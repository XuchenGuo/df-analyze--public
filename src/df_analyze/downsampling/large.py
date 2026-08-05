from __future__ import annotations

from time import perf_counter
from typing import Any, Optional
from warnings import warn

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

from df_analyze._constants import (
    DOWNSAMPLE_DENSE_PEAK_FACTOR,
    DOWNSAMPLE_MAX_CHUNK_BYTES,
    DOWNSAMPLE_MAX_DENSE_BYTES,
)
from df_analyze.downsampling.base import auto_chunk_size, resolve_n_features
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.methods import (
    resolve_feature_downsample_method,
    select_indexed_columns,
)
from df_analyze.downsampling.screening import (
    resolve_screening_split,
)
from df_analyze.enumerables import FeatureDownsampleMethod, ValidationMethod
from df_analyze.preprocessing.inspection.containers import ColumnType, InspectionInfo
from df_analyze.preprocessing.inspection.inference import Inference, InferredKind
from df_analyze.preprocessing.inspection.inspection import inspect_data, unify_nans
from df_analyze.preprocessing.prepare import (
    PreparedData,
    prepare_data,
    raw_train_test_indices,
    usable_training_indices,
)
from df_analyze.preprocessing.targets import as_target_list


def _selection_targets(
    frame: DataFrame,
    targets: list[str],
    is_classification: bool,
    rows: NDArray[np.int_],
) -> Series | DataFrame:
    encoded = {}
    for target in targets:
        values = frame[target].iloc[rows].reset_index(drop=True)
        if is_classification:
            if values.nunique(dropna=False) <= 1:
                suffix = (
                    " Remove this target column and re-run df-analyze."
                    if len(targets) > 1
                    else ""
                )
                raise ValueError(f"Target variable {target} is constant.{suffix}")
            encoder = LabelEncoder()
            encoded[target] = encoder.fit_transform(values)
        else:
            numeric = pd.to_numeric(values, errors="raise").astype(float)
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(
                    f"Regression target {target} contains NaN or infinite values."
                )
            if numeric.nunique(dropna=False) <= 1:
                suffix = (
                    " Remove this target column and re-run df-analyze."
                    if len(targets) > 1
                    else ""
                )
                raise ValueError(f"Regression target {target} is constant.{suffix}")
            encoded[target] = numeric.to_numpy()
    y = DataFrame(encoded)
    if len(targets) == 1:
        return y.iloc[:, 0].rename(targets[0])
    return y


def _validate_predictors(X: DataFrame, chunk_size: int) -> None:
    chunk_size = auto_chunk_size(
        n_rows=len(X),
        itemsize=8,
        max_chunk_bytes=DOWNSAMPLE_MAX_CHUNK_BYTES,
        requested=chunk_size,
    )
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        try:
            values = X.iloc[:, start:stop].to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Large-feature table predictors must convert to numeric values."
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(
                "Large-feature table predictors contain NaN or infinite values."
            )


def _treat_numeric_predictors_as_continuous(inspection) -> None:
    numeric_infos = dict(inspection.conts.infos)
    for inferred in (inspection.ords, inspection.cats, inspection.binaries):
        for name in inferred.infos:
            numeric_infos[name] = Inference(
                InferredKind.CertainCont,
                "Large-feature mode treats numeric predictors as continuous.",
            )
    inspection.conts = InspectionInfo(ColumnType.Continuous, numeric_infos)
    inspection.ords = InspectionInfo(ColumnType.Ordinal)
    inspection.cats = InspectionInfo(ColumnType.Categorical)
    inspection.binaries = InspectionInfo(ColumnType.Binary)
    inspection.big_cats = {}
    inspection.multi_cats = []
    inspection.inflation = []
    inspection.user_cats = set()
    inspection.user_ords = set()


def large_table_prepared_splits(
    frame: DataFrame,
    options: Any,
    train_indices: Optional[NDArray[np.int_]] = None,
    test_indices: Optional[list[NDArray[np.int_]]] = None,
) -> list[tuple[PreparedData, PreparedData, FeatureDownsampleResult]]:
    if options.feat_downsample is FeatureDownsampleMethod.None_:
        raise ValueError("--large-feature-mode requires --feat-downsample.")
    if options.categoricals or options.ordinals:
        raise ValueError(
            "Large-feature table mode accepts numeric predictors only; categorical "
            "and ordinal columns require the normal preparation pipeline."
        )

    targets = as_target_list(options.targets)
    if len(targets) != len(set(targets)):
        raise ValueError("Target column names must be unique.")
    missing = [name for name in targets if name not in frame.columns]
    if missing:
        raise KeyError(f"Target columns not found: {missing}")
    original_rows = len(frame)
    groups = None
    if options.grouper is not None:
        if options.grouper not in frame.columns:
            raise KeyError(f"Grouping column not found: {options.grouper}")
        # Never call DataFrame.map over the complete ultra-wide table.  Only
        # normalize the metadata column needed before group-aware splitting;
        # target cleaning is likewise column-first in usable_training_indices.
        cleaned_groups = unify_nans(frame[[options.grouper]].copy())
        frame = frame.copy(deep=False)
        frame[options.grouper] = cleaned_groups[options.grouper].to_numpy()
        groups = cleaned_groups[options.grouper].reset_index(drop=True)
    excluded = set(targets) | set(options.drops)
    if options.grouper is not None:
        excluded.add(options.grouper)
    X = frame.drop(columns=list(excluded), errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain for feature downsampling.")
    original_features = X.shape[1]
    if not options.assume_numeric_features:
        non_numeric = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if non_numeric:
            shown = ", ".join(map(str, non_numeric[:10]))
            raise TypeError(
                "Large-feature table mode requires numeric predictors. Non-numeric "
                f"columns include: {shown}"
            )
    _validate_predictors(X, options.downsample_chunk_size)

    target_spec: str | list[str] = targets[0] if len(targets) == 1 else targets
    if train_indices is None:
        train, test, split_audit = raw_train_test_indices(
            frame,
            target_spec,
            options.grouper,
            options.is_classification,
            options.test_val_size,
            options.seed,
        )
        split_indices = [(train, test, split_audit)]
    else:
        partitions = [
            np.asarray(train_indices, dtype=int),
            *[np.asarray(test, dtype=int) for test in (test_indices or [])],
        ]
        if len(partitions) < 2:
            raise ValueError("External large-feature input did not provide a test split.")
        method = getattr(options, "tests_method", ValidationMethod.List)
        if method is ValidationMethod.List:
            split_indices = [(partitions[0], test, None) for test in partitions[1:]]
        elif method is ValidationMethod.LODO:
            split_indices = []
            for train_idx, train in enumerate(partitions):
                test_parts = partitions[:train_idx] + partitions[train_idx + 1 :]
                split_indices.append((train, np.concatenate(test_parts), None))
        else:
            raise ValueError(f"Invalid external validation method: {method}")

    outputs = []
    selection_cache: dict[
        tuple[int, ...], tuple[NDArray[np.int_], FeatureDownsampleResult]
    ] = {}
    feature_names = X.columns.astype(str).tolist()
    protected_requested = set(getattr(options, "downsample_protected_features", []))
    protected_lookup = {
        name: idx for idx, name in enumerate(feature_names) if name in protected_requested
    }
    missing_protected = sorted(protected_requested - protected_lookup.keys())
    if missing_protected:
        raise KeyError(
            "Protected large-table features were not found among predictors: "
            f"{missing_protected}"
        )
    protected_indices = sorted(protected_lookup.values())
    for train_rows, test_rows, split_audit in split_indices:
        fit_rows = usable_training_indices(
            frame,
            target_spec,
            options.is_classification,
            np.asarray(train_rows, dtype=int),
        )
        y_train = _selection_targets(frame, targets, options.is_classification, fit_rows)
        groups_train = (
            None if groups is None else groups.iloc[fit_rows].reset_index(drop=True)
        )
        n_select = resolve_n_features(options.n_feat_downsample, X.shape[1])
        resolved = resolve_feature_downsample_method(
            options.feat_downsample,
            len(fit_rows),
            X.shape[1],
            n_select,
            options.is_classification,
            y_train,
        )
        effective, screening, tuning, fallback_note = resolve_screening_split(
            options.feat_downsample,
            resolved,
            y_train,
            options.downsample_screening_fraction,
            options.is_classification,
            groups_train,
            options.seed,
        )
        cache_key = tuple(np.asarray(fit_rows, dtype=int).tolist())
        cached = selection_cache.get(cache_key)
        if cached is None:
            selection_request = (
                options.feat_downsample if fallback_note is None else effective
            )
            try:
                selected, result = select_indexed_columns(
                    X,
                    fit_rows,
                    y_train,
                    options.is_classification,
                    selection_request,
                    options.n_feat_downsample,
                    options.downsample_chunk_size,
                    screening,
                    feature_names,
                    options.downsample_variance_threshold,
                    options.downsample_save_scores,
                    input_format="table-large",
                    seed=options.seed,
                    large_feature_mode=True,
                    protected_indices=protected_indices,
                )
            except ValueError as error:
                if (
                    options.feat_downsample is not FeatureDownsampleMethod.Auto
                    or screening is None
                    or "usable downsampling scores" not in str(error)
                ):
                    raise
                fallback_note = (
                    f"Auto feature downsampling fell back from {effective.value} to "
                    "normalized-variance because supervised scoring produced no "
                    f"usable feature scores: {error}"
                )
                warn(fallback_note, stacklevel=2)
                screening = None
                tuning = np.arange(len(y_train), dtype=int)
                selected, result = select_indexed_columns(
                    X,
                    fit_rows,
                    y_train,
                    options.is_classification,
                    FeatureDownsampleMethod.NormalizedVariance,
                    options.n_feat_downsample,
                    options.downsample_chunk_size,
                    None,
                    feature_names,
                    options.downsample_variance_threshold,
                    options.downsample_save_scores,
                    input_format="table-large",
                    seed=options.seed,
                    large_feature_mode=True,
                    protected_indices=protected_indices,
                )
            result.requested_method = options.feat_downsample.value
            if (
                options.feat_downsample is FeatureDownsampleMethod.Auto
                and result.auto_reason is None
            ):
                result.auto_reason = fallback_note
            if fallback_note is not None:
                result.notes.append(fallback_note)
            result.tuning_rows = tuning.tolist()
            result.tuning_samples = len(tuning)
            selection_cache[cache_key] = (
                selected.copy(),
                result.clone_for_reuse(),
            )
        else:
            selected, result = (
                cached[0].copy(),
                cached[1].clone_for_reuse(),
            )
            result.fit_seconds = 0.0
            result.notes.append("Reused feature scores from the same training partition.")
        if result.resolved_method == FeatureDownsampleMethod.None_.value:
            result.notes.append(
                "The requested feature limit did not reduce the large table; all "
                "source predictors were materialized for normal preprocessing."
            )
        train_rows = np.asarray(train_rows, dtype=int)
        test_rows = np.asarray(test_rows, dtype=int)
        dense_bytes = (len(train_rows) + len(test_rows)) * len(selected) * 8
        peak_bytes = dense_bytes * DOWNSAMPLE_DENSE_PEAK_FACTOR
        if peak_bytes > DOWNSAMPLE_MAX_DENSE_BYTES:
            gib = peak_bytes / 1024**3
            raise MemoryError(
                "Selected train/test matrices are estimated to require about "
                f"{gib:.2f} GiB of peak dense working memory. Reduce "
                "--n-feat-downsample."
            )
        started = perf_counter()
        selected_names = [feature_names[idx] for idx in selected]
        used_rows = np.concatenate([train_rows, test_rows])
        local_train = np.arange(len(train_rows), dtype=int)
        local_test = np.arange(len(train_rows), len(used_rows), dtype=int)
        fit_local = np.flatnonzero(np.isin(train_rows, fit_rows))
        selected_frame = X.iloc[used_rows, selected].copy().reset_index(drop=True)
        selected_frame.columns = selected_names
        if options.grouper is not None:
            selected_frame[options.grouper] = (
                frame[options.grouper].iloc[used_rows].to_numpy()
            )
        for target in targets:
            selected_frame[target] = frame[target].iloc[used_rows].to_numpy()

        _, inspection = inspect_data(
            selected_frame.iloc[fit_local].reset_index(drop=True),
            target_spec,
            options.grouper,
            [],
            [],
            [],
            _warn=True,
        )
        _treat_numeric_predictors_as_continuous(inspection)
        prepared = prepare_data(
            selected_frame,
            target_spec,
            options.grouper,
            inspection,
            options.is_classification,
            local_train,
            [local_test],
            ValidationMethod.List,
        )
        prepared_train, prepared_test = next(
            prepared.get_splits(test_size=options.test_val_size, seed=options.seed)
        )
        result.transform_seconds = perf_counter() - started
        final_names = prepared_train.X.columns.astype(str).tolist()
        # The final names are a tiny selected subset; never build a million-key
        # reverse map for the complete source table.
        index_by_name = dict(zip(selected_names, selected))
        result.selected_features = final_names
        result.selected_indices = [int(index_by_name[name]) for name in final_names]
        result.n_features_out = len(final_names)
        if len(final_names) < len(selected_names):
            result.notes.append(
                "Normal preprocessing removed unusable selected features."
            )
        if prepared_train.info is not None:
            prepared_train.info.original_shape = (original_rows, original_features)
            prepared_train.info.split_audit = split_audit
            prepared_train.info.runtimes["feature downsampling"] = result.total_seconds
        if prepared_test.info is not None:
            prepared_test.info.original_shape = (original_rows, original_features)
            prepared_test.info.split_audit = split_audit
            prepared_test.info.runtimes["feature downsampling"] = result.total_seconds
        outputs.append((prepared_train, prepared_test, result))
    return outputs

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

from df_analyze._constants import (
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
        values = frame.iloc[rows][target].reset_index(drop=True)
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
    frame = unify_nans(frame)
    groups = None
    if options.grouper is not None:
        if options.grouper not in frame.columns:
            raise KeyError(f"Grouping column not found: {options.grouper}")
        groups = frame[options.grouper].reset_index(drop=True)
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
            split_indices = [
                (partitions[0], test, None) for test in partitions[1:]
            ]
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
    for train_rows, test_rows, split_audit in split_indices:
        fit_rows = usable_training_indices(
            frame,
            target_spec,
            options.is_classification,
            np.asarray(train_rows, dtype=int),
        )
        y_train = _selection_targets(
            frame, targets, options.is_classification, fit_rows
        )
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
        )
        cache_key = tuple(np.asarray(fit_rows, dtype=int).tolist())
        cached = selection_cache.get(cache_key)
        if cached is None:
            selected, result = select_indexed_columns(
                X,
                fit_rows,
                y_train,
                options.is_classification,
                effective,
                options.n_feat_downsample,
                options.downsample_chunk_size,
                screening,
                feature_names,
                options.downsample_variance_threshold,
                options.downsample_save_scores,
                input_format="table-large",
            )
            result.requested_method = options.feat_downsample.value
            if fallback_note is not None:
                result.notes.append(fallback_note)
            result.tuning_rows = tuning.tolist()
            result.tuning_samples = len(tuning)
            selection_cache[cache_key] = (selected.copy(), deepcopy(result))
        else:
            selected, result = cached[0].copy(), deepcopy(cached[1])
            result.fit_seconds = 0.0
            result.notes.append("Reused feature scores from the same training partition.")
        dense_bytes = (len(fit_rows) + len(test_rows)) * len(selected) * 8
        if dense_bytes > DOWNSAMPLE_MAX_DENSE_BYTES:
            gib = dense_bytes / 1024**3
            raise MemoryError(
                f"Selected train/test matrices require about {gib:.2f} GiB. "
                "Reduce --n-feat-downsample."
            )
        started = perf_counter()
        selected_names = [feature_names[idx] for idx in selected]
        selected_frame = X.iloc[:, selected].copy()
        selected_frame.columns = selected_names
        if options.grouper is not None:
            selected_frame[options.grouper] = frame[options.grouper].to_numpy()
        for target in targets:
            selected_frame[target] = frame[target].to_numpy()

        _, inspection = inspect_data(
            selected_frame.iloc[fit_rows].reset_index(drop=True),
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
            np.asarray(train_rows, dtype=int),
            [np.asarray(test_rows, dtype=int)],
            ValidationMethod.List,
        )
        prepared_train, prepared_test = next(
            prepared.get_splits(test_size=options.test_val_size, seed=options.seed)
        )
        result.transform_seconds = perf_counter() - started
        final_names = prepared_train.X.columns.astype(str).tolist()
        index_by_name = dict(zip(feature_names, range(len(feature_names))))
        result.selected_features = final_names
        result.selected_indices = [index_by_name[name] for name in final_names]
        result.n_features_out = len(final_names)
        if len(final_names) < len(selected_names):
            result.notes.append(
                "Normal preprocessing removed unusable selected features."
            )
        if prepared_train.info is not None:
            prepared_train.info.original_shape = (original_rows, original_features)
            prepared_train.info.split_audit = split_audit
            prepared_train.info.runtimes["feature downsampling"] = (
                result.fit_seconds + result.transform_seconds
            )
        if prepared_test.info is not None:
            prepared_test.info.original_shape = (original_rows, original_features)
            prepared_test.info.split_audit = split_audit
            prepared_test.info.runtimes["feature downsampling"] = (
                result.fit_seconds + result.transform_seconds
            )
        outputs.append((prepared_train, prepared_test, result))
    return outputs

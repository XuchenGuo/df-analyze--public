from __future__ import annotations

from time import perf_counter
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from df_analyze._constants import (
    DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
    DOWNSAMPLE_MAX_CHUNK_BYTES,
    DOWNSAMPLE_SCORE_LIMIT,
    N_FEAT_DOWNSAMPLE_DEFAULT,
    SEED,
)
from df_analyze.downsampling.base import auto_chunk_size, resolve_n_features
from df_analyze.downsampling.chunked import (
    chunked_f_test_scores,
    chunked_range_normalized_variance_scores,
)
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.ranking import descending_rank_percentiles, top_k_indices
from df_analyze.downsampling.screening import resolve_screening_split
from df_analyze.enumerables import FeatureDownsampleMethod
from df_analyze.preprocessing.prepare import PreparedData

INDEXED_SAFE_METHODS = {
    FeatureDownsampleMethod.None_,
    FeatureDownsampleMethod.NormalizedVariance,
    FeatureDownsampleMethod.FTest,
}


def _method(value: Any) -> FeatureDownsampleMethod:
    if value is None:
        return FeatureDownsampleMethod.None_
    if isinstance(value, FeatureDownsampleMethod):
        return value
    return FeatureDownsampleMethod(str(value))


def _rank_indices(
    scores: NDArray[np.float64],
    n_select: int,
    threshold: Optional[float] = None,
    *,
    seed: int = SEED,
    tie_keys: Optional[Sequence[str]] = None,
) -> NDArray[np.int_]:
    return top_k_indices(
        scores,
        n_select,
        threshold,
        seed=seed,
        tie_keys=tie_keys,
    )


def _rank_score(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    return descending_rank_percentiles(scores)


def _effective_chunk_size(X: Any, requested: int) -> int:
    dtype = getattr(X, "dtype", None)
    itemsize = getattr(dtype, "itemsize", 8)
    return auto_chunk_size(
        n_rows=X.shape[0],
        itemsize=itemsize,
        max_chunk_bytes=DOWNSAMPLE_MAX_CHUNK_BYTES,
        requested=requested,
    )


def resolve_feature_downsample_method(
    requested: FeatureDownsampleMethod | str | None,
    n_samples: int,
    n_features: int,
    n_features_out: int,
    is_classification: bool,
    y: Optional[Series | DataFrame] = None,
) -> FeatureDownsampleMethod:
    del n_samples, is_classification, y
    method = _method(requested)
    if method is FeatureDownsampleMethod.None_ or n_features <= n_features_out:
        return FeatureDownsampleMethod.None_
    return method


def _score_details(
    scores: Optional[NDArray[np.float64]],
    feature_names: Sequence[str],
    save_all: bool,
    seed: int,
) -> tuple[Optional[list[float]], Optional[list[int]], Optional[list[str]]]:
    if scores is None:
        return None, None, None
    limit = min(DOWNSAMPLE_SCORE_LIMIT, len(scores))
    ranked = _rank_indices(
        scores,
        limit,
        seed=seed,
        tie_keys=feature_names,
    )
    return (
        [float(scores[idx]) for idx in ranked],
        ranked.astype(int).tolist(),
        [str(feature_names[idx]) for idx in ranked],
    )


def select_indexed_columns(
    X: Any,
    train_indices: NDArray[np.int_],
    y_train: Series | DataFrame,
    is_classification: bool,
    requested_method: FeatureDownsampleMethod | str | None,
    n_features_out: int | float = N_FEAT_DOWNSAMPLE_DEFAULT,
    chunk_size: int = DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
    screening_positions: Optional[NDArray[np.int_]] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_all_scores: bool = False,
    sparse_input: bool = False,
    input_format: str = "table",
    seed: int = SEED,
    large_feature_mode: bool = False,
    protected_indices: Optional[Sequence[int]] = None,
) -> tuple[NDArray[np.int_], FeatureDownsampleResult]:
    n_features = int(X.shape[1])
    n_select = resolve_n_features(n_features_out, n_features)
    requested = _method(requested_method)
    resolved = resolve_feature_downsample_method(
        requested,
        len(train_indices),
        n_features,
        n_select,
        is_classification,
        y_train,
    )
    names = (
        X.columns.astype(str).tolist()
        if feature_names is None and isinstance(X, DataFrame)
        else (
            [f"feature_{idx}" for idx in range(n_features)]
            if feature_names is None
            else feature_names
        )
    )
    if len(names) != n_features:
        raise ValueError("Feature names must match the input feature count.")

    protected = np.asarray(
        sorted(
            set(
                int(idx)
                for idx in ([] if protected_indices is None else protected_indices)
            )
        ),
        dtype=int,
    )
    if len(protected) and (protected[0] < 0 or protected[-1] >= n_features):
        raise ValueError("Protected feature indices must refer to source columns.")
    if len(protected) > n_select:
        raise ValueError(
            f"{len(protected):,} protected features exceed the requested output "
            f"count of {n_select:,}. Increase --n-feat-downsample."
        )
    n_ranked_select = n_select - len(protected)

    started = perf_counter()
    actual_chunk = _effective_chunk_size(X, chunk_size)
    notes: list[str] = []
    if actual_chunk < int(chunk_size):
        budget_mib = DOWNSAMPLE_MAX_CHUNK_BYTES / 1024**2
        notes.append(
            f"Score chunk size was reduced from {int(chunk_size):,} to "
            f"{actual_chunk:,} to stay within the {budget_mib:g} MiB "
            "working-memory budget."
        )

    scores: Optional[NDArray[np.float64]] = None
    fit_rows = train_indices
    y_fit = y_train
    if resolved is FeatureDownsampleMethod.None_:
        selected = np.arange(n_features, dtype=int)
    else:
        if screening_positions is not None:
            fit_rows = train_indices[screening_positions]
            y_fit = y_train.iloc[screening_positions]
        if resolved is FeatureDownsampleMethod.NormalizedVariance:
            scores = chunked_range_normalized_variance_scores(X, actual_chunk, fit_rows)
            threshold = np.nextafter(0.0, np.inf)
        elif resolved is FeatureDownsampleMethod.FTest:
            scores = chunked_f_test_scores(
                X, y_fit, is_classification, actual_chunk, fit_rows
            )
            threshold = None
        else:  # pragma: no cover - enum and CLI constrain this path
            raise ValueError(f"Unsupported feature downsampling method: {resolved.value}")

        if n_ranked_select == 0:
            selected = protected.copy()
        else:
            assert scores is not None
            ranking_scores = scores.copy()
            if len(protected):
                ranking_scores[protected] = -np.inf
            ranked = _rank_indices(
                ranking_scores,
                n_ranked_select,
                threshold,
                seed=seed,
                tie_keys=names,
            )
            if len(ranked) != n_ranked_select:
                raise ValueError(
                    "Fewer usable downsampling scores were available among "
                    "non-protected features than requested."
                )
            selected = np.concatenate([protected, ranked])

    score_values, score_indices, score_names = _score_details(
        scores, names, save_all_scores, seed
    )
    aggregation = (
        "not_targeted"
        if resolved is not FeatureDownsampleMethod.FTest
        else (
            "mean_rank_percentile_across_targets"
            if isinstance(y_train, DataFrame) and y_train.shape[1] > 1
            else "single_target"
        )
    )
    result = FeatureDownsampleResult(
        requested_method=requested.value,
        resolved_method=resolved.value,
        n_features_in=n_features,
        n_features_out=len(selected),
        selected_features=[str(names[int(idx)]) for idx in selected],
        selected_indices=selected.astype(int).tolist(),
        protected_features=[str(names[int(idx)]) for idx in protected],
        protected_indices=protected.astype(int).tolist(),
        scores=score_values,
        score_feature_indices=score_indices,
        score_feature_names=score_names,
        full_score_values=scores if save_all_scores else None,
        full_score_feature_names=names
        if save_all_scores and scores is not None
        else None,
        screening_rows=(
            [] if screening_positions is None else screening_positions.tolist()
        ),
        screening_samples=0 if screening_positions is None else len(fit_rows),
        tuning_samples=(
            len(y_train) - len(fit_rows)
            if screening_positions is not None
            else len(y_train)
        ),
        chunk_size=actual_chunk,
        sparse_input=sparse_input,
        input_format=input_format,
        fit_seconds=perf_counter() - started,
        notes=notes,
        score_aggregation=aggregation,
        large_feature_mode=large_feature_mode,
    )
    return selected, result


def _downsample_frame(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Series | DataFrame,
    is_classification: bool,
    requested: FeatureDownsampleMethod,
    n_features_out: int | float,
    chunk_size: int,
    screening_rows: Optional[NDArray[np.int_]],
    save_all_scores: bool,
    seed: int,
    protected_names: Sequence[str],
) -> tuple[DataFrame, DataFrame, FeatureDownsampleResult]:
    missing = sorted(set(protected_names) - set(X_train.columns.astype(str)))
    if missing:
        raise ValueError(f"Protected downsampling features were not found: {missing}")
    protected = [
        idx
        for idx, name in enumerate(X_train.columns.astype(str))
        if name in set(protected_names)
    ]
    selected, result = select_indexed_columns(
        X_train,
        np.arange(len(X_train), dtype=int),
        y_train,
        is_classification,
        requested,
        n_features_out,
        chunk_size,
        screening_rows,
        X_train.columns.astype(str).tolist(),
        save_all_scores=save_all_scores,
        seed=seed,
        protected_indices=protected,
    )
    transform_started = perf_counter()
    train_out = X_train.iloc[:, selected].copy()
    test_out = X_test.iloc[:, selected].copy()
    result.transform_seconds = perf_counter() - transform_started
    return train_out, test_out, result


def downsample_split(
    train: PreparedData,
    test: PreparedData,
    options: Any,
) -> tuple[PreparedData, PreparedData, FeatureDownsampleResult]:
    requested = _method(getattr(options, "feat_downsample", None))
    n_features_out = getattr(options, "n_feat_downsample", N_FEAT_DOWNSAMPLE_DEFAULT)
    seed = int(getattr(options, "seed", SEED))
    resolved = resolve_feature_downsample_method(
        requested,
        len(train.X),
        train.X.shape[1],
        resolve_n_features(n_features_out, train.X.shape[1]),
        train.is_classification,
        train.y,
    )
    _, screening_rows, tuning_rows, _ = resolve_screening_split(
        requested,
        resolved,
        train.y,
        getattr(options, "downsample_screening_fraction", 0.25),
        train.is_classification,
        train.groups,
        seed,
    )
    X_train, X_test, result = _downsample_frame(
        train.X,
        test.X,
        train.y,
        train.is_classification,
        requested,
        n_features_out,
        getattr(options, "downsample_chunk_size", DOWNSAMPLE_CHUNK_SIZE_DEFAULT),
        screening_rows,
        getattr(options, "downsample_save_scores", False),
        seed,
        getattr(options, "downsample_protected_features", []),
    )
    result.screening_rows = [] if screening_rows is None else screening_rows.tolist()
    result.tuning_rows = tuning_rows.tolist()
    result.screening_samples = 0 if screening_rows is None else len(screening_rows)
    result.tuning_samples = len(tuning_rows)
    if result.resolved_method == FeatureDownsampleMethod.None_.value:
        return train, test, result
    return (
        train.with_features(X_train, result.resolved_method),
        test.with_features(X_test, result.resolved_method),
        result,
    )

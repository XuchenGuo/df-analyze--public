from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy import sparse
from sklearn.feature_selection import f_classif, f_regression


def validate_chunk_size(chunk_size: int) -> int:
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("Downsample chunk size must be positive.")
    return chunk_size


def _column_chunk(
    X: Any, start: int, stop: int, row_indices: Optional[NDArray[np.int_]]
) -> Any:
    if hasattr(X, "iloc"):
        if row_indices is None:
            return X.iloc[:, start:stop]
        return X.iloc[row_indices, start:stop]
    if row_indices is None:
        return X[:, start:stop]
    return X[row_indices, start:stop]


def _sparse_variance(X: Any) -> NDArray[np.float64]:
    n_samples = int(X.shape[0])
    if n_samples <= 1:
        return np.full(X.shape[1], np.nan, dtype=np.float64)
    means = np.asarray(X.mean(axis=0)).ravel().astype(np.float64, copy=False)
    squared = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(
        np.float64, copy=False
    )
    population = np.maximum(0.0, squared - means * means)
    return population * n_samples / (n_samples - 1)


def chunked_variance_scores(
    X: Any,
    chunk_size: int,
    row_indices: Optional[NDArray[np.int_]] = None,
) -> NDArray[np.float64]:
    chunk_size = validate_chunk_size(chunk_size)
    scores = np.empty(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        chunk = _column_chunk(X, start, stop, row_indices)
        if sparse.issparse(chunk):
            values = _sparse_variance(chunk)
        elif hasattr(chunk, "var") and hasattr(chunk, "to_numpy"):
            values = chunk.var(axis=0).to_numpy(dtype=np.float64)
        else:
            values = np.var(np.asarray(chunk), axis=0, ddof=1, dtype=np.float64)
        scores[start:stop] = values
    return scores


def chunked_range_normalized_variance_scores(
    X: Any,
    chunk_size: int,
    row_indices: Optional[NDArray[np.int_]] = None,
) -> NDArray[np.float64]:
    """Return sample variance divided by squared observed feature range.

    The normalization makes this ensemble member invariant to a feature's
    non-zero linear rescaling while retaining the cheap, chunked implementation
    needed for very wide inputs.
    """
    chunk_size = validate_chunk_size(chunk_size)
    scores = np.empty(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        chunk = _column_chunk(X, start, stop, row_indices)
        if sparse.issparse(chunk):
            variance = _sparse_variance(chunk)
            minimum = chunk.min(axis=0)
            maximum = chunk.max(axis=0)
            if sparse.issparse(minimum):
                minimum = minimum.toarray()
            if sparse.issparse(maximum):
                maximum = maximum.toarray()
            feature_range = np.asarray(maximum).ravel() - np.asarray(minimum).ravel()
        elif hasattr(chunk, "var") and hasattr(chunk, "to_numpy"):
            variance = chunk.var(axis=0).to_numpy(dtype=np.float64)
            feature_range = (
                chunk.max(axis=0).to_numpy(dtype=np.float64)
                - chunk.min(axis=0).to_numpy(dtype=np.float64)
            )
        else:
            values = np.asarray(chunk)
            variance = np.var(values, axis=0, ddof=1, dtype=np.float64)
            feature_range = np.ptp(values, axis=0).astype(np.float64, copy=False)

        normalized = np.zeros(stop - start, dtype=np.float64)
        np.divide(
            variance,
            feature_range**2,
            out=normalized,
            where=np.isfinite(feature_range) & (feature_range > 0.0),
        )
        normalized[~np.isfinite(variance)] = np.nan
        scores[start:stop] = normalized
    return scores


def _target_series(y: Union[Series, DataFrame]) -> list[Series]:
    if isinstance(y, DataFrame):
        return [y[col] for col in y.columns]
    return [y]


def _f_scores(X: Any, y: Series, is_classification: bool) -> NDArray[np.float64]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"sklearn\.feature_selection\._univariate_selection",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"sklearn\.feature_selection\._univariate_selection",
        )
        raw = f_classif(X, y) if is_classification else f_regression(X, y)
    return np.asarray(raw[0], dtype=np.float64)


def _rank_percentiles(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=np.float64)
    usable = ~np.isnan(values) & ~np.isneginf(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(usable)
    if len(indices) == 0:
        return result
    order = np.lexsort((indices, -values[indices]))
    ranked = indices[order]
    result[ranked] = (len(ranked) - np.arange(len(ranked))) / len(ranked)
    return result


def aggregate_target_scores(
    target_scores: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    if len(target_scores) == 0:
        raise ValueError("Expected at least one target score array.")
    if len(target_scores) == 1:
        return target_scores[0]
    ranked = np.vstack([_rank_percentiles(scores) for scores in target_scores])
    counts = np.sum(~np.isnan(ranked), axis=0)
    totals = np.nansum(ranked, axis=0)
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def chunked_f_test_scores(
    X: Any,
    y: Union[Series, DataFrame],
    is_classification: bool,
    chunk_size: int,
    row_indices: Optional[NDArray[np.int_]] = None,
) -> NDArray[np.float64]:
    chunk_size = validate_chunk_size(chunk_size)
    targets = _target_series(y)
    target_scores = [np.empty(X.shape[1], dtype=np.float64) for _ in targets]
    for start in range(0, X.shape[1], chunk_size):
        stop = min(start + chunk_size, X.shape[1])
        chunk = _column_chunk(X, start, stop, row_indices)
        for target_idx, target in enumerate(targets):
            target_scores[target_idx][start:stop] = _f_scores(
                chunk, target, is_classification
            )
    return aggregate_target_scores(target_scores)

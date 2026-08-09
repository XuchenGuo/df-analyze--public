from __future__ import annotations

from hashlib import blake2b
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def usable_score_mask(scores: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return scores that can participate in feature ranking.

    Positive infinity is a valid limiting F statistic for perfectly separated
    features, so it remains rankable. NaN and negative infinity are unavailable.
    """
    values = np.asarray(scores, dtype=np.float64)
    return ~np.isnan(values) & ~np.isneginf(values)


def descending_rank_percentiles(
    scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return tie-aware descending rank percentiles.

    Equal scores receive the same average rank, so rank aggregation is invariant
    to the input column order. Unavailable scores remain NaN and can therefore be
    ignored by downstream ``nanmean`` aggregation.
    """
    values = np.asarray(scores, dtype=np.float64)
    usable = usable_score_mask(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    indices = np.flatnonzero(usable)
    if len(indices) == 0:
        return result
    ranks = rankdata(-values[indices], method="average")
    result[indices] = (len(indices) - ranks + 1.0) / len(indices)
    return result


def _tie_priorities(
    indices: NDArray[np.int_],
    seed: int,
    tie_keys: Optional[Sequence[str]],
) -> NDArray[np.uint64]:
    priorities = np.empty(len(indices), dtype=np.uint64)
    for position, index in enumerate(indices):
        key = str(index) if tie_keys is None else str(tie_keys[int(index)])
        digest = blake2b(
            f"{int(seed)}\0{key}".encode("utf-8", errors="surrogatepass"),
            digest_size=8,
        ).digest()
        priorities[position] = int.from_bytes(digest, "big", signed=False)
    return priorities


def top_k_indices(
    scores: NDArray[np.float64],
    n_select: int,
    threshold: Optional[float] = None,
    *,
    seed: int = 0,
    tie_keys: Optional[Sequence[str]] = None,
) -> NDArray[np.int_]:
    """Select exact top-k indices without sorting the full score vector.

    Ties at the selection boundary are resolved by a stable hash of the feature
    identity and seed instead of raw column position whenever ``tie_keys`` are
    available.
    """
    values = np.asarray(scores, dtype=np.float64)
    if n_select <= 0:
        return np.array([], dtype=int)
    usable = usable_score_mask(values)
    if threshold is not None:
        usable &= values >= threshold
    indices = np.flatnonzero(usable)
    if len(indices) == 0:
        return np.array([], dtype=int)

    if len(indices) > n_select:
        usable_values = values[indices]
        cutoff_position = len(indices) - n_select
        cutoff = np.partition(usable_values, cutoff_position)[cutoff_position]
        above = indices[usable_values > cutoff]
        boundary = indices[usable_values == cutoff]
        needed = n_select - len(above)
        if needed < len(boundary):
            priorities = _tie_priorities(boundary, seed, tie_keys)
            boundary = boundary[np.argsort(priorities, kind="stable")[:needed]]
        indices = np.concatenate([above, boundary])

    priorities = _tie_priorities(indices, seed, tie_keys)
    order = np.lexsort((priorities, -values[indices]))
    return indices[order].astype(int, copy=False)

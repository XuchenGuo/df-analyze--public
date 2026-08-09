from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.model_selection import (
    GroupShuffleSplit,
    ShuffleSplit,
    StratifiedShuffleSplit,
)

from df_analyze._constants import SEED
from df_analyze.enumerables import FeatureDownsampleMethod

SUPERVISED_DOWNSAMPLERS = {
    FeatureDownsampleMethod.FTest,
}


def _stratification_target(y: Series | DataFrame) -> Optional[Series]:
    if isinstance(y, Series):
        return y
    if y.shape[1] == 1:
        return y.iloc[:, 0]
    combined = Series(y.astype(str).agg("\x1f".join, axis=1), index=y.index)
    counts = combined.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return combined
    return None


def _supports_supervised_scoring(
    y: Series | DataFrame,
    rows: NDArray[np.int_],
    is_classification: bool,
) -> bool:
    if len(rows) < 3:
        return False
    frame = y.to_frame() if isinstance(y, Series) else y
    subset = frame.iloc[rows]
    if not is_classification:
        # Only multi-target scoring requires every target to vary in both
        # subsets. Keep the existing single-target check unchanged.
        if frame.shape[1] == 1:
            return True
        for col in frame:
            values = subset[col].to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.unique(values).size < 2:
                return False
        return True
    for col in frame:
        if set(frame[col].unique()) != set(subset[col].unique()):
            return False
        counts = subset[col].value_counts(dropna=False)
        if len(counts) < 2 or counts.min() < 2:
            return False
    return True


def screening_tuning_indices(
    y: Series | DataFrame,
    fraction: float,
    is_classification: bool,
    groups: Optional[Series] = None,
    seed: int = SEED,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    n_samples = len(y)
    if n_samples < 6:
        raise ValueError(
            "Supervised feature downsampling requires at least six training samples "
            "so screening and tuning both have usable statistical support."
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError("Downsample screening fraction must be in (0, 1).")

    screen_size = max(3, min(n_samples - 3, int(round(fraction * n_samples))))
    indices = np.arange(n_samples, dtype=int)
    if groups is not None:
        if len(groups) != n_samples:
            raise ValueError(
                "Grouping data must have the same number of rows as the training target."
            )
        if groups.nunique(dropna=False) < 2:
            raise ValueError(
                "Group-disjoint feature screening requires at least two distinct "
                "training groups."
            )
        splitter = GroupShuffleSplit(n_splits=100, train_size=fraction, random_state=seed)
        candidates = []
        for screening, tuning in splitter.split(indices, groups=groups.to_numpy()):
            if _supports_supervised_scoring(
                y, screening, is_classification
            ) and _supports_supervised_scoring(y, tuning, is_classification):
                candidates.append(
                    (
                        abs(len(screening) - screen_size),
                        np.sort(screening),
                        np.sort(tuning),
                    )
                )
        if candidates:
            _, screening, tuning = min(candidates, key=lambda item: item[0])
            return screening, tuning
        raise ValueError(
            "Could not create group-disjoint screening and tuning subsets that "
            "both have usable statistical support."
        )

    target = _stratification_target(y) if is_classification else None
    if target is not None:
        n_classes = target.nunique()
        counts = target.value_counts()
        min_partition = 2 * n_classes
        if counts.min() >= 4 and min_partition <= n_samples // 2:
            screen_size = max(
                min_partition,
                min(n_samples - min_partition, screen_size),
            )
            splitter = StratifiedShuffleSplit(
                n_splits=20, train_size=screen_size, random_state=seed
            )
            for screening, tuning in splitter.split(indices, target):
                if _supports_supervised_scoring(
                    y, screening, is_classification
                ) and _supports_supervised_scoring(y, tuning, is_classification):
                    return np.sort(screening), np.sort(tuning)

    splitter = ShuffleSplit(n_splits=100, train_size=screen_size, random_state=seed)
    for screening, tuning in splitter.split(indices):
        if _supports_supervised_scoring(
            y, screening, is_classification
        ) and _supports_supervised_scoring(y, tuning, is_classification):
            return np.sort(screening), np.sort(tuning)
    raise ValueError(
        "Could not create disjoint screening and tuning subsets that both have "
        "usable statistical support."
    )


def method_needs_screening(method: FeatureDownsampleMethod) -> bool:
    return method in SUPERVISED_DOWNSAMPLERS


def resolve_screening_split(
    requested: FeatureDownsampleMethod,
    resolved: FeatureDownsampleMethod,
    y: Series | DataFrame,
    fraction: float,
    is_classification: bool,
    groups: Optional[Series] = None,
    seed: int = SEED,
) -> tuple[
    FeatureDownsampleMethod,
    Optional[NDArray[np.int_]],
    NDArray[np.int_],
    Optional[str],
]:
    tuning = np.arange(len(y), dtype=int)
    if not method_needs_screening(resolved):
        return resolved, None, tuning, None
    screening, tuning = screening_tuning_indices(
        y,
        fraction,
        is_classification,
        groups,
        seed,
    )
    return resolved, screening, tuning, None

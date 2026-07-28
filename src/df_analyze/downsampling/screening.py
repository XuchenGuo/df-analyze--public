from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit

from df_analyze._constants import SEED
from df_analyze.enumerables import FeatureDownsampleMethod

SUPERVISED_DOWNSAMPLERS = {
    FeatureDownsampleMethod.FTest,
    FeatureDownsampleMethod.MutualInfo,
    FeatureDownsampleMethod.Linear,
    FeatureDownsampleMethod.LGBM,
    FeatureDownsampleMethod.RankEnsemble,
    FeatureDownsampleMethod.SelectorEnsemble,
    FeatureDownsampleMethod.StableRank,
}


def _stratification_target(y: Series | DataFrame) -> Optional[Series]:
    if isinstance(y, Series):
        return y
    if y.shape[1] == 1:
        return y.iloc[:, 0]
    combined = y.astype(str).agg("\x1f".join, axis=1)
    counts = combined.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return combined
    return None


def _has_all_levels(y: Series | DataFrame, rows: NDArray[np.int_]) -> bool:
    frame = y.to_frame() if isinstance(y, Series) else y
    subset = frame.iloc[rows]
    return all(set(frame[col].unique()) == set(subset[col].unique()) for col in frame)


def screening_tuning_indices(
    y: Series | DataFrame,
    fraction: float,
    is_classification: bool,
    groups: Optional[Series] = None,
    seed: int = SEED,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    n_samples = len(y)
    if n_samples < 4:
        raise ValueError(
            "Supervised feature downsampling requires at least four training samples."
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError("Downsample screening fraction must be in (0, 1).")

    screen_size = max(2, min(n_samples - 2, int(round(fraction * n_samples))))
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
        splitter = GroupShuffleSplit(n_splits=20, train_size=fraction, random_state=seed)
        for screening, tuning in splitter.split(indices, groups=groups.to_numpy()):
            if not is_classification or (
                _has_all_levels(y, screening) and _has_all_levels(y, tuning)
            ):
                return np.sort(screening), np.sort(tuning)
        raise ValueError(
            "Could not create group-disjoint screening and tuning subsets that "
            "both contain every target level."
        )

    target = _stratification_target(y) if is_classification else None
    if target is not None:
        n_classes = target.nunique()
        counts = target.value_counts()
        if counts.min() >= 2 and n_classes <= n_samples // 2:
            screen_size = max(n_classes, min(n_samples - n_classes, screen_size))
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=screen_size, random_state=seed
            )
            screening, tuning = next(splitter.split(indices, target))
            if _has_all_levels(y, screening) and _has_all_levels(y, tuning):
                return np.sort(screening), np.sort(tuning)

    splitter = ShuffleSplit(n_splits=1, train_size=screen_size, random_state=seed)
    screening, tuning = next(splitter.split(indices))
    if is_classification and not (
        _has_all_levels(y, screening) and _has_all_levels(y, tuning)
    ):
        raise ValueError(
            "Could not create disjoint screening and tuning subsets that both "
            "contain every target level."
        )
    return np.sort(screening), np.sort(tuning)


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
    try:
        screening, tuning = screening_tuning_indices(
            y,
            fraction,
            is_classification,
            groups,
            seed,
        )
        return resolved, screening, tuning, None
    except ValueError as error:
        if requested is not FeatureDownsampleMethod.Auto:
            raise
        note = (
            f"Auto feature downsampling fell back from {resolved.value} to variance "
            f"because disjoint screening and tuning subsets were not feasible: {error}"
        )
        warn(note, stacklevel=2)
        return FeatureDownsampleMethod.Variance, None, np.arange(len(y), dtype=int), note

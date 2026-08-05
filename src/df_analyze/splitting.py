from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Type, Union
from warnings import catch_warnings, filterwarnings, warn

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Index, Series, factorize
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit as StratSplit,
)

from df_analyze._constants import (
    N_MULTITARGET_LEVEL_MIN,
    N_TARG_LEVEL_MIN_TEST_INTERNAL,
    N_TARG_LEVEL_MIN_TRAIN_INTERNAL,
    SEED,
)

AnyKFold = Union[
    Type[KFold],
    Type[GroupKFold],
    Type[StratifiedKFold],
    Type[StratifiedGroupKFold],
]

AnySplitter = Union[
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
]

APPROXIMATE_GROUP_SPLIT_DIFF_WARN_THRESHOLD = 0.1
_UNSTRATIFIED_SPLIT_ATTR = "df_analyze_unstratified_split"
_MULTITARGET_SPLIT_RETRIES = 128


@dataclass
class MultiTargetSplitInfo:
    initial_targets: list[str]
    used_targets: list[str]
    dropped_targets: list[str]
    initial_n_unique_combinations: int
    initial_min_combination_count: int
    final_n_unique_combinations: int
    final_min_combination_count: int
    min_count: int
    max_unique_combinations: int
    reason: str

    @property
    def fallback_used(self) -> bool:
        return len(self.dropped_targets) > 0

    def to_markdown(self) -> str:
        row = {
            "initial targets": ", ".join(self.initial_targets),
            "used targets": ", ".join(self.used_targets),
            "dropped targets": ", ".join(self.dropped_targets) or "(none)",
            "initial combinations": self.initial_n_unique_combinations,
            "initial minimum count": self.initial_min_combination_count,
            "final combinations": self.final_n_unique_combinations,
            "final minimum count": self.final_min_combination_count,
            "required minimum count": self.min_count,
            "maximum combinations": self.max_unique_combinations,
            "fallback used": self.fallback_used,
        }
        return (
            "# Multi-Target Split Audit\n\n"
            f"{self.reason}\n\n"
            f"{DataFrame([row]).to_markdown(index=False)}\n"
        )


def _combination_summary(label: Series) -> tuple[int, int]:
    counts = label.value_counts()
    if counts.empty:
        return 0, 0
    return int(len(counts)), int(counts.min())


def _target_combination_label(y_df: DataFrame, cols: list[Any]) -> Series:
    """Encode target-value tuples without lossy string concatenation."""
    combinations = Index(
        list(y_df[cols].itertuples(index=False, name=None)),
        tupleize_cols=False,
    )
    codes, _ = factorize(combinations, sort=False)
    return Series(codes, index=y_df.index, name="split_proxy")


def regression_split_label(y: Union[Series, DataFrame]) -> Series:
    if isinstance(y, Series):
        return y
    return y.rank(pct=True).mean(axis=1)


def y_split_label_info(
    y_df: DataFrame,
    min_count: int = N_MULTITARGET_LEVEL_MIN,
    warn_on_fallback: bool = True,
) -> tuple[Series, MultiTargetSplitInfo]:
    if y_df.shape[1] == 0:
        raise ValueError("Cannot split with empty target columns.")

    cols = list(y_df.columns)
    initial_targets = [str(col) for col in cols]
    max_unique = max(1, len(y_df) // min_count)
    initial_label = _target_combination_label(y_df, cols)
    initial_unique, initial_min = _combination_summary(initial_label)
    dropped: list[str] = []

    while True:
        label = _target_combination_label(y_df, cols)
        n_unique, smallest = _combination_summary(label)
        feasible = smallest >= min_count and n_unique <= max_unique
        if feasible:
            if not dropped:
                reason = "All target combinations met the stratification thresholds."
            else:
                reason = (
                    "Target columns were removed from the split label until the "
                    "remaining combinations met the stratification thresholds."
                )
            info = MultiTargetSplitInfo(
                initial_targets=initial_targets,
                used_targets=[str(col) for col in cols],
                dropped_targets=dropped,
                initial_n_unique_combinations=initial_unique,
                initial_min_combination_count=initial_min,
                final_n_unique_combinations=n_unique,
                final_min_combination_count=smallest,
                min_count=min_count,
                max_unique_combinations=max_unique,
                reason=reason,
            )
            return label, info
        if len(cols) == 1:
            dropped.append(str(cols[0]))
            if warn_on_fallback:
                warn(
                    "No target column has enough support for multi-target "
                    "stratification. Falling back to a shuffled split."
                )
            label = Series(0, index=y_df.index, name="split_proxy", dtype=np.int8)
            label.attrs[_UNSTRATIFIED_SPLIT_ATTR] = True
            info = MultiTargetSplitInfo(
                initial_targets=initial_targets,
                used_targets=[],
                dropped_targets=dropped,
                initial_n_unique_combinations=initial_unique,
                initial_min_combination_count=initial_min,
                final_n_unique_combinations=1,
                final_min_combination_count=len(label),
                min_count=min_count,
                max_unique_combinations=max_unique,
                reason=(
                    "No target column met the stratification thresholds; the split "
                    "was shuffled without stratification."
                ),
            )
            return label, info
        candidates = []
        for col in cols:
            remaining = [candidate for candidate in cols if candidate != col]
            candidate_label = _target_combination_label(y_df, remaining)
            candidate_unique, candidate_min = _combination_summary(candidate_label)
            feasible = candidate_min >= min_count and candidate_unique <= max_unique
            quality = (
                candidate_unique if feasible else candidate_min,
                candidate_min if feasible else -candidate_unique,
            )
            candidates.append((feasible, *quality, str(col), col))
        drop_col = max(candidates, key=lambda candidate: candidate[:-1])[-1]
        cols.remove(drop_col)
        dropped.append(str(drop_col))
        if warn_on_fallback:
            warn(
                "Multi-target stratification produced undersampled target combinations. "
                f"Dropping target column '{drop_col}' to improve split feasibility."
            )


def y_split_label(
    y_df: DataFrame,
    min_count: int = N_MULTITARGET_LEVEL_MIN,
    warn_on_fallback: bool = True,
) -> Series:
    label, _ = y_split_label_info(
        y_df=y_df,
        min_count=min_count,
        warn_on_fallback=warn_on_fallback,
    )
    return label


def resolve_final_cv_folds(
    y: Series | DataFrame,
    groups: Optional[Series] = None,
    max_splits: int = 5,
) -> int:
    """Resolve a feasible final-CV fold count for a holdout partition.

    Grouped validation is limited by the number of distinct groups; ordinary
    validation is limited by the number of samples. At least two independent
    validation units are required.
    """
    capacity = len(y) if groups is None else int(groups.nunique(dropna=False))
    if capacity < 2:
        unit = "samples" if groups is None else "groups"
        raise ValueError(
            "Final cross-validation on the holdout set requires at least "
            f"two {unit}; found {capacity}."
        )
    return min(max_splits, capacity)


def validate_multitarget_cv_support(
    y: DataFrame,
    n_splits: int = 5,
    phase: str = "internal cross-validation",
) -> None:
    required = max(
        n_splits * N_TARG_LEVEL_MIN_TEST_INTERNAL,
        int(np.ceil(n_splits * N_TARG_LEVEL_MIN_TRAIN_INTERNAL / max(1, n_splits - 1))),
    )
    failures = []
    for target in y.columns:
        counts = y[target].value_counts(dropna=False)
        if len(counts) < 2:
            failures.append(f"'{target}' has only one target level")
            continue
        for level, count in counts.items():
            if int(count) < required:
                failures.append(f"'{target}' level {level!r} has {int(count)} samples")
    if failures:
        details = "; ".join(failures)
        raise ValueError(
            f"Multi-target classification cannot run {phase} with {n_splits} folds. "
            f"Each target level needs at least {required} samples in this partition; "
            f"{details}. Low-support rows are retained, but df-analyze cannot fit and "
            "score every target reliably with the requested cross-validation design."
        )


def validate_multitarget_holdout_coverage(
    y_train: DataFrame,
    y_holdout: DataFrame,
    *,
    external: bool,
) -> None:
    """Report target levels that are absent from a holdout."""

    missing: list[str] = []
    for target in y_train.columns:
        train_levels = set(y_train[target].astype(str))
        holdout_levels = set(y_holdout[target].astype(str))
        omitted = sorted(train_levels - holdout_levels)
        if omitted:
            missing.append(f"'{target}' is missing level(s) {omitted}")
    if not missing:
        return

    message = (
        "The holdout partition does not represent every multi-target "
        f"classification level seen in training: {'; '.join(missing)}. Metrics "
        "for the omitted levels cannot be estimated."
    )
    if external:
        warn(
            f"{message} This holdout was supplied externally, so evaluation will "
            "continue and the omission must be considered when interpreting results."
        )
        return
    raise ValueError(
        f"{message} Change the split seed/size or grouping design; an internally "
        "generated holdout must cover every training level."
    )


def validate_multitarget_regression_support(
    y: DataFrame,
    phase: str,
) -> None:
    """Reject regression partitions that cannot identify every target."""

    failures: list[str] = []
    for target in y.columns:
        values = np.asarray(y[target], dtype=float)
        if not np.isfinite(values).all():
            failures.append(f"'{target}' contains non-finite values")
        elif np.unique(values).size < 2:
            failures.append(f"'{target}' is constant")
    if failures:
        raise ValueError(
            f"Multi-target regression cannot use the {phase}: "
            f"{'; '.join(failures)}. Every target must vary within each outer "
            "training and holdout partition."
        )


def validate_multitarget_model_cv_support(
    y: DataFrame,
    model_classes: Iterable[type],
) -> None:
    """Validate each selected model's declared K-fold tuning design.

    Models that do not use K-fold tuning declare ``tuning_cv_folds=None`` and
    are skipped here. Identical fold designs are checked once, with every
    affected model named in the error context.
    """
    designs: dict[int, list[str]] = {}
    for model_cls in model_classes:
        n_splits = getattr(model_cls, "tuning_cv_folds", 5)
        if n_splits is None:
            continue
        n_splits = int(n_splits)
        if n_splits < 2:
            name = str(getattr(model_cls, "shortname", model_cls.__name__))
            raise ValueError(
                f"Model '{name}' declares invalid tuning_cv_folds={n_splits}; "
                "K-fold tuning requires at least two folds."
            )
        name = str(getattr(model_cls, "shortname", model_cls.__name__))
        designs.setdefault(n_splits, []).append(name)

    for n_splits in sorted(designs):
        names = ", ".join(sorted(set(designs[n_splits])))
        validate_multitarget_cv_support(
            y,
            n_splits=n_splits,
            phase=f"tuning cross-validation for model(s): {names}",
        )


class OmniKFold:
    def __init__(
        self,
        n_splits: int = 5,
        is_classification: bool = True,
        grouped: bool = False,
        labels: Optional[dict[int, str]] = None,
        shuffle: bool = False,
        seed: Optional[int] = SEED,
        warn_on_fallback: bool = True,
        allow_group_fallback: bool = False,
        df_analyze_phase: Optional[str] = None,
    ) -> None:
        self.n_splits: int = n_splits
        self.is_cls = self.is_classification = is_classification
        labels = labels or {}
        self.labels: dict[str, str] = {
            str(encoded): str(orig) for encoded, orig in labels.items()
        }
        self.grouped: bool = grouped
        self.shuffle: bool = shuffle
        self.seed: Optional[int] = seed
        self.warn: bool = warn_on_fallback
        # This option may relax stratification or reduce the fold count, but it
        # must never relax group disjointness.
        self.allow_group_fallback = allow_group_fallback
        self.effective_n_splits = n_splits
        self.fallback_strategy: Optional[str] = None
        self.kf: AnyKFold
        self.kf_fallback: AnyKFold
        self.df_analyze_phase: Optional[str] = df_analyze_phase
        self.phase_info = (
            ""
            if self.df_analyze_phase is None
            else f"Splitting error occurred at df-analyze phase: {self.df_analyze_phase}"
        )

        if self.is_cls:
            self.kf = StratifiedGroupKFold if self.grouped else StratifiedKFold
            self.kf_fallback = GroupKFold if self.grouped else StratifiedKFold
        else:
            self.kf = GroupKFold if self.grouped else KFold
            self.kf_fallback = GroupKFold if self.grouped else KFold

        self.no_fallback = self.kf is self.kf_fallback

    def split(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series] = None,
        multitarget_y: Optional[DataFrame] = None,
    ) -> tuple[list[tuple[ndarray, ndarray]], bool]:
        """
        Returns
        -------
        splits: list[tuple[ndarray, ndarray]]
            NumPy array of split indices (as int64)

        did_fail: bool
            Boolean indicating if the initial grouping split attempt failed, so
            that subsequent split attempts can disable the warnings.
        """
        if self.grouped and g_train is None:
            raise ValueError(
                f"{self.__class__.__name__} was initialized with `grouped=True`, "
                f"but no grouping data was provided. {self.phase_info}"
            )
        if multitarget_y is not None:
            if len(multitarget_y) != len(y_train):
                raise ValueError(
                    "Multi-target fold-support data must align with the split proxy. "
                    f"Got {len(multitarget_y)} target rows and {len(y_train)} proxy rows."
                )
            multitarget_y = multitarget_y.reset_index(drop=True)
            if self.is_cls:
                validate_multitarget_cv_support(
                    multitarget_y,
                    n_splits=self.n_splits,
                    phase=self.df_analyze_phase or "internal cross-validation",
                )
            else:
                validate_multitarget_regression_support(
                    multitarget_y,
                    phase=self.df_analyze_phase or "internal cross-validation",
                )

        unstratified = bool(y_train.attrs.get(_UNSTRATIFIED_SPLIT_ATTR, False))
        y_cnts = np.unique(y_train.apply(str), return_counts=True)[1]
        if len(y_cnts) <= 1 and not unstratified:
            raise RuntimeError(
                "Split function recieved target variable which appears to be "
                "constant. This should have been caught much earlier in df-analyze "
                f"and so is likely a bug in df-analyze code. df-analyze phase: {self.phase_info}\n"
                f"Target variable:\n{y_train}"
            )
        y: Series
        ix_shuf: Optional[ndarray] = None
        if self.shuffle and not self.grouped:
            rng = np.random.default_rng(seed=self.seed)
            n = len(X_train)
            ix_shuf = rng.permutation(n)
            X = X_train.iloc[ix_shuf]
            y = y_train.iloc[ix_shuf]
            g = None if g_train is None else g_train.iloc[ix_shuf]
        else:
            X, y, g = X_train, y_train, g_train

        def to_original(ix: ndarray) -> ndarray:
            if ix_shuf is None:
                return ix
            return ix_shuf[ix]

        # First attempt: try doing grouped stratified and check target level counts
        kf = (GroupKFold if self.grouped else KFold) if unstratified else self.kf
        kf_args = self._kf_args(kf=kf)
        cv = kf(**kf_args)
        if isinstance(cv, StratifiedKFold):
            cv_args = dict(X=X, y=y)  # only reachable for non-grouped splitting
        else:
            cv_args = dict(X=X, y=y, groups=g)
        splits: list[tuple[ndarray, ndarray]] = []
        ix_tr: ndarray
        ix_t: ndarray
        initial_split_fail = False
        insufficient_groups = (
            self.grouped
            and g is not None
            and int(g.nunique(dropna=False)) < self.n_splits
        )
        if insufficient_groups:
            initial_split_fail = True
        else:
            for ix_tr, ix_t in cv.split(**cv_args):  # type: ignore
                if (
                    self.is_cls
                    and not unstratified
                    and self._split_fail(y, ix_tr=ix_tr, ix_t=ix_t)
                ):
                    initial_split_fail = True
                    splits = []
                    break
                splits.append((to_original(ix_tr), to_original(ix_t)))
        if not initial_split_fail and (
            multitarget_y is None
            or not self._multitarget_split_fail(multitarget_y, splits)
        ):
            self._assert_group_disjoint(splits, g)
            self.effective_n_splits = len(splits)
            return splits, initial_split_fail

        if multitarget_y is not None:
            retried = self._retry_multitarget_splits(
                X_train=X_train,
                y_train=y_train,
                g_train=g_train,
                multitarget_y=multitarget_y,
                unstratified=unstratified,
            )
            if retried is not None:
                if self.warn:
                    warn(
                        "The initial multi-target cross-validation partition did not "
                        "preserve every target level in every fold. A deterministic "
                        "re-seeded partition with adequate per-target support was used."
                    )
                self._assert_group_disjoint(retried, g_train)
                self.effective_n_splits = len(retried)
                return retried, False
            if self.grouped and not self.allow_group_fallback:
                raise self._multitarget_informative_error(multitarget_y)
            if self.no_fallback:
                raise self._multitarget_informative_error(multitarget_y)
            splits = []

        if self.grouped and not self.allow_group_fallback:
            raise RuntimeError(
                "Could not create group-disjoint folds with adequate target-level "
                f"support. {self.phase_info}"
            )
        if self.grouped:
            safe_fallback = self._group_safe_fallback_splits(
                X_train=X,
                y_train=y,
                g_train=g,
                multitarget_y=multitarget_y,
                unstratified=unstratified,
            )
            if safe_fallback is None:
                raise RuntimeError(
                    "Could not create group-disjoint folds with adequate target-level "
                    "support, even after trying fewer stratified folds and "
                    f"unstratified GroupKFold. {self.phase_info}"
                )
            splits, strategy = safe_fallback
            self.effective_n_splits = len(splits)
            self.fallback_strategy = strategy
            if self.warn:
                requested_design = (
                    "grouped, stratified" if not unstratified else "grouped"
                )
                warn(
                    f"Could not create the requested {requested_design} "
                    f"{self.n_splits}-fold partition with adequate target support. "
                    f"Using {strategy}. Group membership remains disjoint between "
                    "training and validation in every fold."
                )
            return splits, True
        if self.no_fallback:
            raise self._informative_error(y_train)
        if self.phase_info is None:
            self.phase_info = ""
        self.phase_info = f"{self.phase_info} - Splitting fallback attempt."

        # fallback attempt
        kf_args = self._kf_args(kf=self.kf_fallback)
        cv = self.kf_fallback(**kf_args)
        if isinstance(cv, StratifiedKFold):
            cv_args = dict(X=X, y=y)  # grouped fallbacks returned above
        else:
            cv_args = dict(X=X, y=y, groups=g)
        for ix_tr, ix_t in cv.split(**cv_args):  # type: ignore
            if self.is_cls and self._split_fail(y, ix_tr=ix_tr, ix_t=ix_t):
                # not worried about the regression fallback case
                raise self._informative_error(y_train)
            splits.append((to_original(ix_tr), to_original(ix_t)))

        if multitarget_y is not None and self._multitarget_split_fail(
            multitarget_y, splits
        ):
            raise self._multitarget_informative_error(multitarget_y)
        return splits, initial_split_fail

    def _group_safe_fallback_splits(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series],
        multitarget_y: Optional[DataFrame],
        unstratified: bool,
    ) -> Optional[tuple[list[tuple[ndarray, ndarray]], str]]:
        """Find a usable fallback without ever splitting a group across folds."""
        if g_train is None:
            return None

        n_groups = int(g_train.nunique(dropna=False))
        max_splits = min(self.n_splits, n_groups)
        candidates: list[tuple[AnyKFold, int]] = []

        # Try fewer stratified folds before dropping stratification.
        if self.is_cls and not unstratified:
            for n_splits in range(min(self.n_splits - 1, max_splits), 1, -1):
                candidates.append((StratifiedGroupKFold, n_splits))

        # If stratification remains infeasible, relax it while retaining groups.
        for n_splits in range(max_splits, 1, -1):
            candidates.append((GroupKFold, n_splits))

        for splitter_cls, n_splits in candidates:
            splitter = splitter_cls(
                n_splits=n_splits,
                shuffle=self.shuffle,
                random_state=self.seed if self.shuffle else None,
            )
            try:
                splits = [
                    (np.asarray(ix_tr, dtype=int), np.asarray(ix_t, dtype=int))
                    for ix_tr, ix_t in splitter.split(
                        X=X_train,
                        y=y_train,
                        groups=g_train,
                    )
                ]
            except ValueError:
                continue
            if (
                self.is_cls
                and not unstratified
                and any(
                    self._split_fail(y_train, ix_tr=ix_tr, ix_t=ix_t)
                    for ix_tr, ix_t in splits
                )
            ):
                continue
            if multitarget_y is not None and self._multitarget_split_fail(
                multitarget_y, splits
            ):
                continue
            self._assert_group_disjoint(splits, g_train)
            strategy = f"{splitter_cls.__name__} with {n_splits} folds"
            return splits, strategy
        return None

    def _assert_group_disjoint(
        self,
        splits: list[tuple[ndarray, ndarray]],
        groups: Optional[Series],
    ) -> None:
        if not self.grouped:
            return
        if groups is None:
            raise RuntimeError(
                "Internal splitting error: grouped folds were created without groups."
            )
        group_codes, _ = factorize(groups, sort=False, use_na_sentinel=True)
        for fold, (ix_tr, ix_t) in enumerate(splits):
            overlap = np.intersect1d(group_codes[ix_tr], group_codes[ix_t])
            if overlap.size > 0:
                raise RuntimeError(
                    "Internal splitting error: grouped cross-validation produced "
                    f"overlapping training and validation groups in fold {fold}."
                )

    def _retry_multitarget_splits(
        self,
        X_train: DataFrame,
        y_train: Series,
        g_train: Optional[Series],
        multitarget_y: DataFrame,
        unstratified: bool,
    ) -> Optional[list[tuple[ndarray, ndarray]]]:
        splitter_cls: AnyKFold
        if unstratified:
            splitter_cls = GroupKFold if self.grouped else KFold
        else:
            splitter_cls = self.kf

        base_seed = SEED if self.seed is None else int(self.seed)
        for attempt in range(_MULTITARGET_SPLIT_RETRIES):
            seed = (base_seed + attempt) % (2**32 - 1)
            splitter = splitter_cls(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed,
            )
            if isinstance(splitter, StratifiedKFold):
                cv_args = dict(X=X_train, y=y_train)
            else:
                cv_args = dict(X=X_train, y=y_train, groups=g_train)
            candidate = [
                (np.asarray(ix_tr, dtype=int), np.asarray(ix_t, dtype=int))
                for ix_tr, ix_t in splitter.split(**cv_args)  # type: ignore
            ]
            if not unstratified and any(
                self._split_fail(y_train, ix_tr=ix_tr, ix_t=ix_t)
                for ix_tr, ix_t in candidate
            ):
                continue
            if not self._multitarget_split_fail(multitarget_y, candidate):
                return candidate
        return None

    def _kf_args(self, kf: AnyKFold) -> dict[str, Any]:
        """

                    KFold(n_splits=5, *, shuffle=False, random_state=None)
          StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
        StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)
                  GroupKFold(n_splits=5)

                       KFold.split(X, y=None, groups=None)
             StratifiedKFold.split(X, y,      groups=None)
        StratifiedGroupKFold.split(X, y=None, groups=None)
                  GroupKFold.split(X, y=None, groups=None)
        """
        if self.grouped:
            # Grouped repetitions must shuffle groups, not rows. GroupKFold gained
            # this seeded behavior in the minimum supported scikit-learn version.
            return {
                "n_splits": self.n_splits,
                "shuffle": self.shuffle,
                "random_state": self.seed if self.shuffle else None,
            }
        return dict(n_splits=self.n_splits, shuffle=False, random_state=None)

    def _split_fail(self, y: Series, ix_tr: ndarray, ix_t: ndarray) -> bool:
        if not self.is_cls:
            return False
        y_str = y.apply(str).to_numpy()  # .astype(str) unreliable
        y_tr = y_str[ix_tr]
        y_t = y_str[ix_t]
        tr_cnts = np.unique(y_tr, return_counts=True)[1]
        t_cnts = np.unique(y_t, return_counts=True)[1]
        if len(tr_cnts) <= 1:
            return True  # definitely cannot proceed, degen training set

        # Can proceed, technically, but will get errors for later AUROC and
        # other metrics...
        if len(t_cnts) <= 1:
            return True

        tr_cnts_n_min = tr_cnts.min()
        if tr_cnts_n_min < N_TARG_LEVEL_MIN_TRAIN_INTERNAL:
            return True

        t_cnts_n_min = t_cnts.min()
        if t_cnts_n_min < N_TARG_LEVEL_MIN_TEST_INTERNAL:
            return True

        return False

    def _multitarget_split_fail(
        self,
        y: DataFrame,
        splits: list[tuple[ndarray, ndarray]],
    ) -> bool:
        return len(self._multitarget_split_failures(y, splits)) > 0

    def _multitarget_split_failures(
        self,
        y: DataFrame,
        splits: list[tuple[ndarray, ndarray]],
    ) -> list[str]:
        failures: list[str] = []
        for fold, (ix_tr, ix_t) in enumerate(splits):
            for target in y.columns:
                if not self.is_cls:
                    train = np.asarray(y[target].iloc[ix_tr], dtype=float)
                    validation = np.asarray(y[target].iloc[ix_t], dtype=float)
                    if np.unique(train).size < 2:
                        failures.append(
                            f"fold {fold}, target '{target}': training target is constant"
                        )
                    if np.unique(validation).size < 2:
                        failures.append(
                            f"fold {fold}, target '{target}': validation target is constant"
                        )
                    continue
                values = y[target].astype(str)
                levels = values.unique()
                train_counts = (
                    values.iloc[ix_tr].value_counts().reindex(levels, fill_value=0)
                )
                test_counts = (
                    values.iloc[ix_t].value_counts().reindex(levels, fill_value=0)
                )
                for level in levels:
                    train_count = int(train_counts.loc[level])
                    test_count = int(test_counts.loc[level])
                    if (
                        train_count < N_TARG_LEVEL_MIN_TRAIN_INTERNAL
                        or test_count < N_TARG_LEVEL_MIN_TEST_INTERNAL
                    ):
                        failures.append(
                            f"fold {fold}, target '{target}', level {level!r}: "
                            f"train={train_count}, validation={test_count}"
                        )
        return failures

    def _multitarget_informative_error(self, y: DataFrame) -> RuntimeError:
        grouped = " group-disjoint" if self.grouped else ""
        if not self.is_cls:
            return RuntimeError(
                f"Could not create{grouped} {self.n_splits}-fold cross-validation "
                "partitions in which every multi-target regression target varies "
                "within every training and validation fold. "
                f"{self.phase_info}\n\n"
                "Distinct values per target:\n\n"
                f"{y.nunique(dropna=False).rename('Distinct Values').to_markdown()}"
            )

        counts = []
        for target in y.columns:
            for level, count in y[target].astype(str).value_counts().items():
                counts.append(
                    {
                        "Target": str(target),
                        "Target Level": str(level),
                        "Count": int(count),
                    }
                )
        return RuntimeError(
            f"Could not create{grouped} {self.n_splits}-fold cross-validation "
            "partitions with every level of every multi-target classification "
            f"target represented by at least {N_TARG_LEVEL_MIN_TRAIN_INTERNAL} "
            "training rows and "
            f"{N_TARG_LEVEL_MIN_TEST_INTERNAL} validation rows in every fold. "
            f"{self.phase_info}\n\n"
            "Global target-level counts:\n\n"
            f"{DataFrame(counts).to_markdown(index=False)}"
        )

    def _informative_error(self, y: Series) -> RuntimeError:
        unqs, cnts = np.unique(y.apply(str).to_numpy(), return_counts=True)
        unqs = (
            Series(unqs)
            .apply(lambda lab: self.labels[lab] if lab in self.labels else lab)
            .values
        )

        df = DataFrame(
            index=Index(data=unqs, name="Target Level"),
            columns=["Count"],
            data=cnts,
        )
        info = df.to_markdown(tablefmt="simple")
        kf = self.kf.__name__
        return RuntimeError(
            f"Attempted to split target variable '{y.name}' with {kf}, but found "
            "insufficient samples per target level, i.e. the classification "
            "target has undersampled levels. This means that one or more of "
            "the target levels (classes) cannot support the requested internal "
            f"{self.n_splits}-fold design. That is, at least one cross-validation "
            "training split will always "
            f"result in less than {N_TARG_LEVEL_MIN_TRAIN_INTERNAL} samples per "
            "level, or that at least one cross-validation test split will always "
            f"have less than {N_TARG_LEVEL_MIN_TEST_INTERNAL} samples per target "
            f"level. df-analyze phase: {self.phase_info}\n\n"
            "The current data thus has target classes with insufficient data for "
            f"meaningful generalization or stable performance estimates via automated "
            "machine learning via df-analyze. Consider removing instances of these "
            "rare target levels, or collect more data.\n\n"
            "Observed target level counts:\n\n"
            f"{info}"
        )


class ApproximateStratifiedGroupSplit:
    def __init__(
        self,
        train_size: float = 0.6,
        is_classification: bool = True,
        grouped: bool = False,
        labels: dict[int, str] | None = None,
        shuffle: bool = False,
        seed: int | None = SEED,
        warn_on_fallback: bool = True,
        allow_group_fallback: bool = False,
        warn_on_large_size_diff: bool = True,
        df_analyze_phase: str | None = None,
    ) -> None:
        self.desired_train_size = train_size
        self.warn_on_large_size_diff: bool = warn_on_large_size_diff
        n_splits = int(1 / (1 - train_size))
        if n_splits == 1:
            n_splits = 2
        self.kf = OmniKFold(
            n_splits=n_splits,
            is_classification=is_classification,
            grouped=grouped,
            labels=labels,
            shuffle=shuffle,
            seed=seed,
            warn_on_fallback=warn_on_fallback,
            allow_group_fallback=allow_group_fallback,
            df_analyze_phase=df_analyze_phase,
        )

    def split(
        self, X_train: DataFrame, y_train: Series, g_train: Series | None = None
    ) -> tuple[tuple[ndarray, ndarray], bool]:
        with catch_warnings():
            filterwarnings(
                "ignore", message="The least populated class in y", category=UserWarning
            )
            splits, group_fail = self.kf.split(X_train, y_train, g_train)

        # TODO:
        desired = self.desired_train_size
        (ix_tr, ix_t), d_min = self._get_best_split(y_train, splits)

        unstratified = bool(y_train.attrs.get(_UNSTRATIFIED_SPLIT_ATTR, False))
        if not unstratified and self.kf._split_fail(y=y_train, ix_tr=ix_tr, ix_t=ix_t):
            if self.kf.grouped:
                raise RuntimeError(
                    "Could not select a group-disjoint holdout split with adequate "
                    f"target-level support. {self.kf.phase_info}"
                )

            ss_args: Mapping = dict(
                train_size=self.desired_train_size, n_splits=1, random_state=self.kf.seed
            )
            ss = StratSplit(**ss_args) if self.kf.is_cls else ShuffleSplit(**ss_args)
            ix_tr, ix_t = next(ss.split(y_train.to_frame(), y_train))
            if self.kf._split_fail(y=y_train, ix_tr=ix_tr, ix_t=ix_t):
                self.kf.phase_info += " - ApproximateStratifiedGroupSplit fallback"
                raise self.kf._informative_error(y_train)

        self.kf._assert_group_disjoint([(ix_tr, ix_t)], g_train)

        # d_min = min(d_train, d_test)
        if self.warn_on_large_size_diff and (
            d_min > APPROXIMATE_GROUP_SPLIT_DIFF_WARN_THRESHOLD
        ):
            tst = round(1 - desired, 2)
            warn(
                f"User requested an initial holdout test set size of approximately {tst}, "
                "But the closest approximate Could not generate initial holdout "
                "https://github.com/scikit-learn/scikit-learn/issues/12076#issuecomment-2047948563"
            )

        return (ix_tr, ix_t), group_fail

    def _get_best_split(
        self, y_train: Series, splits: list[tuple[ndarray, ndarray]]
    ) -> tuple[tuple[ndarray, ndarray], float]:
        n = len(y_train)
        desired = self.desired_train_size

        p_splits = [len(splits[i][0]) / n for i in range(len(splits))]
        ds = [abs(p - desired) for p in p_splits]
        ix_min = np.argmin(ds)
        d_min = min(ds)
        ix_tr, ix_t = splits[ix_min]
        return (ix_tr, ix_t), d_min

# ref: Multiclass & multioutput learning in scikit-learn: https://scikit-learn.org/stable/modules/multiclass.html

from __future__ import annotations

import json
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, cast
from typing import Generator as Gen
from warnings import warn

import jsonpickle
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

from df_analyze._constants import (
    N_CAT_LEVEL_MIN,
    N_MULTITARGET_LEVEL_MIN,
    N_TARG_LEVEL_MIN,
    SEED,
    UNIVARIATE_PRED_MAX_N_SAMPLES,
)
from df_analyze.analysis.univariate.describe import describe_all_features
from df_analyze.enumerables import FeatureDownsampleMethod, NanHandling, ValidationMethod
from df_analyze.preprocessing.cleaning import (
    clean_regression_target,
    clean_regression_targets,
    deflate_categoricals,
    drop_target_nans,
    drop_unusable,
    encode_categoricals,
    encode_target,
    encode_targets,
    handle_continuous_nans,
    normalize_continuous,
)
from df_analyze.preprocessing.inspection.inspection import (
    ClsTargetInfo,
    InspectionResults,
    RegTargetInfo,
    convert_categoricals,
    inspect_target,
    unify_nans,
)
from df_analyze.preprocessing.targets import TargetSpec, as_target_list
from df_analyze.splitting import (
    ApproximateStratifiedGroupSplit,
    MultiTargetSplitInfo,
    regression_split_label,
    y_split_label,
    y_split_label_info,
)
from df_analyze.timing import timed


@dataclass
class PrepFiles:
    X_raw: str = "X.parquet"
    X_tabpfn_raw: str = "X_tabpfn.parquet"
    X_cont_raw: str = "X_cont.parquet"
    X_cat_raw: str = "X_cat.parquet"
    y_raw: str = "y.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    ix_train: str = "idx_train.json"
    ix_tests: str = "idx_tests.json"
    info: str = "info.json"


@dataclass
class PrepFilesTrain:
    X_raw: str = "X_train.parquet"
    X_tabpfn_raw: str = "X_train_tabpfn.parquet"
    X_cont_raw: str = "X_train_cont.parquet"
    X_cat_raw: str = "X_train_cat.parquet"
    y_raw: str = "y_train.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


def usable_training_indices(
    df: DataFrame,
    target: TargetSpec,
    is_classification: bool,
    indices: ndarray,
) -> ndarray:
    """Return training rows that remain after target cleaning."""
    target_cols = as_target_list(target)
    targets = unify_nans(df.iloc[indices][target_cols].copy())
    keep = ~targets.isna().any(axis=1)

    if is_classification and len(target_cols) == 1:
        values = targets.loc[keep, target_cols[0]].astype(str)
        levels, counts = np.unique(values, return_counts=True)
        rare = levels[counts <= N_TARG_LEVEL_MIN]
        if len(rare) > 0:
            keep &= ~targets[target_cols[0]].astype(str).isin(rare)
    return np.asarray(indices, dtype=int)[keep.to_numpy()]


def _ensure_target_levels_in_training(
    y: DataFrame,
    idx_train: ndarray,
    idx_test: ndarray,
    groups: Optional[Series] = None,
) -> tuple[ndarray, ndarray, int]:
    train = np.asarray(idx_train, dtype=int)
    test = np.asarray(idx_test, dtype=int)
    moved = np.array([], dtype=int)
    for target in y.columns:
        train_levels = set(y.iloc[train][target].astype(str))
        missing = ~y.iloc[test][target].astype(str).isin(train_levels)
        if not missing.any():
            continue
        rows = test[missing.to_numpy()]
        if groups is not None:
            moved_groups = groups.iloc[rows].unique()
            rows = test[groups.iloc[test].isin(moved_groups).to_numpy()]
        moved = np.union1d(moved, rows).astype(int)
        train = np.union1d(train, rows).astype(int)
        test = np.setdiff1d(test, rows).astype(int)
    if len(test) == 0:
        raise ValueError(
            "Could not create a non-empty holdout while keeping every target level "
            "in the training data."
        )
    return np.sort(train), np.sort(test), len(moved)


def raw_train_test_indices(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str],
    is_classification: bool,
    test_size: Union[int, float],
    seed: int | None,
) -> tuple[ndarray, ndarray, Optional[MultiTargetSplitInfo]]:
    """Split rows before fitting any predictor preprocessing."""
    target_cols = as_target_list(target)
    targets = unify_nans(df[target_cols].copy())
    rows = usable_training_indices(
        df,
        target,
        is_classification,
        np.arange(len(df), dtype=int),
    )
    if len(rows) < 2:
        raise ValueError("Not enough samples remain to create a train/test split.")
    if isinstance(test_size, int):
        test_size = test_size / len(rows)
    train_size = 1 - test_size

    y = targets.iloc[rows].reset_index(drop=True)
    split_audit: Optional[MultiTargetSplitInfo] = None
    if grouper is None:
        if is_classification:
            if len(target_cols) > 1:
                split_y, split_audit = y_split_label_info(y)
            else:
                split_y = y.iloc[:, 0].astype(str)
            splitter = StratifiedShuffleSplit(
                train_size=train_size, n_splits=1, random_state=seed
            )
            idx_train, idx_test = next(
                splitter.split(split_y.to_frame(), split_y)
            )
        else:
            splitter = ShuffleSplit(
                train_size=train_size, n_splits=1, random_state=seed
            )
            idx_train, idx_test = next(splitter.split(y))
    else:
        groups = df.iloc[rows][grouper].reset_index(drop=True)
        splitter = ApproximateStratifiedGroupSplit(
            train_size=train_size,
            is_classification=is_classification,
            grouped=True,
            labels=None,
            seed=seed,
            warn_on_fallback=True,
            allow_group_fallback=False,
            warn_on_large_size_diff=True,
            df_analyze_phase="Initial holdout splitting",
        )
        if is_classification and len(target_cols) > 1:
            split_y, split_audit = y_split_label_info(y)
        elif is_classification:
            split_y = y.iloc[:, 0].astype(str)
        else:
            split_y = regression_split_label(
                y if len(target_cols) > 1 else y.iloc[:, 0]
            )
        (idx_train, idx_test), _ = splitter.split(
            split_y.to_frame(), split_y, groups
        )

    if is_classification and len(target_cols) > 1:
        local_groups = (
            None
            if grouper is None
            else df.iloc[rows][grouper].reset_index(drop=True)
        )
        idx_train, idx_test, moved = _ensure_target_levels_in_training(
            y,
            idx_train,
            idx_test,
            local_groups,
        )
        if moved > 0:
            message = (
                f"Moved {moved} holdout rows to training so every multi-target "
                "classification level is represented in the training data."
            )
            warn(message)
            if split_audit is not None:
                split_audit.reason = f"{split_audit.reason} {message}"

    return rows[idx_train], rows[idx_test], split_audit


@dataclass
class PrepFilesTest:
    X_raw: str = "X_test.parquet"
    X_tabpfn_raw: str = "X_test_tabpfn.parquet"
    X_cont_raw: str = "X_test_cont.parquet"
    X_cat_raw: str = "X_test_cat.parquet"
    y_raw: str = "y_test.parquet"
    g_raw: str = "g.parquet"
    labels: str = "labels.parquet"
    info: str = "info.json"


@dataclass
class MultiTargetAudit:
    target_names: list[str]
    is_classification: bool
    n_original_rows: int
    n_final_rows: int
    missing_by_target: dict[str, int]
    class_counts_by_target: dict[str, dict[str, int]]
    low_support_labels_by_target: dict[str, dict[str, int]]

    @property
    def n_missing_target_rows(self) -> int:
        return self.n_original_rows - self.n_final_rows

    def to_markdown(self) -> str:
        task = "classification" if self.is_classification else "regression"
        sections = [
            "# Multi-Target Target Audit\n\n",
            f"Task type: {task}\n",
            f"Targets: {', '.join(self.target_names)}\n",
            f"Original rows: {self.n_original_rows}\n",
            f"Final rows: {self.n_final_rows}\n",
            f"Rows dropped due to a missing target: {self.n_missing_target_rows}\n\n",
        ]
        missing = DataFrame(
            {
                "target": list(self.missing_by_target),
                "missing rows": list(self.missing_by_target.values()),
            }
        )
        sections.extend(["## Missing Targets\n\n", missing.to_markdown(index=False)])

        if self.is_classification:
            rows = []
            for target, counts in self.class_counts_by_target.items():
                low_support = self.low_support_labels_by_target.get(target, {})
                for label, count in counts.items():
                    rows.append(
                        {
                            "target": target,
                            "level": label,
                            "count": count,
                            "low support": label in low_support,
                        }
                    )
            if rows:
                sections.extend(
                    [
                        "\n\n## Target Level Counts\n\n",
                        DataFrame(rows).to_markdown(index=False),
                    ]
                )
        return "".join(sections)


@dataclass
class PreparationInfo:
    is_classification: bool
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    n_samples_dropped_via_target_NaNs: int
    n_cont_indicator_added: int
    target_info: Optional[Union[RegTargetInfo, ClsTargetInfo]]
    runtimes: dict[str, float]
    multitarget_audit: Optional[MultiTargetAudit] = None
    split_audit: Optional[MultiTargetSplitInfo] = None

    def to_markdown(self) -> str:
        sections = []
        og = self.original_shape
        fs = self.final_shape
        task = "Classification" if self.is_classification else "Regression"
        orig_shape = f"{og[0]} samples × {og[1]} features"
        final_shape = f"{fs[0]} samples × {fs[1]} features"
        n_drop = self.n_samples_dropped_via_target_NaNs
        n_ind = self.n_cont_indicator_added
        funcs, times = zip(*self.runtimes.items())

        sections.append("# Data Preparation Summary\n\n")
        sections.append(f"Task:                   {task}\n")
        sections.append(f"Data original shape:    {orig_shape}\n")
        sections.append(f"Data final shape:       {final_shape}\n")
        if self.target_info is not None:
            sections.append(f"Target feature:         {self.target_info.name}\n")
        sections.append("\n")
        sections.append(f"Samples dropped due to NaN target: {n_drop}\n")
        sections.append(f"Indicator variables added for continuous NaNs: {n_ind}\n\n")
        sections.append("# Processing Times\n\n")
        sections.append(
            DataFrame(
                data=(np.array(times) * 1000).round(),
                columns=["runtime (ms)"],
                index=Series(name="computation", data=funcs),
            ).to_markdown()
        )

        multitarget_audit = getattr(self, "multitarget_audit", None)
        split_audit = getattr(self, "split_audit", None)
        if multitarget_audit is not None:
            sections.extend(["\n\n", multitarget_audit.to_markdown()])
        if split_audit is not None:
            sections.extend(["\n\n", split_audit.to_markdown()])

        return "".join(sections)

    def to_json(self, path: Path) -> None:
        path.write_text(str(jsonpickle.encode(self)), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> Optional[PreparationInfo]:
        content = path.read_text()
        if content.strip().replace("\n", "") == "":
            return None
        return cast(PreparationInfo, jsonpickle.decode(content))


def _decoded_target_level(
    target: str,
    value: Any,
    labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]],
) -> str:
    if labels is None or not all(isinstance(item, dict) for item in labels.values()):
        return str(value)
    mapping = cast(dict[str, dict[int, str]], labels).get(target, {})
    try:
        return str(mapping.get(int(value), value))
    except (TypeError, ValueError):
        return str(value)


def _build_multitarget_audit(
    raw_targets: DataFrame,
    y: DataFrame,
    labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]],
    is_classification: bool,
) -> MultiTargetAudit:
    missing = raw_targets.isna()
    counts_by_target: dict[str, dict[str, int]] = {}
    low_support_by_target: dict[str, dict[str, int]] = {}

    if is_classification:
        for col in y.columns:
            target = str(col)
            counts: dict[str, int] = {}
            low_support: dict[str, int] = {}
            for encoded, count in y[col].value_counts().sort_index().items():
                label = _decoded_target_level(target, encoded, labels)
                count_int = int(count)
                counts[label] = count_int
                if count_int <= N_MULTITARGET_LEVEL_MIN:
                    low_support[label] = count_int
            counts_by_target[target] = counts
            if low_support:
                low_support_by_target[target] = low_support

    return MultiTargetAudit(
        target_names=[str(col) for col in raw_targets.columns],
        is_classification=is_classification,
        n_original_rows=int(len(raw_targets)),
        n_final_rows=int(len(y)),
        missing_by_target={str(col): int(missing[col].sum()) for col in raw_targets},
        class_counts_by_target=counts_by_target,
        low_support_labels_by_target=low_support_by_target,
    )


def viable_subsample(
    df: DataFrame,
    target: Series,
    n_sub: int = 2000,
    rng: Optional[Generator] = None,
) -> ndarray:
    rng = rng or np.random.default_rng()
    unqs, cnts = np.unique(target, return_counts=True)
    n_min = N_CAT_LEVEL_MIN
    idx_count = cnts < n_min
    idx_final = np.arange(len(target))
    drop_vals = unqs[idx_count]
    unqs = unqs[~idx_count]
    idx_keep = ~target.isin(drop_vals)
    X: ndarray
    y: ndarray
    idx_final = idx_final[idx_keep]
    X = df.copy(deep=True).values[idx_keep]
    y = target.copy(deep=True).to_numpy()[idx_keep]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Keep fail"

    # shuffle once to allow getting random first n of each class later
    idx_shuffle = rng.permutation(len(y))
    idx_final = idx_final[idx_shuffle]
    X = X[idx_shuffle]
    y = y[idx_shuffle]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Shuffle fail"
    #
    idx = np.argsort(y)
    idx_final = idx_final[idx]
    X = X[idx]
    y = y[idx]
    assert np.bincount(y).min() >= N_CAT_LEVEL_MIN, "Argsort fail"
    # this gives e.g. stops[i]:stops[i+1] are class i
    stops = np.searchsorted(y, unqs)
    cls_idxs = []
    for i in range(len(stops) - 1):
        start, stop = stops[i], stops[i + 1]
        shuf = rng.permutation(stop - start)
        cls_idxs.append(np.arange(start, stop)[shuf][:n_min])
    shuf = rng.permutation(len(y) - stops[-1])
    cls_idxs.append(np.arange(stops[-1], len(y))[shuf][:n_min])

    idx_required = np.concatenate(cls_idxs).ravel()
    idx_final_req = idx_final[idx_required]
    y_req = y[idx_required]
    assert np.bincount(y_req).min() >= N_CAT_LEVEL_MIN, "collect fail"

    n_remain = n_sub - len(y_req)
    if n_remain <= 0:
        return idx_final_req

    idx_remain = np.ones_like(y, dtype=bool)
    idx_remain[idx_required] = False
    idx_final_rem = idx_final[idx_remain]
    X_remain = X[idx_remain]
    n_remain = min(n_remain, len(y))
    if n_remain >= len(y):
        idx_full = np.concatenate([idx_final_req, idx_final_rem])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "remain fail"
        return idx_full
    else:
        smax = 2**32 - 1
        if hasattr(rng.bit_generator, "seed_seq"):
            ent = rng.bit_generator.seed_seq.entropy  # type: ignore
        elif hasattr(rng.bit_generator, "_seed_seq"):
            ent = rng.bit_generator._seed_seq.entropy  # type: ignore
        else:
            ent = np.random.randint(1, smax)
        while ent > smax:
            ent //= 2

        ss = ShuffleSplit(
            n_splits=1,
            train_size=n_remain,
            random_state=ent,
        )
        idx = next(ss.split(X_remain))[0]
        idx_final_strat = idx_final_rem[idx]
        idx_full = np.concatenate([idx_final_req, idx_final_strat])
        assert np.bincount(target[idx_full]).min() >= N_CAT_LEVEL_MIN, "concat fail"
        return idx_full


class PreparedData:
    def __init__(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        groups: Optional[Series],
        is_classification: Optional[bool] = None,
        X_cont: Optional[DataFrame] = None,
        X_cat: Optional[DataFrame] = None,
        X_tabpfn: Optional[DataFrame] = None,
        feature_lineage: Optional[dict[str, str]] = None,
        labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]] = None,
        ix_train: Optional[ndarray] = None,
        ix_tests: Optional[list[ndarray]] = None,
        tests_method: Optional[ValidationMethod] = ValidationMethod.List,
        inspection: Optional[InspectionResults] = None,
        info: Optional[PreparationInfo] = None,
        phase: Optional[Literal["train", "test"]] = None,
        validate: bool = True,
    ) -> None:
        # Attempt to automatically infer classification problem based on y.dtype
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        if is_classification is None:
            kind = y.dtypes.iloc[0].kind if isinstance(y, DataFrame) else y.dtype.kind
            if kind == "O":
                if isinstance(y, DataFrame):
                    raise ValueError(
                        "Got object dtype for multi-target y. Please encode targets "
                        "before constructing PreparedData."
                    )
                # run cleaning.encode_target with X, y
                warn(
                    f"Found 'object' dtype for argument y (name='{y.name}'). Attempting to "
                    "automatically label encode. If this is undesirable, ensure that your "
                    "target `y` has the proper dtype, e.g. `y = y.astype(np.float64)` if "
                    "regression, or `y = y.astype(np.int64)` if classification. "
                )
                X, y, labels, _, _ = encode_target(X, y, ix_train, ix_tests, _warn=True)
                self.is_classification = True
            else:
                kinds = {"f": False, "i": True, "b": True, "u": True}
                if kind not in kinds:
                    raise ValueError(
                        f"Got argument for parameter `y` with unsupported data type: {y.dtype}"
                    )
                tname = (
                    "floating point"
                    if kind == "f"
                    else "signed/unsigned integer or boolean type"
                )
                self.is_classification = kinds[kind]
                warn(
                    f"Argument `is_classification` left at default. Inferred "
                    f"`is_classification={self.is_classification}`, since y is {tname}. "
                    "If this is incorrect, or to silence this warning, specify the value for "
                    f"`is_classification`.\ny={y}"
                )
        else:
            self.is_classification = is_classification
        self.inspection: Optional[InspectionResults] = inspection
        self.info: Optional[PreparationInfo] = info
        self.files: PrepFiles = PrepFiles()
        self.phase = phase

        if validate:
            X, X_cont, X_cat, y = self.validate(X, X_cont, X_cat, y)

        if X_tabpfn is None:
            X_tabpfn = X.copy(deep=True)
        elif len(X_tabpfn) != len(X):
            raise ValueError(
                f"TabPFN data number of samples ({len(X_tabpfn)}) does not "
                f"match number of samples in processed data ({len(X)})"
            )
        else:
            X_tabpfn = X_tabpfn.copy(deep=True)
        X_tabpfn.reset_index(drop=True, inplace=True)
        X_tabpfn.index = X.index.copy(deep=True)

        if groups is not None:
            if len(groups) != len(X):
                raise ValueError(
                    f"Grouping data number of samples ({len(groups)}) does not "
                    f"match number of samples in processed data ({len(X)})"
                )
            groups = groups.copy()
            groups.reset_index(drop=True, inplace=True)
            groups.index = X.index.copy(deep=True)

        self.X = self.rename_cols(X)
        self.X_tabpfn = self.rename_cols(X_tabpfn)
        self.X_cont: Optional[DataFrame] = (
            None if X_cont is None else self.rename_cols(X_cont)
        )
        self.X_cat: Optional[DataFrame] = (
            None if X_cat is None else self.rename_cols(X_cat)
        )
        self.feature_lineage = feature_lineage or self._infer_feature_lineage()
        if isinstance(y, DataFrame):
            self.y: Union[Series, DataFrame] = y
            self.target_cols = y.columns.tolist()
            self.target = self.target_cols[0] if len(self.target_cols) > 0 else ""
        else:
            name = "target" if y.name is None else str(y.name)
            if y.name != name:
                y = y.rename(name)
            self.y = y
            self.target_cols = [name]
            self.target = name
        self.labels = labels or {}
        self.split_labels = (
            self.labels
            if isinstance(self.labels, dict)
            and all(isinstance(k, int) for k in self.labels.keys())
            else None
        )
        self.groups: Optional[Series] = groups

        self.ix_train: Optional[ndarray] = ix_train
        self.ix_tests: list[ndarray] = ix_tests or []
        self.tests_method = tests_method

    @property
    def num_classes(self) -> int:
        if not self.is_classification:
            return 1
        if isinstance(self.y, DataFrame):
            return int(max(self.y[col].nunique() for col in self.y.columns))
        return len(np.unique(self.y))

    def _infer_feature_lineage(self) -> dict[str, str]:
        raw_cols = sorted(
            (str(col) for col in self.X_tabpfn.columns), key=len, reverse=True
        )
        raw_set = set(raw_cols)
        cat_cols = sorted(
            (() if self.X_cat is None else (str(col) for col in self.X_cat.columns)),
            key=len,
            reverse=True,
        )
        lineage: dict[str, str] = {}
        for processed in (str(col) for col in self.X.columns):
            if processed in raw_set:
                lineage[processed] = processed
                continue
            source = next(
                (
                    raw
                    for raw in raw_cols
                    if processed == f"{raw}_NAN"
                    or processed.startswith(f"{raw}_NAN_")
                ),
                None,
            )
            if source is None:
                source = next(
                    (
                        raw
                        for raw in cat_cols
                        if processed.startswith(f"{raw}_")
                        or processed.startswith(f"{raw}__")
                    ),
                    None,
                )
            if source is None:
                source = next(
                    (
                        raw
                        for raw in raw_cols
                        if processed.startswith(f"{raw}_")
                        or processed.startswith(f"{raw}__")
                    ),
                    processed,
                )
            lineage[processed] = source
        return lineage

    @staticmethod
    def _is_tabpfn_model(model: Any) -> bool:
        cls = model if isinstance(model, type) else model.__class__
        return cls.__name__.startswith("TabPFN")

    def model_matrix(
        self,
        model: Any,
        selected_cols: Optional[list[str] | slice] = None,
    ) -> DataFrame:
        if not self._is_tabpfn_model(model):
            if selected_cols is None or isinstance(selected_cols, slice):
                return self.X
            return self.X.loc[:, [col for col in selected_cols if col in self.X]]
        if selected_cols is None or isinstance(selected_cols, slice):
            return self.X_tabpfn
        mapped: list[str] = []
        for col in selected_cols:
            source = self.feature_lineage.get(str(col), str(col))
            if source in self.X_tabpfn.columns and source not in mapped:
                mapped.append(source)
        return self.X_tabpfn.loc[:, mapped]

    def get_splits(
        self, test_size: Union[int, float] = 0.4, seed: int | None = SEED
    ) -> Union[
        list[tuple[PreparedData, PreparedData]],
        Gen[tuple[PreparedData, PreparedData], None, None],
    ]:
        if isinstance(test_size, int):
            test_size = test_size / len(self.y)
        train_size = 1 - test_size

        if self.ix_train is None or self.ix_tests is None:
            # Old df-analyze behaviour prior to multiple test sets
            yield (self.split(train_size=train_size, seed=seed))
            return

        prep_train = self.subsample(self.ix_train)

        if self.tests_method is ValidationMethod.List:
            for ix_test in self.ix_tests:
                yield (prep_train, self.subsample(ix_test))
            return

        if self.tests_method is not ValidationMethod.LODO:
            raise ValueError(f"Impossible! Invalid ValidationMethod: {self.tests_method}")

        # now construct the indices needed
        ix_all = [self.ix_train, *self.ix_tests]
        ix_pairs = []
        for i, ix in enumerate(ix_all):
            ix_test = ix
            ix_trains = ix_all[:i] + ix_all[i + 1 :]
            ix_train = np.concatenate(ix_trains)
            ix_pairs.append((ix_train, ix_test))

        for ix_train, ix_test in ix_pairs:
            yield (self.subsample(ix_train), self.subsample(ix_test))

    def split(
        self,
        train_size: Union[int, float] = 0.6,
        seed: int | None = SEED,
    ) -> tuple[PreparedData, PreparedData]:
        y = self.y.copy()
        split_audit: Optional[MultiTargetSplitInfo] = None
        if self.groups is None:
            if self.is_classification:
                ss = StratifiedShuffleSplit(
                    train_size=train_size, n_splits=1, random_state=seed
                )
                if isinstance(y, DataFrame):
                    split_y, split_audit = y_split_label_info(y)
                    idx_train, idx_test = next(ss.split(split_y.to_frame(), split_y))
                else:
                    idx_train, idx_test = next(ss.split(y.to_frame(), y))
            else:
                ss = ShuffleSplit(train_size=train_size, n_splits=1, random_state=seed)
                idx_train, idx_test = next(ss.split(self.X))
        else:
            ss = ApproximateStratifiedGroupSplit(
                train_size=train_size,
                is_classification=self.is_classification,
                grouped=self.groups is not None,
                labels=self.split_labels,
                seed=seed,
                warn_on_fallback=True,
                allow_group_fallback=False,
                warn_on_large_size_diff=True,
                df_analyze_phase="Initial holdout splitting",
            )
            if self.is_classification and isinstance(y, DataFrame):
                split_y, split_audit = y_split_label_info(y)
                (idx_train, idx_test), group_fail = ss.split(
                    split_y.to_frame(), split_y, self.groups
                )
            else:
                split_y = regression_split_label(y)
                (idx_train, idx_test), group_fail = ss.split(
                    split_y.to_frame(), split_y, self.groups
                )

        prep_train = self.subsample(idx_train)
        prep_train.phase = "train"

        prep_test = self.subsample(idx_test)
        prep_test.phase = "test"

        if split_audit is not None:
            if prep_train.info is not None:
                prep_train.info.split_audit = deepcopy(split_audit)
            if prep_test.info is not None:
                prep_test.info.split_audit = deepcopy(split_audit)

        return prep_train, prep_test

    def subsample(self, idx: ndarray, validate: bool = True) -> PreparedData:
        try:
            X_sub = self.X.iloc[idx].reset_index(drop=True)
        except IndexError as e:
            raise IndexError(
                f"Couldn't subsample prepared data. Data shape: {self.X.shape}, "
                f"and subsampling indices range: [{idx.min()}, {idx.max()}]"
            ) from e
        X_cont, X_cat = self.X_cont, self.X_cat
        groups = None if self.groups is None else self.groups.iloc[idx]
        if self.info is not None:
            info_sub = deepcopy(self.info)
            info_sub.final_shape = X_sub.shape
        else:
            info_sub = None
        return PreparedData(
            X=X_sub,
            X_tabpfn=self.X_tabpfn.iloc[idx].reset_index(drop=True),
            feature_lineage=self.feature_lineage,
            X_cont=None if X_cont is None else X_cont.iloc[idx].reset_index(drop=True),
            X_cat=None if X_cat is None else X_cat.iloc[idx].reset_index(drop=True),
            y=self.y.iloc[idx].copy().reset_index(drop=True),
            groups=groups,
            labels=self.labels,
            inspection=self.inspection,
            info=info_sub,
            is_classification=self.is_classification,
            validate=validate,
        )

    def with_features(self, X: DataFrame, feature_origin: str) -> PreparedData:
        info = None if self.info is None else deepcopy(self.info)
        if info is not None:
            info.final_shape = X.shape
            info.runtimes.setdefault(f"feature downsampling ({feature_origin})", 0.0)
        is_projection = feature_origin in {
            FeatureDownsampleMethod.SVD.value,
            FeatureDownsampleMethod.SparseRandomProjection.value,
        }
        if is_projection:
            lineage = {str(col): str(col) for col in X.columns}
            X_tabpfn = X
        else:
            lineage = {
                str(col): self.feature_lineage.get(str(col), str(col))
                for col in X.columns
            }
            raw_cols: list[str] = []
            for source in lineage.values():
                if source in self.X_tabpfn.columns and source not in raw_cols:
                    raw_cols.append(source)
            X_tabpfn = self.X_tabpfn.loc[:, raw_cols]
        return PreparedData(
            X=X,
            X_tabpfn=X_tabpfn,
            feature_lineage=lineage,
            X_cont=X,
            X_cat=None,
            y=self.y.copy(),
            groups=None if self.groups is None else self.groups.copy(),
            labels=self.labels,
            inspection=self.inspection,
            info=info,
            is_classification=self.is_classification,
            phase=self.phase,
            validate=True,
        )

    def for_target(self, target: str) -> PreparedData:
        if not isinstance(self.y, DataFrame):
            if self.y.name != target:
                raise ValueError(
                    f"Single-target PreparedData has y.name={self.y.name}, "
                    f"requested={target}"
                )
            return self

        if target not in self.y.columns:
            raise ValueError(f"Target '{target}' not found in PreparedData.y.")

        y_col = self.y[target].copy()
        labels: Optional[dict[int, str]]
        if (
            isinstance(self.labels, dict)
            and target in self.labels
            and isinstance(self.labels[target], dict)
        ):
            labels = cast(dict[int, str], self.labels[target])
        else:
            labels = None

        return PreparedData(
            X=self.X,
            X_tabpfn=self.X_tabpfn,
            feature_lineage=self.feature_lineage,
            X_cont=self.X_cont,
            X_cat=self.X_cat,
            y=y_col,
            groups=self.groups,
            labels=labels,
            ix_train=self.ix_train,
            ix_tests=self.ix_tests,
            tests_method=self.tests_method,
            inspection=self.inspection,
            info=self.info,
            is_classification=self.is_classification,
            phase=self.phase,
            validate=True,
        )

    def representative_subsample(
        self,
        n_sub: int = UNIVARIATE_PRED_MAX_N_SAMPLES,
        rng: Optional[Generator] = None,
    ) -> tuple[PreparedData, ndarray]:
        rng = rng or np.random.default_rng()
        X, X_tabpfn, X_cont, X_cat = (
            self.X,
            self.X_tabpfn,
            self.X_cont,
            self.X_cat,
        )
        y = self.y

        g = self.groups
        if g is not None:
            warn(
                "Grouping is currently NOT implemented for `representative_subsample`. "
                "The grouping variable will be ignored when creating a minimal viable "
                "subsample. This may introduce a significant bias in the unviariate "
                "predictions if samples from the same group end up distributed across "
                "subsequent training and test splits. However, this bias will be limited "
                "to the univariate predictive stats. Grouping is handled properly in all "
                "subsequent df-analyze splitting procedures."
            )

        if len(X) <= UNIVARIATE_PRED_MAX_N_SAMPLES:
            return self, np.arange(len(X), dtype=np.int64)

        if self.is_classification:
            if isinstance(y, DataFrame):
                split_y = y_split_label(y)
                split_codes = split_y.astype("category").cat.codes
                idx = viable_subsample(df=X, target=split_codes, n_sub=n_sub, rng=rng)
            else:
                idx = viable_subsample(df=X, target=y, n_sub=n_sub, rng=rng)
            X = X.iloc[idx]
            X_tabpfn = X_tabpfn.iloc[idx]
            if X_cont is not None:
                X_cont = X_cont.iloc[idx]
            if X_cat is not None:
                X_cat = X_cat.iloc[idx]
            if g is not None:
                g = g.iloc[idx]
            y = y.iloc[idx]
        else:
            kb = KBinsDiscretizer(
                n_bins=5,
                encode="ordinal",
                quantile_method="linear",
            )
            if isinstance(y, DataFrame):
                y_vals = regression_split_label(y)
            else:
                y_vals = y
            strat = kb.fit_transform(y_vals.to_numpy().reshape(-1, 1))
            n_train = UNIVARIATE_PRED_MAX_N_SAMPLES
            ss = StratifiedShuffleSplit(n_splits=1, train_size=n_train)
            idx = next(ss.split(strat, strat))[0]
            X = cast(DataFrame, self.X.iloc[idx, :].copy(deep=True))
            X_tabpfn = self.X_tabpfn.iloc[idx, :].copy(deep=True)
            y = self.y.loc[idx].copy(deep=True)
            if X_cont is not None:
                X_cont = X_cont.loc[idx, :].copy(deep=True)
            if X_cat is not None:
                X_cat = X_cat.loc[idx, :].copy(deep=True)
            if g is not None:
                g = g.iloc[idx]

        return PreparedData(
            X=X,
            X_tabpfn=X_tabpfn,
            feature_lineage=self.feature_lineage,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            groups=g,
            labels=self.labels,
            inspection=self.inspection,
            info=self.info,
            is_classification=self.is_classification,
        ), idx

    def validate(
        self,
        X: DataFrame,
        X_cont: Optional[DataFrame],
        X_cat: Optional[DataFrame],
        y: Union[Series, DataFrame],
    ) -> tuple[
        DataFrame,
        Optional[DataFrame],
        Optional[DataFrame],
        Union[Series, DataFrame],
    ]:
        n_samples = len(X)
        if X_cont is not None and len(X_cont) != n_samples:
            raise ValueError(
                f"Continuous data number of samples ({len(X_cont)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if X_cat is not None and len(X_cat) != n_samples:
            raise ValueError(
                f"Categorical data number of samples ({len(X_cat)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"Target number of samples ({len(X)}) does not "
                f"match number of samples in processed data ({n_samples})"
            )

        if self.is_classification:
            if isinstance(y, DataFrame):
                undersampled_counts: list[str] = []
                for col in y.columns:
                    cnts = y[col].value_counts()
                    if cnts.empty:
                        raise ValueError(
                            f"Target '{col}' has no valid (non-missing) samples."
                        )
                    if cnts.min() <= N_MULTITARGET_LEVEL_MIN:
                        df = DataFrame(
                            index=pd.Index(data=cnts.index, name="Target Level"),
                            columns=["Count"],
                            data=cnts.values,
                        )
                        info = df.to_markdown(tablefmt="simple")
                        undersampled_counts.append(f"Target '{col}' counts:\n\n{info}")
                if len(undersampled_counts) > 0:
                    info = "\n\n".join(undersampled_counts)
                    warn(
                        "One or more multi-target classification targets have "
                        "undersampled levels. This means that one or more target levels "
                        f"(classes) have fewer than or equal to {N_MULTITARGET_LEVEL_MIN} samples. "
                        "Preprocessing will continue, but splitting and classification "
                        "metrics for these levels may be unstable. This may also cause "
                        "downstream CV splitting to fail (especially when class counts "
                        "are below the number of folds).\n\n"
                        "Observed target level counts:\n\n"
                        f"{info}"
                    )
            elif np.bincount(y).min() < N_TARG_LEVEL_MIN:
                unqs, cnts = np.unique(y.to_numpy(), return_counts=True)
                df = DataFrame(
                    index=pd.Index(data=unqs, name="Target Level"),
                    columns=["Count"],
                    data=cnts,
                )
                info = df.to_markdown(tablefmt="simple")
                raise ValueError(
                    f"Target '{y.name}' has undersampled levels. This means that one or "
                    f"more of the target levels (classes) has less than {N_TARG_LEVEL_MIN} "
                    "samples, either before or after splitting into a holdout set. This is "
                    "simply far too few samples for meaningful generalization or stable "
                    "performance estimates, and means your data is far too small to use for "
                    "automated machine learning via df-analyze.\n\n"
                    "Observed target level counts:\n\n"
                    f"{info}"
                )

        # Handle some BS due to stupid Pandas index behaviour
        X.reset_index(drop=True, inplace=True)
        if X_cont is not None:
            X_cont.index = X.index.copy(deep=True)
        if X_cat is not None:
            X_cat.index = X.index.copy(deep=True)
        y.index = X.index.copy(deep=True)
        return X, X_cont, X_cat, y

    def rename_cols(self, df: DataFrame) -> DataFrame:
        # see https://github.com/microsoft/LightGBM/issues/6202#issuecomment-1820286842
        # for LightGBM disallowed characters
        df = df.rename(
            columns=lambda col: re.sub(r"[\[\]\{\},:\"]+", "", str(col)),
        )
        dupe_cols = set(df.columns[df.columns.duplicated()])
        counts = {col: 0 for col in dupe_cols}
        new_cols = []
        for col in df.columns:
            if col not in dupe_cols:
                new_cols.append(col)
                continue
            if counts[col] == 0:  # leave name unchanged
                new_cols.append(col)
            else:
                new_cols.append(f"{col}_{counts[col]}")
            counts[col] += 1
        df.columns = new_cols
        return df

    def describe_features(
        self,
    ) -> tuple[Optional[DataFrame], Optional[DataFrame], DataFrame]:
        if isinstance(self.y, DataFrame):
            raise ValueError(
                "describe_features does not support multi-target data. "
                "Call for_target(...) first."
            )
        return describe_all_features(
            continuous=self.X_cont,
            categoricals=self.X_cat,
            target=self.y,
            is_classification=self.is_classification,
        )

    def to_markdown(self) -> Optional[str]:
        if self.info is not None:
            return self.info.to_markdown()
        warn("No preparation info found, no Markdown report to make")

    def save_raw(self, root: Path) -> None:
        try:
            self.X.to_parquet(root / self.files.X_raw)
            self.X_tabpfn.to_parquet(root / self.files.X_tabpfn_raw)
            if self.X_cont is not None:
                self.X_cont.to_parquet(root / self.files.X_cont_raw)
            if self.X_cat is not None:
                self.X_cat.to_parquet(root / self.files.X_cat_raw)
            if isinstance(self.y, DataFrame):
                self.y.to_parquet(root / self.files.y_raw)
            else:
                self.y.to_frame().to_parquet(root / self.files.y_raw)
            if self.groups is not None:
                self.groups.to_frame().to_parquet(root / self.files.g_raw)
            if self.labels is not None:
                if all(isinstance(v, dict) for v in self.labels.values()):
                    DataFrame(self.labels).to_parquet(root / self.files.labels)
                else:
                    Series(self.labels).to_frame().to_parquet(root / self.files.labels)
            if self.info is not None:
                self.info.to_json(root / self.files.info)
            if self.ix_train is not None:
                js = json.dumps(self.ix_train.tolist())
                (root / self.files.ix_train).write_text(js)
            if self.ix_tests is not None:
                obj = [ix_test.tolist() for ix_test in self.ix_tests]
                js = json.dumps(obj)
                (root / self.files.ix_tests).write_text(js)
        except Exception as e:
            warn(
                f"Exception while saving {self.__class__.__name__} to {root}."
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    @staticmethod
    def from_saved(root: Path, inspection: InspectionResults) -> PreparedData:
        files = PrepFiles()
        X = pd.read_parquet(root / files.X_raw)
        tabpfn_path = root / files.X_tabpfn_raw
        X_tabpfn = (
            pd.read_parquet(tabpfn_path) if tabpfn_path.exists() else X.copy()
        )
        X_cont = pd.read_parquet(root / files.X_cont_raw)
        X_cat = pd.read_parquet(root / files.X_cat_raw)
        y_raw = pd.read_parquet(root / files.y_raw)
        gfile = root / files.g_raw
        g_raw = pd.read_parquet(gfile) if gfile.exists() else None
        y: Union[Series, DataFrame]
        if y_raw.shape[1] == 1:
            y = Series(
                name=y_raw.columns[0], data=y_raw.values.ravel(), index=y_raw.index
            )
        else:
            y = y_raw
        g = (
            Series(name=g_raw.columns[0], data=g_raw.values.ravel(), index=g_raw.index)
            if g_raw is not None
            else None
        )
        labelpath = root / files.labels
        labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]]
        if labelpath.exists():
            labels_raw = pd.read_parquet(labelpath)
            if labels_raw.shape[1] == 1:
                labels = cast(dict[int, str], labels_raw.iloc[:, 0].dropna().to_dict())
            else:
                labels = {
                    str(col): cast(dict[int, str], labels_raw[col].dropna().to_dict())
                    for col in labels_raw.columns
                }
        else:
            labels = None
        info_path = root / files.info
        info = PreparationInfo.from_json(info_path) if info_path.exists() else None
        if info is not None:
            is_cls = info.is_classification
        else:
            is_cls = None

        ix_train_path = root / files.ix_train
        if ix_train_path.exists():
            obj = json.loads(ix_train_path.read_text())
            ix_train = np.array(obj, dtype=np.int64)
        else:
            ix_train = None

        ix_tests_path = root / files.ix_tests
        if ix_tests_path.exists():
            obj: list[list[int]] = json.loads(ix_tests_path.read_text())
            ix_tests = [np.array(ix_test, dtype=np.int64) for ix_test in obj]
        else:
            ix_tests = None

        return PreparedData(
            X=X,
            X_tabpfn=X_tabpfn,
            X_cont=X_cont,
            X_cat=X_cat,
            y=y,
            groups=g,
            labels=labels,
            inspection=inspection,
            info=info,
            is_classification=is_cls,
            ix_train=ix_train,
            ix_tests=ix_tests,
        )


def prepare_target(
    df: DataFrame,
    target: str,
    is_classification: bool,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
    _warn: bool = True,
) -> tuple[
    DataFrame,
    Series,
    Optional[dict[int, str]],
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    y = df[target]
    df = df.drop(columns=target)
    if is_classification:
        df, y, labels, ix_train, ix_tests = encode_target(
            df, y, ix_train, ix_tests, _warn=_warn
        )
    else:
        labels = None
        df, y, ix_train, ix_tests = clean_regression_target(df, y, ix_train, ix_tests)
    return df, y, labels, ix_train, ix_tests


def prepare_data(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str],
    results: InspectionResults,
    is_classification: bool,
    ix_train: Optional[ndarray] = None,
    ix_tests: Optional[list[ndarray]] = None,
    tests_method: Optional[ValidationMethod] = ValidationMethod.List,
    _warn: bool = True,
) -> PreparedData:
    """
    Returns
    -------
    X_encoded: DataFrame
        All encoded and processed predictors.

    X_cat: DataFrame
        The categorical variables remaining after processing (no encoding,
        for univariate metrics and the like).

    X_cont: DataFrame
        The continues variables remaining after processing (no encoding,
        for univariate metrics and the like).

    y: Series
        The regression or classification target, also encoded.

    info: dict[str, str]
        Other information regarding warnings and cleaning effects.

    """
    times: dict[str, float] = {}
    timer = partial(timed, times=times)
    target_cols = as_target_list(target)
    orig_shape = (df.shape[0], df.shape[1] - len(target_cols))

    df = timer(unify_nans)(df)
    df = timer(convert_categoricals)(df=df, target=target_cols, grouper=grouper)
    info: Optional[Union[RegTargetInfo, ClsTargetInfo]] = None
    multitarget_audit: Optional[MultiTargetAudit] = None
    raw_targets = df[target_cols].copy() if len(target_cols) > 1 else None
    n_targ_drop = 0
    labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]]
    if is_classification:
        if len(target_cols) > 1:
            orig_n = df.shape[0]
            df_no_y, y, labels, ix_train, ix_tests = timer(encode_targets)(
                df, target_cols, ix_train, ix_tests, _warn=_warn
            )
            n_targ_drop = orig_n - df_no_y.shape[0]
            df = pd.concat([df_no_y, y], axis=1)
            if raw_targets is not None:
                multitarget_audit = _build_multitarget_audit(
                    raw_targets, y, labels, is_classification=True
                )
        else:
            df, n_targ_drop, ix_train, ix_tests = timer(drop_target_nans)(
                df, target_cols[0], ix_train, ix_tests
            )
            df, y, labels, ix_train, ix_tests = timer(encode_target)(
                df, df[target_cols[0]], ix_train, ix_tests
            )
    else:
        if len(target_cols) > 1:
            orig_n = len(df)
            df, y, ix_train, ix_tests = timer(clean_regression_targets)(
                df, target_cols, ix_train, ix_tests
            )
            n_targ_drop = orig_n - len(df)
            if raw_targets is not None:
                multitarget_audit = _build_multitarget_audit(
                    raw_targets, y, labels=None, is_classification=False
                )
        else:
            df, n_targ_drop, ix_train, ix_tests = timer(drop_target_nans)(
                df, target_cols[0], ix_train, ix_tests
            )
            df, y, ix_train, ix_tests = timer(clean_regression_target)(
                df, df[target_cols[0]], ix_train, ix_tests
            )
        labels = None

    for i, target_col in enumerate(target_cols):
        target_info = timer(inspect_target)(
            df, target_col, is_classification=is_classification
        )
        if len(target_cols) == 1 and i == 0:
            info = target_info
    fit_indices = ix_train
    df = timer(drop_unusable)(
        df,
        results,
        target=target_cols,
        _warn=_warn,
        fit_indices=fit_indices,
    )
    X_tabpfn = df.drop(
        columns=[*target_cols, *([] if grouper is None else [grouper])],
        errors="ignore",
    ).reset_index(drop=True)
    tabpfn_categoricals = set(results.cats.infos) | set(results.binaries.infos)
    for col in tabpfn_categoricals.intersection(X_tabpfn.columns):
        X_tabpfn[col] = X_tabpfn[col].astype("category")
    df, X_cont, n_ind_added = handle_continuous_nans(
        df=df,
        target=target_cols,
        grouper=grouper,
        results=results,
        nans=NanHandling.Median,
        fit_indices=fit_indices,
    )
    X_cont = normalize_continuous(
        X_cont, robust=True, fit_indices=fit_indices
    )
    df[X_cont.columns] = X_cont

    df = timer(deflate_categoricals)(df, grouper, results, _warn=_warn)
    df, X_cat = timer(encode_categoricals)(
        df=df,
        target=target_cols,
        grouper=grouper,
        results=results,
        warn_explosion=_warn,
        fit_indices=fit_indices,
    )

    X = df.drop(columns=target_cols).reset_index(drop=True)
    if grouper is not None:
        g = X[grouper]
        X = X.drop(columns=grouper)
    else:
        g = None
    return PreparedData(
        X=X,
        X_tabpfn=X_tabpfn,
        X_cont=X_cont,
        X_cat=X_cat,
        y=y,
        groups=g,
        labels=labels,
        ix_train=ix_train,
        ix_tests=ix_tests,
        tests_method=tests_method,
        info=PreparationInfo(
            original_shape=orig_shape,
            final_shape=X.shape,
            n_samples_dropped_via_target_NaNs=n_targ_drop,
            n_cont_indicator_added=n_ind_added,
            target_info=info,
            runtimes=times,
            is_classification=is_classification,
            multitarget_audit=multitarget_audit,
        ),
        inspection=results,
        is_classification=is_classification,
    )

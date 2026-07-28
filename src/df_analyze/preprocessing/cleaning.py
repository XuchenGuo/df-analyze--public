import re
import sys
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from df_analyze._constants import (
    MAX_PERF_N_FEATURES,
    N_CAT_LEVEL_MIN,
    N_MULTITARGET_LEVEL_MIN,
    N_TARG_LEVEL_MIN,
    NAN_STRINGS,
)
from df_analyze.enumerables import NanHandling
from df_analyze.loading import load_spreadsheet
from df_analyze.preprocessing.inspection.inspection import (
    InspectionInfo,
    InspectionResults,
    convert_categoricals,
    messy_inform,
    unify_nans,
)
from df_analyze.preprocessing.targets import TargetSpec, as_target_list


class DataCleaningWarning(UserWarning):
    """For when df-analyze significantly and automatically alters input data"""

    def __init__(self, message: str) -> None:
        cols = get_terminal_size((81, 24))[0]
        sep = "=" * cols
        underline = "." * (len(self.__class__.__name__) + 1)
        self.message = (
            f"\n{sep}\n{self.__class__.__name__}\n{underline}\n{message}\n{sep}"
        )

    def __str__(self) -> str:
        return str(self.message)


class RenameInfo:
    def __init__(self, renames: list[tuple[str, str]]) -> None:
        self.renames = renames
        self.changed: list[tuple[str, str]] = []
        for original, renamed in self.renames:
            if original != renamed:
                self.changed.append((original, renamed))
        self.changed_map: dict[str, str] = {
            original: renamed for original, renamed in self.changed
        }

    def to_markdown(self) -> Optional[str]:
        if len(self.changed) == 0:
            return None
        lines = []
        lines.append("# Feature Renames\n\n")
        lines.append(
            "Some of your feature names contained characters that are problems\n"
            "for either regular expressions or the LightGBM estimator, or\n"
            "which are duplicated. These have been renamed to resolve this\n"
            "issue. Duplicated column renames are given an integer suffix based\n"
            "on the order in which they appeared in the original data, except\n"
            "for the first instance of a duplicated name, which is left as is.\n"
            "\n\n"
        )
        originals, renamed = list(zip(*self.changed))
        table = DataFrame({"New name": renamed, "Original name": originals})
        lines.append(table.to_markdown(index=False))

        return "".join(lines)

    def rename_columns(self, cols: list[str]) -> list[str]:
        renamed = []
        for col in cols:
            if col in self.changed_map:
                renamed.append(self.changed_map[col])
            else:
                renamed.append(col)
        return renamed


def dedup_names(
    df: DataFrame, target: TargetSpec
) -> tuple[DataFrame, list[tuple[str, str]]]:
    target_cols = as_target_list(target)
    y_df = df[target_cols].copy()
    df = df.drop(columns=target_cols, errors="ignore")
    dupe_cols = set(df.columns[df.columns.duplicated()])
    counts = {col: 0 for col in dupe_cols}
    renames: list[tuple[str, str]] = []
    for col in df.columns:
        if col not in dupe_cols:
            renames.append((col, col))
            continue
        # have to rename first, because if you have e.g. "F" with
        # values "1", "2", "NaN", and a second "F" with the same set of
        # values, then one-hot encoding will make "F_1", "F_2", "F_NaN",
        # and the rename here would have created "F_1", so we get a
        # duplicate. Renaming with a 0 (and double underscores) prevents
        # this
        renames.append((col, f"{col}__{counts[col]}"))  # even have to rem
        counts[col] += 1
    newnames = [new for old, new in renames]
    df.columns = newnames
    df = pd.concat([df, y_df], axis=1)
    df.columns = df.columns.astype(str)  # type: ignore
    df = df.rename(str, axis="columns")  # https://stackoverflow.com/a/77046151
    return df, renames


def sanitize_names(df: DataFrame, target: TargetSpec) -> tuple[DataFrame, RenameInfo]:
    """Rename trashy feature names (e.g. with regex characters, spaces)
    to be less problematic

    Returns
    -------
    df: DataFrame
        Renamed DataFrame

    info: RenameInfo
        Container holding tuples of renamed columns. Does NOT include target
        (which is never renamed).
    """
    target_cols = as_target_list(target)
    names = df.columns.to_series().apply(str).to_list()
    # https://docs.python.org/3/library/re.html
    for target_col in target_cols:
        if target_col not in names:
            raise RuntimeError(
                f"Unrecoverable error. Specified target: `{target_col}` not found in "
                f"data with feature names: {names}."
            )
        names.remove(target_col)

        # if `target` still in names, there was a dupe
        if target_col in names:
            raise RuntimeError(
                f"Unrecoverable error. Specified target: `{target_col}` appears "
                f"multiple times as a column name. Column names: {names}."
            )

    trash = r"[\\\.\^\$\*\+\?\{\}\[\]\(\)\| ]"
    lgbm = r"[,:\"]"

    renames = {}
    for name in names:
        renamed = re.sub(trash, "_", name)
        renamed = re.sub(lgbm, "_", renamed)
        renamed = re.sub(r"[_]+", "_", renamed)  # readability
        if renamed[-1] == "_":  # ugly dangling
            renamed = renamed[:-1]
        renames[name] = renamed

    # At this point we have a serious problem if any features have the same
    # name as the target. We have to decide between renaming the target or
    # renaming the conflicting features. Obviously we rename the features.
    if any(target_col in renames.values() for target_col in target_cols):
        for original, renamed in renames.items():
            if renamed not in target_cols:
                continue
            renames[original] = f"feat_{renamed}"

    df = df.rename(columns=renames)
    df.columns = df.columns.astype(str)  # double assurance needed, apparently
    df = df.rename(str, axis="columns")  # https://stackoverflow.com/a/77046151
    df, renames = dedup_names(df, target=target_cols)
    info = RenameInfo(renames)
    return df, info


def restore_names(df: DataFrame, renames: list[tuple[str, str]]) -> DataFrame:
    raise NotImplementedError()
    for original, renamed in renames:
        ...


def cleaning_inform(message: str) -> None:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Cleaning Data Features"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)


def normalize(
    df: DataFrame,
    target: Optional[str],
    robust: bool = True,
    fit_indices: Optional[ndarray] = None,
) -> DataFrame:
    """
    Clip data to within twice of its "robust range" (range of 90% of the data),
    and then min-max normalize.

    Notes
    -----
    Since df-analyze accepts arbitrary data, it needs to be somewhat robust to
    extreme values, and so a naive MinMax normalization seems unwise as a
    default (i.e. a single outlier can essentially "destroy" the information
    value of a feature for classifiers that are sensitive to scale, or in
    general). An ideal automated method might use more advanced outlier
    detection methods for each feature (e.g. sklearn's OneClassSVM or
    https://scikit-learn.org/stable/modules/outlier_detection.html).

    Robust, percentile-based normalization (as opposed to quantile-based
    normalization or normalization based on the probability integral transform)
    is powerful, generally not being obviously harmful to regular data, but
    being much better on highly skewed data or data with outliers.

    However, the actual impact of outliers (or rather, anchor points) surely in
    general depends more on the cumulative magnitudes of those outliers. For
    example, a feature value that is *double* the maximum value of all other
    values from the same feature is probably a clear outlier, but is not
    necessarily going to meaningfully affect the fits.

    Likewise, being an "outlier" may be a useful feature for prediction (e.g.
    by NOT normalizing or clipping, an estimator like a decision tree may learn
    an outlier threshold internally, and use this threshold effectively in
    prediction).

    NOTE: Thus, the optimal normalization strategy depends on the classier.
    Estimators highly sensitive to feature magnitudes, like linear or
    piecewise-linear classifers including LinearSVM, ElasticNet, or in theory
    shallow neural networks, and also support vector machines (for a visual
    proof of the sensitivity to scale see
    https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
    perhaps benefit immensely from robust normalization.

    Estimators involving decision trees could be harmed by any robust
    normalization that employs clipping. A nearest-neighbors-based estimator
    will be harmed by robust normalization and/or clipping if "being an
    outlier" is something that is correlated across features, i.e. if a sample
    that is an outlier on one feature also tends to be an outlier on most other
    features. However, if outlier samples are identified as such by just a
    small number of features relative to the total number of featurs, then
    normalization will mask their outlier status. (By contrast, for a
    neighbors-based estimator, if there are non-predictive features that
    contain outliers, these will skew the computed distances in a way that is
    not useful for prediction).



    """
    if (target in df.columns) and (target is not None):
        X = df.drop(columns=target)
    else:
        X = df

    cols = X.columns
    X_fit = X if fit_indices is None else X.iloc[fit_indices]

    # need to not normalize one-hot columns...

    if robust:
        medians = np.nanmedian(X_fit, axis=0)
        X = X - medians  # robust center
        X_fit = X_fit - medians

        # clip values 2 times more extreme than 95% of the data
        rmins = np.nanpercentile(X_fit, 5, axis=0)
        rmaxs = np.nanpercentile(X_fit, 95, axis=0)
        rranges = rmaxs - rmins
        rmins -= 2 * rranges
        rmaxs += 2 * rranges
        X = np.clip(X, a_min=rmins, a_max=rmaxs)
        X_fit = np.clip(X_fit, a_min=rmins, a_max=rmaxs)

    scaler = MinMaxScaler().fit(X_fit)
    X_norm = DataFrame(data=scaler.transform(X), columns=cols, index=X.index)
    if (target in df.columns) and (target is not None):
        X_norm = pd.concat([X_norm, df[target]], axis=1)
    return X_norm


def normalize_continuous(
    X_cont: DataFrame,
    robust: bool = True,
    fit_indices: Optional[ndarray] = None,
) -> DataFrame:
    if X_cont.empty:
        return X_cont
    return normalize(
        df=X_cont, target=None, robust=robust, fit_indices=fit_indices
    )


def reindex(
    idx_keep: ndarray | Series,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
) -> tuple[
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    if ix_train is None or ix_tests is None:
        return ix_train, ix_tests

    n_dropped = (~idx_keep).sum()
    if n_dropped == 0:
        return ix_train, ix_tests

    keep = np.asarray(idx_keep, dtype=bool)
    # Regenerate compact, increasing indices after rows are removed so the
    # saved train/test partitions still address the re-indexed DataFrame.
    old_to_new = np.full(len(keep), -1, dtype=int)
    old_to_new[np.flatnonzero(keep)] = np.arange(keep.sum())

    train_new = old_to_new[ix_train]
    ix_train = train_new[train_new >= 0]
    remapped_tests = []
    for ix_test in ix_tests:
        test_new = old_to_new[ix_test]
        remapped_tests.append(test_new[test_new >= 0])
    return ix_train, remapped_tests


def drop_target_nans(
    df: DataFrame,
    target: str,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
) -> tuple[
    DataFrame,
    int,
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    y_str = df[target].copy(deep=True)
    y_str = y_str.apply(lambda x: np.nan if x in NAN_STRINGS else x)

    idx_keep = ~y_str.isna()

    df = df.loc[idx_keep]
    n_dropped = (~idx_keep).sum()
    ix_train, ix_tests = reindex(idx_keep, ix_train, ix_tests)
    return df, n_dropped, ix_train, ix_tests


def handle_continuous_nans(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str],
    results: InspectionResults,
    nans: NanHandling,
    add_indicators: bool = True,
    fit_indices: Optional[ndarray] = None,
) -> tuple[DataFrame, DataFrame, int]:
    """Impute or drop nans based on values not in `cat_cols`

    Returns
    -------
    df: DataFrame
        Imputed and indicator-augmented DataFrame with target included

    X_cont: DataFrame
        Imputed DataFrame with no target, no indicators (for used in
        univariate analyses and feature selection, etc)

    n_indicators: int
        Number of NaN indicator columns added
    """
    # drop rows where target is NaN: meaningless
    # NaNs in categoricals are handled as another dummy indicator
    target_cols = as_target_list(target)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    cats = sorted(set(cats).intersection(df.columns.to_list()))
    X_cat = df[cats].copy(deep=True)
    y = df[target_cols].copy()
    g = None if grouper is None else df[grouper]
    df = df.drop(columns=target_cols, errors="ignore")
    if g is not None:
        df = df.drop(columns=grouper)

    X = df.drop(columns=results.drop_cols(), errors="ignore")
    X = X.drop(columns=cats, errors="ignore")  # now only cats and ords

    X_fit = X if fit_indices is None else X.iloc[fit_indices]

    # construct NaN indicators
    if add_indicators:
        X_nan = X.isna().astype(float)
        X_nan.rename(columns=lambda s: f"{s}_NAN", inplace=True)
        # remove constant indicators
        fit_nan = X_nan if fit_indices is None else X_nan.iloc[fit_indices]
        X_nan = X_nan.loc[:, fit_nan.sum(axis=0) != 0]

    if nans is NanHandling.Drop:
        warn(
            "Dropping NaNs is currently not implemented as it can render "
            "usable data unusable, and is unnecessarily destructive of "
            "potential information relative to the use of indicator features "
            "and/or imputation strategies. NaN handling will be set to "
            f"'{NanHandling.Median.value}' instead."
        )
        nans = NanHandling.Median

    if X.empty:
        X_cont = X
    elif nans is NanHandling.Drop:
        raise NotImplementedError("Not implemented currently due to shape changes.")
        X_cont = X.dropna(axis="columns", how="any").dropna(axis="index", how="any")
        if 0 in X_cont.shape:
            others = [na.value for na in NanHandling if na is not NanHandling.Drop]
            raise RuntimeError(
                "Dropping NaNs resulted in either no remaining samples or features. "
                "Consider changing the option `--nan` to another valid option. Other "
                f"valid options: {others}"
            )
    elif nans in [NanHandling.Mean, NanHandling.Median]:
        strategy = "mean" if nans is NanHandling.Mean else "median"
        imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
        imputer.fit(X_fit)
        X_fitted = imputer.transform(X)
        X_cont = DataFrame(data=X_fitted, columns=X.columns, index=X.index)
    elif nans is NanHandling.Impute:
        warn(
            "Using experimental multivariate imputation. This could take a very "
            "long time for even tiny (<500 samples, <30 features) datasets."
        )
        imputer = IterativeImputer(verbose=2, keep_empty_features=True)
        imputer.fit(X_fit)
        X_cont = DataFrame(
            data=imputer.transform(X), columns=X.columns, index=X.index
        )
    else:
        raise NotImplementedError(f"Unhandled enum case: {nans}")

    if add_indicators:
        if g is None:
            X = pd.concat([X_nan, X_cont, X_cat, y], axis=1)  # type: ignore
        else:
            X = pd.concat([X_nan, X_cont, X_cat, g, y], axis=1)  # type: ignore
        return X, X_cont, int(X_nan.shape[1])  # type: ignore

    if g is None:
        X = pd.concat([X_cont, X_cat, y], axis=1)
    else:
        X = pd.concat([X_cont, X_cat, g, y], axis=1)
    return X, X_cont, 0


def encode_target(
    df: DataFrame,
    target: Series,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
    _warn: bool = False,
) -> tuple[DataFrame, Series, dict[int, str], Optional[ndarray], Optional[list[ndarray]]]:
    target = unify_nans(target)
    target_train = target if ix_train is None else target.iloc[ix_train]
    target_train = target_train.dropna()
    unqs, cnts = np.unique(target_train.astype(str), return_counts=True)
    if len(unqs) <= 1:
        raise ValueError(f"Target variable {target.name} is constant.")
    idx = cnts <= N_TARG_LEVEL_MIN
    n_cls = len(unqs)
    if np.sum(idx).item() > 0:
        if _warn:
            cleaning_inform(
                "The target variable has a number of class labels "
                f"({unqs[idx]}) with less than or equal to {N_TARG_LEVEL_MIN} members. This "
                "will cause problems with splitting in various nested k-fold "
                "procedures used in `df-analyze`. In addition, any estimates "
                "or metrics produced for such a class will not be "
                "statistically meaningful (i.e. the uncertainty on those "
                "metrics or estimates will be exceedingly large). We thus "
                "remove all samples that belong to these labels, bringing the "
                f"total number of classes down to {n_cls - np.sum(idx).item()}"
            )
        idx_keep = ~target.astype(str).isin(unqs[idx])
        ix_train, ix_tests = reindex(idx_keep, ix_train, ix_tests)
        df = df.copy().loc[idx_keep].reset_index(drop=True)
        # reset index extremely important for later concats
        target = target[idx_keep].reset_index(drop=True)

    # drop NaNs: Makes no sense to count correct NaN predictions toward
    # classification performance
    idx_keep = ~target.isna()
    ix_train, ix_tests = reindex(idx_keep, ix_train, ix_tests)

    df = df.copy().loc[idx_keep].reset_index(drop=True)
    target = target[idx_keep].reset_index(drop=True)

    enc = LabelEncoder()
    target_train = target if ix_train is None else target.iloc[ix_train]
    if target_train.nunique() <= 1:
        raise ValueError(
            f"Target variable {target.name} is constant in the training data after "
            "removing undersampled labels."
        )
    enc.fit(target_train)
    try:
        encoded = np.array(enc.transform(target))
    except ValueError as error:
        raise ValueError(
            f"Target '{target.name}' contains labels in a holdout set that are "
            "not present in its training data."
        ) from error
    classes = enc.classes_.tolist()
    ints = np.asarray(enc.transform(classes)).tolist()
    return (
        df,
        Series(encoded, name=target.name),
        {i: cls for i, cls in zip(ints, classes)},
        ix_train,
        ix_tests,
    )


def encode_targets(
    df: DataFrame,
    targets: TargetSpec,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
    _warn: bool = False,
) -> tuple[
    DataFrame,
    DataFrame,
    dict[str, dict[int, str]],
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    target_cols = as_target_list(targets)
    n_targ_level_min_mt = N_MULTITARGET_LEVEL_MIN
    has_missing_target = Series(False, index=df.index, dtype=bool)
    for col in target_cols:
        series = unify_nans(df[col])
        is_missing = series.isna()
        has_missing_target |= is_missing

    keep_mask = ~has_missing_target

    ix_train, ix_tests = reindex(keep_mask, ix_train, ix_tests)
    df = df.loc[keep_mask].reset_index(drop=True)
    y_df_raw = df[target_cols].copy()
    df = df.drop(columns=target_cols, errors="ignore")

    labels_map: dict[str, dict[int, str]] = {}
    y_encoded = DataFrame(index=y_df_raw.index)
    for col in target_cols:
        series = unify_nans(y_df_raw[col]).astype(str)
        series_train = series if ix_train is None else series.iloc[ix_train]
        unqs, cnts = np.unique(series_train, return_counts=True)
        if len(unqs) <= 1:
            raise ValueError(
                f"Target variable {col} is constant in the training data after "
                "removing rows with missing multi-target labels. Remove this target "
                "column and re-run df-analyze."
            )
        rare = unqs[cnts <= n_targ_level_min_mt]
        if len(rare) > 0 and _warn:
            cleaning_inform(
                "The multi-target classification variable "
                f"`{col}` has class labels ({rare.tolist()}) with less than or "
                f"equal to {n_targ_level_min_mt} training members. These rows are "
                "kept because removing them would also discard valid labels from "
                "the other targets. Splits and metrics for the rare labels may be "
                "unstable."
            )
        enc = LabelEncoder()
        enc.fit(series_train)
        try:
            encoded = np.array(enc.transform(series))
        except ValueError as error:
            raise ValueError(
                f"Target '{col}' contains labels in a holdout set that are not "
                "present in its training data."
            ) from error
        classes = enc.classes_.tolist()
        ints = np.asarray(enc.transform(classes)).tolist()
        labels_map[col] = {i: cls for i, cls in zip(ints, classes)}
        y_encoded[col] = encoded

    return df, y_encoded, labels_map, ix_train, ix_tests


def clean_regression_target(
    df: DataFrame,
    target: Series,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
) -> tuple[
    DataFrame,
    Series,
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    """Remove regression targets that cannot be predicted.

    Missing targets are dropped, then the remaining values are converted to
    finite floats while preserving their original units. Normalization, when
    requested, is handled later from training-only data.
    """
    target = unify_nans(target)
    idx_keep = ~target.isna()
    ix_train, ix_tests = reindex(idx_keep, ix_train, ix_tests)
    # reset index extremely important for later concats
    df = df.loc[idx_keep].reset_index(drop=True)
    target = target[idx_keep].reset_index(drop=True)

    values = pd.to_numeric(target, errors="raise").astype(float)
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Regression target '{target.name}' contains infinite values.")
    target = Series(numeric, name=target.name)

    return df, target, ix_train, ix_tests


def clean_regression_targets(
    df: DataFrame,
    targets: TargetSpec,
    ix_train: Optional[ndarray],
    ix_tests: Optional[list[ndarray]],
) -> tuple[
    DataFrame,
    DataFrame,
    Optional[ndarray],
    Optional[list[ndarray]],
]:
    """Drop rows missing any regression target and preserve original units."""
    target_cols = as_target_list(targets)
    target_frame = DataFrame(
        {col: unify_nans(df[col]) for col in target_cols}, index=df.index
    )
    keep_mask = ~target_frame.isna().any(axis=1)
    ix_train, ix_tests = reindex(keep_mask, ix_train, ix_tests)

    df = df.loc[keep_mask].reset_index(drop=True)
    target_frame = target_frame.loc[keep_mask].reset_index(drop=True)
    y = DataFrame(index=target_frame.index)
    for col in target_cols:
        values = pd.to_numeric(target_frame[col], errors="raise").astype(float)
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Regression target '{col}' contains infinite values.")
        y[col] = numeric
    df[target_cols] = y
    return df, y, ix_train, ix_tests


def drop_cols(
    df: DataFrame, kind: str, *col_dicts: InspectionInfo, _warn: bool = False
) -> DataFrame:
    cols = set()
    cols_descs = []
    for d in col_dicts:
        for col, desc in d.infos.items():
            if col in df:
                cols.add(col)
                cols_descs.append((col, desc))
    cols_descs = sorted(cols_descs, key=lambda pair: pair[0])

    if len(cols) <= 0:  # nothing to drop
        return df

    w = max(len(col) for col in cols) + 2
    info = "\n".join([f"{col:<{w}} {desc}" for col, desc in cols_descs])
    if _warn:
        cleaning_inform(
            f"Dropping features that appear to be {kind}. Additional information "
            "should be available above.\n\n"
            f"Dropped features:\n{info}"
        )
    drops = list(cols)

    df = df.drop(columns=drops, errors="ignore")
    return df


def floatify(df: DataFrame) -> DataFrame:
    df = df.copy()
    cols = df.select_dtypes(include=["object", "string[python]"]).columns.tolist()  # type: ignore
    for col in cols:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


def drop_unusable(
    df: DataFrame,
    results: InspectionResults,
    target: Optional[TargetSpec] = None,
    _warn: bool = False,
    fit_indices: Optional[ndarray] = None,
) -> DataFrame:
    """Drops identifiers, datetime, constants"""
    target_cols = set(as_target_list(target)) if target is not None else set()
    df = drop_cols(df, "identifiers", results.ids, _warn=_warn)
    df = drop_cols(df, "datetime data", results.times, _warn=_warn)
    df = drop_cols(df, "constant", results.consts, _warn=_warn)
    fit_df = df if fit_indices is None else df.iloc[fit_indices]
    const_drops = fit_df[
        fit_df.columns[fit_df.map(str).apply(lambda col: len(np.unique(col)) <= 1)]
    ].columns.to_list()
    const_drops = [col for col in const_drops if str(col) not in target_cols]
    if len(const_drops) > 0:
        df = df.drop(columns=const_drops, errors="ignore")
    return df


def deflate_categoricals(
    df: DataFrame,
    grouper: Optional[str],
    results: InspectionResults,
    _warn: bool = True,
) -> DataFrame:
    df = df.copy()

    existing_cols = set(df.columns.tolist())
    infos = results.inflation
    infos = [info for info in infos if info.col in existing_cols]
    infos = sorted(infos, key=lambda info: info.n_total, reverse=True)
    cols = [info.col for info in infos]

    if (grouper is not None) and (grouper in cols):
        raise RuntimeError(f"Found grouping column `{grouper}` in inspection info")

    for info in tqdm(
        infos, desc="Deflating categoricals", total=len(infos), disable=len(infos) < 50
    ):
        col = info.col
        nan = df[col].isna()  # remember, we have done NaN unification already
        df[col] = df[col].astype(str)  # same as comment above, this is safe
        df.loc[nan, col] = np.nan
        names = df[col].unique()
        rename = "DEFLATED_OTHER" if "DEFLATED" in names else "DEFLATED"
        if "DEFLATED_OTHER" in names:  # getting a bit crazy here...
            rename = "DEFLATED_OTHER_DFANALYZE"
        ix_deflate = df[col].isin(info.to_deflate)
        df.loc[ix_deflate, col] = rename

    if len(infos) > 0:
        w = max(len(info.col) for info in infos) + 2
    else:
        w = 0
    if _warn and len(infos) > 0:
        message = "\n".join(
            [f"{info.col:<{w}} {info.n_total} --> {info.n_keep}" for info in infos]
        )
        messy_inform(
            "Found and 'deflated' a number of categorical variables with "
            "levels that are unlikely to be reliable or useful in prediction. "
            "For each level of categorical variable to be predictively useful, "
            "there must be enough samples to be statistically meaningful (or to "
            "allow some reasonable generalization) in each fold used for fitting "
            "or analyses. Roughly, this means each fold needs to see at least "
            "10-20 samples of each level (depending on how strongly / cleanly "
            "the level relates to other features - ultimately this is just a "
            "heuristic). Internal 5-fold validations on the training set will "
            "have approximately half the amount of samples (0.6 training split * "
            "0.8 k-fold training split) as the original full data. Thus, if there "
            f"are less than about {N_CAT_LEVEL_MIN} samples per class level in the "
            "original data, then training splits are expected to have about 10 "
            "samples per level, on average (but in fact, anywhere from 0-20, "
            "since stratification is on the target only). Thus, we bin all class "
            f"levels with less than {N_CAT_LEVEL_MIN} samples into a DEFLATED "
            "category (or 'DEFLATED_OTHER' if 'DEFLATED' is already present "
            "as a label for a class level).\n\n"
            "Deflated categorical variables (before --> after):\n"
            f"{message}"
        )

    return df


def encode_categoricals(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str],
    results: InspectionResults,
    warn_explosion: bool = True,
    fit_indices: Optional[ndarray] = None,
) -> tuple[DataFrame, DataFrame]:
    """

    Returns
    -------
    encoded: DataFrame
        Pandas DataFrame with categorical variables one-hot encoded

    unencoded: DataFrame
        Pandas DataFrame with original categorical variables
    """
    target_cols = as_target_list(target)
    y = df[target_cols].copy()
    df = df.drop(columns=target_cols, errors="ignore")
    df = convert_categoricals(df, target_cols, grouper=grouper)
    df = unify_nans(df)
    df = drop_unusable(df, results, _warn=False, fit_indices=fit_indices)
    df = deflate_categoricals(df, grouper, results, _warn=warn_explosion)
    df = drop_unusable(
        df, results, _warn=False, fit_indices=fit_indices
    )  # get rid of NaNs from deflation
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    cats = sorted(set(cats).intersection(df.columns.to_list()))
    X_cat = df.loc[:, cats].copy(deep=True)
    to_convert = cats

    # below will FAIL if we didn't remove timestamps or etc.
    try:
        bins = sorted(set(results.binaries.cols).intersection(to_convert))
        multis = sorted(set(results.multi_cats).intersection(to_convert))
        new = df
        fit_new = new if fit_indices is None else new.iloc[fit_indices]
        for col in to_convert:
            categories = fit_new[col].dropna().unique().tolist()
            new[col] = pd.Categorical(new[col], categories=categories)
        # note `bins` includes variables that are (1) "constant-binary" (i.e.
        # either a constant value or NaN), (2) true binary (i.e. two unique
        # non-NaN values), or (3) binary plus NaN (two unique non-NaN values
        # and a NaN). In all these cases there is no intepretive confusion or
        # information loss when representing them as:
        #
        # (1) {0, 1}
        # (2) {0, 1}
        # (3) {0, 1} + {0, 1} NaN indicator
        #
        # i.e. by using pd.get_dummies(..., dummy_na=True, drop_first=True)
        fit_new = new if fit_indices is None else new.iloc[fit_indices]
        nan_cols = set(fit_new.columns[fit_new.isna().any()].to_list())
        bin_nans = sorted(nan_cols.intersection(bins))
        bin_no_nans = sorted(set(bins).difference(bin_nans))
        multi_nans = sorted(nan_cols.intersection(multis))
        multi_no_nans = sorted(set(multis).difference(multi_nans))

        new = pd.get_dummies(new, columns=bin_nans, dummy_na=True, drop_first=True)
        new = pd.get_dummies(new, columns=multi_nans, dummy_na=True, drop_first=False)

        new = pd.get_dummies(new, columns=bin_no_nans, dummy_na=False, drop_first=True)
        new = pd.get_dummies(new, columns=multi_no_nans, dummy_na=False, drop_first=False)

        fit_encoded = new if fit_indices is None else new.iloc[fit_indices]
        has_const = fit_encoded.apply(
            lambda col: len(np.unique(col.apply(str))) == 1
        ).any()
        if has_const:
            new_consts = fit_encoded.columns[
                fit_encoded.apply(lambda col: len(np.unique(col)) == 1)
            ]
            raise ValueError(f"pd.get_dummies created constant columns: {new_consts}")

        if new.columns.has_duplicates:
            dupes = new.columns[new.columns.duplicated()]
            raise ValueError(f"pd.get_dummies created duplicates: {dupes}")

        # TODO: the lines above: pd.get_dummies(... dummy_na=True, drop_first=True)
        # IS BUGGED, and introduces an indicator column even when no NaN values
        # are present! So we now need to manually drop any constant columns with
        # with _nan in the title... I filed an issue:
        #
        # https://github.com/pandas-dev/pandas/issues/59968
        #
        # but likely this will be claimed as intended design.

        new = convert_categoricals(new, target=target_cols, grouper=grouper)
        new = new.astype(float)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            "Could not convert data to floating point after cleaning. Some "
            "messy data must be removed or cleaned before `df-analyze` can "
            "continue. See information above."
        ) from e

    # warn about large number of features
    old_feats = df.shape[1]
    enc_feats = new.shape[1]
    big_cats = results.big_cats
    n_orig = len(to_convert)
    n_cont = old_feats - n_orig
    n_enc_total = enc_feats - n_cont
    n_big_orig = len(big_cats)
    n_small_orig = n_orig - n_big_orig
    n_big_enc = sum(big_cats.values(), start=0)
    n_small_enc = n_enc_total - n_big_enc
    n_small_added = n_small_enc - n_small_orig
    n_big_added = n_big_enc - n_big_orig

    if enc_feats < MAX_PERF_N_FEATURES:
        new = pd.concat([new, y], axis=1)
        return new, df.loc[:, to_convert]

    names = reversed(sorted(big_cats.keys(), key=lambda col: big_cats[col]))
    if n_big_enc > 0:
        info = "\n".join([f"{cat}: {big_cats[cat]} levels" for cat in names])
        big_message = (
            "\n\n"
            "Some of the increase in data size was due to encoding categorical "
            "features with a large (20+) number of levels / categories. These "
            f"features in total added {n_big_added} additional features to "
            "the data. Often, a categorical variable with a large number of "
            "levels will not contain much predictive information in most of "
            "the levels (i.e. only a few classes will be useful). Inspect "
            "df-analyze's univariate association and prediction reports and "
            "consider whether either manually removing some of these features "
            "using the `--drops` option is warranted, or otherwise specify "
            "a filter-based automatic selection method.\n\n"
            f"Large categoricals:\n{info}"
        )
    else:
        big_message = ""

    small_names = sorted(set(to_convert).difference(names))
    if len(small_names) > 0:
        small_message = (
            "\n\n"
            "Some of the increase in data size was due to encoding categorical "
            f"features. These were the features: {small_names}, which in "
            f"total added {n_small_added} additional features to the data. "
            "Inspect df-analyze's univariate association and prediction "
            "reports and consider whether either manually removing some of "
            "these features using the `--drops` option is warranted, or "
            "otherwise specify a filter-based automatic selection method. "
        )
    else:
        small_message = ""

    # handle data explosion due to one-hot encoding
    if n_orig > 0 and ((n_enc_total / n_orig) > 10) and warn_explosion:
        messy_inform(
            "Encoding the categoricals of your data has increased the number "
            "of effective features in your data by an order of magnitude, and "
            f"your data also has now over {MAX_PERF_N_FEATURES} effective "
            "features. This will cause performance issues if using stepwise "
            "feature selection methods, or if you do not use any filter-based "
            "selection methods."
            f"{big_message}"
            f"{small_message}"
        )

    new = pd.concat([new, y], axis=1)
    return new, X_cat


def load_as_df(path: Path, spreadsheet: bool) -> DataFrame:
    FILETYPES = [".json", ".csv", ".npy", "xlsx", ".parquet"]
    if path.suffix not in FILETYPES:
        raise ValueError(f"Invalid data file. Currently must be one of: {FILETYPES}")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_csv(str(path))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(str(path))
    elif path.suffix == ".xlsx":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_excel(str(path))
    elif path.suffix == ".npy":
        arr: ndarray = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Invalid NumPy data in {path}. NumPy array must be two-dimensional."
            )
        cols = [f"c{i}" for i in range(arr.shape[1])]
        df = DataFrame(data=arr, columns=cols)
    else:
        raise RuntimeError("Unreachable!")
    return df

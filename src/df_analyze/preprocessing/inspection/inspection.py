from __future__ import annotations

import sys
import traceback
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from tqdm import tqdm

from df_analyze._constants import N_CAT_LEVEL_MIN, NAN_STRINGS

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions

from df_analyze.preprocessing.inspection.containers import (
    ClsTargetInfo,
    ColumnDescriptions,
    ColumnType,
    InflationInfo,
    InspectionInfo,
    InspectionResults,
    RegTargetInfo,
)
from df_analyze.preprocessing.inspection.inference import (
    Inference,
    InferredKind,
    has_cat_name,
    infer_binary,
    infer_categorical,
    infer_constant,
    infer_floatlike,
    infer_identifier,
    infer_ordinal,
    infer_timelike,
)
from df_analyze.preprocessing.targets import TargetSpec, as_target_list


class InspectionError(Exception):
    pass


def messy_inform(message: str) -> str:
    cols = get_terminal_size((81, 24))[0]
    sep = "=" * cols
    title = "Found Messy Data"
    underline = "." * (len(title) + 1)
    message = f"\n{sep}\n{title}\n{underline}\n{message}\n{sep}"
    print(message, file=sys.stderr)
    return message


def get_str_cols(df: DataFrame, target: TargetSpec) -> list[str]:
    target_cols = as_target_list(target)
    X = df.drop(columns=target_cols, errors="ignore")
    return X.select_dtypes(
        include=["object", "string[python]", "category"]
    ).columns.tolist()


def get_int_cols(df: DataFrame, target: TargetSpec) -> list[str]:
    target_cols = as_target_list(target)
    X = df.drop(columns=target_cols, errors="ignore")
    return X.select_dtypes(include="int").columns.tolist()


def get_unq_counts(
    df: DataFrame, target: TargetSpec
) -> tuple[dict[str, int], dict[str, int]]:
    target_cols = as_target_list(target)
    X = df.drop(columns=target_cols, errors="ignore").infer_objects().convert_dtypes()
    unique_counts = {}
    nanless_counts = {}
    for colname in X.columns:
        nans = df[colname].isna().sum() > 0
        try:
            unqs = np.unique(df[colname])
        except TypeError:  # happens when can't sort for unique
            unqs = np.unique(df[colname].astype(str))
        unique_counts[colname] = len(unqs)
        nanless_counts[colname] = len(unqs) - int(nans)
    return unique_counts, nanless_counts


def inflation(series: Series) -> float:
    unqs, cnts = np.unique(series.astype(str), return_counts=True)
    index = np.mean(cnts < N_CAT_LEVEL_MIN)
    return index.item()


def detect_big_cats(
    df: DataFrame,
    unique_counts: dict[str, int],
    all_cats: list[str],
    _warn: bool = True,
) -> tuple[dict[str, int], list[InflationInfo], Optional[InspectionInfo]]:
    """Detect categoricals with more than 20 levels, and "inflating" categoricals
    (see notes).

    Notes
    -----
    Inflated categoricals we deflate by converting all inflated levels to NaN.
    """
    big_cats = [
        col for col in all_cats if (col in unique_counts) and (unique_counts[col] >= 20)
    ]
    big_cols = {
        col: Inference(InferredKind.BigCat, f"{unique_counts[col]} levels")
        for col in big_cats
    }
    inspect_info = InspectionInfo(ColumnType.BigCat, big_cols)
    if _warn:
        inspect_info.print_message()

    # dict is {colname: list[columns to deflate...]}
    inflation_infos: list[InflationInfo] = []
    for col in all_cats:
        unqs, cnts = np.unique(df[col].astype(str), return_counts=True)
        n_total = len(unqs)
        if n_total <= 2:  # do not mangle boolean indicators
            continue
        keep_idx = cnts >= N_CAT_LEVEL_MIN
        if np.all(keep_idx):
            continue
        n_keep = keep_idx.sum()
        n_deflate = len(keep_idx) - n_keep
        inflation_infos.append(
            InflationInfo(
                col=col,
                to_deflate=unqs[~keep_idx].tolist(),
                to_keep=unqs[keep_idx].tolist(),
                n_deflate=n_deflate,
                n_keep=n_keep,
                n_total=n_total,
            )
        )
    return {col: unique_counts[col] for col in big_cats}, inflation_infos, inspect_info


def inspect_str_column(
    series: Series,
) -> Union[ColumnDescriptions, tuple[str, Exception]]:
    col = str(series.name)
    series = series.copy(deep=True)
    try:
        result = ColumnDescriptions(col)

        # this sould override even user-specified cardinality, as a constant
        # column as we define it here is useless no matter what
        is_const = infer_constant(series.copy(deep=True))
        if is_const:
            result.const = is_const
            if is_const.is_certain():
                return result

        # Likewise, we are not equipped to handle timeseries features, so this
        # too must override use-specified cardinality
        maybe_time = infer_timelike(series.copy(deep=True))
        if maybe_time:
            result.time = maybe_time
            if maybe_time.is_certain():
                return result  # time is bad enough we don't need other checks

        maybe_id = infer_identifier(series.copy(deep=True))
        if maybe_id:
            result.id = maybe_id
            if maybe_id.is_certain():
                return result

        is_bin = infer_binary(series)
        if is_bin:
            result.bin = is_bin
            # binary inference is always certain
            return result

        maybe_ord = infer_ordinal(series.copy(deep=True))
        if maybe_ord:
            result.ord = maybe_ord
            if maybe_ord.is_certain():
                return result

        maybe_float = infer_floatlike(series.copy(deep=True))
        if maybe_float:
            result.cont = maybe_float
            if maybe_float.is_certain():
                return result

        if maybe_id or maybe_time:
            return result

        maybe_cat = infer_categorical(series.copy(deep=True))
        if maybe_cat:
            result.cat = maybe_cat

        return result
    except Exception as e:
        traceback.print_exc()
        return col, e


def inspect_str_columns(
    df: DataFrame,
    str_cols: list[str],
    max_rows: int = 5000,
    _warn: bool = True,
) -> tuple[
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
    InspectionInfo,
]:
    """
    Returns
    -------
    float_cols: dict[str, str]
        Columns that may be continuous
    ord_cols: dict[str, str]
        Columns that may be ordinal
    id_cols: dict[str, str]
        Columns that may be identifiers
    time_cols: dict[str, str]
        Columns that may be timestamps or timedeltas
    bin_cols: dict[str, str]
        Columns that are binary
    cat_cols: dict[str, str]
        Columns that may be categorical and not specified in `--categoricals`
    const_cols: dict[str, str]
        Columns with one value (either all NaN or one value with no NaNs).
    """
    float_cols: dict[str, Inference] = {}
    ord_cols: dict[str, Inference] = {}
    id_cols: dict[str, Inference] = {}
    time_cols: dict[str, Inference] = {}
    bin_cols: dict[str, Inference] = {}
    cat_cols: dict[str, Inference] = {}
    const_cols: dict[str, Inference] = {}

    # args = tqdm()
    descs: list[Union[ColumnDescriptions, tuple[str, Exception]]] = Parallel(n_jobs=-1)(
        delayed(inspect_str_column)(df[col])
        for col in tqdm(
            str_cols,
            desc="Inspecting features",
            total=len(str_cols),
            disable=len(str_cols) < 50,
        )
    )  # type: ignore
    for desc in descs:
        if isinstance(desc, tuple):  # i.e. an error
            col, error = desc
            raise InspectionError(
                f"Could not interpret data in feature {col}. Additional information "
                "should be above."
            ) from error
        if desc.cont is not None:
            float_cols[desc.col] = desc.cont
        if desc.ord is not None:
            ord_cols[desc.col] = desc.ord
        if desc.bin is not None:
            bin_cols[desc.col] = desc.bin
        if desc.id is not None:
            id_cols[desc.col] = desc.id
        if desc.time is not None:
            time_cols[desc.col] = desc.time
        if desc.cat is not None:
            cat_cols[desc.col] = desc.cat
        if desc.const is not None:
            const_cols[desc.col] = desc.const

    float_info = InspectionInfo(ColumnType.Continuous, float_cols)
    ord_info = InspectionInfo(ColumnType.Ordinal, ord_cols)
    id_info = InspectionInfo(ColumnType.Id, id_cols)
    time_info = InspectionInfo(ColumnType.Time, time_cols)
    bin_info = InspectionInfo(ColumnType.Binary, bin_cols)
    cat_info = InspectionInfo(ColumnType.Categorical, cat_cols)
    const_info = InspectionInfo(ColumnType.Const, const_cols)

    if _warn:
        id_info.print_message()
        time_info.print_message()
        const_info.print_message()
        bin_info.print_message()
        float_info.print_message()
        ord_info.print_message()
        cat_info.print_message()

    return float_info, ord_info, id_info, time_info, bin_info, cat_info, const_info


def coerce_inferred_ambig(
    df: DataFrame,
    ambigs: dict[str, list[Inference]],
    infer_cols: set[str],
) -> dict[str, Inference]:
    final_inferences = {}
    for col in infer_cols:
        infers = ambigs[col]
        if len(infers) == 1:
            final_inferences[col] = infers[0]
            continue

        series = df[col]
        kinds = [infer.kind for infer in infers]
        if InferredKind.MaybeCont in kinds:
            if series.astype(str).str.contains(".", regex=False).any():
                final_inferences[col] = Inference(
                    InferredKind.CoercedCont,
                    "Coerced to float due to presence of decimal",
                )
                continue
            else:
                kinds.remove(InferredKind.MaybeCont)
        # now len(kinds) == 2 and we decide between cat and ord

        cat_named, match = has_cat_name(series)
        if cat_named and (inflation(series) <= 0.0):
            final_inferences[col] = Inference(
                InferredKind.CoercedCat,
                f"Coerced to categorical since well-sampled and matches pattern {match}",
            )
            continue
        else:
            final_inferences[col] = Inference(
                InferredKind.CoercedOrd,
                "Coerced to ordinal due to no decimal, feature name or undersampled levels",
            )
    return final_inferences


def coerce_user_ambig(
    df: DataFrame,
    ambigs: dict[str, list[Inference]],
    user_cols: set[str],
    arg_cats: set[str],
    arg_ords: set[str],
) -> dict[str, Inference]:
    """
    Case 1) if we are certain:                                    distrust user
    Case 2) if the feature is even MAYBE time or identifier:      distrust user
    Case 3) if the user specifies ordinal:                           trust user
    Case 4) if the user specifies categorical, then if we think:
            a) maybe categorical only:                               trust user
            b) maybe float only:
               i) if big / inflated categorical:                  distrust user
               ii) if not inflated:                                  trust user
            c) maybe ordinal and maybe categorical:
               i) if big / inflated categorical:                  distrust user
               ii) if not inflated:                                  trust user
            d) maybe float and maybe ordinal and maybe categorical:
    """
    coercions = {}
    for col in user_cols:
        ### Case (3) ###
        if col in arg_ords:  # always trust user ordinals
            coercions[col] = Inference(InferredKind.UserOrdinal, "User-specified ordinal")
            continue
        # now we know user specified categorical
        assert (
            col in arg_cats
        ), f"Logic error: {col} missing from user-specified categoricals"

        ### Case (4) ###

        infers = ambigs[col]
        if len(infers) == 1:
            infer = infers[0]
            ### Case (4a) ###
            if infer.kind is InferredKind.MaybeCat:
                coercions[col] = Inference(
                    InferredKind.UserCategorical,
                    "User-specified and inferred categorical",
                )
                continue
            ### Case (4b) ###
            elif infer.kind is InferredKind.MaybeCont:
                ### Case (4b.ii) ###
                if inflation(df[col]) <= 0.0:
                    coercions[col] = Inference(
                        InferredKind.UserCategorical,
                        "User-specified and non-inflated categorical",
                    )
                ### Case (4b.i) ###
                else:
                    coercions[col] = Inference(
                        InferredKind.CoercedCont,
                        "Looks continuous and is inflated as categorical",
                    )
            else:
                raise ValueError(f"Unaccounted for case: {infers}")

        series = df[col]
        kinds = [infer.kind for infer in infers]
        if InferredKind.MaybeCont in kinds:
            if series.astype(str).str.contains(".", regex=False).any():
                coercions[col] = Inference(
                    InferredKind.CoercedCont,
                    "Coerced to float due to presence of decimal",
                )
                continue
            else:
                kinds.remove(InferredKind.MaybeCont)
        # now len(kinds) == 2 and we decide between cat and ord

        cat_named, match = has_cat_name(series)
        if cat_named and (inflation(series) <= 0.0):
            coercions[col] = Inference(
                InferredKind.CoercedCat,
                f"Coerced to categorical since well-sampled and matches pattern {match}",
            )
            continue
        coercions[col] = Inference(
            InferredKind.CoercedOrd,
            "Coerced to ordinal due to no decimal, feature name or undersampled levels",
        )
    return coercions


def coerce_ambiguous_cols(
    df: DataFrame,
    ambigs: dict[str, list[Inference]],
    arg_cats: set[str],
    arg_ords: set[str],
    _warn: bool = True,
) -> dict[str, Inference]:
    """
    Coerce columns that cannot be clearly determined to be categorical,
    orfinal, or float to either float or categorical.


    Returns
    -------
    float_cols: dict[str, str]
        Columns coerced to float
    ord_cols: dict[str, str]
        Columns coerced to float
    cat_cols: dict[str, str]
        Columns coerced to categorical

    Notes
    -----
    This shoud generally only happen with columns where values are like:

        0.0, 1.0, 2.0, 1.0, 3.0, ...

    I.e. conversion to integer can happen without loss of precision. We examine
    the string-form of the feature and if any contain "." assume the feature
    was meant as a float, although incompetent data entry or saving was always
    possible in this case.

    In general, categoricals are expensive, so we do not want to coerce to
    categorical without good reason. We only do so when ALL the following hold:

    1. the coerced categorical has zero inflation (all levels have 50+ samples)
    2. the feature name strongly suggests a categorical interpretation

    Also, ordinals are ultimately treated no differently than continuous
    currently.

    We must infer types of columns not specified as ordinal or categorical by
    the user in order to determine one-hot encoding vs. normalization. However,
    we must also check that user-specified categoricals and ordinals can in
    fact be treated as such.

    There are thus two classes of features: user-specified and unspecified.
    For unspecified features, we need only resolve *ambiguous* types. However,
    for specified features, we must make a choice between *trusting* and
    *distrusting* the user. There should be very good reason to distrust the
    user, so we limit distrust to certain cases of user-specified categoricals,
    or when we are 100% certain of the type.

    .
    . or for each column:
    .
    .    remove  <-- certain <---- column  ----> maybe time --> distrust --> remove
    .      or          or             |             or
    .    coerce       const       ambiguous      maybe id
    .                                 |
    .                                 |
    . unspecified ____________________|_______ user-specified
    .      |                                        |
    .      |              __________________________|______________
    .      v             |                                         |
    .    remove      one maybe                            maybe cat and maybe ord
    .      or            |                   ______________________|_____
    .    coerce          |                   |                           |
    .       _____________|                   |                           |
    .      |             |                   |                           |
    .      |             |                 user                        user
    .    infer         infer                cat                         ord
    .    agree        disagree               |                           |
    .      |        _____|_____              |                           |
    .    trust     |           |             |                           |
    .    user    user        user            |                         trust
    .      |      ord         cat            |                          user
    .      |       |       ____|___       ___|_____                      |
    .    final   trust    |        |     |         |                     |
    .    type     user   big            big       cat                    |
    .              |     cat            cat        |                     |
    .              |      |              |         |                     |
    .            final  deflate        reject    trust                 final
    .             ord     |              |         |                    cat
    .                  ___|__          final     final
    .                 |      |          ord       cat
    .                >10    <=10
    .               levels  levels
    .                 |     |
    .               final  final
    .                ord    cat
    """

    ...

    final_inferences = {}
    for col, infers in ambigs.items():
        for infer in infers:
            if infer.overrides_user():
                final_inferences[col] = infer
                break
    for col in final_inferences:
        ambigs.pop(col)

    # now all that remains are maybe floats/cats/ords
    all_user_cols = arg_cats.union(arg_ords)
    infer_cols = set([*ambigs.keys()]).difference(all_user_cols)
    user_cols = set([*ambigs.keys()]).difference(infer_cols)

    coercions = coerce_inferred_ambig(df, ambigs=ambigs, infer_cols=infer_cols)
    final_inferences.update(coercions)
    coercions = coerce_user_ambig(
        df, ambigs=ambigs, user_cols=user_cols, arg_cats=arg_cats, arg_ords=arg_ords
    )
    final_inferences.update(coercions)

    return final_inferences


def convert_categorical(series: Series) -> Series:
    # https://stackoverflow.com/a/70442594
    # Pandas so dumb, why it would silently ignore casting to string and retain
    # the categorical dtype makes no sense, e.g.
    #
    #       df[col] = df[col].astype("string")
    #
    # fails to do anything but silently passes without warning or error
    if series.dtype == "category":
        return series.astype(series.cat.categories.to_numpy().dtype)
    return series


def convert_categoricals(
    df: DataFrame, target: TargetSpec, grouper: Optional[str]
) -> DataFrame:
    # convert screwy categorical columns which can have all sorts of
    # annoying behaviours when incorrectly labeled as such
    target_cols = as_target_list(target)
    df = df.copy()
    for col in df.columns:
        if str(col) in target_cols:
            continue
        if (grouper is not None) and str(col) == grouper:
            continue
        df[col] = convert_categorical(df[col])
    if grouper is not None:
        df[grouper] = df[grouper].astype("category").cat.codes

    return df


@overload
def unify_nans(df: DataFrame) -> DataFrame: ...


@overload
def unify_nans(df: Series) -> Series: ...


def unify_nans(df: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
    df = df.map(lambda x: np.nan if str(x) in NAN_STRINGS else x)  # type: ignore
    return df


def inspect_data(
    df: DataFrame,
    target: TargetSpec,
    grouper: Optional[str] = None,
    categoricals: Optional[list[str]] = None,
    ordinals: Optional[list[str]] = None,
    drops: Optional[list[str]] = None,
    _warn: bool = True,
) -> tuple[DataFrame, InspectionResults]:
    """Attempt to infer column types"""
    categoricals = categoricals or []
    ordinals = ordinals or []
    target_cols = as_target_list(target)
    y_df = df[target_cols].copy()
    g = df[grouper] if (grouper is not None) else None

    df = df.drop(columns=target_cols, errors="ignore")
    if g is not None:
        df = df.drop(columns=grouper, errors="ignore")

    if drops is not None:
        drops = sorted(set(drops))
        df = df.drop(columns=drops, errors="ignore")

    arg_cats, arg_ords = set(categoricals), set(ordinals)

    all_cols = set(df.columns.to_list())
    user_cols = arg_cats.union(arg_ords).intersection(all_cols)
    unk_cols = list(all_cols.difference(user_cols))

    (
        floats,
        ords,
        ids,
        times,
        bins,
        cats,
        consts,
    ) = inspect_str_columns(df, unk_cols, _warn=_warn)
    (
        user_floats,
        user_ords,
        user_ids,
        user_times,
        user_bins,
        user_cats,
        user_consts,
    ) = inspect_str_columns(df, list(user_cols), _warn=_warn)

    certains, ambigs = InspectionInfo.conflicts(
        floats,
        ords,
        ids,
        times,
        bins,
        cats,
        consts,
        user_floats,
        user_ords,
        user_ids,
        user_times,
        user_bins,
        user_cats,
        user_consts,
    )
    if sorted(all_cols) != sorted([*certains.keys()]):
        diffs = all_cols.symmetric_difference([*certains.keys()])
        raise ValueError(f"Missing inference for columns: {diffs}")

    # if we are certain, that is the type, regardles of user specification
    certain_types = {
        col: infer[0]
        for col, infer in certains.items()
        if (len(infer) == 1) and infer[0].is_certain()
    }
    for col in certain_types:
        ambigs.pop(col)

    final_coercions = coerce_ambiguous_cols(
        df, ambigs, arg_cats=arg_cats, arg_ords=arg_ords, _warn=_warn
    )
    if len(ambigs) > 0 and _warn:
        messy_inform(
            "df-analyze could not determine the types of some features. These "
            "have been coerced to our best guess for the appropriate type. See "
            "reports for details. This warning cannot be silenced unless you "
            "properly identify these features with the `--categoricals` and "
            "`--ordinals` options to df-analyze.\n\n"
            f"Ambiguous columns: {ambigs}."
        )

    final_types = {**certain_types, **final_coercions}

    all_cats = [col for col, info in final_types.items() if info.is_cat()]

    unique_counts, nanless_cnts = get_unq_counts(df=df, target=target_cols)
    bigs, inflation = detect_big_cats(df, unique_counts, all_cats, _warn=_warn)[:-1]

    multi_cats = {col for col, cnt in nanless_cnts.items() if cnt > 2}
    multi_cats = sorted(multi_cats.intersection(all_cats))

    # fmt: off
    final_cats   = {col: info for col, info in final_types.items() if info.is_cat()}
    final_conts  = {col: info for col, info in final_types.items() if info.is_cont()}
    final_ords   = {col: info for col, info in final_types.items() if info.is_ord()}
    final_ids    = {col: info for col, info in final_types.items() if info.is_id()}
    final_times  = {col: info for col, info in final_types.items() if info.is_time()}
    final_consts = {col: info for col, info in final_types.items() if info.is_const()}
    final_bins   = {col: info for col, info in final_types.items() if info.is_bin()}
    # fmt: on

    df = pd.concat([df, y_df], axis=1)
    df[grouper] = g
    df = df.rename(str, axis="columns")  # https://stackoverflow.com/a/77046151

    return df, InspectionResults(
        conts=InspectionInfo(ColumnType.Continuous, final_conts),
        ords=InspectionInfo(ColumnType.Ordinal, final_ords),
        ids=InspectionInfo(ColumnType.Id, final_ids),
        times=InspectionInfo(ColumnType.Time, final_times),
        consts=InspectionInfo(ColumnType.Const, final_consts),
        cats=InspectionInfo(ColumnType.Categorical, final_cats),
        binaries=InspectionInfo(ColumnType.Binary, final_bins),
        big_cats=bigs,
        inflation=inflation,
        multi_cats=multi_cats,
        user_cats=arg_cats,
        user_ords=arg_ords,
        drops=set(drops) if drops is not None else set(),
    )


def inspect_data_cached(
    options: ProgramOptions,
    memory: Memory,
) -> tuple[DataFrame, InspectionResults]:
    target: TargetSpec = options.targets
    if options.program_dirs.root is None:  # can't write files, no cache
        return inspect_data(
            df=options.load_df(),
            target=target,
            grouper=None,
            categoricals=options.categoricals,
            ordinals=options.ordinals,
            _warn=True,
        )
    raise NotImplementedError()


def inspect_cls_target(series: Series) -> ClsTargetInfo:
    series = unify_nans(series.copy(deep=True))
    inflation, unqs, cnts = InflationInfo.from_series(series)
    if len(unqs) <= 1:
        raise ValueError(f"Classification target '{series.name}' is constant.")
    n_values = len(series.dropna().unique())
    if n_values <= 1:
        raise ValueError(
            f"Classification target '{series.name}' is constant after dropping NaNs."
        )

    p_max = np.max(cnts) / np.sum(cnts)
    p_min = np.min(cnts) / np.sum(cnts)
    p_nan = unify_nans(series).isna().mean()
    return ClsTargetInfo(
        name=str(series.name),
        inflation=inflation,
        unqs=unqs,
        cnts=cnts,
        p_max_cls=p_max,
        p_min_cls=p_min,
        p_nan=p_nan,
    )


def inspect_reg_target(series: Series) -> RegTargetInfo:
    p_nan = unify_nans(series).isna().mean()
    if p_nan >= 1.0:
        raise ValueError(f"Regression target '{series.name}' is all NaN.")
    if series.dtype.kind == "c":
        y = np.asarray(series.values, dtype=np.complex128)
        y = y.real
    else:
        y = np.asarray(series.values, dtype=np.float64)

    if np.isinf(y).any():
        raise ValueError(f"Regression target '{series.name}' contains infinite values.")
    var = np.nanvar(y, ddof=1)
    if float(var) <= 0:
        raise ValueError(f"Regression target {series.name} is constant.")

    return RegTargetInfo(
        name=str(series.name), needs_logarithm=False, has_outliers=False, p_nan=p_nan
    )


def inspect_target(
    df: DataFrame, target: str, is_classification: bool
) -> Union[ClsTargetInfo, RegTargetInfo]:
    y = df[target]
    if is_classification:
        return inspect_cls_target(y)
    return inspect_reg_target(y)

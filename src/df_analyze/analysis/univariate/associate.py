from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
import traceback
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import choice
from typing import (
    Any,
    Optional,
    Type,
)
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame, Series
from scipy.stats import (
    brunnermunzel,
    kruskal,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
)
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif as minfo_cat
from sklearn.feature_selection import mutual_info_regression as minfo_cont
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from df_analyze.analysis.metrics import auroc, cohens_d, cramer_v
from df_analyze.enumerables import RandEnum
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.saving import add_fold_idx


class Association:
    def has_significance(self) -> bool:
        raise NotImplementedError()

    def higher_is_better(self) -> bool:
        raise NotImplementedError()

    def p_value(self) -> Optional[str]:
        raise NotImplementedError()

    @staticmethod
    def default() -> Enum:
        raise NotImplementedError()


class ContAssociation(Association):
    @classmethod
    def random(cls: Type[Association]) -> ContAssociation:
        return choice(
            [
                ContClsStats.random(),
                ContRegStats.random(),
            ]
        )


class CatAssociation(Association):
    @classmethod
    def random(cls: Type[Association]) -> CatAssociation:
        return choice(
            [
                CatClsStats.random(),
                CatRegStats.random(),
            ]
        )


class ContClsStats(ContAssociation, RandEnum, Enum):
    TTest = "t"
    MannWhitneyU = "U"
    BrunnerMunzelW = "W"
    Correlation = "corr"
    CohensD = "cohen_d"
    AUROC = "AUROC"
    MutualInfo = "mut_info"

    @staticmethod
    def default() -> ContClsStats:
        return ContClsStats.MutualInfo

    def has_significance(self) -> bool:
        return {
            ContClsStats.TTest: True,
            ContClsStats.MannWhitneyU: True,
            ContClsStats.BrunnerMunzelW: True,
            ContClsStats.Correlation: True,
            ContClsStats.CohensD: False,
            ContClsStats.AUROC: False,
            ContClsStats.MutualInfo: False,
        }[self]

    def higher_is_better(self) -> bool:
        return {
            ContClsStats.TTest: True,
            ContClsStats.MannWhitneyU: True,
            ContClsStats.BrunnerMunzelW: True,
            ContClsStats.Correlation: True,
            ContClsStats.CohensD: True,
            ContClsStats.AUROC: True,
            ContClsStats.MutualInfo: True,
        }[self]

    def p_value(self) -> str:
        if not self.has_significance():
            raise ValueError(f"No statistical signficance for {self}")
        return {
            ContClsStats.TTest: "t_p",
            ContClsStats.MannWhitneyU: "U_p",
            ContClsStats.BrunnerMunzelW: "W_p",
            ContClsStats.Correlation: "corr_p",
        }[self]

    def longname(self) -> str:
        return {
            ContClsStats.TTest: "T-Test",
            ContClsStats.MannWhitneyU: "Mann-Whitney U",
            ContClsStats.BrunnerMunzelW: "Brunner-Munzel W",
            ContClsStats.Correlation: "Pearson Correlation",
            ContClsStats.CohensD: "Cohen's d",
            ContClsStats.AUROC: "Directionless AUROC Association Strength",
            ContClsStats.MutualInfo: "Mutual Information",
        }[self]

    @classmethod
    def random(cls: Type[ContClsStats]) -> ContClsStats:
        return choice([*cls])  # type: ignore


class CatClsStats(CatAssociation, RandEnum, Enum):
    MutualInfo = "mut_info"  # sklearn.feature_selection.mutual_info_regression
    KruskalWallaceH = "H"
    CramerV = "cramer_v"  # Cramer's V

    @staticmethod
    def default() -> CatClsStats:
        # on test datasets, Cramer V and mut_info have Pearson correlations
        # over 0.94 and Spearman correlations of up to 1.0 and most above
        # 0.96. CramerV is in [0, 1], this gives the illusion of
        # interpretability and comparability that people usually want, but
        # also it can fail where mutual information does not.
        return CatClsStats.MutualInfo

    def has_significance(self) -> bool:
        return {
            CatClsStats.MutualInfo: False,
            CatClsStats.KruskalWallaceH: True,
            CatClsStats.CramerV: False,
        }[self]

    def higher_is_better(self) -> bool:
        return {
            CatClsStats.MutualInfo: True,
            CatClsStats.KruskalWallaceH: True,
            CatClsStats.CramerV: True,
        }[self]

    def p_value(self) -> Optional[str]:
        if not self.has_significance():
            raise ValueError(f"No statistical signficance for {self}")
        return {
            CatClsStats.MutualInfo: None,
            CatClsStats.KruskalWallaceH: "H_p",
            CatClsStats.CramerV: None,
        }[self]

    def longname(self) -> str:
        return {
            CatClsStats.MutualInfo: "Mutual Information",
            CatClsStats.KruskalWallaceH: "Kruskal-Wallace H",
            CatClsStats.CramerV: "Cramer's V",
        }[self]

    @classmethod
    def random(cls: Type[CatClsStats]) -> CatClsStats:
        return choice([*cls])  # type: ignore


class ContRegStats(ContAssociation, RandEnum, Enum):
    PearsonR = "pearson_r"
    SpearmanR = "spearman_r"
    MutualInfo = "mut_info"
    F = "F"

    @staticmethod
    def default() -> ContRegStats:
        return ContRegStats.F

    def has_significance(self) -> bool:
        return {
            ContRegStats.PearsonR: True,
            ContRegStats.SpearmanR: True,
            ContRegStats.MutualInfo: False,
            ContRegStats.F: True,
        }[self]

    def higher_is_better(self) -> bool:
        return {
            ContRegStats.PearsonR: True,
            ContRegStats.SpearmanR: True,
            ContRegStats.MutualInfo: True,
            ContRegStats.F: True,
        }[self]

    def p_value(self) -> Optional[str]:
        if not self.has_significance():
            raise ValueError(f"No statistical signficance for {self}")
        return {
            ContRegStats.PearsonR: "pearson_p",
            ContRegStats.SpearmanR: "spearman_p",
            ContRegStats.MutualInfo: None,
            ContRegStats.F: "F_p",
        }[self]

    def longname(self) -> str:
        return {
            ContRegStats.PearsonR: "Pearson's Correlation",
            ContRegStats.SpearmanR: "Spearman's Correlation",
            ContRegStats.MutualInfo: "Mutual Information",
            ContRegStats.F: "F-test",
        }[self]

    @classmethod
    def random(cls: Type[ContRegStats]) -> ContRegStats:
        return choice([*cls])  # type: ignore


class CatRegStats(CatAssociation, RandEnum, Enum):
    MutualInfo = "mut_info"  # sklearn.feature_selection.mutual_info_regression
    H = "H"  # Kruskal-Wallace H

    @staticmethod
    def default() -> CatRegStats:
        return CatRegStats.H

    def has_significance(self) -> bool:
        return {
            CatRegStats.MutualInfo: False,
            CatRegStats.H: True,
        }[self]

    def higher_is_better(self) -> bool:
        return {
            CatRegStats.MutualInfo: True,
            CatRegStats.H: True,
        }[self]

    def p_value(self) -> Optional[str]:
        if not self.has_significance():
            raise ValueError(f"No statistical signficance for {self}")
        return {
            CatRegStats.MutualInfo: None,
            CatRegStats.H: "H_p",
        }[self]

    def longname(self) -> str:
        return {
            CatRegStats.MutualInfo: "Mutual Information",
            CatRegStats.H: "Kruskal-Wallace H",
        }[self]

    @classmethod
    def random(cls: Type[CatRegStats]) -> CatRegStats:
        return choice([*cls])  # type: ignore


CONT_FEATURE_CAT_TARGET_LEVEL_STATS = [
    "t",
    "t_p",
    "U",
    "U_p",
    "W",
    "W_p",
    "cohen_d",
    "AUROC",
    "corr",
    "corr_p",
    "mut_info",
]

CONT_FEATURE_CONT_TARGET_STATS = [
    "pearson_r",
    "pearson_p",
    "spearman_r",
    "spearman_p",
    "F",
    "F_p",
    "mut_info",
]

CAT_FEATURE_CONT_TARGET_STATS = [
    "mut_info",  # sklearn.feature_selection.mutual_info_regression
    "H",  # Kruskal-Wallace H
    "H_p",
]

CAT_FEATURE_CAT_TARGET_LEVEL_STATS = ["cramer_v"]
CAT_FEATURE_CAT_TARGET_STATS = ["cramer_v", "H", "mut_info"]


@dataclass
class AssocFiles:
    conts_raw = "continuous_features.parquet"
    cats_raw = "categorical_features.parquet"
    conts_csv = "continuous_features.csv"
    cats_csv = "categorical_features.csv"


class AssocResults:
    def __init__(
        self,
        conts: Optional[DataFrame],
        cats: Optional[DataFrame],
        is_classification: bool,
    ) -> None:
        self.conts = conts
        self.cats = cats
        self.is_classification = is_classification
        self.files = AssocFiles()

    def to_markdown(self, path: Optional[Path] = None) -> Optional[str]:
        try:
            if self.conts is not None:
                conts = self.conts.sort_values(by="mut_info", ascending=False)
                conts = conts.map(lambda x: x if abs(x) > 1e-10 else 0)
                cols = conts.columns.to_list()
                cols.remove("mut_info")
                cols = ["mut_info"] + cols
                conts_table = conts.loc[:, cols].to_markdown(floatfmt="<0.03g")
            else:
                conts_table = ""
            if self.cats is not None:
                cats = self.cats.sort_values(by="mut_info", ascending=False)
                cats = cats.map(lambda x: x if abs(x) > 1e-10 else 0)
                cols = cats.columns.to_list()
                cols.remove("mut_info")
                cols = ["mut_info"] + cols
                cats_table = cats.loc[:, cols].to_markdown(floatfmt="<0.03g")
            else:
                cats_table = ""

            conts_table = conts_table.replace("nan", "   ")
            cats_table = cats_table.replace("nan", "   ")

            conts_table = (
                f"# Continuous associations\n\n{conts_table}\n\n"
                if self.conts is not None
                else ""
            )
            cats_table = (
                f"# Categorical associations\n\n{cats_table}"
                if self.cats is not None
                else ""
            )
            tables = conts_table + cats_table
            if tables.replace("\n", "") != "":
                tables = (
                    f"{tables}\n\n**Note**: values less than 1e-10 are rounded to zero.\n"
                )
                if path is not None:
                    path.write_text(tables, encoding="utf-8")
                return tables
        except Exception as e:
            warn(
                "Got exception when attempting to make associations report. "
                f"Details:\n{e}\n{traceback.format_exc()}"
            )

    def save_tables(self, root: Path, fold_idx: Optional[int]) -> None:
        if self.conts is not None:
            self.conts.to_csv(add_fold_idx(root / self.files.conts_csv, fold_idx))
        if self.cats is not None:
            self.cats.to_csv(add_fold_idx(root / self.files.cats_csv, fold_idx))

    def save_raw(self, root: Path, fold_idx: Optional[int]) -> None:
        conts = DataFrame() if self.conts is None else self.conts
        cats = DataFrame() if self.cats is None else self.cats

        conts.to_parquet(add_fold_idx(root / self.files.conts_raw, fold_idx))
        cats.to_parquet(add_fold_idx(root / self.files.cats_raw, fold_idx))

    @staticmethod
    def is_saved(cachedir: Path) -> bool:
        files = AssocFiles()
        conts = cachedir / files.conts_raw
        cats = cachedir / files.cats_raw
        return conts.exists() and cats.exists()

    @staticmethod
    def load(cachedir: Path, is_classification: bool) -> AssocResults:
        preds = AssocResults(conts=None, cats=None, is_classification=is_classification)
        try:
            conts = pd.read_parquet(cachedir / preds.files.conts_raw)
            cats = pd.read_parquet(cachedir / preds.files.cats_raw)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing cached associations at {cachedir}")

        preds.conts = None if conts.empty else conts
        preds.cats = None if cats.empty else cats
        return preds


def cont_feature_cat_target_level_stats(x: Series, y: Series, level: Any) -> DataFrame:
    stats = CONT_FEATURE_CAT_TARGET_LEVEL_STATS

    idx_level = y == level
    y_bin = idx_level.astype(float)
    g0 = x[~idx_level]
    g1 = x[idx_level]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation",
            category=RuntimeWarning,
        )
        tt_res = ttest_ind(g0, g1, equal_var=False)
    t, t_p = tt_res.statistic, tt_res.pvalue  # type: ignore
    U_res = mannwhitneyu(g0, g1)
    U, U_p = U_res.statistic, U_res.pvalue
    W_res = brunnermunzel(
        g0, g1, distribution="normal"
    )  # avoid errors for `distribution="t"`
    W, W_p = W_res.statistic, W_res.pvalue

    if np.nanstd(x) == 0:
        r, r_p = np.nan, np.nan
        warn(f"Input `x`: {x.name} is constant")
    elif np.nanstd(y_bin) == 0:
        raise ValueError(f"Target `y_bin` ({y.name}) is constant")
    else:
        r_res = pearsonr(x, y_bin)
        r, r_p = r_res.statistic, r_res.pvalue  # type: ignore

    try:
        # copy is needed to prevent a strange error:
        # ValueError: cannot set WRITEABLE flag to True of this array
        auc = auroc(x.copy(), idx_level.astype(int).copy())
    except Exception as e:
        print(f"Got error:\n{e}")
        traceback.print_exc()
        auc = np.nan

    try:
        # copy is needed to prevent a strange error:
        # ValueError: cannot set WRITEABLE flag to True of this array
        minfo = minfo_cat(x.to_frame().copy(), y_bin)
    except Exception as e:
        print(f"Got error:\n{e}")
        traceback.print_exc()
        minfo = np.nan

    data = {
        "t": t,
        "t_p": t_p,
        "U": U,
        "U_p": U_p,
        "W": W,
        "W_p": W_p,
        "cohen_d": cohens_d(g0, g1),
        # copies below to maybe avoid an odd WRITABLE bug...
        "AUROC": auc,
        "corr": r,
        "corr_p": r_p,
        "mut_info": minfo,
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}__{y.name}.{level}"],
        columns=stats,
    )


def cont_feature_cont_target_stats(x: Series, y: Series) -> DataFrame:
    """
    Parameters
    ----------

    x: Series
        Continuous feature

    y: Series
        Continuous target

    Returns
    -------
    stats: DataFrame
        Table of stats
    """
    stats = CONT_FEATURE_CONT_TARGET_STATS

    xx = x.to_numpy().ravel()
    yy = y.to_numpy().ravel()

    r_res = pearsonr(xx, yy)
    r, r_p = r_res.statistic, r_res.pvalue  # type: ignore
    rs_res = spearmanr(xx, yy)
    rs, rs_p = rs_res.statistic, rs_res.pvalue  # type: ignore
    F, F_p = f_regression(xx.reshape(-1, 1), yy)

    data = {
        "pearson_r": r,
        "pearson_p": r_p,
        "spearman_r": rs,
        "spearman_p": rs_p,
        "F": F,
        "F_p": F_p,
        "mut_info": minfo_cont(xx.reshape(-1, 1), yy),
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}"],
        columns=stats,
    )


def continuous_feature_target_stats(
    continuous: DataFrame,
    column: str,
    target: Series,
    is_classification: bool,
) -> DataFrame:
    ...
    x = continuous[column]
    y = target
    if len(x) != len(y):
        raise ValueError(
            "Continuous features and target do not have same number of samples."
        )
    if is_classification:
        levels = np.unique(y).tolist()
        descs = []
        for level in levels:
            desc = cont_feature_cat_target_level_stats(x, y, level=level)
            descs.append(desc)
        desc = pd.concat(descs, axis=0)
        is_multiclass = len(levels) > 2
        if is_multiclass:
            # TODO: collect mean stats when this makes sense?
            # TODO: collect some other fancy stat?
            ...
        return desc

    return cont_feature_cont_target_stats(x, y)


def cat_feature_cont_target_stats(x: Series, y: Series) -> DataFrame:
    stats = CAT_FEATURE_CONT_TARGET_STATS

    xx = x.astype(str).to_numpy().ravel()
    x_enc = np.asarray(LabelEncoder().fit_transform(xx)).reshape(-1, 1)

    minfo = minfo_cont(x_enc, y, discrete_features=True)
    H, H_p = kruskal(x_enc.ravel(), y)

    data = {
        "mut_info": minfo,
        "H": H,
        "H_p": H_p,
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}"],
        columns=stats,
    )


def cat_feature_cat_target_level_stats(
    x: Series, y: Series, level: str, label: str
) -> DataFrame:
    stats = ["cramer_v", "H", "H_p", "mut_info"]
    idx_level = y == level
    y_bin = idx_level.astype(np.int64)

    xx = x.astype(str).to_numpy().ravel()
    x_enc = np.asarray(LabelEncoder().fit_transform(xx)).reshape(-1, 1)
    try:
        H, H_p = kruskal(x_enc.ravel(), y_bin)
    except ValueError:  # "All numbers are identical in kruskal"
        H, H_p = 0, np.nan

    minfo = minfo_cat(x_enc.reshape(-1, 1), y_bin, discrete_features=True)

    data = {
        "cramer_v": cramer_v(x_enc, y_bin),
        "H": H,
        "H_p": H_p,
        "mut_info": minfo.item(),
    }

    return DataFrame(
        data=data,
        index=[f"{x.name}__{y.name}.{label}"],
        columns=stats,
    )


def categorical_feature_target_stats(
    categoricals: DataFrame,
    column: str,
    target: Series,
    labels: Optional[dict[int, str]],
    is_classification: bool,
) -> DataFrame:
    try:
        x = categoricals[column]
        y = target

        if is_classification:
            assert labels is not None, "Missing target labels"
            xx = x.astype(str).to_numpy().ravel()
            x_enc = np.asarray(LabelEncoder().fit_transform(xx)).ravel()
            xs = Series(data=x_enc, name=x.name)
            levels = np.unique(y).tolist()

            descs = []
            for level in levels:
                label = labels[int(level)]
                desc = cat_feature_cat_target_level_stats(xs, y, level=level, label=label)
                descs.append(desc)
            desc = pd.concat(descs, axis=0)

            # if y.dtype == "object":
            #     y = y.astype(str)
            V = cramer_v(x_enc.reshape(-1, 1), y.astype(str))
            minfo = minfo_cat(x_enc.reshape(-1, 1), y.astype(str), discrete_features=True)
            df = DataFrame(data={"cramer_v": V, "mut_info": minfo}, index=[f"{x.name}"])
            desc = desc.dropna(axis="columns", how="all")
            desc = pd.concat([desc, df], axis=0)
            # TODO: collect mean stats when this makes sense?
            # TODO: collect some other fancy stat?
            return desc

        return cat_feature_cont_target_stats(x, y)
    except Exception as e:
        print(f"Got exception: {e}")
        traceback.print_exc()
        return DataFrame()


def target_associations(
    prepared: PreparedData,
) -> AssocResults:
    """
    For each non-categorical (including ordinal) feature:

        Binary classification target:
            - t-test
            - Mann-Whitney U
            - Brunner-Munzel W
            - Cohen's d
            - AUROC
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - as above

        Regression target:
            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    For each categorical feature:

        Binary classificataion target:
            - Cramer's V
            - Mutual Info (sklearn.feature_selection.mutual_info_classif)

        Multiclass classification target (for each target class / level):
            - as above

        Multiclass classification target (means over each target class above):
            - means of above

        Regression target:
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)
            - Kruskal-Wallace H? (scipy.stats.kruskal) (or ANOVA)
            - mean AUROC of each level? (No...)
            - max AUROC of each level? (Yes?)

        Kruskal-Wallace H basically looks at the distribution of continuous
        values for each level of the categorical, and compares if the medians
        are different. Thus, a low H value implies each level sort of looks
        the same on the continuous target, and implies there is not much
        predictive value of the categorical variable, whereas a high H value
        implies the opposite.

            - Pearson correlation (scipy.stats.pearsonr)
            - Spearman correlation (scipy.stats.spearmanr)
            - F-statistic (sklearn.feature_selection.f_regression)
            - Mutual Info (sklearn.feature_selection.mutual_info_regression)

    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df_cats = []
        df_conts = []

        cont = prepared.X_cont
        if cont is not None:
            df_conts: list[DataFrame] = Parallel(n_jobs=-1)(
                delayed(continuous_feature_target_stats)(
                    continuous=cont,
                    column=col,
                    target=prepared.y,
                    is_classification=prepared.is_classification,
                )
                for col in tqdm(
                    cont.columns,
                    desc="Computing associations for continuous features",
                    total=cont.shape[1],
                )
            )  # type: ignore

        # Currently X_cat is the raw cat data, neither label- nor one-hot-
        # encoded.
        cats = prepared.X_cat
        if cats is not None:
            df_cats: list[DataFrame] = Parallel(n_jobs=-1)(
                delayed(categorical_feature_target_stats)(
                    categoricals=cats,
                    column=col,
                    target=prepared.y,
                    labels=prepared.labels,
                    is_classification=prepared.is_classification,
                )
                for col in tqdm(
                    cats.columns,
                    desc="Computing associations for categorical features",
                    total=cats.shape[1],
                )
            )  # type: ignore

        df_cont = pd.concat(df_conts, axis=0) if len(df_conts) > 0 else None
        df_cat = pd.concat(df_cats, axis=0) if len(df_cats) > 0 else None

    return AssocResults(
        conts=df_cont,
        cats=df_cat,
        is_classification=prepared.is_classification,
    )

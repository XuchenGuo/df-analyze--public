from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from random import choice, randint, uniform
from typing import TYPE_CHECKING, Literal, Optional, Type, Union, overload
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from df_analyze.analysis.univariate.associate import (
    AssocResults,
    CatAssociation,
    CatClsStats,
    CatRegStats,
    ContAssociation,
    ContClsStats,
    ContRegStats,
)
from df_analyze.analysis.univariate.predict.predict import PredResults

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
    from df_analyze.testing.datasets import TestDataset
from df_analyze.enumerables import ClsScore, RegScore
from df_analyze.preprocessing.prepare import PreparedData


@dataclass
class FilterSelected:
    selected: list[str]
    cont_scores: Optional[Series]
    cat_scores: Optional[Series]
    method: Literal["association", "prediction"]
    is_classification: bool

    def to_markdown(self) -> str:
        cont_name = self.cont_scores.name if self.cont_scores is not None else None
        cat_name = self.cat_scores.name if self.cat_scores is not None else None
        cont_cls, cat_cls = FilterSelected.get_cont_cat_metrics(
            method=self.method, is_classification=self.is_classification
        )
        cont_metric = None if cont_name is None else cont_cls(cont_name)
        cat_metric = None if cat_name is None else cat_cls(cat_name)

        text = (
            "# Filter-Based Feature Selection Summary\n\n"
            "\n"
            f"## Selected Features\n\n"
            f"{self.selected}\n\n"
            f"## Selection {self.method.capitalize()} Scores \n\n"
        )
        if cont_metric is not None:
            higher_better = cont_metric.higher_is_better()
            direction = "Higher" if higher_better else "Lower"
            text = text + (
                f"### Continuous Features ({cont_metric.longname()}: "
                f"{direction} = More important)\n\n"
                f"{self.cont_scores.to_frame().to_markdown(floatfmt='0.3e')}\n\n"  # type: ignore
            )
        if cat_metric is not None:
            higher_better = cat_metric.higher_is_better()
            direction = "Higher" if higher_better else "Lower"
            text = text + (
                f"### Categorical Features ({cat_metric.longname()}: "
                f"{direction} = More important)\n\n"
                f"{self.cat_scores.to_frame().to_markdown(floatfmt='0.3e')}"  # type: ignore
            )
        return text

    @staticmethod
    def get_cont_cat_metrics(
        method: Literal["association", "prediction"], is_classification: bool
    ) -> tuple[
        Union[
            Type[ContClsStats],
            Type[ContRegStats],
            Type[RegScore],
            Type[ClsScore],
        ],
        Union[
            Type[CatClsStats],
            Type[CatRegStats],
            Type[RegScore],
            Type[ClsScore],
        ],
    ]:
        if is_classification:
            if method == "association":
                cont_metric = ContClsStats
                cat_metric = CatClsStats
            else:
                cont_metric = cat_metric = ClsScore
        else:
            if method == "association":
                cont_metric = ContRegStats
                cat_metric = CatRegStats
            else:
                cont_metric = cat_metric = RegScore
        return cont_metric, cat_metric

    @staticmethod
    def random(
        ds: TestDataset, method: Optional[Literal["association", "prediction"]] = None
    ) -> FilterSelected:
        df = ds.load().drop(columns="target", errors="ignore")
        cols = df.columns.to_list()
        n_selected = randint(min(10, df.shape[1]), df.shape[1])
        n_cont = randint(4, n_selected)
        selected = np.random.choice(cols, size=n_selected, replace=False).tolist()
        cont_selected = np.random.choice(selected, size=n_cont, replace=False).tolist()
        cat_selected = [s for s in selected if s not in cont_selected]
        cont_scores = Series({name: uniform(0, 1) for name in cont_selected})
        cat_scores = Series({name: uniform(0, 1) for name in cat_selected})
        if method is None:
            method = choice(["association", "prediction"])

        cont_cls, cat_cls = FilterSelected.get_cont_cat_metrics(
            method=method,  # type: ignore
            is_classification=ds.is_classification,
        )
        cont_metric = cont_cls.random()
        cat_metric = cat_cls.random()

        cont_scores.name = cont_metric.value
        cat_scores.name = cat_metric.value
        return FilterSelected(
            selected=selected,
            cont_scores=cont_scores,
            cat_scores=cat_scores,
            method=method,  # type: ignore
            is_classification=ds.is_classification,
        )


def _default_from_X(
    X: DataFrame,
    default_p: float,
    default_min: int,
    default_max: int,
    requested: Optional[Union[int, float]] = None,
) -> int:
    if X is None or X.empty:
        return 0

    n_feat = X.shape[1]

    if isinstance(requested, int):
        # do not limit to max if user specifies valid int number of features
        return min(n_feat, requested)

    if isinstance(requested, float):
        if (requested < 0) or (requested > 1):
            warn(
                f"Invalid percent of features requested: {requested}. Must be "
                f"value in [0, 1]. Setting to default value of {default_p}"
            )
            requested = default_p
        else:  # valid float
            percent = requested
            n = ceil(percent * n_feat)
            return min(n_feat, n)

    # now either caller provided invalid request or it was None
    if requested is None:
        requested = default_p

    assert isinstance(requested, float)

    if n_feat <= default_min:
        return n_feat
    n = ceil(requested * n_feat)

    return min(default_max, n)


def n_cont_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        prepared.X_cont,
        default_p=0.2,
        default_min=10,
        default_max=50,
        requested=requested,
    )


def n_cat_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        prepared.X_cat,
        default_p=0.1,
        default_min=5,
        default_max=20,
        requested=requested,
    )


def n_total_select_default(
    prepared: PreparedData, requested: Optional[Union[int, float]] = None
) -> int:
    return _default_from_X(
        pd.concat([prepared.X_cat, prepared.X_cont], axis=1),
        default_p=0.25,
        default_min=8,
        default_max=50,
        requested=requested,
    )


def _resolve_filter_counts(
    prepared: PreparedData,
    n_cont: Optional[Union[int, float]],
    n_cat: Optional[Union[int, float]],
    n_total: Optional[Union[int, float]],
) -> tuple[int, int]:
    """Resolve the documented two-of-three filter feature-count contract."""
    available_cont = (
        0 if prepared.X_cont is None else int(prepared.X_cont.shape[1])
    )
    available_cat = 0 if prepared.X_cat is None else int(prepared.X_cat.shape[1])

    if n_cont is not None and n_cat is not None:
        resolved_cont = n_cont_select_default(prepared, n_cont)
        resolved_cat = n_cat_select_default(prepared, n_cat)
        if n_total is not None:
            resolved_total = n_total_select_default(prepared, n_total)
            if resolved_total != resolved_cont + resolved_cat:
                warn(
                    "When specifying all of `n_cat`, `n_cont`, and `n_total`, "
                    "will ignore count specified for `n_total`."
                )
        return resolved_cont, resolved_cat

    resolved_total = n_total_select_default(prepared, n_total)
    resolved_total = min(resolved_total, available_cont + available_cat)

    if n_cont is not None:
        resolved_cont = min(
            n_cont_select_default(prepared, n_cont),
            resolved_total,
        )
        resolved_cat = min(available_cat, resolved_total - resolved_cont)
        return resolved_cont, resolved_cat

    if n_cat is not None:
        resolved_cat = min(
            n_cat_select_default(prepared, n_cat),
            resolved_total,
        )
        resolved_cont = min(available_cont, resolved_total - resolved_cat)
        return resolved_cont, resolved_cat

    available_total = available_cont + available_cat
    if available_total == 0 or resolved_total == 0:
        return 0, 0

    resolved_cont = round(resolved_total * available_cont / available_total)
    resolved_cont = min(available_cont, max(0, resolved_cont))
    resolved_cat = min(available_cat, resolved_total - resolved_cont)
    if resolved_cont + resolved_cat < resolved_total:
        resolved_cont = min(
            available_cont,
            resolved_total - resolved_cat,
        )
    return resolved_cont, resolved_cat


def _collapse_association_scores(
    scores: Series,
    feature_columns: list[str],
    *,
    higher_is_better: bool,
) -> Series:
    """Collapse per-target-level association rows back to source features."""
    columns = sorted((str(col) for col in feature_columns), key=len, reverse=True)
    buckets: dict[str, list[float]] = {col: [] for col in columns}
    for row_name, value in scores.items():
        row = str(row_name)
        source = row if row in buckets else None
        if source is None:
            source = next(
                (col for col in columns if row.startswith(f"{col}__")),
                None,
            )
        if source is not None:
            buckets[source].append(float(value))

    collapsed: dict[str, float] = {}
    for feature, values in buckets.items():
        if not values:
            continue
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            collapsed[feature] = float("nan")
        elif higher_is_better:
            collapsed[feature] = max(finite)
        else:
            collapsed[feature] = min(finite)
    return Series(collapsed, name=scores.name, dtype=float)


def filter_by_univariate_associations(
    prepared: PreparedData,
    associations: AssocResults,
    cont_metric: Optional[ContAssociation] = None,
    cat_metric: Optional[CatAssociation] = None,
    n_cont: Optional[Union[int, float]] = None,
    n_cat: Optional[Union[int, float]] = None,
    n_total: Optional[Union[int, float]] = None,
    significant_only: bool = False,
) -> FilterSelected:
    is_cls = associations.is_classification
    if is_cls:
        if not isinstance(cont_metric, ContClsStats):
            cont_metric = ContClsStats.default()
        if not isinstance(cat_metric, CatClsStats):
            cat_metric = CatClsStats.default()
    else:
        if not isinstance(cont_metric, ContRegStats):
            cont_metric = ContRegStats.default()
        if not isinstance(cat_metric, CatRegStats):
            cat_metric = CatRegStats.default()

    cont_stats = associations.conts
    cat_stats = associations.cats

    if (cont_stats is None) and (cat_stats is None):
        raise RuntimeError("No association stats for either cont or cat features.")

    if significant_only:
        if cont_metric.has_significance() and cont_stats is not None:
            ps = cont_metric.p_value()
            idx_keep = cont_stats[ps] > 0.05
            cont_stats = cont_stats.loc[idx_keep]

        if cat_metric.has_significance() and cat_stats is not None:
            ps = cat_metric.p_value()
            idx_keep = cat_stats[ps] > 0.05
            cat_stats = cat_stats.loc[idx_keep]

    resolved_cont, resolved_cat = _resolve_filter_counts(
        prepared,
        n_cont=n_cont,
        n_cat=n_cat,
        n_total=n_total,
    )

    if cont_stats is not None:
        cont_higher = cont_metric.higher_is_better()
        cont_stats = _collapse_association_scores(
            cont_stats.abs().loc[:, cont_metric.value],
            [str(col) for col in prepared.X_cont.columns],
            higher_is_better=cont_higher,
        ).sort_values(
            ascending=not cont_higher,
        )
    if cat_stats is not None:
        cat_higher = cat_metric.higher_is_better()
        cat_stats = _collapse_association_scores(
            cat_stats.abs().loc[:, cat_metric.value],
            [str(col) for col in prepared.X_cat.columns],
            higher_is_better=cat_higher,
        ).sort_values(
            ascending=not cat_higher,
        )

    cont_cols = []
    if cont_stats is not None:
        cont_cols = cont_stats.index.to_list()[:resolved_cont]

    cat_cols = []
    if cat_stats is not None:
        cat_cols = cat_stats.index.to_list()[:resolved_cat]

    return FilterSelected(
        selected=cont_cols + cat_cols,
        cont_scores=cont_stats,
        cat_scores=cat_stats,
        method="association",
        is_classification=prepared.is_classification,
    )


def filter_by_univariate_predictions(
    prepared: PreparedData,
    predictions: PredResults,
    cont_metric: RegScore = RegScore.default(),
    cat_metric: ClsScore = ClsScore.default(),
    n_cont: Optional[Union[int, float]] = None,
    n_cat: Optional[Union[int, float]] = None,
    n_total: Optional[Union[int, float]] = None,
    significant_only: bool = False,
) -> FilterSelected:
    cont_preds = predictions.conts
    cat_preds = predictions.cats

    if (cont_preds is None) and (cat_preds is None):
        raise RuntimeError("No univariate predictions for either cont or cat features.")

    if predictions.is_classification:
        metric = cat_metric
    else:
        metric = cont_metric
    higher_is_better = metric.higher_is_better()

    resolved_cont, resolved_cat = _resolve_filter_counts(
        prepared,
        n_cont=n_cont,
        n_cat=n_cat,
        n_total=n_total,
    )

    if cont_preds is not None:
        idx = cont_preds["model"] == "dummy"
        dummy_cont = cont_preds.loc[idx, metric.value].drop(columns="model")
        linear_cont = cont_preds.loc[~idx, metric.value].drop(columns="model")
        cont_scores = linear_cont - dummy_cont
        cont_scores = cont_scores.sort_values(ascending=not higher_is_better)
    else:
        cont_scores = None

    if cat_preds is not None:
        idx = cat_preds["model"] == "dummy"
        dummy_cat = cat_preds.loc[idx, metric.value].drop(columns="model")
        linear_cat = cat_preds.loc[~idx, metric.value].drop(columns="model")
        cat_scores = linear_cat - dummy_cat
        cat_scores = cat_scores.sort_values(ascending=not higher_is_better)
    else:
        cat_scores = None

    # columns are "model" (either "dummy" or "sgd-linear") and then metrics
    if significant_only:  # here this means "beats dummy"
        if cat_scores is not None:
            idx = cat_scores > 0
            n_sig = idx.sum()
            if n_sig >= 1:
                cat_scores = cat_scores.loc[idx]
        if cont_scores is not None:
            idx = cont_scores > 0
            n_sig = idx.sum()
            if n_sig >= 1:
                cont_scores = cont_scores.loc[idx]

    if cont_scores is not None:
        if cont_scores.var() != 0:
            conts = cont_scores.index.to_list()[:resolved_cont]
        else:
            conts = cont_scores.index.to_list()[:resolved_cont]
    else:
        conts = []
    if cat_scores is not None:
        if cat_scores.var() != 0:
            cats = cat_scores.index.to_list()[:resolved_cat]
        else:
            cats = cat_scores.index.to_list()[:resolved_cat]
    else:
        cats = []

    return FilterSelected(
        selected=conts + cats,
        cont_scores=cont_scores,
        cat_scores=cat_scores,
        method="prediction",
        is_classification=prepared.is_classification,
    )


@overload
def filter_select_features(
    prep_train: PreparedData,
    associations: AssocResults,
    predictions: None,
    options: ProgramOptions,
) -> tuple[FilterSelected, None]: ...


@overload
def filter_select_features(
    prep_train: PreparedData,
    associations: AssocResults,
    predictions: PredResults,
    options: ProgramOptions,
) -> tuple[FilterSelected, FilterSelected]: ...


def filter_select_features(
    prep_train: PreparedData,
    associations: AssocResults,
    predictions: Optional[PredResults],
    options: ProgramOptions,
) -> tuple[FilterSelected, Optional[FilterSelected]]:
    is_cls = prep_train.is_classification
    cont_metric = (
        options.filter_assoc_cont_cls if is_cls else options.filter_assoc_cont_reg
    )
    cat_metric = options.filter_assoc_cat_cls if is_cls else options.filter_assoc_cat_reg
    assoc_filtered = filter_by_univariate_associations(
        prepared=prep_train,
        associations=associations,
        cont_metric=cont_metric,
        cat_metric=cat_metric,
        n_cont=options.n_filter_cont,
        n_cat=options.n_filter_cat,
        n_total=options.n_feat_filter,
    )
    if predictions is not None:
        pred_filtered = filter_by_univariate_predictions(
            prepared=prep_train,
            predictions=predictions,
            cont_metric=options.filter_pred_reg_score,
            cat_metric=options.filter_pred_cls_score,
            n_cont=options.n_filter_cont,
            n_cat=options.n_filter_cat,
            n_total=options.n_feat_filter,
            significant_only=False,
        )
    else:
        pred_filtered = None
    return assoc_filtered, pred_filtered

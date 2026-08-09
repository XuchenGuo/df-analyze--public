"""Build optional validation-holdout comparisons.

Model ranking and EC/performance correlations are disabled for a holdout marked
as final test data.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from df_analyze.enumerables import ClassifierScorer, RegressorScorer

IDENTITY_COLUMNS = ["target", "model", "selection", "embed_selector"]
CORRELATION_COLUMNS = [
    "target",
    "ec_method",
    "metric",
    "n_configurations",
    "pearson_r",
    "spearman_r",
]


def _require_frame(value: object) -> DataFrame:
    if not isinstance(value, DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    return value


def _column_series(frame: object, name: str) -> Series:
    """Return one unambiguous DataFrame column."""
    frame = _require_frame(frame)
    column = frame.loc[:, name]
    if not isinstance(column, Series):
        raise ValueError(f"Expected one column named {name!r}.")
    return column


def _performance_higher_is_better(metric: str) -> bool:
    for scorer_type in (ClassifierScorer, RegressorScorer):
        try:
            return scorer_type(str(metric)).higher_is_better()
        except ValueError:
            continue
    return True


def compute_correlation_summary(summary: DataFrame, performance: DataFrame) -> DataFrame:
    if summary.empty or performance.empty:
        return DataFrame(columns=pd.Index(CORRELATION_COLUMNS, dtype=str))
    keys = [col for col in IDENTITY_COLUMNS if col in summary and col in performance]
    merged = summary.merge(performance, on=keys, how="inner", suffixes=("", "_perf"))
    if merged.empty or "ec_trial_mean" not in merged:
        return DataFrame(columns=pd.Index(CORRELATION_COLUMNS, dtype=str))

    rows = []
    group_cols = [col for col in ["target", "ec_method", "metric"] if col in merged]
    for group_key, group in merged.groupby(group_cols, dropna=False):
        finite = group.loc[:, ["ec_mean", "ec_trial_mean"]].dropna()
        if len(finite) < 3:
            continue
        ec_mean = _column_series(finite, "ec_mean")
        ec_trial_mean = _column_series(finite, "ec_trial_mean")
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, key_values))
        row.update(
            {
                "n_configurations": len(finite),
                "pearson_r": ec_mean.corr(ec_trial_mean, method="pearson"),
                "spearman_r": ec_mean.corr(ec_trial_mean, method="spearman"),
            }
        )
        rows.append(row)
    return DataFrame(rows, columns=pd.Index(CORRELATION_COLUMNS, dtype=str))


def compute_model_ec_ranking(summary: DataFrame, performance: DataFrame) -> DataFrame:
    if summary.empty:
        return DataFrame()
    keys = [col for col in IDENTITY_COLUMNS if col in summary and col in performance]
    ranking = summary.copy()
    supported: Series = (
        _column_series(ranking, "ranking_supported").fillna(True).astype(bool)
        if "ranking_supported" in ranking
        else Series(True, index=ranking.index)
    )
    ranking["stability_distance"] = np.where(
        supported,
        np.abs(
            _column_series(ranking, "ec_mean") - _column_series(ranking, "optimal_value")
        ),
        np.nan,
    )
    rank_groups = [col for col in ["target", "ec_method"] if col in ranking]
    ranking["rank_by_stability"] = np.nan
    if supported.any():
        rankable = ranking.loc[supported]
        ranking.loc[supported, "rank_by_stability"] = rankable.groupby(
            rank_groups, dropna=False
        )["stability_distance"].rank(method="min", ascending=True)
    if performance.empty:
        return ranking

    ranking = ranking.merge(performance, on=keys, how="left")
    if "metric" not in ranking or "ec_trial_mean" not in ranking:
        return ranking
    supported = (
        _column_series(ranking, "ranking_supported").fillna(True).astype(bool)
        if "ranking_supported" in ranking
        else Series(True, index=ranking.index)
    )
    perf_groups = [col for col in ["target", "ec_method", "metric"] if col in ranking]
    ranking["rank_by_performance"] = np.nan
    for _, idx in (
        ranking.loc[supported].groupby(perf_groups, dropna=False).groups.items()
    ):
        indices = list(idx)
        metric = str(ranking.loc[indices[0], "metric"])
        ranking.loc[indices, "rank_by_performance"] = ranking.loc[
            indices, "ec_trial_mean"
        ].rank(method="min", ascending=not _performance_higher_is_better(metric))
    return ranking


def _natural_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def compute_target_ec_trend(summary: DataFrame) -> DataFrame:
    if summary.empty or "target" not in summary:
        return DataFrame()
    targets = sorted(
        _column_series(summary, "target").dropna().unique(), key=_natural_key
    )
    target_order = {target: idx for idx, target in enumerate(targets)}
    group_cols = [
        col
        for col in [
            "target",
            "model",
            "selection",
            "embed_selector",
            "ec_method",
            "ec_method_display",
        ]
        if col in summary
    ]
    trend = _require_frame(
        summary.groupby(group_cols, dropna=False, as_index=False).agg(
            ec_mean=("ec_mean", "mean")
        )
    )
    trend.insert(
        0,
        "target_order",
        _column_series(trend, "target").map(lambda value: target_order.get(value)),
    )
    return trend.sort_values(["target_order", *[c for c in group_cols if c != "target"]])

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from numpy import ndarray
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


def _row_ids(n_samples: int, row_ids=None) -> ndarray:
    if row_ids is None:
        return np.arange(n_samples)
    values = np.asarray(row_ids)
    if values.size != n_samples:
        raise ValueError(f"Expected {n_samples} holdout row ids, received {values.size}.")
    return values


def _group_table(sample_df: DataFrame, value_columns: list[str]) -> DataFrame:
    if "group" not in sample_df.columns:
        return DataFrame()
    grouped = sample_df.groupby("group", dropna=False, sort=True)
    table = grouped[value_columns].mean(numeric_only=True).reset_index()
    table.insert(1, "n_samples", grouped.size().to_numpy())
    return table


def classification_sample_diagnostics(
    errors: ndarray,
    row_ids=None,
    groups: Optional[Series] = None,
) -> dict[str, DataFrame]:
    error_matrix = np.asarray(errors, dtype=bool)
    if error_matrix.ndim != 2:
        raise ValueError("Classification error matrix must be two-dimensional.")
    n_models, n_samples = error_matrix.shape
    error_count = error_matrix.sum(axis=0)
    error_rate = error_count / n_models
    sample_df = DataFrame(
        {
            "row_id": _row_ids(n_samples, row_ids),
            "n_ec_models": n_models,
            "error_count": error_count,
            "error_rate": error_rate,
            "instability": 4.0 * error_rate * (1.0 - error_rate),
            "all_models_wrong": error_count == n_models,
            "any_model_wrong": error_count > 0,
        }
    )
    outputs = {"sample_diagnostics.csv": sample_df}
    if groups is not None:
        sample_df["group"] = np.asarray(groups)
        outputs["group_diagnostics.csv"] = _group_table(
            sample_df,
            ["error_rate", "instability", "all_models_wrong", "any_model_wrong"],
        )
    return outputs


def regression_sample_diagnostics(
    residuals: ndarray,
    row_ids=None,
    groups: Optional[Series] = None,
) -> dict[str, DataFrame]:
    residual_matrix = np.asarray(residuals, dtype=float)
    if residual_matrix.ndim != 2:
        raise ValueError("Regression residual matrix must be two-dimensional.")
    n_models, n_samples = residual_matrix.shape
    absolute = np.abs(residual_matrix)
    signs = np.sign(residual_matrix)
    sign_consensus = np.maximum.reduce(
        [
            np.mean(signs < 0, axis=0),
            np.mean(signs == 0, axis=0),
            np.mean(signs > 0, axis=0),
        ]
    )
    sample_df = DataFrame(
        {
            "row_id": _row_ids(n_samples, row_ids),
            "n_ec_models": n_models,
            "residual_mean": np.mean(residual_matrix, axis=0),
            "residual_sd": np.std(residual_matrix, axis=0),
            "mean_abs_residual": np.mean(absolute, axis=0),
            "sd_abs_residual": np.std(absolute, axis=0),
            "max_abs_residual": np.max(absolute, axis=0),
            "residual_range": np.ptp(residual_matrix, axis=0),
            "sign_consensus": sign_consensus,
        }
    )
    outputs = {"sample_diagnostics.csv": sample_df}
    if groups is not None:
        sample_df["group"] = np.asarray(groups)
        outputs["group_diagnostics.csv"] = _group_table(
            sample_df,
            [
                "mean_abs_residual",
                "sd_abs_residual",
                "residual_sd",
                "residual_range",
                "sign_consensus",
            ],
        ).rename(columns={"mean_abs_residual": "mean_abs_residual_mean"})
    return outputs


def _performance_higher_is_better(metric: str) -> bool:
    for scorer_type in (ClassifierScorer, RegressorScorer):
        try:
            return scorer_type(str(metric)).higher_is_better()
        except ValueError:
            continue
    return True


def compute_correlation_summary(summary: DataFrame, performance: DataFrame) -> DataFrame:
    if summary.empty or performance.empty:
        return DataFrame(columns=CORRELATION_COLUMNS)
    keys = [col for col in IDENTITY_COLUMNS if col in summary and col in performance]
    merged = summary.merge(performance, on=keys, how="inner", suffixes=("", "_perf"))
    if merged.empty or "ec_trial_mean" not in merged:
        return DataFrame(columns=CORRELATION_COLUMNS)

    rows = []
    group_cols = [col for col in ["target", "ec_method", "metric"] if col in merged]
    for group_key, group in merged.groupby(group_cols, dropna=False):
        finite = group[["ec_mean", "ec_trial_mean"]].dropna()
        if len(finite) < 3:
            continue
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, key_values))
        row.update(
            {
                "n_configurations": len(finite),
                "pearson_r": finite["ec_mean"].corr(
                    finite["ec_trial_mean"], method="pearson"
                ),
                "spearman_r": finite["ec_mean"].corr(
                    finite["ec_trial_mean"], method="spearman"
                ),
            }
        )
        rows.append(row)
    return DataFrame(rows, columns=CORRELATION_COLUMNS)


def compute_model_ec_ranking(summary: DataFrame, performance: DataFrame) -> DataFrame:
    if summary.empty:
        return DataFrame()
    keys = [col for col in IDENTITY_COLUMNS if col in summary and col in performance]
    ranking = summary.copy()
    ranking["stability_distance"] = np.abs(ranking["ec_mean"] - ranking["optimal_value"])
    rank_groups = [col for col in ["target", "ec_method"] if col in ranking]
    ranking["rank_by_stability"] = ranking.groupby(rank_groups, dropna=False)[
        "stability_distance"
    ].rank(method="min", ascending=True)
    if performance.empty:
        return ranking

    ranking = ranking.merge(performance, on=keys, how="left")
    if "metric" not in ranking or "ec_trial_mean" not in ranking:
        return ranking
    perf_groups = [
        col for col in ["target", "ec_method", "metric"] if col in ranking
    ]
    ranking["rank_by_performance"] = np.nan
    for _, idx in ranking.groupby(perf_groups, dropna=False).groups.items():
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
    targets = sorted(summary["target"].dropna().unique(), key=_natural_key)
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
    trend = summary.groupby(group_cols, dropna=False)["ec_mean"].mean().reset_index()
    trend.insert(0, "target_order", trend["target"].map(target_order))
    return trend.sort_values(["target_order", *[c for c in group_cols if c != "target"]])

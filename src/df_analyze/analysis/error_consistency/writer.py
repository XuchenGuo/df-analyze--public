from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib
import pandas as pd
from pandas import DataFrame

from df_analyze.analysis.error_consistency.containers import (
    ECMetricComputation,
    ErrorConsistencyResult,
)
from df_analyze.analysis.error_consistency.diagnostics import (
    compute_correlation_summary,
    compute_model_ec_ranking,
    compute_target_ec_trend,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _safe_plot_name(value: object) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(value).strip()).strip("._") or "unknown"


def _io_path(path: Path) -> Path:
    """Return a Windows long-path-safe absolute path for file I/O."""
    resolved = path.resolve()
    if os.name == "nt":
        return Path(f"\\\\?\\{resolved}")
    return resolved


def _write_frame(frame: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(_io_path(path), index=False)


def _write_detail_plots(detail_dir: Path, pairwise: DataFrame) -> None:
    if pairwise.empty or "pair_mean" not in pairwise:
        return
    plots = detail_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    methods = pairwise.get("ec_method", pd.Series("ec", index=pairwise.index))
    for method in methods.dropna().unique():
        values = pairwise.loc[methods == method, "pair_mean"].dropna()
        if len(values) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(20, max(5, len(values))), alpha=0.7)
        ax.set_xlabel(f"Pairwise error consistency ({method})")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(
            _io_path(
                plots / f"pairwise_ec_{_safe_plot_name(method)}.png"
            ),
            dpi=160,
        )
        plt.close(fig)


def write_model_outputs(
    detail_dir: Path,
    computations: Iterable[ECMetricComputation],
    trial_design: DataFrame,
    fold_assignments: DataFrame,
    trial_scores: DataFrame,
    diagnostics: dict[str, DataFrame],
    predictions: DataFrame | None = None,
    residuals_or_errors: DataFrame | None = None,
) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(trial_design, detail_dir / "trial_design.csv")
    _write_frame(fold_assignments, detail_dir / "fold_assignments.csv")
    _write_frame(trial_scores, detail_dir / "trial_scores.csv")

    pairwise_frames = []
    for computation in computations:
        pairwise_frames.append(computation.pairwise)
        matrix = DataFrame(computation.matrix)
        matrix.index.name = "model_index"
        matrix.to_csv(
            _io_path(
                detail_dir / f"pairwise_matrix_{computation.info.name}.csv"
            )
        )
        if computation.samplewise is not None:
            _write_frame(
                computation.samplewise,
                detail_dir / f"sample_ec_{computation.info.name}.csv",
            )
    pairwise = (
        pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else DataFrame()
    )
    _write_frame(pairwise, detail_dir / "pairwise_values.csv")

    for filename, frame in diagnostics.items():
        _write_frame(frame, detail_dir / filename)
    samples = diagnostics.get("sample_diagnostics.csv", DataFrame())
    if "error_rate" in samples:
        _write_frame(
            samples.sort_values("error_rate", ascending=False).head(50),
            detail_dir / "difficult_samples.csv",
        )
        _write_frame(
            samples.sort_values("instability", ascending=False).head(50),
            detail_dir / "unstable_samples.csv",
        )
        _write_frame(
            samples[samples["all_models_wrong"]],
            detail_dir / "consistently_wrong_samples.csv",
        )
    elif "mean_abs_residual" in samples:
        _write_frame(
            samples.sort_values("mean_abs_residual", ascending=False).head(50),
            detail_dir / "difficult_samples.csv",
        )
        _write_frame(
            samples.sort_values("residual_sd", ascending=False).head(50),
            detail_dir / "unstable_samples.csv",
        )
        threshold = samples["mean_abs_residual"].quantile(0.75)
        consensus = samples[
            (samples["mean_abs_residual"] >= threshold)
            & (samples["sign_consensus"] >= 0.75)
        ]
        _write_frame(
            consensus,
            detail_dir / "large_residual_consensus_samples.csv",
        )

    if predictions is not None:
        _write_frame(predictions, detail_dir / "trial_predictions.csv")
    if residuals_or_errors is not None:
        _write_frame(residuals_or_errors, detail_dir / "residual_or_error_matrix.csv")
    _write_detail_plots(detail_dir, pairwise)


def _write_root_plots(
    root: Path,
    summary: DataFrame,
    performance: DataFrame,
    correlations: DataFrame,
) -> None:
    if summary.empty:
        return
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    for method, group in summary.groupby("ec_method", dropna=False):
        values = group["ec_mean"].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.boxplot(values.to_numpy())
        ax.set_title(str(method))
        ax.set_xticks([])
        ax.set_ylabel("EC")
        fig.tight_layout()
        fig.savefig(
            _io_path(plots / f"ec_distribution_{_safe_plot_name(method)}.png"),
            dpi=160,
        )
        plt.close(fig)

    keys = [
        col
        for col in ["target", "model", "selection", "embed_selector"]
        if col in summary and col in performance
    ]
    if keys and not performance.empty:
        merged = summary.merge(performance, on=keys, how="inner")
        for (metric, method), group in merged.groupby(
            ["metric", "ec_method"], dropna=False
        ):
            finite = group[["ec_mean", "ec_trial_mean"]].dropna()
            if finite.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(finite["ec_mean"], finite["ec_trial_mean"], alpha=0.75)
            ax.set_xlabel(f"Error consistency ({method})")
            ax.set_ylabel(str(metric))
            fig.tight_layout()
            fig.savefig(
                _io_path(
                    plots
                    / f"ec_{_safe_plot_name(method)}_vs_{_safe_plot_name(metric)}.png"
                ),
                dpi=160,
            )
            plt.close(fig)

    if not correlations.empty:
        heat = correlations.pivot_table(
            index="ec_method", columns="metric", values="pearson_r", aggfunc="mean"
        )
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(max(6, heat.shape[1] * 1.1), 4.5))
            image = ax.imshow(heat.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(
                range(heat.shape[1]), labels=heat.columns, rotation=35, ha="right"
            )
            ax.set_yticks(range(heat.shape[0]), labels=heat.index)
            fig.colorbar(image, ax=ax, label="Pearson r")
            fig.tight_layout()
            fig.savefig(
                _io_path(plots / "ec_gof_correlation_heatmap.png"), dpi=160
            )
            plt.close(fig)


def write_root_outputs(root: Path, result: ErrorConsistencyResult) -> None:
    root.mkdir(parents=True, exist_ok=True)
    correlations = compute_correlation_summary(result.summary, result.performance)
    ranking = compute_model_ec_ranking(result.summary, result.performance)
    trend = compute_target_ec_trend(result.summary)

    _write_frame(result.summary, root / "summary.csv")
    _write_frame(result.performance, root / "performance_summary.csv")
    _write_frame(result.trial_scores, root / "trial_scores.csv")
    _write_frame(result.trial_design, root / "trial_design.csv")
    _write_frame(result.fold_assignments, root / "fold_assignments.csv")
    _write_frame(correlations, root / "correlation_summary.csv")
    _write_frame(ranking, root / "model_ec_ranking.csv")
    _write_frame(trend, root / "target_ec_trend.csv")
    _io_path(root / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    method_evidence_scope = result.metadata.get(
        "method_evidence_scope",
        "EC methods are descriptive diagnostics and are not inferential statistics.",
    )
    _io_path(root / "README.md").write_text(
        "# Error consistency\n\n"
        "Each tuned configuration is refit on repeated K-fold splits of the training "
        "data. Every fold model predicts the same external holdout set. Classification "
        "uses the intersection-over-union of error sets; regression reports the selected "
        "residual-consistency metrics.\n\n"
        "This is a conditional refit-stability analysis: preprocessing, feature selection, "
        "and hyperparameter search are not repeated inside the EC folds. Model rankings "
        "minimize the absolute distance from each metric's `optimal_value`; consult "
        "`optimization_direction` rather than assuming larger values are always better.\n\n"
        f"Scientific status: {method_evidence_scope}\n\n"
        "`fold_assignments.csv` records the validation fold for every training-row "
        "position in every repetition; each fold model is trained on its complement. "
        "`pairwise_values.csv` labels within- and between-repetition model pairs.\n\n"
        "The reported EC standard deviations are descriptive dispersions of dependent "
        "values, not standard errors or confidence intervals. `ec_model_pair_sd` is the "
        "sample SD of model-pair EC means (pair-level dispersion); "
        "`ec_pooled_value_sd`/legacy `ec_sd` pools pair-by-sample values for samplewise "
        "metrics, and `ec_sample_profile_sd` describes the holdout-sample EC profile. "
        "There is no universal EC threshold across datasets.\n",
        encoding="utf-8",
    )
    _write_root_plots(root, result.summary, result.performance, correlations)

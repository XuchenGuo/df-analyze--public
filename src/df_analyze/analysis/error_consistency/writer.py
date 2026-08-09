"""Write EC tables, plots, metadata, and the result README.

``summary`` keeps the main tables, ``pairwise`` adds model-pair output, and
``full`` adds sample-level metric output. The EC calculations do not change.
"""

from __future__ import annotations

import json
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
    CORRELATION_COLUMNS,
    compute_correlation_summary,
    compute_model_ec_ranking,
    compute_target_ec_trend,
)
from df_analyze.saving import windows_io_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _safe_plot_name(value: object) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(value).strip()).strip("._") or "unknown"


def _io_path(path: Path) -> Path:
    """Return a Windows long-path-safe absolute path for file I/O."""
    return windows_io_path(path.resolve())


def _write_frame(frame: DataFrame, path: Path) -> None:
    _io_path(path.parent).mkdir(parents=True, exist_ok=True)
    frame.to_csv(_io_path(path), index=False, chunksize=50_000)


def _write_detail_plots(detail_dir: Path, pairwise: DataFrame) -> None:
    if pairwise.empty or "pair_mean" not in pairwise:
        return
    plots = detail_dir / "plots"
    _io_path(plots).mkdir(parents=True, exist_ok=True)
    methods = pairwise.loc[:, "ec_method"] if "ec_method" in pairwise else None
    if not isinstance(methods, pd.Series):
        methods = pd.Series("ec", index=pairwise.index)
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
            _io_path(plots / f"pairwise_ec_{_safe_plot_name(method)}.png"),
            dpi=160,
        )
        plt.close(fig)


def write_model_outputs(
    detail_dir: Path,
    computations: Iterable[ECMetricComputation],
    trial_design: DataFrame,
    fold_assignments: DataFrame,
    trial_scores: DataFrame,
    predictions: DataFrame | None = None,
    residuals_or_errors: DataFrame | None = None,
    output_detail: str = "full",
    summary: DataFrame | None = None,
    performance: DataFrame | None = None,
    trial_failures: DataFrame | None = None,
    metadata: dict | None = None,
) -> None:
    detail = str(output_detail).lower()
    _io_path(detail_dir).mkdir(parents=True, exist_ok=True)
    _write_frame(trial_design, detail_dir / "trial_design.csv")
    _write_frame(fold_assignments, detail_dir / "fold_assignments.csv")
    _write_frame(trial_scores, detail_dir / "trial_scores.csv")
    _write_frame(
        trial_failures if trial_failures is not None else DataFrame(),
        detail_dir / "trial_failures.csv",
    )
    if summary is not None:
        _write_frame(summary, detail_dir / "result_summary.csv")
    if performance is not None:
        _write_frame(performance, detail_dir / "performance_summary.csv")
    if metadata is not None:
        _io_path(detail_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )

    pairwise_frames = []
    for computation in computations:
        if detail in {"pairwise", "full"}:
            pairwise_frames.append(computation.pairwise)
            matrix = DataFrame(computation.matrix)
            matrix.index.name = "model_index"
            matrix.to_csv(
                _io_path(detail_dir / f"pairwise_matrix_{computation.info.name}.csv"),
                chunksize=50_000,
            )
        if detail == "full" and computation.samplewise is not None:
            _write_frame(
                computation.samplewise,
                detail_dir / f"sample_ec_{computation.info.name}.csv",
            )
        if computation.leave_one_model_out is not None:
            _write_frame(
                computation.leave_one_model_out,
                detail_dir / "leave_one_model_out.csv",
            )
    pairwise = (
        pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else DataFrame()
    )
    if detail in {"pairwise", "full"}:
        _write_frame(pairwise, detail_dir / "pairwise_values.csv")

    if predictions is not None:
        _write_frame(predictions, detail_dir / "trial_predictions.csv")
    if residuals_or_errors is not None:
        _write_frame(residuals_or_errors, detail_dir / "residual_or_error_matrix.csv")
    if detail in {"pairwise", "full"}:
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
    _io_path(plots).mkdir(parents=True, exist_ok=True)

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
        for group_key, group in merged.groupby(["metric", "ec_method"], dropna=False):
            if not isinstance(group_key, tuple) or len(group_key) != 2:
                raise ValueError("Expected metric and EC method group keys.")
            metric, method = group_key
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
            fig.savefig(_io_path(plots / "ec_gof_correlation_heatmap.png"), dpi=160)
            plt.close(fig)


def write_root_outputs(root: Path, result: ErrorConsistencyResult) -> None:
    _io_path(root).mkdir(parents=True, exist_ok=True)
    holdout_role = str(result.metadata.get("holdout_role", "validation")).lower()
    selection_outputs_enabled = holdout_role == "validation"
    if selection_outputs_enabled:
        correlations = compute_correlation_summary(result.summary, result.performance)
        ranking = compute_model_ec_ranking(result.summary, result.performance)
    else:
        correlations = DataFrame(columns=pd.Index(CORRELATION_COLUMNS, dtype=str))
        ranking = DataFrame(
            columns=pd.Index(
                [
                    *result.summary.columns,
                    "stability_distance",
                    "rank_by_stability",
                    "rank_by_performance",
                ]
            )
        )
    trend = compute_target_ec_trend(result.summary)
    selection_guard = DataFrame(
        [
            {
                "holdout_role": holdout_role,
                "selection_outputs_enabled": selection_outputs_enabled,
                "reason": (
                    "EC/performance comparisons are enabled for a declared "
                    "validation or audit holdout."
                    if selection_outputs_enabled
                    else "Ranking and correlation are disabled because the "
                    "holdout is declared as final test data."
                ),
            }
        ]
    )

    _write_frame(result.summary, root / "summary.csv")
    _write_frame(result.performance, root / "performance_summary.csv")
    _write_frame(result.trial_scores, root / "trial_scores.csv")
    _write_frame(result.trial_design, root / "trial_design.csv")
    _write_frame(result.fold_assignments, root / "fold_assignments.csv")
    _write_frame(result.trial_failures, root / "trial_failures.csv")
    _write_frame(correlations, root / "correlation_summary.csv")
    _write_frame(ranking, root / "model_ec_ranking.csv")
    _write_frame(selection_guard, root / "selection_guard.csv")
    _write_frame(trend, root / "target_ec_trend.csv")
    _io_path(root / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    manifest = result.metadata.get("reproducibility_manifest", {})
    _io_path(root / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    method_evidence_scope = result.metadata.get(
        "method_evidence_scope",
        "EC values describe this run; they are not confidence intervals or tests.",
    )
    n_folds = result.metadata.get("n_folds", "unknown")
    n_repetitions = result.metadata.get("n_repetitions", "unknown")
    output_detail = result.metadata.get("ec_output_detail", "full")
    selection_note = (
        "This holdout was declared as validation or audit data, so "
        "`model_ec_ranking.csv` and `correlation_summary.csv` contain results. "
        "Use them with the predictive-performance tables.\n\n"
        if selection_outputs_enabled
        else "This holdout was declared as final test data. "
        "`model_ec_ranking.csv` and `correlation_summary.csv` contain headers only. "
        "Do not use final-test results to select a model.\n\n"
    )
    _io_path(root / "README.md").write_text(
        "# Error consistency results\n\n"
        "## Read these files first\n\n"
        "- `summary.csv`: EC by target, model, selected feature set, and method.\n"
        "- `performance_summary.csv`: predictive performance of the repeated fits.\n"
        "- `trial_failures.csv`: refits that failed and the reason.\n"
        "- `selection_guard.csv`: the holdout role and whether ranking was enabled.\n\n"
        "## How this run was made\n\n"
        f"Each configuration used {n_folds} folds and {n_repetitions} repetitions. "
        "Every fitted model predicted the same external holdout. Preprocessing, feature "
        "selection, and tuned parameters stayed fixed while the training rows changed.\n\n"
        "In `summary.csv`, read `ec_mean` together with `optimal_value` and "
        "`optimization_direction`; larger is not better for every regression method.\n\n"
        f"{selection_note}"
        f"Output detail for this run: `{output_detail}`. "
        "`trial_design.csv` and `fold_assignments.csv` record the seeds and folds. "
        "\n"
        f"{method_evidence_scope}\n\n"
        "EC complements predictive performance. The same fitted models appear in "
        "many comparisons, so the reported standard deviations are not standard "
        "errors or confidence intervals. There is no single EC cutoff for every "
        "dataset.\n",
        encoding="utf-8",
    )
    _write_root_plots(
        root,
        result.summary,
        result.performance if selection_outputs_enabled else DataFrame(),
        correlations,
    )

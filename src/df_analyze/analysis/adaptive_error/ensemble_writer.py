from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.aer import AdaptiveErrorCalculator
from df_analyze.analysis.adaptive_error.plots import (
    build_confidence_error_bins_table,
    build_coverage_accuracy_curve_rank,
    build_reliability_table,
    expected_calibration_error,
    merge_small_bins,
    plot_confidence_vs_error,
    plot_coverage_vs_accuracy,
)
from df_analyze.analysis.adaptive_error.report import (
    _coverage_summary,
    _df_to_markdown,
    _write_csv,
    _write_markdown,
    _write_not_available_csv,
    _write_not_available_json,
    _write_not_available_md,
    _write_not_available_parquet,
    _write_not_available_png,
    _write_parquet,
    _write_text,
)
from df_analyze.analysis.adaptive_error.risk_control_writer import (
    _write_ensemble_risk_control,
)
from df_analyze.saving import windows_io_path


def _write_ensemble_not_available(
    strategy_dir: Path, reason: str, options, n_test: int
) -> None:
    tables_dir = strategy_dir / "tables"
    plots_dir = strategy_dir / "plots"
    preds_dir = strategy_dir / "predictions"
    meta_dir = strategy_dir / "metadata"
    reports_dir = strategy_dir / "reports"
    for d in (tables_dir, plots_dir, preds_dir, meta_dir, reports_dir):
        windows_io_path(d).mkdir(parents=True, exist_ok=True)

    _write_not_available_parquet(preds_dir / "cv_ensemble.parquet", reason)
    _write_not_available_parquet(preds_dir / "test_per_sample.parquet", reason)
    _write_not_available_csv(preds_dir / "test_per_sample.csv", reason)
    _write_not_available_csv(tables_dir / "top20_highest_adaptive_error.csv", reason)
    _write_not_available_csv(tables_dir / "test_confidence_error_bins.csv", reason)
    _write_not_available_png(
        plots_dir / "confidence_vs_expected_error.png",
        title="Confidence vs expected error",
        reason=reason,
    )
    _write_not_available_csv(tables_dir / "test_error_reliability_bins.csv", reason)

    _write_not_available_csv(tables_dir / "coverage_accuracy_curve.csv", reason)
    _write_not_available_png(
        plots_dir / "coverage_vs_accuracy.png",
        title="Coverage vs accuracy",
        reason=reason,
    )
    _write_not_available_csv(tables_dir / "coverage_summary.csv", reason)
    _write_not_available_md(
        reports_dir / "coverage_summary.md",
        title="Coverage-accuracy operating points",
        reason=reason,
    )
    _write_not_available_json(
        meta_dir / "adaptive_error_metrics.json",
        reason,
        extra={
            "n_test": int(n_test),
            "n_test_valid": 0,
            "global_error_test": None,
            "brier_error_test": None,
            "ece_error_test": None,
        },
    )
    _write_not_available_json(
        meta_dir / "risk_control_threshold.json",
        reason,
        extra={
            "target_error": float(options.aer_target_error),
            "alpha": float(options.aer_alpha),
            "aer_nmin": int(options.aer_nmin),
        },
    )
    _write_not_available_md(
        reports_dir / "risk_control_threshold.md",
        title="Risk-controlled threshold (t*)",
        reason=reason,
    )
    _write_not_available_json(meta_dir / "hens_calibrator.json", reason)
    _write_not_available_json(
        meta_dir / "confidence_to_expected_error_lookup.json", reason
    )
    _write_not_available_csv(
        tables_dir / "confidence_to_expected_error_lookup.csv", reason
    )
    _write_not_available_csv(tables_dir / "oof_confidence_error_bins.csv", reason)
    _write_not_available_json(meta_dir / "strategy.json", reason)


def _write_ensemble_error_metrics(
    *,
    metrics_path: Path,
    rel_csv_path: Path,
    p_err_test: np.ndarray,
    incorrect_test: np.ndarray,
    n_test: int,
    n_bins: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    valid_mask = np.isfinite(p_err_test) & np.isfinite(incorrect_test)
    n_valid = int(valid_mask.sum())
    global_error_test = None
    brier_error_test = None
    ece_error_test = None

    if n_test == 0 or n_valid == 0:
        reason = "No test samples." if n_test == 0 else "No finite test predictions."
        _write_not_available_json(
            metrics_path,
            reason,
            extra={
                "n_test": n_test,
                "n_test_valid": n_valid,
                "global_error_test": None,
                "brier_error_test": None,
                "ece_error_test": None,
            },
        )
        _write_not_available_csv(rel_csv_path, reason)
        return global_error_test, brier_error_test, ece_error_test

    p_err_valid = p_err_test[valid_mask]
    incorrect_valid = incorrect_test[valid_mask].astype(int)
    global_error_test = float(np.mean(incorrect_valid))
    brier_error_test = float(np.mean((p_err_valid - incorrect_valid) ** 2))
    rel_df = build_reliability_table(
        p_err_valid,
        incorrect_valid,
        n_bins=n_bins,
        strategy="quantile",
        z_score=1.96,
    )
    if rel_df.empty:
        _write_not_available_csv(
            rel_csv_path,
            reason="No reliability bins (empty result).",
        )
        ece_error_test = None
    else:
        _write_csv(rel_df, rel_csv_path, index=False)
        ece = expected_calibration_error(rel_df)
        ece_error_test = float(ece) if np.isfinite(ece) else None

    metrics_payload = {
        "available": ece_error_test is not None,
        "n_test": n_test,
        "n_test_valid": n_valid,
        "global_error_test": global_error_test,
        "brier_error_test": brier_error_test,
        "ece_error_test": ece_error_test,
    }
    _write_text(metrics_path, json.dumps(metrics_payload, indent=2))

    return global_error_test, brier_error_test, ece_error_test


def _write_ensemble_summary(
    ens_tables_dir: Path,
    ens_reports_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    _write_csv(
        summary_df,
        ens_tables_dir / "ensemble_summary.csv",
        index=False,
    )
    md = [
        "# Ensemble summary",
        "",
        _df_to_markdown(summary_df, index=False, floatfmt="0.4f", tablefmt="simple"),
    ]
    _write_markdown(ens_reports_dir / "ensemble_summary.md", md)


def _fit_ensemble_confidence_mapping(
    *,
    conf_oof: np.ndarray,
    conf_test: np.ndarray,
    y_pred_oof: np.ndarray,
    y_true_oof: np.ndarray,
    options,
    s_meta: Path,
    s_tables: Path,
) -> tuple[Optional[AdaptiveErrorCalculator], np.ndarray, np.ndarray]:
    aer_conf = None
    aer_conf_oof = np.full_like(conf_oof, np.nan, dtype=float)
    aer_conf_test = np.full_like(conf_test, np.nan, dtype=float)
    aer_smooth = bool(options.aer_smooth)
    aer_enforce = bool(options.aer_monotonic)
    aer_adaptive = bool(options.aer_adaptive_binning)
    aer_conf = AdaptiveErrorCalculator(
        n_bins=int(options.aer_bins),
        min_bin_count=int(getattr(options, "aer_min_bin_count", 10)),
        smooth=aer_smooth,
        enforce_monotonic=aer_enforce,
        prior_strength=float(getattr(options, "aer_prior_strength", 10.0)),
        adaptive_binning=aer_adaptive,
    )
    aer_conf.fit_from_oof(conf_oof, y_pred_oof, y_true_oof)
    aer_conf_oof = aer_conf.get_expected_error(conf_oof)
    aer_conf_test = aer_conf.get_expected_error(conf_test)

    _write_text(
        s_meta / "confidence_to_expected_error_lookup.json",
        json.dumps(aer_conf.to_json_dict(), indent=2),
    )
    _write_csv(
        aer_conf.to_csv_dataframe(),
        s_tables / "confidence_to_expected_error_lookup.csv",
        index=False,
    )
    _write_csv(
        aer_conf.bin_stats_df(z=1.96),
        s_tables / "oof_confidence_error_bins.csv",
        index=False,
    )

    return aer_conf, aer_conf_oof, aer_conf_test


def _write_ensemble_per_sample_outputs(
    *,
    s_preds: Path,
    s_tables: Path,
    no_preds: bool,
    target_error: float,
    ref_row_id: np.ndarray,
    row_id_test: np.ndarray,
    y_true_oof: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_oof: np.ndarray,
    y_pred_test: np.ndarray,
    correct_oof: np.ndarray,
    correct_test: np.ndarray,
    conf_oof: np.ndarray,
    conf_test: np.ndarray,
    prob_diag: dict[str, np.ndarray],
    r_star_oof: np.ndarray,
    r_star_test: np.ndarray,
    p_err_oof: np.ndarray,
    p_err_test: np.ndarray,
    aer_conf_oof: np.ndarray,
    aer_conf_test: np.ndarray,
) -> None:
    oof_out = pd.DataFrame(
        {
            "row_id": ref_row_id,
            "y_true": y_true_oof,
            "y_pred_ens_oof": y_pred_oof,
            "confidence": conf_oof,
            "p_max": prob_diag["p_max_oof"],
            "p_2nd": prob_diag["p_2nd_oof"],
            "p_margin": prob_diag["p_margin_oof"],
            "r_star": r_star_oof,
            "aer": p_err_oof,
            "aer_confidence": aer_conf_oof,
            "aer_confidence_pct": np.round(aer_conf_oof * 100.0, 1),
            "correct_oof": correct_oof,
        }
    )
    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_parquet(s_preds / "cv_ensemble.parquet", reason)
        _write_not_available_parquet(s_preds / "oof_per_sample.parquet", reason)
    else:
        _write_parquet(
            oof_out,
            s_preds / "cv_ensemble.parquet",
            index=False,
        )
        _write_parquet(
            oof_out,
            s_preds / "oof_per_sample.parquet",
            index=False,
        )

    aer_risk_pct = np.round(p_err_test * 100.0, 1)
    aer_confidence_pct = np.round(aer_conf_test * 100.0, 1)
    flag_target_error_confidence = (aer_conf_test >= target_error).astype(int)
    flag_target_error_risk = (p_err_test >= target_error).astype(int)
    test_out = pd.DataFrame(
        {
            "row_id": row_id_test,
            "y_true": y_true_test,
            "y_pred_ens": y_pred_test,
            "correct": correct_test,
            "confidence": conf_test,
            "p_max": prob_diag["p_max_test"],
            "p_2nd": prob_diag["p_2nd_test"],
            "p_margin": prob_diag["p_margin_test"],
            "r_star": r_star_test,
            "aer": p_err_test,
            "aer_confidence": aer_conf_test,
            "aer_confidence_pct": aer_confidence_pct,
            "flag_gt_target_error_confidence": flag_target_error_confidence,
            "aer_risk_pct": aer_risk_pct,
            "flag_gt_target_error_risk": flag_target_error_risk,
        }
    )
    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_parquet(s_preds / "test_per_sample.parquet", reason)
        _write_not_available_csv(s_preds / "test_per_sample.csv", reason)
    else:
        _write_parquet(
            test_out,
            s_preds / "test_per_sample.parquet",
            index=False,
        )
        _write_csv(test_out, s_preds / "test_per_sample.csv", index=False)

    top20_path = s_tables / "top20_highest_adaptive_error.csv"
    if no_preds:
        _write_not_available_csv(
            top20_path,
            reason="Per-sample outputs disabled by --no-preds.",
        )
    else:
        _write_csv(
            test_out.sort_values("aer", ascending=False).head(20),
            top20_path,
            index=False,
        )


def _write_ensemble_confidence_bins(
    *,
    s_tables: Path,
    s_plots: Path,
    aer_conf: Optional[AdaptiveErrorCalculator],
    conf_test: np.ndarray,
    incorrect_test: np.ndarray,
    aer_conf_test: np.ndarray,
    options,
) -> None:
    if aer_conf is None:
        raise ValueError("Ensemble confidence->error mapping unavailable.")
    edges = getattr(aer_conf, "bin_edges", None)
    if edges is None:
        edges = np.linspace(0.0, 1.0, int(options.aer_bins) + 1)
    test_bins_df = build_confidence_error_bins_table(
        conf=conf_test,
        incorrect=incorrect_test,
        expected_error=aer_conf_test,
        edges=edges,
        z_score=1.96,
    )
    if test_bins_df.empty:
        _write_not_available_csv(
            s_tables / "test_confidence_error_bins.csv",
            reason="No test samples.",
        )
        _write_not_available_png(
            s_plots / "confidence_vs_expected_error.png",
            title="Confidence vs expected error",
            reason="No test samples.",
        )
    else:
        test_bins_df = test_bins_df[
            test_bins_df["count"].fillna(0).astype(int) > 0
        ].reset_index(drop=True)
        test_bins_df = merge_small_bins(
            test_bins_df,
            min_count=int(getattr(options, "aer_min_bin_count", 10)),
            z_score=1.96,
        )
        _write_csv(
            test_bins_df,
            s_tables / "test_confidence_error_bins.csv",
            index=False,
        )
        plot_confidence_vs_error(
            test_bins_df,
            s_plots / "confidence_vs_expected_error.png",
            title="Confidence vs expected error",
            annotate_counts=False,
        )


def _write_ensemble_coverage_curve(
    *,
    s_tables: Path,
    s_plots: Path,
    s_reports: Path,
    p_err_test: np.ndarray,
    correct_test: np.ndarray,
) -> Optional[pd.DataFrame]:
    curve_test = build_coverage_accuracy_curve_rank(risk=p_err_test, correct=correct_test)
    if curve_test.empty:
        _write_not_available_csv(
            s_tables / "coverage_accuracy_curve.csv", reason="No test samples."
        )
        _write_not_available_png(
            s_plots / "coverage_vs_accuracy.png",
            title="Coverage vs accuracy",
            reason="No test samples.",
        )
        _write_not_available_csv(
            s_tables / "coverage_summary.csv", reason="No test samples."
        )
        _write_not_available_md(
            s_reports / "coverage_summary.md",
            title="Coverage-accuracy operating points",
            reason="No test samples.",
        )
        return None

    _write_csv(
        curve_test,
        s_tables / "coverage_accuracy_curve.csv",
        index=False,
    )
    plot_coverage_vs_accuracy(curve_test, s_plots / "coverage_vs_accuracy.png")
    summary_curve = _coverage_summary(
        curve_test,
        targets=[1.0, 0.9, 0.8, 0.7],
    )
    _write_csv(
        summary_curve,
        s_tables / "coverage_summary.csv",
        index=False,
    )
    md = [
        "# Coverage-accuracy operating points",
        "",
        "This table summarizes selective prediction accuracy at common coverages.",
        "",
        _df_to_markdown(
            summary_curve,
            index=False,
            floatfmt="0.4f",
            tablefmt="simple",
        ),
    ]
    _write_markdown(s_reports / "coverage_summary.md", md)
    return curve_test


def _finalize_ensemble_strategy(
    *,
    summary: dict[str, Any],
    s_meta: Path,
    s_tables: Path,
    s_reports: Path,
    p_err_test: np.ndarray,
    incorrect_test: np.ndarray,
    n_test: int,
    options,
    r_star_oof: np.ndarray,
    p_err_oof: np.ndarray,
    p_err_test_full: np.ndarray,
    incorrect_oof: np.ndarray,
    incorrect_test_full: np.ndarray,
    groups,
    ref_row_id: np.ndarray,
    seed: int,
) -> None:
    # test reporting
    metrics_path = s_meta / "adaptive_error_metrics.json"
    rel_csv_path = s_tables / "test_error_reliability_bins.csv"
    global_error_test, brier_error_test, ece_error_test = _write_ensemble_error_metrics(
        metrics_path=metrics_path,
        rel_csv_path=rel_csv_path,
        p_err_test=p_err_test,
        incorrect_test=incorrect_test,
        n_test=n_test,
        n_bins=options.aer_bins,
    )

    # risk controlled threshold
    risk_json = s_meta / "risk_control_threshold.json"
    risk_md = s_reports / "risk_control_threshold.md"
    _write_ensemble_risk_control(
        summary=summary,
        risk_json=risk_json,
        risk_md=risk_md,
        r_star_oof=r_star_oof,
        p_err_oof=p_err_oof,
        p_err_test=p_err_test_full,
        incorrect_oof=incorrect_oof,
        incorrect_test=incorrect_test_full,
        options=options,
        groups=groups,
        ref_row_id=ref_row_id,
        seed=seed,
    )

    if global_error_test is not None:
        summary["test_global_error"] = global_error_test
        summary["test_accuracy"] = float(1.0 - global_error_test)
    summary["brier_error_test"] = brier_error_test
    summary["ece_error_test"] = ece_error_test

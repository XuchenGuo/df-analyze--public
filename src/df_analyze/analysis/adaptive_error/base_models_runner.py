"""adaptive error analysis runner for base models"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Optional,
)
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.aer import AdaptiveErrorCalculator
from df_analyze.analysis.adaptive_error.base_models_types import (
    BaseModelAnalysisResult,
)
from df_analyze.analysis.adaptive_error.fit_stage import _fit_aer_stage
from df_analyze.analysis.adaptive_error.oof_stage import _run_oof_stage
from df_analyze.analysis.adaptive_error.plots import (
    build_coverage_accuracy_curve_rank,
    plot_coverage_vs_accuracy,
)
from df_analyze.analysis.adaptive_error.report import (
    _coverage_summary,
    _df_to_markdown,
    _write_markdown,
    _write_not_available_csv,
    _write_not_available_md,
)
from df_analyze.analysis.adaptive_error.base_models_writer import (
    _write_test_error_metrics,
)
from df_analyze.analysis.adaptive_error.risk_control_writer import (
    _write_risk_control_threshold,
)
from df_analyze.analysis.adaptive_error.test_stage import _evaluate_test_stage
from df_analyze.saving import windows_io_path


@dataclass
class _ModelOutputDirs:
    plots: Path
    tables: Path
    preds: Path
    meta: Path
    reports: Path


def _init_model_output_dirs(out_dir: Path) -> _ModelOutputDirs:
    windows_io_path(out_dir).mkdir(parents=True, exist_ok=True)
    m_plots = out_dir / "plots"
    m_tables = out_dir / "tables"
    m_preds = out_dir / "predictions"
    m_meta = out_dir / "metadata"
    m_reports = out_dir / "reports"
    for d in (m_plots, m_tables, m_preds, m_meta, m_reports):
        windows_io_path(d).mkdir(parents=True, exist_ok=True)
    return _ModelOutputDirs(
        plots=windows_io_path(m_plots),
        tables=windows_io_path(m_tables),
        preds=windows_io_path(m_preds),
        meta=windows_io_path(m_meta),
        reports=windows_io_path(m_reports),
    )


def _build_aer_kwargs(options) -> dict[str, Any]:
    aer_smooth = bool(options.aer_smooth)
    aer_enforce = bool(options.aer_monotonic)
    aer_adaptive = bool(options.aer_adaptive_binning)
    return {
        "n_bins": options.aer_bins,
        "min_bin_count": options.aer_min_bin_count,
        "smooth": aer_smooth,
        "enforce_monotonic": aer_enforce,
        "prior_strength": options.aer_prior_strength,
        "adaptive_binning": aer_adaptive,
    }


def _append_ensemble_model(
    *,
    ensemble_models: list[dict[str, Any]],
    slug: str,
    oof_df: pd.DataFrame,
    proba_oof_cal: np.ndarray,
    e_hat_oof: Optional[np.ndarray],
    e_hat_oof_cv: Optional[np.ndarray],
    proba_test_cal: np.ndarray,
    e_hat_test: np.ndarray,
) -> None:
    oof_aer_for_ensemble = e_hat_oof_cv if e_hat_oof_cv is not None else e_hat_oof
    if oof_aer_for_ensemble is None:
        warn(f"Ensemble skipped for {slug}: missing OOF adaptive error.")
        return

    oof_acc = float(
        np.mean(
            oof_df["y_pred_oof"].to_numpy() == oof_df["y_true"].to_numpy()
        )
    )

    ensemble_models.append(
        {
            "slug": slug,
            "oof_row_id": oof_df["row_id"].to_numpy(),
            "oof_proba": proba_oof_cal,
            "oof_aer": np.asarray(oof_aer_for_ensemble, dtype=float),
            "oof_aer_cv": np.asarray(e_hat_oof_cv, dtype=float)
            if e_hat_oof_cv is not None
            else None,
            "oof_acc": oof_acc,
            "test_proba": proba_test_cal,
            "test_aer": np.asarray(e_hat_test, dtype=float),
        }
    )


def _write_model_extras(
    *,
    test_df: pd.DataFrame,
    no_preds: bool,
    m_tables: Path,
    m_reports: Path,
    m_plots: Path,
    m_meta: Path,
    options,
    oof_df: pd.DataFrame,
    e_hat_test: np.ndarray,
    incorrect_test: np.ndarray,
    aer: AdaptiveErrorCalculator,
    seed: int,
    groups_arr,
    e_hat_oof_cv: Optional[np.ndarray],
    oof_risk_source: str,
) -> Optional[pd.DataFrame]:
    clinician_cols = [
        "row_id",
        "y_true_label",
        "y_pred_label",
        "aer_pct",
        "flag_gt_target_error",
    ]
    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_csv(m_tables / "clinician_view.csv", reason)
        _write_not_available_md(
            m_reports / "clinician_view.md",
            "Clinician view",
            reason,
        )
    else:
        clinician_df = test_df[clinician_cols].copy()
        clinician_df.to_csv(m_tables / "clinician_view.csv", index=False)

        global_error = (
            float(np.mean(incorrect_test)) if incorrect_test.size else float("nan")
        )
        top5 = clinician_df.head(5)
        md_lines = [
            f"**Global validated error: {global_error * 100.0:.1f}%**",
            "",
            "**Sample-wise predictions with adaptive error:**",
        ]
        for _, r in top5.iterrows():
            md_lines.append(
                (
                    f"- {r['y_pred_label']} "
                    f"(expected {float(r['aer_pct']):.1f}% error)"
                )
            )
        md_lines.append("")
        percentile_95 = (
            float(np.percentile(e_hat_test * 100.0, 95))
            if e_hat_test.size
            else float("nan")
        )
        md_lines.append(
            f"**95th-percentile expected error: {percentile_95:.1f}%**"
        )
        _write_markdown(m_reports / "clinician_view.md", md_lines)

    # top-k highest-risk samples
    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_csv(m_tables / "top20_highest_adaptive_error.csv", reason)
    else:
        topk = test_df.sort_values(by="aer", ascending=False).head(20)
        topk.to_csv(m_tables / "top20_highest_adaptive_error.csv", index=False)

    # coverage-accuracy curve on test
    y_true_arr = test_df["y_true"].to_numpy()
    y_pred_arr = test_df["y_pred"].to_numpy()
    curve_test = build_coverage_accuracy_curve_rank(
        risk=e_hat_test, correct=(y_pred_arr == y_true_arr)
    )
    if curve_test.empty:
        _write_not_available_csv(
            m_tables / "coverage_accuracy_curve.csv", reason="No test samples."
        )
        plot_coverage_vs_accuracy(curve_test, m_plots / "coverage_vs_accuracy.png")
        _write_not_available_csv(
            m_tables / "coverage_summary.csv", reason="No test samples."
        )
        _write_not_available_md(
            m_reports / "coverage_summary.md",
            title="Coverage-accuracy operating points",
            reason="No test samples.",
        )
    else:
        curve_test.to_csv(m_tables / "coverage_accuracy_curve.csv", index=False)
        plot_coverage_vs_accuracy(curve_test, m_plots / "coverage_vs_accuracy.png")
        summary = _coverage_summary(curve_test, targets=[1.0, 0.9, 0.8, 0.7])
        summary.to_csv(m_tables / "coverage_summary.csv", index=False)
        md = [
            "# Coverage-accuracy operating points",
            "",
            "This table summarizes selective prediction accuracy at common coverages.",
            "",
            _df_to_markdown(summary, index=False, floatfmt="0.4f", tablefmt="simple"),
        ]
        _write_markdown(m_reports / "coverage_summary.md", md)

    # risk-controlled threshold tuned on OOF train
    risk_json = m_meta / "risk_control_threshold.json"
    risk_md = m_reports / "risk_control_threshold.md"
    _write_risk_control_threshold(
        risk_json=risk_json,
        risk_md=risk_md,
        oof_df=oof_df,
        test_df=test_df,
        e_hat_test=e_hat_test,
        aer=aer,
        options=options,
        seed=seed,
        aer_kwargs=_build_aer_kwargs(options),
        groups_arr=groups_arr,
        e_hat_oof_cv=e_hat_oof_cv,
        oof_risk_source=oof_risk_source,
    )

    return curve_test


def run_base_model_analyses(
    *,
    top_results: list,
    slugs: list[str],
    prep_train,
    prep_test,
    options,
    base_dir: Path,
    no_preds: bool,
    seed: int,
) -> BaseModelAnalysisResult:
    y_test_arr = prep_test.y.to_numpy()

    def _test_acc(res: Any, X_train, y_train, X_test) -> float:
        model = getattr(res, "model", None)
        if model is None:
            return float("nan")
        tuned_model = getattr(model, "tuned_model", None)
        if tuned_model is None:
            tuned_args = getattr(model, "tuned_args", None) or res.params
            y_train_fit = y_train
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
                y_train_fit = y_train.iloc[:, 0]
            model.refit_tuned(
                X_train,
                y_train_fit,
                tuned_args=tuned_args,
            )
        preds = np.asarray(model.tuned_predict(X_test)).ravel()
        if preds.size != y_test_arr.size:
            return float("nan")
        return float(np.mean(preds == y_test_arr))

    best_result = top_results[0]
    compare_bins: dict[str, pd.DataFrame] = {}
    compare_test_bins: dict[str, pd.DataFrame] = {}
    compare_tests: dict[str, Optional[pd.DataFrame]] = {}
    compare_metrics: list[dict[str, Any]] = []
    model_run_info: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    ensemble_models: list[dict[str, Any]] = []

    models_dir = base_dir / "models"
    windows_io_path(models_dir).mkdir(parents=True, exist_ok=True)

    y_train = prep_train.y
    y_test = prep_test.y
    groups = prep_train.groups
    labels_map = getattr(prep_train, "labels", None)

    base_test = pd.DataFrame(
        {"row_id": prep_test.X.index, "y_true": y_test.to_numpy()}
    )

    best_slug: Optional[str] = None
    best_curve_test: Optional[pd.DataFrame] = None

    for rank, (result, slug) in enumerate(zip(top_results, slugs), start=1):
        is_best = result is best_result
        if is_best:
            best_slug = slug
        X_train = prep_train.X[result.selected_cols]
        X_test = prep_test.X[result.selected_cols]

        # write per-model outputs under the directory names
        out_dir = models_dir / slug
        output_dirs = _init_model_output_dirs(out_dir)
        m_plots = output_dirs.plots
        m_tables = output_dirs.tables
        m_preds = output_dirs.preds
        m_meta = output_dirs.meta
        m_reports = output_dirs.reports

        metric_name = (
            result.metric.value
            if hasattr(result.metric, "value")
            else str(result.metric)
        )
        if out_dir.is_relative_to(base_dir):
            out_dir_rel = out_dir.relative_to(base_dir).as_posix()
        else:
            out_dir_rel = out_dir.as_posix()

        model_rows.append(
            {
                "rank": rank,
                "model_slug": slug,
                "metric": metric_name,
                "cv_score": result.score,
                "test_accuracy": _test_acc(result, X_train, y_train, X_test),
                "n_features": len(result.selected_cols),
                "out_dir_rel": out_dir_rel,
            }
        )

        aer_kwargs = _build_aer_kwargs(options)

        oof_result = _run_oof_stage(
            result=result,
            X_train=X_train,
            y_train=y_train,
            groups=groups,
            options=options,
            seed=seed,
            aer_kwargs=aer_kwargs,
            m_meta=m_meta,
            slug=slug,
        )

        _calibrator = oof_result.calibrator

        # fit aer mapping
        aer_result = _fit_aer_stage(
            oof_df=oof_result.oof_df,
            proba_oof_cal=oof_result.proba_oof_cal,
            aer_kwargs=aer_kwargs,
            options=options,
            seed=seed,
            groups_arr=oof_result.groups_arr,
            m_meta=m_meta,
            m_tables=m_tables,
            m_preds=m_preds,
            no_preds=no_preds,
        )

        if aer_result.bins_df is not None and not aer_result.bins_df.empty:
            compare_bins[slug] = aer_result.bins_df

        # evaluate on the test
        test_result = _evaluate_test_stage(
            result=result,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels_map=labels_map,
            aer=aer_result.aer,
            calibrator=_calibrator,
            selected_conf_metric=oof_result.selected_conf_metric,
            selected_conf_params=oof_result.selected_conf_params,
            options=options,
            no_preds=no_preds,
            m_preds=m_preds,
            m_tables=m_tables,
            m_plots=m_plots,
            m_meta=m_meta,
        )

        test_df = test_result.test_df
        used_conf_metric = test_result.used_conf_metric
        drift_rate = test_result.drift_rate
        compare_tests[slug] = test_df
        if test_result.test_bins_df is not None and not test_result.test_bins_df.empty:
            compare_test_bins[slug] = test_result.test_bins_df.copy()

        if options.aer_ensemble:
            _append_ensemble_model(
                ensemble_models=ensemble_models,
                slug=slug,
                oof_df=oof_result.oof_df,
                proba_oof_cal=oof_result.proba_oof_cal,
                e_hat_oof=aer_result.e_hat_oof,
                e_hat_oof_cv=aer_result.e_hat_oof_cv,
                proba_test_cal=test_result.proba_test_cal,
                e_hat_test=test_result.e_hat_test,
            )

        selected_conf_metric = oof_result.selected_conf_metric
        selected_conf_params = oof_result.selected_conf_params

        # brier_error_test: mean squared error of the aER estimates
        # ece_error_test: calibration error of the aER estimates

        metrics_path = m_meta / "adaptive_error_metrics.json"
        rel_csv_path = m_tables / "test_error_reliability_bins.csv"
        y_true_arr = test_df["y_true"].to_numpy()
        y_pred_arr = test_df["y_pred"].to_numpy()
        e_hat_test = test_df["aer"].to_numpy(dtype=float)
        incorrect_test = (y_pred_arr != y_true_arr).astype(int)
        n_test = int(y_true_arr.size)
        (
            global_error_test,
            brier_error_test,
            ece_error_test,
        ) = _write_test_error_metrics(
            metrics_path=metrics_path,
            rel_csv_path=rel_csv_path,
            e_hat_test=e_hat_test,
            incorrect_test=incorrect_test,
            n_test=n_test,
            n_bins=options.aer_bins,
            selected_conf_metric=selected_conf_metric,
            used_conf_metric=used_conf_metric,
            selected_conf_params=selected_conf_params,
            drift_rate=drift_rate,
        )

        compare_metrics.append(
            {
                "rank": rank,
                "model_slug": slug,
                "confidence_metric": used_conf_metric,
                "global_error_test": global_error_test,
                "brier_error_test": brier_error_test,
                "ece_error_test": ece_error_test,
            }
        )

        model_run_info.append(
            {
                "model_slug": slug,
                "confidence_metric_selected": selected_conf_metric,
                "confidence_metric_used": used_conf_metric,
                "confidence_metric_params": selected_conf_params,
                "proba_calibrator_method": getattr(_calibrator, "method", None),
                "external_proba_calibration": getattr(_calibrator, "method", "none")
                != "none",
                "internal_model_calibration": bool(
                    getattr(getattr(result, "model", None), "needs_calibration", False)
                ),
                "prediction_source": "tuned_predict",
                "pred_drift_rate": drift_rate,
                "oof_risk_source": aer_result.oof_risk_source,
                "oof_risk_status": (
                    aer_result.oof_risk_diag.get("status")
                    if aer_result.oof_risk_diag is not None
                    else None
                ),
                "oof_risk_condition": (
                    aer_result.oof_risk_diag.get("condition")
                    if aer_result.oof_risk_diag is not None
                    else None
                ),
                "oof_risk_reason": (
                    aer_result.oof_risk_diag.get("reason")
                    if aer_result.oof_risk_diag is not None
                    else None
                ),
                "calibration_status": oof_result.calibration_status,
                "calibration_condition": oof_result.calibration_condition,
                "calibration_reason": oof_result.calibration_reason,
            }
        )

        model_curve = _write_model_extras(
            test_df=test_df,
            no_preds=no_preds,
            m_tables=m_tables,
            m_reports=m_reports,
            m_plots=m_plots,
            m_meta=m_meta,
            options=options,
            oof_df=oof_result.oof_df,
            e_hat_test=test_result.e_hat_test,
            incorrect_test=test_result.incorrect_test,
            aer=aer_result.aer,
            seed=seed,
            groups_arr=oof_result.groups_arr,
            e_hat_oof_cv=aer_result.e_hat_oof_cv,
            oof_risk_source=aer_result.oof_risk_source,
        )
        if is_best and model_curve is not None and not model_curve.empty:
            best_curve_test = model_curve.copy()

    return BaseModelAnalysisResult(
        compare_bins=compare_bins,
        compare_test_bins=compare_test_bins,
        compare_tests=compare_tests,
        compare_metrics=compare_metrics,
        model_run_info=model_run_info,
        model_rows=model_rows,
        ensemble_models=ensemble_models,
        best_slug=best_slug,
        best_curve_test=best_curve_test,
        base_test=base_test,
    )

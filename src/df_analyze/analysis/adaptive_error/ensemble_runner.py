"""adaptive error analysis runner for ensemble strategies"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.confidence_metrics import (
    proba_margin,
    proba_margin_for_pred,
    proba_max_for_pred,
    proba_p2nd,
    proba_pmax,
)
from df_analyze.analysis.adaptive_error.ensemble_config import _build_strategy_labels
from df_analyze.analysis.adaptive_error.ensemble_registry import (
    EnsembleStrategyContext,
    run_ensemble_strategy,
)
from df_analyze.analysis.adaptive_error.ensemble_writer import (
    _finalize_ensemble_strategy,
    _fit_ensemble_confidence_mapping,
    _write_ensemble_confidence_bins,
    _write_ensemble_coverage_curve,
    _write_ensemble_not_available,
    _write_ensemble_per_sample_outputs,
    _write_ensemble_summary,
)
from df_analyze.analysis.adaptive_error.plots import (
    plot_coverage_vs_accuracy_overlay,
)
from df_analyze.analysis.adaptive_error.proba import normalize_proba
from df_analyze.analysis.adaptive_error.report import (
    _coverage_summary_by_accuracy,
    _coverage_summary_with_auc,
    _write_csv,
    _write_text,
)
from df_analyze.analysis.adaptive_error.risk_control import _fit_hens_calibrator
from df_analyze.saving import windows_io_path


def _align_to_reference(
    ref_row_id: np.ndarray,
    row_id: np.ndarray,
    arrays: list[np.ndarray],
) -> list[np.ndarray]:
    if np.array_equal(row_id, ref_row_id):
        return arrays
    idx_map = {rid: i for i, rid in enumerate(row_id.tolist())}
    try:
        order = np.array([idx_map[rid] for rid in ref_row_id.tolist()], dtype=int)
    except KeyError as e:
        raise RuntimeError(f"Missing row_id during alignment: {e}")
    return [arr[order] for arr in arrays]


@dataclass
class _StrategyDirs:
    plots: Path
    tables: Path
    preds: Path
    meta: Path
    reports: Path


def _init_ensemble_dirs(
    base_dir: Path,
    options,
) -> tuple[Path, Path, Path, Path, dict[str, str], dict[str, Path]]:
    ensemble_dir = base_dir / "ensemble"
    windows_io_path(ensemble_dir).mkdir(parents=True, exist_ok=True)
    ens_plots_dir = ensemble_dir / "plots"
    ens_tables_dir = ensemble_dir / "tables"
    ens_reports_dir = ensemble_dir / "reports"
    for _d in (ens_plots_dir, ens_tables_dir, ens_reports_dir):
        windows_io_path(_d).mkdir(parents=True, exist_ok=True)

    # 10 adaptive-error ensemble strategies
    # when --aer-ensemble is enabled
    strategy_labels = _build_strategy_labels(options)
    strategy_dirs = {key: ensemble_dir / label for key, label in strategy_labels.items()}
    for sdir in strategy_dirs.values():
        windows_io_path(sdir).mkdir(parents=True, exist_ok=True)

    return (
        ensemble_dir,
        ens_plots_dir,
        ens_tables_dir,
        ens_reports_dir,
        strategy_labels,
        strategy_dirs,
    )


def _collect_valid_models(
    *,
    ensemble_models: list[dict[str, Any]],
    ref_row_id: np.ndarray,
    n_test: int,
) -> list[dict[str, Any]]:
    ref_n_classes = None
    valid_models: list[dict[str, Any]] = []
    for info in ensemble_models:
        slug = info["slug"]
        proba_oof = normalize_proba(np.asarray(info["oof_proba"]))
        proba_test = normalize_proba(np.asarray(info["test_proba"]))

        if ref_n_classes is None:
            ref_n_classes = int(proba_oof.shape[1])

        if proba_oof.ndim != 2 or proba_test.ndim != 2:
            warn(f"Ensemble skipped for {slug}: probability array shape invalid.")
            continue

        if proba_oof.shape[1] != ref_n_classes or proba_test.shape[1] != ref_n_classes:
            warn(f"Ensemble skipped for {slug}: class count mismatch.")
            continue

        if proba_oof.shape[0] != ref_row_id.size or proba_test.shape[0] != n_test:
            warn(f"Ensemble skipped for {slug}: sample count mismatch.")
            continue

        aer_oof = np.asarray(info["oof_aer"], dtype=float).ravel()
        aer_test = np.asarray(info["test_aer"], dtype=float).ravel()
        if aer_oof.shape[0] != ref_row_id.size or aer_test.shape[0] != n_test:
            warn(f"Ensemble skipped for {slug}: adaptive error length mismatch.")
            continue

        row_id = np.asarray(info["oof_row_id"])
        proba_oof, aer_oof = _align_to_reference(
            ref_row_id, row_id, [proba_oof, aer_oof]
        )

        aer_oof = np.clip(
            np.nan_to_num(aer_oof, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        aer_test = np.clip(
            np.nan_to_num(aer_test, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )

        valid_models.append(
            {
                "slug": slug,
                "oof_proba": proba_oof,
                "oof_aer": aer_oof,
                "oof_acc": float(info.get("oof_acc", float("nan"))),
                "test_proba": proba_test,
                "test_aer": aer_test,
            }
        )

    return valid_models


def _build_strategy_context(
    valid_models: list[dict[str, Any]],
) -> EnsembleStrategyContext:
    slugs_ens = [m["slug"] for m in valid_models]
    oof_acc = np.asarray([m["oof_acc"] for m in valid_models], dtype=float)
    proba_oof_stack = np.stack([m["oof_proba"] for m in valid_models], axis=0)
    aer_oof_stack = np.stack([m["oof_aer"] for m in valid_models], axis=0)
    proba_test_stack = np.stack([m["test_proba"] for m in valid_models], axis=0)
    aer_test_stack = np.stack([m["test_aer"] for m in valid_models], axis=0)
    return EnsembleStrategyContext(
        proba_oof_stack=proba_oof_stack,
        aer_oof_stack=aer_oof_stack,
        oof_acc=oof_acc,
        proba_test_stack=proba_test_stack,
        aer_test_stack=aer_test_stack,
        slugs_ens=slugs_ens,
    )


def _prepare_strategy_dirs(sdir: Path) -> _StrategyDirs:
    s_plots = sdir / "plots"
    s_tables = sdir / "tables"
    s_preds = sdir / "predictions"
    s_meta = sdir / "metadata"
    s_reports = sdir / "reports"
    for d in (s_plots, s_tables, s_preds, s_meta, s_reports):
        windows_io_path(d).mkdir(parents=True, exist_ok=True)
    return _StrategyDirs(
        plots=s_plots,
        tables=s_tables,
        preds=s_preds,
        meta=s_meta,
        reports=s_reports,
    )


def _normalize_requested_conf_metric(requested_conf_metric) -> Optional[str]:
    if requested_conf_metric is None:
        return None
    requested_conf_metric = str(requested_conf_metric).strip().lower().replace("-", "_")
    if requested_conf_metric in ("", "auto", "best", "select", "default", "none"):
        return None
    return requested_conf_metric


def _compute_ensemble_prob_diagnostics(
    p_ens_oof: np.ndarray,
    p_ens_test: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "p_max_oof": proba_pmax(p_ens_oof),
        "p_2nd_oof": proba_p2nd(p_ens_oof),
        "p_margin_oof": proba_margin(p_ens_oof),
        "p_max_test": proba_pmax(p_ens_test),
        "p_2nd_test": proba_p2nd(p_ens_test),
        "p_margin_test": proba_margin(p_ens_test),
    }


def _select_ensemble_confidence_metric(
    *,
    requested_conf_metric: Optional[str],
    p_ens_oof: np.ndarray,
    p_ens_test: np.ndarray,
    y_pred_oof: np.ndarray,
    y_pred_test: np.ndarray,
    prob_diag: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str]:
    conf_metric_used = "p_margin"
    conf_oof = prob_diag["p_margin_oof"]
    conf_test = prob_diag["p_margin_test"]
    if requested_conf_metric:
        if requested_conf_metric in ("proba_margin", "p_pred_margin"):
            conf_oof = proba_margin_for_pred(p_ens_oof, y_pred_oof)
            conf_test = proba_margin_for_pred(p_ens_test, y_pred_test)
            conf_metric_used = "proba_margin"
        elif requested_conf_metric in ("p_pred", "proba_max_for_pred"):
            conf_oof = proba_max_for_pred(p_ens_oof, y_pred_oof)
            conf_test = proba_max_for_pred(p_ens_test, y_pred_test)
            conf_metric_used = "p_pred"
        elif requested_conf_metric == "p_max":
            conf_oof = prob_diag["p_max_oof"]
            conf_test = prob_diag["p_max_test"]
            conf_metric_used = "p_max"
        elif requested_conf_metric == "p_2nd":
            conf_oof = prob_diag["p_2nd_oof"]
            conf_test = prob_diag["p_2nd_test"]
            conf_metric_used = "p_2nd"
        elif requested_conf_metric == "p_margin":
            conf_oof = prob_diag["p_margin_oof"]
            conf_test = prob_diag["p_margin_test"]
            conf_metric_used = "p_margin"
        else:
            warn(
                f"Requested confidence metric '{requested_conf_metric}' not "
                f"recognized for ensembles; using '{conf_metric_used}'."
            )
    return conf_oof, conf_test, conf_metric_used


def _run_strategy_analysis(
    *,
    strategy_key: str,
    strategy_label: str,
    sdir: Path,
    strategy_context: EnsembleStrategyContext,
    options,
    n_test: int,
    no_preds: bool,
    y_true_oof: np.ndarray,
    y_true_test: np.ndarray,
    ref_row_id: np.ndarray,
    row_id_test: np.ndarray,
    groups,
    seed: int,
    ensemble_curves: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    summary = {
        "strategy_name": strategy_label,
        "n_models_used": int(strategy_context.proba_oof_stack.shape[0]),
        "test_accuracy": None,
        "test_global_error": None,
        "brier_error_test": None,
        "ece_error_test": None,
        "risk_control_feasible": None,
        "t_star": None,
        "test_coverage_at_t_star": None,
        "test_error_at_t_star": None,
        "status": "not_available",
        "reason": "",
    }

    s_dirs = _prepare_strategy_dirs(sdir)

    strategy_result = run_ensemble_strategy(strategy_key, strategy_context, options)
    strategy_params = strategy_result.params
    p_ens_oof = strategy_result.p_ens_oof
    r_star_oof = strategy_result.r_star_oof
    p_ens_test = strategy_result.p_ens_test
    r_star_test = strategy_result.r_star_test

    p_ens_oof = normalize_proba(p_ens_oof)
    p_ens_test = normalize_proba(p_ens_test)

    y_pred_oof = np.argmax(p_ens_oof, axis=1)
    y_pred_test = np.argmax(p_ens_test, axis=1)
    incorrect_oof = (y_pred_oof != y_true_oof).astype(int)
    incorrect_test = (y_pred_test != y_true_test).astype(int)
    correct_oof = (incorrect_oof == 0).astype(int)
    correct_test = (incorrect_test == 0).astype(int)

    prob_diag = _compute_ensemble_prob_diagnostics(p_ens_oof, p_ens_test)
    requested_conf_metric = _normalize_requested_conf_metric(
        options.aer_confidence_metric
    )
    conf_oof, conf_test, _ = _select_ensemble_confidence_metric(
        requested_conf_metric=requested_conf_metric,
        p_ens_oof=p_ens_oof,
        p_ens_test=p_ens_test,
        y_pred_oof=y_pred_oof,
        y_pred_test=y_pred_test,
        prob_diag=prob_diag,
    )

    p_err_oof, p_err_test, hens_payload = _fit_hens_calibrator(
        r_star_oof, incorrect_oof, r_star_test
    )

    _write_text(
        s_dirs.meta / "hens_calibrator.json",
        json.dumps(hens_payload, indent=2),
    )

    _write_text(
        s_dirs.meta / "strategy.json",
        json.dumps(
            {
                "strategy_key": strategy_key,
                "strategy_label": strategy_label,
                "parameters": strategy_params,
            },
            indent=2,
        ),
    )

    aer_conf, aer_conf_oof, aer_conf_test = _fit_ensemble_confidence_mapping(
        conf_oof=conf_oof,
        conf_test=conf_test,
        y_pred_oof=y_pred_oof,
        y_true_oof=y_true_oof,
        options=options,
        s_meta=s_dirs.meta,
        s_tables=s_dirs.tables,
    )

    target_error = getattr(options, "aer_target_error", 0.05)
    if target_error is None:
        target_error = 0.05
    target_error = float(target_error)
    _write_ensemble_per_sample_outputs(
        s_preds=s_dirs.preds,
        s_tables=s_dirs.tables,
        no_preds=no_preds,
        target_error=target_error,
        ref_row_id=ref_row_id,
        row_id_test=row_id_test,
        y_true_oof=y_true_oof,
        y_true_test=y_true_test,
        y_pred_oof=y_pred_oof,
        y_pred_test=y_pred_test,
        correct_oof=correct_oof,
        correct_test=correct_test,
        conf_oof=conf_oof,
        conf_test=conf_test,
        prob_diag=prob_diag,
        r_star_oof=r_star_oof,
        r_star_test=r_star_test,
        p_err_oof=p_err_oof,
        p_err_test=p_err_test,
        aer_conf_oof=aer_conf_oof,
        aer_conf_test=aer_conf_test,
    )

    _write_ensemble_confidence_bins(
        s_tables=s_dirs.tables,
        s_plots=s_dirs.plots,
        aer_conf=aer_conf,
        conf_test=conf_test,
        incorrect_test=incorrect_test,
        aer_conf_test=aer_conf_test,
        options=options,
    )

    curve_test = _write_ensemble_coverage_curve(
        s_tables=s_dirs.tables,
        s_plots=s_dirs.plots,
        s_reports=s_dirs.reports,
        p_err_test=p_err_test,
        correct_test=correct_test,
    )
    if curve_test is not None and not curve_test.empty:
        ensemble_curves[strategy_label] = curve_test.copy()

    _finalize_ensemble_strategy(
        summary=summary,
        s_meta=s_dirs.meta,
        s_tables=s_dirs.tables,
        s_reports=s_dirs.reports,
        p_err_test=p_err_test,
        incorrect_test=incorrect_test,
        n_test=n_test,
        options=options,
        r_star_oof=r_star_oof,
        p_err_oof=p_err_oof,
        p_err_test_full=p_err_test,
        incorrect_oof=incorrect_oof,
        incorrect_test_full=incorrect_test,
        groups=groups,
        ref_row_id=ref_row_id,
        seed=seed,
    )

    summary["status"] = "ok"
    summary["reason"] = ""
    return summary


def run_ensemble_analysis(
    *,
    base_dir: Path,
    plots_dir: Path,
    tables_dir: Path,
    prep_train,
    prep_test,
    options,
    no_preds: bool = False,
    ensemble_models: list[dict[str, Any]],
    best_slug: Optional[str],
    best_curve_test: Optional[pd.DataFrame],
    seed: int,
) -> None:
    ensemble_curves: dict[str, pd.DataFrame] = {}
    y_train = prep_train.y
    y_test = prep_test.y
    groups = prep_train.groups
    if not options.aer_ensemble:
        return

    (
        _ensemble_dir,
        _ens_plots_dir,
        ens_tables_dir,
        ens_reports_dir,
        strategy_labels,
        strategy_dirs,
    ) = _init_ensemble_dirs(base_dir, options)

    n_test = int(y_test.size)
    summary_rows: list[dict[str, Any]] = []

    if len(ensemble_models) < 2:
        reason = "Fewer than 2 models with usable probabilities for ensemble."
        for strategy_key, sdir in strategy_dirs.items():
            strategy_label = strategy_labels.get(strategy_key, strategy_key)
            _write_ensemble_not_available(sdir, reason, options, n_test)
            summary_rows.append(
                {
                    "strategy_name": strategy_label,
                    "n_models_used": int(len(ensemble_models)),
                    "test_accuracy": None,
                    "test_global_error": None,
                    "brier_error_test": None,
                    "ece_error_test": None,
                    "risk_control_feasible": None,
                    "t_star": None,
                    "test_coverage_at_t_star": None,
                    "test_error_at_t_star": None,
                    "status": "not_available",
                    "reason": reason,
                }
            )
        _write_ensemble_summary(ens_tables_dir, ens_reports_dir, summary_rows)
        return

    ref_row_id = np.asarray(prep_train.X.index)
    valid_models = _collect_valid_models(
        ensemble_models=ensemble_models,
        ref_row_id=ref_row_id,
        n_test=n_test,
    )

    if len(valid_models) < 2:
        reason = "Fewer than 2 models with aligned probabilities for ensemble."
        for strategy_key, sdir in strategy_dirs.items():
            strategy_label = strategy_labels.get(strategy_key, strategy_key)
            _write_ensemble_not_available(sdir, reason, options, n_test)
            summary_rows.append(
                {
                    "strategy_name": strategy_label,
                    "n_models_used": int(len(valid_models)),
                    "test_accuracy": None,
                    "test_global_error": None,
                    "brier_error_test": None,
                    "ece_error_test": None,
                    "risk_control_feasible": None,
                    "t_star": None,
                    "test_coverage_at_t_star": None,
                    "test_error_at_t_star": None,
                    "status": "not_available",
                    "reason": reason,
                }
            )
        _write_ensemble_summary(ens_tables_dir, ens_reports_dir, summary_rows)
        return

    strategy_context = _build_strategy_context(valid_models)

    y_true_oof = y_train.to_numpy()
    y_true_test = y_test.to_numpy()
    row_id_test = prep_test.X.index.to_numpy()

    for strategy_key, sdir in strategy_dirs.items():
        strategy_label = strategy_labels.get(strategy_key, strategy_key)
        summary_rows.append(
            _run_strategy_analysis(
                strategy_key=strategy_key,
                strategy_label=strategy_label,
                sdir=sdir,
                strategy_context=strategy_context,
                options=options,
                n_test=n_test,
                no_preds=no_preds,
                y_true_oof=y_true_oof,
                y_true_test=y_true_test,
                ref_row_id=ref_row_id,
                row_id_test=row_id_test,
                groups=groups,
                seed=seed,
                ensemble_curves=ensemble_curves,
            )
        )

    _write_ensemble_summary(ens_tables_dir, ens_reports_dir, summary_rows)

    overlay_curves: dict[str, pd.DataFrame] = {}
    if (
        best_slug is not None
        and best_curve_test is not None
        and not best_curve_test.empty
    ):
        overlay_curves[f"Base-{best_slug}"] = best_curve_test
    for name, curve in ensemble_curves.items():
        if curve is None or curve.empty:
            continue
        overlay_curves[name] = curve

    if not overlay_curves:
        return

    coverage_targets = [1.0, 0.9, 0.8, 0.7]
    accuracy_targets = [0.99, 0.995]
    overlay_summary_rows = []
    overlay_summary_rows_by_acc = []
    for name, curve in overlay_curves.items():
        summary_df, _ = _coverage_summary_with_auc(curve, targets=coverage_targets)
        if not summary_df.empty:
            summary_df.insert(0, "method", name)
            overlay_summary_rows.append(summary_df)

        acc_summary_df, _ = _coverage_summary_by_accuracy(
            curve, target_accuracies=accuracy_targets
        )
        if not acc_summary_df.empty:
            acc_summary_df.insert(0, "method", name)
            overlay_summary_rows_by_acc.append(acc_summary_df)

    if overlay_summary_rows:
        overlay_summary = pd.concat(overlay_summary_rows, ignore_index=True)
        _write_csv(
            overlay_summary,
            tables_dir / "coverage_accuracy_overlay_summary.csv",
            index=False,
        )

    if overlay_summary_rows_by_acc:
        overlay_acc_summary = pd.concat(
            overlay_summary_rows_by_acc, ignore_index=True
        )
        _write_csv(
            overlay_acc_summary,
            tables_dir / "coverage_accuracy_overlay_by_accuracy_summary.csv",
            index=False,
        )

    plot_coverage_vs_accuracy_overlay(
        overlay_curves,
        plots_dir / "coverage_vs_accuracy_overlay.png",
        title="Coverage vs accuracy (overlay)",
    )

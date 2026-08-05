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
from df_analyze.analysis.adaptive_error.base_models_compute import _crossfit_oof_risk
from df_analyze.analysis.adaptive_error.report import (
    _write_markdown,
    _write_text,
)
from df_analyze.analysis.adaptive_error.risk_control import (
    _crossfit_hens_error_probability,
    _select_risk_control_threshold,
)


def _write_risk_control_threshold(
    *,
    risk_json: Path,
    risk_md: Path,
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    e_hat_test: np.ndarray,
    aer: AdaptiveErrorCalculator,
    options,
    seed: int,
    aer_kwargs: dict[str, Any],
    groups_arr,
    e_hat_oof_cv: Optional[np.ndarray],
    oof_risk_source: Optional[str],
) -> None:
    target_err = float(options.aer_target_error)
    alpha = float(options.aer_alpha)
    nmin = int(options.aer_nmin)
    if oof_risk_source is None:
        oof_risk_source = "naive"
    conf_oof = oof_df["conf_oof"].to_numpy()
    e_hat_oof = aer.get_expected_error(conf_oof)
    incorrect_oof = (
        oof_df["y_pred_oof"].to_numpy() != oof_df["y_true"].to_numpy()
    ).astype(int)
    if e_hat_oof_cv is None:
        n_splits = int(min(5, int(max(2, options.aer_oof_folds))))
        e_hat_oof_cv, oof_risk_source, oof_risk_diag = _crossfit_oof_risk(
            conf_oof=conf_oof,
            incorrect_oof=incorrect_oof,
            groups_arr=groups_arr,
            row_id=oof_df["row_id"],
            n_splits=n_splits,
            seed=seed,
            aer_kwargs=aer_kwargs,
        )
    else:
        oof_risk_diag = {
            "status": "DONE",
            "reason": "",
            "condition": "",
            "source": oof_risk_source,
        }

    risk_for_tstar = e_hat_oof_cv if e_hat_oof_cv is not None else e_hat_oof
    if risk_for_tstar is None:
        raise ValueError("Risk estimates unavailable for threshold selection.")

    best, reason, meta = _select_risk_control_threshold(
        risk=risk_for_tstar,
        incorrect=incorrect_oof,
        target_err=target_err,
        alpha=alpha,
        nmin=nmin,
    )
    n_all = int((np.isfinite(risk_for_tstar) & np.isfinite(incorrect_oof)).sum())
    bonferroni_m = int(meta.get("bonferroni_m", 0))
    alpha_adj = float(meta.get("alpha_adj", alpha))
    oof_n_total = int(meta.get("oof_n_total", n_all))

    oof_risk_status = (
        oof_risk_diag.get("status", "DONE") if oof_risk_diag else "DONE"
    )
    oof_risk_reason = oof_risk_diag.get("reason", "") if oof_risk_diag else ""
    oof_risk_condition = (
        oof_risk_diag.get("condition", "") if oof_risk_diag else ""
    )

    if best is None:
        payload = {
            "feasible": False,
            "reason": reason,
            "target_error": target_err,
            "alpha": alpha,
            "alpha_adj": alpha_adj,
            "bonferroni_m": bonferroni_m,
            "aer_nmin": nmin,
            "oof_n_total": oof_n_total,
            "oof_risk_source": oof_risk_source,
            "oof_risk_status": oof_risk_status,
            "oof_risk_reason": oof_risk_reason,
            "oof_risk_condition": oof_risk_condition,
        }
        _write_text(risk_json, json.dumps(payload, indent=2))

        md = [
            "# Risk-controlled threshold (t*)",
            "",
            f"Target error: **{target_err:.2%}**",
            "",
            f"alpha: {alpha:.4g}",
            f"m (Bonferroni): {bonferroni_m}",
            f"alpha_adj: {alpha_adj:.4g}",
            "",
            "Threshold selected using OOF calibration data.",
            "",
            "Not available.",
            "",
            f"Reason: {reason}",
        ]
        _write_markdown(risk_md, md)
        return
    else:
        correct_mask = test_df["correct"].to_numpy(dtype=int).astype(bool)
        mask_test = (e_hat_test <= best["t_star"]).astype(bool)
        n_test = int(mask_test.sum())
        k_test = int((~correct_mask[mask_test]).sum()) if n_test > 0 else 0
        test_cov = float(n_test / len(e_hat_test)) if len(e_hat_test) else 0.0
        test_err_val = float(k_test / n_test) if n_test > 0 else float("nan")
        test_acc_val = (
            float((correct_mask[mask_test]).mean()) if n_test > 0 else float("nan")
        )
        test_err = test_err_val if np.isfinite(test_err_val) else None
        test_acc = test_acc_val if np.isfinite(test_acc_val) else None

        best.update(
            {
                "feasible": True,
                "test_n": n_test,
                "test_k": k_test,
                "test_coverage": test_cov,
                "test_error_rate": test_err,
                "test_selective_accuracy": test_acc,
                "target_error": target_err,
                "alpha": alpha,
                "alpha_adj": alpha_adj,
                "bonferroni_m": bonferroni_m,
                "aer_nmin": nmin,
                "oof_risk_source": oof_risk_source,
                "oof_risk_status": oof_risk_status,
                "oof_risk_reason": oof_risk_reason,
                "oof_risk_condition": oof_risk_condition,
            }
        )

        _write_text(risk_json, json.dumps(best, indent=2))

        t_star = float(best["t_star"])
        oof_coverage = float(best["oof_coverage"])
        oof_n = int(best["oof_n"])
        oof_n_total_selected = int(best.get("oof_n_total", oof_n_total))
        oof_error_rate = float(best["oof_error_rate"])
        oof_k = int(best["oof_k"])
        oof_cp_upper = float(best["oof_cp_upper"])

        test_coverage = float(test_cov)
        test_n = int(n_test)
        test_total = int(len(e_hat_test))

        md = [
            "# Risk-controlled threshold (t*)",
            "",
            f"Target error: **{target_err:.2%}**",
            "",
            f"alpha: {alpha:.4g}",
            f"m (Bonferroni): {bonferroni_m}",
            f"alpha_adj: {alpha_adj:.4g}",
            "",
            "Threshold selected using OOF calibration data.",
            "",
            "## Selected threshold on OOF train",
            f"- t*: **{t_star:.4f}**",
            (
                f"- OOF coverage: **{oof_coverage:.2%}** "
                f"({oof_n} / {oof_n_total_selected})"
            ),
            f"- OOF observed error: **{oof_error_rate:.2%}** ({oof_k} / {oof_n})",
            f"- OOF CP upper bound: **{oof_cp_upper:.2%}**",
            "",
            "## Applied to holdout test",
            f"- Test coverage: **{test_coverage:.2%}** ({test_n} / {test_total})",
            f"- Test selective accuracy: **{test_acc_val:.2%}**",
            f"- Test observed error: **{test_err_val:.2%}**",
        ]
        _write_markdown(risk_md, md)


def _write_ensemble_risk_control(
    *,
    summary: dict[str, Any],
    risk_json: Path,
    risk_md: Path,
    r_star_oof: np.ndarray,
    p_err_oof: np.ndarray,
    p_err_test: np.ndarray,
    incorrect_oof: np.ndarray,
    incorrect_test: np.ndarray,
    options,
    groups,
    ref_row_id: np.ndarray,
    seed: int,
) -> None:
    target_err = float(options.aer_target_error)
    alpha = float(options.aer_alpha)
    nmin = int(options.aer_nmin)
    oof_risk_source = "naive"
    p_err_oof_cv = None
    oof_risk_status = "DONE"
    oof_risk_reason = ""
    oof_risk_condition = ""
    groups_for_risk = None
    if groups is not None:
        if hasattr(groups, "reindex"):
            groups_for_risk = groups.reindex(ref_row_id)
        else:
            groups_for_risk = groups
        if groups_for_risk is not None:
            groups_for_risk = np.asarray(groups_for_risk)
    if groups_for_risk is not None and groups_for_risk.shape[0] != r_star_oof.shape[0]:
        groups_for_risk = None

    mask = np.isfinite(r_star_oof) & np.isfinite(incorrect_oof)
    if groups_for_risk is not None:
        mask = mask & (~pd.isna(groups_for_risk))
    n_splits = int(min(5, int(max(2, options.aer_oof_folds))))
    n_valid = int(mask.sum())
    uniq_incorrect = int(np.unique(incorrect_oof[mask]).size)
    can_crossfit = (
        n_splits >= 2
        and n_valid >= n_splits
        and uniq_incorrect >= 2
        and np.unique(r_star_oof[mask]).size >= 2
    )
    if can_crossfit:
        p_err_oof_cv = _crossfit_hens_error_probability(
            r_star_oof,
            incorrect_oof,
            groups=groups_for_risk,
            n_splits=n_splits,
            seed=seed,
        )
        if np.isfinite(p_err_oof_cv).sum() > 0:
            oof_risk_source = "crossfit"
        else:
            p_err_oof_cv = None
            oof_risk_status = "FALLBACK"
            oof_risk_condition = "crossfit_nonfinite"
            oof_risk_reason = "Cross-fit HENS risk returned no finite values."
    else:
        oof_risk_status = "FALLBACK"
        oof_risk_condition = "insufficient_samples_or_variance"
        oof_risk_reason = (
            "Cross-fit HENS risk disabled due to insufficient samples"
        )

    risk_for_tstar = p_err_oof_cv if p_err_oof_cv is not None else p_err_oof
    best, reason, meta = _select_risk_control_threshold(
        risk_for_tstar, incorrect_oof, target_err, alpha, nmin
    )
    bonferroni_m = int(meta.get("bonferroni_m", 0))
    alpha_adj = float(meta.get("alpha_adj", alpha))
    oof_n_total = int(meta.get("oof_n_total", len(risk_for_tstar)))
    if best is None:
        payload = {
            "feasible": False,
            "reason": reason,
            "target_error": target_err,
            "alpha": alpha,
            "alpha_adj": alpha_adj,
            "bonferroni_m": bonferroni_m,
            "aer_nmin": nmin,
            "oof_n": oof_n_total,
            "oof_n_total": oof_n_total,
            "oof_risk_source": oof_risk_source,
            "oof_risk_status": oof_risk_status,
            "oof_risk_reason": oof_risk_reason,
            "oof_risk_condition": oof_risk_condition,
        }
        _write_text(risk_json, json.dumps(payload, indent=2))

        md = [
            "# Risk-controlled threshold (t*)",
            "",
            f"Target error: **{target_err:.2%}**",
            "",
            f"alpha: {alpha:.4g}",
            f"m (Bonferroni): {bonferroni_m}",
            f"alpha_adj: {alpha_adj:.4g}",
            "",
            "Threshold selected using OOF calibration data.",
            "",
            "Not available.",
            "",
            f"Reason: {reason}",
        ]
        _write_markdown(risk_md, md)

        summary["risk_control_feasible"] = False
        return
    else:
        mask_test = (p_err_test <= best["t_star"]).astype(bool)
        n_acc = int(mask_test.sum())
        k_err = int((incorrect_test[mask_test]).sum()) if n_acc > 0 else 0
        test_cov = float(n_acc / len(p_err_test)) if len(p_err_test) else 0.0
        test_err_val = float(k_err / n_acc) if n_acc > 0 else float("nan")
        test_acc_val = (
            float((1 - incorrect_test[mask_test]).mean()) if n_acc > 0 else float("nan")
        )

        best.update(
            {
                "feasible": True,
                "test_n": n_acc,
                "test_k": k_err,
                "test_coverage": test_cov,
                "test_error_rate": test_err_val if np.isfinite(test_err_val) else None,
                "test_selective_accuracy": test_acc_val
                if np.isfinite(test_acc_val)
                else None,
                "target_error": target_err,
                "alpha": alpha,
                "alpha_adj": alpha_adj,
                "bonferroni_m": bonferroni_m,
                "aer_nmin": nmin,
                "oof_risk_source": oof_risk_source,
                "oof_risk_status": oof_risk_status,
                "oof_risk_reason": oof_risk_reason,
                "oof_risk_condition": oof_risk_condition,
            }
        )

        _write_text(risk_json, json.dumps(best, indent=2))

        t_star = float(best["t_star"])
        oof_coverage = float(best["oof_coverage"])
        oof_n = int(best["oof_n"])
        oof_n_total_selected = int(best.get("oof_n_total", oof_n_total))
        oof_error_rate = float(best["oof_error_rate"])
        oof_k = int(best["oof_k"])
        oof_cp_upper = float(best["oof_cp_upper"])

        test_coverage = float(test_cov)
        test_n = int(n_acc)
        test_total = int(len(p_err_test))

        md = [
            "# Risk-controlled threshold (t*)",
            "",
            f"Target error: **{target_err:.2%}**",
            "",
            f"alpha: {alpha:.4g}",
            f"m (Bonferroni): {bonferroni_m}",
            f"alpha_adj: {alpha_adj:.4g}",
            "",
            "Threshold selected using OOF calibration data.",
            "",
            "## Selected threshold on OOF train",
            f"- t*: **{t_star:.4f}**",
            (
                f"- OOF coverage: **{oof_coverage:.2%}** "
                f"({oof_n} / {oof_n_total_selected})"
            ),
            f"- OOF observed error: **{oof_error_rate:.2%}** ({oof_k} / {oof_n})",
            f"- OOF CP upper bound: **{oof_cp_upper:.2%}**",
            "",
            "## Applied to holdout test",
            f"- Test coverage: **{test_coverage:.2%}** ({test_n} / {test_total})",
            f"- Test selective accuracy: **{test_acc_val:.2%}**",
            f"- Test observed error: **{test_err_val:.2%}**",
        ]
        _write_markdown(risk_md, md)

        summary["risk_control_feasible"] = True
        summary["t_star"] = float(best["t_star"])
        summary["test_coverage_at_t_star"] = float(test_cov)
        summary["test_error_at_t_star"] = (
            float(test_err_val) if np.isfinite(test_err_val) else None
        )

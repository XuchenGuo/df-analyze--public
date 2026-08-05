from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.confidence_metrics import (
    apply_exp_gamma,
    apply_minmax,
    fit_exp_gamma_params,
    fit_minmax_params,
    proba_margin,
    proba_margin_for_pred,
    proba_max_for_pred,
    proba_p2nd,
    proba_pmax,
)
from df_analyze.analysis.adaptive_error.confidence_selection import (
    select_confidence_metric,
)
from df_analyze.analysis.adaptive_error.oof import build_oof_for_result
from df_analyze.analysis.adaptive_error.proba import (
    ProbaCalibrator,
    crossfit_calibrated_proba,
    normalize_proba,
    select_best_proba_calibrator,
)


@dataclass
class _OofSelectionResult:
    oof_df: pd.DataFrame
    proba_oof_cal: np.ndarray
    calibrator: Any
    selected_conf_metric: str
    selected_conf_params: dict[str, Any]
    groups_arr: Any
    y_pred_oof_raw: np.ndarray
    calibration_status: str
    calibration_condition: str
    calibration_reason: str


def _build_confidence_candidates(
    *,
    oof_df: pd.DataFrame,
    proba_oof_cal: np.ndarray,
    y_pred_oof_raw: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    conf_candidates: dict[str, np.ndarray] = {}
    conf_params: dict[str, Any] = {}
    oof_df["p_max"] = proba_pmax(proba_oof_cal)
    oof_df["p_2nd"] = proba_p2nd(proba_oof_cal)
    oof_df["p_margin"] = proba_margin(proba_oof_cal)
    oof_df["p_pred"] = proba_max_for_pred(proba_oof_cal, y_pred_oof_raw)
    oof_df["p_pred_margin"] = proba_margin_for_pred(proba_oof_cal, y_pred_oof_raw)
    raw_margin = oof_df["p_pred_margin"].to_numpy(dtype=float)
    conf_candidates["proba_margin"] = raw_margin
    conf_params["proba_margin"] = {"kind": "identity", "params": {}}
    finite = raw_margin[np.isfinite(raw_margin)]
    if finite.size > 0:
        q_low = float(np.quantile(finite, 0.01))
        q_high = float(np.quantile(finite, 0.99))
        if (q_high - q_low) < 0.8:
            p = fit_minmax_params(raw_margin)
            conf_candidates["proba_margin"] = apply_minmax(raw_margin, p)
            conf_params["proba_margin"] = {
                "kind": p.kind,
                "params": dict(p.params),
            }

    def _add_minmax_candidate(name: str, col: str) -> None:
        if col not in oof_df.columns:
            return
        arr = np.asarray(oof_df[col].to_numpy(dtype=float)).ravel()
        if np.isfinite(arr).sum() == 0:
            return
        p = fit_minmax_params(arr)
        conf = apply_minmax(arr, p)
        oof_df[col] = conf
        conf_candidates[name] = conf
        conf_params[name] = {
            "kind": p.kind,
            "params": dict(p.params),
        }

    _add_minmax_candidate("tree_vote_agreement", "conf_tree_vote_agreement")
    _add_minmax_candidate("tree_leaf_support", "conf_tree_leaf_support")
    _add_minmax_candidate("knn_vote", "conf_knn_vote")
    _add_minmax_candidate("knn_dist_weighted", "conf_knn_dist_weighted")
    if "raw_knn_min_dist" in oof_df.columns:
        raw = np.asarray(oof_df["raw_knn_min_dist"].to_numpy(dtype=float)).ravel()
        if np.isfinite(raw).sum() > 0:
            p = fit_exp_gamma_params(raw)
            conf = apply_exp_gamma(raw, p)
            oof_df["conf_knn_min_dist"] = conf
            conf_candidates["knn_min_dist"] = conf
            conf_params["knn_min_dist"] = {
                "kind": p.kind,
                "params": dict(p.params),
            }

    return conf_candidates, conf_params


def _run_oof_stage(
    *,
    result,
    X_train,
    y_train,
    groups,
    options,
    seed: int,
    aer_kwargs: dict[str, Any],
    m_meta: Path,
    slug: str,
) -> _OofSelectionResult:
    oof_df, proba_oof = build_oof_for_result(
        result=result,
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        n_folds=options.aer_oof_folds,
        seed=seed,
        options=options,
    )
    groups_arr = None
    if groups is not None:
        if hasattr(groups, "reindex"):
            groups_arr = groups.reindex(X_train.index)
        else:
            groups_arr = groups

    y_pred_oof_raw = oof_df["y_pred_oof"].to_numpy()
    drift_threshold = 0.02
    params = getattr(result, "params", None)
    loss = params.get("loss") if isinstance(params, dict) else None
    if isinstance(loss, str) and loss.lower() == "modified_huber":
        drift_threshold = 0.1
    calibrator, cal_diag = select_best_proba_calibrator(
        proba_oof,
        oof_df["y_true"].to_numpy(),
        n_splits=3,
        seed=seed,
        groups=groups_arr,
        y_pred=y_pred_oof_raw,
        drift_threshold=drift_threshold,
    )
    cal_cv_diag: dict[str, Any] = {}
    calibration_status = "DONE"
    calibration_condition = ""
    calibration_reason = ""
    proba_oof_cal, cal_cv_diag = crossfit_calibrated_proba(
        proba_oof,
        oof_df["y_true"].to_numpy(),
        method=calibrator.method,
        n_splits=min(5, int(max(2, options.aer_oof_folds))),
        seed=seed,
        groups=groups_arr,
    )

    if cal_cv_diag.get("fallback") == "full_fit":
        warn(
            f"Crossfit calibration unavailable for {slug}; disabling calibration "
            "to avoid OOF label leakage."
        )
        cal_cv_diag["fallback"] = "none"
        cal_cv_diag["disabled"] = "crossfit_full_fit"
        cal_cv_diag["status"] = "FALLBACK"
        cal_cv_diag["condition"] = "crossfit_full_fit"
        cal_cv_diag["reason"] = (
            "Crossfit fallback full_fit disabled to avoid OOF label leakage."
        )
        calibration_status = "FALLBACK"
        calibration_condition = "crossfit_full_fit"
        calibration_reason = (
            "Crossfit fallback full_fit disabled to avoid OOF label leakage."
        )
        calibrator = ProbaCalibrator(
            method="none",
            n_classes=int(np.asarray(proba_oof).shape[1]),
            models=None,
        )
        proba_oof_cal = normalize_proba(proba_oof)
    else:
        fallback = cal_cv_diag.get("fallback")
        if fallback not in (None, "none"):
            cal_cv_diag["status"] = "FALLBACK"
            cal_cv_diag["condition"] = str(fallback)
            cal_cv_diag["reason"] = str(fallback)
            calibration_status = "FALLBACK"
            calibration_condition = str(fallback)
            calibration_reason = str(fallback)
        else:
            cal_cv_diag.setdefault("status", "DONE")
            cal_cv_diag.setdefault("condition", "")
            cal_cv_diag.setdefault("reason", "")

    y_pred_oof_from_cal_proba = np.argmax(proba_oof_cal, axis=1)
    oof_df["y_pred_oof_from_cal_proba"] = y_pred_oof_from_cal_proba

    conf_candidates, conf_params = _build_confidence_candidates(
        oof_df=oof_df,
        proba_oof_cal=proba_oof_cal,
        y_pred_oof_raw=y_pred_oof_raw,
    )
    incorrect_oof = (y_pred_oof_raw != oof_df["y_true"].to_numpy()).astype(int)
    requested_conf_metric = options.aer_confidence_metric
    sel = select_confidence_metric(
        conf_candidates,
        incorrect_oof,
        groups=groups_arr,
        aer_kwargs=aer_kwargs,
        n_splits=min(5, int(max(2, options.aer_oof_folds))),
        seed=seed,
        metric=requested_conf_metric,
    )
    selected_conf_metric = sel.selected
    oof_df["conf_oof"] = conf_candidates[selected_conf_metric]
    selected_conf_params = conf_params.get(
        selected_conf_metric, {"kind": "identity", "params": {}}
    )

    selection_payload = {
        **sel.to_json(),
        "params": conf_params,
    }
    (m_meta / "confidence_metric_selection.json").write_text(
        json.dumps(selection_payload, indent=2),
        encoding="utf-8",
    )

    (m_meta / "proba_calibrator.json").write_text(
        json.dumps(
            {
                "calibrator": calibrator.to_json_dict(),
                "selection": cal_diag,
                "oof_calibration": cal_cv_diag,
                "status": calibration_status,
                "condition": calibration_condition,
                "reason": calibration_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return _OofSelectionResult(
        oof_df=oof_df,
        proba_oof_cal=proba_oof_cal,
        calibrator=calibrator,
        selected_conf_metric=selected_conf_metric,
        selected_conf_params=selected_conf_params,
        groups_arr=groups_arr,
        y_pred_oof_raw=y_pred_oof_raw,
        calibration_status=calibration_status,
        calibration_condition=calibration_condition,
        calibration_reason=calibration_reason,
    )

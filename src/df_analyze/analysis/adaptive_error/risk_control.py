# ref: Selective prediction / riskâ€“coverage framing https://arxiv.org/abs/1705.08500

from __future__ import annotations

from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd

from df_analyze.splitting import OmniKFold


def _fit_hens_calibrator(
    r_star_oof: np.ndarray,
    incorrect_oof: np.ndarray,
    r_star_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    r_oof = np.asarray(r_star_oof, dtype=float).ravel()
    inc = np.asarray(incorrect_oof, dtype=float).ravel()
    if r_oof.shape[0] != inc.shape[0]:
        raise ValueError("r_star_oof and incorrect_oof must have the same length")

    mask = np.isfinite(r_oof) & np.isfinite(inc)
    r_fit = r_oof[mask]
    inc_fit = inc[mask].astype(int)
    mean_err = float(np.mean(inc_fit)) if inc_fit.size else 0.0

    def _constant_fallback(reason: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        p_err_oof = np.full_like(r_oof, mean_err, dtype=float)
        r_test = np.asarray(r_star_test, dtype=float).ravel()
        p_err_test = np.full_like(r_test, mean_err, dtype=float)
        payload = {
            "method": "constant",
            "value": mean_err,
            "fallback_reason": reason,
            "status": "FALLBACK",
            "n_fit": int(inc_fit.size),
        }
        return p_err_oof, p_err_test, payload

    if r_fit.size == 0:
        return _constant_fallback("No OOF samples for hens.")

    if np.unique(r_fit).size < 2 or np.unique(inc_fit).size < 2:
        return _constant_fallback("Degenerate OOF data for hens.")

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(r_fit, inc_fit)
    r_oof_safe = np.clip(
        np.nan_to_num(r_oof, nan=mean_err, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    r_test = np.asarray(r_star_test, dtype=float).ravel()
    r_test_safe = np.clip(
        np.nan_to_num(r_test, nan=mean_err, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    p_err_oof = iso.predict(r_oof_safe)
    p_err_test = iso.predict(r_test_safe)
    payload = {
        "method": "isotonic",
        "x": iso.X_thresholds_.tolist(),
        "y": iso.y_thresholds_.tolist(),
        "fallback_reason": "",
        "status": "DONE",
        "n_fit": int(r_fit.size),
    }
    return p_err_oof, p_err_test, payload


def _crossfit_hens_error_probability(
    r_star: np.ndarray,
    incorrect: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression

    r = np.asarray(r_star, dtype=float).ravel()
    inc = np.asarray(incorrect, dtype=int).ravel()
    if r.shape[0] != inc.shape[0]:
        raise ValueError("r_star and incorrect must have the same length")

    g = None
    if groups is not None:
        g = np.asarray(groups).ravel()
        if g.shape[0] != r.shape[0]:
            raise ValueError("groups must have same length as r_star")

    mask = np.isfinite(r) & np.isfinite(inc)
    if g is not None:
        mask = mask & (~pd.isna(g))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.full_like(r, np.nan, dtype=float)

    r_valid = r[idx]
    inc_valid = inc[idx]
    g_valid = g[idx] if g is not None else None
    mean_err = float(np.mean(inc_valid)) if inc_valid.size else 0.0

    if (
        np.unique(r_valid).size < 2
        or np.unique(inc_valid).size < 2
        or int(n_splits) < 2
        or idx.size < int(n_splits)
    ):
        pred = np.full_like(r, np.nan, dtype=float)
        pred[idx] = mean_err
        return pred

    pred_valid = np.full_like(r_valid, np.nan, dtype=float)
    X_df = pd.DataFrame({"r_star": r_valid})
    y_ser = pd.Series(inc_valid)
    g_ser = pd.Series(g_valid) if g_valid is not None else None

    splitter = OmniKFold(
        n_splits=int(n_splits),
        is_classification=True,
        grouped=g_valid is not None,
        shuffle=True,
        seed=int(seed),
        warn_on_fallback=True,
        df_analyze_phase="adaptive_error_hens_crossfit",
    )
    splits, _ = splitter.split(X_df, y_ser, g_ser)

    for trn, val in splits:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(r_valid[trn], inc_valid[trn])
        pred_valid[val] = iso.predict(r_valid[val])

    pred = np.full_like(r, np.nan, dtype=float)
    pred[idx] = pred_valid
    return pred


def _select_risk_control_threshold(
    risk: np.ndarray,
    incorrect: np.ndarray,
    target_err: float,
    alpha: float,
    nmin: int,
) -> tuple[Optional[dict[str, Any]], str, dict[str, Any]]:
    from scipy.stats import beta as _beta_dist

    def _cp_upper(k: int, n: int, alpha: float) -> float:
        if n <= 0:
            return 1.0
        if k >= n:
            return 1.0
        return float(_beta_dist.ppf(1.0 - alpha, k + 1, n - k))

    risk_arr = np.asarray(risk, dtype=float).ravel()
    inc_arr = np.asarray(incorrect, dtype=float).ravel()
    mask = np.isfinite(risk_arr) & np.isfinite(inc_arr)
    risk_arr = risk_arr[mask]
    inc_arr = inc_arr[mask].astype(int)
    n_all = int(risk_arr.size)
    if n_all == 0:
        return (
            None,
            "No OOF samples.",
            {"bonferroni_m": 0, "alpha_adj": float(alpha), "oof_n_total": 0},
        )

    order = np.argsort(risk_arr)
    risk_sorted = risk_arr[order]
    inc_sorted = inc_arr[order]
    cum_err = np.cumsum(inc_sorted)
    last_idxs = np.r_[np.where(risk_sorted[1:] != risk_sorted[:-1])[0], n_all - 1]
    m = int(last_idxs.size)
    alpha_adj = float(alpha / m) if m > 0 else float(alpha)
    meta = {"bonferroni_m": m, "alpha_adj": alpha_adj, "oof_n_total": n_all}

    best = None
    for idx in last_idxs:
        n_acc = int(idx + 1)
        if n_acc < nmin:
            continue
        k_err = int(cum_err[idx])
        ub = _cp_upper(k_err, n_acc, alpha=alpha_adj)
        if ub <= target_err:
            best = {
                "t_star": float(risk_sorted[idx]),
                "oof_n": n_acc,
                "oof_k": k_err,
                "oof_error_rate": k_err / n_acc,
                "oof_cp_upper": ub,
                "oof_coverage": n_acc / n_all,
            }

    if best is None:
        return None, "No threshold satisfies the target error bound.", meta
    best.update(meta)
    return best, "", meta

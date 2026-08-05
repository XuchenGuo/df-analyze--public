from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Optional,
)

import numpy as np

from df_analyze.analysis.adaptive_error.plots import (
    build_reliability_table,
    expected_calibration_error,
)
from df_analyze.analysis.adaptive_error.report import (
    _write_csv,
    _write_not_available_csv,
    _write_not_available_json,
    _write_text,
)


def _write_test_error_metrics(
    *,
    metrics_path: Path,
    rel_csv_path: Path,
    e_hat_test: np.ndarray,
    incorrect_test: np.ndarray,
    n_test: int,
    n_bins: int,
    selected_conf_metric: str,
    used_conf_metric: str,
    selected_conf_params: dict[str, Any],
    drift_rate: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    valid_mask = np.isfinite(e_hat_test) & np.isfinite(incorrect_test)
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

    e_hat_valid = e_hat_test[valid_mask]
    incorrect_valid = incorrect_test[valid_mask].astype(int)
    global_error_test = float(np.mean(incorrect_valid))
    brier_error_test = float(np.mean((e_hat_valid - incorrect_valid) ** 2))

    rel_df = build_reliability_table(
        e_hat_valid,
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
        "confidence_metric_selected": selected_conf_metric,
        "confidence_metric_used": used_conf_metric,
        "confidence_metric_params": selected_conf_params,
        "pred_drift_rate": drift_rate,
        "global_error_test": global_error_test,
        "brier_error_test": brier_error_test,
        "ece_error_test": ece_error_test,
    }
    _write_text(metrics_path, json.dumps(metrics_payload, indent=2))

    return global_error_test, brier_error_test, ece_error_test

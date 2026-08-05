from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Optional,
)

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.aer import AdaptiveErrorCalculator
from df_analyze.analysis.adaptive_error.base_models_compute import (
    _crossfit_oof_risk,
)
from df_analyze.analysis.adaptive_error.base_models_helpers import _proba_to_list
from df_analyze.analysis.adaptive_error.report import (
    _write_csv,
    _write_not_available_csv,
    _write_not_available_parquet,
    _write_parquet,
    _write_text,
)


@dataclass
class _AerFitResult:
    aer: AdaptiveErrorCalculator
    e_hat_oof: Optional[np.ndarray]
    e_hat_oof_cv: Optional[np.ndarray]
    incorrect_oof: Optional[np.ndarray]
    oof_risk_source: str
    oof_risk_diag: Optional[dict[str, Any]]
    bins_df: Optional[pd.DataFrame]


def _fit_aer_stage(
    *,
    oof_df: pd.DataFrame,
    proba_oof_cal: np.ndarray,
    aer_kwargs: dict[str, Any],
    options,
    seed: int,
    groups_arr,
    m_meta: Path,
    m_tables: Path,
    m_preds: Path,
    no_preds: bool,
) -> _AerFitResult:
    aer = AdaptiveErrorCalculator(**aer_kwargs)
    aer.fit_from_oof(
        oof_df["conf_oof"],
        oof_df["y_pred_oof"],
        oof_df["y_true"],
    )

    conf_oof = oof_df["conf_oof"].to_numpy()
    oof_risk_source = "naive"
    oof_risk_diag: Optional[dict[str, Any]] = None
    e_hat_oof = aer.get_expected_error(conf_oof)
    incorrect_oof = (
        oof_df["y_pred_oof"].to_numpy() != oof_df["y_true"].to_numpy()
    ).astype(int)
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

    if oof_risk_diag is not None:
        _write_text(
            m_meta / "oof_risk_diag.json",
            json.dumps(oof_risk_diag, indent=2),
        )

    _write_text(
        m_meta / "confidence_to_expected_error_lookup.json",
        json.dumps(aer.to_json_dict(), indent=2),
    )
    _write_csv(
        aer.to_csv_dataframe(),
        m_tables / "confidence_to_expected_error_lookup.csv",
        index=False,
    )

    bins_df = aer.bin_stats_df(z=1.96)
    _write_csv(bins_df, m_tables / "oof_confidence_error_bins.csv", index=False)

    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_parquet(m_preds / "oof_per_sample.parquet", reason)
        _write_not_available_csv(m_preds / "oof_per_sample.csv", reason)
    else:
        oof_out = oof_df.copy()
        oof_out["confidence"] = np.asarray(oof_out["conf_oof"], dtype=float)
        aer_oof_arr = np.asarray(e_hat_oof, dtype=float).ravel()
        if aer_oof_arr.shape[0] == oof_out.shape[0]:
            oof_out["aer"] = aer_oof_arr
            oof_out["aer_pct"] = np.round(aer_oof_arr * 100.0, 1)
            oof_out["correct"] = (incorrect_oof == 0).astype(int)
        if e_hat_oof_cv is not None:
            aer_oof_cv_arr = np.asarray(e_hat_oof_cv, dtype=float).ravel()
            if aer_oof_cv_arr.shape[0] == oof_out.shape[0]:
                oof_out["aer_cv"] = aer_oof_cv_arr
                oof_out["aer_cv_pct"] = np.round(aer_oof_cv_arr * 100.0, 1)
        oof_out["proba_calibrated"] = _proba_to_list(proba_oof_cal)
        _write_parquet(
            oof_out,
            m_preds / "oof_per_sample.parquet",
            index=False,
        )
        oof_csv = oof_out.drop(columns=["proba_calibrated"], errors="ignore")
        _write_csv(oof_csv, m_preds / "oof_per_sample.csv", index=False)

    return _AerFitResult(
        aer=aer,
        e_hat_oof=e_hat_oof,
        e_hat_oof_cv=e_hat_oof_cv,
        incorrect_oof=incorrect_oof,
        oof_risk_source=oof_risk_source,
        oof_risk_diag=oof_risk_diag,
        bins_df=bins_df,
    )

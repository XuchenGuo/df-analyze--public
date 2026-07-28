from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from df_analyze.analysis.adaptive_error.base_models_selection import (
    _select_unique_models,
    _slug_from_options,
)
from df_analyze.analysis.error_consistency.runner import detail_output_dir


REPORT_COLUMNS = [
    "target",
    "model",
    "selection",
    "embed_selector",
    "row_id",
    "correct",
    "aer",
    "error_rate",
    "instability",
    "aer_threshold",
    "ec_recurrence_threshold",
    "threshold_basis",
    "high_aer",
    "high_error_recurrence",
    "risk_stability_quadrant",
]
SUMMARY_COLUMNS = [
    "target",
    "model",
    "selection",
    "embed_selector",
    "risk_stability_quadrant",
    "n_samples",
    "sample_fraction",
    "observed_error_rate",
    "mean_aer",
    "mean_ec_error_rate",
    "mean_ec_instability",
    "aer_threshold",
    "ec_recurrence_threshold",
    "threshold_basis",
]
SKIPPED_COLUMNS = ["target", "model", "selection", "reason"]


def risk_stability_summary(report: DataFrame) -> DataFrame:
    if report.empty:
        return DataFrame(columns=SUMMARY_COLUMNS)
    report = report.copy()
    rows = []
    identity = [
        col
        for col in ["target", "model", "selection", "embed_selector"]
        if col in report.columns
    ]
    report["_identity_total"] = (
        report.groupby(identity, dropna=False)["risk_stability_quadrant"].transform(
            "size"
        )
        if identity
        else len(report)
    )
    group_cols = [*identity, "risk_stability_quadrant"]
    for group_key, group in report.groupby(group_cols, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, key_values))
        aer_threshold = (
            float(group["aer_threshold"].iloc[0]) if "aer_threshold" in group else np.nan
        )
        recurrence_threshold = (
            float(group["ec_recurrence_threshold"].iloc[0])
            if "ec_recurrence_threshold" in group
            else np.nan
        )
        threshold_basis = (
            str(group["threshold_basis"].iloc[0])
            if "threshold_basis" in group
            else "not_recorded"
        )
        row.update(
            {
                "n_samples": len(group),
                "sample_fraction": len(group) / int(group["_identity_total"].iloc[0]),
                "observed_error_rate": 1.0 - group["correct"].astype(float).mean(),
                "mean_aer": group["aer"].astype(float).mean(),
                "mean_ec_error_rate": group["error_rate"].astype(float).mean(),
                "mean_ec_instability": group["instability"].astype(float).mean(),
                "aer_threshold": aer_threshold,
                "ec_recurrence_threshold": recurrence_threshold,
                "threshold_basis": threshold_basis,
            }
        )
        rows.append(row)
    return DataFrame(rows)


def write_risk_stability_report(
    target: str,
    eval_results,
    options,
    aer_base_dir: Path,
    ec_base_dir: Path,
) -> DataFrame:
    tables = aer_base_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    skipped = []
    if not getattr(eval_results, "is_classification", False):
        report = DataFrame(columns=REPORT_COLUMNS)
        skipped.append(
            {
                "target": str(target),
                "model": "",
                "selection": "",
                "reason": "risk stability is available only for classification",
            }
        )
        report.attrs["skipped_configurations"] = skipped
        report.to_csv(tables / "risk_stability_report.csv", index=False)
        risk_stability_summary(report).to_csv(
            tables / "risk_stability_summary.csv", index=False
        )
        DataFrame(skipped, columns=SKIPPED_COLUMNS).to_csv(
            tables / "risk_stability_skipped.csv", index=False
        )
        return report
    selected = _select_unique_models(
        eval_results.results,
        metric_pref=options.htune_cls_metric,
        options=options,
        top_k=0,
    )
    selected = [
        result
        for result in selected
        if _slug_from_options(result.model_cls, options) != "dummy"
    ]
    top_k = int(getattr(options, "aer_top_k", 0) or 0)
    if top_k > 0:
        selected = selected[:top_k]
    if not selected:
        skipped.append(
            {
                "target": str(target),
                "model": "",
                "selection": "",
                "reason": "no eligible models were available",
            }
        )
    reports = []
    for result in selected:
        slug = _slug_from_options(result.model_cls, options)
        if slug == "dummy":
            continue
        aer_path = aer_base_dir / "models" / slug / "predictions" / "test_per_sample.csv"
        selector = getattr(result, "embed_select_model", None)
        embed = "none" if selector is None else str(getattr(selector, "value", selector))
        model_name = str(getattr(result.model, "shortname", slug))
        ec_path = detail_output_dir(
            ec_base_dir,
            {
                "target": str(target),
                "model": model_name,
                "selection": str(result.selection),
                "embed_selector": embed,
            },
        ) / "sample_diagnostics.csv"
        if not aer_path.exists() or not ec_path.exists():
            missing = []
            if not aer_path.exists():
                missing.append("AER predictions")
            if not ec_path.exists():
                missing.append("EC diagnostics")
            skipped.append(
                {
                    "target": str(target),
                    "model": model_name,
                    "selection": str(result.selection),
                    "reason": f"missing {' and '.join(missing)}",
                }
            )
            continue
        aer = pd.read_csv(aer_path)
        ec = pd.read_csv(ec_path)
        required_aer = {"row_id", "correct", "aer"}
        required_ec = {"row_id", "error_rate", "instability"}
        if not required_aer.issubset(aer) or not required_ec.issubset(ec):
            skipped.append(
                {
                    "target": str(target),
                    "model": model_name,
                    "selection": str(result.selection),
                    "reason": "required AER or EC columns are missing",
                }
            )
            continue
        joined = aer.merge(ec, on="row_id", how="inner", validate="one_to_one")
        if joined.empty:
            skipped.append(
                {
                    "target": str(target),
                    "model": model_name,
                    "selection": str(result.selection),
                    "reason": "AER and EC outputs have no matching holdout rows",
                }
            )
            continue
        aer_threshold = float(getattr(options, "aer_target_error", 0.05))
        recurrence_threshold = float(getattr(options, "ec_recurrence_threshold", 0.5))
        high_aer = joined["aer"].astype(float) >= aer_threshold
        high_recurrence = joined["error_rate"].astype(float) >= recurrence_threshold
        joined["aer_threshold"] = aer_threshold
        joined["ec_recurrence_threshold"] = recurrence_threshold
        joined["threshold_basis"] = "user_configurable_heuristic"
        joined["high_aer"] = high_aer
        joined["high_error_recurrence"] = high_recurrence
        joined["risk_stability_quadrant"] = np.select(
            [
                high_aer & high_recurrence,
                high_aer & ~high_recurrence,
                ~high_aer & high_recurrence,
            ],
            [
                "high_aer_high_recurrence",
                "high_aer_low_recurrence",
                "low_aer_high_recurrence",
            ],
            default="low_aer_low_recurrence",
        )
        joined.insert(0, "target", str(target))
        joined.insert(1, "model", model_name)
        joined.insert(2, "selection", str(result.selection))
        joined.insert(3, "embed_selector", embed)
        reports.append(joined)

    report = (
        pd.concat(reports, ignore_index=True)
        if reports
        else DataFrame(columns=REPORT_COLUMNS)
    )
    report.attrs["skipped_configurations"] = skipped
    report.to_csv(tables / "risk_stability_report.csv", index=False)
    risk_stability_summary(report).to_csv(
        tables / "risk_stability_summary.csv", index=False
    )
    DataFrame(skipped, columns=SKIPPED_COLUMNS).to_csv(
        tables / "risk_stability_skipped.csv", index=False
    )
    return report

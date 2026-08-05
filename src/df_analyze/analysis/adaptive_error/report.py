from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Optional,
)
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.plots import (
    plot_compare_confidence_vs_error,
    plot_placeholder,
)
from df_analyze.saving import windows_io_path


def _write_text(path: Path, text: str) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    windows_io_path(path).write_text(text, encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(windows_io_path(path), **kwargs)


def _write_parquet(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    df.to_parquet(windows_io_path(path), **kwargs)


def _write_markdown(path: Path, lines: list[str]) -> None:
    _write_text(path, "\n".join(lines))


def _df_to_markdown(df: pd.DataFrame, **kwargs: Any) -> str:
    with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
        return df.to_markdown(**kwargs)


def _write_sanity_checks(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2))


def _write_not_available_json(
    path: Path, reason: str, extra: Optional[dict[str, Any]] = None
) -> None:
    payload: dict[str, Any] = dict(extra or {})
    payload["available"] = False
    payload["status"] = "NOT_AVAILABLE"
    payload["reason"] = str(reason)
    _write_text(path, json.dumps(payload, indent=2))


def _write_not_available_csv(path: Path, reason: str) -> None:
    df = pd.DataFrame(
        [
            {
                "status": "NOT_AVAILABLE",
                "available": False,
                "reason": str(reason),
            }
        ]
    )
    _write_csv(df, path, index=False)


def _write_not_available_parquet(path: Path, reason: str) -> None:
    df = pd.DataFrame(
        [
            {
                "status": "NOT_AVAILABLE",
                "available": False,
                "reason": str(reason),
            }
        ]
    )
    _write_parquet(df, path, index=False)


def _write_not_available_md(path: Path, title: str, reason: str) -> None:
    lines = [
        f"# {title}",
        "",
        "NOT_AVAILABLE",
        "",
        f"Reason: {reason}",
    ]
    _write_text(path, "\n".join(lines))


def _write_not_available_png(path: Path, title: str, reason: str) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    message = str(reason).strip() or "Not available"
    plot_placeholder(windows_io_path(path), title=title, message=message)


def _display_label(slug: str) -> str:
    s = str(slug).lower().strip()
    mapping = {
        "knn": "KNN",
        "lgbm": "LGBM",
        "rf": "RF",
        "sgd": "SGD",
        "lr": "LR",
        "mlp": "MLP",
        "svm": "SVM",
        "gandalf": "GANDALF",
        "dummy": "Dummy",
    }
    if s in mapping:
        return mapping[s]
    return s.upper()


def _coverage_summary(
    curve: pd.DataFrame,
    targets: list[float],
    risk: Optional[np.ndarray] = None,
    correct: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame(
            columns=[
                "target_coverage",
                "coverage",
                "selective_accuracy",
                "threshold",
                "n_accepted",
            ]
        )
    rows = []
    for t in targets:
        idx = (curve["coverage"] - t).abs().idxmin()
        row = curve.loc[idx].to_dict()
        row["target_coverage"] = t
        rows.append(row)

    out = pd.DataFrame(rows)
    if risk is None or correct is None:
        cols = [
            "target_coverage",
            "coverage",
            "selective_accuracy",
            "threshold",
            "n_accepted",
        ]
        return out[cols] if set(cols).issubset(out.columns) else out

    risk_arr = np.asarray(risk, dtype=float).ravel()
    correct_arr = np.asarray(correct).ravel()
    if risk_arr.size != correct_arr.size:
        warn("Coverage summary inputs have mismatched lengths; using curve values.")
        cols = [
            "target_coverage",
            "coverage",
            "selective_accuracy",
            "threshold",
            "n_accepted",
        ]
        return out[cols] if set(cols).issubset(out.columns) else out

    mask = np.isfinite(risk_arr)
    risk_arr = np.clip(risk_arr[mask], 0.0, 1.0)
    correct_arr = correct_arr[mask].astype(int)
    n_total = int(risk_arr.size)
    if n_total == 0:
        cols = [
            "target_coverage",
            "coverage",
            "selective_accuracy",
            "threshold",
            "n_accepted",
        ]
        return out[cols] if set(cols).issubset(out.columns) else out

    recomputed_rows = []
    for row in rows:
        threshold = float(row["threshold"])
        accept = risk_arr <= threshold
        n_accepted = int(accept.sum())
        coverage = n_accepted / n_total if n_total else 0.0
        selective_accuracy = (
            float(correct_arr[accept].mean()) if n_accepted > 0 else float("nan")
        )
        row["n_accepted"] = n_accepted
        row["coverage"] = coverage
        row["selective_accuracy"] = selective_accuracy
        recomputed_rows.append(row)

    out = pd.DataFrame(recomputed_rows)
    cols = [
        "target_coverage",
        "coverage",
        "selective_accuracy",
        "threshold",
        "n_accepted",
    ]
    return out[cols] if set(cols).issubset(out.columns) else out


def _curve_for_auc(curve_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = curve_df.copy()
    df = df.dropna(subset=["coverage", "selective_accuracy"]).sort_values("coverage")
    if df.empty:
        return df, float("nan")

    if float(df["coverage"].iloc[0]) > 0.0:
        first_acc = float(df["selective_accuracy"].iloc[0])
        first_thr = (
            float(df["threshold"].iloc[0]) if "threshold" in df.columns else float("nan")
        )
        df = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "threshold": first_thr,
                            "coverage": 0.0,
                            "n_accepted": 0,
                            "selective_accuracy": first_acc,
                        }
                    ]
                ),
                df,
            ],
            ignore_index=True,
        )

    auc_val = float(
        np.trapz(df["selective_accuracy"].to_numpy(), df["coverage"].to_numpy())
    )
    return df, auc_val


def _coverage_summary_with_auc(
    curve_df: pd.DataFrame,
    targets: list[float],
) -> tuple[pd.DataFrame, float]:
    cols = [
        "target_coverage",
        "coverage",
        "selective_accuracy",
        "n_accepted",
        "reject_rate",
        "threshold",
        "auc_coverage_accuracy",
    ]
    if curve_df.empty:
        return pd.DataFrame(columns=cols), float("nan")

    df, auc_val = _curve_for_auc(curve_df)
    if df.empty:
        return pd.DataFrame(columns=cols), auc_val

    cov_vals = df["coverage"].to_numpy()
    rows = []
    for t in targets:
        idx = int(np.abs(cov_vals - t).argmin())
        r = df.iloc[idx]
        coverage = float(r["coverage"])
        rows.append(
            {
                "target_coverage": float(t),
                "coverage": coverage,
                "selective_accuracy": float(r["selective_accuracy"]),
                "n_accepted": int(r["n_accepted"]) if "n_accepted" in r else 0,
                "reject_rate": float(1.0 - coverage),
                "threshold": float(r["threshold"]) if "threshold" in r else float("nan"),
                "auc_coverage_accuracy": auc_val,
            }
        )

    return pd.DataFrame(rows), auc_val


def _coverage_summary_by_accuracy(
    curve_df: pd.DataFrame,
    target_accuracies: list[float],
) -> tuple[pd.DataFrame, float]:
    cols = [
        "target_accuracy",
        "coverage",
        "selective_accuracy",
        "n_accepted",
        "reject_rate",
        "threshold",
        "auc_coverage_accuracy",
        "status",
        "note",
    ]
    if curve_df.empty:
        return pd.DataFrame(columns=cols), float("nan")

    df, auc_val = _curve_for_auc(curve_df)
    if df.empty:
        return pd.DataFrame(columns=cols), auc_val

    rows = []
    for a in target_accuracies:
        a = float(a)
        eligible = df[df["selective_accuracy"] >= a]
        if eligible.empty:
            best_acc = float(df["selective_accuracy"].max())
            best_rows = df[df["selective_accuracy"] >= best_acc - 1e-12]
            r = best_rows.iloc[int(np.argmax(best_rows["coverage"].to_numpy()))]
            coverage = float(r["coverage"])
            rows.append(
                {
                    "target_accuracy": a,
                    "coverage": coverage,
                    "selective_accuracy": float(r["selective_accuracy"]),
                    "n_accepted": int(r["n_accepted"]) if "n_accepted" in r else 0,
                    "reject_rate": float(1.0 - coverage),
                    "threshold": float(r["threshold"])
                    if "threshold" in r
                    else float("nan"),
                    "auc_coverage_accuracy": auc_val,
                    "status": "below_target",
                    "note": f"Max achievable selective_accuracy={best_acc:.4f} < target",
                }
            )
            continue

        r = eligible.iloc[int(np.argmax(eligible["coverage"].to_numpy()))]
        coverage = float(r["coverage"])
        rows.append(
            {
                "target_accuracy": a,
                "coverage": coverage,
                "selective_accuracy": float(r["selective_accuracy"]),
                "n_accepted": int(r["n_accepted"]) if "n_accepted" in r else 0,
                "reject_rate": float(1.0 - coverage),
                "threshold": float(r["threshold"]) if "threshold" in r else float("nan"),
                "auc_coverage_accuracy": auc_val,
                "status": "ok",
                "note": "",
            }
        )

    return pd.DataFrame(rows), auc_val


def write_cross_model_summaries(
    *,
    plots_dir: Path,
    tables_dir: Path,
    preds_dir: Path,
    base_test: pd.DataFrame,
    slugs: list[str],
    compare_tests: dict[str, Optional[pd.DataFrame]],
    compare_bins: dict[str, pd.DataFrame],
    compare_test_bins: dict[str, pd.DataFrame],
    compare_metrics: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    no_preds: bool,
) -> None:
    _write_csv(
        pd.DataFrame(model_rows),
        tables_dir / "models_ranked.csv",
        index=False,
    )

    test_multi = base_test.copy()
    for slug in slugs:
        test_df = compare_tests.get(slug)
        if test_df is None or test_df.empty:
            test_multi[f"y_pred__{slug}"] = np.nan
            test_multi[f"confidence__{slug}"] = np.nan
            test_multi[f"aer__{slug}"] = np.nan
            test_multi[f"correct__{slug}"] = np.nan
            continue

        use_cols = ["row_id", "y_pred", "confidence", "aer", "correct"]
        rename = {
            "y_pred": f"y_pred__{slug}",
            "confidence": f"confidence__{slug}",
            "aer": f"aer__{slug}",
            "correct": f"correct__{slug}",
        }
        merge_df = test_df[use_cols].rename(columns=rename)
        test_multi = test_multi.merge(merge_df, on="row_id", how="left")

    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_parquet(
            preds_dir / "test_per_sample_multi_model.parquet", reason
        )
        _write_not_available_csv(preds_dir / "test_per_sample_multi_model.csv", reason)
    else:
        _write_parquet(
            test_multi,
            preds_dir / "test_per_sample_multi_model.parquet",
            index=False,
        )
        _write_csv(
            test_multi,
            preds_dir / "test_per_sample_multi_model.csv",
            index=False,
        )

    compare_bins_for_plot: dict[str, pd.DataFrame] = {}
    for slug in slugs:
        bins_df = compare_test_bins.get(slug)
        if bins_df is None or bins_df.empty:
            bins_df = compare_bins.get(slug)
        if bins_df is None or bins_df.empty:
            continue
        compare_bins_for_plot[_display_label(slug)] = bins_df
    plot_compare_confidence_vs_error(
        compare_bins_for_plot,
        plots_dir / "confidence_vs_expected_error_compare.png",
        title="Confidence vs adaptive expected error rate",
    )

    _write_csv(
        pd.DataFrame(compare_metrics),
        tables_dir / "aer_metrics_by_model.csv",
        index=False,
    )


def write_run_config(
    *,
    base_dir: Path,
    options,
    seed: int,
    best_slug: Optional[str],
    model_run_info: list[dict[str, Any]],
    no_preds: bool,
    strategy_labels: dict[str, str],
    strategy_params: dict[str, Any],
) -> None:
    executed_strategies = (
        list(strategy_labels.values()) if options.aer_ensemble else []
    )
    aer_smooth = bool(options.aer_smooth)
    aer_enforce = bool(options.aer_monotonic)
    aer_adaptive = bool(options.aer_adaptive_binning)
    aer_confidence_metric = options.aer_confidence_metric
    if aer_confidence_metric is not None:
        aer_confidence_metric = str(aer_confidence_metric)
    aer_ensemble_strategies = getattr(options, "aer_ensemble_strategies", None)
    run_config = {
        "decision_label_source": "df-analyze tuned predictions",
        "best_model_slug": best_slug,
        "best_model_selection_rule": "highest_cv_score",
        "confidence_source": (
            "OOF-calibrated probabilities anchored to y_pred "
            "(fallback to uncalibrated if calibration is disabled)"
        ),
        "no_preds": bool(no_preds),
        "oof_folds": int(options.aer_oof_folds),
        "seed": int(seed),
        "aer_bins": int(options.aer_bins),
        "aer_min_bin_count": int(options.aer_min_bin_count),
        "aer_prior_strength": float(options.aer_prior_strength),
        "aer_smooth": aer_smooth,
        "aer_monotonic": aer_enforce,
        "aer_adaptive_binning": aer_adaptive,
        "aer_confidence_metric": aer_confidence_metric,
        "aer_target_error": float(options.aer_target_error),
        "aer_alpha": float(options.aer_alpha),
        "aer_nmin": int(options.aer_nmin),
        "ensemble_enabled": bool(options.aer_ensemble),
        "aer_ensemble_strategies": aer_ensemble_strategies,
        "ensemble_strategies_executed": executed_strategies,
        "ensemble_strategy_params": strategy_params,
        "models": model_run_info,
    }
    _write_text(base_dir / "run_config.json", json.dumps(run_config, indent=2))

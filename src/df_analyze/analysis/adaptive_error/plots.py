# ref: Risk–coverage curves for selective prediction: https://arxiv.org/abs/1705.08500
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from df_analyze.analysis.adaptive_error.stats import _wilson_interval
from df_analyze.saving import windows_io_path


def _save_figure(fig, outpath: Path) -> None:
    windows_io_path(outpath.parent).mkdir(parents=True, exist_ok=True)
    fig.savefig(windows_io_path(outpath), dpi=150)


def build_confidence_error_bins_table(
    conf: np.ndarray,
    incorrect: np.ndarray,
    expected_error: np.ndarray,
    edges: np.ndarray,
    *,
    z_score: float = 1.96,
) -> pd.DataFrame:
    conf = np.asarray(conf, dtype=float).ravel()
    incorrect = np.asarray(incorrect).ravel()
    expected_error = np.asarray(expected_error, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float).ravel()

    if conf.size != incorrect.size or conf.size != expected_error.size:
        raise ValueError("conf, incorrect, and expected_error must have the same length")

    mask = np.isfinite(conf) & np.isfinite(expected_error)
    if mask.sum() == 0 or edges.size < 2:
        return pd.DataFrame(
            columns=[
                "bin_left",
                "bin_right",
                "bin_center",
                "count",
                "error_count",
                "raw_error",
                "expected_error",
                "wilson_low",
                "wilson_high",
            ]
        )

    conf = np.clip(conf[mask], 0.0, 1.0)
    incorrect = incorrect[mask].astype(int)
    expected_error = expected_error[mask]

    rows: list[dict[str, Union[float, int]]] = []
    n_bins = int(edges.size - 1)
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < n_bins - 1:
            in_bin = (conf >= lo) & (conf < hi)
        else:
            in_bin = (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        err_cnt = int(incorrect[in_bin].sum()) if n > 0 else 0
        raw_error = float(err_cnt / n) if n > 0 else float("nan")
        exp = float(np.mean(expected_error[in_bin])) if n > 0 else float("nan")
        lo_ci, hi_ci = _wilson_interval(err_cnt, n, z=z_score)
        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "bin_center": (lo + hi) / 2.0,
                "count": n,
                "error_count": err_cnt,
                "raw_error": raw_error,
                "expected_error": exp,
                "wilson_low": lo_ci,
                "wilson_high": hi_ci,
            }
        )
    return pd.DataFrame(rows)


def merge_small_bins(
    bins_df: pd.DataFrame,
    *,
    min_count: int = 10,
    z_score: float = 1.96,
) -> pd.DataFrame:
    if bins_df is None or bins_df.empty:
        return bins_df

    df = bins_df.copy()
    if "count" not in df.columns:
        return df

    df = df.loc[df["count"].fillna(0).astype(int) > 0].copy()
    if df.empty:
        return df

    df = df.sort_values(["bin_left", "bin_right"]).reset_index(drop=True)

    merged_rows: list[dict[str, Union[float, int]]] = []
    cur_lo: Optional[float] = None
    cur_hi: Optional[float] = None
    cur_n: int = 0
    cur_err: int = 0
    cur_exp_sum: float = 0.0

    def _flush() -> None:
        nonlocal cur_lo, cur_hi, cur_n, cur_err, cur_exp_sum
        if cur_lo is None or cur_hi is None or cur_n <= 0:
            return
        raw_error = float(cur_err / cur_n)
        expected_error = float(cur_exp_sum / cur_n)
        lo_ci, hi_ci = _wilson_interval(cur_err, cur_n, z=z_score)
        merged_rows.append(
            {
                "bin_left": float(cur_lo),
                "bin_right": float(cur_hi),
                "bin_center": float((cur_lo + cur_hi) / 2.0),
                "count": int(cur_n),
                "error_count": int(cur_err),
                "raw_error": raw_error,
                "expected_error": expected_error,
                "wilson_low": lo_ci,
                "wilson_high": hi_ci,
            }
        )
        cur_lo, cur_hi, cur_n, cur_err, cur_exp_sum = None, None, 0, 0, 0.0

    for _, r in df.iterrows():
        lo = float(r["bin_left"])
        hi = float(r["bin_right"])
        n = int(r["count"])
        err = int(r.get("error_count", 0))
        exp = float(r.get("expected_error", np.nan))
        if not np.isfinite(exp):
            exp = 0.0

        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
            cur_n, cur_err = n, err
            cur_exp_sum = exp * n
            continue

        if cur_n < int(min_count):
            cur_hi = hi
            cur_n += n
            cur_err += err
            cur_exp_sum += exp * n
        else:
            _flush()
            cur_lo, cur_hi = lo, hi
            cur_n, cur_err = n, err
            cur_exp_sum = exp * n
    _flush()

    if len(merged_rows) >= 2 and merged_rows[-1]["count"] < int(min_count):
        last = merged_rows.pop()
        prev = merged_rows.pop()
        lo = float(prev["bin_left"])
        hi = float(last["bin_right"])
        n = int(prev["count"]) + int(last["count"])
        err = int(prev["error_count"]) + int(last["error_count"])
        exp_sum = float(prev["expected_error"]) * int(prev["count"]) + float(
            last["expected_error"]
        ) * int(last["count"])
        raw_error = float(err / n)
        expected_error = float(exp_sum / n)
        lo_ci, hi_ci = _wilson_interval(err, n, z=z_score)
        merged_rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "bin_center": float((lo + hi) / 2.0),
                "count": n,
                "error_count": err,
                "raw_error": raw_error,
                "expected_error": expected_error,
                "wilson_low": lo_ci,
                "wilson_high": hi_ci,
            }
        )

    return pd.DataFrame(merged_rows)


def plot_placeholder(outpath: Path, title: str, message: Optional[str] = None) -> None:
    fig, ax = plt.subplots()
    ax.axis("off")
    lines = [title]
    if message:
        lines.append(message)
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save_figure(fig, outpath)
    plt.close(fig)


def plot_confidence_vs_error(
    bins_df: pd.DataFrame,
    outpath: Path,
    *,
    title: str = "Confidence vs Expected Error",
    annotate_counts: bool = False,
) -> None:
    df = bins_df.copy()
    req = {
        "bin_center",
        "raw_error",
        "expected_error",
        "wilson_low",
        "wilson_high",
        "count",
    }
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"bins_df is missing required columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df_expected = df.dropna(subset=["bin_center", "expected_error"])
    df_expected = df_expected[df_expected["count"].fillna(0).astype(float) > 0]
    df_emp = df.copy()
    df_emp = df_emp[df_emp["count"].fillna(0).astype(float) > 0]
    df_emp = df_emp.dropna(
        subset=["bin_center", "raw_error", "wilson_low", "wilson_high"]
    )
    if df_expected.empty and df_emp.empty:
        warn("No non-empty bins available for confidence vs error plot.")
        plot_placeholder(outpath, title=title, message="Not available")
        return

    df_emp = df_emp.sort_values("bin_center")
    df_expected = df_expected.sort_values("bin_center")
    fig, ax = plt.subplots(figsize=(8, 5))

    if not df_emp.empty:
        x_emp = df_emp["bin_center"].to_numpy(dtype=float)
        y_emp = df_emp["raw_error"].to_numpy(dtype=float)
        lo = df_emp["wilson_low"].to_numpy(dtype=float)
        hi = df_emp["wilson_high"].to_numpy(dtype=float)
        ax.fill_between(
            x_emp,
            lo,
            hi,
            alpha=0.20,
            label="95% CI (Wilson)",
        )

    if not df_expected.empty:
        x_exp = df_expected["bin_center"].to_numpy(dtype=float)
        y_smooth = df_expected["expected_error"].to_numpy(dtype=float)
        ax.plot(
            x_exp,
            y_smooth,
            marker="o",
            markersize=4,
            linewidth=2.0,
            label="Adaptive expected error (aER)",
        )

    ax.set_xlabel("Predictive confidence")
    ax.set_ylabel("Expected error rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    if annotate_counts and not df_emp.empty:
        if int(df_emp.shape[0]) <= 12:
            n = df_emp["count"].to_numpy(dtype=int)
            y_max = float(ax.get_ylim()[1])
            for xi, yi, hi_i, ni in zip(x_emp, y_emp, hi, n):
                y_text = float(max(yi, hi_i))
                dy = 6
                va = "bottom"
                if np.isfinite(y_text) and y_text >= 0.92 * y_max:
                    dy = -12
                    va = "top"

                ax.annotate(
                    str(int(ni)),
                    (float(xi), y_text),
                    textcoords="offset points",
                    xytext=(0, dy),
                    ha="center",
                    va=va,
                    fontsize=8,
                    alpha=0.9,
                )
        else:
            warn("Skipping count annotations in confidence-vs-error plot: too many bins")

    fig.tight_layout()
    _save_figure(fig, outpath)
    plt.close(fig)


def plot_compare_confidence_vs_error(
    bins_by_model: dict[str, pd.DataFrame],
    outpath: Path,
    *,
    title: str = "Confidence vs expected error",
) -> None:
    if not bins_by_model:
        plot_placeholder(outpath, title=title, message="Not available")
        return

    x_grid = np.linspace(0.0, 1.0, 21)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for label, bins_df in bins_by_model.items():
        df = bins_df.copy()
        if "bin_center" not in df.columns or "expected_error" not in df.columns:
            continue
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["bin_center", "expected_error"])
        if "count" in df.columns:
            df = df[df["count"].fillna(0).astype(float) > 0]
        if df.empty:
            continue
        df = df.sort_values("bin_center")
        x = df["bin_center"].to_numpy(dtype=float)
        y = df["expected_error"].to_numpy(dtype=float)
        if x.size < 1:
            continue

        if x.size == 1:
            y_grid = np.full_like(x_grid, float(y[0]), dtype=float)
        else:
            y_grid = np.interp(
                x_grid,
                x,
                y,
                left=float(y[0]),
                right=float(y[-1]),
            )
        ax.plot(
            x_grid,
            y_grid,
            marker="o",
            markersize=3,
            linewidth=2.0,
            label=label,
        )
        has_data = True

    if not has_data:
        plot_placeholder(outpath, title=title, message="Not available")
        return

    ax.set_xlabel("Predictive confidence")
    ax.set_ylabel("Adaptive expected error rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, outpath)
    plt.close(fig)


def build_coverage_accuracy_curve_rank(
    risk: np.ndarray,
    correct: np.ndarray,
) -> pd.DataFrame:
    risk = np.asarray(risk, dtype=float).ravel()
    correct = np.asarray(correct).ravel().astype(int)
    if risk.shape[0] != correct.shape[0]:
        raise ValueError("risk and correct must have the same length")

    mask = np.isfinite(risk)
    risk = np.clip(risk[mask], 0.0, 1.0)
    correct = correct[mask].astype(int)

    n = int(risk.size)
    if n == 0:
        return pd.DataFrame(
            columns=["coverage", "selective_accuracy", "threshold", "n_accepted"]
        )

    order = np.argsort(risk, kind="mergesort")
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    n_accepted = np.arange(1, n + 1)
    coverage = n_accepted / n
    selective_accuracy = cum_correct / n_accepted

    return pd.DataFrame(
        {
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "threshold": risk[order],
            "n_accepted": n_accepted,
        }
    )


def plot_coverage_vs_accuracy(curve_df: pd.DataFrame, outpath: Path) -> None:
    if curve_df.empty:
        plot_placeholder(outpath, title="Coverage vs accuracy", message="Not available")
        return
    fig, ax = plt.subplots()
    ax.plot(
        curve_df["coverage"], curve_df["selective_accuracy"], marker="o", markersize=2
    )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy (accepted predictions)")
    ax.set_title("Coverage vs accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    _save_figure(fig, outpath)
    plt.close(fig)


def plot_coverage_vs_accuracy_overlay(
    curves: dict[str, pd.DataFrame],
    outpath: Path,
    *,
    title: str = "Coverage vs accuracy (overlay)",
) -> None:
    if not curves:
        plot_placeholder(outpath, title=title, message="Not available")
        return

    fig, ax = plt.subplots()
    has_data = False
    min_acc = None
    max_acc = None
    n_labels = 0
    for label, curve_df in curves.items():
        if curve_df is None or curve_df.empty:
            continue
        if (
            "coverage" not in curve_df.columns
            or "selective_accuracy" not in curve_df.columns
        ):
            continue
        df = curve_df.dropna(subset=["coverage", "selective_accuracy"])
        if df.empty:
            continue
        ax.plot(
            df["coverage"],
            df["selective_accuracy"],
            marker="o",
            markersize=2,
            label=str(label),
        )
        acc_min = float(df["selective_accuracy"].min())
        acc_max = float(df["selective_accuracy"].max())
        min_acc = acc_min if min_acc is None else min(min_acc, acc_min)
        max_acc = acc_max if max_acc is None else max(max_acc, acc_max)
        n_labels += 1
        has_data = True

    if not has_data:
        plot_placeholder(outpath, title=title, message="Not available")
        return

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy (accepted predictions)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    if (
        min_acc is None
        or max_acc is None
        or not np.isfinite(min_acc)
        or not np.isfinite(max_acc)
    ):
        ax.set_ylim(0.0, 1.0)
    else:
        if min_acc >= 0.94:
            ax.set_ylim(0.94, 1.0)
        elif min_acc >= 0.90:
            ax.set_ylim(0.90, 1.0)
        else:
            ymin = max(0.0, min_acc - 0.02)
            ymax = min(1.0, max_acc + 0.002)
            if ymax <= ymin:
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylim(ymin, ymax)
    legend_kwargs = {"loc": "best", "fontsize": 8}
    if n_labels > 8:
        legend_kwargs["ncol"] = 2
        legend_kwargs["fontsize"] = 7
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    _save_figure(fig, outpath)
    plt.close(fig)


def build_reliability_table(
    p_pred: np.ndarray,
    incorrect_indicator: np.ndarray,
    n_bins: int = 20,
    strategy: str = "quantile",
    z_score: float = 1.96,
) -> pd.DataFrame:
    p = np.asarray(p_pred, dtype=float).ravel()
    inc = np.asarray(incorrect_indicator, dtype=float).ravel()
    if p.size != inc.size:
        raise ValueError("p_pred and incorrect_indicator must have the same length")

    mask = np.isfinite(p) & np.isfinite(inc)
    p = np.clip(p[mask], 0.0, 1.0)
    inc = inc[mask].astype(int)

    if p.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_left",
                "bin_right",
                "bin_center",
                "count",
                "error_count",
                "error_rate",
                "p_pred_mean",
                "wilson_low",
                "wilson_high",
            ]
        )

    n_bins = max(1, int(n_bins))
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0], dtype=float)

    n_bins_eff = int(edges.size - 1)
    bin_idx = np.searchsorted(edges, p, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins_eff - 1)

    counts = np.bincount(bin_idx, minlength=n_bins_eff).astype(int)
    err_counts = np.bincount(bin_idx, weights=inc, minlength=n_bins_eff).astype(int)
    pred_sum = np.bincount(bin_idx, weights=p, minlength=n_bins_eff).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_rate = err_counts / counts
        pred_mean = pred_sum / counts

    lows, highs = [], []
    for k, n in zip(err_counts.tolist(), counts.tolist()):
        lo, hi = _wilson_interval(int(k), int(n), z=z_score)
        lows.append(lo)
        highs.append(hi)

    df = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": (edges[:-1] + edges[1:]) / 2.0,
            "count": counts,
            "error_count": err_counts,
            "error_rate": error_rate,
            "p_pred_mean": pred_mean,
            "wilson_low": np.asarray(lows, dtype=float),
            "wilson_high": np.asarray(highs, dtype=float),
        }
    )
    df = df[df["count"] > 0].reset_index(drop=True)
    return df


def expected_calibration_error(rel_df: pd.DataFrame) -> float:
    if rel_df.empty:
        return float("nan")
    n = rel_df["count"].sum()
    if n <= 0:
        return float("nan")
    gap = (rel_df["error_rate"] - rel_df["p_pred_mean"]).abs()
    return float((rel_df["count"] / n * gap).sum())

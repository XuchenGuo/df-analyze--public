"""Apply a fitted adaptive-error mapping to the untouched holdout.

The confidence transform and error mapping are learned from training OOF
predictions. This stage refits the selected base model, predicts the holdout,
and reports expected error without fitting any risk component on holdout rows.
"""

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
from df_analyze.analysis.adaptive_error.base_models_helpers import (
    _apply_confidence_transform,
    _decode_labels,
)
from df_analyze.analysis.adaptive_error.confidence_metrics import (
    knn_distance_weighted_conf,
    knn_min_dist_raw,
    knn_neighbor_vote_conf,
    proba_margin,
    proba_margin_for_pred,
    proba_max_for_pred,
    proba_p2nd,
    proba_pmax,
    tree_leaf_support_conf,
    tree_vote_agreement_conf,
)
from df_analyze.analysis.adaptive_error.plots import (
    build_confidence_error_bins_table,
    merge_small_bins,
    plot_confidence_vs_error,
)
from df_analyze.analysis.adaptive_error.proba import (
    align_proba_with_predictions,
    predict_proba_or_scores,
    predict_scores,
    scores_to_proba,
)
from df_analyze.analysis.adaptive_error.report import (
    _write_csv,
    _write_not_available_csv,
    _write_not_available_parquet,
    _write_not_available_png,
    _write_parquet,
    _write_sanity_checks,
)
from df_analyze.runtime.hardware import (
    DeviceIntent,
    RuntimeComponent,
    clear_fitted_model_state,
    device_reason_text,
    is_cuda_runtime_error,
    release_accelerator_memory,
)


@dataclass
class _TestEvalResult:
    test_df: pd.DataFrame
    used_conf_metric: str
    e_hat_test: np.ndarray
    incorrect_test: np.ndarray
    drift_rate: Optional[float]
    proba_test_cal: np.ndarray
    test_bins_df: Optional[pd.DataFrame]


def _coerce_pred_vector(preds, X_test_index: pd.Index) -> Optional[np.ndarray]:
    if isinstance(preds, pd.Series):
        if not preds.index.equals(X_test_index):
            preds = preds.reindex(X_test_index)
            if preds.isna().any():
                return None
        return preds.to_numpy()
    if isinstance(preds, pd.DataFrame):
        if preds.shape[1] != 1:
            return None
        series = preds.iloc[:, 0]
        if not series.index.equals(X_test_index):
            series = series.reindex(X_test_index)
            if series.isna().any():
                return None
        return series.to_numpy()
    return np.asarray(preds).ravel()


def _predict_test_outputs(
    *,
    result,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple[np.ndarray, np.ndarray, Any, Optional[np.ndarray]]:
    y_pred_test_raw = None
    proba_test = None
    scores_test = None
    expect_single_target = isinstance(y_test, pd.Series) or (
        isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1
    )

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    model = result.model
    tuned_model = getattr(model, "tuned_model", None)
    if tuned_model is None:
        tuned_args = getattr(model, "tuned_args", None) or result.params
        model.refit_tuned(
            X_train,
            y_train,
            tuned_args=tuned_args,
        )
        tuned_model = getattr(model, "tuned_model", None)
    if tuned_model is None:
        raise RuntimeError(
            "tuned_model is still unavailable after refit; "
            "cannot compute test predictions."
        )

    preds = model.tuned_predict(X_test)
    if expect_single_target and isinstance(preds, pd.DataFrame) and preds.shape[1] > 1:
        raise RuntimeError(
            "Adaptive error expects single-target predictions, "
            "but tuned_predict returned multi-target output."
        )
    y_pred_test_raw = _coerce_pred_vector(preds, X_test.index)
    if y_pred_test_raw is None:
        raise RuntimeError("Could not align tuned predictions with test index.")
    proba_test = None
    try:
        proba_test = model.predict_proba(X_test)
    except (AttributeError, TypeError, ValueError):
        proba_test, scores_test = predict_proba_or_scores(tuned_model, X_test)
    if proba_test is None and scores_test is not None:
        proba_test = scores_to_proba(scores_test)
    if proba_test is None:
        raise RuntimeError(
            "Adaptive error analysis requires predict_proba or decision_function."
        )
    if scores_test is None:
        scores_test = predict_scores(tuned_model, X_test)

    return y_pred_test_raw, proba_test, tuned_model, scores_test


def _compute_test_confidence(
    *,
    selected_conf_metric: str,
    selected_conf_params: dict[str, Any],
    proba_test_cal: np.ndarray,
    y_pred_test_raw: np.ndarray,
    tuned_model,
    X_test,
    y_train,
    X_train,
) -> tuple[np.ndarray, str]:
    used_conf_metric = selected_conf_metric
    conf_raw = None
    if selected_conf_metric == "proba_margin":
        conf_raw = proba_margin_for_pred(proba_test_cal, y_pred_test_raw)
    elif selected_conf_metric == "knn_vote":
        conf_raw = knn_neighbor_vote_conf(
            tuned_model,
            X_test,
            y_pred_test_raw,
            y_train.to_numpy(),
        )
    elif selected_conf_metric == "knn_dist_weighted":
        conf_raw = knn_distance_weighted_conf(
            tuned_model,
            X_test,
            y_pred_test_raw,
            y_train.to_numpy(),
        )
    elif selected_conf_metric == "knn_min_dist":
        conf_raw = knn_min_dist_raw(tuned_model, X_test)
        if conf_raw is None:
            raise RuntimeError("kneighbors not available")
    elif selected_conf_metric == "tree_vote_agreement":
        conf_raw = tree_vote_agreement_conf(
            tuned_model,
            X_test,
            y_pred_test_raw,
        )
    elif selected_conf_metric == "tree_leaf_support":
        conf_raw = tree_leaf_support_conf(
            tuned_model,
            X_test,
            n_train=len(X_train),
        )
    else:
        conf_raw = proba_margin_for_pred(proba_test_cal, y_pred_test_raw)
        used_conf_metric = "proba_margin"

    if conf_raw is None:
        raise RuntimeError(
            f"Selected confidence metric '{selected_conf_metric}' returned None."
        )

    params_to_apply = (
        selected_conf_params
        if used_conf_metric == selected_conf_metric
        else {"kind": "identity", "params": {}}
    )
    conf_test = _apply_confidence_transform(conf_raw, params_to_apply)
    if conf_test is None:
        raise RuntimeError(
            f"Confidence transform failed for metric '{used_conf_metric}'."
        )

    return conf_test, used_conf_metric


def _evaluate_test_stage_attempt(
    *,
    result,
    X_train,
    X_test,
    y_train,
    y_test,
    labels_map,
    aer: AdaptiveErrorCalculator,
    calibrator,
    selected_conf_metric: str,
    selected_conf_params: dict[str, Any],
    options,
    no_preds: bool,
    m_preds: Path,
    m_tables: Path,
    m_plots: Path,
    m_meta: Path,
) -> _TestEvalResult:
    """Run one complete holdout evaluation on the currently resolved device."""
    y_pred_test_raw, proba_test, tuned_model, scores_test = _predict_test_outputs(
        result=result,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    proba_test = align_proba_with_predictions(
        proba_test,
        y_pred_test_raw,
        scores_test,
    )
    proba_test_cal = calibrator.transform(proba_test)

    y_pred_test_from_cal_proba = np.argmax(proba_test_cal, axis=1)

    # compute confidence using the selected confidence metric
    conf_test, used_conf_metric = _compute_test_confidence(
        selected_conf_metric=selected_conf_metric,
        selected_conf_params=selected_conf_params,
        proba_test_cal=proba_test_cal,
        y_pred_test_raw=y_pred_test_raw,
        tuned_model=tuned_model,
        X_test=X_test,
        y_train=y_train,
        X_train=X_train,
    )

    e_hat_test = aer.get_expected_error(conf_test)

    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError(
                "Adaptive error test stage expects a single target column in y_test."
            )
        y_true_arr = y_test.iloc[:, 0].to_numpy()
    elif isinstance(y_test, pd.Series):
        y_true_arr = y_test.to_numpy()
    else:
        y_true_arr = np.asarray(y_test)
    y_true_arr = np.asarray(y_true_arr).ravel()
    if y_true_arr.shape[0] != y_pred_test_raw.shape[0]:
        raise ValueError(
            "Length mismatch between predictions and y_test in adaptive error test stage."
        )
    y_true_label = _decode_labels(y_true_arr, labels_map)
    y_pred_label = _decode_labels(y_pred_test_raw, labels_map)

    incorrect_test = (y_pred_test_raw != y_true_arr).astype(int)
    correct = incorrect_test == 0

    aer_pct = np.round(e_hat_test * 100.0, 1)
    target_error = getattr(options, "aer_target_error", 0.05)
    if target_error is None:
        target_error = 0.05
    target_error = float(target_error)
    flag_target_error = (e_hat_test >= target_error).astype(int)
    p_max = proba_pmax(proba_test_cal)
    p_2nd = proba_p2nd(proba_test_cal)
    p_margin = proba_margin(proba_test_cal)

    # predicted class probability diagnostics
    p_pred = proba_max_for_pred(proba_test_cal, y_pred_test_raw)
    p_pred_margin = proba_margin_for_pred(proba_test_cal, y_pred_test_raw)
    test_payload = {
        "row_id": X_test.index,
        "y_true": y_true_arr,
        "y_pred": y_pred_test_raw,
        "correct": correct.astype(int),
        "confidence": conf_test,
        "aer": e_hat_test,
        "aer_pct": aer_pct,
        "flag_gt_target_error": flag_target_error,
        "y_true_label": y_true_label,
        "y_pred_label": y_pred_label,
        "y_pred_from_cal_proba": y_pred_test_from_cal_proba,
    }
    test_payload["p_max"] = p_max
    test_payload["p_2nd"] = p_2nd
    test_payload["p_margin"] = p_margin
    test_payload["p_pred"] = p_pred
    test_payload["p_pred_margin"] = p_pred_margin
    test_df = pd.DataFrame(test_payload)

    if no_preds:
        reason = "Per-sample outputs disabled by --no-preds."
        _write_not_available_parquet(m_preds / "test_per_sample.parquet", reason)
        _write_not_available_csv(m_preds / "test_per_sample.csv", reason)
    else:
        _write_parquet(
            test_df,
            m_preds / "test_per_sample.parquet",
            index=False,
        )
        _write_csv(test_df, m_preds / "test_per_sample.csv", index=False)

    sanity_issues: dict[str, Any] = {}
    sanity_info: dict[str, Any] = {}
    lengths = {
        "y_true": int(y_true_arr.size),
        "y_pred": int(np.asarray(y_pred_test_raw).size),
        "confidence": int(np.asarray(conf_test).size),
        "aer": int(np.asarray(e_hat_test).size),
    }
    sanity_info["lengths"] = lengths
    if len(set(lengths.values())) > 1:
        sanity_issues["length_mismatch"] = lengths

    drift_rate = float(np.mean(y_pred_test_raw != y_pred_test_from_cal_proba))
    if drift_rate is not None:
        sanity_info["pred_drift_rate"] = drift_rate
        if drift_rate >= 0.05:
            sanity_issues["pred_drift_rate"] = {
                "value": drift_rate,
                "threshold": 0.05,
            }

    spearman = float(
        pd.Series(conf_test).corr(pd.Series(incorrect_test), method="spearman")
    )
    if spearman is not None and np.isfinite(spearman):
        sanity_info["spearman_conf_vs_incorrect"] = spearman
        if spearman > 0.1:
            sanity_issues["spearman_conf_vs_incorrect"] = {
                "value": spearman,
                "threshold": 0.1,
            }

    if sanity_issues:
        sanity_payload = {
            "status": "issues_found",
            "issues": sanity_issues,
            "info": sanity_info,
        }
        _write_sanity_checks(m_meta / "sanity_checks.json", sanity_payload)

    # evaluation: confidence vs error
    test_bins_df = None
    if aer.bin_edges is None:
        raise RuntimeError("aER bin edges are unavailable.")
    test_bins_df = build_confidence_error_bins_table(
        conf=conf_test,
        incorrect=incorrect_test,
        expected_error=e_hat_test,
        edges=aer.bin_edges,
    )
    if test_bins_df.empty:
        test_bins_df = None
        _write_not_available_csv(
            m_tables / "test_confidence_error_bins.csv",
            reason="No test samples.",
        )
        _write_not_available_png(
            m_plots / "confidence_vs_expected_error.png",
            title="Confidence vs Expected Error",
            reason="No test samples.",
        )
    else:
        original_test_bins_df = test_bins_df[
            test_bins_df["count"].fillna(0).astype(int) > 0
        ].reset_index(drop=True)
        merged_test_bins_df = merge_small_bins(
            original_test_bins_df,
            min_count=int(getattr(options, "aer_min_bin_count", 10)),
            z_score=1.96,
        )
        if merged_test_bins_df is None or merged_test_bins_df.empty:
            test_bins_df = original_test_bins_df
        elif (
            int(original_test_bins_df.shape[0]) > 1
            and int(merged_test_bins_df.shape[0]) == 1
        ):
            test_bins_df = original_test_bins_df
        else:
            test_bins_df = merged_test_bins_df
        _write_csv(
            test_bins_df,
            m_tables / "test_confidence_error_bins.csv",
            index=False,
        )
        plot_confidence_vs_error(
            test_bins_df,
            m_plots / "confidence_vs_expected_error.png",
            title="Confidence vs Expected Error",
            annotate_counts=False,
        )

    return _TestEvalResult(
        test_df=test_df,
        used_conf_metric=used_conf_metric,
        e_hat_test=e_hat_test,
        incorrect_test=incorrect_test,
        drift_rate=drift_rate,
        proba_test_cal=proba_test_cal,
        test_bins_df=test_bins_df,
    )


def _clear_model_state_for_cpu_retry(model) -> None:
    try:
        clear_fitted_model_state(model)
    except Exception:
        pass
    cleanup = getattr(model, "_cleanup_after_fold", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception:
            pass
    release_accelerator_memory()


def _evaluate_test_stage(
    *,
    result,
    X_train,
    X_test,
    y_train,
    y_test,
    labels_map,
    aer: AdaptiveErrorCalculator,
    calibrator,
    selected_conf_metric: str,
    selected_conf_params: dict[str, Any],
    options,
    no_preds: bool,
    m_preds: Path,
    m_tables: Path,
    m_plots: Path,
    m_meta: Path,
) -> _TestEvalResult:
    """Evaluate once, restarting the complete stage on CPU after an auto-CUDA failure."""
    attempt_args = {
        "result": result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "labels_map": labels_map,
        "aer": aer,
        "calibrator": calibrator,
        "selected_conf_metric": selected_conf_metric,
        "selected_conf_params": selected_conf_params,
        "options": options,
        "no_preds": no_preds,
        "m_preds": m_preds,
        "m_tables": m_tables,
        "m_plots": m_plots,
        "m_meta": m_meta,
    }
    model = result.model
    base_runtime = getattr(model, "runtime", None)
    if base_runtime is None:
        return _evaluate_test_stage_attempt(**attempt_args)

    component = getattr(
        result.model_cls,
        "runtime_component",
        RuntimeComponent.Sklearn,
    )
    runtime = base_runtime.for_task(
        len(X_train),
        X_train.shape[1],
        n_queries=len(X_test),
    )
    decision = runtime.decision_for(component)
    print(
        "[device] Adaptive error holdout "
        f"{getattr(model, 'shortname', result.model_cls.__name__)} / "
        f"{getattr(result, 'selection', 'none')}: {decision.resolved.upper()} "
        f"({len(X_train)} train rows x {len(X_test)} query rows x "
        f"{X_train.shape[1]} features; {device_reason_text(decision)})"
    )

    try:
        model.set_runtime(runtime)
        return _evaluate_test_stage_attempt(**attempt_args)
    except Exception as error:
        cuda_failure = (
            decision.resolved == "cuda" and is_cuda_runtime_error(error)
        )
        if runtime.intent is DeviceIntent.Auto and cuda_failure:
            _clear_model_state_for_cpu_retry(model)
            runtime.record_cpu_fallback(
                component,
                f"cuda_runtime_fallback:{type(error).__name__}",
            )
            model.set_runtime(runtime)
            warn(
                "Adaptive error holdout encountered a CUDA runtime failure. "
                "Restarting this complete model/selection holdout task on CPU once."
            )
            return _evaluate_test_stage_attempt(**attempt_args)
        if runtime.intent is DeviceIntent.CUDA and cuda_failure:
            raise RuntimeError(
                "Adaptive error holdout failed on CUDA while --device cuda is "
                "strict. CPU fallback is disabled."
            ) from error
        raise

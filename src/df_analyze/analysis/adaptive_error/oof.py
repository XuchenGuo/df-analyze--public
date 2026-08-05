"""Build leakage-safe predictions for fitting adaptive error estimates.

Every training row is predicted by a fold model that did not fit that row.
The resulting confidence/error pairs fit the adaptive-error mapping; the
unchanged holdout is evaluated later in ``test_stage``.
"""

from __future__ import annotations

# ref: Out-of-fold predictions via cross-validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html

import inspect
from typing import Any, Optional
from warnings import warn

import numpy as np
import pandas as pd

from df_analyze._constants import SEED
from df_analyze.analysis.adaptive_error.confidence_metrics import (
    knn_distance_weighted_conf,
    knn_min_dist_raw,
    knn_neighbor_vote_conf,
    proba_margin_for_pred,
    tree_leaf_support_conf,
    tree_vote_agreement_conf,
)
from df_analyze.analysis.adaptive_error.proba import (
    align_proba_with_predictions,
    predict_proba_or_scores,
    predict_scores,
    scores_to_proba,
)
from df_analyze.models.base import classification_output_dim
from df_analyze.runtime.hardware import (
    DeviceIntent,
    RuntimeComponent,
    RuntimePolicy,
    clear_fitted_model_state,
    device_reason_text,
    get_runtime,
    is_cuda_runtime_error,
    release_accelerator_memory,
)
from df_analyze.splitting import OmniKFold


def _init_model(model_cls: type, y_train: pd.Series, model_args=None) -> Any:
    parameters = inspect.signature(model_cls).parameters
    kwargs = {}
    if "num_classes" in parameters:
        kwargs["num_classes"] = classification_output_dim(y_train)
    if "model_args" in parameters and model_args:
        kwargs["model_args"] = dict(model_args)
    return model_cls(**kwargs)


def build_oof_for_result(
    result,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: Optional[pd.Series],
    n_folds: int,
    seed: int = SEED,
    options=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Refit one tuned configuration across folds and collect OOF outputs.

    Hyperparameters remain fixed so the mapping measures prediction confidence
    rather than another round of tuning. Groups remain disjoint when supplied.
    """
    splitter = OmniKFold(
        n_splits=n_folds,
        is_classification=True,
        grouped=groups is not None,
        labels=None,
        shuffle=True,
        seed=seed,
        warn_on_fallback=True,
        df_analyze_phase="Adaptive error OOF",
    )
    splits = splitter.split(
        X_train=X_train,
        y_train=y_train,
        g_train=groups,
    )[0]
    if not splits:
        raise RuntimeError(
            "Adaptive error analysis could not create any cross-validation folds."
        )
    component = getattr(
        result.model_cls, "runtime_component", RuntimeComponent.Sklearn
    )
    fold_sizes = [
        (len(idx_train), len(idx_query))
        for idx_train, idx_query in splits
    ]
    if component is RuntimeComponent.KNN:
        largest_train, largest_query = max(
            fold_sizes,
            key=lambda sizes: sizes[0] * sizes[1],
        )
    else:
        largest_train, largest_query = max(
            fold_sizes,
            key=lambda sizes: sizes[0],
        )
    base_runtime = getattr(result.model, "runtime", None) or get_runtime(
        DeviceIntent.CPU
    )
    runtime = base_runtime.for_task(
        largest_train,
        X_train.shape[1],
        n_queries=largest_query,
    )
    decision = runtime.decision_for(component)
    print(
        "[device] Adaptive error OOF "
        f"{getattr(result.model, 'shortname', result.model_cls.__name__)} / "
        f"{getattr(result, 'selection', 'none')}: {decision.resolved.upper()} "
        f"({largest_train} train rows x {largest_query} query rows x "
        f"{X_train.shape[1]} features; "
        f"{device_reason_text(decision)})"
    )
    try:
        output = _build_oof_attempt(
            result,
            X_train,
            y_train,
            groups,
            runtime,
            splits,
        )
        _record_oof_device(
            options,
            result,
            runtime,
            component,
            attempt=1,
            stage="completed",
        )
        return output
    except Exception as error:
        _record_oof_device(
            options,
            result,
            runtime,
            component,
            attempt=1,
            stage="failed",
            error=error,
        )
        cuda_failure = (
            decision.resolved == "cuda" and is_cuda_runtime_error(error)
        )
        if runtime.intent is DeviceIntent.Auto and cuda_failure:
            reason = f"cuda_runtime_fallback:{type(error).__name__}"
            runtime.record_cpu_fallback(component, reason)
            release_accelerator_memory()
            warn(
                "Adaptive error OOF encountered a CUDA runtime failure. "
                "Restarting this complete model/selection OOF task on CPU once."
            )
            try:
                output = _build_oof_attempt(
                    result,
                    X_train,
                    y_train,
                    groups,
                    runtime,
                    splits,
                )
            except Exception as retry_error:
                _record_oof_device(
                    options,
                    result,
                    runtime,
                    component,
                    attempt=2,
                    stage="failed",
                    error=retry_error,
                )
                raise
            _record_oof_device(
                options,
                result,
                runtime,
                component,
                attempt=2,
                stage="completed",
            )
            return output
        if runtime.intent is DeviceIntent.CUDA and cuda_failure:
            raise RuntimeError(
                "Adaptive error OOF failed on CUDA while --device cuda is "
                "strict. CPU fallback is disabled."
            ) from error
        raise


def _record_oof_device(
    options,
    result,
    runtime: RuntimePolicy,
    component: RuntimeComponent,
    *,
    attempt: int,
    stage: str,
    error: Optional[BaseException] = None,
) -> None:
    if options is None:
        return
    record: dict[str, object] = {
        "fold": getattr(options, "_runtime_current_fold", None),
        "model": getattr(result.model, "shortname", result.model_cls.__name__),
        "selection": f"aer-oof:{getattr(result, 'selection', 'none')}",
        "phase": "adaptive-error-oof",
        "attempt": attempt,
        "stage": stage,
        **runtime.decision_for(component).to_dict(),
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    audit = getattr(options, "_runtime_model_audit", [])
    audit.append(record)
    options._runtime_model_audit = audit


def _fit_oof_fold(
    result,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    g_tr: Optional[pd.Series],
    X_val: pd.DataFrame,
    runtime: RuntimePolicy,
) -> dict[str, Optional[np.ndarray]]:
    model = _init_model(
        result.model_cls,
        y_tr,
        model_args=getattr(result.model, "model_args", None),
    )
    model.set_runtime(runtime)
    try:
        model.refit_tuned(X_tr, y_tr, g=g_tr, tuned_args=result.params)
        tuned_model = getattr(model, "tuned_model", None)
        if tuned_model is None:
            raise RuntimeError(
                "Adaptive error OOF requires a tuned estimator after refit."
            )

        y_pred = np.asarray(model.tuned_predict(X_val)).ravel()
        proba, scores = predict_proba_or_scores(tuned_model, X_val)
        if proba is None and scores is None:
            proba, scores = predict_proba_or_scores(model, X_val)
        if proba is None and scores is not None:
            proba = scores_to_proba(scores)
        if proba is None:
            raise RuntimeError(
                "Adaptive error analysis requires predict_proba or "
                "decision_function."
            )
        if scores is None:
            scores = predict_scores(tuned_model, X_val)
            if scores is None:
                scores = predict_scores(model, X_val)

        proba = align_proba_with_predictions(proba, y_pred, scores)
        y_tr_arr = y_tr.to_numpy()
        return {
            "y_pred": y_pred,
            "proba": np.asarray(proba),
            "conf": np.asarray(proba_margin_for_pred(proba, y_pred)),
            "knn_vote": knn_neighbor_vote_conf(
                tuned_model, X_val, y_pred, y_tr_arr
            ),
            "knn_dist_weighted": knn_distance_weighted_conf(
                tuned_model, X_val, y_pred, y_tr_arr
            ),
            "knn_min_dist": knn_min_dist_raw(tuned_model, X_val),
            "tree_vote_agreement": tree_vote_agreement_conf(
                tuned_model, X_val, y_pred
            ),
            "tree_leaf_support": tree_leaf_support_conf(
                tuned_model, X_val, n_train=len(X_tr)
            ),
        }
    finally:
        try:
            clear_fitted_model_state(model)
            model._cleanup_after_fold()
        finally:
            del model


def _build_oof_attempt(
    result,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: Optional[pd.Series],
    runtime: RuntimePolicy,
    splits,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_samples = len(X_train)
    y_pred_oof = np.full(n_samples, None, dtype=object)
    conf_oof = np.full(n_samples, np.nan, dtype=float)
    raw_knn_min_dist = np.full(n_samples, np.nan, dtype=float)
    conf_knn_vote = np.full(n_samples, np.nan, dtype=float)
    conf_knn_dist_weighted = np.full(n_samples, np.nan, dtype=float)
    conf_tree_vote_agreement = np.full(n_samples, np.nan, dtype=float)
    conf_tree_leaf_support = np.full(n_samples, np.nan, dtype=float)
    proba_rows: list[Optional[np.ndarray]] = [None] * n_samples

    for idx_train, idx_val in splits:
        X_tr = X_train.iloc[idx_train]
        y_tr = y_train.iloc[idx_train]
        g_tr = groups.iloc[idx_train] if groups is not None else None
        X_val = X_train.iloc[idx_val]
        outputs = _fit_oof_fold(
            result, X_tr, y_tr, g_tr, X_val, runtime
        )
        y_pred = np.asarray(outputs["y_pred"])
        proba = np.asarray(outputs["proba"])
        conf = np.asarray(outputs["conf"])

        kv = outputs["knn_vote"]
        if kv is not None:
            conf_knn_vote[idx_val] = np.asarray(kv, dtype=float)

        kd = outputs["knn_dist_weighted"]
        if kd is not None:
            conf_knn_dist_weighted[idx_val] = np.asarray(kd, dtype=float)

        md = outputs["knn_min_dist"]
        if md is not None:
            raw_knn_min_dist[idx_val] = np.asarray(md, dtype=float)

        ta = outputs["tree_vote_agreement"]
        if ta is not None:
            conf_tree_vote_agreement[idx_val] = np.asarray(ta, dtype=float)

        ts = outputs["tree_leaf_support"]
        if ts is not None:
            conf_tree_leaf_support[idx_val] = np.asarray(ts, dtype=float)
        y_pred_oof[idx_val] = y_pred
        conf_oof[idx_val] = conf
        for row_idx, row in zip(idx_val, proba):
            proba_rows[row_idx] = np.asarray(row, dtype=float)

    if any(val is None for val in y_pred_oof.tolist()):
        raise RuntimeError("Missing OOF predictions for one or more samples.")
    if any(val is None for val in proba_rows):
        raise RuntimeError("Missing OOF probabilities for one or more samples.")

    proba_oof = np.vstack([row for row in proba_rows if row is not None])
    y_true_arr = y_train.to_numpy()
    correct_oof = (np.asarray(y_pred_oof) == y_true_arr).astype(int)

    data = {
        "row_id": X_train.index,
        "y_true": y_true_arr,
        "y_pred_oof": y_pred_oof,
        "conf_oof": conf_oof,
        "correct_oof": correct_oof,
        "raw_knn_min_dist": raw_knn_min_dist,
        "conf_knn_vote": conf_knn_vote,
        "conf_knn_dist_weighted": conf_knn_dist_weighted,
        "conf_tree_vote_agreement": conf_tree_vote_agreement,
        "conf_tree_leaf_support": conf_tree_leaf_support,
    }
    return pd.DataFrame(data=data), proba_oof

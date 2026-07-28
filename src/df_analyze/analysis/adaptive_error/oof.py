# ref: Out-of-fold predictions via cross-validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html

from __future__ import annotations

import inspect
from typing import Any, Optional

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

    kf = OmniKFold(
        n_splits=n_folds,
        is_classification=True,
        grouped=groups is not None,
        labels=None,
        shuffle=True,
        seed=seed,
        warn_on_fallback=True,
        df_analyze_phase="Adaptive error OOF",
    )
    splits = kf.split(X_train=X_train, y_train=y_train, g_train=groups)[0]

    for idx_train, idx_val in splits:
        X_tr = X_train.iloc[idx_train]
        y_tr = y_train.iloc[idx_train]
        g_tr = groups.iloc[idx_train] if groups is not None else None
        X_val = X_train.iloc[idx_val]

        model = _init_model(
            result.model_cls,
            y_train,
            model_args=getattr(result.model, "model_args", None),
        )
        runtime = getattr(result.model, "runtime", None)
        if runtime is not None:
            model.set_runtime(runtime)
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
                "Adaptive error analysis requires predict_proba or decision_function."
            )
        if scores is None:
            scores = predict_scores(tuned_model, X_val)
            if scores is None:
                scores = predict_scores(model, X_val)

        proba = align_proba_with_predictions(proba, y_pred, scores)
        # universal probability-margin confidence
        conf = proba_margin_for_pred(proba, y_pred)

        # additional confidence metrics
        y_tr_arr = y_tr.to_numpy()
        kv = knn_neighbor_vote_conf(tuned_model, X_val, y_pred, y_tr_arr)
        if kv is not None:
            conf_knn_vote[idx_val] = np.asarray(kv, dtype=float)

        kd = knn_distance_weighted_conf(tuned_model, X_val, y_pred, y_tr_arr)
        if kd is not None:
            conf_knn_dist_weighted[idx_val] = np.asarray(kd, dtype=float)

        md = knn_min_dist_raw(tuned_model, X_val)
        if md is not None:
            raw_knn_min_dist[idx_val] = np.asarray(md, dtype=float)

        ta = tree_vote_agreement_conf(tuned_model, X_val, y_pred)
        if ta is not None:
            conf_tree_vote_agreement[idx_val] = np.asarray(ta, dtype=float)

        ts = tree_leaf_support_conf(tuned_model, X_val, n_train=len(X_tr))
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

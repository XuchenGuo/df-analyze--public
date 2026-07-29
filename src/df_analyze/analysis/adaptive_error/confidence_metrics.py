"""confidence metrics used by adaptive error rate, details can view in the readme"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from df_analyze.analysis.adaptive_error.proba import normalize_proba

Array = np.ndarray


def _as_2d(X: Any) -> Any:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X
    if sparse.issparse(X):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _as_2d_without_feature_names(X: Any) -> Any:
    """Prepare input for ensemble members fitted internally on NumPy arrays."""
    arr = _as_2d(X)
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.to_numpy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    return arr


def _pred_indices_for_proba(y_pred: Array, n_classes: int) -> Optional[np.ndarray]:
    y = np.asarray(y_pred).ravel()
    if y.size == 0:
        return np.asarray([], dtype=int)
    try:
        idx = y.astype(int)
    except (TypeError, ValueError):
        return None
    if idx.shape[0] != y.shape[0]:
        return None
    if np.any(idx < 0) or np.any(idx >= int(n_classes)):
        return None
    return idx


def proba_max_for_pred(proba: Array, y_pred: Array) -> Array:
    p = normalize_proba(np.asarray(proba, dtype=float))
    if p.ndim != 2 or p.shape[0] == 0:
        return np.asarray([], dtype=float)
    n_classes = int(p.shape[1])
    idx = _pred_indices_for_proba(y_pred, n_classes)
    if idx is None:
        return np.max(p, axis=1)
    row_idx = np.arange(p.shape[0])
    out = p[row_idx, idx]
    return np.clip(out.astype(float), 0.0, 1.0)


def proba_pmax(proba: Array) -> Array:
    p = normalize_proba(np.asarray(proba, dtype=float))
    if p.ndim != 2 or p.shape[0] == 0:
        return np.asarray([], dtype=float)
    return np.clip(np.max(p, axis=1).astype(float), 0.0, 1.0)


def proba_p2nd(proba: Array) -> Array:
    p = normalize_proba(np.asarray(proba, dtype=float))
    if p.ndim != 2 or p.shape[0] == 0:
        return np.asarray([], dtype=float)
    if p.shape[1] < 2:
        return np.zeros(p.shape[0], dtype=float)
    p_sorted = np.sort(p, axis=1)
    return np.clip(p_sorted[:, -2].astype(float), 0.0, 1.0)


def proba_margin(proba: Array) -> Array:
    p = normalize_proba(np.asarray(proba, dtype=float))
    if p.ndim != 2 or p.shape[0] == 0:
        return np.asarray([], dtype=float)

    n_classes = int(p.shape[1])
    if n_classes < 2:
        return np.ones(p.shape[0], dtype=float)

    if n_classes == 2:
        p1 = p[:, 1].astype(float)
        return np.clip(2.0 * np.abs(p1 - 0.5), 0.0, 1.0)

    p_sorted = np.sort(p, axis=1)
    margin = (p_sorted[:, -1] - p_sorted[:, -2]).astype(float)
    return np.clip(margin, 0.0, 1.0)


def proba_margin_for_pred(proba: Array, y_pred: Array) -> Array:
    p = normalize_proba(np.asarray(proba, dtype=float))
    if p.ndim != 2 or p.shape[0] == 0:
        return np.asarray([], dtype=float)
    n_classes = int(p.shape[1])
    if n_classes < 2:
        return np.ones(p.shape[0], dtype=float)

    idx = _pred_indices_for_proba(y_pred, n_classes)
    if idx is None:
        p_sorted = np.sort(p, axis=1)
        margin = p_sorted[:, -1] - p_sorted[:, -2]
        return np.clip(margin.astype(float), 0.0, 1.0)

    row_idx = np.arange(p.shape[0])
    p_pred = p[row_idx, idx]
    p_other = p.copy()
    p_other[row_idx, idx] = -np.inf
    p_runnerup = np.max(p_other, axis=1)
    margin = p_pred - p_runnerup
    return np.clip(margin.astype(float), 0.0, 1.0)


def knn_neighbor_vote_conf(
    estimator: Any,
    X: Any,
    y_pred: Array,
    y_train_fold: Array,
) -> Optional[Array]:
    if not hasattr(estimator, "kneighbors"):
        return None
    idx = estimator.kneighbors(_as_2d(X), return_distance=False)
    idx = np.asarray(idx, dtype=int)
    y_tr = np.asarray(y_train_fold)
    neigh_y = y_tr[idx]
    yhat = np.asarray(y_pred).reshape(-1, 1)
    return np.mean(neigh_y == yhat, axis=1).astype(float)


def knn_distance_weighted_conf(
    estimator: Any,
    X: Any,
    y_pred: Array,
    y_train_fold: Array,
    *,
    eps: float = 1e-12,
) -> Optional[Array]:
    if not hasattr(estimator, "kneighbors"):
        return None
    dist, idx = estimator.kneighbors(_as_2d(X), return_distance=True)
    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=int)
    y_tr = np.asarray(y_train_fold)
    neigh_y = y_tr[idx]
    yhat = np.asarray(y_pred).reshape(-1, 1)
    w = 1.0 / (dist + float(eps))
    mask = (neigh_y == yhat).astype(float)
    num = np.sum(w * mask, axis=1)
    den = np.sum(w, axis=1)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out.astype(float)


def knn_min_dist_raw(estimator: Any, X: Any) -> Optional[Array]:
    if not hasattr(estimator, "kneighbors"):
        return None
    dist, _ = estimator.kneighbors(_as_2d(X), return_distance=True)
    dist = np.asarray(dist, dtype=float)
    if dist.ndim != 2 or dist.shape[1] < 1:
        return None
    return dist[:, 0].astype(float)


def _sklearn_ensemble_estimators(estimator: Any) -> Optional[list[Any]]:
    ests = getattr(estimator, "estimators_", None)
    if ests is None:
        return None
    if isinstance(ests, list):
        return ests
    try:
        arr = np.asarray(ests, dtype=object)
        flat = arr.ravel().tolist()
        flat = [e for e in flat if e is not None]
        return flat if flat else None
    except (TypeError, ValueError):
        return None


def tree_vote_agreement_conf(estimator: Any, X: Any, y_pred: Array) -> Optional[Array]:
    base = _sklearn_ensemble_estimators(estimator)
    if base is not None:
        tree_X = _as_2d_without_feature_names(X)
        preds = np.vstack([np.asarray(t.predict(tree_X)).ravel() for t in base])
        yhat = np.asarray(y_pred).ravel()[None, :]
        return np.mean(preds == yhat, axis=0).astype(float)
    booster = getattr(estimator, "booster_", None)
    n_classes = getattr(estimator, "n_classes_", None)
    if booster is None or n_classes not in (None, 2):
        return None

    leaf_idx = booster.predict(_as_2d(X), pred_leaf=True)
    leaf_idx = np.asarray(leaf_idx)
    if leaf_idx.ndim == 1:
        leaf_idx = leaf_idx.reshape(-1, 1)
    dump = booster.dump_model()
    tree_info = dump.get("tree_info", [])
    n_trees = min(len(tree_info), leaf_idx.shape[1])
    if n_trees <= 0:
        return None
    leaf_values: list[np.ndarray] = []
    for t in range(n_trees):
        struct = tree_info[t].get("tree_structure", {})
        leaf_map = _lgbm_leaf_attr_map(struct, attr="leaf_value")
        if not leaf_map:
            return None
        max_idx = max(leaf_map)
        arr = np.full(max_idx + 1, np.nan, dtype=float)
        for k, v in leaf_map.items():
            if 0 <= k <= max_idx:
                arr[k] = float(v)
        leaf_values.append(arr)

    votes = np.zeros((n_trees, leaf_idx.shape[0]), dtype=int)
    for t in range(n_trees):
        idx_t = leaf_idx[:, t].astype(int)
        arr = leaf_values[t]
        idx_t = np.clip(idx_t, 0, arr.shape[0] - 1)
        v = arr[idx_t]
        votes[t, :] = (v > 0.0).astype(int)

    yhat = np.asarray(y_pred).astype(int).ravel()[None, :]
    return np.mean(votes == yhat, axis=0).astype(float)


def tree_leaf_support_conf(estimator: Any, X: Any, *, n_train: int) -> Optional[Array]:
    n_train_f = float(max(int(n_train), 1))

    base = _sklearn_ensemble_estimators(estimator)
    if base is not None:
        tree_X = _as_2d_without_feature_names(X)
        support = np.zeros(len(tree_X), dtype=float)
        n_trees = 0
        for t in base:
            if not hasattr(t, "apply") or not hasattr(t, "tree_"):
                continue
            leaf_nodes = np.asarray(t.apply(tree_X), dtype=int).ravel()
            node_counts = getattr(getattr(t, "tree_", None), "n_node_samples", None)
            if node_counts is None:
                continue
            node_counts = np.asarray(node_counts, dtype=float)
            leaf_nodes = np.clip(leaf_nodes, 0, node_counts.shape[0] - 1)
            support += node_counts[leaf_nodes] / n_train_f
            n_trees += 1
        if n_trees <= 0:
            return None
        return np.clip(support / float(n_trees), 0.0, 1.0)

    booster = getattr(estimator, "booster_", None)
    if booster is None:
        return None

    leaf_idx = booster.predict(_as_2d(X), pred_leaf=True)
    leaf_idx = np.asarray(leaf_idx)
    if leaf_idx.ndim == 1:
        leaf_idx = leaf_idx.reshape(-1, 1)

    dump = booster.dump_model()
    tree_info = dump.get("tree_info", [])
    n_trees = min(len(tree_info), leaf_idx.shape[1])
    if n_trees <= 0:
        return None

    leaf_counts: list[np.ndarray] = []
    for t in range(n_trees):
        struct = tree_info[t].get("tree_structure", {})
        leaf_map = _lgbm_leaf_attr_map(struct, attr="leaf_count")
        if not leaf_map:
            return None
        max_idx = max(leaf_map)
        arr = np.full(max_idx + 1, np.nan, dtype=float)
        for k, v in leaf_map.items():
            if 0 <= k <= max_idx:
                arr[k] = float(v)
        leaf_counts.append(arr)

    support = np.zeros(leaf_idx.shape[0], dtype=float)
    for t in range(n_trees):
        idx_t = leaf_idx[:, t].astype(int)
        arr = leaf_counts[t]
        idx_t = np.clip(idx_t, 0, arr.shape[0] - 1)
        cnt = arr[idx_t]
        cnt = np.nan_to_num(cnt, nan=0.0)
        support += cnt / n_train_f

    return np.clip(support / float(n_trees), 0.0, 1.0)


def _lgbm_leaf_attr_map(tree_structure: dict[str, Any], *, attr: str) -> dict[int, float]:
    out: dict[int, float] = {}
    stack: list[dict[str, Any]] = [tree_structure]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if "leaf_index" in node:
            try:
                idx = int(node.get("leaf_index"))
                out[idx] = float(node.get(attr))
            except (TypeError, ValueError):
                continue
        else:
            left = node.get("left_child")
            right = node.get("right_child")
            if isinstance(left, dict):
                stack.append(left)
            if isinstance(right, dict):
                stack.append(right)
    return out


@dataclass(frozen=True)
class ConfidenceMetricParams:
    kind: Literal[
        "identity",
        "minmax",
        "exp_gamma",
    ]
    params: dict[str, float]


def fit_minmax_params(
    raw: Array,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> ConfidenceMetricParams:
    raw = np.asarray(raw, dtype=float).ravel()
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return ConfidenceMetricParams(kind="minmax", params={"min": 0.0, "max": 1.0})

    mn_raw = float(np.min(finite))
    mx_raw = float(np.max(finite))
    try:
        mn = float(np.quantile(finite, q_low))
        mx = float(np.quantile(finite, q_high))
    except (TypeError, ValueError):
        mn, mx = mn_raw, mx_raw

    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = mn_raw, mx_raw

    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = 0.0, 0.0

    return ConfidenceMetricParams(
        kind="minmax",
        params={
            "min": mn,
            "max": mx,
            "min_raw": mn_raw,
            "max_raw": mx_raw,
            "q_low": float(q_low),
            "q_high": float(q_high),
        },
    )


def apply_minmax(raw: Array, p: ConfidenceMetricParams) -> Array:
    raw = np.asarray(raw, dtype=float).ravel()
    mn = float(p.params.get("min", 0.0))
    mx = float(p.params.get("max", 1.0))
    if mx <= mn:
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            out = np.full_like(raw, 0.5, dtype=float)
            out[~np.isfinite(raw)] = np.nan
            return out
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo and 0.0 <= lo <= 1.0:
            out = np.full_like(raw, lo, dtype=float)
            out[~np.isfinite(raw)] = np.nan
            return out
        out = np.full_like(raw, 0.5, dtype=float)
        out[~np.isfinite(raw)] = np.nan
        return out
    conf = (raw - mn) / (mx - mn)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~np.isfinite(conf)] = np.nan
    return conf


def fit_exp_gamma_params(dist: Array, *, eps: float = 1e-12) -> ConfidenceMetricParams:
    dist = np.asarray(dist, dtype=float).ravel()
    finite = dist[np.isfinite(dist)]
    finite = finite[finite >= 0]
    if finite.size == 0:
        gamma = 1.0
    else:
        med = float(np.median(finite))
        if not np.isfinite(med) or med <= 0:
            med = float(np.mean(finite)) if finite.size else 1.0
        gamma = 1.0 / (med + float(eps))
    return ConfidenceMetricParams(kind="exp_gamma", params={"gamma": float(gamma)})


def apply_exp_gamma(dist: Array, p: ConfidenceMetricParams) -> Array:
    dist = np.asarray(dist, dtype=float).ravel()
    gamma = float(p.params.get("gamma", 1.0))
    conf = np.exp(-gamma * dist)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~np.isfinite(conf)] = np.nan
    return conf

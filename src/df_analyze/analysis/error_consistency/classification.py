"""Classification EC based on error-set intersection over union.

Each row of the input matrix is one fitted model and each column is the same
holdout sample. Pairwise EC is the IoU of the models' misclassification sets,
with an explicit policy for pairs whose error union is empty.
"""

from __future__ import annotations

from itertools import combinations
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray

from df_analyze.analysis.error_consistency.backend import ECBackendDecision
from df_analyze.analysis.error_consistency.containers import (
    ECMetricComputation,
    ECMetricInfo,
)

CLASSIFICATION_EC_INFO = ECMetricInfo(
    name="classification_iou",
    display_name="Classification error IoU",
    higher_is_better=True,
    optimal_value=1.0,
    range_min=0.0,
    range_max=1.0,
    paper_equation="1",
    samplewise_defined=False,
    scientific_status="published_classification_error_iou",
    reference_url="https://doi.org/10.3390/diagnostics13071315",
)
EMPTY_UNION_POLICIES = {"0", "1", "nan", "drop", "error", "warn"}
CUDA_PAIR_WORK_ITEMS = 2_000_000


def _as_prediction_matrix(
    predictions: ndarray, y_true: ndarray
) -> tuple[ndarray, ndarray]:
    preds = np.asarray(predictions)
    truth = np.asarray(y_true).ravel()
    if preds.ndim != 2:
        raise ValueError(
            "Classification EC predictions must have shape (n_models, n_samples). "
            f"Got shape: {preds.shape}."
        )
    if preds.shape[0] < 2:
        raise ValueError("Classification EC requires at least two prediction vectors.")
    if preds.shape[1] != truth.size:
        raise ValueError(
            "Classification predictions and target have different sample counts: "
            f"{preds.shape[1]} and {truth.size}."
        )
    return preds, truth


def _pair_counts(errors: ndarray, pairs: list[tuple[int, int]], use_cuda: bool):
    if not use_cuda:
        intersections = np.asarray(
            [np.count_nonzero(errors[i] & errors[j]) for i, j in pairs], dtype=int
        )
        unions = np.asarray(
            [np.count_nonzero(errors[i] | errors[j]) for i, j in pairs], dtype=int
        )
        return intersections, unions

    import torch

    err = torch.as_tensor(errors, dtype=torch.bool, device="cuda")
    batch_size = max(1, CUDA_PAIR_WORK_ITEMS // max(errors.shape[1], 1))
    intersections = np.empty(len(pairs), dtype=int)
    unions = np.empty(len(pairs), dtype=int)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        pair_i = torch.as_tensor([i for i, _ in batch], device="cuda")
        pair_j = torch.as_tensor([j for _, j in batch], device="cuda")
        stop = start + len(batch)
        intersections[start:stop] = (
            torch.sum(err[pair_i] & err[pair_j], dim=1).cpu().numpy()
        )
        unions[start:stop] = torch.sum(err[pair_i] | err[pair_j], dim=1).cpu().numpy()
    return intersections, unions


def _empty_union_value(policy: str) -> float | None:
    if policy == "0":
        return 0.0
    if policy == "1":
        return 1.0
    if policy == "error":
        raise ZeroDivisionError(
            "Classification EC is undefined when neither model makes an error."
        )
    if policy == "drop":
        return None
    return np.nan


def compute_classification_ec(
    predictions: ndarray,
    y_true: ndarray,
    empty_unions: str = "warn",
    backend: ECBackendDecision | None = None,
    output_detail: str = "full",
) -> ECMetricComputation:
    preds, truth = _as_prediction_matrix(predictions, y_true)
    detail = str(output_detail).lower()
    if detail not in {"summary", "pairwise", "full"}:
        raise ValueError(
            "Classification EC output detail must be summary, pairwise, or full."
        )
    retain_pairwise = detail in {"pairwise", "full"}
    policy = str(empty_unions).lower()
    if policy not in EMPTY_UNION_POLICIES:
        raise ValueError(
            f"Unknown empty-union policy '{empty_unions}'. "
            f"Expected one of {sorted(EMPTY_UNION_POLICIES)}."
        )

    errors = preds != truth[None, :]
    pairs = list(combinations(range(preds.shape[0]), 2))
    use_cuda = backend is not None and backend.resolved == "torch_cuda"
    intersections, unions = _pair_counts(errors, pairs, use_cuda)
    if policy == "warn" and np.any(unions == 0):
        warn(
            "Classification EC encountered model pairs with empty error unions; "
            "recording NaN for those pairs."
        )

    matrix = np.full((preds.shape[0], preds.shape[0]), np.nan, dtype=float)
    for model_idx in range(preds.shape[0]):
        diagonal = _empty_union_value(policy) if not np.any(errors[model_idx]) else 1.0
        if diagonal is not None:
            matrix[model_idx, model_idx] = diagonal
    values: list[float] = []
    rows = []
    for (i, j), intersection, union in zip(pairs, intersections, unions):
        value = (
            _empty_union_value(policy) if int(union) == 0 else float(intersection / union)
        )
        if value is None:
            continue
        matrix[i, j] = matrix[j, i] = value
        values.append(value)
        if retain_pairwise:
            rows.append(
                {
                    "ec_method": CLASSIFICATION_EC_INFO.name,
                    "model_i": i,
                    "model_j": j,
                    "error_intersection": int(intersection),
                    "error_union": int(union),
                    "pair_mean": value,
                }
            )

    pair_values = np.asarray(values, dtype=float)
    total_intersection = int(np.count_nonzero(np.all(errors, axis=0)))
    total_union = int(np.count_nonzero(np.any(errors, axis=0)))
    total_consistency = (
        _empty_union_value(policy)
        if total_union == 0
        else float(total_intersection / total_union)
    )

    leave_one_model_out_rows = []
    aggregate_empty_union = total_union == 0
    if preds.shape[0] >= 3:
        for model_idx in range(preds.shape[0]):
            retained = np.delete(errors, model_idx, axis=0)
            intersection = int(np.count_nonzero(np.all(retained, axis=0)))
            union = int(np.count_nonzero(np.any(retained, axis=0)))
            aggregate_empty_union = aggregate_empty_union or union == 0
            value = (
                _empty_union_value(policy) if union == 0 else float(intersection / union)
            )
            stored_value = np.nan if value is None else float(value)
            leave_one_model_out_rows.append(
                {
                    "model_removed": model_idx,
                    "n_models_retained": preds.shape[0] - 1,
                    "error_intersection": intersection,
                    "error_union": union,
                    "consistency": stored_value,
                    "empty_union": union == 0,
                    "included_in_summary": bool(np.isfinite(stored_value)),
                    "empty_union_policy": policy,
                }
            )

    if policy == "warn" and aggregate_empty_union and not np.any(unions == 0):
        warn(
            "Classification EC encountered an empty total or leave-one-out error "
            "union; recording NaN."
        )

    leave_one_model_out = pd.DataFrame(
        leave_one_model_out_rows,
        columns=pd.Index(
            [
                "model_removed",
                "n_models_retained",
                "error_intersection",
                "error_union",
                "consistency",
                "empty_union",
                "included_in_summary",
                "empty_union_policy",
            ],
            dtype=str,
        ),
    )
    consistency = leave_one_model_out.loc[:, "consistency"]
    if not isinstance(consistency, pd.Series):
        raise ValueError("Expected one consistency column.")
    finite_loo = consistency.to_numpy(dtype=float)
    finite_loo = finite_loo[np.isfinite(finite_loo)]
    leave_one_model_out_mean = (
        float(np.mean(finite_loo)) if finite_loo.size > 0 else np.nan
    )
    leave_one_model_out_sd = (
        float(np.std(finite_loo, ddof=1)) if finite_loo.size > 1 else np.nan
    )
    extra = {
        "n_model_pairs": len(pairs),
        "n_valid_model_pairs": int(np.count_nonzero(np.isfinite(pair_values))),
        "n_pair_sample_values": len(pairs) * preds.shape[1],
        "comparison_unit": "model_pair",
        "total_consistency": (np.nan if total_consistency is None else total_consistency),
        "total_error_intersection": total_intersection,
        "total_error_union": total_union,
        "empty_union_policy": policy,
        "empty_union_pairs": int(np.count_nonzero(unions == 0)),
        "leave_one_model_out_mean": leave_one_model_out_mean,
        "leave_one_model_out_sd": leave_one_model_out_sd,
        "n_leave_one_model_out": len(leave_one_model_out),
        "output_detail": detail,
    }
    if backend is not None:
        extra.update(backend.as_metadata())

    return ECMetricComputation(
        info=CLASSIFICATION_EC_INFO,
        values=pair_values,
        matrix=matrix,
        pairwise=pd.DataFrame(rows),
        leave_one_model_out=leave_one_model_out,
        extra=extra,
    )

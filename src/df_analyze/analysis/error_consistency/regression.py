from __future__ import annotations

from itertools import combinations
from typing import Iterable
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray

from df_analyze.analysis.error_consistency.backend import ECBackendDecision
from df_analyze.analysis.error_consistency.containers import (
    ECMetricComputation,
    ECMetricInfo,
)

CUDA_PAIR_WORK_ITEMS = 2_000_000
REGRESSION_EC_METRICS: dict[str, ECMetricInfo] = {
    "ratio": ECMetricInfo(
        "ratio", "Ratio", True, 1.0, 0.0, 1.0, None, True, legacy_equation_label="1"
    ),
    "ratio_diff": ECMetricInfo(
        "ratio_diff",
        "Ratio-diff",
        False,
        0.0,
        0.0,
        1.0,
        None,
        True,
        legacy_equation_label="2",
    ),
    "ratio_sign": ECMetricInfo(
        "ratio_sign",
        "Ratio-sign",
        True,
        1.0,
        -1.0,
        1.0,
        None,
        True,
        legacy_equation_label="3",
    ),
    "ratio_diff_sign": ECMetricInfo(
        "ratio_diff_sign",
        "Ratio-diff-sign magnitude",
        False,
        0.0,
        0.0,
        1.0,
        None,
        True,
        legacy_equation_label="4",
    ),
    "intersection_union_sample": ECMetricInfo(
        "intersection_union_sample",
        "Intersection-union-sample",
        True,
        1.0,
        0.0,
        1.0,
        None,
        True,
        legacy_equation_label="6",
    ),
    "intersection_union_all": ECMetricInfo(
        "intersection_union_all",
        "Intersection-union-all",
        True,
        1.0,
        0.0,
        1.0,
        None,
        False,
        legacy_equation_label="5",
    ),
    "intersection_union_distance": ECMetricInfo(
        "intersection_union_distance",
        "Intersection-union-distance",
        False,
        0.0,
        0.0,
        np.inf,
        None,
        True,
    ),
}
METHOD_ALIASES = {
    "ratio-diff": "ratio_diff",
    "ratio-signed": "ratio_sign",
    "ratio-sign": "ratio_sign",
    # ``ratio_diff_sign`` is retained as the serialized compatibility name.
    # The preferred alias makes clear that primary summaries aggregate the
    # signed value's magnitude; signed direction remains an auxiliary output.
    "ratio_diff_sign_magnitude": "ratio_diff_sign",
    "ratio-diff-sign-magnitude": "ratio_diff_sign",
    "ratio-diff-sign": "ratio_diff_sign",
    "ratio-diff-signed": "ratio_diff_sign",
    "intersection-union-sample": "intersection_union_sample",
    "intersection-union-all": "intersection_union_all",
    "intersection-union-distance": "intersection_union_distance",
}


def normalize_regression_method(method: str) -> str:
    clean = str(method).strip().lower().replace(" ", "_")
    clean = METHOD_ALIASES.get(clean, clean)
    if clean not in REGRESSION_EC_METRICS:
        known = ", ".join(REGRESSION_EC_METRICS)
        raise ValueError(
            f"Unknown regression EC method '{method}'. Known methods: {known}"
        )
    return clean


def default_regression_methods() -> list[str]:
    return list(REGRESSION_EC_METRICS)


def _same_sign_parts(r1: ndarray, r2: ndarray) -> tuple[ndarray, ndarray]:
    abs1, abs2 = np.abs(r1), np.abs(r2)
    both_positive = (r1 >= 0) & (r2 >= 0)
    both_negative = (r1 <= 0) & (r2 <= 0)
    numerator = np.where(
        both_positive,
        np.minimum(r1, r2),
        np.where(both_negative, np.minimum(abs1, abs2), 0.0),
    )
    denominator = np.where(
        both_positive,
        np.maximum(r1, r2),
        np.where(both_negative, np.maximum(abs1, abs2), abs1 + abs2),
    )
    return numerator, denominator


def regression_pairwise_consistency(
    r1: ndarray,
    r2: ndarray,
    method: str,
    epsilon: float = 0.0,
) -> ndarray:
    method = normalize_regression_method(method)
    r1 = np.asarray(r1, dtype=float).ravel()
    r2 = np.asarray(r2, dtype=float).ravel()
    if r1.shape != r2.shape:
        raise ValueError(
            f"Residual vectors have different shapes: {r1.shape}, {r2.shape}."
        )
    epsilon = max(float(epsilon), 0.0)
    abs1, abs2 = np.abs(r1), np.abs(r2)
    sign = np.sign(r1 * r2)

    with np.errstate(divide="ignore", invalid="ignore"):
        if method in {"ratio", "ratio_sign"}:
            numerator = np.minimum(abs1, abs2)
            denominator = np.maximum(abs1, abs2)
            values = np.divide(
                numerator + epsilon,
                denominator + epsilon,
                out=np.ones_like(denominator, dtype=float),
                where=denominator > 0,
            )
            # Epsilon stabilizes two non-zero residual magnitudes. It must not
            # turn a true zero-over-nonzero endpoint into positive consistency.
            values[(numerator == 0) & (denominator > 0)] = 0.0
            if method == "ratio_sign":
                values = sign * values
                values[denominator == 0] = 1.0
            return np.nan_to_num(values, nan=1.0)
        if method in {"ratio_diff", "ratio_diff_sign"}:
            denominator = abs1 + abs2
            values = np.abs(abs1 - abs2) / (denominator + epsilon)
            values[denominator == 0] = 0.0
            if method == "ratio_diff_sign":
                values = sign * values
            return np.nan_to_num(values, nan=0.0)
        if method == "intersection_union_sample":
            numerator, denominator = _same_sign_parts(r1, r2)
            return np.nan_to_num(numerator / denominator, nan=1.0)
        if method == "intersection_union_all":
            numerator, denominator = _same_sign_parts(r1, r2)
            return np.asarray(
                [np.nan_to_num(numerator.sum() / denominator.sum(), nan=1.0)]
            )
        if method == "intersection_union_distance":
            same_sign = ((r1 >= 0) & (r2 >= 0)) | ((r1 <= 0) & (r2 <= 0))
            return np.where(same_sign, np.abs(abs1 - abs2), abs1 + abs2)
    raise ValueError(f"Unhandled regression EC method: {method}")


def _torch_pairwise(r1, r2, method: str, epsilon: float):
    import torch

    abs1, abs2 = torch.abs(r1), torch.abs(r2)
    sign = torch.sign(r1 * r2)
    if method in {"ratio", "ratio_sign"}:
        numerator = torch.minimum(abs1, abs2)
        denominator = torch.maximum(abs1, abs2)
        values = torch.where(
            denominator > 0,
            (numerator + epsilon) / (denominator + epsilon),
            torch.ones_like(denominator),
        )
        values = torch.where(
            (numerator == 0) & (denominator > 0),
            torch.zeros_like(values),
            values,
        )
        if method == "ratio_sign":
            values = sign * values
            values = torch.where(denominator == 0, torch.ones_like(values), values)
        return torch.nan_to_num(values, nan=1.0)
    if method in {"ratio_diff", "ratio_diff_sign"}:
        denominator = abs1 + abs2
        values = torch.abs(abs1 - abs2) / (denominator + epsilon)
        values = torch.where(denominator == 0, torch.zeros_like(values), values)
        if method == "ratio_diff_sign":
            values = sign * values
        return torch.nan_to_num(values, nan=0.0)

    both_positive = (r1 >= 0) & (r2 >= 0)
    both_negative = (r1 <= 0) & (r2 <= 0)
    numerator = torch.where(
        both_positive,
        torch.minimum(r1, r2),
        torch.where(both_negative, torch.minimum(abs1, abs2), torch.zeros_like(r1)),
    )
    denominator = torch.where(
        both_positive,
        torch.maximum(r1, r2),
        torch.where(both_negative, torch.maximum(abs1, abs2), abs1 + abs2),
    )
    if method == "intersection_union_sample":
        return torch.nan_to_num(numerator / denominator, nan=1.0)
    if method == "intersection_union_all":
        return torch.nan_to_num(
            numerator.sum(dim=1, keepdim=True) / denominator.sum(dim=1, keepdim=True),
            nan=1.0,
        )
    if method == "intersection_union_distance":
        same_sign = both_positive | both_negative
        return torch.where(same_sign, torch.abs(abs1 - abs2), abs1 + abs2)
    raise ValueError(f"Unhandled regression EC method: {method}")


def _sample_sd(values: ndarray) -> float:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan


def compute_regression_ec(
    residuals: ndarray,
    methods: Iterable[str] | None = None,
    epsilon: float = 0.0,
    backend: ECBackendDecision | None = None,
    row_ids=None,
) -> list[ECMetricComputation]:
    residual_matrix = np.asarray(residuals, dtype=float)
    if residual_matrix.ndim != 2 or residual_matrix.shape[0] < 2:
        raise ValueError(
            "Regression EC residuals must have shape (n_models, n_samples) with "
            "at least two models."
        )
    if residual_matrix.shape[1] == 0:
        raise ValueError("Regression EC requires at least one test sample.")
    sample_ids = (
        np.arange(residual_matrix.shape[1])
        if row_ids is None
        else np.asarray(row_ids)
    )
    if sample_ids.size != residual_matrix.shape[1]:
        raise ValueError(
            f"Expected {residual_matrix.shape[1]} holdout row ids, "
            f"received {sample_ids.size}."
        )
    epsilon = max(float(epsilon), 0.0)
    selected = (
        default_regression_methods()
        if methods is None
        else list(dict.fromkeys(normalize_regression_method(m) for m in methods))
    )
    pairs = list(combinations(range(residual_matrix.shape[0]), 2))

    residual_t = None
    if backend is not None and backend.resolved == "torch_cuda":
        try:
            import torch

            residual_t = torch.as_tensor(
                residual_matrix, dtype=torch.float64, device="cuda"
            )
        except Exception as error:
            warn(
                "Could not compute regression EC on CUDA; falling back to numpy. "
                f"Details: {error}"
            )
            return compute_regression_ec(
                residual_matrix,
                methods=selected,
                epsilon=epsilon,
                backend=backend.fallback(f"torch_cuda_error:{type(error).__name__}"),
                row_ids=sample_ids,
            )

    computations = []
    for method in selected:
        info = REGRESSION_EC_METRICS[method]
        matrix = np.zeros((residual_matrix.shape[0], residual_matrix.shape[0]))
        np.fill_diagonal(matrix, info.optimal_value)
        rows = []
        pair_means: list[float] = []
        flat_count = 0
        flat_mean = 0.0
        flat_m2 = 0.0
        flat_min = np.inf
        flat_max = -np.inf
        sample_sum = (
            np.zeros(residual_matrix.shape[1], dtype=float)
            if info.samplewise_defined
            else None
        )
        sample_sum_sq = None if sample_sum is None else np.zeros_like(sample_sum)
        sample_count = (
            None if sample_sum is None else np.zeros(sample_sum.size, dtype=int)
        )
        signed_sample_sum = (
            np.zeros(residual_matrix.shape[1], dtype=float)
            if method == "ratio_diff_sign"
            else None
        )
        signed_flat_sum = 0.0

        def record_pair(pair: tuple[int, int], values: ndarray) -> None:
            nonlocal flat_count, flat_mean, flat_m2, flat_min, flat_max
            nonlocal signed_flat_sum
            i, j = pair
            values = np.asarray(values, dtype=float).ravel()
            valid = np.isfinite(values)
            signed_finite = values[valid]
            primary_values = np.abs(values) if method == "ratio_diff_sign" else values
            finite = primary_values[valid]
            pair_mean = float(np.mean(finite)) if finite.size else np.nan
            pair_means.append(pair_mean)
            matrix[i, j] = matrix[j, i] = pair_mean
            pair_row = {
                "ec_method": method,
                "model_i": i,
                "model_j": j,
                "pair_mean": pair_mean,
                "pair_sd": _sample_sd(finite),
                "pair_min": float(np.min(finite)) if finite.size else np.nan,
                "pair_max": float(np.max(finite)) if finite.size else np.nan,
            }
            if method == "ratio_diff_sign":
                pair_row.update(
                    pair_signed_mean=(
                        float(np.mean(signed_finite))
                        if signed_finite.size
                        else np.nan
                    ),
                    pair_signed_sd=_sample_sd(signed_finite),
                )
                signed_flat_sum += float(np.sum(signed_finite))
            rows.append(pair_row)

            if finite.size:
                batch_count = int(finite.size)
                batch_mean = float(np.mean(finite))
                batch_m2 = float(np.sum((finite - batch_mean) ** 2))
                combined = flat_count + batch_count
                if flat_count == 0:
                    flat_mean = batch_mean
                    flat_m2 = batch_m2
                else:
                    delta = batch_mean - flat_mean
                    flat_mean += delta * batch_count / combined
                    flat_m2 += (
                        batch_m2
                        + delta * delta * flat_count * batch_count / combined
                    )
                flat_count = combined
                flat_min = min(flat_min, float(np.min(finite)))
                flat_max = max(flat_max, float(np.max(finite)))

            if sample_sum is not None:
                if primary_values.size != sample_sum.size:
                    raise RuntimeError(
                        f"Regression EC method '{method}' returned "
                        f"{primary_values.size} "
                        f"values for {sample_sum.size} samples."
                    )
                sample_sum[valid] += primary_values[valid]
                sample_sum_sq[valid] += primary_values[valid] ** 2
                sample_count[valid] += 1
                if signed_sample_sum is not None:
                    signed_sample_sum[valid] += values[valid]

        if residual_t is None:
            for pair in pairs:
                i, j = pair
                record_pair(
                    pair,
                    regression_pairwise_consistency(
                        residual_matrix[i], residual_matrix[j], method, epsilon
                    ),
                )
        else:
            try:
                import torch

                batch_size = max(
                    1, CUDA_PAIR_WORK_ITEMS // max(residual_matrix.shape[1], 1)
                )
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start : start + batch_size]
                    pair_i = torch.as_tensor([i for i, _ in batch], device="cuda")
                    pair_j = torch.as_tensor([j for _, j in batch], device="cuda")
                    torch_values = _torch_pairwise(
                        residual_t[pair_i], residual_t[pair_j], method, epsilon
                    )
                    batch_values = torch_values.detach().cpu().numpy()
                    for pair, values in zip(batch, batch_values):
                        record_pair(pair, values)
            except Exception as error:
                warn(
                    "Could not compute regression EC on CUDA; falling back to "
                    f"numpy. Details: {error}"
                )
                return compute_regression_ec(
                    residual_matrix,
                    methods=selected,
                    epsilon=epsilon,
                    backend=backend.fallback(
                        f"torch_cuda_error:{type(error).__name__}"
                    ),
                    row_ids=sample_ids,
                )

        samplewise = None
        if info.samplewise_defined:
            with np.errstate(divide="ignore", invalid="ignore"):
                sample_mean = sample_sum / sample_count
                sample_var = (sample_sum_sq - sample_sum**2 / sample_count) / (
                    sample_count - 1
                )
                sample_var[sample_count <= 1] = np.nan
            sample_sd = np.sqrt(np.maximum(sample_var, 0.0))
            samplewise_data = {
                "row_id": sample_ids,
                "sample_idx": np.arange(residual_matrix.shape[1]),
                "ec_method": method,
                "ec_mean": sample_mean,
                "ec_sd": sample_sd,
            }
            if signed_sample_sum is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    samplewise_data["ec_signed_mean"] = (
                        signed_sample_sum / sample_count
                    )
            samplewise = pd.DataFrame(samplewise_data)
            ec_vec_sd = _sample_sd(sample_mean)
            ec_scalar_sd = _sample_sd(np.asarray(pair_means))
        else:
            ec_vec_sd = _sample_sd(np.asarray(pair_means))
            ec_scalar_sd = np.nan

        flat_sd = (
            float(np.sqrt(max(flat_m2, 0.0) / (flat_count - 1)))
            if flat_count > 1
            else np.nan
        )

        extra: dict[str, object] = {
            "ec_mean": flat_mean if flat_count else np.nan,
            "ec_sd": flat_sd,
            "ec_min": flat_min if flat_count else np.nan,
            "ec_max": flat_max if flat_count else np.nan,
            "n_comparisons": flat_count,
            "n_model_pairs": len(pairs),
            "n_pair_sample_values": flat_count,
            "comparison_unit": (
                "model_pair_x_sample" if info.samplewise_defined else "model_pair"
            ),
            "EC_scalar_sd": ec_scalar_sd,
            "EC_vec_sd": ec_vec_sd,
            # Explicit aliases avoid treating the legacy paper labels as
            # interchangeable. The model-pair SD is defined for every metric.
            "ec_pooled_value_sd": flat_sd,
            "ec_model_pair_sd": _sample_sd(np.asarray(pair_means)),
            "ec_sample_profile_sd": ec_vec_sd if info.samplewise_defined else np.nan,
            "ec_epsilon": epsilon,
        }
        if method == "ratio_diff_sign":
            extra.update(
                primary_aggregation="mean_absolute_signed_value",
                preferred_method_name="ratio_diff_sign_magnitude",
                compatibility_method_name="ratio_diff_sign",
                ec_signed_mean=(
                    signed_flat_sum / flat_count if flat_count else np.nan
                ),
            )
        if backend is not None:
            extra.update(backend.as_metadata())
        computations.append(
            ECMetricComputation(
                info=info,
                values=np.asarray(pair_means),
                matrix=matrix,
                pairwise=pd.DataFrame(rows),
                samplewise=samplewise,
                extra=extra,
            )
        )
    return computations

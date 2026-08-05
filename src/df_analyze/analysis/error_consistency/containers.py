"""Data classes and summary helpers used by EC calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import ndarray
from pandas import DataFrame


def finite_summary(values: ndarray) -> dict[str, float | int]:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "ec_mean": np.nan,
            "ec_sd": np.nan,
            "ec_min": np.nan,
            "ec_max": np.nan,
            "n_comparisons": 0,
        }
    return {
        "ec_mean": float(np.mean(vals)),
        "ec_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
        "ec_min": float(np.min(vals)),
        "ec_max": float(np.max(vals)),
        "n_comparisons": int(vals.size),
    }


@dataclass(frozen=True)
class ECMetricInfo:
    name: str
    display_name: str
    higher_is_better: bool
    optimal_value: float
    range_min: float
    range_max: float
    paper_equation: str | None
    samplewise_defined: bool
    scientific_status: str = "experimental_descriptive_diagnostic"
    reference_url: str | None = None
    legacy_equation_label: str | None = None
    ranking_supported: bool = True


@dataclass
class ECMetricComputation:
    info: ECMetricInfo
    values: ndarray
    matrix: ndarray
    pairwise: DataFrame
    samplewise: DataFrame | None = None
    leave_one_model_out: DataFrame | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        if np.isclose(self.info.optimal_value, self.info.range_min):
            optimization_direction = "minimize"
        elif np.isclose(self.info.optimal_value, self.info.range_max):
            optimization_direction = "maximize"
        else:
            optimization_direction = "toward_optimum"
        row: dict[str, Any] = {
            "ec_method": self.info.name,
            "ec_method_display": self.info.display_name,
            "higher_is_better": self.info.higher_is_better,
            "optimal_value": self.info.optimal_value,
            "optimization_direction": optimization_direction,
            "ranking_rule": (
                "minimize_absolute_distance_to_optimal_value"
                if self.info.ranking_supported
                else "not_ranked"
            ),
            "range_min": self.info.range_min,
            "range_max": self.info.range_max,
            "paper_equation": self.info.paper_equation,
            "legacy_equation_label": self.info.legacy_equation_label,
            "samplewise_defined": self.info.samplewise_defined,
            "scientific_status": self.info.scientific_status,
            "reference_url": self.info.reference_url,
            "inferential_status": "descriptive_only_not_confidence_interval",
            "ranking_supported": self.info.ranking_supported,
        }
        row.update(finite_summary(self.values))
        row.update(self.extra)
        return row


@dataclass
class ErrorConsistencyResult:
    summary: DataFrame
    performance: DataFrame
    trial_scores: DataFrame
    trial_design: DataFrame
    fold_assignments: DataFrame = field(default_factory=DataFrame)
    trial_failures: DataFrame = field(default_factory=DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

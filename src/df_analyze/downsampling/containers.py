from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from df_analyze._constants import DOWNSAMPLE_SCORE_LIMIT


@dataclass
class FeatureDownsampleResult:
    requested_method: str
    resolved_method: str
    n_features_in: int
    n_features_out: int
    selected_features: list[str]
    selected_indices: list[int] = field(default_factory=list)
    protected_features: list[str] = field(default_factory=list)
    protected_indices: list[int] = field(default_factory=list)
    scores: Optional[list[float]] = None
    score_feature_indices: Optional[list[int]] = None
    score_feature_names: Optional[list[str]] = None
    screening_rows: list[int] = field(default_factory=list, repr=False)
    tuning_rows: list[int] = field(default_factory=list, repr=False)
    screening_samples: int = 0
    tuning_samples: int = 0
    chunk_size: Optional[int] = None
    projection: bool = False
    sparse_input: bool = False
    input_format: str = "table"
    source_index_base: Optional[int] = None
    input_scan_seconds: float = 0.0
    train_load_seconds: float = 0.0
    test_load_seconds: float = 0.0
    fit_seconds: float = 0.0
    transform_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    auto_reason: Optional[str] = None
    strategy: Optional[str] = None
    ensemble_members: list[str] = field(default_factory=list)
    score_aggregation: Optional[str] = None
    stability_repeats: Optional[int] = None
    stability_subsample: Optional[float] = None
    large_feature_mode: bool = False
    # Keep full score vectors as NumPy arrays. Python lists for millions of
    # scores, indices, and names use much more memory. Stream them to CSV.
    full_score_values: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    full_score_feature_names: Optional[Sequence[str]] = field(default=None, repr=False)

    @property
    def applied(self) -> bool:
        return self.resolved_method != "none" and self.n_features_out < self.n_features_in

    @property
    def total_seconds(self) -> float:
        """Return all measured downsampling phases without double counting."""
        return float(
            self.input_scan_seconds
            + self.train_load_seconds
            + self.fit_seconds
            + self.test_load_seconds
            + self.transform_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        excluded = {
            "screening_rows",
            "tuning_rows",
            "full_score_values",
            "full_score_feature_names",
        }
        # Avoid dataclasses.asdict(): it deep-copies NumPy arrays before they
        # can be removed, defeating the bounded-memory full-score path.
        data = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in excluded
        }
        if self.scores is not None and len(self.scores) <= DOWNSAMPLE_SCORE_LIMIT:
            data["scores"] = [
                score if math.isfinite(score) else None for score in self.scores
            ]
        elif self.scores is not None:
            data.pop("scores", None)
            data.pop("score_feature_indices", None)
            data.pop("score_feature_names", None)
            data["scores_saved_separately"] = True
        if self.full_score_values is not None:
            values = np.asarray(self.full_score_values, dtype=np.float64)
            data["scores_saved_separately"] = True
            data["n_scores_saved"] = int(values.size)
            data["invalid_score_count"] = int(
                np.count_nonzero(np.isnan(values) | np.isneginf(values))
            )
        if self.source_index_base is not None:
            data["selected_source_indices"] = [
                index + self.source_index_base for index in self.selected_indices
            ]
            data["protected_source_indices"] = [
                index + self.source_index_base for index in self.protected_indices
            ]
            score_indices = data.get("score_feature_indices")
            if score_indices is not None:
                data["score_source_feature_indices"] = [
                    index + self.source_index_base for index in score_indices
                ]
        data["total_seconds"] = self.total_seconds
        data["applied"] = self.applied
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    def clone_for_reuse(self) -> FeatureDownsampleResult:
        """Clone mutable metadata while sharing score arrays treated as read-only."""
        return replace(
            self,
            selected_features=self.selected_features.copy(),
            selected_indices=self.selected_indices.copy(),
            protected_features=self.protected_features.copy(),
            protected_indices=self.protected_indices.copy(),
            scores=None if self.scores is None else self.scores.copy(),
            score_feature_indices=(
                None
                if self.score_feature_indices is None
                else self.score_feature_indices.copy()
            ),
            score_feature_names=(
                None
                if self.score_feature_names is None
                else self.score_feature_names.copy()
            ),
            screening_rows=self.screening_rows.copy(),
            tuning_rows=self.tuning_rows.copy(),
            notes=self.notes.copy(),
            ensemble_members=self.ensemble_members.copy(),
        )

    def selected_frame(self) -> pd.DataFrame:
        indices: list[Optional[int]] = self.selected_indices
        if len(indices) != len(self.selected_features):
            indices = [None] * len(self.selected_features)
        columns: dict[str, Any] = {
            "rank": range(1, len(self.selected_features) + 1),
            "feature_index": indices,
        }
        if self.source_index_base is not None:
            columns["source_feature_index"] = [
                None if index is None else index + self.source_index_base
                for index in indices
            ]
        columns["feature"] = self.selected_features
        return pd.DataFrame(columns)

    def scores_frame(self) -> Optional[pd.DataFrame]:
        """Return the bounded ranked score summary.

        Full score exports use :meth:`iter_score_frames` so a CRUSH-scale score
        vector is never expanded into one giant DataFrame.
        """
        if self.scores is None:
            return None
        names = self.score_feature_names or [None] * len(self.scores)
        indices = self.score_feature_indices or list(range(len(self.scores)))
        return self._score_frame(
            np.asarray(indices, dtype=np.int64),
            names,
            np.asarray(self.scores, dtype=np.float64),
            self.source_index_base,
        )

    @staticmethod
    def _score_frame(
        indices: NDArray[np.int64],
        names: Sequence[Optional[str]] | Sequence[str],
        values: NDArray[np.float64],
        source_index_base: Optional[int] = None,
    ) -> pd.DataFrame:
        valid = ~np.isnan(values) & ~np.isneginf(values)
        reasons = np.full(len(values), "", dtype=object)
        reasons[np.isnan(values)] = "nan"
        reasons[np.isneginf(values)] = "negative_infinity"
        columns: dict[str, Any] = {"feature_index": indices}
        if source_index_base is not None:
            columns["source_feature_index"] = indices + source_index_base
        columns.update(
            {
                "feature": list(names),
                "score": values,
                "score_valid": valid,
                "invalid_reason": reasons,
            }
        )
        return pd.DataFrame(columns)

    def iter_score_frames(self, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
        """Yield every requested score in bounded, input-order chunks."""
        if self.full_score_values is None:
            summary = self.scores_frame()
            if summary is not None:
                yield summary
            return
        if chunk_size <= 0:
            raise ValueError("Score export chunk size must be positive.")
        values = np.asarray(self.full_score_values, dtype=np.float64)
        names = self.full_score_feature_names
        if names is None or len(names) != len(values):
            raise ValueError("Full score feature names must match the score count.")
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            yield self._score_frame(
                np.arange(start, stop, dtype=np.int64),
                names[start:stop],
                values[start:stop],
                self.source_index_base,
            )

    def to_markdown(self) -> str:
        rows = [
            "# Feature downsampling",
            "",
            f"- Requested method: `{self.requested_method}`",
            f"- Resolved method: `{self.resolved_method}`",
            f"- Features: {self.n_features_in:,} -> {self.n_features_out:,}",
            f"- Screening samples: {self.screening_samples:,}",
            f"- Tuning samples: {self.tuning_samples:,}",
        ]
        if self.sparse_input:
            rows.extend(
                [
                    f"- Input layout/target scan time: {self.input_scan_seconds:.3f} s",
                    f"- Sparse training matrix load time: {self.train_load_seconds:.3f} s",
                    f"- Feature scoring time: {self.fit_seconds:.3f} s",
                    f"- Sparse test matrix load time: {self.test_load_seconds:.3f} s",
                    (
                        "- Selected-feature materialization/scaling time: "
                        f"{self.transform_seconds:.3f} s"
                    ),
                    f"- Total measured downsampling time: {self.total_seconds:.3f} s",
                ]
            )
        else:
            rows.extend(
                [
                    f"- Fit time: {self.fit_seconds:.3f} s",
                    f"- Transform time: {self.transform_seconds:.3f} s",
                    f"- Total measured downsampling time: {self.total_seconds:.3f} s",
                ]
            )
        if self.chunk_size is not None:
            rows.append(f"- Effective score chunk size: {self.chunk_size:,}")
        if self.source_index_base is not None:
            rows.append(f"- SVMlight source index base: {self.source_index_base}")
        if self.sparse_input:
            rows.append(
                "- Memory accounting: the score-chunk budget does not include the "
                "source CSR matrix, temporary sparse conversions, full-length score "
                "vectors, or selected dense outputs."
            )
        if self.notes:
            rows.extend(["", "## Notes", "", *[f"- {note}" for note in self.notes]])
        if self.protected_features:
            rows.extend(
                [
                    "",
                    "## Protected source features",
                    "",
                    f"- Count: {len(self.protected_features):,}",
                    f"- Features: {', '.join(self.protected_features[:25])}",
                ]
            )
        if self.auto_reason:
            rows.extend(["", "## Auto selection", "", self.auto_reason])
        if self.strategy:
            rows.extend(
                [
                    "",
                    "## Selection strategy",
                    "",
                    f"- Strategy: `{self.strategy}`",
                    f"- Members: {', '.join(self.ensemble_members) or 'none'}",
                    f"- Score aggregation: `{self.score_aggregation or 'none'}`",
                ]
            )
        if self.stability_repeats is not None:
            rows.extend(
                [
                    f"- Stability repeats: {self.stability_repeats}",
                    f"- Stability subsample: {self.stability_subsample}",
                ]
            )
        return "\n".join(rows) + "\n"

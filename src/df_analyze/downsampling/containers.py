from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from df_analyze._constants import DOWNSAMPLE_SCORE_LIMIT


@dataclass
class FeatureDownsampleResult:
    requested_method: str
    resolved_method: str
    n_features_in: int
    n_features_out: int
    selected_features: list[str]
    selected_indices: list[int] = field(default_factory=list)
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

    @property
    def applied(self) -> bool:
        return self.resolved_method != "none" and self.n_features_out < self.n_features_in

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("screening_rows", None)
        data.pop("tuning_rows", None)
        if self.scores is not None and len(self.scores) <= DOWNSAMPLE_SCORE_LIMIT:
            data["scores"] = [
                score if math.isfinite(score) else None for score in self.scores
            ]
        elif self.scores is not None:
            data.pop("scores", None)
            data.pop("score_feature_indices", None)
            data.pop("score_feature_names", None)
            data["scores_saved_separately"] = True
        data["applied"] = self.applied
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    def selected_frame(self) -> pd.DataFrame:
        indices: list[Optional[int]] = self.selected_indices
        if len(indices) != len(self.selected_features):
            indices = [None] * len(self.selected_features)
        return pd.DataFrame(
            {
                "rank": range(1, len(self.selected_features) + 1),
                "feature_index": indices,
                "feature": self.selected_features,
            }
        )

    def scores_frame(self) -> Optional[pd.DataFrame]:
        if self.scores is None:
            return None
        names = self.score_feature_names or [None] * len(self.scores)
        indices = self.score_feature_indices or list(range(len(self.scores)))
        return pd.DataFrame(
            {"feature_index": indices, "feature": names, "score": self.scores}
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
            f"- Fit time: {self.fit_seconds:.3f} s",
            f"- Transform time: {self.transform_seconds:.3f} s",
        ]
        if self.notes:
            rows.extend(["", "## Notes", "", *[f"- {note}" for note in self.notes]])
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

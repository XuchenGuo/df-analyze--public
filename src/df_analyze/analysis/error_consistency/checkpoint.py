"""Save and restore progress for repeated K-fold EC fits.

A checkpoint is reused only when the model, tuned parameters, EC settings,
seeds, and prepared training and holdout data match. How often checkpoints are
saved does not affect that match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.errors import EmptyDataError

from df_analyze.analysis.error_consistency.containers import ErrorConsistencyResult
from df_analyze.analysis.error_consistency.provenance import (
    EC_METHOD_IMPLEMENTATION_VERSION,
    canonical_fingerprint,
)
from df_analyze.saving import windows_io_path

CHECKPOINT_SCHEMA_VERSION = "1.0"


@dataclass
class ECPartialCheckpoint:
    completed_repetitions: int
    predictions: ndarray
    trial_design: DataFrame
    fold_assignments: DataFrame
    trial_scores: DataFrame
    trial_failures: DataFrame


def _qualified_name(value: object) -> str:
    candidate = value if isinstance(value, type) else type(value)
    return f"{candidate.__module__}.{candidate.__qualname__}"


def _json_safe(value: Any) -> Any:
    """Return a stable, JSON-serializable representation for a fingerprint."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, type):
        return _qualified_name(value)
    if (
        callable(value)
        and hasattr(value, "__module__")
        and hasattr(value, "__qualname__")
    ):
        return f"{value.__module__}.{value.__qualname__}"
    return {"type": _qualified_name(value), "value": str(value)}


def checkpoint_directory(detail_dir: Path) -> Path:
    return detail_dir / ".ec_checkpoint"


def _atomic_text(path: Path, text: str) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary_io = windows_io_path(temporary)
    temporary_io.write_text(text, encoding="utf-8")
    temporary_io.replace(windows_io_path(path))


def _atomic_frame(path: Path, frame: DataFrame) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary_io = windows_io_path(temporary)
    frame.to_csv(temporary_io, index=False, chunksize=50_000)
    temporary_io.replace(windows_io_path(path))


def _atomic_array(path: Path, values: ndarray) -> None:
    windows_io_path(path.parent).mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary_io = windows_io_path(temporary)
    DataFrame(values).to_parquet(temporary_io, index=False)
    temporary_io.replace(windows_io_path(path))


def _read_frame(path: Path) -> DataFrame:
    path_io = windows_io_path(path)
    if not path_io.is_file():
        return DataFrame()
    try:
        return pd.read_csv(path_io)
    except EmptyDataError:
        return DataFrame()


def _hash_pandas(value: DataFrame | Series) -> str:
    digest_payload = pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes()
    metadata = {
        "shape": value.shape,
        "columns": list(value.columns) if isinstance(value, DataFrame) else [value.name],
        "dtypes": (
            [str(dtype) for dtype in value.dtypes]
            if isinstance(value, DataFrame)
            else [str(value.dtype)]
        ),
        "values_hash": hashlib.sha256(digest_payload).hexdigest(),
    }
    return canonical_fingerprint(metadata)


def configuration_fingerprint(
    *,
    identity: dict[str, str],
    options,
    X_train: DataFrame,
    y_train: Series,
    X_holdout: DataFrame,
    y_holdout: Series,
    is_classification: bool,
    model_class: type,
    tuned_parameters: Any,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "identity": identity,
        "problem_type": "classification" if is_classification else "regression",
        "method_implementation_version": EC_METHOD_IMPLEMENTATION_VERSION,
        "model_class": _qualified_name(model_class),
        "tuned_parameters": _json_safe(tuned_parameters),
        "train_features": _hash_pandas(X_train),
        "train_target": _hash_pandas(y_train),
        "holdout_features": _hash_pandas(X_holdout),
        "holdout_target": _hash_pandas(y_holdout),
        "folds": int(options.ec_folds),
        "repetitions": int(options.ec_repetitions),
        "base_seed": int(options.seed),
        "model_seed_mode": str(options.ec_model_seed_mode),
        "methods": list(getattr(options, "ec_methods", None) or []),
        "empty_unions": str(options.ec_empty_unions),
        "epsilon": float(options.ec_epsilon),
        "output_detail": str(getattr(options, "ec_output_detail", "full")),
    }
    return canonical_fingerprint(payload), payload


def read_checkpoint_state(detail_dir: Path) -> dict[str, Any] | None:
    state_path = windows_io_path(checkpoint_directory(detail_dir) / "state.json")
    if not state_path.is_file():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def save_partial_checkpoint(
    detail_dir: Path,
    *,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    completed_repetitions: int,
    predictions: ndarray,
    trial_design: DataFrame,
    fold_assignments: DataFrame,
    trial_scores: DataFrame,
    trial_failures: DataFrame,
) -> None:
    directory = checkpoint_directory(detail_dir)
    _atomic_array(directory / "predictions.parquet", np.asarray(predictions))
    _atomic_frame(directory / "trial_design.csv", trial_design)
    _atomic_frame(directory / "fold_assignments.csv", fold_assignments)
    _atomic_frame(directory / "trial_scores.csv", trial_scores)
    _atomic_frame(directory / "trial_failures.csv", trial_failures)
    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "partial",
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "completed_repetitions": int(completed_repetitions),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(directory / "state.json", json.dumps(state, indent=2) + "\n")


def load_partial_checkpoint(
    detail_dir: Path,
    *,
    fingerprint: str,
) -> ECPartialCheckpoint | None:
    directory = checkpoint_directory(detail_dir)
    state = read_checkpoint_state(detail_dir)
    if state is None or state.get("fingerprint") != fingerprint:
        return None
    if state.get("status") not in {"partial", "complete"}:
        return None
    prediction_path = directory / "predictions.parquet"
    prediction_io = windows_io_path(prediction_path)
    if not prediction_io.is_file():
        return None
    predictions = pd.read_parquet(prediction_io).to_numpy()
    return ECPartialCheckpoint(
        completed_repetitions=int(state.get("completed_repetitions", 0)),
        predictions=predictions,
        trial_design=_read_frame(directory / "trial_design.csv"),
        fold_assignments=_read_frame(directory / "fold_assignments.csv"),
        trial_scores=_read_frame(directory / "trial_scores.csv"),
        trial_failures=_read_frame(directory / "trial_failures.csv"),
    )


def mark_checkpoint_complete(
    detail_dir: Path,
    *,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    completed_repetitions: int,
) -> None:
    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "complete",
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "completed_repetitions": int(completed_repetitions),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(
        checkpoint_directory(detail_dir) / "state.json",
        json.dumps(state, indent=2) + "\n",
    )
    # The final configuration file contains the complete reusable result.
    # Remove duplicate partial data but keep the small completion record.
    directory = checkpoint_directory(detail_dir)
    for name in (
        "predictions.parquet",
        "trial_design.csv",
        "fold_assignments.csv",
        "trial_scores.csv",
        "trial_failures.csv",
    ):
        windows_io_path(directory / name).unlink(missing_ok=True)


def load_completed_result(
    detail_dir: Path,
    *,
    fingerprint: str,
) -> ErrorConsistencyResult | None:
    state = read_checkpoint_state(detail_dir)
    if (
        state is None
        or state.get("status") != "complete"
        or state.get("fingerprint") != fingerprint
    ):
        return None
    required = [
        "result_summary.csv",
        "performance_summary.csv",
        "trial_scores.csv",
        "trial_design.csv",
        "fold_assignments.csv",
        "trial_failures.csv",
        "metadata.json",
    ]
    if any(not windows_io_path(detail_dir / name).is_file() for name in required):
        return None
    try:
        metadata = json.loads(
            windows_io_path(detail_dir / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None
    metadata["resumed_from_complete_checkpoint"] = True
    return ErrorConsistencyResult(
        summary=_read_frame(detail_dir / "result_summary.csv"),
        performance=_read_frame(detail_dir / "performance_summary.csv"),
        trial_scores=_read_frame(detail_dir / "trial_scores.csv"),
        trial_design=_read_frame(detail_dir / "trial_design.csv"),
        fold_assignments=_read_frame(detail_dir / "fold_assignments.csv"),
        trial_failures=_read_frame(detail_dir / "trial_failures.csv"),
        metadata=metadata,
    )

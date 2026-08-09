"""Record the data, settings, versions, and command used for an EC run.

The hashes identify the inputs used for a run. Matching records do not
guarantee identical results on every platform.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from df_analyze._constants import VERSION

EC_OUTPUT_SCHEMA_VERSION = "1.0"
EC_METHOD_IMPLEMENTATION_VERSION = "2026-08-reference-and-magnitude-v1"
_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "matplotlib",
)
_SENSITIVE_MARKERS = ("token", "password", "secret", "api-key", "api_key")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_record(path: object) -> dict[str, Any]:
    candidate = Path(str(path)).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(candidate),
        "exists": candidate.is_file(),
    }
    if candidate.is_file():
        stat = candidate.stat()
        record.update(
            {
                "size_bytes": int(stat.st_size),
                "sha256": sha256_file(candidate),
            }
        )
    return record


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> str | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        git = parent / ".git"
        if git.is_file():
            text = git.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                git = (parent / text.split(":", 1)[1].strip()).resolve()
        if not git.is_dir():
            continue
        head = git / "HEAD"
        if not head.is_file():
            return None
        value = head.read_text(encoding="utf-8", errors="replace").strip()
        if not value.startswith("ref:"):
            return value or None
        ref = git / value.split(":", 1)[1].strip()
        if ref.is_file():
            return ref.read_text(encoding="utf-8", errors="replace").strip() or None
        packed = git / "packed-refs"
        if packed.is_file():
            ref_name = value.split(":", 1)[1].strip()
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref_name:
                        return commit
        return None
    return None


def _redacted_argv(argv: Iterable[object]) -> list[str]:
    values = [str(item) for item in argv]
    redacted: list[str] = []
    hide_next = False
    for value in values:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        lowered = value.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            if "=" in value:
                redacted.append(value.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(value)
                hide_next = True
        else:
            redacted.append(value)
    return redacted


def build_reproducibility_manifest(
    options,
    *,
    metadata: dict[str, Any],
    trial_failures: int,
) -> dict[str, Any]:
    input_paths = []
    datapath = getattr(options, "datapath", None)
    if datapath is not None:
        input_paths.append(datapath)
    input_paths.extend(getattr(options, "test_paths", []) or [])
    inputs = []
    seen = set()
    for path in input_paths:
        resolved = str(Path(path).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        inputs.append(_input_record(path))

    settings = {
        "folds": getattr(options, "ec_folds", None),
        "repetitions": getattr(options, "ec_repetitions", None),
        "model_seed_mode": getattr(options, "ec_model_seed_mode", None),
        "base_seed": getattr(options, "seed", None),
        "methods": metadata.get("ec_methods", []),
        "holdout_role": getattr(options, "ec_holdout_role", None),
        "test_val_size": getattr(options, "test_val_size", None),
        "empty_union_policy": getattr(options, "ec_empty_unions", None),
        "epsilon": getattr(options, "ec_epsilon", None),
        "output_detail": getattr(options, "ec_output_detail", "full"),
        "save_predictions": bool(getattr(options, "ec_save_predictions", False)),
    }
    return {
        "schema_version": EC_OUTPUT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "df_analyze_version": VERSION,
        "git_commit": _git_commit(),
        "method_implementation_version": EC_METHOD_IMPLEMENTATION_VERSION,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": _dependency_versions(),
        "command": _redacted_argv(getattr(options, "cli_argv", sys.argv)),
        "working_directory": os.getcwd(),
        "inputs": inputs,
        "resolved_ec_settings": settings,
        "targets": metadata.get("targets", []),
        "n_configurations": metadata.get("n_configurations", 0),
        "n_failed_trials": int(trial_failures),
        "audit_files": {
            "fold_assignments": "fold_assignments.csv",
            "trial_design": "trial_design.csv",
            "trial_failures": "trial_failures.csv",
            "selection_guard": "selection_guard.csv",
        },
    }


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

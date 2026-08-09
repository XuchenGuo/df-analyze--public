from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mode", "model_flag", "holdout_role"),
    [
        ("classify", "--classifiers", "test"),
        ("regress", "--regressors", "validation"),
    ],
    ids=["classification", "regression"],
)
def test_error_consistency_full_cli(
    tmp_path: Path,
    mode: str,
    model_flag: str,
    holdout_role: str,
) -> None:
    rng = np.random.default_rng(20260802)
    n_samples = 240
    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    target = (
        (x0 + 0.35 * x1 + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)
        if mode == "classify"
        else 1.5 * x0 - 0.75 * x1 + rng.normal(scale=0.25, size=n_samples)
    )
    dataset = tmp_path / f"{mode}_ec_input.parquet"
    pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "target": target}).to_parquet(
        dataset, index=False
    )
    output = tmp_path / f"{mode}_ec"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "df-analyze.py"),
            "--df",
            str(dataset),
            "--target",
            "target",
            "--mode",
            mode,
            model_flag,
            "dtree",
            "--feat-select",
            "none",
            "--htune-trials",
            "1",
            "--test-val-size",
            "0.2",
            "--error-consistency",
            "--ec-folds",
            "2",
            "--ec-repetitions",
            "2",
            "--ec-holdout-role",
            holdout_role,
            "--ec-save-predictions",
            "--outdir",
            str(output),
            "--device",
            "cpu",
            "--seed",
            "42",
            "--verbosity",
            "0",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )

    roots = [
        path
        for path in output.rglob("error_consistency")
        if path.parent.name == "results"
    ]
    assert len(roots) == 1
    ec_root = roots[0]
    summary = pd.read_csv(ec_root / "summary.csv")
    assert set(summary["n_ec_models"]) == {4}
    assert set(summary["n_folds"]) == {2}
    assert set(summary["n_repetitions"]) == {2}
    failures = pd.read_csv(ec_root / "trial_failures.csv")
    assert failures.empty
    manifest = json.loads(
        (ec_root / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resolved_ec_settings"]["folds"] == 2
    assert manifest["resolved_ec_settings"]["repetitions"] == 2
    assert manifest["inputs"][0]["path"] == str(dataset.resolve())
    assert len(manifest["inputs"][0]["sha256"]) == 64

    predictions_path = next(ec_root.rglob("trial_predictions.csv"))
    predictions = pd.read_csv(predictions_path)
    assert {"row_id", "holdout_position", "y_true"}.issubset(predictions.columns)
    assert predictions.filter(regex=r"^model_\d+$").shape[1] == 4

    guard = pd.read_csv(ec_root / "selection_guard.csv")
    assert guard["holdout_role"].item() == holdout_role
    ranking = pd.read_csv(ec_root / "model_ec_ranking.csv")
    if holdout_role == "test":
        assert ranking.empty
    else:
        assert not ranking.empty

    if mode == "classify":
        leave_one_model_out = pd.read_csv(next(ec_root.rglob("leave_one_model_out.csv")))
        assert leave_one_model_out["model_removed"].tolist() == [0, 1, 2, 3]
    else:
        assert "ratio_diff_sign_magnitude" in set(summary["ec_method"])

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from benchmark_df_analyze_gpu import (  # noqa: E402
    _compare_performance,
    _validate_resolution,
    summarize_runs,
)
from benchmark_gpu_learners import DATASETS, _configs  # noqa: E402
import benchmark_downsampling as downsampling_benchmark  # noqa: E402
from gpu_common import write_json  # noqa: E402


@pytest.mark.fast
def test_summarize_paired_runs_uses_medians() -> None:
    runs = [
        {"device": "cpu", "reported_seconds": 10.0, "wall_seconds": 12.0},
        {"device": "cuda", "reported_seconds": 4.0, "wall_seconds": 6.0},
        {"device": "cpu", "reported_seconds": 14.0, "wall_seconds": 16.0},
        {"device": "cuda", "reported_seconds": 6.0, "wall_seconds": 8.0},
    ]

    summary = summarize_runs(runs)

    assert summary["cpu_median_reported_seconds"] == 12.0
    assert summary["gpu_median_reported_seconds"] == 5.0
    assert summary["reported_seconds_saved"] == 7.0
    assert summary["reported_percent_saved"] == pytest.approx(100 * 7 / 12)
    assert summary["reported_speedup"] == pytest.approx(12 / 5)


@pytest.mark.fast
def test_validate_resolution_rejects_silent_cpu_fallback() -> None:
    timing = {"resolved_devices": {"knn": "cpu", "xgboost": "cuda"}}

    with pytest.raises(RuntimeError, match="knn"):
        _validate_resolution(timing, ["knn", "xgb"], "cuda")

    _validate_resolution(
        {"resolved_devices": {"knn": "cuda", "xgboost": "cuda"}},
        ["knn", "xgb"],
        "cuda",
    )


@pytest.mark.fast
def test_compare_performance_matches_identity_columns() -> None:
    base = {
        "repeat": 1,
        "model": "knn",
        "selection": "none",
        "embed_selector": "none",
        "metric": "acc",
        "split": "holdout",
    }
    rows = [
        {**base, "device": "cpu", "value": 0.8},
        {**base, "device": "cuda", "value": 0.79},
    ]

    comparison = _compare_performance(rows)

    assert len(comparison) == 1
    assert comparison[0]["cpu"] == 0.8
    assert comparison[0]["cuda"] == 0.79
    assert comparison[0]["abs_diff"] == pytest.approx(0.01)


@pytest.mark.fast
def test_learner_benchmark_covers_multiple_tasks_and_knn_configs() -> None:
    assert {item.task for item in DATASETS.values()} == {"classification", "regression"}
    configs = _configs("knn")
    assert len(configs) == 6
    assert {item.args["metric"] for item in configs} == {
        "l2",
        "cosine",
        "correlation",
    }


@pytest.mark.fast
def test_benchmark_json_is_strict_for_nonfinite_values(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    write_json(output, {"nan": float("nan"), "infinity": float("inf")})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "infinity": None,
        "nan": None,
    }


@pytest.mark.fast
def test_downsampling_benchmark_warms_up_and_keeps_raw_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def select(*args, **kwargs):
        calls.append(True)
        return [0, 1], SimpleNamespace(resolved_method="rank_ensemble", chunk_size=4)

    ticks = iter([0.0, 1.0, 2.0, 5.0, 6.0, 11.0])
    monkeypatch.setattr(downsampling_benchmark, "select_indexed_columns", select)
    monkeypatch.setattr(downsampling_benchmark, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        downsampling_benchmark, "environment_metadata", lambda: {"python": "test"}
    )

    result = downsampling_benchmark.benchmark(8, 4, 2, 4, repeats=3, warmups=2)

    assert len(calls) == 5
    assert result["median_seconds"] == 3.0
    assert [run["seconds"] for run in result["runs"]] == [1.0, 3.0, 5.0]
    assert result["environment"] == {"python": "test"}

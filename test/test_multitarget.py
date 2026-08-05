from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cli_test_helpers import ArgvContext
from pandas import DataFrame, Series

from df_analyze.cli.cli import get_options
from df_analyze.enumerables import ClassifierScorer
from df_analyze.hypertune import EvaluationResults, HtuneResult
from df_analyze.models.dummy import DummyClassifier
from df_analyze.multitarget import _eval_results_for_target
from df_analyze.selection.multitarget import _rank_aggregate


def test_multitarget_cli_options(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    DataFrame({"feature": [0], "a": [0], "b": [1]}).to_csv(data, index=False)

    with ArgvContext(
        "df-analyze.py",
        "--df",
        str(data),
        "--targets",
        "a,b",
        "--mt-agg-strategy",
        "freq",
        "--mt-top-k",
        "12",
        "--outdir",
        str(tmp_path / "output"),
    ):
        options = get_options()

    assert options.targets == ["a", "b"]
    assert options.target == "a"
    assert options.mt_agg_strategy == "freq"
    assert options.mt_top_k == 12


def test_multitarget_cli_rejects_space_separated_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data.csv"
    DataFrame({"feature": [0], "a": [0], "b": [1]}).to_csv(data, index=False)

    with ArgvContext(
        "df-analyze.py",
        "--df",
        str(data),
        "--targets",
        "a",
        "b",
        "--outdir",
        str(tmp_path / "output"),
    ):
        with pytest.raises(SystemExit) as error:
            get_options()

    assert error.value.code == 2
    assert "unrecognized arguments: b" in capsys.readouterr().err


def test_multitarget_cli_rejects_duplicate_targets(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    DataFrame({"feature": [0], "a": [0]}).to_csv(data, index=False)

    with ArgvContext(
        "df-analyze.py",
        "--df",
        str(data),
        "--targets",
        "a,a",
        "--outdir",
        str(tmp_path / "output"),
    ):
        with pytest.raises(ValueError, match="must be unique"):
            get_options()


def test_borda_scores_are_comparable_across_candidate_set_sizes() -> None:
    long_scores = Series(
        np.arange(100, 0, -1, dtype=float),
        index=["long_best", *[f"f{i}" for i in range(98)], "shared"],
    )
    scores = {
        "short": Series([2.0, 1.0], index=["shared", "short_other"]),
        "long": long_scores,
    }

    ranked = _rank_aggregate(scores, higher_is_better=True)

    assert ranked.index[0] == "shared"


def test_target_result_reconstruction_uses_internal_target_score_and_table() -> None:
    X_train = DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    X_test = DataFrame({"feature": [4.0, 5.0]})
    y_train = DataFrame({"target_a": [0, 1, 0, 1], "target_b": [1, 0, 1, 0]})
    y_test = DataFrame({"target_a": [0, 1], "target_b": [1, 0]})
    preds_train = y_train.copy()
    preds_test = DataFrame({"target_a": [1, 1], "target_b": [1, 0]})
    probs_train = {
        target: np.column_stack([1.0 - y_train[target], y_train[target]])
        for target in y_train
    }
    probs_test = {
        target: np.column_stack([1.0 - y_test[target], y_test[target]])
        for target in y_test
    }
    model = DummyClassifier()
    result = HtuneResult(
        selection="none",
        selected_cols=["feature"],
        embed_select_model=None,
        model_cls=DummyClassifier,
        model=model,
        params={},
        metric=ClassifierScorer.Accuracy,
        score=0.625,
        preds_test=preds_test,
        preds_train=preds_train,
        probs_test=probs_test,
        probs_train=probs_train,
        per_target_tuning_scores={
            "target_a": 0.61,
            "target_b": 0.79,
        },
    )
    aggregate = DataFrame(
        {
            "metric": ["acc"],
            "trainset": [1.0],
            "holdout": [0.75],
            "5-fold": [0.7],
            "model": ["dummy"],
            "selection": ["none"],
            "embed_selector": ["none"],
        }
    )
    per_target = DataFrame(
        {
            "target": ["target_a", "target_b"],
            "metric": ["acc", "acc"],
            "trainset": [1.0, 1.0],
            "holdout": [0.5, 1.0],
            "5-fold": [0.6, 0.8],
            "model": ["dummy", "dummy"],
            "selection": ["none", "none"],
            "embed_selector": ["none", "none"],
        }
    )
    evaluation = EvaluationResults(
        df=aggregate,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        results=[result],
        is_classification=True,
        per_target_df=per_target,
    )

    target_eval = _eval_results_for_target(
        evaluation,
        SimpleNamespace(X=X_train, y=Series(y_train["target_a"])),
        SimpleNamespace(X=X_test, y=Series(y_test["target_a"])),
        "target_a",
    )

    assert target_eval.results[0].score == 0.61
    assert target_eval.results[0].per_target_tuning_scores == {"target_a": 0.61}
    assert target_eval.df["target"].tolist() == ["target_a"]
    assert target_eval.df["holdout"].tolist() == [0.5]


def test_multitarget_tuning_records_internal_target_scores() -> None:
    n = 60
    X = DataFrame(
        {
            "x1": np.linspace(-1.0, 1.0, n),
            "x2": np.tile([0.0, 1.0, 2.0], n // 3),
        }
    )
    y = DataFrame(
        {
            "target_a": np.tile([0, 0, 1, 1], n // 4),
            "target_b": np.tile([0, 1, 0, 1], n // 4),
        }
    )
    model = DummyClassifier()

    study = model.htune_optuna(
        X_train=X,
        y_train=y,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=1,
        n_jobs=1,
    )

    expected_targets = {"target_a", "target_b"}
    assert set(model.per_target_tuning_scores) == expected_targets
    assert set(study.best_trial.user_attrs["per_target_tuning_scores"]) == expected_targets
    assert all(np.isfinite(score) for score in model.per_target_tuning_scores.values())


def test_per_target_tuning_scores_prediction_json_roundtrip(tmp_path: Path) -> None:
    result = HtuneResult(
        selection="none",
        selected_cols=["feature"],
        embed_select_model=None,
        model_cls=DummyClassifier,
        model=DummyClassifier(),
        params={"strategy": "prior"},
        metric=ClassifierScorer.Accuracy,
        score=0.7,
        preds_test=Series([0, 1]),
        preds_train=Series([0, 1, 0, 1]),
        probs_test=None,
        probs_train=None,
        downsample_requested="auto",
        downsample_resolved="f-test",
        n_downsampled_features=17,
        per_target_tuning_scores={"target_a": 0.61, "target_b": 0.79},
    )
    payload = '{"predictions": [' + result.to_preds_json() + "]}"
    (tmp_path / "prediction_results.json").write_text(payload, encoding="utf-8")

    loaded = EvaluationResults.load_preds(tmp_path)

    assert loaded[0].per_target_tuning_scores == {
        "target_a": pytest.approx(0.61),
        "target_b": pytest.approx(0.79),
    }
    assert loaded[0].downsample_requested == "auto"
    assert loaded[0].downsample_resolved == "f-test"
    assert loaded[0].n_downsampled_features == 17

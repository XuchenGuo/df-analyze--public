from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import jsonpickle
import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Linear, Module

import df_analyze.models.tabpfn as tabpfn_module
from df_analyze._main import _run
from df_analyze.cli.cli import ProgramOptions, get_options, make_parser
from df_analyze.enumerables import (
    ClassifierScorer,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    TabPFNVersion,
)
from df_analyze.hypertune import EvaluationResults, HtuneResult, evaluate_tuned
from df_analyze.models.kan import KANEstimator, SkorchKAN
from df_analyze.models.tabpfn import (
    TABPFN_CLASSIFIERS,
    TABPFN_PRETRAINING_LIMITS,
    TABPFN_REGRESSORS,
    TABPFN_V3_MODEL_CARD_MAX_FEATURES,
    TabPFNClassifierV3,
    TabPFNClassifierV26,
    TabPFNRegressorV25,
    TabPFNSetupError,
)
from df_analyze.models.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from df_analyze.models.xgboost import XGBoostClassifier, XGBoostRegressor
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import PreparedData, prepare_data
from df_analyze.runtime.hardware import get_runtime
from df_analyze.selection.models import ModelSelected


def numeric_data(n: int = 40) -> DataFrame:
    rng = np.random.default_rng(42)
    return DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])


RUN_REAL_MODEL_TESTS = os.getenv("DF_ANALYZE_RUN_REAL_MODEL_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
RUN_REAL_TABPFN_TESTS = os.getenv(
    "DF_ANALYZE_RUN_REAL_TABPFN_TESTS", ""
).lower() in {"1", "true", "yes"}


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="set DF_ANALYZE_RUN_REAL_MODEL_TESTS=1 to exercise real model backends",
)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_kan_backend_smoke(task: str) -> None:
    X = numeric_data(24)
    if task == "classification":
        y = Series((X["x0"] + X["x1"] > 0).astype(int), name="target")
        num_classes = 2
    else:
        y = Series(X["x0"] - 0.5 * X["x1"], name="target")
        num_classes = 1
    model = KANEstimator(
        num_classes=num_classes,
        model_args={
            "max_epochs": 1,
            "batch_size": 8,
            "train_split": None,
            "verbose": 0,
        },
    )
    model.set_runtime(get_runtime("cpu"))

    model.fit(X, y)

    predictions = np.asarray(model.predict(X))
    assert predictions.shape == (len(y),)
    assert np.isfinite(predictions).all()
    if task == "classification":
        assert np.asarray(model.predict_proba_untuned(X)).shape == (len(y), 2)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_REAL_TABPFN_TESTS,
    reason=(
        "set DF_ANALYZE_RUN_REAL_TABPFN_TESTS=1 after accepting the checkpoint "
        "license and configuring TABPFN_TOKEN"
    ),
)
@pytest.mark.parametrize("version", ["v3", "v2.6", "v2.5"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_tabpfn_backend_smoke(version: str, task: str) -> None:
    X = numeric_data(24)
    if task == "classification":
        model_cls = TABPFN_CLASSIFIERS[version]
        y = Series((X["x0"] + X["x1"] > 0).astype(int), name="target")
    else:
        model_cls = TABPFN_REGRESSORS[version]
        y = Series(X["x0"] - 0.5 * X["x1"], name="target")
    model = model_cls(
        model_args={
            "n_estimators": 1,
            "auto_scale_n_estimators": False,
        }
    )
    model.set_runtime(get_runtime("cpu"))

    model.fit(X, y)

    predictions = np.asarray(model.predict(X))
    assert predictions.shape == (len(y),)
    assert np.isfinite(predictions).all()
    if task == "classification":
        assert np.asarray(model.predict_proba_untuned(X)).shape == (len(y), 2)


@pytest.mark.fast
def test_prepared_data_preserves_tabpfn_raw_view_and_lineage() -> None:
    n = 40
    processed = DataFrame(
        {
            "age": np.arange(n, dtype=float),
            "age_NAN": [1.0, *([0.0] * (n - 1))],
            "city_a": [1.0, 0.0] * (n // 2),
            "city_b": [0.0, 1.0] * (n // 2),
        }
    )
    raw = DataFrame(
        {
            "age": [np.nan, *np.arange(1, n, dtype=float)],
            "city": Series(["a", "b"] * (n // 2), dtype="category"),
        }
    )
    prepared = PreparedData(
        X=processed,
        X_tabpfn=raw,
        X_cont=DataFrame({"age": np.arange(n, dtype=float)}),
        X_cat=DataFrame({"city": ["a", "b"] * (n // 2)}),
        y=Series([0, 1] * (n // 2), name="target"),
        groups=None,
        is_classification=True,
        validate=False,
    )

    selected = prepared.model_matrix(TabPFNClassifierV3, ["city_a", "age_NAN"])
    assert selected.columns.tolist() == ["city", "age"]
    assert selected["age"].isna().sum() == 1
    assert prepared.model_matrix(object, ["city_a"]).columns.tolist() == ["city_a"]
    assert prepared.model_matrix(TabPFNClassifierV3, slice(None)).equals(raw)
    assert prepared.model_matrix(object, slice(None)).equals(processed)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        prepared.save_raw(root)
        loaded = PreparedData.from_saved(root, inspection=None)  # type: ignore[arg-type]

    assert loaded.X_tabpfn.equals(prepared.X_tabpfn)
    assert loaded.feature_lineage == prepared.feature_lineage


@pytest.mark.fast
def test_prepare_data_builds_unimputed_tabpfn_view() -> None:
    n = 60
    frame = DataFrame(
        {
            "age": [np.nan, *[float(i % 10) for i in range(1, n)]],
            "city": ["a", "b"] * (n // 2),
            "target": [0, 1] * (n // 2),
        }
    )
    inspected, inspection = inspect_data(
        df=frame,
        target="target",
        grouper=None,
        categoricals=["city"],
        _warn=False,
    )
    prepared = prepare_data(
        inspected,
        target="target",
        grouper=None,
        results=inspection,
        is_classification=True,
        _warn=False,
    )

    assert prepared.X_tabpfn.columns.tolist() == ["age", "city"]
    assert prepared.X_tabpfn["age"].isna().sum() == 1
    assert isinstance(prepared.X_tabpfn["city"].dtype, pd.CategoricalDtype)
    assert not prepared.X.isna().any().any()


@pytest.mark.fast
def test_added_models_are_registered() -> None:
    classifiers = {
        DfAnalyzeClassifier.XGBoost: XGBoostClassifier,
        DfAnalyzeClassifier.TabPFN: TABPFN_CLASSIFIERS["v3"],
        DfAnalyzeClassifier.DecisionTree: DecisionTreeClassifier,
        DfAnalyzeClassifier.ExtraTrees: ExtraTreesClassifier,
        DfAnalyzeClassifier.KAN: KANEstimator,
    }
    regressors = {
        DfAnalyzeRegressor.XGBoost: XGBoostRegressor,
        DfAnalyzeRegressor.TabPFN: TABPFN_REGRESSORS["v3"],
        DfAnalyzeRegressor.DecisionTree: DecisionTreeRegressor,
        DfAnalyzeRegressor.ExtraTrees: ExtraTreesRegressor,
        DfAnalyzeRegressor.KAN: KANEstimator,
    }
    for source, model in classifiers.items():
        assert source.value in DfAnalyzeClassifier.choices()
        assert source.get_model() is model
    for source, model in regressors.items():
        assert source.value in DfAnalyzeRegressor.choices()
        assert source.get_model() is model


@pytest.mark.fast
@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(),
        ExtraTreesClassifier(model_args={"n_estimators": 5, "n_jobs": 1}),
    ],
)
def test_tree_classifiers_support_multitarget(model: Any) -> None:
    X = numeric_data()
    y = DataFrame(
        {
            "first": (X["x0"] > 0).astype(int),
            "second": (X["x1"] + X["x2"] > 0).astype(int),
        }
    )
    model.fit(X, y)
    assert np.asarray(model.predict(X)).shape == y.shape
    model.refit_tuned(X, y, tuned_args={"max_depth": 3})
    assert np.asarray(model.tuned_predict(X)).shape == y.shape
    probabilities = model.predict_proba(X)
    assert isinstance(probabilities, list)
    assert len(probabilities) == y.shape[1]


@pytest.mark.fast
@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(),
        ExtraTreesRegressor(model_args={"n_estimators": 5, "n_jobs": 1}),
    ],
)
def test_tree_regressors_support_multitarget(model: Any) -> None:
    X = numeric_data()
    y = DataFrame({"first": X["x0"] + X["x1"], "second": X["x2"] ** 2})
    model.fit(X, y)
    assert np.asarray(model.predict(X)).shape == y.shape
    model.refit_tuned(X, y, tuned_args={"max_depth": 3})
    assert np.asarray(model.tuned_predict(X)).shape == y.shape


@pytest.mark.parametrize(
    ("mode", "model_option", "expected_metrics"),
    [
        (
            "classify",
            "--classifiers dtree et",
            {"subset-acc", "hamming-loss", "hamming-acc"},
        ),
        (
            "regress",
            "--regressors dtree et",
            {"multi-rmse", "multi-rmse-uniform", "multi-r2"},
        ),
    ],
    ids=["classification", "regression"],
)
def test_tree_models_complete_multitarget_cli_pipeline(
    tmp_path: Path,
    mode: str,
    model_option: str,
    expected_metrics: set[str],
) -> None:
    rng = np.random.default_rng(20260726)
    n_rows = 300
    frame = DataFrame(
        {
            "x0": rng.normal(size=n_rows),
            "x1": rng.normal(size=n_rows),
            "x2": rng.normal(size=n_rows),
            "group": np.repeat(np.arange(12), n_rows // 12),
        }
    )
    if mode == "classify":
        rows = np.arange(n_rows)
        frame["target_a"] = rows % 2
        frame["target_b"] = (rows // 2) % 3
        frame["x0"] += frame["target_a"]
        frame["x1"] += frame["target_b"]
    else:
        frame["target_a"] = 2.0 * frame["x0"] - frame["x1"]
        frame["target_b"] = frame["x0"] + frame["x2"] ** 2

    data_path = tmp_path / f"tree_{mode}.csv"
    output_path = tmp_path / f"tree_{mode}_results"
    frame.to_csv(data_path, index=False)
    options = get_options(
        f"--df {data_path} --targets target_a,target_b --mode {mode} "
        f"--grouper group {model_option} --feat-select none --no-preds --htune-trials 1 "
        f"--test-val-size 0.25 --device cpu --outdir {output_path}"
    )

    _run(options)

    tables = list(output_path.rglob("final_performances.csv"))
    assert len(tables) == 1
    results = pd.read_csv(tables[0])
    assert {"dtree", "et"} <= set(results["model"])
    assert expected_metrics <= set(results["metric"])
    assert results["final_cv_folds"].nunique() == 1
    assert 2 <= int(results["final_cv_folds"].iloc[0]) < 5
    assert {"dtree", "et"} <= {
        entry["model"] for entry in options._model_successes
    }
    assert options._model_failures == []


@pytest.mark.fast
def test_xgboost_multitarget_protocol() -> None:
    pytest.importorskip("xgboost")
    X = numeric_data(30)
    y_cls = DataFrame(
        {
            "first": (X["x0"] > 0).astype(int),
            "second": (X["x1"] > 0).astype(int),
        }
    )
    args = {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    classifier = XGBoostClassifier(model_args=args)
    classifier.fit(X, y_cls)
    assert classifier.predict(X).shape == y_cls.shape
    probabilities = classifier.predict_proba_untuned(X)
    assert isinstance(probabilities, dict)
    assert set(probabilities) == set(y_cls.columns)
    study = classifier.htune_optuna(
        X,
        y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=1,
    )
    assert study.best_trial.number == 0

    y_reg = DataFrame({"first": X["x0"] + X["x1"], "second": X["x2"]})
    regressor = XGBoostRegressor(model_args=args)
    regressor.refit_tuned(X, y_reg, tuned_args={"learning_rate": 0.2})
    assert regressor.tuned_predict(X).shape == y_reg.shape


@pytest.mark.fast
def test_xgboost_classifier_handles_noncontiguous_encoded_labels() -> None:
    pytest.importorskip("xgboost")
    X = numeric_data(30)
    y = Series(np.where(X["x0"] > 0, 2, 0), name="target")
    model = XGBoostClassifier(
        model_args={"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    )

    model.fit(X, y)

    assert set(np.asarray(model.predict(X))) <= {0, 2}
    assert model.predict_proba_untuned(X).shape[1] == 2


@pytest.mark.fast
def test_kan_output_width_preserves_missing_encoded_class() -> None:
    assert KANEstimator._num_classes(Series([0, 2, 2]), True) == 3


@pytest.mark.fast
def test_xgboost_jsonpickle_multitarget_roundtrip() -> None:
    pytest.importorskip("xgboost")
    X = numeric_data(30)
    y = DataFrame(
        {
            "first": (X["x0"] > 0).astype(int),
            "second": (X["x1"] > 0).astype(int),
        }
    )
    model = XGBoostClassifier(
        model_args={"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    )
    model.fit(X, y)
    model.refit_tuned(X, y, tuned_args={"learning_rate": 0.2})

    restored = jsonpickle.decode(jsonpickle.encode(model, unpicklable=True))

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(restored.tuned_predict(X), model.tuned_predict(X))
    expected = model.predict_proba(X)
    actual = restored.predict_proba(X)
    assert isinstance(expected, dict)
    assert isinstance(actual, dict)
    for target in y.columns:
        np.testing.assert_allclose(actual[target], expected[target])

    y_reg = DataFrame({"first": X["x0"] + X["x1"], "second": X["x2"]})
    regressor = XGBoostRegressor(
        model_args={"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    )
    regressor.refit_tuned(X, y_reg, tuned_args={"learning_rate": 0.2})
    restored_regressor = jsonpickle.decode(
        jsonpickle.encode(regressor, unpicklable=True)
    )
    np.testing.assert_allclose(
        restored_regressor.tuned_predict(X), regressor.tuned_predict(X)
    )


@pytest.mark.fast
def test_xgboost_evaluation_results_roundtrip() -> None:
    pytest.importorskip("xgboost")
    X = numeric_data(40)
    y = Series((X["x0"] + X["x1"] > 0).astype(int), name="target")
    model = XGBoostClassifier(
        model_args={"n_estimators": 3, "max_depth": 2, "n_jobs": 1}
    )
    model.refit_tuned(X, y, tuned_args={})
    predictions = model.tuned_predict(X)
    probabilities = model.predict_proba(X)
    result = HtuneResult(
        selection="none",
        selected_cols=X.columns.to_list(),
        embed_select_model=None,
        model_cls=XGBoostClassifier,
        model=model,
        params={},
        metric=ClassifierScorer.Accuracy,
        score=1.0,
        preds_test=predictions,
        preds_train=predictions,
        probs_test=probabilities,
        probs_train=probabilities,
    )
    evaluation = EvaluationResults(
        df=DataFrame({"model": ["xgb"], "selection": ["none"]}),
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        results=[result],
        is_classification=True,
    )

    with TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        evaluation.save(root, fold_idx=None)
        loaded = EvaluationResults.load(root)

    restored = loaded.results[0].model
    np.testing.assert_array_equal(restored.tuned_predict(X), predictions)
    np.testing.assert_allclose(restored.predict_proba(X), probabilities)


class FakeKAN(Module):
    def __init__(self, width: list[int], **_: Any) -> None:
        super().__init__()
        self.layer = Linear(width[0], width[-1])
        self.input_id = torch.arange(width[0])

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


@pytest.mark.fast
def test_kan_skorch_adapter_and_multitarget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("df_analyze.models.kan.official_kan_cls", lambda: FakeKAN)
    module = SkorchKAN(input_dim=5, width=4, depth=1, num_classes=2)
    assert module(torch.zeros((3, 5))).shape == (3, 2)

    X = numeric_data(40)
    y = DataFrame(
        {
            "first": (X["x0"] > 0).astype(int),
            "second": (X["x1"] > 0).astype(int),
        }
    )
    model = KANEstimator(
        num_classes=2,
        model_args={
            "max_epochs": 1,
            "batch_size": 8,
            "train_split": None,
        },
    )
    model.fit(X, y)
    assert model.predict(X).shape == y.shape
    probabilities = model.predict_proba_untuned(X)
    assert isinstance(probabilities, dict)
    assert set(probabilities) == set(y.columns)
    fixed_trial = optuna.trial.FixedTrial(
        {
            "module__width": 8,
            "module__depth": 1,
            "module__grid_size": 3,
            "module__spline_order": 2,
            "module__grid_eps": 0.1,
            "module__sparse_init": False,
            "early_stopping": False,
            "restarts": False,
            "optimizer__lr": 0.01,
            "optimizer__weight_decay": 1e-6,
        }
    )
    objective = model.optuna_objective(
        X,
        y[["first"]],
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_folds=2,
    )
    assert np.isfinite(objective(fixed_trial))


class FakeTabPFNClassifier:
    def __init__(self, version: str, **kwargs: Any) -> None:
        self.version = version
        self.kwargs = kwargs
        self.classes_ = np.asarray([])

    @classmethod
    def create_default_for_version(
        cls, version: str, **kwargs: Any
    ) -> FakeTabPFNClassifier:
        return cls(version, **kwargs)

    def get_inference_config(self) -> dict[str, int]:
        return {
            "MAX_NUMBER_OF_SAMPLES": 10_000,
            "MAX_NUMBER_OF_FEATURES": 1_000,
            "MAX_NUMBER_OF_CLASSES": 100,
        }

    def to(self, device: str) -> None:
        self.kwargs["device"] = device

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNClassifier:
        self.classes_ = np.asarray(sorted(y.unique()))
        return self

    def predict_proba(self, X: DataFrame) -> np.ndarray:
        values = X.iloc[:, 0].fillna(0.0).to_numpy()
        score = 1.0 / (1.0 + np.exp(-values))
        return np.column_stack([1.0 - score, score])

    def predict(self, X: DataFrame) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def score(self, X: DataFrame, y: Series) -> float:
        return float(np.mean(self.predict(X) == y.to_numpy()))


class FakeTabPFNRegressor:
    def __init__(self, version: str, **kwargs: Any) -> None:
        self.version = version
        self.kwargs = kwargs
        self.mean_ = 0.0

    @classmethod
    def create_default_for_version(
        cls, version: str, **kwargs: Any
    ) -> FakeTabPFNRegressor:
        return cls(version, **kwargs)

    def get_inference_config(self) -> dict[str, int]:
        return {"MAX_NUMBER_OF_SAMPLES": 10_000, "MAX_NUMBER_OF_FEATURES": 1_000}

    def to(self, device: str) -> None:
        self.kwargs["device"] = device

    def fit(self, X: DataFrame, y: Series) -> FakeTabPFNRegressor:
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)

    def score(self, X: DataFrame, y: Series) -> float:
        return -float(np.mean(np.abs(self.predict(X) - y.to_numpy())))


@pytest.mark.fast
def test_tabpfn_versions_and_multitarget_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "df_analyze.models.tabpfn.OfficialTabPFNClassifier", FakeTabPFNClassifier
    )
    monkeypatch.setattr(
        "df_analyze.models.tabpfn.OfficialTabPFNRegressor", FakeTabPFNRegressor
    )
    monkeypatch.setattr("df_analyze.models.tabpfn.ModelVersion", str)
    monkeypatch.setattr("df_analyze.models.tabpfn._TABPFN_IMPORT_ERROR", None)
    monkeypatch.setattr("df_analyze.models.tabpfn.torch.cuda.is_available", lambda: False)

    X = numeric_data(30)
    y_cls = DataFrame(
        {
            "first": (X["x0"] > 0).astype(int),
            "second": (X["x1"] > 0).astype(int),
        }
    )
    classifier = TabPFNClassifierV26(model_args={"n_estimators": 2})
    classifier.fit(X, y_cls)
    assert classifier.version == "v2.6"
    assert classifier.predict(X).shape == y_cls.shape
    probabilities = classifier.predict_proba_untuned(X)
    assert isinstance(probabilities, dict)
    assert set(probabilities) == set(y_cls.columns)
    study = classifier.htune_optuna(
        X,
        y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=1,
    )
    assert study.best_trial.number == 0

    regressor = TabPFNRegressorV25(model_args={"n_estimators": 2})
    target = Series(X["x0"] + X["x1"], name="target")
    regressor.fit(X, target)
    assert regressor.version == "v2.5"
    assert len(regressor.predict(X)) == len(target)


@pytest.mark.fast
def test_tabpfn_full_feature_evaluation_accepts_native_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "df_analyze.models.tabpfn.OfficialTabPFNClassifier", FakeTabPFNClassifier
    )
    monkeypatch.setattr("df_analyze.models.tabpfn.ModelVersion", str)
    monkeypatch.setattr("df_analyze.models.tabpfn._TABPFN_IMPORT_ERROR", None)

    n = 60
    processed = DataFrame(
        {
            "age": np.arange(n, dtype=float),
            "city_a": [1.0, 0.0] * (n // 2),
            "city_b": [0.0, 1.0] * (n // 2),
        }
    )
    raw = DataFrame(
        {
            "age": [np.nan, *np.arange(1, n, dtype=float)],
            "city": Series(["a", "b"] * (n // 2), dtype="category"),
        }
    )
    y = Series([0, 1] * (n // 2), name="target")

    def prepared_part(rows: slice) -> PreparedData:
        return PreparedData(
            X=processed.iloc[rows].reset_index(drop=True),
            X_tabpfn=raw.iloc[rows].reset_index(drop=True),
            y=y.iloc[rows].reset_index(drop=True),
            groups=None,
            is_classification=True,
            validate=False,
        )

    prepared = prepared_part(slice(None))
    train = prepared_part(slice(0, 40))
    test = prepared_part(slice(40, None))
    options = SimpleNamespace(
        models=[TabPFNClassifierV26],
        htune_cls_metric=ClassifierScorer.Accuracy,
        htune_reg_metric=None,
        htune_trials=1,
        runtime=get_runtime("cpu"),
        seed=42,
    )

    evaluated = evaluate_tuned(
        prepared=prepared,
        prep_train=train,
        prep_test=test,
        assoc_filtered=None,
        pred_filtered=None,
        model_selected=ModelSelected(embed_selected=None, wrap_selected=None),
        options=options,
    )

    tabpfn = next(
        result
        for result in evaluated.results
        if result.model_cls is TabPFNClassifierV26
    )
    assert tabpfn.failure_reason is None
    assert np.isfinite(tabpfn.score)
    assert tabpfn.model.shortname == "tabpfn-v2_6"
    assert tabpfn.selected_cols == ["age", "city"]


@pytest.mark.fast
def test_tabpfn_preflight_revalidates_cached_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "df_analyze.models.tabpfn.OfficialTabPFNClassifier", FakeTabPFNClassifier
    )
    monkeypatch.setattr("df_analyze.models.tabpfn.ModelVersion", str)
    monkeypatch.setattr("df_analyze.models.tabpfn._TABPFN_IMPORT_ERROR", None)
    monkeypatch.delenv("TABPFN_ALLOW_CPU_LARGE_DATASET", raising=False)

    model = TabPFNClassifierV26()
    X = numeric_data(30)
    model.preflight(X, Series((X["x0"] > 0).astype(int), name="target"))

    larger = numeric_data(1_001)
    target = Series((larger["x0"] > 0).astype(int), name="target")
    with pytest.raises(RuntimeError, match="cannot run on CPU with 1001 rows"):
        model.preflight(larger, target)


@pytest.mark.fast
def test_tabpfn_v3_pretraining_envelope_uses_documented_shape_regimes() -> None:
    class ShapeOnlyFrame:
        def __init__(self, n_samples: int, n_features: int) -> None:
            self.shape = (n_samples, n_features)

        def __len__(self) -> int:
            return self.shape[0]

    model = TabPFNClassifierV3()
    y = Series([0, 1], name="target")

    assert TABPFN_PRETRAINING_LIMITS["v3"]["features"] == 20_000
    assert TABPFN_V3_MODEL_CARD_MAX_FEATURES == 2_000
    model._validate_limits(ShapeOnlyFrame(1_000_000, 200), y)  # type: ignore[arg-type]
    model._validate_limits(ShapeOnlyFrame(100_000, 2_000), y)  # type: ignore[arg-type]
    with pytest.warns(
        UserWarning,
        match=r"wider TabPFN 8.x row/feature regime.*experimental",
    ):
        model._validate_limits(  # type: ignore[arg-type]
            ShapeOnlyFrame(1_000, 20_000),
            y,
        )

    with pytest.raises(
        ValueError,
        match=r"1,000 samples x 20,000 features",
    ):
        model._validate_limits(ShapeOnlyFrame(1_000, 20_001), y)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match=r"documented row/feature regimes",
    ):
        model._validate_limits(  # type: ignore[arg-type]
            ShapeOnlyFrame(1_001, 20_000),
            y,
        )

    with pytest.raises(
        ValueError,
        match=r"documented row/feature regimes",
    ):
        model._validate_limits(  # type: ignore[arg-type]
            ShapeOnlyFrame(100_001, 2_000),
            y,
        )

    with pytest.raises(
        ValueError,
        match=r"documented row/feature regimes",
    ):
        model._validate_limits(  # type: ignore[arg-type]
            ShapeOnlyFrame(1_000_001, 200),
            y,
        )

    with pytest.raises(
        ValueError,
        match=r"documented row/feature regimes",
    ):
        model._validate_limits(  # type: ignore[arg-type]
            ShapeOnlyFrame(1_000_000, 201),
            y,
        )


@pytest.mark.fast
def test_tabpfn_v3_checkpoint_feature_limit_is_per_estimator() -> None:
    config = {
        "MAX_NUMBER_OF_SAMPLES": 1_000_000,
        "MAX_NUMBER_OF_FEATURES": 1_000,
        "MAX_NUMBER_OF_CLASSES": 160,
    }
    X = DataFrame(np.zeros((2, 2_001), dtype=np.float32))
    y = Series([0, 1], name="target")

    TabPFNClassifierV3()._validate_checkpoint_limits(X, y, config)
    with pytest.raises(ValueError, match=r"at most 1000 features"):
        TabPFNClassifierV26()._validate_checkpoint_limits(X, y, config)


@pytest.mark.fast
def test_tabpfn_preflight_rejects_explicit_unwritable_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked-cache"
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", str(blocked))
    monkeypatch.setattr(tabpfn_module, "_get_tabpfn_cache_dir", lambda: blocked)
    monkeypatch.setattr(tabpfn_module, "_TABPFN_IMPORT_ERROR", None)
    monkeypatch.setattr(tabpfn_module, "ModelVersion", str)
    monkeypatch.setattr(
        tabpfn_module, "OfficialTabPFNClassifier", FakeTabPFNClassifier
    )

    def reject_cache(path: Path) -> None:
        raise PermissionError(f"denied: {path}")

    monkeypatch.setattr(tabpfn_module, "_assert_cache_writable", reject_cache)
    model = TabPFNClassifierV3()
    X = numeric_data(30)
    y = Series((X["x0"] > 0).astype(int), name="target")

    with pytest.raises(
        TabPFNSetupError,
        match=r"TABPFN_MODEL_CACHE_DIR.*writable directory",
    ):
        model.preflight(X, y)


@pytest.mark.fast
def test_tabpfn_default_unwritable_cache_uses_writable_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked-cache"
    state_dir = tmp_path / "state"
    settings = SimpleNamespace(tabpfn=SimpleNamespace(model_cache_dir=None))
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setenv("TABPFN_STATE_DIR", str(state_dir))
    monkeypatch.setattr(tabpfn_module, "_get_tabpfn_cache_dir", lambda: blocked)
    monkeypatch.setattr(tabpfn_module, "_tabpfn_settings", settings)

    def check_cache(path: Path) -> None:
        if path == blocked:
            raise PermissionError(f"denied: {path}")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tabpfn_module, "_assert_cache_writable", check_cache)

    with pytest.warns(UserWarning, match="default model cache is not writable"):
        resolved = tabpfn_module._prepare_tabpfn_cache_dir()

    fallback = state_dir / "model-cache"
    assert resolved == fallback
    assert os.environ["TABPFN_MODEL_CACHE_DIR"] == str(fallback)
    assert settings.tabpfn.model_cache_dir == fallback


@pytest.mark.fast
def test_tabpfn_cli_version_selects_model_class() -> None:
    parser = make_parser()
    args = parser.parse_args(["--tabpfn-version", "v2_6"])
    assert args.tabpfn_version == "v2_6"
    dotted = parser.parse_args(["--tabpfn-version", "v2.5"])
    assert dotted.tabpfn_version == "v2_5"
    assert "--tabpfn-version {v3,v2.6,v2.5}" in parser.format_help()
    options = object.__new__(ProgramOptions)
    options.is_classification = True
    options.classifiers = (DfAnalyzeClassifier.TabPFN,)
    options.tabpfn_version = TabPFNVersion.V26
    assert options.models == [TabPFNClassifierV26]


@pytest.mark.fast
def test_model_tuning_fold_declarations_match_implementations() -> None:
    assert DecisionTreeClassifier.tuning_cv_folds == 5
    assert XGBoostClassifier.tuning_cv_folds == 5
    assert KANEstimator.tuning_cv_folds == 3
    assert TabPFNClassifierV3.tuning_cv_folds == 3

from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import logging
import os
import sys
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from typing import Literal, Union

import jsonpickle
import numpy as np
import pandas as pd
import pytest
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytest import CaptureFixture
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import KBinsDiscretizer

from df_analyze.analysis.adaptive_error.oof import _init_model as init_aer_model
from df_analyze.enumerables import ClassifierScorer, RegressorScorer
from df_analyze.models.base import (
    DfAnalyzeModel,
    _calibration_folds,
)
from df_analyze.models.catboost import CatBoostClassifier, CatBoostRegressor
from df_analyze.models.dummy import DummyClassifier, DummyRegressor
from df_analyze.models.gandalf import (
    LOGS,
    GandalfEstimator,
    WindowsSafeModelCheckpoint,
)
from df_analyze.models.knn import KNNClassifier, KNNRegressor
from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifier,
    SGDRegressor,
)
from df_analyze.models.mlp import MLPEstimator
from df_analyze.models.svm import SVMClassifier, SVMRegressor
from df_analyze.models.trees import DecisionTreeRegressor

C = 10


def test_adaptive_error_model_reconstruction_keeps_model_args() -> None:
    model_args = {"strategy": "uniform", "random_state": 17}
    model = init_aer_model(DummyClassifier, Series([0, 1, 0, 1]), model_args=model_args)

    assert model.model_args == model_args


def test_calibration_folds_follow_smallest_class() -> None:
    assert _calibration_folds(Series([0, 0, 0, 1, 1, 1, 1])) == 3
    assert _calibration_folds(Series([0, 1, 1])) is None


def fake_data(
    mode: Literal["classify", "regress"], noise: float = 1.0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    N = 100
    X_cont_tr = np.random.standard_normal([N, C])
    X_cont_test = np.random.standard_normal([N, C])

    cat_sizes = np.random.randint(2, 5, C)
    cats_tr = [np.random.randint(0, c, N) for c in cat_sizes]
    cats_test = [np.random.randint(0, c, N) for c in cat_sizes]

    X_cat_tr = np.empty([N, C])
    for i, cat in enumerate(cats_tr):
        X_cat_tr[:, i] = cat

    X_cat_test = np.empty([N, C])
    for i, cat in enumerate(cats_test):
        X_cat_test[:, i] = cat

    df_cat_tr = pd.get_dummies(DataFrame(X_cat_tr))
    df_cat_test = pd.get_dummies(DataFrame(X_cat_test))

    df_cont_tr = DataFrame(X_cont_tr)
    df_cont_test = DataFrame(X_cont_test)

    df_tr = pd.concat([df_cont_tr, df_cat_tr], axis=1)
    df_test = pd.concat([df_cont_test, df_cat_test], axis=1)

    cols = [f"f{i}" for i in range(df_tr.shape[1])]
    df_tr.columns = cols
    df_test.columns = cols

    weights = np.random.uniform(0, 1, 2 * C)
    y_tr = np.dot(df_tr.values, weights) + np.random.normal(0, noise, N)
    y_test = np.dot(df_test.values, weights) + np.random.normal(0, noise, N)

    if mode == "classify":
        encoder = KBinsDiscretizer(
            n_bins=2,
            encode="ordinal",
            quantile_method="linear",
        )
        encoder.fit(np.concatenate([y_tr.reshape(-1, 1), y_test.reshape(-1, 1)]))
        y_tr = encoder.transform(y_tr.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    target_tr = Series(np.asarray(y_tr).ravel(), name="target")
    target_test = Series(np.asarray(y_test).ravel(), name="target")

    return df_tr, df_test, target_tr, target_test


def lightweight_multitarget_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    rng = np.random.default_rng(7)
    n = 60
    X = DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y_cls = DataFrame(
        {
            "target_a": np.tile([0, 1], n // 2),
            "target_b": np.tile([0, 0, 1, 1], n // 4),
        }
    )
    base = X["f0"].to_numpy() * 0.8 - X["f1"].to_numpy() * 0.3
    y_reg = DataFrame(
        {
            "target_a": base,
            "target_b": base * 0.5 + 0.1 * X["f2"].to_numpy(),
        }
    )
    return X, y_cls, y_reg


def test_linear_models_use_explicit_multioutput_wrappers() -> None:
    X, y_cls, y_reg = lightweight_multitarget_data()

    classifier = LRClassifier(model_args={"C": 1.0, "l1_ratio": 0.0})
    classifier.refit_tuned(X, y_cls, tuned_args={})
    cls_preds = classifier.tuned_predict(X)
    cls_probs = classifier.predict_proba(X)

    assert isinstance(classifier.tuned_model, MultiOutputClassifier)
    assert isinstance(cls_preds, DataFrame)
    assert list(cls_preds.columns) == list(y_cls.columns)
    assert isinstance(cls_probs, dict)
    assert sorted(cls_probs) == sorted(y_cls.columns)

    regressor = SGDRegressor(model_args={"max_iter": 50, "tol": 1e-3})
    regressor.refit_tuned(X, y_reg, tuned_args={})
    reg_preds = regressor.tuned_predict(X)

    assert isinstance(regressor.tuned_model, TransformedTargetRegressor)
    assert isinstance(regressor.tuned_model.regressor_, MultiOutputRegressor)
    assert isinstance(reg_preds, DataFrame)
    assert list(reg_preds.columns) == list(y_reg.columns)


def test_scaled_per_target_regressor_jsonpickle_roundtrip() -> None:
    X, _, y_reg = lightweight_multitarget_data()
    model = SGDRegressor(model_args={"max_iter": 100, "tol": 1e-3, "random_state": 0})
    model.refit_tuned(X, y_reg, tuned_args={})
    expected = model.tuned_predict(X)

    restored = jsonpickle.decode(jsonpickle.encode(model, unpicklable=True))

    np.testing.assert_allclose(restored.tuned_predict(X), expected)


def test_logistic_regression_avoids_deprecated_penalty_argument() -> None:
    X, y_cls, _ = lightweight_multitarget_data()
    model = LRClassifier(model_args={"C": 1.0, "l1_ratio": 0.5})

    assert "penalty" not in model.fixed_args
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.refit_tuned(X, y_cls["target_a"], tuned_args={})


def test_multitarget_optuna_refits_explicit_wrapper() -> None:
    X, y_cls, _ = lightweight_multitarget_data()
    model = LRClassifier()

    model.htune_optuna(
        X_train=X,
        y_train=y_cls,
        g_train=None,
        metric=ClassifierScorer.Accuracy,
        n_trials=1,
        n_jobs=1,
    )

    assert isinstance(model.tuned_model, MultiOutputClassifier)
    predictions = model.tuned_predict(X)
    assert isinstance(predictions, DataFrame)
    assert list(predictions.columns) == list(y_cls.columns)


def test_multitarget_regression_tuning_rejects_degenerate_cv_target() -> None:
    n = 50
    X = DataFrame({"feature": np.arange(n, dtype=float)})
    y = DataFrame(
        {
            "dense": np.arange(n, dtype=float),
            "sparse": np.r_[np.ones(4), np.zeros(n - 4)],
        }
    )
    objective = DummyRegressor().optuna_objective(
        X,
        y,
        g_train=None,
        metric=RegressorScorer.MAE,
        n_folds=5,
    )

    with pytest.raises(
        RuntimeError,
        match="every multi-target regression target varies",
    ):
        objective(object())  # type: ignore[arg-type]


def test_non_deep_models_support_multitarget_outputs() -> None:
    X, y_cls, y_reg = lightweight_multitarget_data()
    classifiers = [
        DummyClassifier(),
        KNNClassifier(model_args={"n_neighbors": 3}),
        LightGBMClassifier(
            model_args={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4}
        ),
        LightGBMRFClassifier(
            model_args={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4}
        ),
        LRClassifier(model_args={"C": 1.0, "l1_ratio": 0.0}),
        SGDClassifier(model_args={"max_iter": 50, "tol": 1e-3}),
        SVMClassifier(model_args={"kernel": "linear", "C": 1.0}),
        CatBoostClassifier(model_args={"iterations": 3, "depth": 2}),
    ]
    regressors = [
        DummyRegressor(),
        KNNRegressor(model_args={"n_neighbors": 3}),
        LightGBMRegressor(
            model_args={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4}
        ),
        LightGBMRFRegressor(
            model_args={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4}
        ),
        ElasticNetRegressor(model_args={"alpha": 0.1, "l1_ratio": 0.5}),
        SGDRegressor(model_args={"max_iter": 50, "tol": 1e-3}),
        SVMRegressor(model_args={"kernel": "linear", "C": 1.0}),
        CatBoostRegressor(model_args={"iterations": 3, "depth": 2}),
    ]

    for model in classifiers:
        model.refit_tuned(X, y_cls, tuned_args={})
        predictions = model.tuned_predict(X)
        probabilities = model.predict_proba(X)
        scores = model._score_outputs(y_cls, predictions, probabilities)
        assert np.asarray(predictions).shape == y_cls.shape, model.longname
        assert "hamming-acc" in scores, model.longname

    for model in regressors:
        model.refit_tuned(X, y_reg, tuned_args={})
        predictions = model.tuned_predict(X)
        scores = model._score_outputs(y_reg, predictions)
        assert np.asarray(predictions).shape == y_reg.shape, model.longname
        assert np.isfinite(scores["multi-nrmse"]), model.longname


def test_multitarget_joint_metrics_are_reported() -> None:
    y_cls = DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]})
    pred_cls = DataFrame({"a": [0, 0, 1], "b": [1, 1, 1]})
    cls_scores = LRClassifier()._score_outputs(y_cls, pred_cls)

    assert {"subset-acc", "hamming-loss", "hamming-acc"} <= set(cls_scores)

    y_reg = DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})
    pred_reg = DataFrame({"a": [0.0, 2.0, 2.0], "b": [2.0, 2.0, 4.0]})
    reg_scores = SGDRegressor()._score_outputs(y_reg, pred_reg)

    expected = {
        "multi-nmae",
        "multi-nmse",
        "multi-nrmse",
        "raw-macro-mae",
        "raw-macro-mse",
        "raw-vector-rmse",
        "raw-macro-rmse",
        "multi-r2",
        "multi-r2-var",
    }
    assert expected <= set(reg_scores)


def test_native_multitarget_regression_fit_is_target_unit_invariant() -> None:
    X = DataFrame(
        {
            "x_first": np.tile([0.0, 0.0, 1.0, 1.0], 100),
            "x_second": np.tile([0.0, 1.0, 0.0, 1.0], 100),
        }
    )
    y = DataFrame(
        {
            "first": 10.0 * X["x_first"],
            "second": X["x_second"],
        }
    )
    rescaled = y.assign(second=y["second"] * 1_000.0)

    original_model = DecisionTreeRegressor(model_args={"max_depth": 1, "random_state": 0})
    rescaled_model = DecisionTreeRegressor(model_args={"max_depth": 1, "random_state": 0})
    original_model.fit(X, y)
    rescaled_model.fit(X, rescaled)

    np.testing.assert_allclose(
        original_model.model.feature_importances_,
        rescaled_model.model.feature_importances_,
    )
    original_predictions = original_model.predict(X)
    rescaled_predictions = rescaled_model.predict(X)
    np.testing.assert_allclose(
        original_predictions["first"], rescaled_predictions["first"]
    )
    np.testing.assert_allclose(
        original_predictions["second"],
        rescaled_predictions["second"] / 1_000.0,
    )


def test_catboost_native_multitarget_regression_scales_targets() -> None:
    class RecordingRegressor(BaseEstimator):
        def __init__(self, **kwargs) -> None:
            self.fit_y = DataFrame()
            self.means = np.array([])

        def fit(self, X: DataFrame, y: DataFrame) -> RecordingRegressor:
            self.fit_y = np.asarray(y, dtype=float).copy()
            self.means = self.fit_y.mean(axis=0)
            return self

        def predict(self, X: DataFrame) -> np.ndarray:
            return np.tile(self.means, (len(X), 1))

    X, _, y = lightweight_multitarget_data()
    model = CatBoostRegressor()
    model.model_cls = RecordingRegressor
    fitted = model._fit_target_models(X, y, {})

    assert isinstance(fitted, TransformedTargetRegressor)
    np.testing.assert_allclose(fitted.regressor_.fit_y.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        fitted.regressor_.fit_y.std(axis=0, ddof=0),
        1.0,
        atol=1e-12,
    )
    predictions = fitted.predict(X)
    expected = np.tile(y.mean(axis=0).to_numpy(dtype=float), (len(X), 1))
    np.testing.assert_allclose(predictions, expected)


def test_scaled_native_multitarget_regressor_jsonpickle_roundtrip() -> None:
    X, _, y = lightweight_multitarget_data()
    model = DecisionTreeRegressor(model_args={"max_depth": 2, "random_state": 0})
    model.fit(X, y)
    expected = model.predict(X)

    restored = jsonpickle.decode(jsonpickle.encode(model, unpicklable=True))

    np.testing.assert_allclose(restored.predict(X), expected)


def test_multitarget_normalized_regression_metrics_are_unit_invariant() -> None:
    model = SGDRegressor()
    y_true = DataFrame(
        {
            "small": [0.0, 1.0, 2.0, 3.0],
            "large": [0.0, 2.0, 4.0, 6.0],
        }
    )
    y_pred = DataFrame(
        {
            "small": [0.5, 1.0, 1.5, 2.5],
            "large": [1.0, 2.0, 3.0, 5.0],
        }
    )
    original = model._score_outputs(y_true, y_pred)
    scaled_true = y_true.assign(large=y_true["large"] * 1_000_000)
    scaled_pred = y_pred.assign(large=y_pred["large"] * 1_000_000)
    rescaled = model._score_outputs(scaled_true, scaled_pred)

    for metric in ("multi-nmae", "multi-nmse", "multi-nrmse", "multi-r2"):
        assert rescaled[metric] == pytest.approx(original[metric])


def test_multitarget_regression_tuning_score_is_scale_invariant() -> None:
    model = SGDRegressor()
    y_true = DataFrame({"small": [0.0, 1.0, 2.0, 3.0], "large": [0.0, 2.0, 4.0, 6.0]})
    y_pred = DataFrame({"small": [0.5, 1.0, 1.5, 2.5], "large": [1.0, 2.0, 3.0, 5.0]})

    score = model._mean_tuning_score(RegressorScorer.MAE, y_true, y_pred)
    scaled_true = y_true.assign(large=y_true["large"] * 1_000_000)
    scaled_pred = y_pred.assign(large=y_pred["large"] * 1_000_000)
    scaled_score = model._mean_tuning_score(RegressorScorer.MAE, scaled_true, scaled_pred)

    assert scaled_score == pytest.approx(score)


def test_mlp_multitarget_refit_uses_target_models() -> None:
    X, y_cls, y_reg = lightweight_multitarget_data()
    model = MLPEstimator(
        num_classes=2,
        model_args={
            "max_epochs": 1,
            "batch_size": 16,
            "iterator_train__drop_last": False,
            "verbose": 0,
        },
    )
    tuned_args = {
        "module__width": 16,
        "module__depth": 3,
        "module__use_bn": False,
        "module__dropout": 0.0,
        "early_stopping": False,
        "restarts": False,
        "optimizer__lr": 1e-3,
        "optimizer__weight_decay": 1e-5,
    }

    model.refit_tuned(X, y_cls, tuned_args=tuned_args)
    predictions = model.tuned_predict(X)
    probabilities = model.predict_proba(X)

    assert isinstance(model.tuned_model, dict)
    assert isinstance(predictions, DataFrame)
    assert list(predictions.columns) == list(y_cls.columns)
    assert isinstance(probabilities, dict)
    assert sorted(probabilities) == sorted(y_cls.columns)

    regressor = MLPEstimator(
        num_classes=1,
        model_args={
            "max_epochs": 1,
            "batch_size": 16,
            "iterator_train__drop_last": False,
            "verbose": 0,
        },
    )
    regressor.refit_tuned(X, y_reg, tuned_args=tuned_args)
    reg_predictions = regressor.tuned_predict(X)

    assert isinstance(regressor.tuned_model, dict)
    assert isinstance(reg_predictions, DataFrame)
    assert list(reg_predictions.columns) == list(y_reg.columns)


def test_mlp_multitarget_optuna_reports_once_per_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, _ = lightweight_multitarget_data()
    fitted = []
    cleaned = []

    class Estimator:
        def fit(self, X, y) -> None:
            fitted.append(len(X))

        def predict(self, X) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

    class TargetModel:
        fixed_args = {}
        default_args = {}
        model_args = {}
        model_cls = Estimator

        @staticmethod
        def _to_torch(X, y):
            return (
                torch.as_tensor(X.to_numpy(), dtype=torch.float32),
                torch.as_tensor(y.to_numpy(), dtype=torch.int64),
            )

        @staticmethod
        def _to_model_args(args, X):
            return {}

        @staticmethod
        def _cleanup_after_fold() -> None:
            cleaned.append(True)

    class Trial:
        def __init__(self) -> None:
            self.steps = []
            self.user_attrs = {}

        def report(self, value: float, step: int) -> None:
            self.steps.append(step)

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

        @staticmethod
        def should_prune() -> bool:
            return False

    model = MLPEstimator(num_classes=2)
    monkeypatch.setattr(model, "_new_target_estimator", lambda y: TargetModel())
    monkeypatch.setattr(model, "optuna_args", lambda trial: {})
    trial = Trial()

    objective = model.optuna_objective(
        X, y, g_train=None, metric=ClassifierScorer.Accuracy, n_folds=2
    )
    score = objective(trial)

    assert np.isfinite(score)
    assert trial.steps == [0, 1]
    assert len(fitted) == len(y.columns) * 2
    assert len(cleaned) == len(fitted)
    assert set(trial.user_attrs["per_target_tuning_scores"]) == set(y.columns)


def test_gandalf_multitarget_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y_cls, _ = lightweight_multitarget_data()

    class TargetModel:
        def refit_tuned(self, X, y, g=None, tuned_args=None) -> None:
            self.value = int(y.mode().iloc[0])

        def tuned_predict(self, X) -> np.ndarray:
            return np.full(len(X), self.value)

        def predict_proba(self, X) -> np.ndarray:
            probs = np.zeros((len(X), 2), dtype=float)
            probs[:, self.value] = 1.0
            return probs

        def tuned_scores(self, X, y) -> float:
            return float(np.mean(self.tuned_predict(X) == y.to_numpy()))

        def optuna_objective(self, X, y, g, metric, n_folds):
            return lambda trial: float(y.mean())

    model = GandalfEstimator(num_classes=2)
    monkeypatch.setattr(model, "_new_target_estimator", lambda y: TargetModel())

    class Trial:
        def __init__(self) -> None:
            self.user_attrs = {}

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    model.refit_tuned(X, y_cls, tuned_args={})
    predictions = model.tuned_predict(X)
    probabilities = model.predict_proba(X)

    assert isinstance(predictions, DataFrame)
    assert list(predictions.columns) == list(y_cls.columns)
    assert isinstance(probabilities, dict)
    assert sorted(probabilities) == sorted(y_cls.columns)
    objective = model.optuna_objective(
        X, y_cls, g_train=None, metric=ClassifierScorer.Accuracy
    )
    trial = Trial()
    assert objective(trial) == 0.5
    assert set(trial.user_attrs["per_target_tuning_scores"]) == set(y_cls.columns)


def test_neural_optuna_objectives_include_model_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    X = DataFrame(np.arange(200, dtype=float).reshape(40, 5))
    y = Series(np.tile([0, 1], 20), name="target")

    class Trial:
        number = 0

        def report(self, value: float, step: int) -> None:
            return

        def should_prune(self) -> bool:
            return False

    mlp_args = []

    class MLPModel:
        def __init__(self, **kwargs) -> None:
            mlp_args.append(kwargs)

        def fit(self, X, y) -> None:
            return

        def predict(self, X) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

    mlp = MLPEstimator(
        num_classes=2,
        model_args={"max_epochs": 3, "iterator_train__drop_last": False},
    )
    mlp.model_cls = MLPModel
    monkeypatch.setattr(mlp, "optuna_args", lambda trial: {"batch_size": 5})
    monkeypatch.setattr(mlp, "_to_model_args", lambda args, X_train: args)

    objective = mlp.optuna_objective(
        X, y, g_train=None, metric=ClassifierScorer.Accuracy, n_folds=2
    )
    assert np.isfinite(objective(Trial()))
    assert len(mlp_args) == 2
    assert all(args["max_epochs"] == 3 for args in mlp_args)
    assert all(args["iterator_train__drop_last"] is False for args in mlp_args)
    assert all(args["batch_size"] == 5 for args in mlp_args)

    gandalf_args = []
    gandalf_prediction_calls = []

    class GandalfModel:
        def __init__(self, **kwargs) -> None:
            gandalf_args.append(kwargs)

    class ValidationData:
        X = torch.zeros((4, 5), dtype=torch.float32)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        def __len__(self) -> int:
            return len(self.y)

    class Trainer:
        log_dir = str(tmp_path)
        checkpoint_callback = SimpleNamespace(
            best_model_path=str(tmp_path / "missing-best.ckpt")
        )

        def fit(self, **kwargs) -> None:
            return

        def predict(self, **kwargs):
            gandalf_prediction_calls.append(kwargs)
            return [torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])]

    gandalf = GandalfEstimator(num_classes=2, model_args={"user_option": "kept"})
    gandalf.model_cls = GandalfModel
    train = SimpleNamespace(dataset=list(range(8)))
    validation = SimpleNamespace(dataset=ValidationData())
    monkeypatch.setattr(
        gandalf, "_train_val_loaders", lambda **kwargs: (train, validation)
    )
    monkeypatch.setattr(gandalf, "_get_trainer", lambda **kwargs: Trainer())
    monkeypatch.setattr(gandalf, "_pred_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(gandalf, "optuna_args", lambda trial: {"virtual_batch": 4})
    monkeypatch.setattr(gandalf, "_to_model_args", lambda args, X_train: args)

    objective = gandalf.optuna_objective(
        X.iloc[:4], y.iloc[:4], None, ClassifierScorer.Accuracy
    )
    assert objective(Trial()) == 1.0
    assert gandalf_args[0]["user_option"] == "kept"
    assert gandalf_args[0]["virtual_batch"] == 4
    assert "ckpt_path" not in gandalf_prediction_calls[0]


def check_basics(model: DfAnalyzeModel, mode: Literal["classify", "regress"]) -> None:
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    try:
        model.fit(X_train=X_tr, y_train=y_tr)
        model.predict(X_test)
    except ValueError as e:  # handle braindead Optuna race condition
        print(e)
        print(e.args)
        if "No trials are completed yet" in str(e):
            pass
        elif "No trials are completed yet" in " ".join(e.args):
            pass
        else:
            traceback.print_exc()
            raise e


def check_optuna_tune_metric(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
    metric: Union[ClassifierScorer, RegressorScorer],
) -> tuple[float, ndarray | Series | None]:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None  # noqa: F841
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    const_target = (len(y_tr.unique()) == 1) or (len(y_test.unique()) == 1)
    while const_target:
        X_tr, X_test, y_tr, y_test = fake_data(mode)
        const_target = (len(y_tr.unique()) == 1) or (len(y_test.unique()) == 1)

    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    model = deepcopy(model)
    try:
        study = model.htune_optuna(
            X_train=X_tr,
            y_train=y_tr,
            g_train=None,
            metric=metric,  # type: ignore
            # shitty Optuna implementation seems to dispatch as many jobs as cores,
            # even if you specify less trials, and but then also have some kind of
            # improper process.join() or other race condition so that it doesn't
            # properly wait for things to finish, resulting in an error like below:
            #
            # def get_best_trial(self, study_id: int) -> FrozenTrial:
            #     with self._lock:
            #         self._check_study_id(study_id)
            #
            #         best_trial_id = self._studies[study_id].best_trial_id
            #
            #         if best_trial_id is None:
            # >               raise ValueError("No trials are completed yet.")
            # E               ValueError: No trials are completed yet.
            #
            # These erors also seem to be uncatchable (raised by some sub-process)
            # so the only way to ignore the issue is to use 1 job for testing, but
            # then of course we aren't testing the paralellism, which is kinf of
            # the whole point. Not sure why Optuna sucks so hard.
            #
            # n_trials=40 if ON_CLUSTER else 8,
            n_trials=4,
            n_jobs=1,
        )

        if not hasattr(study, "best_params"):
            raise RuntimeError("No trials ever ran for some reason.")
        overrides = study.best_params
        print(overrides)
    except ValueError as e:  # handle braindead Optuna race condition
        msg = (
            f"Got error for metric: {metric.name}. Targets:\n"
            f"y_tr unique values: {np.unique(y_tr, return_counts=True)}\n"
            f"{y_tr}\n"
            f"y_test unique values: {np.unique(y_test, return_counts=True)}\n"
            f"{y_test}\n"
        )
        print(e)
        if "No trials are completed yet" in str(e):
            raise ValueError(
                f"No trials completed by Optuna for some reason. Info:\n{msg}"
            )
        raise ValueError(msg) from e
    # print(f"Best params: {overrides}")
    model.refit_tuned(X_tr, y_tr, tuned_args=overrides)
    preds = model.tuned_predict(X_test)
    try:
        score = metric.tuning_score(y_true=y_test, y_pred=preds)
    except Exception:
        raise ValueError(
            "Could not get score on final generated test data. Maybe all same class?"
        )
    return score, preds


def check_optuna_tune(
    model: DfAnalyzeModel,
    mode: Literal["classify", "regress"],
) -> None:
    is_cls = model.is_classifier
    metrics = ClassifierScorer if is_cls else RegressorScorer
    # metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()
    for metric in metrics:
        print(
            metric.name,
            check_optuna_tune_metric(model=model, mode=mode, metric=metric)[0],
        )


@pytest.mark.fast
class TestDummy:
    def test_dummy_cls(self) -> None:
        model = DummyClassifier()
        check_basics(model, "classify")

    def test_dummy_reg(self) -> None:
        model = DummyRegressor()
        check_basics(model, "regress")

    def test_dummy_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        model = DummyClassifier()
        check_optuna_tune(model, "classify")

    def test_dummy_reg_tune(self, capsys: CaptureFixture) -> None:
        model = DummyRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestLinear:
    def test_lin_cls(self) -> None:
        model = LRClassifier()
        check_basics(model, "classify")

    def test_lin_reg(self) -> None:
        model = ElasticNetRegressor()
        check_basics(model, "regress")

    def test_lin_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = LRClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_lin_reg_tune(self, capsys: CaptureFixture) -> None:
        model = ElasticNetRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSGDLinear:
    def test_sgd_cls(self) -> None:
        model = SGDClassifier()
        check_basics(model, "classify")

    def test_sgd_reg(self) -> None:
        model = SGDRegressor()
        check_basics(model, "regress")

    def test_sgd_cls_tune(self, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = SGDClassifier()
            check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    def test_sgd_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SGDRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestKNN:
    def test_knn_grid_preserves_public_metrics(self) -> None:
        assert KNNClassifier().grid["metric"] == [
            "cosine",
            "l2",
            "correlation",
        ]

    def test_knn_cls(self) -> None:
        model = KNNClassifier()
        check_basics(model, "classify")

    def test_knn_reg(self) -> None:
        model = KNNRegressor()
        check_basics(model, "regress")

    def test_knn_cls_tune(self, capsys: CaptureFixture) -> None:
        model = KNNClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_knn_reg_tune(self, capsys: CaptureFixture) -> None:
        model = KNNRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestSVM:
    def test_svm_cls(self) -> None:
        model = SVMClassifier()
        check_basics(model, "classify")

    def test_svm_reg(self) -> None:
        model = SVMRegressor()
        check_basics(model, "regress")

    def test_svm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = SVMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_svm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = SVMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestLightGBM:
    def test_lgbm_cls(self) -> None:
        model = LightGBMClassifier()
        check_basics(model, "classify")

    def test_lgbm_reg(self) -> None:
        model = LightGBMRegressor()
        check_basics(model, "regress")

    def test_lgbm_rf_cls(self) -> None:
        model = LightGBMRFClassifier()
        check_basics(model, "classify")

    def test_lgbm_rf_reg(self) -> None:
        model = LightGBMRFRegressor()
        check_basics(model, "regress")

    def test_lgbm_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")

    def test_lgbm_rf_cls_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, "classify")

    def test_lgbm_rf_reg_tune(self, capsys: CaptureFixture) -> None:
        model = LightGBMRFRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, "regress")


@pytest.mark.fast
class TestGandalf:
    def test_checkpoint_removal_retries_windows_file_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0

        def transiently_locked(*args: object, **kwargs: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError(32, "checkpoint is in use")

        monkeypatch.setattr(
            "pytorch_lightning.callbacks.ModelCheckpoint._remove_checkpoint",
            transiently_locked,
        )
        monkeypatch.setattr("df_analyze.models.gandalf.sleep", lambda _: None)
        monkeypatch.setattr("df_analyze.models.gandalf.sys.platform", "win32")

        callback = WindowsSafeModelCheckpoint()
        callback._remove_checkpoint(object(), "checkpoint.ckpt")  # type: ignore[arg-type]

        assert attempts == 3

    def test_gandalf_cls(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=C)
            with capsys.disabled():
                check_basics(model, "classify")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_reg(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=1)
            with capsys.disabled():
                check_basics(model, "regress")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_cls_tune(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=C)
            with capsys.disabled():
                check_optuna_tune(model, "classify")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

    def test_gandalf_reg_tune(self, capsys: CaptureFixture) -> None:
        try:
            model = GandalfEstimator(num_classes=1)
            with capsys.disabled():
                check_optuna_tune(model, "regress")
        except Exception as e:
            raise e
        finally:
            rmtree(LOGS, ignore_errors=True)

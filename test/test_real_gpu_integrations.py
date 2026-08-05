from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from df_analyze.models.catboost import CatBoostClassifier
from df_analyze.models.kan import KANEstimator
from df_analyze.models.knn import KNNClassifier, TorchKNNClassifier
from df_analyze.models.mlp import MLPEstimator
from df_analyze.models.xgboost import XGBoostClassifier
from df_analyze.runtime.hardware import RuntimeComponent, get_runtime


RUN_REAL_CUDA_TESTS = os.getenv(
    "DF_ANALYZE_RUN_REAL_CUDA_TESTS", ""
).lower() in {"1", "true", "yes"}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_REAL_CUDA_TESTS,
        reason="set DF_ANALYZE_RUN_REAL_CUDA_TESTS=1 to run real CUDA backends",
    ),
]

TORCH_CUDA_COMPONENTS = frozenset(
    {RuntimeComponent.KNN, RuntimeComponent.MLP, RuntimeComponent.KAN}
)


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(20260726)
    X = pd.DataFrame(
        rng.normal(size=(48, 6)), columns=[f"x{idx}" for idx in range(6)]
    )
    y = pd.Series((X["x0"] - X["x1"] > 0).astype(int), name="target")
    return X, y


def _cuda_policy(component: RuntimeComponent):
    if component in TORCH_CUDA_COMPONENTS and not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is not available on this test host")
    policy = get_runtime("cuda").with_workload(48, 6)
    decision = policy.decision_for(component)
    if decision.resolved != "cuda":
        pytest.skip(
            f"{component.value} CUDA backend is unavailable: {decision.reason}"
        )
    return policy


def test_real_cuda_knn_allocates_and_predicts_on_device() -> None:
    X, y = _classification_data()
    policy = _cuda_policy(RuntimeComponent.KNN)
    model = KNNClassifier(
        model_args={"n_neighbors": 3, "metric": "l2", "weights": "distance"}
    ).set_runtime(policy)

    model.fit(X, y)

    assert isinstance(model.model, TorchKNNClassifier)
    assert model.model._X is not None
    assert model.model._X.device.type == "cuda"
    assert np.asarray(model.predict(X)).shape == (len(y),)


@pytest.mark.parametrize(
    ("model_cls", "component"),
    [
        (MLPEstimator, RuntimeComponent.MLP),
        (KANEstimator, RuntimeComponent.KAN),
    ],
    ids=["mlp", "kan"],
)
def test_real_cuda_torch_learner_trains_on_device(model_cls, component) -> None:
    X, y = _classification_data()
    policy = _cuda_policy(component)
    model = model_cls(
        num_classes=2,
        model_args={
            # Runtime policy must remain authoritative over custom model args.
            "device": "cpu",
            "max_epochs": 1,
            "batch_size": 8,
            "train_split": None,
            "verbose": 0,
        },
    ).set_runtime(policy)

    model.fit(X, y)

    assert next(model.model.module_.parameters()).device.type == "cuda"
    assert np.asarray(model.predict(X)).shape == (len(y),)


@pytest.mark.parametrize(
    ("model_cls", "component", "model_args"),
    [
        (
            CatBoostClassifier,
            RuntimeComponent.CatBoost,
            {"iterations": 3, "depth": 2, "verbose": 0},
        ),
        (
            XGBoostClassifier,
            RuntimeComponent.XGBoost,
            {"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
        ),
    ],
    ids=["catboost", "xgboost"],
)
def test_real_cuda_boosted_tree_trains_and_predicts(
    model_cls, component, model_args
) -> None:
    X, y = _classification_data()
    policy = _cuda_policy(component)
    model = model_cls(model_args=model_args).set_runtime(policy)

    model.fit(X, y)

    if component is RuntimeComponent.CatBoost:
        assert str(model.model.get_param("task_type")).upper() == "GPU"
    else:
        estimator = getattr(model.model, "estimator", model.model)
        config = estimator.get_booster().save_config().replace(" ", "").lower()
        assert '"device":"cuda' in config

    predictions = np.asarray(model.predict(X))
    assert predictions.shape == (len(y),)
    assert np.isfinite(predictions).all()

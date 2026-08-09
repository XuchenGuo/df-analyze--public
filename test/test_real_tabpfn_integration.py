from __future__ import annotations

import os

import numpy as np
import pytest
from pandas import DataFrame, Series

from df_analyze.models.tabpfn import TabPFNClassifierV3, TabPFNRegressorV3
from df_analyze.runtime.hardware import get_runtime


RUN_REAL_TABPFN_TESTS = os.getenv("DF_ANALYZE_RUN_REAL_TABPFN_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.licensed_integration,
    pytest.mark.skipif(
        not RUN_REAL_TABPFN_TESTS,
        reason=(
            "set DF_ANALYZE_RUN_REAL_TABPFN_TESTS=1 after accepting the checkpoint "
            "licenses and configuring authentication or cached weights"
        ),
    ),
]


def _numeric_data(n: int = 24) -> DataFrame:
    rng = np.random.default_rng(42)
    return DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_real_tabpfn_backend_smoke(task: str, tmp_path) -> None:
    X = _numeric_data()
    if task == "classification":
        model_cls = TabPFNClassifierV3
        y = Series((X["x0"] + X["x1"] > 0).astype(int), name="target")
    else:
        model_cls = TabPFNRegressorV3
        y = Series(X["x0"] - 0.5 * X["x1"], name="target")
    model = model_cls(model_args={"n_estimators": 1, "auto_scale_n_estimators": False})
    model.set_runtime(get_runtime("cpu"))

    model.fit(X, y)

    predictions = np.asarray(model.predict(X))
    assert predictions.shape == (len(y),)
    assert np.isfinite(predictions).all()
    model.externalize_fitted_models(tmp_path / "fitted_model")
    model.load_externalized_models(tmp_path)
    np.testing.assert_array_equal(model.predict(X), predictions)
    if task == "classification":
        probabilities = np.asarray(model.predict_proba_untuned(X))
        assert probabilities.shape == (len(y), 2)
        np.testing.assert_allclose(model.predict_proba_untuned(X), probabilities)

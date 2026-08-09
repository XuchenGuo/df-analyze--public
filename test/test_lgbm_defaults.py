from __future__ import annotations

import pytest

from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)


@pytest.mark.fast
def test_lgbm_wrappers_preserve_public_defaults() -> None:
    models = [
        LightGBMClassifier(),
        LightGBMRegressor(),
        LightGBMRFClassifier(),
        LightGBMRFRegressor(),
    ]

    for model in models:
        assert model.fixed_args["verbosity"] == -1
        assert "n_jobs" not in model.fixed_args
        assert "force_col_wise" not in model.fixed_args

    assert LightGBMRFClassifier().fixed_args["boosting_type"] == "rf"
    assert LightGBMRFRegressor().fixed_args["boosting_type"] == "rf"


@pytest.mark.fast
def test_lgbm_thread_count_remains_user_configurable() -> None:
    model = LightGBMClassifier(model_args={"n_jobs": 3})
    args = {**model.fixed_args, **model.default_args, **model.model_args}

    assert args["n_jobs"] == 3

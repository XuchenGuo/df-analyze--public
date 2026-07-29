from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Mapping, Optional

from optuna import Trial

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Any, Type

from lightgbm import LGBMClassifier, LGBMRegressor

from df_analyze.models.base import DfAnalyzeModel


class LightGBMEstimator(DfAnalyzeModel):
    shortname = "lgbm"
    longname = "LightGBM Estimator"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1)
        self.model_cls: Type[Any] = type(None)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    # https://neptune.ai/blog/lightgbm-parameters-guide
    # https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )


class LightGBMRFEstimator(DfAnalyzeModel):
    shortname = "rf"
    longname = "LightGBM Random Forest Estimator"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        # https://github.com/microsoft/LightGBM/issues/1333
        self.fixed_args = dict(verbosity=-1)
        self.default_args = dict(bagging_freq=1, bagging_fraction=0.75)
        self.model_cls: Type[Any] = type(None)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    # https://neptune.ai/blog/lightgbm-parameters-guide
    # https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.05, 0.95),
        )


class LightGBMClassifier(LightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Classifier"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.fixed_args = dict(verbosity=-1)
        self.model_cls = LGBMClassifier


class LightGBMRegressor(LightGBMEstimator):
    shortname = "lgbm"
    longname = "LightGBM Regressor"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args = dict(verbosity=-1)
        self.model_cls = LGBMRegressor


class LightGBMRFClassifier(LightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Classifier"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMClassifier


class LightGBMRFRegressor(LightGBMRFEstimator):
    shortname = "rf"
    longname = "LightGBM Random Forest Regressor"
    timeout_s = 3600  # one hour for tuning is plenty

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.fixed_args.update(dict(boosting_type="rf", verbosity=-1))
        self.model_cls = LGBMRegressor

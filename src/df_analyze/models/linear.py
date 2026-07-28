from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional

from optuna import Trial

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor

from df_analyze.models.base import DfAnalyzeModel

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html


def _normalize_sgd_average_arg(full_args: dict[str, Any]) -> dict[str, Any]:
    # sklearn validates `average` as bool or int > 0; Optuna best_params may return 0.
    if full_args.get("average") == 0:
        full_args = dict(full_args)
        full_args["average"] = False
    return full_args


class ElasticNetRegressor(DfAnalyzeModel):
    shortname = "elastic"
    longname = "ElasticNet Regressor"
    timeout_s = 30 * 60  # should never hit this

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = ElasticNet
        self.fixed_args = dict(max_iter=2000)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            alpha=trial.suggest_float("alpha", 0.01, 1.0),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
        )


class LRClassifier(DfAnalyzeModel):
    shortname = "lr"
    longname = "Logistic Regression"
    timeout_s = 15 * 60  # if it isn't done will be bad fit at this point

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = LogisticRegression
        self.fixed_args = dict(max_iter=2000, solver="saga")
        self.default_args = dict(l1_ratio=0.5)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 1.0),
            C=trial.suggest_float("C", 1e-10, 1e2, log=True),
        )


class SGDClassifier(DfAnalyzeModel):
    shortname = "sgd"
    longname = "SGD Linear Classifer"
    timeout_s = 15 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnSGDClassifier
        self.default_args = dict(
            loss="log_loss", learning_rate="adaptive", penalty="l2", eta0=3e-4
        )

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            loss=trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
        )


class SGDRegressor(DfAnalyzeModel):
    shortname = "sgd"
    longname = "SGD Linear Regressor"
    timeout_s = 15 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnSGDRegressor
        self.default_args = dict(eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            loss=trial.suggest_categorical(
                "loss", ["squared_error", "huber", "epsilon_insensitive"]
            ),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
        )


class SGDClassifierSelector(DfAnalyzeModel):
    shortname = "sgd-select"
    longname = "SGD Linear Selector"
    timeout_s = 10 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnSGDClassifier
        self.default_args = dict(learning_rate="adaptive", penalty="l1", eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        full_args = _normalize_sgd_average_arg(full_args)
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        args = dict(
            loss=trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            alpha=trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
            average=trial.suggest_int("average", 0, 20),
        )
        return _normalize_sgd_average_arg(args)


class SGDRegressorSelector(DfAnalyzeModel):
    shortname = "sgd-select"
    longname = "SGD Linear Selector"
    timeout_s = 10 * 60  # can't have this eating up all the time

    def __init__(self, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnSGDRegressor
        self.default_args = dict(learning_rate="adaptive", penalty="l1", eta0=3e-4)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        full_args = _normalize_sgd_average_arg(full_args)
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        args = dict(
            loss=trial.suggest_categorical(
                "loss",
                [
                    "squared_error",
                    "huber",
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ],
            ),
            eta0=trial.suggest_float("eta0", 1e-5, 5.0, log=True),
            alpha=trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            early_stopping=trial.suggest_categorical("early_stopping", [True, False]),
            average=trial.suggest_int("average", 0, 20),
        )
        return _normalize_sgd_average_arg(args)

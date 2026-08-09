from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional, Type

from optuna import Trial

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor as SklearnExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

from df_analyze._constants import SEED
from df_analyze.models.base import DfAnalyzeModel


class DecisionTreeEstimator(DfAnalyzeModel):
    shortname = "dtree"
    longname = "Decision Tree Estimator"
    timeout_s = 15 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.model_cls: Type[Any] = type(None)
        self.fixed_args = dict(random_state=SEED)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, Any]:
        return dict(
            max_depth=trial.suggest_categorical("max_depth", [None, 3, 5, 10, 20]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
        )


class ExtraTreesEstimator(DfAnalyzeModel):
    shortname = "et"
    longname = "Extra Trees Estimator"
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.model_cls: Type[Any] = type(None)
        self.fixed_args = dict(random_state=SEED, n_jobs=1)
        self.default_args = dict(n_estimators=100)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, Any]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            bootstrap=trial.suggest_categorical("bootstrap", [False, True]),
        )


class DecisionTreeClassifier(DecisionTreeEstimator):
    shortname = "dtree"
    longname = "Decision Tree Classifier"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnDecisionTreeClassifier


class DecisionTreeRegressor(DecisionTreeEstimator):
    shortname = "dtree"
    longname = "Decision Tree Regressor"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnDecisionTreeRegressor


class ExtraTreesClassifier(ExtraTreesEstimator):
    shortname = "et"
    longname = "Extra Trees Classifier"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnExtraTreesClassifier


class ExtraTreesRegressor(ExtraTreesEstimator):
    shortname = "et"
    longname = "Extra Trees Regressor"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnExtraTreesRegressor

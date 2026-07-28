from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import optuna
from optuna import Study, Trial

from df_analyze.enumerables import Scorer

ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from df_analyze._constants import SEED
from df_analyze.models.base import DfAnalyzeModel


class DummyEstimator(DfAnalyzeModel):
    shortname = "dummy-est"
    longname = "Dummy Estimator"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.target_cols: list[str] = []

    def htune_optuna(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_trials: int = 100,
        n_jobs: int = -1,
        verbosity: int = optuna.logging.ERROR,
    ) -> Study:
        """
        Too many jobs will blow memory and also since all grids are at most 4
        waste time and compute.
        """
        return super().htune_optuna(
            X_train, y_train, g_train, metric, n_trials, n_jobs=4, verbosity=verbosity
        )

    def _target_cols_for_output(self, n_targets: int) -> list[str]:
        if len(self.target_cols) == n_targets:
            return self.target_cols
        return [f"target_{i}" for i in range(n_targets)]

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            name = y.name if y.name is not None else "target"
            self.target_cols = [str(name)]
        super().refit_tuned(X=X, y=y, g=g, tuned_args=tuned_args)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame, np.ndarray]:
        preds = super().tuned_predict(X)
        if (
            isinstance(preds, np.ndarray)
            and preds.ndim == 2
            and preds.shape[1] > 1
        ):
            cols = self._target_cols_for_output(preds.shape[1])
            return DataFrame(preds, index=X.index, columns=cols)
        return preds


class DummyRegressor(DummyEstimator):
    shortname = "dummy"
    longname = "Dummy Regressor"
    timeout_s = 5 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnDummyRegressor
        self.fixed_args = dict()
        self.grid = {
            "strategy": ["mean", "median"],
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            strategy=trial.suggest_categorical(
                "strategy", ["mean", "median", "quantile"]
            ),
        )


class DummyClassifier(DummyEstimator):
    shortname = "dummy"
    longname = "Dummy Classifier"
    timeout_s = 5 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnDummyClassifier
        self.fixed_args = dict(random_state=SEED)
        self.grid = {"strategy": ["most_frequent", "prior", "stratified", "uniform"]}

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            strategy=trial.suggest_categorical(
                "strategy", ["most_frequent", "prior", "stratified", "uniform"]
            ),
        )

    def predict_proba(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        probs = super().predict_proba(X)
        if isinstance(probs, (list, tuple)):
            if len(probs) == 1:
                return np.asarray(probs[0])
            cols = self._target_cols_for_output(len(probs))
            return {col: np.asarray(arr) for col, arr in zip(cols, probs)}
        if (
            isinstance(probs, np.ndarray)
            and probs.ndim == 3
            and probs.shape[1] > 1
        ):
            cols = self._target_cols_for_output(probs.shape[1])
            return {col: np.asarray(probs[:, i, :]) for i, col in enumerate(cols)}
        return probs

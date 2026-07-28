from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
import warnings
from abc import ABC
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, SVR

from df_analyze.nonsense import enable_spam, silence_spam
from df_analyze.scoring import CLASSIFIER_TEST_SCORERS as CLS_SCORING
from df_analyze.scoring import REGRESSION_TEST_SCORERS as REG_SCORING

AnyModel = Union[
    LGBMClassifier,
    LGBMRegressor,
    DummyClassifier,
    DummyRegressor,
    ElasticNetCV,
    LinearRegression,
    LogisticRegressionCV,
    SklearnSGDRegressor,
    SklearnSGDClassifier,
    SVC,
    SVR,
]


class UnivariatePredictor(ABC):
    """Mixin needed for unified interface"""

    def __init__(self) -> None:
        self.model: Type[AnyModel]
        self.grid: dict[str, list[Any]] = {}
        self.fixed_args: dict[str, Any] = {}
        self.is_classifier: bool = False
        self.__opt: Optional[GridSearchCV] = None
        self.short: str = "abstract"

    def evaluate(
        self, X: Union[Series, DataFrame, ndarray], target: Series
    ) -> tuple[DataFrame, str]:
        opt = self.optimizer
        y = target

        # so much freaking uncatchable spam from this, captuing stdout only way
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = "ignore"
        silence_spam()
        f = StringIO()
        with redirect_stdout(f), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="One or more of the test scores are non-finite",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            opt.fit(X, y)  # type: ignore
        os.environ["PYTHONWARNINGS"] = before
        enable_spam()
        res = self.get_best_scores(opt)
        lines = []
        for line in f.readlines():
            if "ConvergenceWarning" in line:
                continue
        return res.to_frame().T.copy(deep=True), "\n".join(lines)

    @property
    def scorers(self) -> dict[str, Callable]:
        return CLS_SCORING if self.is_classifier else REG_SCORING

    @property
    def refit(self) -> str:
        return "acc" if self.is_classifier else "mae"

    @property
    def optimizer(self) -> GridSearchCV:
        if self.__opt is None:
            self.__opt = GridSearchCV(
                estimator=self.model(**self.fixed_args),  # type: ignore
                param_grid=self.grid,
                scoring=self.scorers,
                refit=self.refit,
                cv=5,
                n_jobs=1,
                # n_jobs=-1,
                # verbose=2,
            )
        return self.__opt

    def get_best_scores(self, opt: GridSearchCV) -> Series:
        """Returns"""
        df = DataFrame(opt.cv_results_)
        filters = [f"mean_test_{scorer}" for scorer in self.scorers]
        df = df[filters].rename(columns=lambda s: s.replace("mean_test_", ""))
        if self.is_classifier:
            return df.iloc[np.argmax(df["acc"])]
        return df.iloc[np.argmin(df["mae"])]


def logspace(start: int, stop: int) -> list[float]:
    n = stop - start + 1
    return np.logspace(start, stop, num=n, endpoint=True).tolist()


class SVMClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC
        self.grid = {
            "kernel": ["rbf"],
            "gamma": logspace(-5, 5),
            "C": np.logspace(-10, 3, num=10, endpoint=True).tolist(),
        }
        self.fixed_args = dict(probability=True)
        self.is_classifier = True
        self.short = "svm"


class LogisticClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegressionCV
        self.grid = {
            "Cs": [logspace(-7, 5)],
            "penalty": ["elasticnet"],
            "l1_ratios": [[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]],
        }
        self.fixed_args = dict(cv=5, solver="saga", max_iter=2000, n_jobs=-1)
        self.is_classifier = True
        self.short = "log"


class LightGBMClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = LGBMClassifier
        self.grid = {
            "n_estimators": [50, 200],
            "reg_alpha": np.logspace(-8, 10, num=4, endpoint=True).tolist(),
            "reg_lambda": np.logspace(-8, 10, num=4, endpoint=True).tolist(),
            # "num_leaves": [2, 31],
            # "colsample_bytree": [0.5, 0.95],
            # "subsample": [0.5, 0.95],
            # "subsample_freq": [0, 2, 4, 6],
            # "min_child_samples": [5, 25, 50],
        }
        self.fixed_args = dict(verbosity=-1)
        self.is_classifier = True
        self.short = "lgbm"
        # raise NotImplementedError("This takes too long to fit on each feature")


class DumbClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = DummyClassifier
        self.grid = {
            "strategy": ["most_frequent", "prior", "stratified", "uniform"],
        }
        self.is_classifier = True
        self.short = "dummy"


class DumbRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = DummyRegressor
        self.grid = {
            "strategy": ["mean", "median", "quantile"],
            "quantile": [0.1, 0.25, 0.75, 0.9],
        }
        self.is_classifier = False
        self.short = "dummy"


class SGDRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SklearnSGDRegressor
        self.grid = {
            "loss": ["squared_error", "huber", "epsilon_insensitive"],
            "penalty": ["l1", "l2"],
            "early_stopping": [True, False],
        }
        self.is_classifier = False
        self.short = "sgd-linear"


class SGDClassifier(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SklearnSGDClassifier
        self.grid = {
            # below are only methods that enable `predict_proba`
            "loss": ["log_loss", "modified_huber"],
            "learning_rate": ["adaptive"],
            "eta0": [3e-1, 3e-2, 3e-3, 3e-4],
            "penalty": ["l2"],
            "early_stopping": [True, False],
        }
        self.is_classifier = True
        self.short = "sgd-linear"


class ElasticNetRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = ElasticNetCV
        self.grid = {"l1_ratio": [[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]]}
        self.is_classifier = False
        self.short = "elastic"


class SVMRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVR
        self.grid = {
            "kernel": ["rbf"],
            "gamma": logspace(-5, 5),
            "C": np.logspace(-10, 3, num=10, endpoint=True).tolist(),
        }
        self.is_classifier = False
        self.short = "svm"


class LinearRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = LinearRegression
        self.grid = {}
        self.is_classifier = False
        self.short = "linear"


class LightGBMRegressor(UnivariatePredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = LGBMRegressor
        self.grid = {
            "n_estimators": [50, 200],
            "reg_alpha": np.logspace(-8, 10, num=4, endpoint=True).tolist(),
            "reg_lambda": np.logspace(-8, 10, num=4, endpoint=True).tolist(),
            # "num_leaves": [2, 31],
            # "colsample_bytree": [0.5, 0.95],
            # "subsample": [0.5, 0.95],
            # "subsample_freq": [0, 2, 4, 6],
            # "min_child_samples": [5, 25, 50],
        }
        self.fixed_args = dict(verbosity=-1)
        self.is_classifier = False
        self.short = "lgbm"


REG_MODELS: list[Type[UnivariatePredictor]] = [
    DumbRegressor,
    # ElasticNetRegressor,
    # LinearRegressor,
    # SVMRegressor,
    SGDRegressor,
    # LightGBMRegressor,
]
CLS_MODELS: list[Type[UnivariatePredictor]] = [
    DumbClassifier,
    # LogisticClassifier,
    # SVMClassifier,
    # LightGBMClassifier,
    SGDClassifier,
]


if __name__ == "__main__":
    X = np.random.uniform(0, 1, [200, 10])
    ws = np.random.uniform(0, 1, 10)
    sums = np.dot(X, ws)
    e = np.random.uniform(0, sums.mean() / 4, 200)
    y = sums + e
    y_cls = (
        KBinsDiscretizer(
            n_bins=5,
            encode="ordinal",
            quantile_method="linear",
        )
        .fit_transform(y.reshape(-1, 1))
        .ravel()
    )
    target = "target"
    y = Series(y, name=target)
    y_cls = Series(y_cls, name=target)
    df = pd.concat([DataFrame(X), y], axis=1)
    df_cls = pd.concat([DataFrame(X), y_cls], axis=1)
    X = DataFrame(X)

    models = [
        DumbRegressor(),
        SVMRegressor(),
        LightGBMRegressor(),
        LinearRegressor(),
    ]
    for model in models:
        print(model.__class__.__name__)
        res, spam = model.evaluate(X, y)
        print(res)

    models = [
        SVMClassifier(),
        LogisticClassifier(),
        # LightGBMClassifier(),
        # DumbClassifier(),
    ]
    for model in models:
        print(model.__class__.__name__)
        res, spam = model.evaluate(X, y)
        print(res)

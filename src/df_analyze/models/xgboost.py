from __future__ import annotations

from base64 import b64decode, b64encode

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Any, Callable, Mapping, Optional, Type, Union

import numpy as np
import optuna
from optuna import Study, Trial
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score

from df_analyze._constants import SEED
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import RuntimeComponent, get_runtime
from df_analyze.splitting import OmniKFold

try:
    from xgboost import XGBClassifier as SklearnXGBClassifier
    from xgboost import XGBRegressor as SklearnXGBRegressor
except ImportError as exc:
    SklearnXGBClassifier = None
    SklearnXGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None


class _EncodedXGBClassifier:
    def __init__(self, estimator: Any, classes: np.ndarray) -> None:
        self.estimator = estimator
        self.classes_ = np.asarray(classes)

    def predict(self, X) -> np.ndarray:
        codes = np.asarray(self.estimator.predict(X), dtype=int)
        return self.classes_[codes]

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))

    def score(self, X, y) -> float:
        return float(accuracy_score(y, self.predict(X)))


class XGBoostEstimator(DfAnalyzeModel):
    shortname = "xgb"
    longname = "XGBoost Estimator"
    timeout_s = 60 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.model_cls: Type[Any] = type(None)
        self.target_cols: list[str] = []
        self.default_args = dict(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=SEED,
            tree_method="hist",
            verbosity=0,
        )

    @staticmethod
    def _serialize_model(model: Any) -> dict[str, Any]:
        classes = None
        if isinstance(model, _EncodedXGBClassifier):
            classes = model.classes_.tolist()
            model = model.estimator
        raw = model.get_booster().save_raw(raw_format="ubj")
        return {
            "params": model.get_params(deep=False),
            "raw": b64encode(raw).decode("ascii"),
            "classes": classes,
        }

    def _serialize_models(self, models: Any) -> Optional[dict[str, Any]]:
        if models is None:
            return None
        if isinstance(models, dict):
            return {
                "kind": "multi",
                "models": {
                    str(target): self._serialize_model(model)
                    for target, model in models.items()
                },
            }
        return {"kind": "single", "model": self._serialize_model(models)}

    def _restore_model(self, state: Mapping[str, Any]) -> Any:
        model_cls, params = self.model_cls_args(dict(state["params"]))
        model = model_cls(**params)
        model.load_model(bytearray(b64decode(state["raw"])))
        classes = state.get("classes")
        if classes is not None:
            return _EncodedXGBClassifier(model, np.asarray(classes))
        return model

    def _restore_models(self, state: Optional[Mapping[str, Any]]) -> Any:
        if state is None:
            return None
        if state["kind"] == "multi":
            return {
                str(target): self._restore_model(model_state)
                for target, model_state in state["models"].items()
            }
        return self._restore_model(state["model"])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_serialized_model"] = self._serialize_models(state.pop("model", None))
        state["_serialized_tuned_model"] = self._serialize_models(
            state.pop("tuned_model", None)
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_state = state.pop("_serialized_model", None)
        tuned_model_state = state.pop("_serialized_tuned_model", None)
        self.__dict__.update(state)
        self.runtime = get_runtime(self.runtime.intent)
        self.model = self._restore_models(model_state)
        self.tuned_model = self._restore_models(tuned_model_state)

    @staticmethod
    def _assert_available() -> None:
        if _XGBOOST_IMPORT_ERROR is not None:
            raise ImportError(
                "XGBoost is not installed. Install it with `pip install xgboost`."
            ) from _XGBOOST_IMPORT_ERROR

    def _set_target_cols(self, y: Union[Series, DataFrame]) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            self.target_cols = [str(y.name if y.name is not None else "target")]

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        self._assert_available()
        args = dict(full_args)
        args["device"] = self.runtime.device_for(RuntimeComponent.XGBoost)
        args["tree_method"] = "hist"
        return self.model_cls, args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1e-2, 20.0, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )

    def _fit_one(self, X: DataFrame, y: Series, args: Mapping[str, Any]) -> Any:
        model_cls, clean_args = self.model_cls_args(dict(args))
        model = model_cls(**clean_args)
        if self.is_classifier:
            classes, encoded = np.unique(np.asarray(y), return_inverse=True)
            if classes.size <= 1:
                raise ValueError("XGBoost classification requires at least two classes.")
            model.fit(X, encoded)
            return _EncodedXGBClassifier(model, classes)
        model.fit(X, y)
        return model

    def _fit_models(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        args: Mapping[str, Any],
    ) -> Any:
        if not isinstance(y, DataFrame):
            return self._fit_one(X, y, args)
        if y.shape[1] == 1:
            return self._fit_one(X, y.iloc[:, 0], args)
        return {str(col): self._fit_one(X, y[col], args) for col in y.columns}

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._set_target_cols(y_train)
        args = {**self.fixed_args, **self.default_args, **self.model_args}
        self.model = self._fit_models(X_train, y_train, args)

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._set_target_cols(y)
        tuned_args = dict(tuned_args or {})
        args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **tuned_args,
        }
        self.tuned_model = self._fit_models(X, y, args)
        self.tuned_args = tuned_args
        self.is_refit = True

    def _predict_with(
        self, model: Union[Any, dict[str, Any]], X: DataFrame
    ) -> Union[Series, DataFrame]:
        if isinstance(model, dict):
            values = {
                target: np.asarray(model[target].predict(X)).reshape(-1)
                for target in self.target_cols
            }
            return DataFrame(values, index=X.index)
        name = self.target_cols[0] if self.target_cols else None
        return Series(np.asarray(model.predict(X)).reshape(-1), name=name, index=X.index)

    def predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self._predict_with(self.model, X)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return self._predict_with(self.tuned_model, X)

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")
        if not isinstance(self.tuned_model, dict):
            return float(self.tuned_model.score(X, y))
        if not isinstance(y, DataFrame):
            raise ValueError("Expected DataFrame targets for a multi-target model.")
        scores = [float(self.tuned_model[str(col)].score(X, y[col])) for col in y.columns]
        return float(np.mean(scores))

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: Optional[int] = None,
    ) -> Callable[[Trial], float]:
        n_folds = self.resolve_tuning_cv_folds(n_folds)
        self._assert_available()
        self._set_target_cols(y_train)
        y_df = y_train.to_frame() if isinstance(y_train, Series) else y_train
        y_split = self._split_target_for_cv(y_df)
        splitter = OmniKFold(
            n_splits=n_folds,
            is_classification=self.is_classifier,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            df_analyze_phase="XGBoost tuning internal splits",
        )
        splits = splitter.split(
            X_train,
            y_split,
            g_train,
            multitarget_y=(
                y_df
                if self.is_classifier and y_df.shape[1] > 1
                else None
            ),
        )[0]

        def objective(trial: Trial) -> float:
            args = {
                **self.fixed_args,
                **self.default_args,
                **self.model_args,
                **self.optuna_args(trial),
            }
            scores = []
            scores_by_target: dict[str, list[float]] = {}
            for step, (idx_train, idx_test) in enumerate(splits):
                X_tr, X_test = X_train.iloc[idx_train], X_train.iloc[idx_test]
                y_tr, y_test = y_df.iloc[idx_train], y_df.iloc[idx_test]
                y_fit: Union[Series, DataFrame]
                y_fit = y_tr.iloc[:, 0] if y_tr.shape[1] == 1 else y_tr
                model = self._fit_models(X_tr, y_fit, args)
                preds = self._predict_with(model, X_test)
                pred_df = self._preds_to_df(preds, y_test, y_test.index)
                target_scores = self._tuning_scores_by_target(
                    metric=metric,
                    y_true_df=y_test,
                    y_pred_df=pred_df,
                    y_baseline_df=y_df,
                )
                for target, target_score in target_scores.items():
                    scores_by_target.setdefault(target, []).append(target_score)
                scores.append(float(np.nanmean(list(target_scores.values()))))
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            self._record_trial_target_scores(trial, scores_by_target)
            return float(np.mean(scores))

        return objective

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
        self._assert_available()
        n_jobs = self.runtime.tuning_jobs(RuntimeComponent.XGBoost, n_jobs)
        return super().htune_optuna(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )


class XGBoostClassifier(XGBoostEstimator):
    shortname = "xgb"
    longname = "XGBoost Classifier"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = SklearnXGBClassifier  # type: ignore

    def _predict_proba_with(
        self, model: Union[Any, dict[str, Any]], X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if isinstance(model, dict):
            return {
                target: np.asarray(model[target].predict_proba(X))
                for target in self.target_cols
            }
        return np.asarray(model.predict_proba(X))

    def predict_proba_untuned(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if self.model is None:
            raise RuntimeError("Need to fit estimator before calling probabilities.")
        return self._predict_proba_with(self.model, X)

    def predict_proba(self, X: DataFrame) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")
        return self._predict_proba_with(self.tuned_model, X)


class XGBoostRegressor(XGBoostEstimator):
    shortname = "xgb"
    longname = "XGBoost Regressor"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = SklearnXGBRegressor  # type: ignore

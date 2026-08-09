# ref: CatBoost paper (ordered boosting, categorical processing): https://arxiv.org/abs/1706.09516

from __future__ import annotations

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from df_analyze._constants import SEED
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import RuntimeComponent
from df_analyze.splitting import OmniKFold

try:
    from catboost import CatBoostClassifier as CBCatBoostClassifier
    from catboost import CatBoostRegressor as CBCatBoostRegressor
except ImportError as exc:
    CBCatBoostClassifier = None
    CBCatBoostRegressor = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None


class CatBoostEstimator(DfAnalyzeModel):
    shortname = "catboost"
    longname = "CatBoost Estimator"
    runtime_component = RuntimeComponent.CatBoost
    timeout_s = 30 * 60

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.needs_calibration = False
        self.model_cls: Type[Any] = type(None)
        self.target_cols: list[str] = []
        self._tuning_thread_count: Optional[int] = None
        self.default_args = dict(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=0,
        )
        self.grid = None

    def _assert_available(self) -> None:
        if _CATBOOST_IMPORT_ERROR is not None:
            raise ImportError(
                "CatBoost is not installed. Install it with `pip install catboost`."
            ) from _CATBOOST_IMPORT_ERROR

    def _has_gpu(self) -> bool:
        """Compatibility helper backed by the centralized runtime policy."""
        return self.runtime.device_for(RuntimeComponent.CatBoost) == "cuda"

    def _maybe_use_gpu(self, kwargs: dict[str, Any]) -> None:
        task_type = str(kwargs.get("task_type", "")).upper()
        if self._has_gpu():
            kwargs["task_type"] = "GPU"
            kwargs.setdefault("devices", "0")
            return
        if task_type == "GPU":
            kwargs["task_type"] = "CPU"
            kwargs.pop("devices", None)

    @staticmethod
    def _uses_gpu(kwargs: Mapping[str, Any]) -> bool:
        return str(kwargs.get("task_type", "")).upper() == "GPU"

    def _set_target_cols(self, y: Union[Series, DataFrame]) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
            return
        name = y.name if y.name is not None else "target"
        self.target_cols = [str(name)]

    def _loss_for_target(self, ycol: Series) -> str:
        return "RMSE"

    def _loss_args_for_target(self, ycol: Series) -> dict[str, Any]:
        return {"loss_function": self._loss_for_target(ycol)}

    def _native_multitarget_loss(self, y: DataFrame) -> Optional[str]:
        if y.shape[1] <= 1:
            return None
        if self.is_classifier:
            for col in y.columns:
                vals = np.asarray(y[col].dropna().to_numpy())
                if vals.size == 0:
                    return None
                uniq = np.unique(vals)
                if not np.isin(uniq, [0, 1]).all():
                    return None
            return "MultiLogloss"
        return "MultiRMSE"

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            iterations=trial.suggest_int("iterations", 100, 400, step=50),
        )

    def _fit_single_target(
        self,
        X: DataFrame,
        y: Series,
        kwargs: Mapping[str, Any],
    ) -> Any:
        model_args = dict(kwargs)
        for key, value in self._loss_args_for_target(y).items():
            model_args.setdefault(key, value)
        model_cls, clean_args = self.model_cls_args(model_args)
        model = model_cls(**clean_args)
        model.fit(X, y)
        return model

    def _fit_target_models(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if isinstance(y, DataFrame):
            if y.shape[1] == 1:
                return self._fit_single_target(X, y.iloc[:, 0], kwargs)
            native_loss = self._native_multitarget_loss(y)
            if native_loss is not None:
                model_args = dict(kwargs)
                model_args.setdefault("loss_function", native_loss)
                model_cls, clean_args = self.model_cls_args(model_args)
                model = model_cls(**clean_args)
                if not self.is_classifier:
                    constant = [
                        str(col)
                        for col in y.columns
                        if y[col].nunique(dropna=False) < 2
                    ]
                    if constant:
                        raise ValueError(
                            "Multi-target regression has constant target(s) in a "
                            f"CatBoost training partition: {constant}."
                        )
                    estimator = TransformedTargetRegressor(
                        regressor=model,
                        transformer=StandardScaler(),
                    )
                    estimator.fit(X, y)
                    return estimator
                model.fit(X, y)
                return model
            model_cls, clean_args = self.model_cls_args(dict(kwargs))
            estimator = MultiOutputClassifier(model_cls(**clean_args))
            estimator.fit(X, y)
            return estimator
        return self._fit_single_target(X, y, kwargs)

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._assert_available()
        self._set_target_cols(y_train)
        kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
        self._maybe_use_gpu(kwargs)
        self.model = self._fit_target_models(X=X_train, y=y_train, kwargs=kwargs)

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._assert_available()
        self._set_target_cols(y)
        tuned_args = tuned_args or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **tuned_args,
        }
        self._maybe_use_gpu(kwargs)
        self.tuned_model = self._fit_target_models(X=X, y=y, kwargs=kwargs)
        self.tuned_args = dict(tuned_args)
        self.is_refit = True

    def _predict_values(self, model: Any, X: DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X))

    def _predict_from_model(
        self, model_obj: Union[Any, dict[str, Any]], X: DataFrame
    ) -> Union[Series, DataFrame]:
        if isinstance(model_obj, dict):
            if len(self.target_cols) == 1:
                target = self.target_cols[0] if len(self.target_cols) > 0 else None
                if target not in model_obj:
                    target = next(iter(model_obj))
                preds = np.asarray(model_obj[target].predict(X)).reshape(-1)
                return Series(preds, name=target, index=X.index)
            out = {
                target: np.asarray(model_obj[target].predict(X)).reshape(-1)
                for target in self.target_cols
            }
            return DataFrame(out, index=X.index)
        preds = self._predict_values(model_obj, X)
        if (
            preds.ndim == 2
            and len(self.target_cols) > 1
            and preds.shape[1] == len(self.target_cols)
        ):
            return DataFrame(preds, index=X.index, columns=self.target_cols)
        name = self.target_cols[0] if len(self.target_cols) > 0 else None
        return Series(preds.reshape(-1), name=name, index=X.index)

    def predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self._predict_from_model(self.model, X)

    def tuned_predict(self, X: DataFrame) -> Union[Series, DataFrame]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return self._predict_from_model(self.tuned_model, X)

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

        kf = OmniKFold(
            n_splits=n_folds,
            is_classification=self.is_classifier,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            df_analyze_phase="Tuning internal splits",
        )
        splits, _ = kf.split(
            X_train=X_train,
            y_train=y_split,
            g_train=g_train,
            multitarget_y=(
                y_df
                if y_df.shape[1] > 1
                else None
            ),
        )

        def objective(trial: Trial) -> float:
            opt_args = self.optuna_args(trial)
            full_args = {
                **self.fixed_args,
                **self.default_args,
                **self.model_args,
                **opt_args,
            }
            self._maybe_use_gpu(full_args)
            if self._tuning_thread_count is not None:
                # Optuna is already parallelizing trials. Keep each CatBoost
                # fit single-threaded to avoid nested CPU oversubscription.
                full_args["thread_count"] = self._tuning_thread_count
            scores = []
            scores_by_target: dict[str, list[float]] = {}
            for step, (idx_train, idx_test) in enumerate(splits):
                X_tr = X_train.iloc[idx_train]
                X_te = X_train.iloc[idx_test]
                y_tr = y_df.iloc[idx_train]
                y_te = y_df.iloc[idx_test]
                y_fit: Union[Series, DataFrame]
                if y_tr.shape[1] == 1:
                    y_fit = y_tr.iloc[:, 0]
                else:
                    y_fit = y_tr
                model = self._fit_target_models(X=X_tr, y=y_fit, kwargs=full_args)
                preds = self._predict_from_model(model, X_te)
                pred_df = self._preds_to_df(preds, y_te, y_te.index)
                target_scores = self._tuning_scores_by_target(
                    metric=metric,
                    y_true_df=y_te,
                    y_pred_df=pred_df,
                    y_baseline_df=y_df,
                )
                for target, target_score in target_scores.items():
                    scores_by_target.setdefault(target, []).append(target_score)
                score = float(np.nanmean(list(target_scores.values())))

                scores.append(score)
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
        base_args = {**self.fixed_args, **self.default_args, **self.model_args}
        self._maybe_use_gpu(base_args)
        n_jobs = self.runtime.tuning_jobs(RuntimeComponent.CatBoost, n_jobs)
        decision = self.runtime.decision_for(RuntimeComponent.CatBoost)
        self._tuning_thread_count = (
            1 if decision.resolved == "cpu" and n_jobs != 1 else None
        )
        try:
            return super().htune_optuna(
                X_train=X_train,
                y_train=y_train,
                g_train=g_train,
                metric=metric,
                n_trials=n_trials,
                n_jobs=n_jobs,
                verbosity=verbosity,
            )
        finally:
            self._tuning_thread_count = None

    def _score_outputs_joint(
        self,
        y_true_df: DataFrame,
        y_pred_df: DataFrame,
        y_prob: Optional[
            Union[
                np.ndarray,
                dict[str, np.ndarray],
                list[np.ndarray],
                tuple[np.ndarray, ...],
            ]
        ] = None,
    ) -> dict[str, float]:
        joint_scores = super()._score_outputs_joint(
            y_true_df=y_true_df, y_pred_df=y_pred_df, y_prob=y_prob
        )
        if y_true_df.shape[1] <= 1:
            return joint_scores
        if self.is_classifier:
            multi_logloss = self._multitarget_logloss(
                y_true_df=y_true_df, y_prob=y_prob
            )
            if multi_logloss is not None:
                joint_scores["multi-logloss"] = multi_logloss
        return joint_scores

    def _multitarget_logloss(
        self,
        y_true_df: DataFrame,
        y_prob: Optional[
            Union[
                np.ndarray,
                dict[str, np.ndarray],
                list[np.ndarray],
                tuple[np.ndarray, ...],
            ]
        ],
    ) -> Optional[float]:
        target_cols = [str(col) for col in y_true_df.columns]
        probs_by_target = self._probs_by_target(y_prob, target_cols)
        if len(probs_by_target) != len(target_cols):
            return None

        y_true = np.asarray(y_true_df.to_numpy(), dtype=float)
        if not np.isin(y_true, [0.0, 1.0]).all():
            return None

        prob_cols: list[np.ndarray] = []
        for col in target_cols:
            p = np.asarray(probs_by_target.get(col))
            if p.ndim == 1:
                pos = p
            elif p.ndim == 2 and p.shape[1] == 2:
                pos = p[:, 1]
            elif p.ndim == 2 and p.shape[1] == 1:
                pos = p[:, 0]
            else:
                return None
            if pos.shape[0] != y_true.shape[0]:
                return None
            prob_cols.append(np.asarray(pos, dtype=float))

        probs = np.column_stack(prob_cols)
        eps = np.finfo(float).eps
        probs = np.clip(probs, eps, 1.0 - eps)
        loss = -(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
        return float(np.mean(loss))


class CatBoostClassifier(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Classifier"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.model_cls = CBCatBoostClassifier  # type: ignore

    def _loss_for_target(self, ycol: Series) -> str:
        n_classes = int(ycol.nunique(dropna=False))
        return "Logloss" if n_classes <= 2 else "MultiClass"

    def predict_proba(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.predict_proba()`"
            )
        if isinstance(self.tuned_model, dict):
            if len(self.target_cols) == 1:
                target = self.target_cols[0] if len(self.target_cols) > 0 else None
                if target not in self.tuned_model:
                    target = next(iter(self.tuned_model))
                return np.asarray(self.tuned_model[target].predict_proba(X))
            return {
                target: np.asarray(self.tuned_model[target].predict_proba(X))
                for target in self.target_cols
            }
        raw_probs = self.tuned_model.predict_proba(X)
        if isinstance(raw_probs, (list, tuple)):
            return {
                target: np.asarray(values)
                for target, values in zip(self.target_cols, raw_probs, strict=True)
            }
        probs = np.asarray(raw_probs)
        if (
            probs.ndim == 2
            and len(self.target_cols) > 1
            and probs.shape[1] == len(self.target_cols)
        ):
            return {
                target: np.stack([1.0 - probs[:, i], probs[:, i]], axis=1)
                for i, target in enumerate(self.target_cols)
            }
        return probs


class CatBoostRegressor(CatBoostEstimator):
    shortname = "catboost"
    longname = "CatBoost Regressor"

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.model_cls = CBCatBoostRegressor  # type: ignore

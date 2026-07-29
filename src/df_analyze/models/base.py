# # ref: Multiclass & multioutput algorithms: https://scikit-learn.org/stable/modules/multiclass.html

from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from math import ceil
from numbers import Integral, Real
from pathlib import Path
from random import choice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    overload,
)
from warnings import warn

if TYPE_CHECKING:
    from df_analyze.testing.datasets import TestDataset

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from optuna import Study, Trial, create_study
from optuna.logging import _get_library_root_logger
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler, TPESampler
from optuna.trial import FrozenTrial
from pandas import DataFrame, Series
from sklearn.calibration import CalibratedClassifierCV as CVCalibrate
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

from df_analyze._constants import SEED
from df_analyze.enumerables import (
    Scorer,
    WrapperSelection,
)
from df_analyze.runtime.hardware import DeviceIntent, RuntimePolicy, get_runtime
from df_analyze.splitting import OmniKFold, resolve_final_cv_folds, y_split_label

NEG_MAE = "neg_mean_absolute_error"

OPT_LOGGER = _get_library_root_logger()


_MULTITARGET_FIT_ERRORS = (
    "y should be a 1d array",
    "bad input shape",
    "multioutput target data is not supported",
    "multi-output target data is not supported",
    "dataframe for label cannot have multiple columns",
    "label should be a 1-dimensional array",
)


def _is_multitarget_fit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in _MULTITARGET_FIT_ERRORS)


def classification_output_dim(y: Series) -> int:
    values = np.asarray(y)
    if values.size == 0:
        raise ValueError("Cannot determine class count from an empty target.")
    if np.issubdtype(values.dtype, np.integer) and np.min(values) >= 0:
        return int(np.max(values)) + 1
    return int(np.unique(values).size)


def _calibration_folds(y: Series) -> Optional[int]:
    counts = y.value_counts(dropna=False)
    if counts.empty:
        return None
    folds = min(5, int(counts.min()))
    return folds if folds >= 2 else None


class PerTargetEstimator:
    """Adapter for estimators that only accept a one-dimensional target."""

    def __init__(
        self,
        model_cls: Type[Any],
        model_args: Mapping[str, Any],
        is_classifier: bool,
        needs_calibration: bool = False,
    ) -> None:
        self.model_cls = model_cls
        self.model_args = dict(model_args)
        self.is_classifier = is_classifier
        self.needs_calibration = needs_calibration
        self.target_cols: list[str] = []
        self.models: dict[str, Any] = {}

    def _new_model(self, y: Series) -> Any:
        model = self.model_cls(**self.model_args)
        if self.needs_calibration:
            folds = _calibration_folds(y)
            if folds is None:
                warn(
                    f"Skipping probability calibration for target '{y.name}' because "
                    "one of its classes has fewer than two training samples."
                )
            else:
                model = CVCalibrate(model, method="sigmoid", cv=folds, n_jobs=folds)
        return model

    def fit(self, X: DataFrame, y: DataFrame) -> PerTargetEstimator:
        if not isinstance(y, DataFrame):
            raise ValueError("PerTargetEstimator requires DataFrame targets.")
        self.target_cols = [str(col) for col in y.columns]
        self.models = {}
        for col in y.columns:
            model = self._new_model(y[col])
            model.fit(X, y[col])
            self.models[str(col)] = model
        return self

    def predict(self, X: DataFrame) -> DataFrame:
        predictions = {
            target: np.asarray(model.predict(X)).reshape(-1)
            for target, model in self.models.items()
        }
        return DataFrame(predictions, index=X.index, columns=self.target_cols)

    def predict_proba(self, X: DataFrame) -> dict[str, ndarray]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        return {
            target: np.asarray(model.predict_proba(X))
            for target, model in self.models.items()
        }

    def score(self, X: DataFrame, y: DataFrame) -> float:
        if not isinstance(y, DataFrame):
            raise ValueError("PerTargetEstimator requires DataFrame targets.")
        scores = [
            float(self.models[str(col)].score(X, y[col]))
            for col in y.columns
            if str(col) in self.models
        ]
        if not scores:
            raise ValueError("No target models were available for scoring.")
        return float(np.mean(scores))


class EarlyStopping:
    def __init__(self, patience: int = 10, min_trials: int = 50) -> None:
        self.patience: int = patience
        self.min_trials: int = min_trials
        self.has_stopped: bool = False

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """https://github.com/optuna/optuna/issues/1001#issuecomment-1351766030"""
        if self.has_stopped:
            # raise optuna.exceptions.TrialPruned()
            study.stop()

        current_trial = trial.number
        if current_trial < self.min_trials:
            return
        best_trial = study.best_trial.number

        # best_score = study.best_value  # TODO: patience
        should_stop = (current_trial - best_trial) >= self.patience
        if should_stop:
            if not self.has_stopped:
                OPT_LOGGER.info(
                    f"Completed {self.patience} trials without metric improvement. "
                    f"Stopping early at trial {current_trial}. Some trials may still "
                    "need to finish, and produce a better result. If so, that better "
                    f"result will be used rather than trial {current_trial}."
                )
            self.has_stopped = True
            study.stop()


class DfAnalyzeModel(ABC):
    shortname: str = ""
    longname: str = ""
    timeout_s: int = 3600  # one hour
    # ``None`` means the model does not use K-fold hyperparameter tuning.
    # Multi-target preflight validation uses this declaration so it checks the
    # same fold design that the model will actually run.
    tuning_cv_folds: Optional[int] = 5

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__()
        self.is_classifier: bool = True
        self.needs_calibration: bool = False
        self.model: Optional[Any] = None
        self.fixed_args: dict[str, Any] = {}
        self.default_args: dict[str, Any] = {}
        self.model_args: Mapping = model_args or {}
        self.grid: Optional[dict[str, Any]] = None
        self._multitarget_fallback_warned: set[str] = set()

        self.tuned_args: Optional[dict[str, Any]] = None
        self.tuned_model: Optional[Any] = None
        self.per_target_tuning_scores: dict[str, float] = {}
        self.is_refit = False
        self.runtime: RuntimePolicy = get_runtime(DeviceIntent.CPU)

    def set_runtime(self, runtime: RuntimePolicy) -> DfAnalyzeModel:
        self.runtime = runtime
        self._configure_runtime()
        return self

    def resolve_tuning_cv_folds(self, n_folds: Optional[int] = None) -> int:
        """Resolve and validate the K-fold design used by this model."""
        resolved = self.tuning_cv_folds if n_folds is None else n_folds
        if resolved is None:
            raise ValueError(
                f"{self.__class__.__name__} does not use K-fold hyperparameter tuning."
            )
        resolved = int(resolved)
        if resolved < 2:
            raise ValueError("K-fold hyperparameter tuning requires at least two folds.")
        return resolved

    def _configure_runtime(self) -> None:
        return

    def _cleanup_after_fold(self) -> None:
        return

    @abstractmethod
    def model_cls_args(
        self, full_args: dict[str, Any]
    ) -> tuple[Type[Any], dict[str, Any]]:
        """Allows for conditioning the model based on args (e.g. SVC vs. LinearSVC
        depending on kernel, and also subsequent removal or addition of necessary
        args because of this.

        Returns
        -------
        model_cls: Type[Any]
            The model class needed based on `full_args`

        clean_args: dict[str, Any]
            The args that now work for the returned `model_cls`

        """
        ...

    def _split_target_for_cv(self, y: Union[Series, DataFrame]) -> Series:
        if isinstance(y, DataFrame):
            if self.is_classifier:
                return y_split_label(y, warn_on_fallback=False)
            for col in y.columns:
                if y[col].nunique(dropna=False) > 1:
                    return y[col]
            return y.iloc[:, 0]
        return y

    @staticmethod
    def _prepare_score_frames(
        y_true: Union[Series, DataFrame],
        y_pred: Union[Series, DataFrame, ndarray],
    ) -> tuple[DataFrame, DataFrame]:
        if isinstance(y_true, DataFrame):
            y_true_df = y_true.copy()
        else:
            name = str(y_true.name) if y_true.name is not None else "target"
            y_true_df = y_true.to_frame(name=name)

        y_pred_df = DfAnalyzeModel._preds_to_df(y_pred, y_true_df, y_true_df.index)
        y_pred_df = DfAnalyzeModel._align_pred_columns(y_pred_df, y_true_df)
        return y_true_df, y_pred_df

    @staticmethod
    def _align_pred_columns(y_pred_df: DataFrame, y_true_df: DataFrame) -> DataFrame:
        if y_pred_df.shape[1] != y_true_df.shape[1]:
            return y_pred_df
        if list(y_pred_df.columns) == list(y_true_df.columns):
            return y_pred_df
        out = y_pred_df.copy()
        if set(out.columns) == set(y_true_df.columns):
            return out.reindex(columns=y_true_df.columns)
        warn("Prediction columns mismatch; aligned by position.")
        out.columns = y_true_df.columns
        return out

    @staticmethod
    def _preds_to_df(
        preds: Union[Series, DataFrame, ndarray],
        y_true_df: DataFrame,
        index: pd.Index,
    ) -> DataFrame:
        if isinstance(preds, DataFrame):
            out = preds.copy().reindex(index=index)
            if out.shape[1] == y_true_df.shape[1]:
                out = DfAnalyzeModel._align_pred_columns(out, y_true_df)
            return out
        if isinstance(preds, Series):
            col = y_true_df.columns[0]
            return DataFrame({col: preds.to_numpy()}, index=index)

        arr = np.asarray(preds)
        if arr.ndim == 1:
            col = y_true_df.columns[0]
            return DataFrame({col: arr}, index=index)
        if arr.ndim == 2:
            cols = (
                list(y_true_df.columns)
                if arr.shape[1] == y_true_df.shape[1]
                else [f"target_{i}" for i in range(arr.shape[1])]
            )
            return DataFrame(arr, index=index, columns=cols)
        raise ValueError("Unsupported prediction output shape for multi-target scoring.")

    @staticmethod
    def _probs_by_target(
        probs: Optional[
            Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
        ],
        target_cols: list[str],
    ) -> dict[str, ndarray]:
        if probs is None:
            return {}
        if isinstance(probs, dict):
            return {str(key): np.asarray(value) for key, value in probs.items()}
        if isinstance(probs, (list, tuple)):
            return {
                str(col): np.asarray(probs[i])
                for i, col in enumerate(target_cols)
                if i < len(probs)
            }
        arr = np.asarray(probs)
        if arr.ndim == 3:
            return {
                str(col): np.asarray(arr[:, i, :])
                for i, col in enumerate(target_cols)
                if i < arr.shape[1]
            }
        if arr.ndim == 2 and len(target_cols) > 1 and arr.shape[1] == len(target_cols):
            return {
                str(col): np.stack([1.0 - arr[:, i], arr[:, i]], axis=1)
                for i, col in enumerate(target_cols)
            }
        if arr.ndim == 2 and len(target_cols) == 1:
            return {str(target_cols[0]): arr}
        return {}

    @staticmethod
    def _mean_scores(per_target: list[dict[str, float]]) -> dict[str, float]:
        if len(per_target) == 0:
            return {}
        keys = list(per_target[0].keys())
        out: dict[str, float] = {}
        for key in keys:
            vals = [float(scores.get(key, np.nan)) for scores in per_target]
            arr = np.asarray(vals, dtype=float)
            out[key] = float(np.nan) if np.all(np.isnan(arr)) else float(np.nanmean(arr))
        return out

    def _score_outputs(
        self,
        y_true: Union[Series, DataFrame],
        y_pred: Union[Series, DataFrame, ndarray],
        y_prob: Optional[
            Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
        ] = None,
    ) -> dict[str, float]:
        y_true_df, y_pred_df = self._prepare_score_frames(y_true=y_true, y_pred=y_pred)
        by_target = self._score_outputs_by_target_df(
            y_true_df=y_true_df, y_pred_df=y_pred_df, y_prob=y_prob
        )
        if len(by_target) == 0:
            return {}
        if len(by_target) == 1:
            return list(by_target.values())[0]
        scores = self._mean_scores(list(by_target.values()))
        joint_scores = self._score_outputs_joint(
            y_true_df=y_true_df, y_pred_df=y_pred_df, y_prob=y_prob
        )
        if len(joint_scores) > 0:
            scores = {**scores, **joint_scores}
        return scores

    def _score_outputs_joint(
        self,
        y_true_df: DataFrame,
        y_pred_df: DataFrame,
        y_prob: Optional[
            Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
        ] = None,
    ) -> dict[str, float]:
        if y_true_df.shape[1] <= 1:
            return {}

        if self.is_classifier:
            y_true = np.asarray(y_true_df.to_numpy())
            y_pred = np.asarray(y_pred_df.to_numpy())
            subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))
            ham_loss = float(np.mean(y_true != y_pred))
            return {
                "subset-acc": subset_acc,
                "hamming-loss": ham_loss,
                "hamming-acc": 1.0 - ham_loss,
            }

        y_true = np.asarray(y_true_df.to_numpy(), dtype=float)
        y_pred = np.asarray(y_pred_df.to_numpy(), dtype=float)
        residual = y_true - y_pred
        sq_norm = np.sum(np.square(residual), axis=1)
        multi_mse = float(mse(y_true, y_pred, multioutput="uniform_average"))
        return {
            "multi-mae": float(mae(y_true, y_pred, multioutput="uniform_average")),
            "multi-mse": multi_mse,
            "multi-rmse": float(np.sqrt(np.mean(sq_norm))),
            "multi-rmse-uniform": float(np.sqrt(multi_mse)),
            "multi-r2": float(r2(y_true, y_pred, multioutput="uniform_average")),
            "multi-r2-var": float(r2(y_true, y_pred, multioutput="variance_weighted")),
        }

    def _score_outputs_by_target(
        self,
        y_true: Union[Series, DataFrame],
        y_pred: Union[Series, DataFrame, ndarray],
        y_prob: Optional[
            Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
        ] = None,
    ) -> dict[str, dict[str, float]]:
        y_true_df, y_pred_df = self._prepare_score_frames(y_true=y_true, y_pred=y_pred)
        return self._score_outputs_by_target_df(
            y_true_df=y_true_df, y_pred_df=y_pred_df, y_prob=y_prob
        )

    def _score_outputs_by_target_df(
        self,
        y_true_df: DataFrame,
        y_pred_df: DataFrame,
        y_prob: Optional[
            Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]
        ] = None,
    ) -> dict[str, dict[str, float]]:
        from df_analyze.hypertune import (
            ClassifierScorer,
            RegressorScorer,
        )

        out: dict[str, dict[str, float]] = {}
        target_cols = [str(col) for col in y_true_df.columns]
        if self.is_classifier:
            probs_by_target = self._probs_by_target(y_prob, target_cols)
            for col in y_true_df.columns:
                key = str(col)
                out[key] = ClassifierScorer.get_scores(
                    y_true=y_true_df[col].to_numpy(),
                    y_pred=y_pred_df[col].to_numpy(),
                    y_prob=probs_by_target.get(key),
                )
            return out

        for col in y_true_df.columns:
            key = str(col)
            out[key] = RegressorScorer.get_scores(
                y_true=y_true_df[col].to_numpy(),
                y_pred=y_pred_df[col].to_numpy(),
            )
        return out

    def _mean_tuning_score(
        self,
        metric: Scorer,
        y_true_df: DataFrame,
        y_pred_df: DataFrame,
        y_baseline_df: Optional[DataFrame] = None,
    ) -> float:
        per_target = self._tuning_scores_by_target(
            metric=metric,
            y_true_df=y_true_df,
            y_pred_df=y_pred_df,
            y_baseline_df=y_baseline_df,
        )
        scores = list(per_target.values())
        if len(scores) == 0:
            raise ValueError("No valid target columns were available for tuning score.")
        score = float(np.nanmean(scores))
        if np.isnan(score):
            raise ValueError("All per-target tuning scores were NaN.")
        return score

    def _tuning_scores_by_target(
        self,
        metric: Scorer,
        y_true_df: DataFrame,
        y_pred_df: DataFrame,
        y_baseline_df: Optional[DataFrame] = None,
    ) -> dict[str, float]:
        y_pred_df = self._align_pred_columns(y_pred_df, y_true_df)
        scores: dict[str, float] = {}
        for col in y_true_df.columns:
            score = metric.tuning_score(
                y_true_df[col].to_numpy(), y_pred_df[col].to_numpy()
            )
            if y_true_df.shape[1] > 1:
                baseline_values = (
                    y_true_df[col]
                    if y_baseline_df is None
                    else y_baseline_df[col]
                )
                score = self._scale_tuning_score(metric, baseline_values, score)
            scores[str(col)] = float(score)
        return scores

    @staticmethod
    def _record_trial_target_scores(
        trial: Trial,
        scores_by_target: Mapping[str, list[float]],
    ) -> None:
        summary: dict[str, float] = {}
        for target, values in scores_by_target.items():
            array = np.asarray(values, dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size > 0:
                summary[str(target)] = float(np.mean(finite))
        if summary:
            trial.set_user_attr("per_target_tuning_scores", summary)

    def _capture_best_target_scores(self, study: Study) -> None:
        raw = study.best_trial.user_attrs.get("per_target_tuning_scores", {})
        if not isinstance(raw, Mapping):
            self.per_target_tuning_scores = {}
            return
        self.per_target_tuning_scores = {
            str(target): float(score)
            for target, score in raw.items()
            if np.isfinite(float(score))
        }

    @staticmethod
    def _scale_tuning_score(
        metric: Scorer, y_true: Union[Series, ndarray], score: float
    ) -> float:
        if metric.higher_is_better():
            return float(score)
        values = np.asarray(y_true, dtype=float).reshape(-1)
        center = np.mean(values) if metric.value == "msqe" else np.median(values)
        baseline = np.full(values.shape, center, dtype=float)
        scale = float(metric.tuning_score(values, baseline))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            return float(score)
        return float(score) / scale

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: Optional[int] = None,
    ) -> Callable[[Trial], float]:
        n_folds = self.resolve_tuning_cv_folds(n_folds)
        y_split = self._split_target_for_cv(y_train)

        def objective(trial: Trial) -> float:
            # if self.is_classifier:
            #     kf = StratifiedKFold if g_train is None else StratifiedGroupKFold
            # else:
            #     kf = KFold if g_train is None else GroupKFold
            kf = OmniKFold(
                n_splits=n_folds,
                is_classification=self.is_classifier,
                grouped=g_train is not None,
                labels=None,
                warn_on_fallback=False,
                allow_group_fallback=False,
                df_analyze_phase="Tuning internal splits",
            )
            splits, group_fail = kf.split(
                X_train=X_train,
                y_train=y_split,
                g_train=g_train,
                multitarget_y=(
                    y_train
                    if self.is_classifier
                    and isinstance(y_train, DataFrame)
                    and y_train.shape[1] > 1
                    else None
                ),
            )

            # if g_train is None:
            #     _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)  # type: ignore
            # else:
            #     _cv = kf(n_splits=n_folds)
            opt_args = self.optuna_args(trial)
            full_args = {
                **self.fixed_args,
                **self.default_args,
                **self.model_args,
                **opt_args,
            }
            scores = []
            scores_by_target: dict[str, list[float]] = {}
            # try:
            #     splits = [split for split in enumerate(_cv.split(X_train, y_train))]
            # except Exception as e:
            #     traceback.print_exc()
            #     print(
            #         f"Got error {e} when attempting to split data. Most likely this means "
            #         "a grouping variable (`--grouper`) was passed into the df-analyze CLI, "
            #         "but that you have insufficient data for each group and target level "
            #         "to ensure that all target levels are present in all train-test splits. "
            #         "Falling back to regular stratified splitting. "
            #     )
            #     kf = StratifiedKFold if self.is_classifier else KFold
            #     _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            #     splits = [split for split in enumerate(_cv.split(X_train, y_train))]

            for step, (idx_train, idx_test) in enumerate(splits):
                estimator = None
                try:
                    X_tr = X_train.iloc[idx_train]
                    X_test = X_train.iloc[idx_test]
                    y_tr = y_train.iloc[idx_train]
                    y_test = y_train.iloc[idx_test]
                    model_cls, clean_args = self.model_cls_args(full_args)
                    estimator = self._fit_estimator(
                        model_cls=model_cls,
                        clean_args=clean_args,
                        X=X_tr,
                        y=y_tr,
                    )
                    preds = estimator.predict(X_test)
                    if isinstance(y_test, DataFrame):
                        pred_df = self._preds_to_df(preds, y_test, y_test.index)
                        target_scores = self._tuning_scores_by_target(
                            metric=metric,
                            y_true_df=y_test,
                            y_pred_df=pred_df,
                            y_baseline_df=y_train,
                        )
                        for target, target_score in target_scores.items():
                            scores_by_target.setdefault(target, []).append(target_score)
                        score = float(np.nanmean(list(target_scores.values())))
                    else:
                        score = float(metric.tuning_score(y_test, preds))
                    scores.append(score)
                    # allows pruning
                    # Reporting each intermediate fold score makes that pruning
                    # decision reflect progress across the completed folds.
                    trial.report(float(np.mean(scores)), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                finally:
                    estimator = None
                    self._cleanup_after_fold()
            self._record_trial_target_scores(trial, scores_by_target)
            return float(np.mean(scores))

        return objective

    @abstractmethod
    def optuna_args(self, trial: Trial) -> dict[str, Union[str, float, int]]:
        return {}

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
        if self.tuned_args is not None:
            raise RuntimeError(
                f"Model {self.__class__.__name__} has already been tuned with Optuna"
            )

        grid = self.grid
        direction = "maximize" if metric.higher_is_better() else "minimize"
        study = create_study(
            direction=direction,
            sampler=(
                GridSampler(grid, seed=SEED)
                if grid is not None
                else TPESampler(seed=SEED)
            ),
            pruner=MedianPruner(n_warmup_steps=0, n_min_trials=5),
        )
        optuna.logging.set_verbosity(verbosity)
        objective = self.optuna_objective(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
        )
        cbs = [EarlyStopping(patience=15, min_trials=50)]
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=cbs if grid is None else [],
            timeout=self.__class__.timeout_s,
            n_jobs=n_jobs,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        finite_trials = [
            trial
            for trial in study.trials
            if trial.value is not None and np.isfinite(float(trial.value))
        ]
        if not finite_trials:
            raise RuntimeError(
                f"Hyperparameter tuning failed for {self.__class__.__name__}: "
                "no trial produced a finite score."
            )
        self._capture_best_target_scores(study)
        self.tuned_args = study.best_params
        self.refit_tuned(X=X_train, y=y_train, tuned_args=self.tuned_args)

        return study

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
        model_cls, clean_args = self.model_cls_args(kwargs)
        self.model = self._fit_estimator(model_cls, clean_args, X_train, y_train)

    def _fit_estimator(
        self,
        model_cls: Type[Any],
        clean_args: Mapping[str, Any],
        X: DataFrame,
        y: Union[Series, DataFrame],
        needs_calibration: bool = False,
    ) -> Any:
        estimator = model_cls(**dict(clean_args))
        if needs_calibration and isinstance(y, Series):
            folds = _calibration_folds(y)
            if folds is None:
                warn(
                    f"Skipping probability calibration for target '{y.name}' because "
                    "one of its classes has fewer than two training samples."
                )
            else:
                estimator = CVCalibrate(
                    estimator, method="sigmoid", cv=folds, n_jobs=folds
                )

        try:
            estimator.fit(X, y)
            return estimator
        except (TypeError, ValueError) as error:
            if (
                not isinstance(y, DataFrame)
                or y.shape[1] <= 1
                or not _is_multitarget_fit_error(error)
            ):
                raise
            model_name = model_cls.__name__
            if model_name not in self._multitarget_fallback_warned:
                warn(
                    f"{model_name} does not support multi-target fitting; "
                    "using one estimator per target."
                )
                self._multitarget_fallback_warned.add(model_name)
            return PerTargetEstimator(
                model_cls=model_cls,
                model_args=clean_args,
                is_classifier=self.is_classifier,
                needs_calibration=needs_calibration,
            ).fit(X, y)

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        tuned_args = tuned_args or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **tuned_args,
        }
        model_cls, clean_args = self.model_cls_args(kwargs)
        self.tuned_model = self._fit_estimator(
            model_cls=model_cls,
            clean_args=clean_args,
            X=X,
            y=y,
            needs_calibration=self.needs_calibration,
        )

    def htune_eval(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        X_test: DataFrame,
        y_test: Union[Series, DataFrame],
        g_test: Optional[Series],
        seed: int,
    ) -> tuple[
        DataFrame,
        Union[Series, DataFrame, ndarray],
        Union[Series, DataFrame, ndarray],
        Optional[Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]],
        Optional[Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]],
        Optional[DataFrame],
    ]:
        """
        Returns
        -------
        df: DataFrame
            DataFrame with columns [trainset, holdout, 5-fold,
            final_cv_folds] and index as the scorers. `5-fold` is retained as a
            compatibility column name; `final_cv_folds` records the actual
            number of group-disjoint folds when fewer than five holdout groups
            are available.
        """
        # TODO: need to specify valiation method, and return confidences, etc.
        # Actually maybe just want to call refit in here...
        if self.tuned_model is None:
            raise RuntimeError("Cannot evaluate tuning because model has not been tuned.")
        holdout_preds_test = self.tuned_predict(X_test)
        preds_train = self.tuned_predict(X_train)
        probs_train = holdout_probs_test = None
        if self.is_classifier:
            holdout_probs_test = self.predict_proba(X_test)
            probs_train = self.predict_proba(X_train)
        holdout_scores = self._score_outputs(
            y_true=y_test,
            y_pred=holdout_preds_test,
            y_prob=holdout_probs_test,
        )
        holdout_scores_by_target = self._score_outputs_by_target(
            y_true=y_test,
            y_pred=holdout_preds_test,
            y_prob=holdout_probs_test,
        )
        train_scores = self._score_outputs(
            y_true=y_train,
            y_pred=preds_train,
            y_prob=probs_train,
        )
        train_scores_by_target = self._score_outputs_by_target(
            y_true=y_train,
            y_pred=preds_train,
            y_prob=probs_train,
        )

        final_cv_folds = resolve_final_cv_folds(y_test, g_test)
        kf = OmniKFold(
            n_splits=final_cv_folds,
            is_classification=self.is_classifier,
            grouped=g_test is not None,
            labels=None,
            warn_on_fallback=True,
            allow_group_fallback=False,
            df_analyze_phase="Final k-fold on holdout set",
            seed=seed,
        )

        tuned_model_orig = self.tuned_model
        scores = []
        fold_scores_by_target: dict[str, list[dict[str, float]]] = {}
        y_cv = self._split_target_for_cv(y_test)
        try:
            for idx_train, idx_test in kf.split(
                y_cv.to_frame(),
                y_cv,
                g_test,
                multitarget_y=(
                    y_test
                    if self.is_classifier
                    and isinstance(y_test, DataFrame)
                    and y_test.shape[1] > 1
                    else None
                ),
            )[0]:
                df_train = X_test.iloc[idx_train]
                df_test = X_test.iloc[idx_test]
                targ_train = y_test.iloc[idx_train]
                targ_test = y_test.iloc[idx_test]
                self.refit_tuned(X=df_train, y=targ_train, tuned_args=self.tuned_args)
                inner_preds_test = self.tuned_predict(X=df_test)
                if self.is_classifier:
                    inner_probs_test = self.predict_proba(X=df_test)
                    fold_score = self._score_outputs(
                        y_true=targ_test,
                        y_pred=inner_preds_test,
                        y_prob=inner_probs_test,
                    )
                    fold_score_by_target = self._score_outputs_by_target(
                        y_true=targ_test,
                        y_pred=inner_preds_test,
                        y_prob=inner_probs_test,
                    )
                else:
                    fold_score = self._score_outputs(
                        y_true=targ_test,
                        y_pred=inner_preds_test,
                        y_prob=None,
                    )
                    fold_score_by_target = self._score_outputs_by_target(
                        y_true=targ_test,
                        y_pred=inner_preds_test,
                        y_prob=None,
                    )
                scores.append(fold_score)
                for target_name, target_scores in fold_score_by_target.items():
                    fold_scores_by_target.setdefault(target_name, []).append(
                        target_scores
                    )
                self.tuned_model = None
                self._cleanup_after_fold()
        finally:
            self.tuned_model = tuned_model_orig
            self._cleanup_after_fold()

        holdout = Series(holdout_scores, name="holdout")
        train = Series(train_scores, name="trainset")
        scores = pd.concat([Series(score) for score in scores], axis=1)
        means = scores.mean(axis=1)
        means.name = "5-fold"
        df = pd.concat([train, holdout, means], axis=1)
        df.index.name = "metric"
        df = df.reset_index()
        df["final_cv_folds"] = final_cv_folds
        df_target: Optional[DataFrame] = None
        if isinstance(y_test, DataFrame):
            rows = []
            target_names = [str(col) for col in y_test.columns]
            for target_name in target_names:
                tr_scores = train_scores_by_target.get(target_name, {})
                ho_scores = holdout_scores_by_target.get(target_name, {})
                fold_target_scores = fold_scores_by_target.get(target_name, [])
                fold_means: dict[str, float] = {}
                if len(fold_target_scores) > 0:
                    fold_means = (
                        DataFrame(fold_target_scores)
                        .mean(axis=0, numeric_only=True)
                        .to_dict()
                    )
                metric_names = sorted(
                    set(tr_scores.keys()) | set(ho_scores.keys()) | set(fold_means.keys())
                )
                for metric_name in metric_names:
                    rows.append(
                        {
                            "target": target_name,
                            "metric": metric_name,
                            "trainset": tr_scores.get(metric_name, np.nan),
                            "holdout": ho_scores.get(metric_name, np.nan),
                            "5-fold": fold_means.get(metric_name, np.nan),
                            "final_cv_folds": final_cv_folds,
                        }
                    )
            if len(rows) > 0:
                df_target = DataFrame(rows)

        return (
            df,
            preds_train,
            holdout_preds_test,
            probs_train,
            holdout_probs_test,
            df_target,
        )

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")

        return self.tuned_model.score(X, y)

    def cv_score(
        self,
        X: DataFrame,
        y: Series,
        groups: Optional[Series],
        metric: Scorer,
        test: bool = False,
    ) -> float:
        # NOTE: VERY IMPORTANT: This must remain single-threaded! As it is
        # used in stepwise selection in the parallel loop

        # TODO: Make sure to make selection use the desired tuning metric?

        if test:
            score = np.random.beta(a=2, b=5)
            if self.is_classifier:
                score = 1 - score
            return score

        kf = OmniKFold(
            n_splits=5,
            is_classification=self.is_classifier,
            grouped=groups is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            df_analyze_phase="Tuning CV Score",
        )

        g = groups.copy() if groups is not None else None
        scores = []

        for idx_train, idx_test in kf.split(y.to_frame(), y.copy(), g)[0]:
            X_train = X.loc[idx_train].copy()
            X_test = X.loc[idx_test].copy()
            y_train = y.loc[idx_train].copy()
            y_test = y.loc[idx_test].copy()
            self.fit(X_train, y_train)
            preds = self.predict(X=X_test)
            self.model = None  # reset for next fit call
            self._cleanup_after_fold()
            if metric is None:
                scorer = acc if self.is_classifier else mae
                score = scorer(preds, y_test)
                score = score if self.is_classifier else -score
            else:
                score = metric.tuning_score(y_test, preds)
                if not metric.higher_is_better():
                    score = -score
            scores.append(score)
        return float(np.mean(scores))

    def tuned_cv_scores(self, X: DataFrame, y: Series) -> Series:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")

        return self.tuned_model.score(X, y)

    def predict(self, X: DataFrame) -> Union[Series, ndarray]:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self.model.predict(X)

    def tuned_predict(self, X: DataFrame) -> Union[Series, ndarray]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return self.tuned_model.predict(X)

    def wrapper_select(
        self,
        X_train: DataFrame,
        y_train: Series,
        n_feat: Optional[Union[float, int]] = None,
        method: WrapperSelection = WrapperSelection.StepUp,
    ) -> Optional[tuple[list[str], dict[str, float]]]:
        n_select = get_n_select(X_train, n_feat=n_feat)
        if n_select is None:
            return None

        raise NotImplementedError()

    def predict_proba(
        self, X: DataFrame
    ) -> Union[ndarray, dict[str, ndarray], list[ndarray], tuple[ndarray, ...]]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")

        if self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")

        return self.tuned_model.predict_proba(X)

    @overload
    @staticmethod
    def random(ds: TestDataset, n: Literal[1]) -> Type[DfAnalyzeModel]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    @staticmethod
    def random(ds: TestDataset, n: int) -> Sequence[Type[DfAnalyzeModel]]: ...

    @staticmethod
    def random(
        ds: TestDataset, n: Union[int, Literal[1]] = 1
    ) -> Union[Type[DfAnalyzeModel], Sequence[Type[DfAnalyzeModel]]]:
        from df_analyze.models.dummy import DummyClassifier, DummyRegressor
        from df_analyze.models.knn import KNNClassifier, KNNRegressor
        from df_analyze.models.lgbm import (
            LightGBMClassifier,
            LightGBMRegressor,
            LightGBMRFClassifier,
            LightGBMRFRegressor,
        )
        from df_analyze.models.linear import (
            ElasticNetRegressor,
            LRClassifier,
            SGDClassifier,
            SGDRegressor,
        )
        from df_analyze.models.mlp import MLPEstimator

        cls_choices = [
            DummyClassifier,
            KNNClassifier,
            LightGBMClassifier,
            LightGBMRFClassifier,
            SGDClassifier,
            LRClassifier,
            MLPEstimator,
        ]
        reg_choices = [
            DummyRegressor,
            KNNRegressor,
            LightGBMRegressor,
            LightGBMRFRegressor,
            SGDRegressor,
            ElasticNetRegressor,
            MLPEstimator,
        ]
        choices = cls_choices if ds.is_classification else reg_choices
        if n <= 1:
            return choice(choices)
        return np.random.choice(choices, replace=False, size=n).tolist()


def get_n_select(
    X_train: DataFrame, n_feat: Optional[Union[float, int]] = None
) -> Optional[int]:
    n_features = X_train.shape[1]
    msg = (
        "`n_feat` must be either None, an integer in [1, n_features - 1] "
        f"(i.e. an integer in [1, {n_features - 1}] for the current data) "
        "representing the number of features, or a float in (0, 1] "
        f"representing a percentage of features to select. Got: {n_feat}"
    )
    if n_feat is None:
        n_select = min(20, n_features - 1)
    elif isinstance(n_feat, Integral):
        if not 0 < n_feat < n_features:
            raise ValueError(msg)
        n_select = n_feat
    elif isinstance(n_feat, Real):
        if not 0 < n_feat <= 1:
            raise ValueError(msg)
        n_select = ceil(n_features * n_feat)
    else:
        raise ValueError(msg)

    if n_select >= n_features:  # from ceil or None case
        return None

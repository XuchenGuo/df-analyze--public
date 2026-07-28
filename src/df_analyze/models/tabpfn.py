from __future__ import annotations

# Reference: https://docs.priorlabs.ai
import os
from math import ceil, prod
from pathlib import Path
from tempfile import gettempdir
from time import time_ns
from typing import Any, Callable, Mapping, Optional, Type, Union
from warnings import warn

import numpy as np
import optuna
import pandas as pd
import torch  # noqa: F401
from optuna import Study, Trial, create_study
from optuna.samplers import GridSampler
from pandas import DataFrame, Series

from df_analyze._constants import SEED
from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import (
    RuntimeComponent,
    cleanup_torch_accelerator,
    configure_torch_cuda,
)
from df_analyze.splitting import OmniKFold

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault(
    "TABPFN_STATE_DIR", os.path.join(gettempdir(), "df-analyze-tabpfn-state")
)

try:
    from tabpfn import TabPFNClassifier as OfficialTabPFNClassifier
    from tabpfn import TabPFNRegressor as OfficialTabPFNRegressor
    from tabpfn.constants import ModelVersion
except ImportError as exc:
    OfficialTabPFNClassifier = None
    OfficialTabPFNRegressor = None
    ModelVersion = None
    _TABPFN_IMPORT_ERROR = exc
else:
    _TABPFN_IMPORT_ERROR = None
    try:
        from tabpfn.model_loading import get_cache_dir as _get_tabpfn_cache_dir
        from tabpfn.settings import settings as _tabpfn_settings
    except ImportError:
        _get_tabpfn_cache_dir = None
        _tabpfn_settings = None

if _TABPFN_IMPORT_ERROR is not None:
    _get_tabpfn_cache_dir = None
    _tabpfn_settings = None

TABPFN_MAX_FEATURES_PER_ESTIMATOR = 200
TABPFN_DEFAULT_N_ESTIMATORS = 8
TABPFN_HIGH_DIM_N_ESTIMATORS = 16
# The TabPFN-3 model card uses this as the feature-count boundary beyond which
# predictive performance is not guaranteed. The current TabPFN 8.x package also
# documents the wider row/feature regimes below.
TABPFN_V3_MODEL_CARD_MAX_FEATURES = 2_000
TABPFN_V3_SUPPORTED_SHAPES = (
    (1_000_000, 200),
    (100_000, 2_000),
    (1_000, 20_000),
)
TABPFN_PRETRAINING_LIMITS: dict[str, dict[str, int | None]] = {
    "v3": {
        "samples": 1_000_000,
        "features": 20_000,
        "sample_feature_values": None,
        "classes": 160,
    },
    "v2.6": {
        "samples": 100_000,
        "features": 2_000,
        "sample_feature_values": None,
        "classes": 10,
    },
    "v2.5": {
        "samples": 50_000,
        "features": 2_000,
        "sample_feature_values": None,
        "classes": 10,
    },
}


class TabPFNSetupError(RuntimeError):
    pass


def _setup_message(model_name: str, error: BaseException) -> str:
    return (
        f"Could not load {model_name}. On first use, sign in at "
        "https://ux.priorlabs.ai/account, accept the selected model license, "
        "and set TABPFN_TOKEN in the same terminal before running df-analyze. "
        "Review the current checkpoint terms before commercial or production use: "
        "https://huggingface.co/Prior-Labs/tabpfn_3. "
        f"Original error: {type(error).__name__}: {error}"
    )


def _assert_cache_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    probe = path / f".df-analyze-write-check-{os.getpid()}-{time_ns()}"
    with probe.open("xb"):
        pass
    probe.unlink()


def _cache_error(path: Path, error: BaseException) -> TabPFNSetupError:
    return TabPFNSetupError(
        f"TabPFN model cache directory is not writable: {path}. Set "
        "TABPFN_MODEL_CACHE_DIR to a writable directory before running df-analyze. "
        f"Original error: {type(error).__name__}: {error}"
    )


def _prepare_tabpfn_cache_dir() -> Optional[Path]:
    if _get_tabpfn_cache_dir is None:
        return None

    configured = os.getenv("TABPFN_MODEL_CACHE_DIR", "").strip()
    cache_dir = Path(configured).expanduser() if configured else _get_tabpfn_cache_dir()
    try:
        _assert_cache_writable(cache_dir)
    except OSError as exc:
        if configured:
            raise _cache_error(cache_dir, exc) from exc

        fallback = Path(os.environ["TABPFN_STATE_DIR"]) / "model-cache"
        try:
            _assert_cache_writable(fallback)
        except OSError as fallback_exc:
            raise _cache_error(fallback, fallback_exc) from fallback_exc

        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(fallback)
        if _tabpfn_settings is not None:
            _tabpfn_settings.tabpfn.model_cache_dir = fallback
        warn(
            f"TabPFN's default model cache is not writable ({cache_dir}). "
            f"Using the writable fallback {fallback}.",
            stacklevel=2,
        )
        return fallback
    return cache_dir


class TabPFNEstimator(DfAnalyzeModel):
    version = "v3"
    shortname = "tabpfn-v3"
    longname = "TabPFN v3 Estimator"
    timeout_s = 60 * 60
    tuning_cv_folds = 3

    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.official_cls: Type[Any] = type(None)
        self.target_cols: list[str] = []
        self.target_classes: dict[str, np.ndarray] = {}
        self._preflight_done = False
        self._preflight_config: Any = None
        self.default_args = dict(
            n_estimators=TABPFN_DEFAULT_N_ESTIMATORS,
            auto_scale_n_estimators=True,
            random_state=SEED,
            n_preprocessing_jobs=1,
            fit_mode="fit_preprocessors",
            memory_saving_mode="auto",
            inference_precision="auto",
            ignore_pretraining_limits=False,
            show_progress_bar=False,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model"] = None
        state["tuned_model"] = None
        state["_preflight_done"] = False
        state["_preflight_config"] = None
        return state

    @staticmethod
    def _assert_available() -> None:
        if _TABPFN_IMPORT_ERROR is not None:
            raise ImportError(
                "TabPFN is not installed. Install it with `pip install tabpfn`."
            ) from _TABPFN_IMPORT_ERROR

    def _configure_runtime(self) -> None:
        configure_torch_cuda(self.runtime, RuntimeComponent.TabPFN)

    def _device(self) -> str:
        return self.runtime.device_for(RuntimeComponent.TabPFN)

    def _cleanup_after_fold(self) -> None:
        cleanup_torch_accelerator(self.runtime, RuntimeComponent.TabPFN)

    @staticmethod
    def _move_model(model: Any, device: str) -> None:
        move = getattr(model, "to", None)
        if callable(move):
            move(device)

    def _run_model(self, model: Any, operation: Callable[[], Any]) -> Any:
        if self._device() == "cpu":
            return operation()
        self._move_model(model, self._device())
        try:
            return operation()
        finally:
            self._move_model(model, "cpu")
            self._cleanup_after_fold()

    @staticmethod
    def _categorical_indices(X: DataFrame) -> list[int]:
        indices = []
        for idx, dtype in enumerate(X.dtypes):
            if str(dtype) in {"object", "string", "bool", "boolean"} or isinstance(
                dtype, pd.CategoricalDtype
            ):
                indices.append(idx)
        return indices

    def _estimator_args(
        self, args: Mapping[str, Any], X: Optional[DataFrame] = None
    ) -> dict[str, Any]:
        out = dict(args)
        out["device"] = self._device()
        if X is not None:
            out["categorical_features_indices"] = self._categorical_indices(X)
        return out

    def _create_estimator(
        self, args: Mapping[str, Any], X: Optional[DataFrame] = None
    ) -> Any:
        self._assert_available()
        if ModelVersion is None:
            raise RuntimeError("TabPFN ModelVersion is unavailable.")
        version = ModelVersion(self.version)
        kwargs = self._estimator_args(args, X)
        return self.official_cls.create_default_for_version(version, **kwargs)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.official_cls, self._estimator_args(full_args)

    def _set_targets(self, y: Union[Series, DataFrame]) -> None:
        y_df = y.to_frame() if isinstance(y, Series) else y
        self.target_cols = [str(col) for col in y_df.columns]
        if not self.is_classifier:
            return
        for col in y_df.columns:
            key = str(col)
            previous = self.target_classes.get(key, np.asarray([])).tolist()
            current = y_df[col].dropna().unique().tolist()
            self.target_classes[key] = np.asarray(sorted(set([*previous, *current])))

    def _validate_limits(self, X: DataFrame, y: Union[Series, DataFrame]) -> None:
        n_samples, n_features = X.shape
        limits = TABPFN_PRETRAINING_LIMITS[self.version]
        max_samples = int(limits["samples"])
        max_features = int(limits["features"])
        max_values = limits["sample_feature_values"]
        max_classes = int(limits["classes"])
        if self.version == "v3":
            exceeds = not any(
                n_samples <= row_limit and n_features <= feature_limit
                for row_limit, feature_limit in TABPFN_V3_SUPPORTED_SHAPES
            )
        else:
            exceeds = n_samples > max_samples or n_features > max_features
            if max_values is not None:
                exceeds = exceeds or n_samples * n_features > int(max_values)
        if exceeds:
            if self.version == "v3":
                regimes = ", ".join(
                    f"{rows:,} samples x {features:,} features"
                    for rows, features in TABPFN_V3_SUPPORTED_SHAPES
                )
                raise ValueError(
                    f"{self.longname} input shape {X.shape} exceeds its documented "
                    f"row/feature regimes ({regimes})."
                )
            raise ValueError(
                f"{self.longname} input shape {X.shape} exceeds its supported "
                f"pretraining envelope (samples <= {max_samples:,}, features <= "
                f"{max_features:,}"
                + (
                    f", sample-feature values <= {int(max_values):,}"
                    if max_values is not None
                    else ""
                )
                + ")."
            )
        if (
            self.version == "v3"
            and n_features > TABPFN_V3_MODEL_CARD_MAX_FEATURES
        ):
            warn(
                f"{self.longname} received {n_features:,} features. This is within "
                "a wider TabPFN 8.x row/feature regime enforced by df-analyze, "
                f"but exceeds the TabPFN-3 model card's {TABPFN_V3_MODEL_CARD_MAX_FEATURES:,}-"
                "feature performance-guarantee boundary. Treat this wide-feature "
                "regime as experimental and validate it against non-TabPFN baselines.",
                stacklevel=2,
            )
        if self.is_classifier:
            y_df = y.to_frame() if isinstance(y, Series) else y
            n_classes = max(int(y_df[col].nunique()) for col in y_df.columns)
            if n_classes > max_classes:
                raise ValueError(
                    f"{self.longname} supports at most {max_classes} classes, "
                    f"but received {n_classes}."
                )

    @staticmethod
    def _config_limit(config: Any, name: str) -> Optional[int]:
        value = (
            config.get(name)
            if isinstance(config, Mapping)
            else getattr(config, name, None)
        )
        return None if value is None else int(value)

    def _validate_checkpoint_limits(
        self, X: DataFrame, y: Union[Series, DataFrame], config: Any
    ) -> None:
        limits = {"samples": self._config_limit(config, "MAX_NUMBER_OF_SAMPLES")}
        # TabPFN v3 supports high-dimensional input through feature subsampling
        # across estimators. MAX_NUMBER_OF_FEATURES describes an individual
        # checkpoint/estimator and must not replace the wrapper-level joint
        # sample-feature envelope checked by ``_validate_limits``.
        if self.version != "v3":
            limits["features"] = self._config_limit(
                config, "MAX_NUMBER_OF_FEATURES"
            )
        for label, limit in limits.items():
            actual = len(X) if label == "samples" else X.shape[1]
            if limit is not None and actual > limit:
                raise ValueError(
                    f"{self.longname} checkpoint supports at most {limit} {label}, "
                    f"but received {actual}."
                )
        if self.is_classifier:
            max_classes = self._config_limit(config, "MAX_NUMBER_OF_CLASSES")
            if max_classes is not None:
                y_df = y.to_frame() if isinstance(y, Series) else y
                actual = max(int(y_df[col].nunique()) for col in y_df.columns)
                if actual > max_classes:
                    raise ValueError(
                        f"{self.longname} checkpoint supports at most {max_classes} "
                        f"classes, but received {actual}."
                    )

    def preflight(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        extra_args: Optional[Mapping[str, Any]] = None,
    ) -> None:
        args = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **dict(extra_args or {}),
        }
        ignore_limits = bool(args.get("ignore_pretraining_limits", False))
        if not ignore_limits:
            self._validate_limits(X, y)
        allow_large_cpu = os.getenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if self._device() == "cpu" and not (ignore_limits or allow_large_cpu):
            if len(X) > 1_000:
                raise RuntimeError(
                    f"{self.longname} cannot run on CPU with {len(X)} rows by default. "
                    "Use CUDA or set TABPFN_ALLOW_CPU_LARGE_DATASET=1."
                )
            if len(X) > 200:
                warn(
                    f"{self.longname} is running on CPU with {len(X)} rows and may be slow."
                )

        config = self._preflight_config
        if not self._preflight_done:
            _prepare_tabpfn_cache_dir()
            model = None
            try:
                model = self._create_estimator(args, X)
                get_config = getattr(model, "get_inference_config", None)
                config = get_config() if callable(get_config) else None
            except Exception as exc:
                raise TabPFNSetupError(_setup_message(self.longname, exc)) from exc
            finally:
                del model
                self._cleanup_after_fold()
            self._preflight_config = config
            self._preflight_done = True
        if config is not None and not ignore_limits:
            self._validate_checkpoint_limits(X, y, config)

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int | bool]:
        args: dict[str, str | float | int | bool] = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators",
                [TABPFN_DEFAULT_N_ESTIMATORS, TABPFN_HIGH_DIM_N_ESTIMATORS],
            )
        }
        if self.is_classifier:
            args["balance_probabilities"] = trial.suggest_categorical(
                "balance_probabilities", [False, True]
            )
            args["average_before_softmax"] = trial.suggest_categorical(
                "average_before_softmax", [False, True]
            )
        return args

    @staticmethod
    def _n_estimators_grid(n_features: int) -> list[int]:
        minimum = max(1, ceil(max(1, n_features) / TABPFN_MAX_FEATURES_PER_ESTIMATOR))
        low = max(TABPFN_DEFAULT_N_ESTIMATORS, minimum)
        high = (
            TABPFN_HIGH_DIM_N_ESTIMATORS
            if low < TABPFN_HIGH_DIM_N_ESTIMATORS
            else low + 4
        )
        return sorted(set([low, high]))

    def _fit_one(self, X: DataFrame, y: Series, args: Mapping[str, Any]) -> Any:
        model = self._create_estimator(args, X)
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
        models = {}
        for col in y.columns:
            model = self._fit_one(X, y[col], args)
            if self._device() == "cuda":
                self._move_model(model, "cpu")
                self._cleanup_after_fold()
            models[str(col)] = model
        return models

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._set_targets(y_train)
        self.preflight(X_train, y_train)
        args = {**self.fixed_args, **self.default_args, **self.model_args}
        self.model = self._fit_models(X_train, y_train, args)

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._set_targets(y)
        tuned_args = dict(tuned_args or {})
        self.preflight(X, y, tuned_args)
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
            values = {}
            for target in self.target_cols:
                target_model = model[target]
                values[target] = self._run_model(
                    target_model,
                    lambda target_model=target_model: np.asarray(
                        target_model.predict(X)
                    ).reshape(-1),
                )
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

    def _aligned_proba(self, model: Any, X: DataFrame, target: str) -> np.ndarray:
        probs = np.asarray(model.predict_proba(X))
        expected = self.target_classes.get(target)
        classes = np.asarray(getattr(model, "classes_", []))
        if expected is None or classes.size == 0 or np.array_equal(classes, expected):
            return probs
        aligned = np.zeros((len(X), len(expected)), dtype=float)
        positions = {value: idx for idx, value in enumerate(expected.tolist())}
        for source_idx, value in enumerate(classes.tolist()):
            if value in positions:
                aligned[:, positions[value]] = probs[:, source_idx]
        return aligned

    def _predict_proba_with(
        self, model: Union[Any, dict[str, Any]], X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if isinstance(model, dict):
            probabilities = {}
            for target in self.target_cols:
                target_model = model[target]
                probabilities[target] = self._run_model(
                    target_model,
                    lambda target_model=target_model, target=target: self._aligned_proba(
                        target_model, X, target
                    ),
                )
            return probabilities
        target = self.target_cols[0] if self.target_cols else "target"
        return self._aligned_proba(model, X, target)

    def predict_proba_untuned(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.model is None:
            raise RuntimeError("Need to fit estimator before calling probabilities.")
        return self._predict_proba_with(self.model, X)

    def predict_proba(self, X: DataFrame) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")
        return self._predict_proba_with(self.tuned_model, X)

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")
        if not isinstance(self.tuned_model, dict):
            return float(self.tuned_model.score(X, y))
        if not isinstance(y, DataFrame):
            raise ValueError("Expected DataFrame targets for a multi-target model.")
        scores = []
        for col in y.columns:
            model = self.tuned_model[str(col)]
            scores.append(
                float(
                    self._run_model(
                        model, lambda model=model, col=col: model.score(X, y[col])
                    )
                )
            )
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
        self._set_targets(y_train)
        y_df = y_train.to_frame() if isinstance(y_train, Series) else y_train
        y_split = self._split_target_for_cv(y_df)
        splitter = OmniKFold(
            n_splits=n_folds,
            is_classification=self.is_classifier,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            df_analyze_phase="TabPFN tuning internal splits",
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
                model = None
                try:
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
                finally:
                    model = None
                    self._cleanup_after_fold()
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
        if self.tuned_args is not None:
            raise RuntimeError(
                f"Model {self.__class__.__name__} has already been tuned with Optuna"
            )
        self.preflight(X_train, y_train)
        estimator_grid = self._n_estimators_grid(X_train.shape[1])
        grid: dict[str, list[Any]] = {"n_estimators": estimator_grid}
        first_trial: dict[str, Any] = {"n_estimators": estimator_grid[0]}
        if self.is_classifier:
            grid.update(
                balance_probabilities=[False, True],
                average_before_softmax=[False, True],
            )
            first_trial.update(
                balance_probabilities=False,
                average_before_softmax=False,
            )
        self.grid = grid
        n_trials = min(n_trials, prod(len(values) for values in grid.values()))
        direction = "maximize" if metric.higher_is_better() else "minimize"
        study = create_study(
            direction=direction,
            sampler=GridSampler(grid, seed=SEED),
        )
        study.enqueue_trial(first_trial)
        optuna.logging.set_verbosity(verbosity)
        study.optimize(
            self.optuna_objective(X_train, y_train, g_train, metric),
            n_trials=n_trials,
            timeout=self.timeout_s,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        self._capture_best_target_scores(study)
        self.tuned_args = study.best_params
        self.refit_tuned(X_train, y_train, tuned_args=self.tuned_args)
        return study


class TabPFNClassifier(TabPFNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = True
        self.official_cls = OfficialTabPFNClassifier  # type: ignore


class TabPFNRegressor(TabPFNEstimator):
    def __init__(self, model_args: Optional[Mapping] = None) -> None:
        super().__init__(model_args)
        self.is_classifier = False
        self.official_cls = OfficialTabPFNRegressor  # type: ignore


class TabPFNClassifierV3(TabPFNClassifier):
    version = "v3"
    shortname = "tabpfn-v3"
    longname = "TabPFN v3 Classifier"


class TabPFNClassifierV26(TabPFNClassifier):
    version = "v2.6"
    shortname = "tabpfn-v2_6"
    longname = "TabPFN v2.6 Classifier"


class TabPFNClassifierV25(TabPFNClassifier):
    version = "v2.5"
    shortname = "tabpfn-v2_5"
    longname = "TabPFN v2.5 Classifier"


class TabPFNRegressorV3(TabPFNRegressor):
    version = "v3"
    shortname = "tabpfn-v3"
    longname = "TabPFN v3 Regressor"


class TabPFNRegressorV26(TabPFNRegressor):
    version = "v2.6"
    shortname = "tabpfn-v2_6"
    longname = "TabPFN v2.6 Regressor"


class TabPFNRegressorV25(TabPFNRegressor):
    version = "v2.5"
    shortname = "tabpfn-v2_5"
    longname = "TabPFN v2.5 Regressor"


TABPFN_CLASSIFIERS = {
    "v3": TabPFNClassifierV3,
    "v2_6": TabPFNClassifierV26,
    "v2_5": TabPFNClassifierV25,
}
TABPFN_REGRESSORS = {
    "v3": TabPFNRegressorV3,
    "v2_6": TabPFNRegressorV26,
    "v2_5": TabPFNRegressorV25,
}

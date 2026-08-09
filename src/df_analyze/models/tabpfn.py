from __future__ import annotations

# Reference: https://docs.priorlabs.ai
import os
from math import ceil, prod
from pathlib import Path
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

try:
    from tabpfn import TabPFNClassifier as OfficialTabPFNClassifier
    from tabpfn import TabPFNRegressor as OfficialTabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import (
        load_fitted_tabpfn_model as load_official_fitted_model,
    )
    from tabpfn.model_loading import (
        save_fitted_tabpfn_model as save_official_fitted_model,
    )
except ImportError as exc:
    OfficialTabPFNClassifier = None
    OfficialTabPFNRegressor = None
    load_official_fitted_model = None
    save_official_fitted_model = None
    ModelVersion = None
    _TABPFN_IMPORT_ERROR = exc
else:
    _TABPFN_IMPORT_ERROR = None
TABPFN_MAX_FEATURES_PER_ESTIMATOR = 200
TABPFN_DEFAULT_N_ESTIMATORS = 8
TABPFN_HIGH_DIM_N_ESTIMATORS = 16
# The model card warns that performance is not guaranteed past this feature
# count, although TabPFN also documents the wider input shapes below.
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
}


class TabPFNSetupError(RuntimeError):
    pass


def _target_series(frame: DataFrame, name: object) -> Series:
    target = frame.loc[:, name]
    if not isinstance(target, Series):
        raise ValueError(f"Expected one target column named {name!r}.")
    return target


def _setup_message(model_name: str, error: BaseException) -> str:
    return (
        f"Could not load {model_name}. Accept the selected checkpoint license and "
        "authenticate with the Prior Labs browser flow or TABPFN_TOKEN. If the "
        "original error reports a gated Hugging Face repository, accept that "
        "repository's terms and use `hf auth login` or a read-only HF_TOKEN. "
        "For offline use, populate TABPFN_MODEL_CACHE_DIR with the required "
        "weights. Setup guide: "
        "https://docs.priorlabs.ai/how-to-access-gated-models. Never commit "
        "authentication tokens. Review the selected checkpoint's current terms "
        "before commercial or production use. "
        f"Original error: {type(error).__name__}: {error}"
    )


class TabPFNEstimator(DfAnalyzeModel):
    runtime_component = RuntimeComponent.TabPFN
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
        self._external_model_manifest: dict[str, Any] | None = None
        self._external_model_directory: str | None = None
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
    def _save_model_group(
        models: Any, directory: Path, prefix: str
    ) -> dict[str, Any] | None:
        if models is None:
            return None
        if save_official_fitted_model is None:
            raise RuntimeError("TabPFN fitted-model saving is unavailable.")
        items = list(models.items()) if isinstance(models, dict) else [(None, models)]
        saved = []
        for index, (target, model) in enumerate(items):
            filename = f"{prefix}_{index}.tabpfn_fit"
            save_official_fitted_model(model, directory / filename)
            saved.append({"target": target, "filename": filename})
        return {
            "kind": "multi" if isinstance(models, dict) else "single",
            "models": saved,
        }

    def externalize_fitted_models(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        originals = {"model": self.model, "tuned_model": self.tuned_model}
        self._external_model_manifest = {
            "model": self._save_model_group(self.model, directory, "model"),
            "tuned_model": self._save_model_group(
                self.tuned_model, directory, "tuned_model"
            ),
        }
        self._external_model_directory = directory.name
        self.model = None
        self.tuned_model = None
        return originals

    def restore_externalized_models(self, originals: Mapping[str, Any]) -> None:
        self.model = originals["model"]
        self.tuned_model = originals["tuned_model"]

    def _load_model_group(self, directory: Path, state: Mapping[str, Any] | None) -> Any:
        if state is None:
            return None
        self._assert_available()
        if load_official_fitted_model is None:
            raise RuntimeError("TabPFN fitted-model loading is unavailable.")

        loaded = [
            (
                item["target"],
                load_official_fitted_model(
                    directory / item["filename"], device=self._device()
                ),
            )
            for item in state["models"]
        ]
        if state["kind"] == "multi":
            return {str(target): model for target, model in loaded}
        return loaded[0][1]

    def load_externalized_models(self, root: Path) -> None:
        if (
            self._external_model_manifest is None
            or self._external_model_directory is None
        ):
            return
        directory = root / self._external_model_directory
        self.model = self._load_model_group(
            directory, self._external_model_manifest["model"]
        )
        self.tuned_model = self._load_model_group(
            directory, self._external_model_manifest["tuned_model"]
        )

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
        max_classes = limits["classes"]
        assert max_classes is not None
        exceeds = not any(
            n_samples <= row_limit and n_features <= feature_limit
            for row_limit, feature_limit in TABPFN_V3_SUPPORTED_SHAPES
        )
        if exceeds:
            regimes = ", ".join(
                f"{rows:,} samples x {features:,} features"
                for rows, features in TABPFN_V3_SUPPORTED_SHAPES
            )
            raise ValueError(
                f"{self.longname} input shape {X.shape} exceeds its documented "
                f"row/feature regimes ({regimes})."
            )
        if n_features > TABPFN_V3_MODEL_CARD_MAX_FEATURES:
            warn(
                f"{self.longname} received {n_features:,} features. This is within "
                "a wider TabPFN 8.x row/feature regime enforced by df-analyze, "
                f"but exceeds the TabPFN-3 model card's {TABPFN_V3_MODEL_CARD_MAX_FEATURES:,}-"
                "feature guidance. Treat this wider regime as experimental and "
                "validate it against non-TabPFN baselines.",
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
                    f"{self.longname} is running on CPU with {len(X)} rows. "
                    "This workload is a candidate for CUDA acceleration."
                )

        config = self._preflight_config
        if not self._preflight_done:
            model = None
            try:
                model = self._create_estimator(args, X)
                get_config = getattr(model, "get_inference_config", None)
                config = get_config() if callable(get_config) else None
            except SystemExit as exc:
                # Some authentication/checkpoint-loading paths in the upstream
                # package terminate the interpreter instead of raising a normal
                # exception. A library model must not be allowed to kill the
                # df-analyze CLI before it can record an actionable failure.
                raise TabPFNSetupError(_setup_message(self.longname, exc)) from exc
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
        try:
            model = self._create_estimator(args, X)
            model.fit(X, y)
        except SystemExit as exc:
            raise TabPFNSetupError(_setup_message(self.longname, exc)) from exc
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
            return self._fit_one(X, _target_series(y, y.columns[0]), args)
        models = {}
        for col in y.columns:
            model = self._fit_one(X, _target_series(y, col), args)
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
                        model,
                        lambda model=model, col=col: model.score(
                            X, _target_series(y, col)
                        ),
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
            multitarget_y=(y_df if y_df.shape[1] > 1 else None),
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


class TabPFNRegressorV3(TabPFNRegressor):
    version = "v3"
    shortname = "tabpfn-v3"
    longname = "TabPFN v3 Regressor"

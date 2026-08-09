from __future__ import annotations

# Reference: https://github.com/KindXiaoming/pykan

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from base64 import b64decode, b64encode
from io import BytesIO
from math import ceil
from typing import Any, Callable, Mapping, Optional, Union, cast

import numpy as np
import optuna
import torch
from optuna import Study, Trial
from pandas import DataFrame, Series
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import LRScheduler
from torch import Tensor
from torch.nn import CrossEntropyLoss, HuberLoss, Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel, classification_output_dim
from df_analyze.models.mlp import BATCH_SIZE, MLPEstimator
from df_analyze.runtime.hardware import (
    RuntimeComponent,
    cleanup_torch_accelerator,
    get_runtime,
)
from df_analyze.splitting import OmniKFold

KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
KAN_WIDTH = 32
KAN_DEPTH = 2


def _target_series(frame: DataFrame, name: object) -> Series:
    target = frame.loc[:, name]
    if not isinstance(target, Series):
        raise ValueError(f"Expected one target column named {name!r}.")
    return target


def official_kan_cls() -> type[Any]:
    try:
        from kan import KAN  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as exc:
        raise ImportError(
            "KAN support requires `pykan`. Install it with `pip install pykan`."
        ) from exc
    return KAN


class SkorchKAN(Module):
    """Expose the official KAN module through the skorch estimator interface."""

    def __init__(
        self,
        input_dim: int,
        width: int = KAN_WIDTH,
        depth: int = KAN_DEPTH,
        grid_size: int = KAN_GRID_SIZE,
        spline_order: int = KAN_SPLINE_ORDER,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        grid_eps: float = 0.02,
        noise_scale: float = 0.3,
        base_fun: str = "silu",
        sparse_init: bool = False,
        seed: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("KAN input_dim must be positive.")

        self.num_classes = int(num_classes)
        self.is_classification = self.num_classes >= 2
        out_channels = self.num_classes
        dims = [int(input_dim), *([int(width)] * int(depth)), out_channels]
        kan_cls = official_kan_cls()
        self.kan: Any = kan_cls(
            width=dims,
            grid=int(grid_size),
            k=int(spline_order),
            noise_scale=float(noise_scale),
            base_fun=base_fun,
            symbolic_enabled=False,
            grid_eps=float(grid_eps),
            grid_range=[float(grid_range[0]), float(grid_range[1])],
            save_act=False,
            sparse_init=bool(sparse_init),
            auto_save=False,
            seed=int(seed),
            device="cpu",
        )

    def _raw_forward(self, x: Tensor) -> Tensor:
        x = x.to(dtype=torch.float32)
        input_id = getattr(self.kan, "input_id", None)
        if isinstance(input_id, Tensor):
            tensor_input_id = cast(Tensor, input_id)
            if tensor_input_id.device != x.device:
                self.kan.input_id = tensor_input_id.to(x.device)
        return self.kan(x)

    def forward(self, x: Tensor) -> Tensor:
        output = self._raw_forward(x)
        if self.is_classification:
            return output
        return torch.flatten(output)


class KANEstimator(MLPEstimator):
    runtime_component = RuntimeComponent.KAN
    shortname = "kan"
    longname = "Kolmogorov-Arnold Network"
    timeout_s = 3600

    def __init__(self, num_classes: int, model_args: Mapping | None = None) -> None:
        super().__init__(num_classes=num_classes, model_args=model_args)
        self.model_cls = NeuralNetClassifier if self.is_classifier else NeuralNetRegressor
        self.target_cols: list[str] = []
        self.fixed_args = dict(
            module=SkorchKAN,
            module__num_classes=num_classes,
            criterion=CrossEntropyLoss if self.is_classifier else HuberLoss,
            optimizer=AdamW,
            max_epochs=50,
            batch_size=BATCH_SIZE,
            iterator_train__drop_last=False,
            device=self.runtime.device_for(RuntimeComponent.KAN),
            verbose=0,
        )

    def _configure_runtime(self) -> None:
        self.fixed_args["device"] = self.runtime.device_for(RuntimeComponent.KAN)

    def _cleanup_after_fold(self) -> None:
        cleanup_torch_accelerator(self.runtime, RuntimeComponent.KAN)

    @staticmethod
    def _serialize_net(model: Any) -> dict[str, Any]:
        module = getattr(model, "module_", None)
        if module is None:
            raise RuntimeError("Cannot serialize an unfitted KAN estimator.")

        params = model.get_params(deep=False)
        module_args = {
            key: value for key, value in params.items() if key.startswith("module__")
        }
        state_dict = {
            key: value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in module.state_dict().items()
        }
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        return {
            "estimator": (
                "classifier" if isinstance(model, NeuralNetClassifier) else "regressor"
            ),
            "module_args": module_args,
            "classes": params.get("classes"),
            "state_dict": b64encode(buffer.getvalue()).decode("ascii"),
        }

    def _serialize_models(self, models: Any) -> Optional[dict[str, Any]]:
        if models is None:
            return None
        if isinstance(models, dict):
            return {
                "kind": "multi",
                "models": {
                    str(target): self._serialize_net(model)
                    for target, model in models.items()
                },
            }
        return {"kind": "single", "model": self._serialize_net(models)}

    def _restore_net(self, state: Mapping[str, Any]) -> Any:
        is_classifier = state["estimator"] == "classifier"
        estimator_cls = NeuralNetClassifier if is_classifier else NeuralNetRegressor
        criterion = CrossEntropyLoss if is_classifier else HuberLoss
        kwargs: dict[str, Any] = {
            "module": SkorchKAN,
            "criterion": criterion,
            "optimizer": AdamW,
            "max_epochs": 1,
            "batch_size": BATCH_SIZE,
            "iterator_train__drop_last": False,
            "device": self.runtime.device_for(RuntimeComponent.KAN),
            "verbose": 0,
            "train_split": None,
            "callbacks": [],
            **dict(state["module_args"]),
        }
        if state.get("classes") is not None:
            kwargs["classes"] = state["classes"]

        model = estimator_cls(**kwargs)
        model.initialize()
        buffer = BytesIO(b64decode(state["state_dict"]))
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        model.module_.load_state_dict(state_dict)
        return model

    def _restore_models(self, state: Optional[Mapping[str, Any]]) -> Any:
        if state is None:
            return None
        if state["kind"] == "multi":
            return {
                str(target): self._restore_net(model_state)
                for target, model_state in state["models"].items()
            }
        return self._restore_net(state["model"])

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
        for name, value in state.items():
            setattr(self, name, value)
        # Load saved models on CPU; callers can choose another device later.
        self.runtime = get_runtime("cpu")
        self._configure_runtime()
        self.model = self._restore_models(model_state)
        self.tuned_model = self._restore_models(tuned_model_state)

    @staticmethod
    def _scheduler_period(
        n_samples: int,
        n_epochs: int,
        has_validation: bool,
        batch_size: int = BATCH_SIZE,
    ) -> int:
        if has_validation:
            # Skorch's default ValidSplit(5) holds out one fifth. The scheduler
            # steps once per *training* batch, not once per validation batch.
            n_samples = n_samples - ceil(n_samples / 5)
        n_batches = max(1, ceil(max(1, n_samples) / max(1, int(batch_size))))
        return max(1, int(n_epochs) * n_batches)

    def _get_scheduler(
        self, X_train: DataFrame, restarts: bool, val_split: bool
    ) -> Union[CosineAnnealingWarmRestarts, CosineAnnealingLR]:
        policy = CosineAnnealingWarmRestarts if restarts else CosineAnnealingLR
        n_epochs = 8 if restarts else 50
        batch_size = int(self.model_args.get("batch_size", BATCH_SIZE))
        period = self._scheduler_period(
            len(X_train), n_epochs, val_split, batch_size=batch_size
        )
        shared: Mapping[str, Any] = dict(eta_min=0, step_every="step")
        if restarts:
            return cast(
                Union[CosineAnnealingWarmRestarts, CosineAnnealingLR],
                LRScheduler(
                    policy=cast(Any, policy),
                    T_0=period,
                    T_mult=2,
                    **shared,
                ),
            )
        return cast(
            Union[CosineAnnealingWarmRestarts, CosineAnnealingLR],
            LRScheduler(
                policy=cast(Any, policy),
                T_max=period,
                **shared,
            ),
        )

    def _set_input_dim(self, X: DataFrame) -> None:
        self.fixed_args["module__input_dim"] = int(X.shape[1])

    @staticmethod
    def _num_classes(y: Series, is_classifier: bool) -> int:
        return classification_output_dim(y) if is_classifier else 1

    def _set_targets(self, y: Union[Series, DataFrame]) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            self.target_cols = [str(y.name if y.name is not None else "target")]

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        bools = (True, False)
        return dict(
            module__width=trial.suggest_categorical("module__width", [8, 16, 32, 64]),
            module__depth=trial.suggest_int("module__depth", 1, 3),
            module__grid_size=trial.suggest_categorical("module__grid_size", [3, 5, 7]),
            module__spline_order=trial.suggest_categorical(
                "module__spline_order", [2, 3]
            ),
            module__grid_eps=trial.suggest_float("module__grid_eps", 0.0, 1.0),
            module__sparse_init=trial.suggest_categorical("module__sparse_init", bools),
            early_stopping=trial.suggest_categorical("early_stopping", bools),
            restarts=trial.suggest_categorical("restarts", bools),
            optimizer__lr=trial.suggest_float("optimizer__lr", 1e-5, 5e-2, log=True),
            optimizer__weight_decay=trial.suggest_float(
                "optimizer__weight_decay", 1e-8, 1e-2, log=True
            ),
        )

    def _to_model_args(self, optuna_args: Mapping, X_train: DataFrame) -> dict[str, Any]:
        args = super()._to_model_args(optuna_args, X_train)
        args["module__input_dim"] = int(X_train.shape[1])
        return args

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._set_input_dim(X_train)
        self._set_targets(y_train)
        if not isinstance(y_train, DataFrame) or y_train.shape[1] == 1:
            target = (
                _target_series(y_train, y_train.columns[0])
                if isinstance(y_train, DataFrame)
                else y_train
            )
            super().fit(X_train, target)
            return

        models = {}
        for col in y_train.columns:
            target = _target_series(y_train, col)
            child = type(self)(
                self._num_classes(target, self.is_classifier), self.model_args
            )
            child.set_runtime(self.runtime)
            child.fit(X_train, target)
            models[str(col)] = child.model
        self.model = models  # type: ignore

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._set_input_dim(X)
        self._set_targets(y)
        if not isinstance(y, DataFrame) or y.shape[1] == 1:
            target = _target_series(y, y.columns[0]) if isinstance(y, DataFrame) else y
            super().refit_tuned(X, target, g=g, tuned_args=tuned_args)
            return

        models = {}
        for col in y.columns:
            target = _target_series(y, col)
            child = type(self)(
                self._num_classes(target, self.is_classifier), self.model_args
            )
            child.set_runtime(self.runtime)
            child.refit_tuned(X, target, g=g, tuned_args=tuned_args)
            models[str(col)] = child.tuned_model
        self.tuned_args = dict(tuned_args or {})
        self.tuned_model = models  # type: ignore
        self.is_refit = True

    def _predict_with(
        self, model: Any, X: DataFrame, probabilities: bool = False
    ) -> Union[np.ndarray, Series, DataFrame, dict[str, np.ndarray]]:
        Xt = self._to_torch(X)
        if isinstance(model, dict):
            if probabilities:
                return {
                    target: np.asarray(model[target].predict_proba(Xt))
                    for target in self.target_cols
                }
            values = {
                target: np.asarray(model[target].predict(Xt)).reshape(-1)
                for target in self.target_cols
            }
            return DataFrame(values, index=X.index)
        if probabilities:
            return np.asarray(model.predict_proba(Xt))
        return np.asarray(model.predict(Xt))

    def predict(self, X: DataFrame) -> Union[np.ndarray, DataFrame]:
        if self.model is None:
            raise RuntimeError("Need to call `model.fit()` before calling `.predict()`")
        return self._predict_with(self.model, X)  # type: ignore

    def tuned_predict(self, X: DataFrame) -> Union[np.ndarray, DataFrame]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        return self._predict_with(self.tuned_model, X)  # type: ignore

    def predict_proba_untuned(
        self, X: DataFrame
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.model is None:
            raise RuntimeError("Need to fit estimator before calling probabilities.")
        return self._predict_with(self.model, X, probabilities=True)  # type: ignore

    def predict_proba(self, X: DataFrame) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if not self.is_classifier:
            raise ValueError("Cannot get probabilities for a regression model.")
        if self.tuned_model is None:
            raise RuntimeError("Need to tune estimator before calling `.predict_proba()`")
        return self._predict_with(self.tuned_model, X, probabilities=True)  # type: ignore

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if self.tuned_model is None:
            raise RuntimeError("Need to tune model before calling `.tuned_scores()`")

        if isinstance(self.tuned_model, dict):
            if not isinstance(y, DataFrame):
                raise ValueError("Expected DataFrame targets for a multi-target model.")
            Xt = self._to_torch(X)
            scores = []
            for col in y.columns:
                target = str(col)
                if target not in self.tuned_model:
                    raise ValueError(f"No tuned KAN model found for target {target!r}.")
                _, yt = self._to_torch(X, _target_series(y, col))
                scores.append(float(self.tuned_model[target].score(Xt, yt)))
            return float(np.mean(scores))

        if isinstance(y, DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Expected one target for a single-target KAN model.")
            y = _target_series(y, y.columns[0])
        Xt, yt = self._to_torch(X, y)
        return float(self.tuned_model.score(Xt, yt))

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: Optional[int] = None,
    ) -> Callable[[Trial], float]:
        n_folds = self.resolve_tuning_cv_folds(n_folds)
        self._set_input_dim(X_train)
        self._set_targets(y_train)
        if not isinstance(y_train, DataFrame):
            return super().optuna_objective(
                X_train, y_train, g_train, metric, n_folds=n_folds
            )

        y_split = self._split_target_for_cv(y_train)
        splitter = OmniKFold(
            n_splits=n_folds,
            is_classification=self.is_classifier,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            df_analyze_phase="KAN tuning internal splits",
        )
        splits = splitter.split(
            X_train,
            y_split,
            g_train,
            multitarget_y=(y_train),
        )[0]

        def objective(trial: Trial) -> float:
            optuna_args = self.optuna_args(trial)
            fold_scores = []
            scores_by_target: dict[str, list[float]] = {}
            for step, (idx_train, idx_test) in enumerate(splits):
                X_tr, X_test = X_train.iloc[idx_train], X_train.iloc[idx_test]
                target_scores = []
                for col in y_train.columns:
                    full_target = _target_series(y_train, col)
                    y_tr, y_test = (
                        full_target.iloc[idx_train],
                        full_target.iloc[idx_test],
                    )
                    child = type(self)(
                        self._num_classes(y_tr, self.is_classifier), self.model_args
                    )
                    child.set_runtime(self.runtime)
                    model_args = child._to_model_args(optuna_args, X_tr)
                    args = child._runtime_model_args(
                        {
                            **child.fixed_args,
                            **child.default_args,
                            **child.model_args,
                            **model_args,
                        }
                    )
                    estimator = None
                    try:
                        estimator = child.model_cls(**args)
                        Xt, yt = child._to_torch(X_tr, y_tr)
                        estimator.fit(Xt, yt)
                        preds = estimator.predict(child._to_torch(X_test))
                        score = float(metric.tuning_score(y_test, preds))
                        if y_train.shape[1] > 1:
                            score = self._scale_tuning_score(metric, full_target, score)
                        target_scores.append(score)
                        scores_by_target.setdefault(str(col), []).append(score)
                    finally:
                        estimator = None
                        child._cleanup_after_fold()
                fold_scores.append(float(np.mean(target_scores)))
                trial.report(float(np.mean(fold_scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            self._record_trial_target_scores(trial, scores_by_target)
            return float(np.mean(fold_scores))

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
        # Parallel trials multiply PyTorch memory and CPU threads, while pykan
        # also uses global random state. Run trials one at a time.
        n_jobs = self.runtime.tuning_jobs(RuntimeComponent.KAN, 1)
        return DfAnalyzeModel.htune_optuna(
            self,
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

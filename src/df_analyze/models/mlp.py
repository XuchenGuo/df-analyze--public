from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import traceback

import matplotlib as mpl

mpl.rcParams["axes.formatter.useoffset"] = False

import os
import platform
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Union,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from numpy import ndarray
from optuna import Study, Trial
from pandas import DataFrame, Series
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.callbacks.lr_scheduler import CosineAnnealingLR
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    CrossEntropyLoss,
    Dropout,
    HuberLoss,
    LazyLinear,
    LeakyReLU,
    Linear,
    Module,
    ModuleList,
    Sequential,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import RuntimeComponent, cleanup_torch_accelerator
from df_analyze.splitting import OmniKFold

"""
See:

Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka: “Well-tuned Simple
Nets Excel on Tabular Datasets”, 2021; [http://arxiv.org/abs/2106.11189
arXiv:2106.11189]. https://arxiv.org/pdf/2106.11189.pdf

Fixed Architecture and Optimization Hyperparameters

> In order to focus exclusively on investigating the effect of regularization we
> fix the neural architecture to a simple multilayer perceptron (MLP) and also
> fix some hyperparameters of the general training procedure. These fixed
> hyperparameter values, as specified in Table 4 of Appendix B.1, have been tuned
> for maximizing the performance of an unregularized neural network on our
> dataset collection (see Table 9 in Appendix D). We use a 9-layer feed-forward
> neural network with 512 units for each layer, a choice motivated by previous
> work [42].
>
> Moreover, we set a low learning rate of 10−3 after performing a grid search for
> finding the best value across datasets. We use AdamW [36], which implements
> decoupled weight decay, and cosine annealing with restarts [35] as a learning
> rate scheduler. Using a learning rate scheduler with restarts helps in our case
> because we keep a fixed initial learning rate. For the restarts, we use an
> initial budget of 15 epochs, with a budget multiplier of 2, following published
> practices [62]. Additionally, since our benchmark includes imbalanced datasets,
> we use a weighted version of categorical cross-entropy and balanced accuracy
> [4] as the evaluation metric.

Relevant Details from Table 4 of Fixed hparams:

Table 4: [Fixed Params]

Category                                 Value
......................................   .....
Cosine Annealing Iterations multiplier   2.0
Max. iterations                          15
Activation                               ReLU
Bias initialization                      Yes
Embeddings                               One-Hot encoding
Units in a layer                         512
Preprocessing Preprocessor               None
Batch size                               128
Imputation                               Median
Initialization method                    Default
Learning rate                            1e-3
Loss module                              Weighted Cross-Entropy
Normalization strategy                   Standardize
Optimizer                                AdamW
Scheduler                                Cosine-Annealing w Warm Restarts
Seed                                     11

Table 1 of The paper describes the regularization cocktails: we replicate just
a few, but also tune the LR (they did not find this to be meaningful, likely
given the large amount of regularization options used).

Batch-Norm                  {True, False}
Use Weight-Decay            {True, False}
Weight-Decay                [1e-5, 1e-1]
Use dropout                 {True, False}
Dropout rate                [0.0, 0.8]


"""

NETWORK_MAX_DEPTH = 8
NETWORK_MIN_DEPTH = 3
NETWORK_MAX_WIDTH = 512
NETWORK_MIN_WIDTH = 32
BATCH_SIZE = 128

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

# silence_spam()
warnings.filterwarnings("ignore", category=UserWarning, message="Lazy modules")


class SkorchMLP(Module):
    def __init__(
        self,
        width: int = NETWORK_MAX_WIDTH,
        depth: int = NETWORK_MAX_DEPTH,
        use_bn: bool = True,
        dropout: float = 0.4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hparams = dict(
            width=width,
            use_bn=use_bn,
            dropout=dropout,
        )
        self.num_classes = num_classes
        self.is_classification = num_classes >= 2
        self.out_channels = num_classes

        W = width
        D = dropout if dropout > 0.0 else 0.0
        self.input = LazyLinear(out_features=width)
        self.layers = ModuleList()
        for i in range(depth):
            self.layers.append(Linear(W, W, bias=not use_bn))
            if use_bn:
                self.layers.append(BatchNorm1d(W))
            if depth - 1:
                self.layers.append(Dropout(D))
            self.layers.append(LeakyReLU())

        self.backbone = Sequential(*self.layers)
        self.output = Linear(W, self.out_channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.backbone(x)
        x = self.output(x)
        if self.is_classification:
            return x
        return torch.flatten(x)


def get_T0(
    X: Union[ndarray, DataFrame],
    n_epochs: int,
    batch_size: int = BATCH_SIZE,
    val_split: Optional[Union[int, float]] = None,
) -> tuple[int, int]:
    N = len(X)
    if val_split is None:
        n_batches = N // batch_size
    elif isinstance(val_split, int):
        n_batches = (N - val_split) // batch_size
    elif isinstance(val_split, float) and (0.0 < val_split < 1.0):
        n_batches = int(N * val_split) // batch_size
    else:
        raise ValueError("Invalid `val_split` argument.")
    if n_batches == 0:
        return n_epochs, n_batches

    return n_epochs * n_batches, n_batches


class MLPEstimator(DfAnalyzeModel):
    shortname = "mlp"
    longname = "Multilayer Perceptron"
    timeout_s = 3600
    tuning_cv_folds = 3

    def __init__(self, num_classes: int, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = num_classes > 1
        # class NeuralNetClassifier: pass
        # class NeuralNetRegressor: pass
        self.model_cls = NeuralNetClassifier if self.is_classifier else NeuralNetRegressor
        self.model: Union[NeuralNetClassifier, NeuralNetRegressor]
        self.target_cols: list[str] = []
        self.fixed_args = dict(
            module=SkorchMLP,
            # https://github.com/skorch-dev/skorch/issues/477#issuecomment-493660800
            iterator_train__drop_last=True,
            module__num_classes=num_classes,
            criterion=CrossEntropyLoss if self.is_classifier else HuberLoss,
            optimizer=AdamW,
            max_epochs=50,
            batch_size=BATCH_SIZE,
            # iterator_train__num_workers=1,  # for some reason causes huge slow
            device=self.runtime.device_for(RuntimeComponent.MLP),
            verbose=0,
        )

    def _configure_runtime(self) -> None:
        self.fixed_args["device"] = self.runtime.device_for(RuntimeComponent.MLP)

    def _cleanup_after_fold(self) -> None:
        cleanup_torch_accelerator(self.runtime, RuntimeComponent.MLP)

    def _set_target_cols(self, y: Union[Series, DataFrame]) -> None:
        if isinstance(y, DataFrame):
            self.target_cols = [str(col) for col in y.columns]
        else:
            self.target_cols = [str(y.name) if y.name is not None else "target"]

    def _target_num_classes(self, y: Series) -> int:
        if not self.is_classifier:
            return 1
        values = np.asarray(y.dropna())
        try:
            encoded = values.astype(int)
            if np.array_equal(values, encoded) and encoded.min() >= 0:
                return max(2, int(encoded.max()) + 1)
        except (TypeError, ValueError):
            pass
        return max(2, int(np.unique(values).shape[0]))

    def _new_target_estimator(self, y: Series) -> MLPEstimator:
        model = type(self)(
            num_classes=self._target_num_classes(y), model_args=self.model_args
        )
        model.set_runtime(self.runtime)
        return model

    def _fit_target_estimators(
        self,
        X: DataFrame,
        y: DataFrame,
        tuned_args: Optional[Mapping] = None,
    ) -> dict[str, MLPEstimator]:
        models: dict[str, MLPEstimator] = {}
        for col in y.columns:
            target = str(col)
            model = self._new_target_estimator(y[col])
            if tuned_args is None:
                model.fit(X, y[col])
            else:
                model.refit_tuned(X, y[col], tuned_args=tuned_args)
            models[target] = model
        return models

    def _predict_target_estimators(
        self, models: dict[str, MLPEstimator], X: DataFrame, tuned: bool
    ) -> DataFrame:
        predictions = {}
        for target in self.target_cols:
            model = models[target]
            pred = model.tuned_predict(X) if tuned else model.predict(X)
            predictions[target] = np.asarray(pred).reshape(-1)
        return DataFrame(predictions, index=X.index, columns=self.target_cols)

    def _predict_target_proba(
        self, models: dict[str, MLPEstimator], X: DataFrame, tuned: bool
    ) -> dict[str, ndarray]:
        return {
            target: np.asarray(
                models[target].predict_proba(X)
                if tuned
                else models[target].predict_proba_untuned(X)
            )
            for target in self.target_cols
        }

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        widths = (16, 32, 64, 128, 256, 512)
        bools = (True, False)
        return dict(
            # SkorchMLP args
            module__width=trial.suggest_categorical("module__width", widths),
            module__depth=trial.suggest_int("module__depth", 3, 8),
            module__use_bn=trial.suggest_categorical("module__use_bn", bools),
            module__dropout=trial.suggest_float("module__dropout", 0.0, 0.7, step=0.1),
            # NeuralNet[Estimator] args
            early_stopping=trial.suggest_categorical("early_stopping", bools),
            restarts=trial.suggest_categorical("restarts", bools),
            optimizer__lr=trial.suggest_float("optimizer__lr", 1e-5, 1e-1, log=True),
            optimizer__weight_decay=trial.suggest_float(
                "optimizer__weight_decay", 1e-7, 1e-1, log=True
            ),
        )

    def _get_scheduler(
        self, X_train: DataFrame, restarts: bool, val_split: bool
    ) -> Union[CosineAnnealingWarmRestarts, CosineAnnealingLR]:
        cls = CosineAnnealingWarmRestarts if restarts else CosineAnnealingLR
        split = 0.2 if val_split else None
        n_epochs = 8 if restarts else 50  # "period" of the annealing
        period = get_T0(X_train, n_epochs=n_epochs, val_split=split)[0]
        shared: Mapping = dict(eta_min=0, step_every="step")
        return (
            LRScheduler(policy=cls, T_0=period, T_mult=2, **shared)  # type: ignore
            if restarts
            else LRScheduler(policy=cls, T_max=period, **shared)  # type: ignore
        )

    @overload
    def _to_torch(self, X: DataFrame) -> Tensor: ...

    @overload
    def _to_torch(self, X: DataFrame, y: Series) -> tuple[Tensor, Tensor]: ...

    def _to_torch(
        self, X: DataFrame, y: Optional[Series] = None
    ) -> Union[tuple[Tensor, Tensor], Tensor]:
        Xt = torch.from_numpy(X.to_numpy()).to(dtype=torch.float32)
        if y is None:
            return Xt

        if self.is_classifier:
            yt = torch.from_numpy(y.to_numpy()).to(dtype=torch.int64)
        else:
            yt = torch.from_numpy(y.to_numpy()).to(dtype=torch.float32)
        return Xt, yt

    def fit(self, X_train: DataFrame, y_train: Union[Series, DataFrame]) -> None:
        self._set_target_cols(y_train)
        if isinstance(y_train, DataFrame) and y_train.shape[1] > 1:
            self.model = self._fit_target_estimators(X_train, y_train)  # type: ignore[assignment]
            return
        if isinstance(y_train, DataFrame):
            y_train = y_train.iloc[:, 0]
        if self.model is None:
            kwargs = {**self.fixed_args, **self.default_args, **self.model_args}
            self.model = self.model_cls(**kwargs)
        X, y = self._to_torch(X_train, y_train)
        try:
            self.model.fit(X, y)
        except ValueError as e:
            raise RuntimeError(
                "Got exception when trying to fit MLP.\n"
                f"X={type(X)}, shape={X.shape}\n"
                f"y={type(y)}, shape={y.shape}\n"
                f"Additional details: {traceback.format_exc()}"
            ) from e

    def refit_tuned(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> None:
        self._set_target_cols(y)
        if isinstance(y, DataFrame) and y.shape[1] > 1:
            self.tuned_model = self._fit_target_estimators(
                X, y, tuned_args=tuned_args or {}
            )
            return
        if isinstance(y, DataFrame):
            y = y.iloc[:, 0]
        tuned_args = tuned_args or {}
        kwargs = {
            **self.fixed_args,
            **self.default_args,
            **self.model_args,
            **self._to_model_args(tuned_args, X),
        }
        self.tuned_model = self.model_cls(**kwargs)
        Xt, yt = self._to_torch(X, y)
        self.tuned_model.fit(Xt, yt)  # type: ignore

    def predict(self, X: DataFrame) -> Union[ndarray, DataFrame]:
        if isinstance(self.model, dict):
            return self._predict_target_estimators(self.model, X, tuned=False)
        Xt = self._to_torch(X)
        return self.model.predict(Xt)

    def tuned_predict(self, X: DataFrame) -> Union[ndarray, DataFrame]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.tuned_predict()`"
            )
        if isinstance(self.tuned_model, dict):
            return self._predict_target_estimators(self.tuned_model, X, tuned=True)
        Xt = self._to_torch(X)
        return self.tuned_model.predict(Xt)

    def predict_proba_untuned(self, X: DataFrame) -> Union[ndarray, dict[str, ndarray]]:
        if self.model is None:
            raise RuntimeError(
                "Need to call `model.fit()` before calling `.predict_proba_untuned()`"
            )
        if isinstance(self.model, dict):
            return self._predict_target_proba(self.model, X, tuned=False)
        Xt = self._to_torch(X)
        return self.model.predict_proba(Xt)

    def predict_proba(self, X: DataFrame) -> Union[ndarray, dict[str, ndarray]]:
        if self.tuned_model is None:
            raise RuntimeError(
                "Need to call `model.tune()` before calling `.predict_proba()`"
            )
        if isinstance(self.tuned_model, dict):
            return self._predict_target_proba(self.tuned_model, X, tuned=True)
        Xt = self._to_torch(X)
        return self.tuned_model.predict_proba(Xt)

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if isinstance(self.tuned_model, dict):
            if not isinstance(y, DataFrame):
                raise ValueError("Expected DataFrame targets for a multi-target model.")
            scores = [
                self.tuned_model[str(col)].tuned_scores(X, y[col]) for col in y.columns
            ]
            return float(np.mean(scores))
        return super().tuned_scores(X, y)

    def _to_model_args(self, optuna_args: Mapping, X_train: DataFrame) -> dict[str, Any]:
        final_args: Mapping = {**deepcopy(optuna_args)}
        restarts = final_args.pop("restarts")
        early = final_args.pop("early_stopping")

        callbacks: list[Any] = [EarlyStopping(patience=20)] if early else []
        sched = self._get_scheduler(X_train, restarts=restarts, val_split=early)
        callbacks.append(sched)

        final_args["callbacks"] = callbacks
        return final_args

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: Optional[int] = None,
    ) -> Callable[[Trial], float]:
        n_folds = self.resolve_tuning_cv_folds(n_folds)
        self._set_target_cols(y_train)
        if isinstance(y_train, DataFrame) and y_train.shape[1] > 1:
            y_split = self._split_target_for_cv(y_train)
            target_data = []
            for col in y_train.columns:
                target = y_train[col]
                model = self._new_target_estimator(target)
                X_target, y_target = model._to_torch(X_train, target)
                target_data.append((model, target, X_target, y_target))

            def objective(trial: Trial) -> float:
                kf = OmniKFold(
                    n_splits=n_folds,
                    is_classification=self.is_classifier,
                    grouped=g_train is not None,
                    labels=None,
                    warn_on_fallback=False,
                    allow_group_fallback=False,
                    df_analyze_phase="Tuning internal splits",
                )
                splits = kf.split(
                    X_train,
                    y_split,
                    g_train,
                    multitarget_y=(
                        y_train if self.is_classifier else None
                    ),
                )[0]
                opt_args = self.optuna_args(trial)
                target_args = []
                for model, target, X_target, y_target in target_data:
                    model_args = model._to_model_args(opt_args, X_train)
                    full_args = {
                        **model.fixed_args,
                        **model.default_args,
                        **model.model_args,
                        **model_args,
                    }
                    target_args.append(
                        (model, target, X_target, y_target, full_args)
                    )

                scores = []
                scores_by_target: dict[str, list[float]] = {}
                for step, (idx_train, idx_test) in enumerate(splits):
                    fold_scores = []
                    for model, target, X_target, y_target, full_args in target_args:
                        estimator = None
                        try:
                            estimator = model.model_cls(**full_args)
                            estimator.fit(X_target[idx_train], y_target[idx_train])
                            preds = estimator.predict(X_target[idx_test])
                            score = metric.tuning_score(
                                y_target[idx_test].numpy(), preds
                            )
                            fold_scores.append(
                                self._scale_tuning_score(metric, target, score)
                            )
                            target_name = (
                                str(target.name)
                                if target.name is not None
                                else f"target_{len(fold_scores) - 1}"
                            )
                            scores_by_target.setdefault(target_name, []).append(
                                fold_scores[-1]
                            )
                        finally:
                            estimator = None
                            model._cleanup_after_fold()
                    scores.append(float(np.mean(fold_scores)))
                    trial.report(float(np.mean(scores)), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                self._record_trial_target_scores(trial, scores_by_target)
                return float(np.mean(scores))

            return objective
        if isinstance(y_train, DataFrame):
            y_train = y_train.iloc[:, 0]
        X, y = self._to_torch(X_train, y_train)

        def objective(trial: Trial) -> float:
            kf = OmniKFold(
                n_splits=n_folds,
                is_classification=self.is_classifier,
                grouped=g_train is not None,
                labels=None,
                warn_on_fallback=False,
                allow_group_fallback=False,
                df_analyze_phase="Tuning internal splits",
            )
            opt_args = self.optuna_args(trial)
            model_args = self._to_model_args(opt_args, X_train)
            full_args = {
                **self.fixed_args,
                **self.default_args,
                **self.model_args,
                **model_args,
            }
            scores = []
            for step, (idx_train, idx_test) in enumerate(
                kf.split(X_train, y_train, g_train)[0]
            ):
                X_tr, y_tr = X[idx_train], y[idx_train]
                X_test, y_test = X[idx_test], y[idx_test]
                estimator = None
                try:
                    estimator = self.model_cls(**full_args)
                    estimator.fit(X_tr, y_tr)
                    preds = estimator.predict(X_test)
                    score = metric.tuning_score(y_test.numpy(), preds)
                    scores.append(score)
                    # Reporting each intermediate fold score allows pruning.
                    trial.report(float(np.mean(scores)), step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                finally:
                    estimator = None
                    self._cleanup_after_fold()
            return float(np.mean(scores))

            # estimator = self.model_cls(**full_args)

            # scoring = "accuracy" if self.is_classifier else NEG_MAE
            # # _cv = kf(n_splits=n_folds, shuffle=True, random_state=SEED)
            # # TODO: maybe use NeuralNet class cv option?
            # # TODO: maybe not use k-fold here due to costs, or e.g. 2-fold
            # idx_tr, idx_test = next(_cv.split(y, y))
            # X_tr, X_test = X[idx_tr], X[idx_test]
            # y_tr, y_test = y[idx_tr], y[idx_test]
            # estimator.fit(X_tr, y_tr)
            # score = estimator.score(X_test, y_test)
            # return -float(score)
            # scores = cv(
            #     estimator,  # type: ignore
            #     X=X,
            #     y=y,
            #     scoring=scoring,
            #     cv=_cv,
            #     n_jobs=1,
            #     verbose=1,
            # )
            # return float(np.mean(scores["test_score"]))

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
        try:
            plat = platform.platform().lower()
        except Exception:
            plat = ""

        if os.environ.get("CC_CLUSTER") == "niagara":
            # Niagara trials
            #
            # n_jobs    50 models   10      20 models    30 models   40 models
            #      1        s                 ~4 min     ~5.4        ~6.8 min
            #      2        s       3 min     ~6 min     ~9 min      ~10.5 min
            #      4        s
            #      8        s                8-9 min     15 min
            #
            # Seems VERY clear to make n_jobs=1 here
            override_jobs = 1

        elif ("macos" in plat) and ("arm64" in plat):
            # Local M2 MacBook Air, 16GB RAM, 500GB
            #
            # n_jobs    50 models
            #     -1     ~7 min
            #      1    6.6 min (6-7 min)
            #      2     ~6 min
            #      4     ~3.5-6 min               # confirmed this is sweet spot
            #      8     ~6 min for 30 models
            override_jobs = 4
        else:
            override_jobs = 2  # seems to work / be least harmful on both tested

        override_jobs = self.runtime.tuning_jobs(RuntimeComponent.MLP, override_jobs)
        return super().htune_optuna(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            verbosity=verbosity,
            n_jobs=override_jobs,
        )


if __name__ == "__main__":
    X, y = make_classification(2000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1000:]
    y_test = y[1000:]
    wd = 1e-4
    lr = 1e-3
    # for lr in tqdm([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]):
    for lr in tqdm([1e-4]):
        T0, n_batches = get_T0(X, n_epochs=8, val_split=0.2)
        sched = LRScheduler(
            policy=CosineAnnealingWarmRestarts,  # type: ignore
            T_0=T0,
            T_mult=2,
            eta_min=0,
            verbose=False,
            step_every="step",
        )
        T0, n_batches = get_T0(X, n_epochs=50, val_split=0.2)
        sched = LRScheduler(
            policy=CosineAnnealingLR,  # type: ignore
            # T_max=T0,
            T_max=T0,
            eta_min=0,
            verbose=True,
            step_every="step",
        )
        # lrs = sched.simulate(steps=50 * n_batches, initial_lr=lr)
        # epochs = [int(i / n_batches) for i in range(len(lrs))]
        # ax: Axes
        # fig, ax = plt.subplots()
        # ax.plot(epochs, lrs, color="black")
        # ax.set_xlabel("Epoch")
        # ax.set_ylabel("LR")
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        # ax.set_yscale("log")
        # plt.show()
        # plt.close()
        # sys.exit()

        net = NeuralNetClassifier(
            module=SkorchMLP,
            module__width=128,
            module__use_bn=True,
            module__dropout=0.4,
            module__num_classes=2,
            criterion=CrossEntropyLoss,  # type: ignore
            criterion__weight=None,
            optimizer=AdamW,
            optimizer__weight_decay=wd,
            optimizer__lr=lr,
            callbacks=[
                sched,
                # Effectively min_epochs=20
                EarlyStopping(patience=20, load_best=False),
            ],
            max_epochs=50,
            batch_size=BATCH_SIZE,
            # train_split=None,  # we do all tuning with 5-fold anyway...
            device="cpu",
            verbose=0,
        )
        net.fit(X_train, y_train)
        # probs = net.predict_proba(X_test)
        # print(probs[:10])
        # preds = net.predict(X_test)
        # print(preds[:10])
        print("Acc:", net.score(X_test, y_test))
        # fig, ax = plt.subplots()
        # ax.plot(net.history[:, "train_loss"], color="black", label="train")
        # ax.plot(net.history[:, "valid_loss"], color="orange", label="val")
        # ax.set_title(f"LR={lr}")
        # plt.show(block=False)
    plt.show()

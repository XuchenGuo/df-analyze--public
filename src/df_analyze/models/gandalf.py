from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import gc
import traceback

import matplotlib as mpl

mpl.rcParams["axes.formatter.useoffset"] = False

import multiprocessing
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from time import sleep
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sized,
    Type,
    Union,
)
from uuid import uuid4

import numpy as np
import optuna
import torch
from numpy import ndarray
from optuna import Study, Trial
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping as LightningEarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import (
    SLURMEnvironment,  # pyright: ignore[reportPrivateImportUsage]
)
from pytorch_tabular.models.common.layers import BatchNorm1d as GhostBatchNorm1d
from pytorch_tabular.models.common.layers.activations import t_softmax
from pytorch_tabular.models.common.layers.gated_units import (
    GatedFeatureLearningUnit as GFLU,
)
from sklearn.preprocessing import OrdinalEncoder
from skorch.callbacks.lr_scheduler import CosineAnnealingLR
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Identity,
    L1Loss,
    Linear,
    Module,
    ModuleList,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, MeanAbsoluteError

from df_analyze.enumerables import Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.runtime.hardware import (
    RuntimeComponent,
    cleanup_torch_accelerator,
    configure_torch_cuda,
)
from df_analyze.splitting import ApproximateStratifiedGroupSplit

"""
If we use the usual df-analyze categorical processing pipeline, then we don't
need to worry about categorical embeddings and the like. At the moment, this
is safe, as there is no evidence supporting the superiority of one-hot vs.
embeddings, with perhaps the only exception being the piecewise linear
encoding paper.
"""

BATCH = 512
"""We fix batch size at 512 like original paper"""
EPOCHS = 50

LOGS = ROOT / "__GANDALF_INTERNAL_LOGS__"
LOGS.mkdir(exist_ok=True, parents=True)


class WindowsSafeModelCheckpoint(ModelCheckpoint):
    """Tolerate transient Windows file locks while rotating best checkpoints."""

    _REMOVE_RETRIES = 5

    def _remove_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        for attempt in range(self._REMOVE_RETRIES):
            try:
                super()._remove_checkpoint(trainer, filepath)
                return
            except PermissionError:
                if sys.platform != "win32":
                    raise
                if attempt == self._REMOVE_RETRIES - 1:
                    warnings.warn(
                        "Windows kept a stale GANDALF checkpoint locked; "
                        f"retaining it instead of aborting training: {filepath}",
                        stacklevel=2,
                    )
                    return
                gc.collect()
                sleep(0.05 * (2**attempt))

# fmt: off
TUNING_SPACE = dict(
    vbatch=[8, 16, 32, 64],              # ghost (virtual) batch norm size
    lr=[3e-4, 3e-3, 3e-2, 1e-1],         # should be large
    wd=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],   # should be quite small
    anneal=[True],                       # use CosineAnnealingLRSchedule
    bnorm_cont=[True],                   # use batch norm on continuous features
    embed=[True, False],                 # use learnable embeddings for categoricals
    embed_dim=[4, 8],                    # embedding dimension size for each, above
    drop=[0.1, 0.3, 0.5, 0.65, 0.8],     # embed_dropout
    gdrop=[0.0, 0.01, 0.02, 0.05, 0.1],  # gflu_dropout
    depth=[8, 12, 16, 20, 24, 32],       # number of GFLU layers
    sparsity=[0.1, 0.3, 0.5, 0.7, 0.9],  # init_sparsity
)
# fmt: on

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")
warnings.filterwarnings("ignore", ".*You are using `torch.load`.*")

# https://github.com/Lightning-AI/pytorch-lightning/issues/6389#issuecomment-1997042135
SLURMEnvironment.detect = lambda: False


class DisabledSLURMEnvironment(SLURMEnvironment):
    @staticmethod
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def get_T0(
    X: Union[ndarray, DataFrame],
    n_epochs: int,
    batch_size: int = BATCH,
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


class TabularData(Dataset):
    def __init__(
        self,
        X_cont: DataFrame,
        X_cat: Optional[DataFrame],
        y: Series,
        is_classification: bool = True,
    ) -> None:
        super().__init__()
        self.is_classification = is_classification
        is_cls = self.is_cls = self.is_classification

        dtype = torch.int64 if is_cls else torch.float32
        nptype = np.int64 if is_cls else np.float32
        # self.y = torch.from_numpy(y.to_numpy().astype(np.float32)).to(dtype=dtype)
        self.y = torch.from_numpy(y.to_numpy().astype(nptype)).to(dtype=dtype)
        self.n_cls = len(y.unique()) if is_cls else 1
        self.X_cont = torch.from_numpy(X_cont.to_numpy().astype(np.float32)).float()

        if X_cat is not None:
            encoder = OrdinalEncoder()
            enc = encoder.fit_transform(X_cat.astype("string").map(str)).astype(np.int64)
            self.X_cat = torch.from_numpy(enc)
            cards = [cat.shape[0] for cat in encoder.categories_]  # type: ignore
            self.cardinalities = cards
        else:
            self.cardinalities = []
            self.X_cat = None

        if self.is_cls:
            unqs, cnts = self.y.unique(return_counts=True)
            self.cls_weights = cnts / cnts.sum()
        else:
            self.cls_weights = None

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.X_cat is None:
            return self.X_cont[i, :], torch.empty([0, 0]), self.y[i]
        else:
            return self.X_cont[i, :], self.X_cat[i, :], self.y[i]

    def __len__(self) -> int:
        return self.y.shape[0]


class ContinuousData(Dataset):
    def __init__(
        self,
        df: DataFrame,
        y: Series,
        g: Optional[Series] = None,
        is_classification: bool = True,
    ) -> None:
        super().__init__()
        self.is_classification = is_classification
        is_cls = self.is_cls = self.is_classification

        dtype = torch.int64 if is_cls else torch.float32
        nptype = np.int64 if is_cls else np.float32
        # self.y = torch.from_numpy(y.to_numpy().astype(np.float32)).to(dtype=dtype)
        self.y = torch.from_numpy(y.to_numpy().astype(nptype)).to(dtype=dtype)
        if g is not None:
            self.g = torch.from_numpy(g.to_numpy().astype(np.int64)).to(dtype=torch.int64)
        else:
            self.g = None
        self.n_cls = len(y.unique()) if is_cls else 1
        self.X = torch.from_numpy(df.to_numpy().astype(np.float32)).float()

        if self.is_cls:
            unqs, cnts = self.y.unique(return_counts=True)
            self.cls_weights = cnts / cnts.sum()
        else:
            self.cls_weights = None
        self.n_feat = self.X.shape[1]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.X[i, :], self.y[i]

    def __len__(self) -> int:
        return self.y.shape[0]


class ContinuousPredData(Dataset):
    def __init__(
        self,
        df: DataFrame | Tensor,
        g: Optional[Series] = None,
        is_classification: bool = True,
    ) -> None:
        super().__init__()
        self.is_classification = is_classification
        self.X: Tensor

        if g is not None:
            self.g = torch.from_numpy(g.to_numpy().astype(np.int64)).to(dtype=torch.int64)
        else:
            self.g = None
        if isinstance(df, DataFrame):
            self.X = torch.from_numpy(df.to_numpy().astype(np.float32)).float()
        else:
            self.X = df

    def __getitem__(self, i: int) -> Tensor:
        return self.X[i, :]

    def __len__(self) -> int:
        return self.X.shape[0]


class AddValue(Module):
    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return x + self.value


def _dropout_layer(probability: float, name: str) -> Module:
    value = float(probability)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= probability < 1. Got {value}.")
    return Dropout(value) if value > 0.0 else Identity()


class GandlfEmbeddingLayer(Module):
    def __init__(
        self,
        dataset: TabularData,
        embed_dim: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cats = len(dataset.cardinalities)
        self.cardinalities = dataset.cardinalities
        self.dropout = _dropout_layer(
            dropout,
            "GANDALF embedding dropout",
        )
        layers = []
        for cardinality in self.cardinalities:
            layers.append(
                Embedding(num_embeddings=cardinality, embedding_dim=self.embed_dim)
            )
        self.embedders = ModuleList(layers)

    def forward(self, x_cat: Tensor) -> Tensor:
        # x_cat is ordinal-encoded categoricals
        # x_cat.shape is [B, P] or [B, 0, 0];
        if x_cat.shape[1] == 0:
            return x_cat

        embeds = []
        for i, embedder in enumerate(self.embedders):
            embeds.append(embedder(x_cat[:, i]))
        if len(embeds) == 0:
            return x_cat
        x_cat = torch.cat(embeds, dim=-1)
        x_cat = self.dropout(x_cat)
        return x_cat


class GandalfContCatModel(Module):
    def __init__(
        self,
        dataset: TabularData,
        depth: int = 8,
        embed_dim: int = 16,
        embed_dropout: float = 0.3,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.is_classification = self.is_cls = dataset.is_classification
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.virtual_batch_size = virtual_batch_size
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        vb = self.virtual_batch_size

        X_cat = dataset.X_cat
        self.n_cls = dataset.n_cls
        self.n_cont = dataset.X_cont.shape[1]
        self.n_cat = 0 if X_cat is None else X_cat.shape[1]
        self.n_feat = self.n_cont + self.n_cat * self.embed_dim

        self.cat_embedder = GandlfEmbeddingLayer(
            dataset, embed_dim=embed_dim, dropout=embed_dropout
        )
        self.model = GFLU(
            n_features_in=self.n_feat,
            n_stages=depth,
            feature_mask_function=t_softmax,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
        )
        self.cont_dropout = _dropout_layer(
            self.gflu_dropout,
            "GANDALF continuous GFLU dropout",
        )
        self.cont_bnorm = (
            GhostBatchNorm1d(self.n_cont, vb)
            if self.batch_norm_continuous_input
            else Identity()
        )
        """
        For all our experiments, we kept the dimensions of the Multi Layer
        Perceptron constant with two hidden layers of 32 and 16 units
        respectively with a ReLU activation, a
        """
        is_cls = self.is_classification
        if not is_cls:
            ymean = dataset.y.mean().reshape([1, -1])
            self.T0 = Parameter(
                torch.tensor(ymean, dtype=torch.float32), requires_grad=True
            )

        self.output = Sequential(
            GhostBatchNorm1d(self.n_feat, vb),
            Linear(in_features=self.n_feat, out_features=32, bias=False),
            # LeakyReLU(),
            ReLU(),
            GhostBatchNorm1d(32, vb),
            Linear(in_features=32, out_features=16, bias=False),
            # LeakyReLU(),
            ReLU(),
            GhostBatchNorm1d(16, vb),
            Linear(in_features=16, out_features=self.n_cls),
            # Identity() if is_cls else Softsign(),
            # Identity()
            # if is_cls
            # else Linear(in_features=self.n_cls, out_features=self.n_cls),
            Identity() if is_cls else AddValue(self.T0),
            # Identity() if is_cls else Sigmoid(),
            # Identity() if is_cls else AddValue(self.T0),
        )

    def forward(self, x_cont: Tensor, x_cat: Tensor, y: Tensor) -> Tensor:
        x_cont = self.cont_bnorm(x_cont)  # x_cont: (B, n_cont)
        x_cont = self.cont_dropout(x_cont)
        if x_cat.shape[1] != 0:
            x_emb = self.cat_embedder(x_cat)  # x_emb: (B, n_cat * embed_dim)
            x = torch.cat([x_cont, x_emb], dim=-1)
        else:
            x = x_cont
        x = self.model(x)
        x = self.output(x)
        return x


class GandalfContModel(Module):
    def __init__(
        self,
        dataset: ContinuousData,
        depth: int = 8,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.is_classification = self.is_cls = dataset.is_classification
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.virtual_batch_size = virtual_batch_size
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        vb = self.virtual_batch_size

        self.n_cls = dataset.n_cls
        self.n_feat = dataset.n_feat

        self.model = GFLU(
            n_features_in=self.n_feat,
            n_stages=depth,
            feature_mask_function=t_softmax,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
        )
        self.dropout = _dropout_layer(
            self.gflu_dropout,
            "GANDALF GFLU dropout",
        )
        self.bnorm = (
            GhostBatchNorm1d(self.n_feat, vb)
            if self.batch_norm_continuous_input
            else Identity()
        )
        """
        For all our experiments, we kept the dimensions of the Multi Layer
        Perceptron constant with two hidden layers of 32 and 16 units
        respectively with a ReLU activation, a
        """
        is_cls = self.is_classification
        if not is_cls:
            ymean = dataset.y.mean().reshape([1, -1]).to(dtype=torch.float32)
            self.T0 = Parameter(
                ymean.clone().detach(),
                requires_grad=True,
            )

        self.output = Sequential(
            GhostBatchNorm1d(self.n_feat, vb),
            Linear(in_features=self.n_feat, out_features=32, bias=False),
            ReLU(),
            GhostBatchNorm1d(32, vb),
            Linear(in_features=32, out_features=16, bias=False),
            ReLU(),
            GhostBatchNorm1d(16, vb),
            Linear(in_features=16, out_features=self.n_cls),
            Identity() if is_cls else AddValue(self.T0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.bnorm(x)  # x: (B, n)
        x = self.dropout(x)
        x = self.model(x)
        x = self.output(x)
        return x


class GandalfContCatLightningModel(LightningModule):
    def __init__(
        self,
        dataset: TabularData,
        lr: float = 3e-4,
        wd: float = 1e-4,
        epochs: int = 10,
        anneal: bool = False,
        depth: int = 8,
        embed: bool = True,
        embed_dim: int = 16,
        embed_dropout: float = 0.3,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = False,
        virtual_batch: int = 32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_cls = dataset.n_cls
        self.cls_weights = dataset.cls_weights
        self.depth = depth
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.anneal = anneal
        self.embed = embed
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.virtual_batch = virtual_batch
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.is_classification = self.is_cls = dataset.is_classification
        self.model = GandalfContCatModel(
            dataset=dataset,
            depth=self.depth,
            embed_dim=self.embed_dim,
            embed_dropout=self.embed_dropout,
            gflu_dropout=self.gflu_dropout,
            gflu_feature_init_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
            virtual_batch_size=self.virtual_batch,
        )
        # self.loss = CrossEntropyLoss() if self.is_cls else HuberLoss()
        self.loss = CrossEntropyLoss(weight=self.cls_weights) if self.is_cls else L1Loss()
        # self.loss = CrossEntropyLoss() if self.is_cls else MSELoss()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def training_step(  # type: ignore
        self,
        batch: tuple[Tensor, Optional[Tensor], Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        loss = self.shared_step(batch)[0]
        return loss

    def validation_step(  # type: ignore
        self, batch: tuple[Tensor, Optional[Tensor], Tensor], *args: Any, **kwargs: Any
    ) -> None:
        with torch.no_grad():
            loss, preds = self.shared_step(batch)
        self.log("val_loss", loss)

    def predict_step(
        self, batch: tuple[Tensor, Optional[Tensor], Tensor], *args: Any, **kwargs: Any
    ) -> Any:
        with torch.no_grad():
            preds = self.shared_step(batch)[1]
        return preds

    def configure_optimizers(
        self,
    ) -> Any:
        if not self.anneal:
            return AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-9)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

    def shared_step(
        self, batch: tuple[Tensor, Optional[Tensor], Tensor]
    ) -> tuple[Tensor, Tensor]:
        X_cont, X_cat, target = batch
        preds = self.model(X_cont, X_cat, target)
        if self.is_cls:
            target = target.long()
        if target.ndim == 1:
            loss = self.loss(preds.squeeze(1), target)
        else:
            loss = self.loss(preds, target)
        return loss, preds


class GandalfContLightningModel(LightningModule):
    def __init__(
        self,
        dataset: ContinuousData,
        lr: float = 3e-4,
        wd: float = 1e-4,
        epochs: int = 50,
        anneal: bool = False,
        depth: int = 8,
        gflu_dropout: float = 0.0,
        init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        bnorm_cont: bool = False,
        virtual_batch: int = 32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_cls = dataset.n_cls
        self.cls_weights = dataset.cls_weights
        self.depth = depth
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.anneal = anneal
        self.gflu_dropout = gflu_dropout
        self.bnorm_cont = bnorm_cont
        self.virtual_batch = virtual_batch
        self.init_sparsity = init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.is_classification = self.is_cls = dataset.is_classification
        self.model = GandalfContModel(
            dataset=dataset,
            depth=self.depth,
            gflu_dropout=self.gflu_dropout,
            gflu_feature_init_sparsity=self.init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
            batch_norm_continuous_input=self.bnorm_cont,
            virtual_batch_size=self.virtual_batch,
        )
        self.loss = (
            # Line below is causing anomalies on Niagara, we have to use
            # unweighted version:
            # CrossEntropyLoss(weight=self.cls_weights) if self.is_cls else MSELoss()
            CrossEntropyLoss() if self.is_cls else MSELoss()
        )
        self.metric = (
            Accuracy(task="multiclass", num_classes=self.n_cls)
            if self.is_cls
            else MeanAbsoluteError()
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def predict_step(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        with torch.no_grad():
            preds = self.model(x)
        return preds

    def training_step(  # type: ignore
        self,
        batch: tuple[Tensor, Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        loss = self.shared_step(batch)[0]
        return loss

    def validation_step(  # type: ignore
        self, batch: tuple[Tensor, Tensor], *args: Any, **kwargs: Any
    ) -> None:
        with torch.no_grad():
            # preds.shape: (B, self.n_cls)
            loss, preds = self.shared_step(batch)

        target = batch[1]  # target.shape (B,)
        self.log("val/loss", loss)
        if self.is_cls:
            self.log(
                "val/metric",
                self.metric(preds, target),
                on_step=False,
                on_epoch=True,
            )
        else:
            self.log(
                "val/metric",
                self.metric(preds.squeeze(1), target),
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(
        self,
    ) -> Any:
        if not self.anneal:
            return AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-9)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

    def shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, target = batch
        preds = self.model(x)
        if self.is_cls:
            target = target.long()
        if target.ndim == 1:
            loss = self.loss(preds.squeeze(1), target)
        else:
            loss = self.loss(preds, target)
        return loss, preds


class GandalfEstimator(DfAnalyzeModel):
    shortname = "gandalf"
    longname = "GANDALF - Gated Adaptive Network"
    runtime_component = RuntimeComponent.Gandalf
    timeout_s = 3600
    # GANDALF tunes against an internal train/validation split rather than
    # OmniKFold, so the shared K-fold support preflight does not apply.
    tuning_cv_folds = None

    def __init__(self, num_classes: int, model_args: Mapping | None = None) -> None:
        super().__init__(model_args)
        self.is_classifier = num_classes > 1
        self.n_cls = num_classes
        self.model_cls: Type[GandalfContLightningModel] = GandalfContLightningModel
        self.model: GandalfContLightningModel
        self.trainer: Optional[Trainer] = None
        self.tuned_trainer: Optional[Trainer] = None
        self.target_cols: list[str] = []

    def _configure_runtime(self) -> None:
        configure_torch_cuda(self.runtime, RuntimeComponent.Gandalf)

    def _cleanup_after_fold(self) -> None:
        cleanup_torch_accelerator(self.runtime, RuntimeComponent.Gandalf)

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

    def _new_target_estimator(self, y: Series) -> GandalfEstimator:
        model = type(self)(
            num_classes=self._target_num_classes(y), model_args=self.model_args
        )
        model.set_runtime(self.runtime)
        return model

    def _fit_target_estimators(
        self,
        X: DataFrame,
        y: DataFrame,
        g: Optional[Series] = None,
        tuned_args: Optional[Mapping] = None,
    ) -> dict[str, GandalfEstimator]:
        models: dict[str, GandalfEstimator] = {}
        for col in y.columns:
            target = str(col)
            model = self._new_target_estimator(y[col])
            if tuned_args is None:
                model.fit(X, y[col], g_train=g)
            else:
                model.refit_tuned(X, y[col], g=g, tuned_args=tuned_args)
            models[target] = model
        return models

    def _predict_target_estimators(
        self, models: dict[str, GandalfEstimator], X: DataFrame, tuned: bool
    ) -> DataFrame:
        predictions = {}
        for target in self.target_cols:
            model = models[target]
            pred = model.tuned_predict(X) if tuned else model.predict(X)
            predictions[target] = np.asarray(pred).reshape(-1)
        return DataFrame(predictions, index=X.index, columns=self.target_cols)

    def _predict_target_proba(
        self, models: dict[str, GandalfEstimator], X: DataFrame, tuned: bool
    ) -> dict[str, ndarray]:
        return {
            target: np.asarray(
                models[target].predict_proba(X)
                if tuned
                else models[target].predict_proba_untuned(X)
            )
            for target in self.target_cols
        }

    def _loader_kwargs(self) -> dict[str, Any]:
        device = self.runtime.device_for(RuntimeComponent.Gandalf)
        workers = 0 if device == "cuda" or sys.platform.startswith("win") else 1
        return {
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "pin_memory": device == "cuda",
        }

    def optuna_args(self, trial: Trial) -> dict[str, str | float | int]:
        return dict(
            lr=trial.suggest_float("lr", 1e-5, 5e-1, log=True),
            wd=trial.suggest_float("wd", 1e-8, 1e-3, log=True),
            anneal=trial.suggest_categorical("anneal", [True]),
            depth=trial.suggest_categorical("depth", [4, 8, 12, 16, 20, 24, 32]),
            gflu_dropout=trial.suggest_float("gflu_dropout", 0.01, 0.1, log=False),
            init_sparsity=trial.suggest_float("init_sparsity", 0.1, 0.9, log=False),
            bnorm_cont=trial.suggest_categorical("bnorm_cont", [True]),
            virtual_batch=trial.suggest_categorical("virtual_batch", [8, 16, 32, 64]),
        )

    def _pred_loader(self, X: DataFrame | Tensor, g: Optional[Series]) -> DataLoader:
        data = ContinuousPredData(df=X, g=g, is_classification=self.is_classifier)
        loader = DataLoader(
            data,
            batch_size=min(BATCH, len(data)),
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )
        return loader

    def _train_val_loaders(
        self, X_train: DataFrame, y_train: Series, g_train: Optional[Series] = None
    ) -> tuple[DataLoader, DataLoader]:
        # data = ContinuousData(
        #     df=X_train, y=y_train, g=g_train, is_classification=self.is_classifier
        # )
        is_cls = self.is_classifier
        ss = ApproximateStratifiedGroupSplit(
            train_size=0.8,
            is_classification=is_cls,
            grouped=g_train is not None,
            labels=None,
            warn_on_fallback=False,
            allow_group_fallback=False,
            warn_on_large_size_diff=False,
            df_analyze_phase="GANDALF train-val split",
        )
        ix_tr, ix_v = ss.split(X_train=X_train, y_train=y_train, g_train=g_train)[0]
        g_tr = g_train.iloc[ix_tr] if g_train is not None else None
        g_v = g_train.iloc[ix_v] if g_train is not None else None
        train = ContinuousData(
            df=X_train.iloc[ix_tr],
            y=y_train.iloc[ix_tr],
            g=g_tr,
            is_classification=is_cls,
        )
        val = ContinuousData(
            df=X_train.iloc[ix_v], y=y_train.iloc[ix_v], g=g_v, is_classification=is_cls
        )

        # if we don't set a generator we get a segfault...
        train_batch = min(BATCH, len(train))
        if train_batch > (len(train) // 2):
            train_batch = 32  # prevent overfitting small datasets
        val_batch = min(BATCH, len(val))
        train_loader = DataLoader(
            train,
            batch_size=train_batch,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )
        val_loader = DataLoader(
            val,
            batch_size=val_batch,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )
        return train_loader, val_loader

    def _trainer_hardware(self) -> tuple[str, int | str]:
        device = self.runtime.device_for(RuntimeComponent.Gandalf)
        if device == "cuda":
            return "gpu", 1
        if device == "mps":
            return "mps", 1
        return "cpu", "auto"

    def _uses_accelerator(self) -> bool:
        return self.runtime.uses_accelerator(RuntimeComponent.Gandalf)

    def model_cls_args(self, full_args: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        return self.model_cls, full_args

    def _get_trainer(
        self, train: DataLoader, val: DataLoader, trial: Optional[Trial] = None
    ) -> Trainer:
        is_cls = self.is_classifier
        run_kind = "refit" if trial is None else f"trial-{trial.number}"
        logs = LOGS / f"{run_kind}-{uuid4().hex}"
        logs.mkdir(exist_ok=True, parents=True)
        (logs / "lightning_logs").mkdir(exist_ok=True, parents=True)
        logger = TensorBoardLogger(save_dir=logs, default_hp_metric=False)
        stop = "val/metric"
        delta = 0.002
        mode = "max" if is_cls else "min"
        # ckpt_metric = "val/loss"
        cbs = [
            # ModelCheckpoint(monitor=ckpt_metric, every_n_epochs=1),
            WindowsSafeModelCheckpoint(
                dirpath=logs / "checkpoints",
                monitor=stop,
                mode=mode,
                every_n_epochs=1,
            ),
            LightningEarlyStopping(monitor=stop, patience=7, min_delta=delta, mode=mode),
        ]
        # ensure we get at least one ckpt file...
        if train.dataset is None:
            raise ValueError("Missing training data.")
        if train.batch_size is None:
            raise ValueError("Missing training batch size data.")
        if not isinstance(train.dataset, Sized):
            raise RuntimeError("Unsupported training dataset: dataset has no len()")

        if val.dataset is None:
            raise ValueError("Missing validation data.")
        if val.batch_size is None:
            raise ValueError("Missing validation batch size data.")
        if not isinstance(val.dataset, Sized):
            raise RuntimeError("Unsupported training dataset: dataset has no len()")

        log_freq_train = max(1, min(len(train.dataset) // train.batch_size, 4))
        log_freq_val = max(1, min(len(val.dataset) // val.batch_size, 4))
        log_freq = min(log_freq_train, log_freq_val)
        accelerator, devices = self._trainer_hardware()
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
            max_epochs=50,
            min_epochs=10,
            log_every_n_steps=log_freq,
            callbacks=cbs,
            enable_progress_bar=False,
            enable_model_summary=False,
            detect_anomaly=False,
            benchmark=self.runtime.device_for(RuntimeComponent.Gandalf) == "cuda",
        )
        return trainer

    def fit(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series] = None,
    ) -> None:
        self._set_target_cols(y_train)
        if isinstance(y_train, DataFrame) and y_train.shape[1] > 1:
            self.model = self._fit_target_estimators(  # type: ignore[assignment]
                X_train, y_train, g=g_train
            )
            return
        if isinstance(y_train, DataFrame):
            y_train = y_train.iloc[:, 0]
        kwargs: Mapping = {**self.fixed_args, **self.default_args, **self.model_args}
        data = ContinuousData(
            df=X_train, y=y_train, g=g_train, is_classification=self.is_classifier
        )
        kwargs["dataset"] = data
        if self.model is None:
            try:
                self.model = self.model_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Error instantiating model {self.model_cls.__name__} with kwargs: `{kwargs}`"
                ) from e
        try:
            train, val = self._train_val_loaders(
                X_train=X_train, y_train=y_train, g_train=g_train
            )
            self.trainer = self._get_trainer(train=train, val=val)
            if self.trainer is None:
                raise ValueError("Could not get Optuna Trainer.")
            self.trainer.fit(
                model=self.model, train_dataloaders=train, val_dataloaders=val
            )
        except ValueError as e:
            raise RuntimeError(
                "Got exception when trying to fit GANDALF.\n"
                f"X={type(X_train)}, shape={X_train.shape}\n"
                f"y={type(y_train)}, shape={y_train.shape}\n"
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
                X, y, g=g, tuned_args=tuned_args or {}
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
        data = ContinuousData(df=X, y=y, g=g, is_classification=self.is_classifier)
        kwargs["dataset"] = data
        train, val = self._train_val_loaders(X_train=X, y_train=y, g_train=g)
        self.tuned_trainer = self._get_trainer(train=train, val=val)
        if self.tuned_trainer is None:
            raise ValueError("Could not get Optuna tuned Trainer.")
        self.tuned_model = self.model_cls(**kwargs)
        self.tuned_trainer.fit(
            model=self.tuned_model, train_dataloaders=train, val_dataloaders=val
        )

    def _predict_logits(
        self,
        *,
        trainer: Trainer,
        model: LightningModule,
        loader: DataLoader,
    ) -> Tensor:
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best_path = getattr(ckpt_cb, "best_model_path", "")
        if (
            isinstance(best_path, str)
            and (len(best_path) > 0)
            and Path(best_path).exists()
        ):
            all_logits = trainer.predict(
                model=model, dataloaders=loader, ckpt_path="best"
            )
        else:
            all_logits = trainer.predict(model=model, dataloaders=loader)
        return torch.concatenate(all_logits, dim=0)  # type: ignore

    def predict(self, X: DataFrame) -> Union[ndarray, DataFrame]:
        if isinstance(self.model, dict):
            return self._predict_target_estimators(self.model, X, tuned=False)
        if self.trainer is None:
            raise RuntimeError("Model has not been trained yet.")
        loader = self._pred_loader(X=X, g=None)
        logits = self._predict_logits(
            trainer=self.trainer,
            model=self.model,
            loader=loader,
        )
        if self.is_classifier:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            return probs.argmax(axis=1)
        return logits.detach().cpu().numpy()

    def tuned_predict(self, X: DataFrame) -> Union[ndarray, DataFrame]:
        if isinstance(self.tuned_model, dict):
            return self._predict_target_estimators(self.tuned_model, X, tuned=True)
        if self.tuned_trainer is None:
            raise RuntimeError("Model has not been trained yet.")
        loader = self._pred_loader(X=X, g=None)
        logits = self._predict_logits(
            trainer=self.tuned_trainer,
            model=self.tuned_model,
            loader=loader,
        )
        if self.is_classifier:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            return probs.argmax(axis=1)
        return logits.detach().cpu().numpy()

    def predict_proba_untuned(self, X: DataFrame) -> Union[ndarray, dict[str, ndarray]]:
        if not self.is_classifier:
            raise ValueError("Can't predict probabilities for regression.")
        if isinstance(self.model, dict):
            return self._predict_target_proba(self.model, X, tuned=False)
        if self.trainer is None:
            raise RuntimeError("Model has not been trained yet.")
        loader = self._pred_loader(X=X, g=None)
        logits = self._predict_logits(
            trainer=self.trainer,
            model=self.model,
            loader=loader,
        )
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def predict_proba(self, X: DataFrame) -> Union[ndarray, dict[str, ndarray]]:
        if not self.is_classifier:
            raise ValueError("Can't predict probabilities for regression.")
        if isinstance(self.tuned_model, dict):
            return self._predict_target_proba(self.tuned_model, X, tuned=True)
        if self.tuned_trainer is None:
            raise RuntimeError("Model has not been tuned yet.")
        loader = self._pred_loader(X=X, g=None)
        logits = self._predict_logits(
            trainer=self.tuned_trainer,
            model=self.tuned_model,
            loader=loader,
        )
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def tuned_scores(self, X: DataFrame, y: Union[Series, DataFrame]) -> float:
        if isinstance(self.tuned_model, dict):
            if not isinstance(y, DataFrame):
                raise ValueError("Expected DataFrame targets for a multi-target model.")
            scores = [
                self.tuned_model[str(col)].tuned_scores(X, y[col]) for col in y.columns
            ]
            return float(np.mean(scores))
        return super().tuned_scores(X, y)

    def _to_model_args(self, optuna_args: Mapping, X_train: DataFrame) -> Mapping:
        final_args: Mapping = deepcopy(optuna_args)
        return final_args

    def optuna_objective(
        self,
        X_train: DataFrame,
        y_train: Union[Series, DataFrame],
        g_train: Optional[Series],
        metric: Scorer,
        n_folds: int = 3,
    ) -> Callable[[Trial], float]:
        self._set_target_cols(y_train)
        if isinstance(y_train, DataFrame) and y_train.shape[1] > 1:

            def objective(trial: Trial) -> float:
                scores = []
                scores_by_target: dict[str, list[float]] = {}
                for col in y_train.columns:
                    model = self._new_target_estimator(y_train[col])
                    target_objective = model.optuna_objective(
                        X_train, y_train[col], g_train, metric, n_folds
                    )
                    score = float(target_objective(trial))
                    scaled = self._scale_tuning_score(metric, y_train[col], score)
                    scores.append(scaled)
                    scores_by_target[str(col)] = [scaled]
                self._record_trial_target_scores(trial, scores_by_target)
                return float(np.mean(scores))

            return objective
        if isinstance(y_train, DataFrame):
            y_train = y_train.iloc[:, 0]
        data = ContinuousData(
            df=X_train, y=y_train, g=g_train, is_classification=self.is_classifier
        )

        def objective(trial: Trial) -> float:
            # these MUST be created in the objective, or we get a bug with
            # the iter() loop "generator already executing"
            try:
                train, val = self._train_val_loaders(
                    X_train=X_train, y_train=y_train, g_train=g_train
                )
                opt_args = self.optuna_args(trial)
                model_args = self._to_model_args(opt_args, X_train)
                full_args = {
                    **self.fixed_args,
                    **self.default_args,
                    **self.model_args,
                    **model_args,
                }
                full_args["dataset"] = data
                if not isinstance(train.dataset, Sized):
                    raise ValueError("Training data is not sized (e.g. has no length).")

                n_train = len(train.dataset)
                if full_args["virtual_batch"] > n_train:
                    full_args["virtual_batch"] = 16
                model = self.model_cls(**full_args)
                trainer = self._get_trainer(train=train, val=val, trial=trial)
                Path(trainer.log_dir).mkdir(exist_ok=True, parents=True)  # type: ignore
                trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
                pred = self._pred_loader(val.dataset.X, g=None)  # type: ignore
                logits = self._predict_logits(
                    trainer=trainer,
                    model=model,
                    loader=pred,
                )
                preds = logits.detach().cpu().numpy()
                if not hasattr(val.dataset, "y"):
                    raise AttributeError(
                        "Validation dataset missing target attribute `y`"
                    )
                y_true = Series(data=val.dataset.y.detach().cpu().numpy())  # type: ignore
                if self.is_classifier:
                    y_pred = Series(data=preds.argmax(axis=1))
                    y_prob = (
                        torch.softmax(torch.from_numpy(preds), dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    score = metric.get_scores(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
                    return score[metric.value]  # type: ignore
                    # and (metric is not ClassifierScorer.AUROC):
                    # if metric is not ClassifierScorer.AUROC:
                    #     preds = preds.argmax(axis=1)
                    # else:
                    #     if self.n_cls == 2:
                    #         preds = preds[:, 1]

                score = metric.tuning_score(y_true=y_true, y_pred=preds)
                return score
            except Exception as e:
                if self._uses_accelerator():
                    raise
                traceback.print_exc()
                print(f"Got error: {e}")
                if metric.higher_is_better():
                    return float(-np.inf)
                else:
                    return float(np.inf)
            finally:
                self._cleanup_after_fold()

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
        # completely arbitrary...
        n_cpu = multiprocessing.cpu_count()
        n_jobs = self.runtime.tuning_jobs(RuntimeComponent.Gandalf, max(1, n_cpu // 2))
        # n_jobs = 4 if os.environ.get("CC_CLUSTER") is None else 8
        return super().htune_optuna(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            metric=metric,
            n_trials=n_trials,
            verbosity=verbosity,
            n_jobs=n_jobs,
        )


if __name__ == "__main__":
    ...

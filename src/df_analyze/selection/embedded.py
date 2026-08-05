from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from random import randint, uniform
from typing import TYPE_CHECKING, Optional, cast
from warnings import warn

import jsonpickle
import numpy as np
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectFromModel

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
    from df_analyze.models.base import DfAnalyzeModel
from df_analyze.enumerables import EmbedSelectionModel
from df_analyze.models.lgbm import LightGBMClassifier, LightGBMRegressor
from df_analyze.models.linear import SGDClassifierSelector, SGDRegressorSelector
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.testing.datasets import TestDataset


@dataclass
class EmbedSelected:
    model: EmbedSelectionModel
    selected: list[str]
    scores: dict[str, float]
    is_classification: bool

    def to_markdown(self) -> str:
        fnames, scores = zip(*self.scores.items())
        scores = DataFrame(
            data=scores, index=Series(name="feature", data=fnames), columns=["score"]
        )
        if self.model is EmbedSelectionModel.LGBM:
            metric = "Importances"
        else:
            metric = "Coefficients"
        text = (
            "# Wrapper-Based Feature Selection Summary\n\n"
            f"Wrapper model:  {self.model.name}\n"
            "\n"
            f"## Selected Features\n\n"
            f"{self.selected}\n\n"
            f"## Selection scores ({metric}: Larger magnitude = More important)\n\n"
            f"{scores.to_markdown(floatfmt='0.3e')}"
        )
        return text

    def to_json(self) -> str:
        return str(jsonpickle.encode(self))

    @staticmethod
    def from_json(path: Path) -> EmbedSelected:
        return cast(EmbedSelected, jsonpickle.decode(path.read_text()))

    @staticmethod
    def random(
        ds: TestDataset, model: Optional[EmbedSelectionModel] = None
    ) -> EmbedSelected:
        model = model or EmbedSelectionModel.random()
        df = ds.load()
        x = df.drop(columns="target", errors="ignore")
        cols = x.columns.to_list()
        n_feat = randint(min(10, x.shape[1]), x.shape[1])
        selected = np.random.choice(cols, size=n_feat, replace=False).tolist()
        scores = {s: uniform(0, 1) for s in selected}
        return EmbedSelected(
            model=model,
            selected=selected,
            scores=scores,
            is_classification=ds.is_classification,
        )


def embed_select_features(
    prep_train: PreparedData,
    options: ProgramOptions,
) -> list[EmbedSelected]:
    y = prep_train.y
    g = prep_train.groups
    X_train = prep_train.X
    is_cls = options.is_classification

    if options.embed_select is None:
        return []

    embed_models = sorted(set([e for e in options.embed_select if e is not None]))
    models = []
    for embed_model in embed_models:
        if embed_model is EmbedSelectionModel.Linear:
            model = SGDClassifierSelector() if is_cls else SGDRegressorSelector()
        else:
            model = LightGBMClassifier() if is_cls else LightGBMRegressor()
        models.append(model)

    if prep_train.is_classification:
        metric = options.htune_cls_metric
    else:
        metric = options.htune_reg_metric
    model: DfAnalyzeModel
    results: list[EmbedSelected] = []
    for model, embed_model in zip(models, embed_models):
        try:
            model.htune_optuna(
                X_train=X_train,
                y_train=y,
                g_train=g,
                metric=metric,
                # Embedded selection has its own fixed search budget;
                # --htune-trials controls only final model tuning.
                n_trials=100,
                n_jobs=-1,
            )
            # `coefs` are floats if Linear, int32 if LGBM
            scores = np.ravel(
                model.tuned_model.coef_  # type: ignore
                if embed_model is EmbedSelectionModel.Linear
                else model.tuned_model.feature_importances_  # type: ignore
            )
            fscores = {feature: score for feature, score in zip(X_train.columns, scores)}

            selector = SelectFromModel(
                model.tuned_model,
                prefit=True,
            )
            idx = np.array(selector.get_support()).astype(bool)
            selected = X_train.loc[:, idx].columns.to_list()  # type: ignore
            result = EmbedSelected(
                model=embed_model,
                selected=selected,
                scores=fscores,
                is_classification=prep_train.is_classification,
            )
            results.append(result)
        except Exception as e:
            warn(
                "Got exception when attempting to perform embedded selection "
                f"via model '{model.__class__.__name__}'. Details:\n{e}\n"
                f"{traceback.format_exc()}"
            )

    return results

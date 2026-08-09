from __future__ import annotations

import traceback
from dataclasses import dataclass
from random import choice
from typing import TYPE_CHECKING, Optional
from warnings import warn

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.enumerables import FeatureSelection
from df_analyze.runtime.hardware import DeviceIntent
from df_analyze.selection.embedded import (
    EmbedSelected,
    EmbedSelectionModel,
    embed_select_features,
)
from df_analyze.selection.wrapper import WrapperSelected, wrap_select_features
from df_analyze.testing.datasets import TestDataset


def _raise_strict_cuda_failure(options: ProgramOptions, error: Exception) -> None:
    runtime = getattr(options, "runtime", None)
    if (
        runtime is not None
        and runtime.intent is DeviceIntent.CUDA
    ):
        raise error


@dataclass
class ModelSelected:
    embed_selected: Optional[list[EmbedSelected]]
    wrap_selected: Optional[WrapperSelected]

    @staticmethod
    def random(ds: TestDataset) -> ModelSelected:
        embeds = [
            None,
            [EmbedSelected.random(ds, model=EmbedSelectionModel.Linear)],
            [EmbedSelected.random(ds, model=EmbedSelectionModel.LGBM)],
            [
                EmbedSelected.random(ds, model=EmbedSelectionModel.Linear),
                EmbedSelected.random(ds, model=EmbedSelectionModel.LGBM),
            ],
        ]
        wraps = [WrapperSelected.random(ds) for _ in range(4)]
        embed_selected = choice(embeds)
        wrap_selected = choice([None, *wraps])
        return ModelSelected(embed_selected=embed_selected, wrap_selected=wrap_selected)


def model_select_features(
    prep_train: PreparedData,
    options: ProgramOptions,
) -> ModelSelected:
    embed_selected = None
    wrap_selected = None
    try:
        if (
            FeatureSelection.Embedded in options.feat_select
            and options.embed_select is not None
        ):
            embed_selected = embed_select_features(
                prep_train=prep_train, options=options
            )
    except Exception as e:
        _raise_strict_cuda_failure(options, e)
        warn(
            f"Got error when attempting embedded feature selection:\n{e}\n"
            f"{traceback.format_exc()}"
        )

    try:
        if (
            FeatureSelection.Wrapper in options.feat_select
            and options.wrapper_select is not None
        ):
            wrap_selected = wrap_select_features(prep_train=prep_train, options=options)
    except Exception as e:
        _raise_strict_cuda_failure(options, e)
        warn(
            f"Got error when attempting wrapped-based feature selection:\n{e}\n"
            f"{traceback.format_exc()}"
        )
    return ModelSelected(embed_selected=embed_selected, wrap_selected=wrap_selected)

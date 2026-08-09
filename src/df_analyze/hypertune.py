from __future__ import annotations

import json
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from random import choice, randint
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from warnings import warn

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
)

from df_analyze._constants import SEED, VAL_SIZE
from df_analyze._types import Classifier, CVMethod, EstimationMode, Estimator
from df_analyze.cli.cli import ProgramOptions
from df_analyze.legacy.src.objectives import (
    bagging_classifier_objective,
    dtree_classifier_objective,
    mlp_classifier_objective,
    rf_classifier_objective,
    svm_classifier_objective,
)
from df_analyze.models.dummy import DummyClassifier, DummyRegressor
from df_analyze.testing.datasets import TestDataset

if TYPE_CHECKING:
    from df_analyze.downsampling.containers import FeatureDownsampleResult
    from df_analyze.models.base import DfAnalyzeModel
import jsonpickle

from df_analyze.enumerables import ClassifierScorer, RegressorScorer
from df_analyze.hypertune_io import (
    PredArray,
    ProbArray,
    _deserialize_preds,
    _deserialize_probs,
    _preds_equal,
    _probs_equal,
    _serialize_preds,
    _serialize_probs,
)
from df_analyze.models.gandalf import GandalfEstimator
from df_analyze.models.kan import KANEstimator
from df_analyze.models.mlp import MLPEstimator
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.runtime.hardware import (
    RuntimeComponent,
    clear_fitted_model_state,
    device_reason_text,
    release_accelerator_memory,
)
from df_analyze.saving import add_fold_idx
from df_analyze.scoring import (
    sensitivity,
    specificity,
)
from df_analyze.selection.embedded import EmbedSelectionModel
from df_analyze.selection.filter import FilterSelected
from df_analyze.selection.models import ModelSelected

Splits = Iterable[Tuple[ndarray, ndarray]]

LR_SOLVER = "liblinear"
# MLP_LAYER_SIZES = [0, 8, 32, 64, 128, 256, 512]
MLP_LAYER_SIZES = [4, 8, 16, 32]
N_SPLITS = 5


@dataclass
class HtuneResult:
    selection: Literal["none", "assoc", "pred", "embed", "wrap"]
    selected_cols: list[str]
    embed_select_model: Optional[EmbedSelectionModel]
    model_cls: Type[DfAnalyzeModel]
    model: DfAnalyzeModel
    params: dict[str, Any]  # optuna best_params
    metric: Union[ClassifierScorer, RegressorScorer]
    score: float
    preds_test: PredArray
    preds_train: PredArray
    probs_test: ProbArray
    probs_train: ProbArray
    target: Optional[str] = None
    downsample_requested: str = "none"
    downsample_resolved: str = "none"
    n_downsampled_features: int = 0
    failure_reason: Optional[str] = None
    per_target_tuning_scores: dict[str, float] = field(default_factory=dict)

    @no_type_check  # Pyright mucks up the conditionals here...
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HtuneResult):
            return False
        preds_test_equal = _preds_equal(self.preds_test, other.preds_test)
        preds_train_equal = _preds_equal(self.preds_train, other.preds_train)
        probs_test_equal = _probs_equal(self.probs_test, other.probs_test)
        probs_train_equal = _probs_equal(self.probs_train, other.probs_train)

        ret = (
            self.selection == other.selection
            and sorted(self.selected_cols) == sorted(other.selected_cols)
            and self.model_cls == other.model_cls
            and self.embed_select_model == other.embed_select_model
            and
            # self.model == other.model and  # NOTE: do not check since weights
            self.params == other.params
            and round(self.score, 8) == round(other.score, 8)
            and self.metric == other.metric
            and preds_test_equal
            and preds_train_equal
            and probs_test_equal
            and probs_train_equal
            and self.target == other.target
            and self.downsample_requested == other.downsample_requested
            and self.downsample_resolved == other.downsample_resolved
            and self.n_downsampled_features == other.n_downsampled_features
            and self.per_target_tuning_scores == other.per_target_tuning_scores
        )
        return bool(ret)

    def to_row(self) -> DataFrame:
        selector = self.embed_select_model
        embed = "none" if selector is None else selector.value
        row = DataFrame(
            {
                "selection": self.selection,
                "embed_selector": embed,
                "model": self.model.shortname,
                "params": str(jsonpickle.encode(self.params)),
                "metric": self.metric.value,
                "score": self.score,
                "downsample_requested": self.downsample_requested,
                "downsample": self.downsample_resolved,
                "n_downsampled_features": self.n_downsampled_features,
                "failure_reason": self.failure_reason,
                "per_target_tuning_scores": json.dumps(
                    self.per_target_tuning_scores,
                    sort_keys=True,
                ),
            },
            index=[0],
        )
        if self.target is not None:
            row["target"] = self.target
        return row

    def to_preds_json(self) -> str:
        preds_kind_test, preds_test, _, preds_columns = _serialize_preds(self.preds_test)
        preds_kind_train, preds_train, preds_dtype, _ = _serialize_preds(self.preds_train)
        if preds_kind_test != preds_kind_train:
            raise ValueError(
                "Expected preds_test and preds_train to share container type."
            )
        preds_kind = preds_kind_test
        if (
            preds_kind == "dataframe"
            and isinstance(self.preds_test, DataFrame)
            and isinstance(self.preds_train, DataFrame)
            and [str(col) for col in self.preds_test.columns]
            != [str(col) for col in self.preds_train.columns]
        ):
            raise ValueError(
                "Expected DataFrame predictions to use matching train/test columns."
            )

        probs_kind_test, probs_test, probs_dtype = _serialize_probs(self.probs_test)
        probs_kind_train, probs_train, _ = _serialize_probs(self.probs_train)
        if probs_kind_test != probs_kind_train:
            raise ValueError(
                "Expected probs_test and probs_train to share container type."
            )
        probs_kind = probs_kind_test
        selector = (
            None if self.embed_select_model is None else self.embed_select_model.value
        )
        obj = {
            "selection": self.selection,
            "selected_cols": self.selected_cols,
            "embed_select_model": selector,
            "model_cls": self.model_cls.__name__,
            "params": str(self.params),
            "metric": self.metric.name,
            "score": self.score,
            "preds_test": preds_test,
            "preds_train": preds_train,
            "probs_test": probs_test,
            "probs_train": probs_train,
            "preds_dtype": preds_dtype,
            "preds_kind": preds_kind,
            "preds_columns": preds_columns,
            "probs_dtype": probs_dtype,
            "probs_kind": probs_kind,
            "target": self.target,
            "downsample_requested": self.downsample_requested,
            "downsample_resolved": self.downsample_resolved,
            "n_downsampled_features": self.n_downsampled_features,
            "per_target_tuning_scores": self.per_target_tuning_scores,
        }
        return json.dumps(obj)

    def to_preds(self) -> PredResult:
        embed_model = (
            self.embed_select_model.value
            if self.embed_select_model is not None
            else "none"
        )
        return PredResult(
            selection=self.selection,
            selected_cols=self.selected_cols,
            embed_select_model=embed_model,
            model_cls=str(self.model_cls.__name__),
            params=str(self.params),
            metric=self.metric.name,
            score=self.score,
            preds_test=self.preds_test,
            preds_train=self.preds_train,
            probs_test=self.probs_test,
            probs_train=self.probs_train,
            target=self.target,
            downsample_requested=self.downsample_requested,
            downsample_resolved=self.downsample_resolved,
            n_downsampled_features=self.n_downsampled_features,
            per_target_tuning_scores=self.per_target_tuning_scores,
        )

    @staticmethod
    def all_from_json(root: Path) -> HtuneResult:
        raise NotImplementedError(
            "This should be handled by src.df_analyze.hypertune.py::EvaluationResults"
        )

    @staticmethod
    def random(
        ds: TestDataset,
        y_test: Series,
        selection: Optional[Literal["none", "assoc", "pred", "embed", "wrap"]] = None,
        selected_cols: Optional[list[str]] = None,
        model_cls: Optional[Type[DfAnalyzeModel]] = None,
    ) -> HtuneResult:
        from df_analyze.models.base import DfAnalyzeModel
        from df_analyze.models.mlp import MLPEstimator

        n_cols = ds.shape[1]
        is_cls = ds.is_classification
        n_cls = 1 if not ds.is_classification else len(np.unique(y_test))
        n_samp = ds.shape[0]
        n_sel = randint(1, n_cols)

        selection = selection or choice(["none", "assoc", "pred", "embed", "wrap"])
        selected = (
            selected_cols
            or np.random.choice([f"f{i}" for i in range(n_cols)], size=n_sel).tolist()
        )
        model_cls = model_cls or DfAnalyzeModel.random(ds, n=1)
        if model_cls is MLPEstimator:
            prep = ds.prepared(load_cached=True)
            model = model_cls(num_classes=prep.num_classes)  # type: ignore
        else:
            model = model_cls()
        params = {**model.fixed_args, **model.default_args}
        # AUROC is handled automatically and changed to BalancedAcc with a warning
        metric = ClassifierScorer.random() if is_cls else RegressorScorer.random()
        score = (
            np.random.uniform(0, 1) if ds.is_classification else np.random.uniform(0, 10)
        )
        n_samp_test = len(y_test)
        n_samp_train = n_samp - n_samp_test
        if is_cls:
            probs_train = softmax(
                np.random.standard_normal([n_samp_train, n_cls]), axis=1
            )
            probs_test = softmax(np.random.standard_normal([n_samp_test, n_cls]), axis=1)
            preds_test = Series(np.argmax(probs_test, axis=1))
            preds_train = Series(np.argmax(probs_train, axis=1))
        else:
            probs_test = None
            probs_train = None
            preds_test = Series(np.random.standard_normal([n_samp_test]))
            preds_train = Series(np.random.standard_normal([n_samp_train]))

        return HtuneResult(
            selection=selection,  # type: ignore
            selected_cols=selected,
            embed_select_model=EmbedSelectionModel.random(),
            model_cls=model_cls,
            model=model,
            params=params,
            metric=metric,
            score=score,
            preds_test=preds_test,
            preds_train=preds_train,
            probs_test=probs_test,
            probs_train=probs_train,
        )


@dataclass
class PredResult:
    selection: Literal["none", "assoc", "pred", "embed", "wrap"]
    selected_cols: list[str]
    embed_select_model: str
    model_cls: str
    params: str  # optuna best_params
    metric: str
    score: float
    preds_test: PredArray
    preds_train: PredArray
    probs_test: ProbArray
    probs_train: ProbArray
    target: Optional[str] = None
    downsample_requested: str = "none"
    downsample_resolved: str = "none"
    n_downsampled_features: int = 0
    per_target_tuning_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """
    df: DataFrame
        Has columns: "model", "selection"
    """

    df: DataFrame
    X_train: DataFrame
    y_train: Union[Series, DataFrame]
    X_test: DataFrame
    y_test: Union[Series, DataFrame]
    results: list[HtuneResult]
    is_classification: bool
    per_target_df: Optional[DataFrame] = None

    @staticmethod
    def random(ds: TestDataset, options: ProgramOptions) -> EvaluationResults:
        from df_analyze.models.base import DfAnalyzeModel

        prep = ds.prepared(load_cached=True)
        prep_train, prep_test = prep.split()

        # Build up list of results with random HtuneResult objects
        selectors = ["assoc", "pred", "embed", "wrap"]
        n_select = np.random.randint(1, len(selectors))
        selected = np.random.choice(selectors, replace=False, size=n_select).tolist()
        selected = ["none"] + selected

        n_model = np.random.randint(2, 5, size=1).item()
        dummy = DummyClassifier if ds.is_classification else DummyRegressor
        rands = DfAnalyzeModel.random(ds, n=n_model)
        models_clses = list(set([dummy, *rands]))

        features = prep.X.columns.to_list()

        results = []
        dfs = []
        for selection in selected:
            for model_cls in models_clses:
                p = prep.X.shape[1]
                n_sel = np.random.randint(1, p)
                selected_cols = np.random.choice(features, size=n_sel, replace=False)
                selected_cols = selected_cols.tolist()
                result = HtuneResult.random(
                    ds=ds,
                    y_test=prep_test.y,
                    selection=selection,
                    selected_cols=selected_cols,
                    model_cls=model_cls,
                )
                df = result.to_row()
                results.append(result)
                dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return EvaluationResults(
            df=df,
            X_train=prep_train.X,
            y_train=prep_train.y,
            X_test=prep_test.X,
            y_test=prep_test.y,
            results=results,
            is_classification=ds.is_classification,
        )

    def hp_table(self, fold_idx: Optional[int] = None) -> DataFrame:
        dfs = []
        for res in self.results:
            dfs.append(res.to_row())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if fold_idx is not None:
            df["test_idx"] = fold_idx
        return df

    def wide_table(
        self, valset: Literal["cv_mean", "trainset", "holdout"] = "cv_mean"
    ) -> DataFrame:
        cols = ["trainset", "holdout", "cv_mean"]
        cols.remove(valset)
        col = valset
        idx = ~self.df.mean(axis=1, numeric_only=True).isna()
        index_cols = ["model", "selection", "embed_selector"]
        if "downsample" in self.df.columns:
            if "downsample_requested" in self.df.columns:
                index_cols.append("downsample_requested")
            index_cols.extend(["downsample", "n_downsampled_features"])
        if "target" in self.df.columns:
            index_cols.append("target")
        if "final_cv_folds" in self.df.columns:
            index_cols.append("final_cv_folds")
        df = (
            self.df.loc[idx]
            .drop(columns=cols)
            .pivot_table(
                columns="metric",
                values=col,
                index=index_cols,
                aggfunc="mean",
            )
            .reset_index()
        )
        sorters = ["acc"] if self.is_classification else ["multi-nmae", "mae"]
        for sorter in sorters:
            if sorter in df.columns:
                return df.sort_values(by=sorter, ascending=not self.is_classification)
        return df

    def to_markdown(self) -> str:
        df_train = self.wide_table("trainset")
        df_hold = self.wide_table("holdout")
        df_fold = self.wide_table("cv_mean")
        tab_train = df_train.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_hold = df_hold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        tab_fold = df_fold.to_markdown(tablefmt="simple", floatfmt="0.3f", index=False)
        if "final_cv_folds" not in self.df.columns:
            raise ValueError("Evaluation results are missing `final_cv_folds`.")
        fold_counts = (
            pd.to_numeric(self.df["final_cv_folds"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(fold_counts) == 0:
            raise ValueError("Evaluation results contain no valid final CV fold count.")
        fold_heading = (
            f"{int(fold_counts[0])}-fold" if len(fold_counts) == 1 else "adaptive-fold"
        )
        text = (
            "# Final Model Performances\n\n"
            "## Training set performance\n\n"
            f"{tab_train}\n\n"
            "## Holdout set performance\n\n"
            f"{tab_hold}\n\n"
            f"## {fold_heading} performance on holdout set\n\n"
            f"{tab_fold}\n\n"
        )
        return text

    def per_target_long_table(self) -> Optional[DataFrame]:
        if self.per_target_df is not None and len(self.per_target_df) > 0:
            return self.per_target_df.copy()
        if "target" in self.df.columns:
            return self.df.copy()
        return None

    def to_json(self) -> str:
        # obj = {
        #     "df": self.df.to_json()
        #     "X_train": DataFrame
        #     "y_train": Series
        #     "X_test": DataFrame
        #     "y_test": Series
        #     "results": list[HtuneResult]
        #     "is_classification": bool
        # }
        # raise NotImplementedError()
        return str(jsonpickle.encode(self))

    def save(self, root: Path, fold_idx: Optional[int]) -> None:
        fix = partial(add_fold_idx, fold_idx=fold_idx)
        self.df.to_csv(fix(root / "performance_long_table.csv"), index=False)
        if self.per_target_df is not None and len(self.per_target_df) > 0:
            self.per_target_df.to_csv(
                fix(root / "performance_long_table_per_target.csv"),
                index=False,
            )
        self.X_train.to_csv(fix(root / "X_train.csv"), index=False)
        self.X_test.to_csv(fix(root / "X_test.csv"), index=False)
        if isinstance(self.y_train, DataFrame):
            self.y_train.to_csv(fix(root / "y_train.csv"), index=True)
        else:
            self.y_train.to_frame().to_csv(fix(root / "y_train.csv"), index=True)
        if isinstance(self.y_test, DataFrame):
            self.y_test.to_csv(fix(root / "y_test.csv"), index=True)
        else:
            self.y_test.to_frame().to_csv(fix(root / "y_test.csv"), index=True)
        results_json = ", ".join([result.to_preds_json() for result in self.results])
        results_json = "{" + '"predictions": [' + results_json + "]}"
        remain = {
            "results": self.results,
            "is_classification": self.is_classification,
        }
        externalized: list[tuple[DfAnalyzeModel, Mapping[str, Any]]] = []
        try:
            for index, result in enumerate(self.results):
                externalize = getattr(result.model, "externalize_fitted_models", None)
                if callable(externalize):
                    directory = fix(root / f"tabpfn_fitted_model_{index}")
                    originals = externalize(directory)
                    externalized.append((result.model, originals))
            try:
                enc = str(jsonpickle.encode(remain, unpicklable=True))
                fix(root / "eval_htune_results_jsonpickle.json").write_text(enc)
            except TypeError:
                enc = str(jsonpickle.encode(remain, unpicklable=True, fail_safe=str))
                fix(root / "eval_htune_results_jsonpickle.json").write_text(enc)
        finally:
            for model, originals in externalized:
                model.restore_externalized_models(originals)
        try:
            fix(root / "prediction_results.json").write_text(results_json)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print(
                "Error saving manual results .json. Information above.", file=sys.stderr
            )

    @classmethod
    def load(cls, root: Path) -> EvaluationResults:
        # TODO: handle fold_idx
        df = pd.read_csv(root / "performance_long_table.csv")
        per_target_path = root / "performance_long_table_per_target.csv"
        per_target_df = pd.read_csv(per_target_path) if per_target_path.exists() else None
        for results_df in (df, per_target_df):
            if results_df is not None and "failure_reason" in results_df.columns:
                reasons = results_df["failure_reason"].astype(object)
                results_df["failure_reason"] = reasons.where(reasons.notna(), None)
        X_train = pd.read_csv(root / "X_train.csv", engine="python")
        X_test = pd.read_csv(root / "X_test.csv", engine="python")
        df_y_tr = pd.read_csv(root / "y_train.csv", index_col=0)
        df_y_test = pd.read_csv(root / "y_test.csv", index_col=0)

        if df_y_tr.shape[1] == 1:
            y_train: Union[Series, DataFrame] = Series(
                data=df_y_tr.values.ravel(),
                name=df_y_tr.columns.to_list()[0],
            )
        else:
            y_train = df_y_tr.reset_index(drop=True)
        if df_y_test.shape[1] == 1:
            y_test: Union[Series, DataFrame] = Series(
                data=df_y_test.values.ravel(),
                name=df_y_test.columns.to_list()[0],
            )
        else:
            y_test = df_y_test.reset_index(drop=True)
        enc = (root / "eval_htune_results_jsonpickle.json").read_text()
        remain = cast(dict[str, Any], jsonpickle.decode(enc))
        loaded_results = cast(list[HtuneResult], remain["results"])
        for result in loaded_results:
            load_externalized = getattr(result.model, "load_externalized_models", None)
            if callable(load_externalized):
                load_externalized(root)
        return EvaluationResults(
            df=df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            results=loaded_results,
            is_classification=remain["is_classification"],
            per_target_df=per_target_df,
        )

    @classmethod
    def load_preds(cls, root: Path) -> list[PredResult]:
        path = root / "prediction_results.json"
        results: dict[Literal["predictions"], list[dict[str, Any]]] = json.loads(
            path.read_text()
        )
        pred_results = []
        for result in results["predictions"]:
            preds_test = _deserialize_preds(result, "preds_test")
            preds_train = _deserialize_preds(result, "preds_train")
            probs_test = _deserialize_probs(result, "probs_test")
            probs_train = _deserialize_probs(result, "probs_train")
            if (probs_test is None) != (probs_train is None):
                probs_test = None
                probs_train = None

            pred_result = PredResult(
                selection=result["selection"],
                selected_cols=result["selected_cols"],
                embed_select_model=result["embed_select_model"],
                model_cls=result["model_cls"],
                params=result["params"],
                metric=result["metric"],
                score=result["score"],
                preds_test=preds_test,
                preds_train=preds_train,
                probs_test=probs_test,
                probs_train=probs_train,
                target=result.get("target"),
                downsample_requested=result.get("downsample_requested", "none"),
                downsample_resolved=result.get("downsample_resolved", "none"),
                n_downsampled_features=int(result.get("n_downsampled_features", 0)),
                per_target_tuning_scores={
                    str(target): float(score)
                    for target, score in result.get(
                        "per_target_tuning_scores", {}
                    ).items()
                },
            )
            pred_results.append(pred_result)
        return pred_results


def _get_cols(
    selection: Union[Literal["none", "assoc", "pred", "embed", "wrap"], str],
    selected: Optional[list[str]],
) -> Union[list[str], slice]:
    if selection == "none" or (selected is None):
        return slice(None)

    if selection == "assoc":
        # dealing with level-specific metrics
        selected = sorted(set([re.sub("__.*", "", str(c)) for c in selected]))
    return selected


def _get_splits(
    prep_train: PreparedData,
    prep_test: PreparedData,
    selection: str,
    cols: Union[list[str], slice],
) -> tuple[DataFrame, DataFrame]:
    if selection == "none" or isinstance(cols, slice):
        X_train = prep_train.X
        X_test = prep_test.X
    else:
        requested = {str(col) for col in cols}

        def resolved_columns(prepared: PreparedData) -> list[str]:
            lineage = getattr(prepared, "feature_lineage", {}) or {}
            return [
                str(column)
                for column in prepared.X.columns
                if str(column) in requested
                or str(lineage.get(str(column), str(column))) in requested
            ]

        train_cols = resolved_columns(prep_train)
        test_cols = resolved_columns(prep_test)
        if train_cols != test_cols:
            raise ValueError(
                "Selected feature columns do not resolve identically in the "
                "training and holdout matrices."
            )
        if len(train_cols) == 0:
            raise ValueError(
                "None of the selected features could be resolved in the prepared "
                "model matrix."
            )
        X_train = prep_train.X.loc[:, train_cols]
        X_test = prep_test.X.loc[:, test_cols]

    return X_train, X_test


def _new_model(
    model_cls: Type[DfAnalyzeModel],
    prepared: PreparedData,
) -> DfAnalyzeModel:
    if model_cls in (MLPEstimator, KANEstimator, GandalfEstimator):
        return model_cls(num_classes=prepared.num_classes)  # type: ignore
    return model_cls()  # type: ignore


def _record_model_device(
    options: ProgramOptions,
    model_name: str,
    selection: str,
    runtime,
    component: RuntimeComponent,
    *,
    attempt: int,
    stage: str,
    error: Optional[BaseException] = None,
) -> None:
    record: dict[str, object] = {
        "fold": getattr(options, "_runtime_current_fold", None),
        "model": model_name,
        "selection": selection,
        "attempt": attempt,
        "stage": stage,
        **runtime.decision_for(component).to_dict(),
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    audit = getattr(options, "_runtime_model_audit", [])
    audit.append(record)
    options._runtime_model_audit = audit


def _release_fitted_model_state(
    model: DfAnalyzeModel,
    component: RuntimeComponent,
    *,
    force: bool = False,
) -> None:
    """Drop fitted estimators after predictions/parameters have been captured."""
    runtime = getattr(model, "runtime", None)
    resolved = "cpu"
    if runtime is not None:
        try:
            resolved = runtime.decision_for(component).resolved
        except Exception:
            resolved = "cpu"
    if not force and resolved not in {"cuda", "mps"}:
        return
    try:
        clear_fitted_model_state(model)
        cleanup = getattr(model, "_cleanup_after_fold", None)
        if callable(cleanup):
            cleanup()
    except Exception as error:
        warn(
            "Could not completely release fitted model state for "
            f"{getattr(model, 'shortname', type(model).__name__)}: {error}"
        )
    if resolved in {"cuda", "mps"}:
        release_accelerator_memory()


def _tune_and_evaluate_model(
    model: DfAnalyzeModel,
    X_train: DataFrame,
    X_test: DataFrame,
    prep_train: PreparedData,
    prep_test: PreparedData,
    options: ProgramOptions,
    metric,
    downsample_result: Optional[FeatureDownsampleResult],
):
    tuning_rows = (
        downsample_result.tuning_rows
        if downsample_result is not None and downsample_result.tuning_rows
        else list(range(len(X_train)))
    )
    X_tune = X_train.iloc[tuning_rows]
    y_tune = prep_train.y.iloc[tuning_rows]
    g_tune = None if prep_train.groups is None else prep_train.groups.iloc[tuning_rows]
    study = model.htune_optuna(
        X_train=X_tune,
        y_train=y_tune,
        g_train=g_tune,
        n_trials=options.htune_trials,
        metric=metric,
        n_jobs=-1,
        verbosity=optuna.logging.ERROR,
    )
    if len(tuning_rows) < len(X_train):
        model.refit_tuned(
            X=X_train,
            y=prep_train.y,
            g=prep_train.groups,
            tuned_args=study.best_params,
        )
    evaluation = model.htune_eval(
        X_train=X_train,
        y_train=prep_train.y,
        g_train=prep_train.groups,
        X_test=X_test,
        y_test=prep_test.y,
        g_test=prep_test.groups,
        seed=options.seed,
    )
    return study, evaluation


def evaluate_tuned(
    prepared: PreparedData,
    prep_train: PreparedData,
    prep_test: PreparedData,
    assoc_filtered: Optional[FilterSelected],
    pred_filtered: Optional[FilterSelected],
    model_selected: ModelSelected,
    options: ProgramOptions,
    downsample_result: Optional[FeatureDownsampleResult] = None,
) -> EvaluationResults:
    model_cls: Union[Type[DfAnalyzeModel], Type[MLPEstimator], Type[KANEstimator]]
    results: list[HtuneResult]
    dfs: list[DataFrame]
    per_target_dfs: list[DataFrame]

    selections: dict[str, Optional[list[str]]] = {"none": None}
    if assoc_filtered is not None:
        selections["assoc"] = assoc_filtered.selected
    if pred_filtered is not None:
        selections["pred"] = pred_filtered.selected

    embed_models: dict[str, EmbedSelectionModel] = {}
    if model_selected.embed_selected is not None:
        for selected in model_selected.embed_selected:
            selections[f"embed_{selected.model.value}"] = selected.selected
            embed_models[f"embed_{selected.model.value}"] = selected.model
    if model_selected.wrap_selected is not None:
        selections["wrap"] = model_selected.wrap_selected.selected

    if prepared.is_classification:
        metric = options.htune_cls_metric
    else:
        metric = options.htune_reg_metric
    n_downsampled_features = (
        prep_train.X.shape[1]
        if downsample_result is None
        else downsample_result.n_features_out
    )

    dfs, results, per_target_dfs = [], [], []
    for model_cls in options.models:
        for selection, cols in selections.items():
            selected_cols = _get_cols(selection=selection, selected=cols)
            X_train, X_test = _get_splits(prep_train, prep_test, selection, selected_cols)
            is_tabpfn_model = prep_train._is_tabpfn_model(model_cls)
            if is_tabpfn_model:
                X_train = prep_train.model_matrix(model_cls, selected_cols)
                X_test = prep_test.model_matrix(model_cls, selected_cols)
            if X_train.empty or X_test.empty:
                raise ValueError(
                    f"Error when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )
            if not is_tabpfn_model and (
                X_train.isna().any().any() or X_test.isna().any().any()
            ):
                raise ValueError(
                    f"Got NaNs when subsetting features for model '{model_cls.shortname}'. Got:\n"
                    f"cols: {cols}\n"
                    f"selected_cols: {selected_cols}\n"
                )

            is_embed = "embed" in selection
            embed_model = embed_models[selection] if is_embed else None
            model_runtime = options.runtime.for_task(
                len(X_train),
                X_train.shape[1],
                n_queries=len(X_train),
            )
            component = getattr(model_cls, "runtime_component", RuntimeComponent.Sklearn)
            model = _new_model(model_cls, prepared)
            attempt = 0
            release_model_state = False
            try:
                while True:
                    attempt += 1
                    decision = model_runtime.decision_for(component)
                    print(
                        f"[device] {model.shortname} / {selection}: "
                        f"{decision.resolved.upper()} "
                        f"({len(X_train)} rows x {X_train.shape[1]} features; "
                        f"{device_reason_text(decision)})"
                    )
                    _record_model_device(
                        options,
                        model.shortname,
                        selection,
                        model_runtime,
                        component,
                        attempt=attempt,
                        stage="started",
                    )
                    try:
                        model.set_runtime(model_runtime)
                        print(f"Tuning {model.longname} for selection={selection}")
                        study, evaluation = _tune_and_evaluate_model(
                            model=model,
                            X_train=X_train,
                            X_test=X_test,
                            prep_train=prep_train,
                            prep_test=prep_test,
                            options=options,
                            metric=metric,
                            downsample_result=downsample_result,
                        )
                        (
                            df,
                            preds_train,
                            preds_test,
                            probs_train,
                            probs_test,
                            df_target,
                        ) = evaluation
                        final_decision = model_runtime.decision_for(component)
                        _record_model_device(
                            options,
                            model.shortname,
                            selection,
                            model_runtime,
                            component,
                            attempt=attempt,
                            stage="completed",
                        )
                        if final_decision.resolved != decision.resolved:
                            print(
                                f"[device] {model.shortname} / {selection}: "
                                f"{final_decision.resolved.upper()} "
                                f"({device_reason_text(final_decision)})"
                            )
                        break
                    except Exception as error:
                        _record_model_device(
                            options,
                            model.shortname,
                            selection,
                            model_runtime,
                            component,
                            attempt=attempt,
                            stage="failed",
                            error=error,
                        )
                        raise
                result = HtuneResult(
                    selection="embed" if is_embed else selection,  # type: ignore
                    selected_cols=X_train.columns.to_list(),
                    embed_select_model=embed_model,
                    model_cls=model_cls,
                    model=model,
                    params=study.best_params,
                    metric=metric,
                    score=study.best_value,
                    preds_train=preds_train,
                    preds_test=preds_test,
                    probs_train=probs_train,
                    probs_test=probs_test,
                    downsample_requested=(
                        "none"
                        if downsample_result is None
                        else downsample_result.requested_method
                    ),
                    downsample_resolved=(
                        "none"
                        if downsample_result is None
                        else downsample_result.resolved_method
                    ),
                    n_downsampled_features=n_downsampled_features,
                    per_target_tuning_scores=dict(model.per_target_tuning_scores),
                )
                results.append(result)
                if df_target is not None and len(df_target) > 0:
                    df_target = df_target.copy()
                    if prepared.is_classification and isinstance(prep_train.labels, dict):
                        positive_classes = {}
                        for target_name in prep_train.target_cols:
                            mapping = prep_train.labels.get(target_name)
                            if isinstance(mapping, dict) and len(mapping) == 2:
                                positive_classes[str(target_name)] = mapping.get(1)
                        df_target["positive_class"] = (
                            df_target["target"].astype(str).map(positive_classes)
                        )
                    df_target["model"] = model.shortname
                    df_target["selection"] = selection
                    df_target["embed_selector"] = (
                        embed_model.value if embed_model is not None else "none"
                    )
                    df_target["downsample_requested"] = result.downsample_requested
                    df_target["downsample"] = result.downsample_resolved
                    df_target["n_downsampled_features"] = result.n_downsampled_features
                    per_target_dfs.append(df_target)
            except Exception as e:
                if prepared.is_classification:
                    nulls = Series(ClassifierScorer.null_scores(), name="metric")
                else:
                    nulls = Series(RegressorScorer.null_scores(), name="metric")
                warn(
                    f"\nGot exception when trying to tune and evaluate {model.shortname}:\n{e}\n"
                    f"{traceback.format_exc()}"
                )
                df = DataFrame(
                    {"trainset": nulls, "holdout": nulls, "cv_mean": nulls}
                ).reset_index()
                if "index" in df.columns:
                    df = df.rename(columns={"index": "metric"})
                result = HtuneResult(
                    selection="embed" if is_embed else selection,  # type: ignore
                    selected_cols=X_train.columns.to_list(),
                    embed_select_model=embed_model,
                    model_cls=model_cls,
                    model=model,
                    params={},
                    score=np.nan,
                    metric=metric,
                    preds_train=Series(),
                    preds_test=Series(),
                    probs_train=None,
                    probs_test=None,
                    downsample_requested=(
                        "none"
                        if downsample_result is None
                        else downsample_result.requested_method
                    ),
                    downsample_resolved=(
                        "none"
                        if downsample_result is None
                        else downsample_result.resolved_method
                    ),
                    n_downsampled_features=n_downsampled_features,
                    failure_reason=str(e),
                )
                release_model_state = True
                results.append(result)
                if isinstance(prep_test.y, DataFrame):
                    rows = []
                    for target_name in prep_test.y.columns:
                        for metric_name, null_val in nulls.items():
                            rows.append(
                                {
                                    "target": str(target_name),
                                    "metric": str(metric_name),
                                    "trainset": float(null_val),
                                    "holdout": float(null_val),
                                    "cv_mean": float(null_val),
                                    "model": model.shortname,
                                    "selection": selection,
                                    "embed_selector": (
                                        embed_model.value
                                        if embed_model is not None
                                        else "none"
                                    ),
                                    "downsample_requested": result.downsample_requested,
                                    "downsample": result.downsample_resolved,
                                    "n_downsampled_features": (
                                        result.n_downsampled_features
                                    ),
                                }
                            )
                    if len(rows) > 0:
                        per_target_dfs.append(DataFrame(rows))
            df["model"] = model.shortname
            df["selection"] = selection
            df["embed_selector"] = (
                embed_model.value if embed_model is not None else "none"
            )
            df["downsample_requested"] = (
                "none"
                if downsample_result is None
                else downsample_result.requested_method
            )
            df["downsample"] = (
                "none" if downsample_result is None else downsample_result.resolved_method
            )
            df["n_downsampled_features"] = n_downsampled_features
            dfs.append(df)
            if release_model_state:
                _release_fitted_model_state(
                    model,
                    component,
                    force=result.failure_reason is not None,
                )
    df = pd.concat(dfs, axis=0, ignore_index=True)
    per_target_df = (
        pd.concat(per_target_dfs, axis=0, ignore_index=True)
        if len(per_target_dfs) > 0
        else None
    )
    return EvaluationResults(
        df=df,
        X_train=prep_train.X,
        X_test=prep_test.X,
        y_train=prep_train.y,
        y_test=prep_test.y,
        results=results,
        is_classification=prepared.is_classification,
        per_target_df=per_target_df,
    )


@dataclass(init=True, repr=True, eq=True, frozen=True)
class HtuneResultLegacy:
    estimator: Estimator
    mode: EstimationMode
    n_trials: int
    cv_method: CVMethod
    val_acc: float = np.nan
    best_params: Dict = field(default_factory=dict)


def train_val_splits(
    df: DataFrame, is_classification: bool, val_size: float = VAL_SIZE
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Wrapper around `sklearn.model_selection.train_test_split` to return splits as `DataFrame`s
    instead of numpy arrays.

    Parameters
    ----------
    df: DataFrame
        Data with target in column named "target".

    mode: Literal["classify", "regress"]
        What kind of model we are running.

    val_size: float = 0.2
        Percent of data to reserve for validation

    Returns
    -------
    splits: [X_train, X_val, y_train, y_val]
    """
    train, val = train_test_split(
        df, test_size=val_size, random_state=SEED, shuffle=True, stratify=df.target
    )
    X_train = train.drop(columns="target")
    X_val = val.drop(columns="target")
    y_train = train.target
    y_val = val.target
    if is_classification:
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
    return X_train, X_val, y_train, y_val


def cv_desc(cv_method: CVMethod, n_folds: Optional[int] = None) -> str:
    """Helper for logging a readable description of the CVMethod to stdout"""
    if isinstance(cv_method, int):
        return f"stratified {cv_method}-fold"
    if isinstance(cv_method, float):  # stratified holdout
        if cv_method <= 0 or cv_method >= 1:
            raise ValueError("Holdout test_size must be in (0, 1)")
        perc = int(100 * cv_method)
        return f"stratified {perc}% holdout"
    cv_method = str(cv_method).lower()  # type: ignore

    if n_folds is None:
        n_folds = 5
    if cv_method == "loocv":
        return "LOOCV"
    if cv_method in ["kfold", "k-fold"]:
        return f"{n_folds}-fold"
    if cv_method == "mc":
        return "stratified Monte-Carlo (20 random 20%-sized test sets)"
    raise ValueError(f"Invalid `cv_method`: {cv_method}")


def package_classifier_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=np.mean(scores["test_sens"]),
        spec=np.mean(scores["test_spec"]),
        acc_sd=np.std(scores["test_acc"], ddof=1),
        auc_sd=np.std(scores["test_auroc"], ddof=1),
        sens_sd=np.std(scores["test_sens"], ddof=1),
        spec_sd=np.std(scores["test_spec"], ddof=1),
    )
    if not log:
        return result

    acc_mean = float(np.mean(scores["test_acc"]))
    auc_mean = float(np.mean(scores["test_auroc"]))
    sens_mean = float(np.mean(scores["test_sens"]))
    spec_mean = float(np.mean(scores["test_spec"]))
    acc_sd = float(np.std(scores["test_acc"], ddof=1))
    auc_sd = float(np.std(scores["test_auroc"], ddof=1))
    sens_sd = float(np.std(scores["test_sens"], ddof=1))
    spec_sd = float(np.std(scores["test_spec"], ddof=1))
    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"Accuracy:           μ = {np.round(acc_mean, 3):0.3f} (sd = {np.round(acc_sd, 4):0.4f})")  # noqa
    print(f"AUC:                μ = {np.round(auc_mean, 3):0.3f} (sd = {np.round(auc_sd, 4):0.4f})")  # noqa
    print(f"Sensitivity:        μ = {np.round(sens_mean, 3):0.3f} (sd = {np.round(sens_sd, 4):0.4f})")  # noqa
    print(f"Specificity:        μ = {np.round(spec_mean, 3):0.3f} (sd = {np.round(spec_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_classifier_scores(
    y_test: ndarray,
    y_pred: ndarray,
    y_score: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    sens = sensitivity(y_test, y_pred)
    spec = specificity(y_test, y_pred)
    percent = int(100 * float(cv_method))
    scores = dict(
        test_accuracy=np.array([acc]).ravel(), test_roc_auc=np.array([auc]).ravel()
    )
    result = dict(
        htuned=htuned,
        cv_method=cv_method,
        acc=np.mean(scores["test_acc"]),
        auc=np.mean(scores["test_auroc"]),
        sens=sens,
        spec=spec,
        acc_sd=np.nan,
        auc_sd=np.nan,
        sens_sd=np.nan,
        spec_sd=np.nan,
    )
    if not log:
        return result
    print(f"Testing validation: {percent}% holdout")
    print(
        f"          Accuracy: μ = {np.round(acc, 3):0.3f} (sd = {np.round(acc, 4):0.4f})"
    )
    print(
        f"               AUC: μ = {np.round(auc, 3):0.3f} (sd = {np.round(auc, 4):0.4f})"
    )
    return result


def package_regressor_cv_scores(
    scores: Dict[str, ndarray],
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    result = dict(
        htuned=htuned,
        cv_method=htuned.cv_method,
        mae=np.mean(scores["test_mae"]),
        msqe=np.mean(scores["test_msqe"]),
        mdae=np.mean(scores["test_mdae"]),
        mape=np.mean(scores["test_mape"]),
        r2=np.mean(scores["test_r2"]),
        var_exp=np.mean(scores["test_var-exp"]),
        mae_sd=np.std(scores["test_mae"], ddof=1),
        msqe_sd=np.std(scores["test_msqe"], ddof=1),
        mdae_sd=np.std(scores["test_mdae"], ddof=1),
        mape_sd=np.std(scores["test_mape"], ddof=1),
        r2_sd=np.std(scores["test_r2"], ddof=1),
        var_exp_sd=np.std(scores["test_var-exp"], ddof=1),
    )
    if not log:
        return result

    mae = np.mean(scores["test_mae"])
    msqe = np.mean(scores["test_msqe"])
    mdae = np.mean(scores["test_mdae"])
    # mape = np.mean(scores["test_mape"])
    r2 = np.mean(scores["test_r2"])
    var_exp = np.mean(scores["test_var-exp"])
    mae_sd = np.std(scores["test_mae"], ddof=1)
    msqe_sd = np.std(scores["test_msqe"], ddof=1)
    mdae_sd = np.std(scores["test_mdae"], ddof=1)
    # mape_sd = np.std(scores["test_mape"], ddof=1)
    r2_sd = np.std(scores["test_r2"], ddof=1)
    var_exp_sd = np.std(scores["test_var exp"], ddof=1)

    desc = cv_desc(cv_method)
    # fmt: off
    print(f"Testing validation: {desc}")
    print(f"MAE             μ = {np.round(mae, 3):0.3f} (sd = {np.round(mae_sd, 4):0.4f})")  # noqa
    print(f"MSqE:           μ = {np.round(msqe, 3):0.3f} (sd = {np.round(msqe_sd, 4):0.4f})")  # noqa
    print(f"Median Abs Err: μ = {np.round(mdae, 3):0.3f} (sd = {np.round(mdae_sd, 4):0.4f})")  # noqa
    # print(f"MAPE:           μ = {np.round(mape, 3):0.3f} (sd = {np.round(mape_sd, 4):0.4f})")  # noqa
    print(f"R-squared:      μ = {np.round(r2, 3):0.3f} (sd = {np.round(r2_sd, 4):0.4f})")  # noqa
    print(f"Var explained:  μ = {np.round(var_exp, 3):0.3f} (sd = {np.round(var_exp_sd, 4):0.4f})")  # noqa
    # fmt: on
    return result


def package_regressor_scores(
    y_test: ndarray,
    y_pred: ndarray,
    htuned: HtuneResultLegacy,
    cv_method: CVMethod,
    log: bool = False,
) -> Dict[str, Any]:
    scores = {
        "test_MAE": mean_absolute_error(y_test, y_pred),
        "test_MSqE": mean_squared_error(y_test, y_pred),
        "test_MdAE": median_absolute_error(y_test, y_pred),
        "test_MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "test_R2": r2_score(y_test, y_pred),
        "test_Var exp": explained_variance_score(y_test, y_pred),
    }
    if log:
        percent = int(100 * float(cv_method))
        print(f"Testing validation: {percent}% holdout")
    with warnings.catch_warnings():
        # suppress warnings for taking std of length 1 ndarray:
        # RuntimeWarning: Degrees of freedom <= 0 for slice
        # RuntimeWarning: invalid value encountered in double_scalars
        warnings.simplefilter("ignore", RuntimeWarning)
        return package_regressor_cv_scores(scores, htuned, cv_method, log)


"""See Optuna docs (https://optuna.org/#code_ScikitLearn) for the motivation behond the closures
below. Currently I am using closures, but this might be a BAD IDEA in parallel contexts. In any
case, they do seem to suggest this is OK https://optuna.readthedocs.io/en/stable/faq.html
#how-to-define-objective-functions-that-have-own-arguments, albeit by using classes or lambdas. """


def hypertune_classifier(
    classifier: Classifier,
    X_train: DataFrame,
    y_train: DataFrame,
    n_trials: int = 200,
    cv_method: CVMethod = 5,
    verbosity: int = optuna.logging.ERROR,
) -> HtuneResultLegacy:
    """Core function. Uses Optuna base TPESampler (Tree-Parzen Estimator Sampler) to perform
    Bayesian hyperparameter optimization via Gaussian processes on the classifier specified in
    `classifier`.

    Parameters
    ----------
    classifier: Classifier
        Classifier to tune.

    X_train: DataFrame
        DataFrame with no target value (features only). Shape (n_samples, n_features)

    y_train: DataFrame
        Target values. Shape (n_samples,).

    n_trials: int = 200
        Number of trials to use with Optuna.

    cv_method: CVMethod = 5
        How to evaluate accuracy during tuning.

    verbosity: int = optuna.logging.ERROR
        See https://optuna.readthedocs.io/en/stable/reference/logging.html. Most useful other option
        is `optuna.logging.INFO`.

    Returns
    -------
    htuned: HtuneResultLegacy
        See top of this file.
    """
    OBJECTIVES: Dict[str, Callable] = {
        "rf": rf_classifier_objective(X_train, y_train, cv_method),
        "svm": svm_classifier_objective(X_train, y_train, cv_method),
        "dtree": dtree_classifier_objective(X_train, y_train, cv_method),
        "mlp": mlp_classifier_objective(X_train, y_train, cv_method),
        "bag": bagging_classifier_objective(X_train, y_train, cv_method),
    }
    # HYPERTUNING
    objective = OBJECTIVES[classifier]
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    optuna.logging.set_verbosity(verbosity)
    if classifier == "mlp":
        # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
        before = os.environ.get("PYTHONWARNINGS", "")
        os.environ["PYTHONWARNINGS"] = (
            "ignore"  # can't kill ConvergenceWarning any other way
        )
        study.optimize(objective, n_trials=n_trials)
        os.environ["PYTHONWARNINGS"] = before
    else:
        study.optimize(objective, n_trials=n_trials)

    val_method = cv_desc(cv_method)
    acc = np.round(study.best_value, 3)
    if verbosity != optuna.logging.ERROR:
        print(f"\n{' Tuning Results ':=^80}")
        print("Best params:")
        pprint(study.best_params, indent=4, width=80)
        print(f"\nTuning validation: {val_method}")
        print(f"Best accuracy:      μ = {acc:0.3f}")
        # print("=" * 80, end="\n")

    return HtuneResultLegacy(
        estimator=classifier,
        mode="classify",
        n_trials=n_trials,
        cv_method=cv_method,
        val_acc=study.best_value,
        best_params=study.best_params,
    )

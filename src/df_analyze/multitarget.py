# transforms multi-target EvaluationResults into single-target EvaluationResults especially for the adaptive error rate analysis.
from __future__ import annotations

import inspect
from typing import Any, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from df_analyze.hypertune import EvaluationResults, HtuneResult


def _match_target_key(keys, target: str) -> Optional[Any]:
    if target in keys:
        return target
    target_str = str(target)
    for key in keys:
        if str(key) == target_str:
            return key
    return None


def _slice_preds(
    preds: Union[Series, DataFrame, np.ndarray],
    target: str,
    index,
    target_index: Optional[int] = None,
    target_cols: Optional[list[str]] = None,
) -> Optional[Series]:
    if isinstance(preds, DataFrame):
        key = _match_target_key(preds.columns, target)
        if key is None:
            return None
        out = preds[key]
    elif isinstance(preds, Series):
        out = preds
    elif isinstance(preds, np.ndarray):
        arr = np.asarray(preds)
        n_rows = len(index)
        if arr.shape[0] != n_rows:
            return None
        if arr.ndim == 1:
            out = Series(arr, index=index, name=target)
        elif arr.ndim == 2:
            if target_cols is not None and len(target_cols) == arr.shape[1]:
                frame = DataFrame(arr, index=index, columns=target_cols)
                key = _match_target_key(frame.columns, target)
                if key is None:
                    return None
                out = frame[key]
            else:
                if target_index is None:
                    if arr.shape[1] != 1:
                        return None
                    target_index = 0
                if target_index < 0 or target_index >= arr.shape[1]:
                    return None
                out = Series(arr[:, target_index], index=index, name=target)
        else:
            return None
    else:
        return None
    if not out.index.equals(index):
        out = out.reindex(index)
    return out


def _slice_probs(
    probs: Optional[
        Union[
            np.ndarray,
            dict[str, np.ndarray],
            list[np.ndarray],
            tuple[np.ndarray, ...],
        ]
    ],
    target: str,
    target_index: Optional[int] = None,
    target_cols: Optional[list[str]] = None,
) -> Optional[np.ndarray]:
    if isinstance(probs, dict):
        key = _match_target_key(probs.keys(), target)
        if key is None:
            return None
        return np.asarray(probs[key])
    if isinstance(probs, (list, tuple)):
        if target_index is None:
            if len(probs) != 1:
                return None
            return np.asarray(probs[0])
        if target_index < 0 or target_index >= len(probs):
            return None
        return np.asarray(probs[target_index])
    if isinstance(probs, np.ndarray):
        arr = np.asarray(probs)
        if arr.ndim == 3:
            if target_index is None:
                if arr.shape[1] != 1:
                    return None
                target_index = 0
            if target_index < 0 or target_index >= arr.shape[1]:
                return None
            return arr[:, target_index, :]
        if (
            arr.ndim == 2
            and target_cols is not None
            and len(target_cols) > 1
            and arr.shape[1] == len(target_cols)
        ):
            if target_index is None:
                return None
            return arr[:, target_index]
        return arr
    return None


def _target_names(eval_results: EvaluationResults) -> list[str]:
    y_train = getattr(eval_results, "y_train", None)
    if isinstance(y_train, DataFrame):
        return [str(col) for col in y_train.columns]

    y_test = getattr(eval_results, "y_test", None)
    if isinstance(y_test, DataFrame):
        return [str(col) for col in y_test.columns]

    return []


def _init_model_for_target(
    model_cls,
    y_train: Series,
    is_classification: bool,
    runtime=None,
    model_args=None,
):
    parameters = inspect.signature(model_cls).parameters
    kwargs = {}
    if "num_classes" in parameters:
        from df_analyze.models.base import classification_output_dim

        kwargs["num_classes"] = (
            classification_output_dim(y_train) if is_classification else 1
        )
    if "model_args" in parameters and model_args is not None:
        kwargs["model_args"] = dict(model_args)
    model = model_cls(**kwargs)
    if runtime is not None:
        model.set_runtime(runtime)
    set_targets = getattr(model, "_set_targets", None)
    if callable(set_targets):
        set_targets(y_train)
    return model


def _eval_results_for_target(
    eval_results: EvaluationResults,
    prep_train_t,
    prep_test_t,
    target: str,
) -> EvaluationResults:
    df = eval_results.df
    if "target" in df.columns:
        df_target = df[df["target"].astype(str) == str(target)].copy()
        if df_target.empty:
            df_target = df.copy()
    else:
        df_target = df.copy()

    per_target_df = None
    src_target_df = eval_results.per_target_long_table()
    if src_target_df is not None and "target" in src_target_df.columns:
        df_target_detail = src_target_df[
            src_target_df["target"].astype(str) == str(target)
        ].copy()
        if len(df_target_detail) > 0:
            per_target_df = df_target_detail
        elif "target" in df_target.columns:
            per_target_df = df_target.copy()
    elif "target" in df_target.columns:
        per_target_df = df_target.copy()
    if per_target_df is not None:
        df_target = per_target_df.copy()

    target_names = _target_names(eval_results)
    target_index = None
    if len(target_names) > 0:
        key = _match_target_key(target_names, target)
        if key is not None:
            target_index = target_names.index(key)

    results: list[HtuneResult] = []
    for res in eval_results.results:
        if getattr(res, "target", None) is not None:
            if str(getattr(res, "target")) != str(target):
                continue
        preds_test = _slice_preds(
            res.preds_test,
            target,
            prep_test_t.X.index,
            target_index=target_index,
            target_cols=target_names,
        )
        preds_train = _slice_preds(
            res.preds_train,
            target,
            prep_train_t.X.index,
            target_index=target_index,
            target_cols=target_names,
        )
        probs_test = _slice_probs(
            res.probs_test,
            target,
            target_index,
            target_cols=target_names,
        )
        probs_train = _slice_probs(
            res.probs_train,
            target,
            target_index,
            target_cols=target_names,
        )

        model = _init_model_for_target(
            res.model_cls,
            prep_train_t.y,
            is_classification=eval_results.is_classification,
            runtime=getattr(res.model, "runtime", None),
            model_args=getattr(res.model, "model_args", None),
        )

        need_proba = bool(eval_results.is_classification)
        if (
            preds_test is None
            or preds_train is None
            or (need_proba and probs_test is None)
        ):
            preds_test = Series(dtype=float)
            preds_train = Series(dtype=float)
            probs_test = None
            probs_train = None
        per_target_tuning_scores = getattr(
            res, "per_target_tuning_scores", {}
        )
        score = float(per_target_tuning_scores.get(str(target), res.score))
        results.append(
            HtuneResult(
                selection=res.selection,
                selected_cols=res.selected_cols,
                embed_select_model=res.embed_select_model,
                model_cls=res.model_cls,
                model=model,
                params=res.params,
                metric=res.metric,
                score=score,
                preds_test=preds_test,
                preds_train=preds_train,
                probs_test=probs_test,
                probs_train=probs_train,
                target=target,
                downsample_requested=res.downsample_requested,
                downsample_resolved=res.downsample_resolved,
                n_downsampled_features=res.n_downsampled_features,
                failure_reason=res.failure_reason,
                per_target_tuning_scores=(
                    {str(target): score} if np.isfinite(score) else {}
                ),
            )
        )

    return EvaluationResults(
        df=df_target,
        X_train=prep_train_t.X,
        y_train=prep_train_t.y,
        X_test=prep_test_t.X,
        y_test=prep_test_t.y,
        results=results,
        is_classification=eval_results.is_classification,
        per_target_df=per_target_df,
    )

"""Orchestrate repeated K-fold refits against one external holdout.

For every repetition the training partition is reshuffled, K models are fitted
on K-1 folds, and every model predicts the unchanged external holdout. The
runner records partitions and seeds, isolates fold failures, checkpoints
completed repetitions, and restores global random state after analysis.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from df_analyze.analysis.error_consistency.backend import resolve_ec_backend
from df_analyze.analysis.error_consistency.checkpoint import (
    configuration_fingerprint,
    load_completed_result,
    load_partial_checkpoint,
    mark_checkpoint_complete,
    save_partial_checkpoint,
)
from df_analyze.analysis.error_consistency.classification import (
    compute_classification_ec,
)
from df_analyze.analysis.error_consistency.containers import (
    ECMetricComputation,
    ErrorConsistencyResult,
)
from df_analyze.analysis.error_consistency.diagnostics import (
    classification_sample_diagnostics,
    regression_sample_diagnostics,
)
from df_analyze.analysis.error_consistency.provenance import (
    build_reproducibility_manifest,
)
from df_analyze.analysis.error_consistency.regression import (
    compute_regression_ec,
    default_regression_methods,
)
from df_analyze.analysis.error_consistency.writer import (
    write_model_outputs,
    write_root_outputs,
)
from df_analyze.runtime.hardware import (
    CUDA_BACKENDS,
    DeviceIntent,
    RuntimeComponent,
    RuntimePolicy,
    clear_fitted_model_state,
    device_reason_text,
    get_runtime,
    is_cuda_runtime_error,
    release_accelerator_memory,
)
from df_analyze.splitting import OmniKFold

MODEL_SEED_MODES = {"vary", "fixed"}
MODEL_SEED_ARGUMENTS = ("random_state", "random_seed", "seed")
TRIAL_FAILURE_COLUMNS = [
    "target",
    "model",
    "selection",
    "embed_selector",
    "trial_id",
    "repetition",
    "fold",
    "split_seed",
    "model_seed",
    "error_type",
    "reason",
    "traceback",
]


def _safe_name(value: object) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", str(value).strip()).strip("._")
    return safe or "unknown"


def _embed_name(result) -> str:
    selector = getattr(result, "embed_select_model", None)
    return "none" if selector is None else str(getattr(selector, "value", selector))


def _identity(result, target: str) -> dict[str, str]:
    model = getattr(result, "model", None)
    model_name = getattr(
        model, "shortname", getattr(result.model_cls, "shortname", "model")
    )
    return {
        "target": str(target),
        "model": str(model_name),
        "selection": str(result.selection),
        "embed_selector": _embed_name(result),
    }


def detail_output_dir(base_dir: Path, identity: dict[str, str]) -> Path:
    config = f"{identity['selection']}_{identity['embed_selector']}"
    return (
        base_dir
        / _safe_name(identity["target"])
        / _safe_name(identity["model"])
        / _safe_name(config)
    )


@contextmanager
def _preserve_random_state(*, use_torch: bool, use_cuda: bool):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch = None
    if use_torch:
        try:
            import torch
        except ImportError:
            torch = None
    torch_state = None
    cuda_states = None
    if torch is not None:
        try:
            torch_state = torch.random.get_rng_state()
            if use_cuda and torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
        except RuntimeError:
            torch_state = None
            cuda_states = None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _target_name(y: Series) -> str:
    return "target" if y.name is None else str(y.name)


def _partition_signature(splits: list[tuple[np.ndarray, np.ndarray]]) -> str:
    validation_folds = sorted(
        ",".join(str(index) for index in np.sort(validation)) for _, validation in splits
    )
    payload = "|".join(validation_folds).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalize_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def _trial_model_seed(base_seed: int, repetition: int, fold: int, mode: str) -> int:
    mode = str(mode).lower()
    if mode not in MODEL_SEED_MODES:
        raise ValueError(
            f"Unknown EC model-seed mode '{mode}'. Expected one of "
            f"{sorted(MODEL_SEED_MODES)}."
        )
    normalized = _normalize_seed(base_seed)
    if mode == "fixed":
        return normalized
    state = np.random.SeedSequence(
        [normalized, int(repetition), int(fold)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _seed_trial_model(model, tuned_args, model_seed: int) -> tuple[dict, str]:
    """Seed global RNGs and explicit estimator seed parameters when available."""
    seed = _normalize_seed(model_seed)
    random.seed(seed)
    np.random.seed(seed)
    component = getattr(model, "runtime_component", RuntimeComponent.Sklearn)
    if CUDA_BACKENDS.get(component) == "torch":
        try:
            import torch

            torch.manual_seed(seed)
            runtime = getattr(model, "runtime", None)
            use_cuda = (
                runtime is not None
                and runtime.decision_for(component).resolved == "cuda"
            )
            if use_cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except (ImportError, RuntimeError):
            pass

    trial_args = dict(tuned_args or {})
    full_args = {
        **getattr(model, "fixed_args", {}),
        **getattr(model, "default_args", {}),
        **dict(getattr(model, "model_args", {})),
        **trial_args,
    }
    # random_state, random_seed, and seed are normally estimator-level aliases,
    # not independent controls. Prefer the alias already used by df-analyze's
    # model arguments so EC does not pass conflicting synonyms (CatBoost, for
    # example, rejects random_seed and random_state when both are present).
    existing = [key for key in MODEL_SEED_ARGUMENTS if key in full_args]
    signature_seed_arguments: set[str] = set()
    try:
        model_cls, _ = model.model_cls_args(dict(full_args))
        parameters = inspect.signature(model_cls).parameters
        signature_seed_arguments.update(
            key for key in MODEL_SEED_ARGUMENTS if key in parameters
        )
    except (AttributeError, TypeError, ValueError):
        pass
    selected_seed_argument = next(
        iter(existing),
        next(
            (key for key in MODEL_SEED_ARGUMENTS if key in signature_seed_arguments),
            None,
        ),
    )
    explicit = {selected_seed_argument} if selected_seed_argument is not None else set()
    if getattr(model, "shortname", None) == "kan":
        explicit.add("module__seed")
    for key in explicit:
        trial_args[key] = seed
    seeded = ",".join(sorted(explicit)) if explicit else "global_rng_only"
    return trial_args, seeded


def _validate_repetition_partition(
    splits: list[tuple[np.ndarray, np.ndarray]], n_rows: int
) -> None:
    validation = np.concatenate([np.asarray(valid, dtype=int) for _, valid in splits])
    if validation.size != n_rows or not np.array_equal(
        np.sort(validation), np.arange(n_rows)
    ):
        raise RuntimeError(
            "Repeated K-fold validation folds must assign every training-row "
            "position exactly once per repetition."
        )
    all_rows = set(range(n_rows))
    for train, valid in splits:
        train_set = set(np.asarray(train, dtype=int))
        valid_set = set(np.asarray(valid, dtype=int))
        if train_set.intersection(valid_set) or train_set.union(valid_set) != all_rows:
            raise RuntimeError(
                "Each EC fold must use disjoint training/validation complements."
            )


def _init_model(
    result,
    y_train: Series,
    is_classification: bool,
    runtime: RuntimePolicy | None = None,
):
    from df_analyze.models.base import classification_output_dim

    parameters = inspect.signature(result.model_cls).parameters
    kwargs = {}
    if "num_classes" in parameters:
        kwargs["num_classes"] = (
            classification_output_dim(y_train) if is_classification else 1
        )
    model_args = getattr(result.model, "model_args", None)
    if "model_args" in parameters and model_args:
        kwargs["model_args"] = model_args
    model = result.model_cls(**kwargs)
    runtime = runtime or getattr(result.model, "runtime", None)
    if runtime is not None:
        model.set_runtime(runtime)
    return model


def _prediction_vector(predictions, expected_index, n_samples: int) -> np.ndarray:
    if isinstance(predictions, DataFrame):
        if predictions.shape[1] != 1:
            raise ValueError("Error consistency expects one target at a time.")
        predictions = predictions.iloc[:, 0]
    if isinstance(predictions, Series):
        if predictions.index.equals(expected_index):
            values = predictions.to_numpy()
        elif len(predictions) == n_samples:
            values = predictions.to_numpy()
        else:
            aligned = predictions.reindex(expected_index)
            if aligned.isna().any():
                raise ValueError("Could not align predictions with holdout rows.")
            values = aligned.to_numpy()
    else:
        values = np.asarray(predictions)
    values = np.asarray(values).ravel()
    if values.size != n_samples:
        raise ValueError(
            f"Expected {n_samples} holdout predictions, received {values.size}."
        )
    return values


def _score_trial(
    model,
    X_test: DataFrame,
    y_test: Series,
    predictions: np.ndarray,
    metric,
) -> dict[str, float]:
    probabilities = None
    if model.is_classifier:
        try:
            probabilities = model.predict_proba(X_test)
        except (AttributeError, TypeError, ValueError):
            probabilities = None
    try:
        return model._score_outputs(y_test, predictions, probabilities)
    except (TypeError, ValueError):
        return {str(metric.value): float(metric.tuning_score(y_test, predictions))}


def _selection_candidates(result) -> set[str]:
    selection = str(result.selection)
    candidates = {selection}
    if selection == "embed":
        candidates.add(f"embed_{_embed_name(result)}")
    return candidates


def _single_model_values(
    eval_df: DataFrame,
    result,
    metric: str,
    target: str,
) -> tuple[float, float]:
    if eval_df.empty:
        return np.nan, np.nan
    selected = eval_df.copy()
    if "model" in selected:
        selected = selected[selected["model"].astype(str) == str(result.model.shortname)]
    if "selection" in selected:
        selected = selected[
            selected["selection"].astype(str).isin(_selection_candidates(result))
        ]
    if "embed_selector" in selected:
        selected = selected[selected["embed_selector"].astype(str) == _embed_name(result)]
    if "metric" in selected:
        selected = selected[selected["metric"].astype(str) == str(metric)]
    if "target" in selected:
        selected = selected[selected["target"].astype(str) == str(target)]
    if selected.empty:
        return np.nan, np.nan
    row = selected.iloc[0]
    return float(row.get("trainset", np.nan)), float(row.get("holdout", np.nan))


def _performance_summary(
    trial_scores: DataFrame,
    eval_df: DataFrame,
    result,
    identity: dict[str, str],
) -> DataFrame:
    if trial_scores.empty:
        return DataFrame()
    rows = []
    for metric, values in trial_scores.groupby("metric", dropna=False)["score"]:
        finite = values[np.isfinite(values)]
        train_score, holdout_score = _single_model_values(
            eval_df, result, str(metric), identity["target"]
        )
        rows.append(
            {
                **identity,
                "metric": metric,
                "ec_trial_mean": float(finite.mean()) if len(finite) else np.nan,
                "ec_trial_sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "n_ec_trials": int(len(finite)),
                "single_model_train": train_score,
                "single_model_holdout": holdout_score,
            }
        )
    return DataFrame(rows)


def _enrich_pairwise_trials(
    computations: list[ECMetricComputation], trial_design: DataFrame
) -> None:
    design = trial_design.set_index("model_index")
    for computation in computations:
        pairwise = computation.pairwise
        if pairwise.empty:
            continue
        pairwise["repetition_i"] = pairwise["model_i"].map(design["repetition"])
        pairwise["fold_i"] = pairwise["model_i"].map(design["fold"])
        pairwise["repetition_j"] = pairwise["model_j"].map(design["repetition"])
        pairwise["fold_j"] = pairwise["model_j"].map(design["fold"])
        pairwise["same_repetition"] = pairwise["repetition_i"] == pairwise["repetition_j"]
        pairwise["pair_scope"] = np.where(
            pairwise["same_repetition"],
            "within_repetition",
            "between_repetition",
        )


def _pair_scope_summary(
    computation: ECMetricComputation, trial_design: DataFrame
) -> dict[str, float | int]:
    pairwise = computation.pairwise
    result: dict[str, float | int] = {}
    if pairwise.empty or "pair_scope" not in pairwise:
        successful = trial_design.dropna(subset=["model_index"]).copy()
        if successful.empty:
            scoped = DataFrame()
        else:
            successful["model_index"] = successful["model_index"].astype(int)
            successful = successful.sort_values("model_index")
            repetitions = successful["repetition"].to_numpy()
            pair_i, pair_j = np.triu_indices(len(successful), k=1)
            values = np.asarray(computation.matrix)[pair_i, pair_j]
            finite = np.isfinite(values)
            scoped = DataFrame(
                {
                    "pair_mean": values[finite],
                    "repetition_i": repetitions[pair_i[finite]],
                    "repetition_j": repetitions[pair_j[finite]],
                }
            )
            scoped["same_repetition"] = scoped["repetition_i"] == scoped["repetition_j"]
            scoped["pair_scope"] = np.where(
                scoped["same_repetition"],
                "within_repetition",
                "between_repetition",
            )
        pairwise = scoped
    if pairwise.empty or "pair_scope" not in pairwise:
        for scope in ("within_repetition", "between_repetition"):
            result[f"ec_{scope}_mean"] = np.nan
            result[f"ec_{scope}_sd"] = np.nan
            result[f"n_{scope}_pairs"] = 0
        result["ec_repetition_mean"] = np.nan
        result["ec_repetition_sd"] = np.nan
        result["n_repetition_estimates"] = 0
        return result
    for scope in ("within_repetition", "between_repetition"):
        values = pairwise.loc[pairwise["pair_scope"] == scope, "pair_mean"]
        values = values[np.isfinite(values)].astype(float)
        result[f"ec_{scope}_mean"] = float(values.mean()) if len(values) else np.nan
        result[f"ec_{scope}_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else np.nan
        )
        result[f"n_{scope}_pairs"] = int(len(values))

    within = pairwise[pairwise["same_repetition"]].copy()
    within = within[np.isfinite(within["pair_mean"])]
    repetition_estimates = within.groupby("repetition_i")["pair_mean"].mean()
    result["ec_repetition_mean"] = (
        float(repetition_estimates.mean()) if len(repetition_estimates) else np.nan
    )
    result["ec_repetition_sd"] = (
        float(repetition_estimates.std(ddof=1))
        if len(repetition_estimates) > 1
        else np.nan
    )
    result["n_repetition_estimates"] = int(len(repetition_estimates))
    return result


def _run_one_result(
    prep_train,
    prep_test,
    eval_results,
    result,
    options,
    base_dir: Path,
) -> ErrorConsistencyResult:
    X_train = prep_train.model_matrix(result.model_cls, result.selected_cols)
    X_test = prep_test.model_matrix(result.model_cls, result.selected_cols)
    base_runtime = getattr(options, "runtime", None)
    if base_runtime is None:
        base_runtime = get_runtime(
            getattr(options, "device", DeviceIntent.CPU)
        )
    model_runtime = base_runtime.for_task(
        len(X_train),
        X_train.shape[1],
        n_queries=len(X_test),
    )
    component = getattr(
        result.model_cls, "runtime_component", RuntimeComponent.Sklearn
    )
    decision = model_runtime.decision_for(component)
    print(
        "[device] Error consistency refits "
        f"{getattr(result.model, 'shortname', result.model_cls.__name__)} / "
        f"{result.selection}: {decision.resolved.upper()} "
        f"({len(X_train)} rows x {X_train.shape[1]} features; "
        f"{device_reason_text(decision)})"
    )
    try:
        return _run_one_result_attempt(
            prep_train,
            prep_test,
            eval_results,
            result,
            options,
            base_dir,
            model_runtime=model_runtime,
            allow_resume=True,
        )
    except Exception as error:
        cuda_failure = (
            decision.resolved == "cuda" and is_cuda_runtime_error(error)
        )
        if (
            model_runtime.intent is DeviceIntent.Auto
            and cuda_failure
        ):
            reason = f"cuda_runtime_fallback:{type(error).__name__}"
            model_runtime.record_cpu_fallback(component, reason)
            model_runtime.record_cpu_fallback(
                RuntimeComponent.ErrorConsistency, reason
            )
            release_accelerator_memory()
            warn(
                "Error consistency encountered a CUDA runtime failure while "
                f"refitting {getattr(result.model, 'shortname', 'a model')}. "
                "Restarting this complete model/selection configuration on CPU "
                "once."
            )
            return _run_one_result_attempt(
                prep_train,
                prep_test,
                eval_results,
                result,
                options,
                base_dir,
                model_runtime=model_runtime,
                allow_resume=False,
            )
        if (
            model_runtime.intent is DeviceIntent.CUDA
            and cuda_failure
        ):
            raise RuntimeError(
                "Error consistency failed while refitting a CUDA-capable model "
                "under strict --device cuda. CPU fallback is disabled."
            ) from error
        raise


def _run_one_result_attempt(
    prep_train,
    prep_test,
    eval_results,
    result,
    options,
    base_dir: Path,
    *,
    model_runtime: RuntimePolicy,
    allow_resume: bool,
) -> ErrorConsistencyResult:
    if not np.isfinite(float(result.score)):
        raise RuntimeError("The tuned configuration did not produce a finite score.")
    if not isinstance(prep_train.y, Series) or not isinstance(prep_test.y, Series):
        raise RuntimeError("Error consistency runs one target at a time.")

    identity = _identity(result, _target_name(prep_test.y))
    X_train = prep_train.model_matrix(result.model_cls, result.selected_cols)
    X_test = prep_test.model_matrix(result.model_cls, result.selected_cols)
    y_train = prep_train.y
    y_test = prep_test.y
    groups = prep_train.groups
    n_folds = int(options.ec_folds)
    repetitions = int(options.ec_repetitions)
    if n_folds > len(X_train):
        raise ValueError(
            f"Error-consistency folds ({n_folds}) exceed training rows ({len(X_train)})."
        )

    output_detail = str(getattr(options, "ec_output_detail", "full")).lower()
    detail_dir = detail_output_dir(base_dir, identity)
    tuned_args = getattr(result.model, "tuned_args", None) or result.params
    fingerprint, fingerprint_payload = configuration_fingerprint(
        identity=identity,
        options=options,
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_test,
        y_holdout=y_test,
        is_classification=eval_results.is_classification,
        model_class=result.model_cls,
        tuned_parameters=tuned_args,
    )
    resume = allow_resume and bool(getattr(options, "ec_resume", False))
    if resume:
        completed_result = load_completed_result(detail_dir, fingerprint=fingerprint)
        if completed_result is not None:
            return completed_result
        partial = load_partial_checkpoint(detail_dir, fingerprint=fingerprint)
    else:
        partial = None

    if partial is None:
        predictions: list[np.ndarray] = []
        design_rows: list[dict] = []
        assignment_rows: list[dict] = []
        score_rows: list[dict] = []
        failure_rows: list[dict] = []
        start_repetition = 0
    else:
        predictions = [row.copy() for row in np.asarray(partial.predictions)]
        design_rows = partial.trial_design.to_dict("records")
        assignment_rows = partial.fold_assignments.to_dict("records")
        score_rows = partial.trial_scores.to_dict("records")
        failure_rows = partial.trial_failures.to_dict("records")
        start_repetition = int(partial.completed_repetitions)
        if start_repetition > repetitions:
            raise RuntimeError(
                "EC checkpoint contains more repetitions than the requested run."
            )

    n_requested_models = n_folds * repetitions
    n_requested_pairs = n_requested_models * (n_requested_models - 1) // 2
    n_methods = (
        1
        if eval_results.is_classification
        else len(getattr(options, "ec_methods", None) or default_regression_methods())
    )
    if n_requested_models >= 100:
        warn(
            "This EC configuration requires "
            f"{n_requested_models} additional model refits before pairwise EC "
            "calculation. --ec-output-detail summary reduces retained artifacts "
            "but does not reduce refit time. Start with fewer repetitions when "
            "estimating runtime."
        )
    if n_requested_pairs >= 100_000 and output_detail == "full":
        warn(
            "This EC run requests "
            f"{n_requested_models} models and {n_requested_pairs:,} model pairs "
            f"for each of {n_methods} method(s). Consider "
            "--ec-output-detail summary or pairwise if full sample-level output "
            "is not required."
        )

    model_seed_mode = str(getattr(options, "ec_model_seed_mode", "vary")).lower()
    checkpoint_every = int(getattr(options, "ec_checkpoint_every", 5))
    if partial is None:
        save_partial_checkpoint(
            detail_dir,
            fingerprint=fingerprint,
            fingerprint_payload=fingerprint_payload,
            completed_repetitions=0,
            predictions=np.empty((0, len(y_test))),
            trial_design=DataFrame(),
            fold_assignments=DataFrame(),
            trial_scores=DataFrame(),
            trial_failures=DataFrame(columns=TRIAL_FAILURE_COLUMNS),
        )

    for repetition in range(start_repetition, repetitions):
        repetition_seed = _normalize_seed(int(options.seed) + repetition)
        splitter = OmniKFold(
            n_splits=n_folds,
            is_classification=eval_results.is_classification,
            grouped=groups is not None,
            labels=getattr(prep_train, "labels", None),
            shuffle=True,
            seed=repetition_seed,
            warn_on_fallback=True,
            allow_group_fallback=False,
            df_analyze_phase="Error consistency repeated K-fold",
        )
        splits, split_fallback = splitter.split(X_train, y_train, groups)
        if groups is not None and split_fallback:
            raise RuntimeError(
                "Grouped error consistency could not create group-disjoint folds."
            )
        _validate_repetition_partition(splits, len(X_train))
        partition_signature = _partition_signature(splits)
        for fold, (idx_train, idx_validation) in enumerate(splits):
            trial_id = repetition * n_folds + fold
            candidate_model_index = len(predictions)
            model_seed = _trial_model_seed(
                int(options.seed), repetition, fold, model_seed_mode
            )
            X_fold = X_train.iloc[idx_train]
            y_fold = y_train.iloc[idx_train]
            g_fold = None if groups is None else groups.iloc[idx_train]
            group_overlap = np.nan
            if groups is not None:
                train_groups = set(groups.iloc[idx_train].astype(str))
                validation_groups = set(groups.iloc[idx_validation].astype(str))
                group_overlap = len(train_groups.intersection(validation_groups))

            model = None
            success = False
            seeded_parameters = "not_initialized"
            failure_reason = ""
            try:
                model = _init_model(
                    result,
                    y_train,
                    eval_results.is_classification,
                    runtime=model_runtime,
                )
                trial_tuned_args, seeded_parameters = _seed_trial_model(
                    model, tuned_args, model_seed
                )
                model.refit_tuned(X_fold, y_fold, g=g_fold, tuned_args=trial_tuned_args)
                pred = _prediction_vector(
                    model.tuned_predict(X_test), X_test.index, len(X_test)
                )
                if (
                    not eval_results.is_classification
                    and not np.isfinite(pred.astype(float)).all()
                ):
                    raise ValueError(
                        "Regression EC received non-finite holdout predictions."
                    )
                predictions.append(pred)
                success = True
                for metric, score in _score_trial(
                    model, X_test, y_test, pred, result.metric
                ).items():
                    score_rows.append(
                        {
                            **identity,
                            "trial_id": trial_id,
                            "model_index": candidate_model_index,
                            "repetition": repetition,
                            "fold": fold,
                            "model_seed": model_seed,
                            "metric": metric,
                            "score": score,
                        }
                    )
            except Exception as error:
                if (
                    model_runtime.decision_for(
                        getattr(
                            result.model_cls,
                            "runtime_component",
                            RuntimeComponent.Sklearn,
                        )
                    ).resolved
                    == "cuda"
                    and is_cuda_runtime_error(error)
                ):
                    raise
                failure_reason = str(error)
                failure_rows.append(
                    {
                        **identity,
                        "trial_id": trial_id,
                        "repetition": repetition,
                        "fold": fold,
                        "split_seed": repetition_seed,
                        "model_seed": model_seed,
                        "error_type": type(error).__name__,
                        "reason": failure_reason,
                        "traceback": traceback.format_exc(),
                    }
                )
                warn(
                    "Error-consistency trial failed and was recorded: "
                    f"{identity['model']}/{identity['selection']} repetition "
                    f"{repetition}, fold {fold}: {error}"
                )
            finally:
                if model is not None:
                    try:
                        clear_fitted_model_state(model)
                        model._cleanup_after_fold()
                    except Exception as cleanup_error:
                        warn(
                            "EC fold cleanup failed after trial completion: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    del model

            model_index = candidate_model_index if success else np.nan
            design_rows.append(
                {
                    **identity,
                    "trial_id": trial_id,
                    "model_index": model_index,
                    "repetition": repetition,
                    "fold": fold,
                    "seed": repetition_seed,
                    "split_seed": repetition_seed,
                    "model_seed": model_seed,
                    "model_seed_mode": model_seed_mode,
                    "seeded_parameters": seeded_parameters,
                    "partition_signature": partition_signature,
                    "n_train": len(idx_train),
                    "n_validation": len(idx_validation),
                    "split_fallback": split_fallback,
                    "group_overlap": group_overlap,
                    "status": "success" if success else "failed",
                    "failure_reason": failure_reason,
                }
            )
            for train_position in np.asarray(idx_validation, dtype=int):
                assignment_rows.append(
                    {
                        **identity,
                        "trial_id": trial_id,
                        "repetition": repetition,
                        "fold": fold,
                        "model_index": model_index,
                        "split_seed": repetition_seed,
                        "partition_signature": partition_signature,
                        "train_position": int(train_position),
                        "row_id": X_train.index[train_position],
                        "trial_status": "success" if success else "failed",
                        "group": (
                            np.nan if groups is None else groups.iloc[train_position]
                        ),
                    }
                )

        completed_repetitions = repetition + 1
        if (
            completed_repetitions % checkpoint_every == 0
            or completed_repetitions == repetitions
        ):
            save_partial_checkpoint(
                detail_dir,
                fingerprint=fingerprint,
                fingerprint_payload=fingerprint_payload,
                completed_repetitions=completed_repetitions,
                predictions=(
                    np.vstack(predictions) if predictions else np.empty((0, len(y_test)))
                ),
                trial_design=DataFrame(design_rows),
                fold_assignments=DataFrame(assignment_rows),
                trial_scores=DataFrame(score_rows),
                trial_failures=DataFrame(failure_rows, columns=TRIAL_FAILURE_COLUMNS),
            )

    if len(predictions) < 2:
        raise RuntimeError(
            "Error consistency requires at least two successful refit trials; "
            f"{len(predictions)} succeeded and {len(failure_rows)} failed."
        )

    prediction_matrix = np.vstack(predictions)
    trial_design = DataFrame(design_rows)
    fold_assignments = DataFrame(assignment_rows)
    trial_scores = DataFrame(score_rows)
    trial_failures = DataFrame(failure_rows, columns=TRIAL_FAILURE_COLUMNS)
    n_unique_partitions = int(trial_design["partition_signature"].nunique())
    if n_unique_partitions < repetitions:
        warn(
            "Error-consistency repetitions produced "
            f"{n_unique_partitions} distinct K-fold partitions from "
            f"{repetitions} repetitions. This can occur when the number of "
            "possible partitions is small."
        )
    backend = resolve_ec_backend(
        options,
        prediction_matrix.shape[0],
        len(y_test),
        runtime=model_runtime,
    )
    backend_device = "CUDA" if backend.resolved == "torch_cuda" else "CPU"
    print(
        "[device] Error consistency pairwise calculation "
        f"{identity['model']} / {identity['selection']}: {backend_device} "
        f"({backend.reason})"
    )
    if eval_results.is_classification:
        computations = [
            compute_classification_ec(
                prediction_matrix,
                y_test.to_numpy(),
                empty_unions=options.ec_empty_unions,
                backend=backend,
                output_detail=output_detail,
            )
        ]
        error_matrix = prediction_matrix != y_test.to_numpy()[None, :]
        diagnostics = (
            classification_sample_diagnostics(
                error_matrix, row_ids=X_test.index, groups=prep_test.groups
            )
            if output_detail == "full"
            else {}
        )
        residual_or_error = error_matrix
        problem_type = "classification"
    else:
        methods = getattr(options, "ec_methods", None)
        computations = compute_regression_ec(
            prediction_matrix.astype(float) - y_test.to_numpy(dtype=float)[None, :],
            methods=methods,
            epsilon=float(options.ec_epsilon),
            backend=backend,
            row_ids=X_test.index,
            output_detail=output_detail,
        )
        residual_or_error = (
            prediction_matrix.astype(float) - y_test.to_numpy(dtype=float)[None, :]
        )
        diagnostics = (
            regression_sample_diagnostics(
                residual_or_error, row_ids=X_test.index, groups=prep_test.groups
            )
            if output_detail == "full"
            else {}
        )
        problem_type = "regression"

    _enrich_pairwise_trials(computations, trial_design)
    summary_rows = []
    for computation in computations:
        row = {
            **identity,
            "problem_type": problem_type,
            "n_ec_models": prediction_matrix.shape[0],
            "n_requested_ec_models": n_requested_models,
            "n_failed_trials": len(trial_failures),
            "n_folds": n_folds,
            "n_repetitions": repetitions,
            "n_unique_partitions": n_unique_partitions,
            "n_holdout_samples": len(y_test),
            "group_split_fallback_used": bool(trial_design["split_fallback"].any()),
            "model_seed_mode": model_seed_mode,
        }
        row.update(computation.summary())
        row.update(_pair_scope_summary(computation, trial_design))
        summary_rows.append(row)

    performance = _performance_summary(trial_scores, eval_results.df, result, identity)
    save_predictions = bool(getattr(options, "ec_save_predictions", False))
    prediction_df = None
    residual_df = None
    if save_predictions:
        columns = [f"model_{idx:03d}" for idx in range(prediction_matrix.shape[0])]
        prediction_df = DataFrame(prediction_matrix.T, columns=columns)
        prediction_df.insert(0, "y_true", y_test.to_numpy())
        prediction_df.insert(0, "holdout_position", np.arange(len(y_test)))
        prediction_df.insert(0, "row_id", X_test.index)
        residual_df = DataFrame(residual_or_error.T, columns=columns)
        residual_df.insert(0, "y_true", y_test.to_numpy())
        residual_df.insert(0, "holdout_position", np.arange(len(y_test)))
        residual_df.insert(0, "row_id", X_test.index)
        if prep_test.groups is not None:
            prediction_df.insert(3, "group", prep_test.groups.to_numpy())
            residual_df.insert(3, "group", prep_test.groups.to_numpy())

    methods = [computation.info.name for computation in computations]
    requested_backend = backend.as_metadata()
    actual_backends = []
    for computation in computations:
        actual = {
            key: computation.extra.get(key, value)
            for key, value in requested_backend.items()
        }
        if actual not in actual_backends:
            actual_backends.append(actual)
    result_metadata = {
        "target": identity["target"],
        "ec_methods": methods,
        "ec_backends": actual_backends,
        "model_seed_mode": model_seed_mode,
        "holdout_role": str(getattr(options, "ec_holdout_role", "validation")).lower(),
        "output_detail": output_detail,
        "checkpoint_fingerprint": fingerprint,
        "resumed_from_partial_checkpoint": partial is not None,
        "n_requested_models": n_requested_models,
        "n_successful_trials": len(predictions),
        "n_failed_trials": len(trial_failures),
    }
    result_summary = DataFrame(summary_rows)
    write_model_outputs(
        detail_dir,
        computations,
        trial_design,
        fold_assignments,
        trial_scores,
        diagnostics,
        predictions=prediction_df,
        residuals_or_errors=residual_df,
        output_detail=output_detail,
        summary=result_summary,
        performance=performance,
        trial_failures=trial_failures,
        metadata=result_metadata,
    )
    mark_checkpoint_complete(
        detail_dir,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        completed_repetitions=repetitions,
    )
    return ErrorConsistencyResult(
        summary=result_summary,
        performance=performance,
        trial_scores=trial_scores,
        trial_design=trial_design,
        fold_assignments=fold_assignments,
        trial_failures=trial_failures,
        metadata=result_metadata,
    )


def combine_error_consistency_results(
    results: list[ErrorConsistencyResult],
    options=None,
    skipped: Optional[list[dict[str, str]]] = None,
) -> ErrorConsistencyResult:
    def combine(attribute: str) -> DataFrame:
        frames = [getattr(result, attribute) for result in results]
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else DataFrame()

    methods = []
    backends = []
    targets = []
    all_skipped = list(skipped or [])
    for result in results:
        for method in result.metadata.get("ec_methods", []):
            if method not in methods:
                methods.append(method)
        for backend in result.metadata.get("ec_backends", []):
            if backend not in backends:
                backends.append(backend)
        target = result.metadata.get("target")
        if target is not None and target not in targets:
            targets.append(target)
        for nested_target in result.metadata.get("targets", []):
            if nested_target not in targets:
                targets.append(nested_target)
        all_skipped.extend(result.metadata.get("skipped_configurations", []))
    if not methods and options is not None:
        if getattr(options, "is_classification", False):
            methods = ["classification_iou"]
        else:
            selected = getattr(options, "ec_methods", None)
            methods = default_regression_methods() if selected is None else list(selected)
    is_classification = "classification_iou" in methods
    holdout_role = str(getattr(options, "ec_holdout_role", "validation")).lower()
    n_resumed_complete = sum(
        bool(result.metadata.get("resumed_from_complete_checkpoint", False))
        for result in results
    )
    n_resumed_partial = sum(
        bool(result.metadata.get("resumed_from_partial_checkpoint", False))
        for result in results
    )

    trial_failures = combine("trial_failures")
    if trial_failures.empty:
        trial_failures = DataFrame(columns=TRIAL_FAILURE_COLUMNS)
    metadata = {
        "n_configurations": len(results),
        "targets": targets,
        "ec_methods": methods,
        "ec_backends": backends,
        "skipped_configurations": all_skipped,
        "n_folds": getattr(options, "ec_folds", None),
        "n_repetitions": getattr(options, "ec_repetitions", None),
        "model_seed_mode": getattr(options, "ec_model_seed_mode", "vary"),
        "holdout_role": holdout_role,
        "selection_outputs_enabled": holdout_role == "validation",
        "ec_profile": getattr(options, "ec_profile", "none"),
        "ec_profile_scope": getattr(options, "ec_profile_scope", "df-analyze defaults"),
        "ec_profile_overrides": getattr(options, "ec_profile_overrides", {}),
        "ec_output_detail": getattr(options, "ec_output_detail", "full"),
        "ec_resume": bool(getattr(options, "ec_resume", False)),
        "ec_checkpoint_every": getattr(options, "ec_checkpoint_every", 5),
        "n_resumed_complete_configurations": n_resumed_complete,
        "n_resumed_partial_configurations": n_resumed_partial,
        "ec_recurrence_threshold": getattr(options, "ec_recurrence_threshold", 0.5),
        "skipped_targets": [
            item for item in all_skipped if item.get("scope") == "target"
        ],
        "scientific_scope": (
            "EC compares repeated K-fold fits on one shared external holdout. "
            "Preprocessing, feature selection, and tuning stay fixed. A final-test "
            "holdout must not be used for model selection."
        ),
        "randomness_scope": (
            "Split seeds change by repetition. Model seeds change by repetition "
            "and fold when model_seed_mode='vary', and stay fixed when it is 'fixed'."
        ),
        "dispersion_note": (
            "EC standard deviations describe variation in this run. They are not "
            "standard errors or confidence intervals."
        ),
        "method_evidence_scope": (
            "Classification EC follows Equation 1 of Levman et al. (2023), "
            "doi:10.3390/diagnostics13071315."
            if is_classification
            else "Regression EC methods compare residuals across repeated fits. "
            "They are descriptive measures, not confidence intervals or tests."
        ),
    }
    if options is not None:
        metadata["reproducibility_manifest"] = build_reproducibility_manifest(
            options,
            metadata=metadata,
            trial_failures=len(trial_failures),
        )
    return ErrorConsistencyResult(
        summary=combine("summary"),
        performance=combine("performance"),
        trial_scores=combine("trial_scores"),
        trial_design=combine("trial_design"),
        fold_assignments=combine("fold_assignments"),
        trial_failures=trial_failures,
        metadata=metadata,
    )


def _run_error_consistency_analysis(
    prep_train,
    prep_test,
    eval_results,
    options,
    prog_dirs=None,
    base_dir: Optional[Path] = None,
    write_root: bool = True,
) -> ErrorConsistencyResult:
    if base_dir is None:
        results_dir = None if prog_dirs is None else getattr(prog_dirs, "results", None)
        if results_dir is not None:
            base_dir = Path(results_dir) / "error_consistency"
    if base_dir is None:
        raise ValueError(
            "An output directory is required for error consistency analysis."
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    holdout_role = str(getattr(options, "ec_holdout_role", "validation")).lower()
    if holdout_role == "test":
        warn(
            "Error consistency is using a holdout declared as final test data. "
            "EC values will be reported, but model-ranking and EC/performance "
            "correlation outputs are disabled. Use --ec-holdout-role validation "
            "only with a separate validation or audit holdout."
        )

    outputs = []
    skipped = []
    for result in getattr(eval_results, "results", []):
        identity = _identity(result, _target_name(prep_test.y))
        try:
            outputs.append(
                _run_one_result(
                    prep_train,
                    prep_test,
                    eval_results,
                    result,
                    options,
                    base_dir,
                )
            )
        except Exception as error:
            runtime = getattr(options, "runtime", None)
            if (
                runtime is not None
                and runtime.intent is DeviceIntent.CUDA
                and is_cuda_runtime_error(error)
            ):
                raise
            skipped.append({**identity, "reason": str(error)})
            warn(
                "Skipping error-consistency configuration "
                f"{identity['model']}/{identity['selection']}: {error}\n"
                f"{traceback.format_exc()}"
            )

    combined = combine_error_consistency_results(outputs, options, skipped)
    if write_root:
        write_root_outputs(base_dir, combined)
    if not outputs:
        reasons = "; ".join(item["reason"] for item in skipped)
        raise RuntimeError(
            "Error consistency did not complete for any tuned configuration. "
            f"{reasons or 'No tuned configurations were available.'}"
        )
    return combined


def run_error_consistency_analysis(
    prep_train,
    prep_test,
    eval_results,
    options,
    prog_dirs=None,
    base_dir: Optional[Path] = None,
    write_root: bool = True,
) -> ErrorConsistencyResult:
    results = getattr(eval_results, "results", [])
    torch_components = [
        getattr(result.model_cls, "runtime_component", RuntimeComponent.Sklearn)
        for result in results
    ]
    use_torch = any(
        CUDA_BACKENDS.get(component) == "torch"
        for component in torch_components
    )
    runtime = getattr(options, "runtime", None)
    intent = getattr(runtime, "intent", getattr(options, "device", DeviceIntent.CPU))
    if not isinstance(intent, DeviceIntent):
        intent = DeviceIntent.from_arg(intent)
    use_cuda = False
    if use_torch:
        base_runtime = runtime or get_runtime(intent)
        for result, component in zip(results, torch_components):
            if CUDA_BACKENDS.get(component) != "torch":
                continue
            X_train = prep_train.model_matrix(
                result.model_cls, result.selected_cols
            )
            X_test = prep_test.model_matrix(
                result.model_cls, result.selected_cols
            )
            task_runtime = base_runtime.for_task(
                len(X_train),
                X_train.shape[1],
                n_queries=len(X_test),
            )
            if task_runtime.decision_for(component).resolved == "cuda":
                use_cuda = True
                break
    with _preserve_random_state(
        use_torch=use_torch,
        use_cuda=use_cuda,
    ):
        return _run_error_consistency_analysis(
            prep_train=prep_train,
            prep_test=prep_test,
            eval_results=eval_results,
            options=options,
            prog_dirs=prog_dirs,
            base_dir=base_dir,
            write_root=write_root,
        )

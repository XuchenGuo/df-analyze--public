from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
SRC = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(SRC))  # isort: skip
# fmt: on


import gc
import json
import logging
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pprint
from time import perf_counter
from typing import List, Optional, TypeVar, Union
from warnings import warn

import numpy as np
from pandas import DataFrame

from df_analyze.analysis.univariate.associate import target_associations
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import ProgramOptions, get_options
from df_analyze.downsampling import FeatureDownsampleResult, downsample_split
from df_analyze.downsampling.large import large_table_prepared_splits
from df_analyze.downsampling.sparse import sparse_prepared_splits
from df_analyze.enumerables import (
    FeatureDownsampleMethod,
    FeatureSelection,
    ValidationMethod,
)
from df_analyze.hypertune import evaluate_tuned
from df_analyze.multitarget import _eval_results_for_target
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import (
    prepare_data,
    raw_train_test_indices,
    usable_training_indices,
)
from df_analyze.preprocessing.targets import as_target_list
from df_analyze.runtime.hardware import (
    DeviceIntent,
    RuntimeComponent,
    format_device_plan,
    is_cuda_runtime_error,
    validate_cuda_request,
)
from df_analyze.selection.filter import FilterSelected, filter_select_features
from df_analyze.selection.models import model_select_features
from df_analyze.selection.multitarget import (
    aggregate_filter_selected,
    aggregate_model_selected,
)
from df_analyze.saving import windows_io_path
from df_analyze.splitting import (
    resolve_final_cv_folds,
    validate_multitarget_cv_support,
    validate_multitarget_holdout_coverage,
    validate_multitarget_model_cv_support,
    validate_multitarget_regression_support,
)

RESULTS_DIR = Path(__file__).parent / "results"

T = TypeVar("T")

# https://github.com/Lightning-AI/pytorch-lightning/issues/3431#issuecomment-2130390858
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")


class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger.addFilter(IgnorePLFilter())


def listify(item: Union[T, list[T], tuple[T, ...]]) -> List[T]:
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return [i for i in item]
    return [item]


def sort_df(df: DataFrame) -> DataFrame:
    """Auto-detect if classification or regression based on columns and sort"""
    cols = [c.lower() for c in df.columns]
    is_regression = False
    sort_col = None
    for col in cols:
        if ("mae" in col) and ("sd" not in col):
            is_regression = True
            sort_col = col
            break
    if sort_col is None:
        return df
    ascending = is_regression
    return df.sort_values(by=sort_col, ascending=ascending)


def print_sorted(df: DataFrame) -> None:
    """Auto-detect if classification or regression based on columns"""
    cols = [c.lower() for c in df.columns]
    is_regression = None
    sort_col = None
    for col in cols:
        if "mae" in col:
            is_regression = True
            break
        if "acc" in col:
            is_regression = False
            break
    if is_regression is None:
        print(df.to_markdown(tablefmt="simple", floatfmt="0.3f"))
        return
    sort_col = "mae" if is_regression else "acc"
    ascending = is_regression
    table = df.sort_values(by=sort_col, ascending=ascending).to_markdown(
        tablefmt="simple", floatfmt="0.3f"
    )
    print(table)


def log_options(options: ProgramOptions) -> None:
    opts = deepcopy(options.__dict__)

    print("Will run analyses with options:")
    pprint(opts, indent=2, depth=2, compact=False)


def _runtime_components(options: ProgramOptions) -> dict[str, RuntimeComponent]:
    components = {
        "preprocessing": RuntimeComponent.Preprocessing,
        "selection": RuntimeComponent.Selection,
        "univariate": RuntimeComponent.Univariate,
    }
    model_components = {
        "catboost": ("catboost", RuntimeComponent.CatBoost),
        "xgb": ("xgboost", RuntimeComponent.XGBoost),
        "knn": ("knn", RuntimeComponent.KNN),
        "tabpfn": ("tabpfn", RuntimeComponent.TabPFN),
        "mlp": ("mlp", RuntimeComponent.MLP),
        "kan": ("kan", RuntimeComponent.KAN),
        "gandalf": ("gandalf", RuntimeComponent.Gandalf),
        "lgbm": ("lgbm", RuntimeComponent.LightGBM),
        "rf": ("rf", RuntimeComponent.LightGBM),
        "dummy": ("dummy", RuntimeComponent.Sklearn),
        "lr": ("lr", RuntimeComponent.Sklearn),
        "sgd": ("sgd", RuntimeComponent.Sklearn),
        "svm": ("svm", RuntimeComponent.Sklearn),
        "elastic": ("elastic", RuntimeComponent.Sklearn),
        "dtree": ("dtree", RuntimeComponent.Sklearn),
        "et": ("et", RuntimeComponent.Sklearn),
    }
    sources = options.classifiers if options.is_classification else options.regressors
    for source in sources:
        name = getattr(source, "value", str(source))
        model_component = model_components.get(name)
        if model_component is not None:
            component_name, component = model_component
            components[component_name] = component
    wrapper = getattr(options, "wrapper_select", None)
    wrapper_model = getattr(options, "wrapper_model", None)
    if wrapper is not None and getattr(wrapper_model, "value", wrapper_model) == "knn":
        components["knn"] = RuntimeComponent.KNN
    if bool(getattr(options, "error_consistency", False)):
        components["error-consistency"] = RuntimeComponent.ErrorConsistency
    return components


def _canonical_device_name(name: object) -> str:
    value = str(name)
    return {"xgb": "xgboost"}.get(value, value)


def _runtime_records(
    options: ProgramOptions,
    fold_idx: Optional[int] = None,
) -> list[dict[str, object]]:
    records = list(getattr(options, "_runtime_model_audit", []))
    if fold_idx is None:
        return records
    return [record for record in records if record.get("fold") == fold_idx]


def _planned_devices(options: ProgramOptions) -> dict[str, str]:
    return {
        name: options.runtime.decision_for(component).resolved
        for name, component in _runtime_components(options).items()
    }


def _planned_device_decisions(
    options: ProgramOptions,
) -> dict[str, dict[str, object]]:
    return {
        name: options.runtime.decision_for(component).to_dict()
        for name, component in _runtime_components(options).items()
    }


def _resolved_devices(
    options: ProgramOptions,
    fold_idx: Optional[int] = None,
) -> dict[str, str]:
    resolved = _planned_devices(options)
    records = _runtime_records(options, fold_idx)
    model_names = {
        _canonical_device_name(record.get("model", ""))
        for record in records
        if record.get("model")
    }
    for model_name in model_names:
        completed = [
            str(record.get("resolved"))
            for record in records
            if _canonical_device_name(record.get("model", "")) == model_name
            and record.get("stage") == "completed"
        ]
        if completed:
            devices = sorted(set(completed))
            resolved[model_name] = (
                devices[0] if len(devices) == 1 else "mixed"
            )
        else:
            resolved[model_name] = "failed"
    ec_backends = [
        backend
        for backend in getattr(options, "_ec_backends", [])
        if fold_idx is None or backend.get("fold") == fold_idx
    ]
    if ec_backends:
        devices = {
            (
                "cuda"
                if backend.get("ec_backend_resolved") == "torch_cuda"
                else "cpu"
            )
            for backend in ec_backends
        }
        resolved["error-consistency"] = (
            next(iter(devices)) if len(devices) == 1 else "mixed"
        )
    return resolved


def _device_decisions(
    options: ProgramOptions,
    fold_idx: Optional[int] = None,
) -> dict[str, dict[str, object]]:
    runtime = options.runtime
    decisions = _planned_device_decisions(options)
    records = _runtime_records(options, fold_idx)
    model_names = {
        _canonical_device_name(record.get("model", ""))
        for record in records
        if record.get("model")
    }
    audit_fields = {
        "requested",
        "resolved",
        "reason",
        "n_samples",
        "n_features",
        "n_queries",
        "work_metric",
        "work_items",
        "threshold",
    }
    for model_name in model_names:
        completed = [
            record
            for record in records
            if _canonical_device_name(record.get("model", "")) == model_name
            and record.get("stage") == "completed"
        ]
        if not completed:
            decisions[model_name] = {
                "requested": runtime.intent.value,
                "resolved": "failed",
                "reason": "all_model_tasks_failed",
            }
            continue
        devices = sorted(
            {str(record.get("resolved")) for record in completed}
        )
        if len(completed) == 1:
            decisions[model_name] = {
                key: value
                for key, value in completed[-1].items()
                if key in audit_fields
            }
        else:
            decisions[model_name] = {
                "requested": runtime.intent.value,
                "resolved": (
                    devices[0] if len(devices) == 1 else "mixed"
                ),
                "reason": "multiple_model_tasks",
                "devices": devices,
                "tasks": len(completed),
            }
    ec_backends = [
        backend
        for backend in getattr(options, "_ec_backends", [])
        if fold_idx is None or backend.get("fold") == fold_idx
    ]
    if ec_backends:
        devices = sorted(
            {
                (
                    "cuda"
                    if backend.get("ec_backend_resolved") == "torch_cuda"
                    else "cpu"
                )
                for backend in ec_backends
            }
        )
        decisions["error-consistency"] = {
            "requested": runtime.intent.value,
            "resolved": devices[0] if len(devices) == 1 else "mixed",
            "reason": "error_consistency_runtime_backends",
            "devices": devices,
            "tasks": len(ec_backends),
        }
    return decisions


def _runtime_snapshot(
    options: ProgramOptions, fold_idx: Optional[int]
) -> dict[str, object]:
    return {
        "fold": fold_idx,
        "planned_devices": _planned_devices(options),
        "resolved_devices": _resolved_devices(options, fold_idx),
        "planned_device_decisions": _planned_device_decisions(options),
        "device_decisions": _device_decisions(options, fold_idx),
    }


def _record_ec_backends(options: ProgramOptions, result) -> None:
    recorded = getattr(options, "_ec_backends", [])
    for backend in result.metadata.get("ec_backends", []):
        entry = {
            **backend,
            "fold": getattr(options, "_runtime_current_fold", None),
        }
        if entry not in recorded:
            recorded.append(entry)
    options._ec_backends = recorded
    for skipped in result.metadata.get("skipped_configurations", []):
        _record_partial_failure(
            options,
            component="error_consistency",
            reason=str(skipped.get("reason", "configuration skipped")),
            details={key: value for key, value in skipped.items() if key != "reason"},
        )


def _record_partial_failure(
    options: ProgramOptions,
    component: str,
    reason: str,
    details: Optional[dict[str, object]] = None,
) -> None:
    failure: dict[str, object] = {
        "component": str(component),
        "reason": str(reason),
    }
    if details:
        failure.update(details)
    recorded = getattr(options, "_partial_failures", [])
    if failure not in recorded:
        recorded.append(failure)
    options._partial_failures = recorded


def _write_run_timing(
    options: ProgramOptions,
    started_at: datetime,
    started_s: float,
    status: str,
    error: Optional[BaseException] = None,
) -> None:
    outdir = options.program_dirs.results
    if outdir is None:
        return
    device = getattr(options, "device", None)
    payload = {
        "total_seconds": round(perf_counter() - started_s, 6),
        "device_requested": getattr(device, "value", str(device)),
        "planned_devices": _planned_devices(options),
        "resolved_devices": _resolved_devices(options),
        "planned_device_decisions": _planned_device_decisions(options),
        "device_decisions": _device_decisions(options),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": getattr(options, "cli_args", " ".join(sys.argv)),
        "command_argv": getattr(options, "cli_argv", list(sys.argv)),
        "status": status,
        "runtime_folds": getattr(options, "_runtime_fold_audit", []),
        "runtime_models": getattr(options, "_runtime_model_audit", []),
        "model_failures": getattr(options, "_model_failures", []),
        "partial_failures": getattr(options, "_partial_failures", []),
        "error_consistency": {
            "enabled": bool(getattr(options, "error_consistency", False)),
            "backends": getattr(options, "_ec_backends", []),
        },
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    try:
        windows_io_path(outdir).mkdir(exist_ok=True, parents=True)
        windows_io_path(outdir / "run_timing.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        warn(f"Could not save run timing: {exc}")


def _adaptive_error_base_dir(
    prog_dirs,
    fold_idx: Optional[int],
    target_name: Optional[str] = None,
) -> Optional[Path]:
    if prog_dirs.results is None:
        return None
    base_dir = prog_dirs.results / "adaptive_error"
    if fold_idx is not None:
        base_dir = base_dir / f"test{fold_idx:02d}"
    if target_name is not None:
        base_dir = base_dir / str(prog_dirs._safe_target_name(target_name))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _error_consistency_base_dir(
    prog_dirs,
    fold_idx: Optional[int],
) -> Optional[Path]:
    if prog_dirs.results is None:
        return None
    base_dir = prog_dirs.results / "error_consistency"
    if fold_idx is not None:
        base_dir = base_dir / f"test{fold_idx:02d}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _run(options: ProgramOptions) -> None:
    # TODO:
    # get arguments
    # run analyses based on args
    # if options.verbosity.value > 0:
    #     log_options(options)

    # print(options.drops)
    # sys.exit(0)
    is_cls = options.is_classification
    options._runtime_fold_audit = []
    options._runtime_model_audit = []
    options._ec_backends = []
    options._model_failures = []
    options._model_successes = []
    options._partial_failures = []
    components = _runtime_components(options)
    validate_cuda_request(options.runtime, components)
    print(format_device_plan(options.runtime, components))
    print()
    prog_dirs = options.program_dirs
    targets = as_target_list(options.targets)
    target_spec: Union[str, list[str]] = targets[0] if len(targets) == 1 else targets
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops
    grouper = options.grouper
    test_size = options.test_val_size
    method = options.tests_method
    seed = options.seed
    downsampling_requested = (
        options.feat_downsample is not FeatureDownsampleMethod.None_
    )
    # joblib_cache = options.program_dirs.joblib_cache
    # if joblib_cache is not None:
    #     memory = Memory(location=joblib_cache)

    sparse_input = options.uses_svmlight_input()
    pre_downsampled = sparse_input or options.large_feature_mode
    has_external_tests = len(options.test_paths) > 0
    if sparse_input:
        if not downsampling_requested:
            raise ValueError("SVMlight input requires --feat-downsample.")
        prep_splits = sparse_prepared_splits(options)
        prepared = prep_splits[0][0]
    else:
        df = options.load_df()
        merges = options.merged_df(df)
        if merges is not None:
            merged_df, ix_train, ix_tests = merges
        else:
            merged_df, ix_train, ix_tests = (None, None, None)

        df, renames = sanitize_names(df, target_spec)
        if merged_df is not None:
            # Test frames have identical columns, so sanitizing them produces
            # the same mapping recorded from the training frame.
            merged_df = sanitize_names(merged_df, target_spec)[0]
        prog_dirs.save_renames(renames)
        categoricals = renames.rename_columns(categoricals)
        ordinals = renames.rename_columns(ordinals)
        drops = renames.rename_columns(drops)
        options.categoricals = categoricals
        options.ordinals = ordinals
        options.drops = drops

        raw_df = merged_df if merged_df is not None else df
        if options.large_feature_mode:
            prep_splits = large_table_prepared_splits(
                raw_df, options, ix_train, ix_tests
            )
            prepared = prep_splits[0][0]
            del raw_df, df, merged_df
            gc.collect()
        else:
            if merged_df is None:
                idx_train, idx_test, split_audit = raw_train_test_indices(
                    raw_df,
                    target_spec,
                    grouper,
                    is_cls,
                    test_size,
                    seed,
                )
                split_specs = [(idx_train, [idx_test], split_audit)]
            elif method is ValidationMethod.List:
                if ix_train is None or ix_tests is None:
                    raise RuntimeError("Missing external train/test row indices.")
                split_specs = [(ix_train, ix_tests, None)]
            elif method is ValidationMethod.LODO:
                if ix_train is None or ix_tests is None:
                    raise RuntimeError("Missing external train/test row indices.")
                partitions = [ix_train, *ix_tests]
                split_specs = []
                # For LODO, train on each supplied partition in turn and
                # validate on all the others combined.
                for train_idx, idx_train in enumerate(partitions):
                    test_parts = partitions[:train_idx] + partitions[train_idx + 1 :]
                    split_specs.append(
                        (idx_train, [np.concatenate(test_parts)], None)
                    )
            else:
                raise ValueError(f"Invalid external validation method: {method}")

            prepared_df = raw_df.drop(columns=drops, errors="ignore")
            prep_splits = []
            for prep_idx, (idx_train, idx_tests, split_audit) in enumerate(
                split_specs
            ):
                inspection_indices = usable_training_indices(
                    raw_df, target_spec, is_cls, idx_train
                )
                _, inspection = inspect_data(
                    raw_df.iloc[inspection_indices].reset_index(drop=True),
                    target_spec,
                    grouper,
                    categoricals,
                    ordinals,
                    drops,
                    _warn=True,
                )
                report_fold = (
                    prep_idx
                    if has_external_tests and method is ValidationMethod.LODO
                    else None
                )
                prog_dirs.save_inspect_reports(inspection, report_fold)
                prog_dirs.save_inspect_tables(inspection, report_fold)
                prepared_fold = prepare_data(
                    prepared_df,
                    target_spec,
                    grouper,
                    inspection,
                    is_cls,
                    idx_train,
                    idx_tests,
                    ValidationMethod.List,
                )
                if split_audit is not None and prepared_fold.info is not None:
                    prepared_fold.info.split_audit = split_audit
                if not (
                    downsampling_requested
                    and options.skip_full_prepared_save_before_downsample
                ):
                    prog_dirs.save_prepared_raw(prepared_fold, report_fold)
                prog_dirs.save_prep_report(
                    prepared_fold.to_markdown(), report_fold
                )
                for prep_train, prep_test in prepared_fold.get_splits(
                    test_size=test_size, seed=seed
                ):
                    prep_splits.append(
                        (prepared_fold, prep_train, prep_test)
                    )

    for fold_idx, split in enumerate(prep_splits):
        if pre_downsampled:
            prep_train, prep_test, precomputed_downsample = split
            prepared = prep_train
        else:
            prepared, prep_train, prep_test = split
            precomputed_downsample = None
        final_cv_folds = resolve_final_cv_folds(prep_test.y, prep_test.groups)
        if final_cv_folds < 5:
            unit = "samples" if prep_test.groups is None else "distinct groups"
            warn(
                "Final holdout cross-validation was reduced from 5 to "
                f"{final_cv_folds} folds because the holdout contains only "
                f"{final_cv_folds} {unit}. The folds remain disjoint, but "
                "performance estimates may be unstable; use more holdout "
                "validation units when feasible."
            )
        if is_cls and isinstance(prep_train.y, DataFrame):
            validate_multitarget_model_cv_support(prep_train.y, options.models)
            validate_multitarget_holdout_coverage(
                prep_train.y,
                prep_test.y,
                external=has_external_tests,
            )
            validate_multitarget_cv_support(
                prep_test.y,
                n_splits=final_cv_folds,
                phase="final holdout cross-validation",
            )
        elif isinstance(prep_train.y, DataFrame):
            validate_multitarget_regression_support(
                prep_train.y, phase="outer training partition"
            )
            validate_multitarget_regression_support(
                prep_test.y, phase="outer holdout partition"
            )
        # prep_train, prep_test = prepared.split()
        if not has_external_tests:
            fold_idx = None
        options._runtime_current_fold = fold_idx
        if pre_downsampled and prep_train.info is not None:
            prog_dirs.save_prep_report(prep_train.to_markdown(), fold_idx)

        downsample_result: Optional[FeatureDownsampleResult] = precomputed_downsample
        if downsampling_requested and not pre_downsampled:
            prep_train, prep_test, downsample_result = downsample_split(
                prep_train, prep_test, options
            )
        if downsample_result is not None:
            prog_dirs.save_downsampling(downsample_result, fold_idx)
            # Describe prepared features after any requested downsampling.
            if fold_idx in (None, 0):
                if isinstance(prep_train.y, DataFrame):
                    for target_name in prep_train.target_cols:
                        desc_cont, desc_cat, desc_target = prep_train.for_target(
                            target_name
                        ).describe_features()
                        prog_dirs.save_feature_descriptions(
                            desc_cont, desc_cat, desc_target, target_name=target_name
                        )
                else:
                    desc_cont, desc_cat, desc_target = prep_train.describe_features()
                    prog_dirs.save_feature_descriptions(
                        desc_cont, desc_cat, desc_target
                    )
        elif not downsampling_requested and fold_idx in (None, 0):
            # Describe prepared features when no downsampling stage was requested.
            if isinstance(prep_train.y, DataFrame):
                for target_name in prep_train.target_cols:
                    desc_cont, desc_cat, desc_target = prep_train.for_target(
                        target_name
                    ).describe_features()
                    prog_dirs.save_feature_descriptions(
                        desc_cont,
                        desc_cat,
                        desc_target,
                        target_name=target_name,
                    )
            else:
                desc_cont, desc_cat, desc_target = prep_train.describe_features()
                prog_dirs.save_feature_descriptions(
                    desc_cont, desc_cat, desc_target
                )

        options.set_runtime_workload(len(prep_train.X), prep_train.X.shape[1])

        prep_selection = prep_train
        if downsample_result is not None and downsample_result.screening_rows:
            prep_selection = prep_train.subsample(
                np.asarray(downsample_result.screening_rows, dtype=int),
                validate=False,
            )

        split_audit = (
            None
            if prep_train.info is None
            else getattr(prep_train.info, "split_audit", None)
        )
        if split_audit is not None:
            prog_dirs.save_multitarget_split_report(
                split_audit.to_markdown(), fold_idx=fold_idx
            )

        if not isinstance(prep_selection.y, DataFrame):
            associations = target_associations(prep_selection)
            prog_dirs.save_univariate_assocs(associations, fold_idx)
            prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

            if options.no_preds:
                predictions = None
            else:
                predictions = univariate_predictions(prep_selection, is_cls)
                prog_dirs.save_univariate_preds(predictions, fold_idx)
                prog_dirs.save_pred_report(predictions.to_markdown(), fold_idx)

            if FeatureSelection.Filter in options.feat_select:
                assoc_filtered, pred_filtered = filter_select_features(
                    prep_selection, associations, predictions, options
                )
                prog_dirs.save_filter_report(assoc_filtered, fold_idx)
                prog_dirs.save_filter_report(pred_filtered, fold_idx)
            else:
                assoc_filtered, pred_filtered = None, None

            # TODO: make embedded and wrapper selection mutually exclusive. Only two
            # phases of feature selection: filter selection, and model-based
            # selection, wher model-based selection means either embedded or wrapper
            # (stepup, stepdown) methods.
            selected = model_select_features(prep_selection, options)
            prog_dirs.save_model_selection_reports(selected, fold_idx)
            prog_dirs.save_model_selection_data(selected, fold_idx)
        else:
            target_cols = prep_selection.target_cols
            per_target_assoc = {}
            per_target_pred = {}
            per_target_selected = {}
            has_pred_for_any_target = False

            for target_name in target_cols:
                prep_train_t = prep_selection.for_target(target_name)
                associations_t = target_associations(prep_train_t)
                prog_dirs.save_univariate_assocs(
                    associations_t, fold_idx=fold_idx, target=target_name
                )
                prog_dirs.save_assoc_report(
                    associations_t.to_markdown(), fold_idx=fold_idx, target=target_name
                )

                if options.no_preds:
                    predictions_t = None
                else:
                    predictions_t = univariate_predictions(prep_train_t, is_cls)
                    prog_dirs.save_univariate_preds(
                        predictions_t, fold_idx=fold_idx, target=target_name
                    )
                    prog_dirs.save_pred_report(
                        predictions_t.to_markdown(),
                        fold_idx=fold_idx,
                        target=target_name,
                    )

                if FeatureSelection.Filter in options.feat_select:
                    assoc_t, pred_t = filter_select_features(
                        prep_train_t, associations_t, predictions_t, options
                    )
                else:
                    assoc_t, pred_t = None, None
                selected_t = model_select_features(prep_train_t, options)
                per_target_assoc[target_name] = assoc_t
                per_target_pred[target_name] = pred_t
                per_target_selected[target_name] = selected_t
                if pred_t is not None:
                    has_pred_for_any_target = True

            mt_top_k = options.mt_top_k
            if isinstance(mt_top_k, int) and mt_top_k <= 0:
                mt_top_k = None

            if FeatureSelection.Filter not in options.feat_select:
                assoc_filtered = None
                pred_filtered = None
            else:
                assoc_filtered = aggregate_filter_selected(
                    [per_target_assoc[target_name] for target_name in target_cols],
                    method="association",
                    is_cls=is_cls,
                    strategy=options.mt_agg_strategy,
                    top_k=mt_top_k,
                    target_names=target_cols,
                )
                if options.no_preds or not has_pred_for_any_target:
                    pred_filtered = None
                else:
                    pred_parts = []
                    for target_name in target_cols:
                        pred_t = per_target_pred[target_name]
                        if pred_t is None:
                            pred_parts.append(
                                FilterSelected(
                                    selected=[],
                                    cont_scores=None,
                                    cat_scores=None,
                                    method="prediction",
                                    is_classification=is_cls,
                                )
                            )
                        else:
                            pred_parts.append(pred_t)
                    pred_filtered = aggregate_filter_selected(
                        pred_parts,
                        method="prediction",
                        is_cls=is_cls,
                        strategy=options.mt_agg_strategy,
                        top_k=mt_top_k,
                        target_names=target_cols,
                    )
            selected = aggregate_model_selected(
                [per_target_selected[target_name] for target_name in target_cols],
                is_cls=is_cls,
                strategy=options.mt_agg_strategy,
                top_k=mt_top_k,
                target_names=target_cols,
            )

            prog_dirs.save_filter_report(assoc_filtered, fold_idx)
            prog_dirs.save_filter_report(pred_filtered, fold_idx)
            prog_dirs.save_model_selection_reports(selected, fold_idx)
            prog_dirs.save_model_selection_data(selected, fold_idx)

        silence_spam()
        eval_results = evaluate_tuned(
            prepared=prepared,
            prep_train=prep_train,
            prep_test=prep_test,
            assoc_filtered=assoc_filtered,
            pred_filtered=pred_filtered,
            model_selected=selected,
            options=options,
            downsample_result=downsample_result,
        )
        for result in eval_results.results:
            if result.failure_reason is None:
                options._model_successes.append(
                    {
                        "fold": fold_idx,
                        "target": result.target,
                        "model": result.model.shortname,
                        "selection": result.selection,
                    }
                )
                continue
            options._model_failures.append(
                {
                    "fold": fold_idx,
                    "target": result.target,
                    "model": result.model.shortname,
                    "selection": result.selection,
                    "reason": result.failure_reason,
                }
            )
        prog_dirs.save_eval_report(eval_results, fold_idx)
        prog_dirs.save_eval_tables(eval_results, fold_idx)
        prog_dirs.save_eval_data(eval_results, fold_idx)
        if options.error_consistency:
            from df_analyze.analysis.error_consistency.runner import (
                combine_error_consistency_results,
                run_error_consistency_analysis,
            )
            from df_analyze.analysis.error_consistency.writer import write_root_outputs

            ec_base_dir = _error_consistency_base_dir(prog_dirs, fold_idx)
            if ec_base_dir is None:
                warn(
                    "No output directory is available; skipping error consistency analysis."
                )
            elif isinstance(prep_train.y, DataFrame):
                target_outputs = []
                skipped_targets = []
                for target_name in prep_train.target_cols:
                    prep_train_t = prep_train.for_target(target_name)
                    prep_test_t = prep_test.for_target(target_name)
                    eval_results_t = _eval_results_for_target(
                        eval_results=eval_results,
                        prep_train_t=prep_train_t,
                        prep_test_t=prep_test_t,
                        target=target_name,
                    )
                    try:
                        target_outputs.append(
                            run_error_consistency_analysis(
                                prep_train=prep_train_t,
                                prep_test=prep_test_t,
                                eval_results=eval_results_t,
                                options=options,
                                prog_dirs=prog_dirs,
                                base_dir=ec_base_dir,
                                write_root=False,
                            )
                        )
                    except RuntimeError as error:
                        if (
                            options.runtime.intent is DeviceIntent.CUDA
                            and is_cuda_runtime_error(error)
                        ):
                            raise
                        skipped_targets.append(
                            {
                                "scope": "target",
                                "target": str(target_name),
                                "model": "*",
                                "selection": "*",
                                "embed_selector": "*",
                                "reason": str(error),
                            }
                        )
                        warn(
                            "Skipping error-consistency target "
                            f"'{target_name}': {error}"
                        )
                if not target_outputs:
                    reasons = "; ".join(
                        f"{item['target']}: {item['reason']}"
                        for item in skipped_targets
                    )
                    raise RuntimeError(
                        "Error consistency did not complete for any target. "
                        f"{reasons or 'No targets were available.'}"
                    )
                combined_ec = combine_error_consistency_results(
                    target_outputs,
                    options=options,
                    skipped=skipped_targets,
                )
                write_root_outputs(ec_base_dir, combined_ec)
                _record_ec_backends(options, combined_ec)
            else:
                ec_result = run_error_consistency_analysis(
                    prep_train=prep_train,
                    prep_test=prep_test,
                    eval_results=eval_results,
                    options=options,
                    prog_dirs=prog_dirs,
                    base_dir=ec_base_dir,
                )
                _record_ec_backends(options, ec_result)
        if options.adaptive_error:
            from df_analyze.analysis.adaptive_error.runner import (
                run_adaptive_error_analysis,
            )

            base_dir = _adaptive_error_base_dir(prog_dirs, fold_idx=fold_idx)

            if isinstance(prep_train.y, DataFrame):
                target_cols = prep_train.target_cols

                for target_name in target_cols:
                    prep_train_t = prep_train.for_target(target_name)
                    prep_test_t = prep_test.for_target(target_name)
                    eval_results_t = _eval_results_for_target(
                        eval_results=eval_results,
                        prep_train_t=prep_train_t,
                        prep_test_t=prep_test_t,
                        target=target_name,
                    )

                    target_base_dir = _adaptive_error_base_dir(
                        prog_dirs, fold_idx=fold_idx, target_name=target_name
                    )

                    run_adaptive_error_analysis(
                        prep_train=prep_train_t,
                        prep_test=prep_test_t,
                        eval_results=eval_results_t,
                        options=options,
                        prog_dirs=prog_dirs,
                        no_preds=options.no_preds,
                        base_dir=target_base_dir,
                    )
                    if options.error_consistency and target_base_dir is not None:
                        from df_analyze.analysis.error_consistency.risk_stability import (
                            write_risk_stability_report,
                        )

                        ec_base_dir = _error_consistency_base_dir(prog_dirs, fold_idx)
                        if ec_base_dir is not None:
                            try:
                                write_risk_stability_report(
                                    target=str(target_name),
                                    eval_results=eval_results_t,
                                    options=options,
                                    aer_base_dir=target_base_dir,
                                    ec_base_dir=ec_base_dir,
                                )
                            except Exception as error:
                                _record_partial_failure(
                                    options,
                                    component="risk_stability",
                                    reason=str(error),
                                    details={"target": str(target_name)},
                                )
                                warn(
                                    "Could not write the AER/EC risk-stability "
                                    f"report for target '{target_name}': {error}"
                                )
            else:
                run_adaptive_error_analysis(
                    prep_train=prep_train,
                    prep_test=prep_test,
                    eval_results=eval_results,
                    options=options,
                    prog_dirs=prog_dirs,
                    no_preds=options.no_preds,
                    base_dir=base_dir,
                )
                if options.error_consistency and base_dir is not None:
                    from df_analyze.analysis.error_consistency.risk_stability import (
                        write_risk_stability_report,
                    )

                    ec_base_dir = _error_consistency_base_dir(prog_dirs, fold_idx)
                    if ec_base_dir is not None:
                        try:
                            write_risk_stability_report(
                                target=str(prep_test.target),
                                eval_results=eval_results,
                                options=options,
                                aer_base_dir=base_dir,
                                ec_base_dir=ec_base_dir,
                            )
                        except Exception as error:
                            _record_partial_failure(
                                options,
                                component="risk_stability",
                                reason=str(error),
                                details={"target": str(prep_test.target)},
                            )
                            warn(
                                "Could not write the AER/EC risk-stability "
                                f"report: {error}"
                            )
        try:
            print(eval_results.to_markdown())
        except ValueError as e:
            warn(
                f"Got error when attempting to print final report:\n{e}\n"
                f"Details:\n{traceback.format_exc()}"
            )
        options._runtime_fold_audit.append(
            _runtime_snapshot(options, fold_idx)
        )
    # TODO: Assemble final summary tables


def _all_predictive_models_failed(options: ProgramOptions) -> bool:
    failures = getattr(options, "_model_failures", [])
    failed_models = {
        str(failure.get("model", ""))
        for failure in failures
        if str(failure.get("model", "")) != "dummy"
    }
    if not failed_models:
        return False

    successes = getattr(options, "_model_successes", [])
    return not any(
        str(success.get("model", "")) != "dummy" for success in successes
    )


def _all_predictive_models_failed_message(options: ProgramOptions) -> str:
    failures = [
        failure
        for failure in getattr(options, "_model_failures", [])
        if str(failure.get("model", "")) != "dummy"
    ]
    failed_models = sorted({str(failure.get("model", "")) for failure in failures})
    names = ", ".join(failed_models) or "unknown"
    message = (
        f"All requested predictive models failed: {names}. "
        "No requested predictive-model result was produced."
    )
    details = []
    seen = set()
    for failure in failures:
        model = str(failure.get("model", "unknown"))
        selection = str(failure.get("selection", "")).strip()
        reason = str(failure.get("reason", "unknown failure")).strip()
        label = f"{model} / {selection}" if selection else model
        detail = f"{label}: {reason}"
        if detail not in seen:
            details.append(detail)
            seen.add(detail)
    if details:
        message += "\nModel failure details:\n- " + "\n- ".join(details)
    message += "\nThe same details are recorded in run_timing.json."
    return message


def main() -> None:
    options = get_options()
    options.to_json()
    started_at = datetime.now().astimezone()
    started_s = perf_counter()
    try:
        _run(options)
        if _all_predictive_models_failed(options):
            raise RuntimeError(_all_predictive_models_failed_message(options))
    except BaseException as error:
        _write_run_timing(options, started_at, started_s, "failed", error)
        raise
    status = (
        "completed_with_failures"
        if (
            getattr(options, "_model_failures", [])
            or getattr(options, "_partial_failures", [])
        )
        else "completed"
    )
    _write_run_timing(options, started_at, started_s, status)


if __name__ == "__main__":
    main()

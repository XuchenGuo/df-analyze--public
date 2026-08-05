from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

from pandas import DataFrame

from df_analyze._constants import SEED
from df_analyze.analysis.adaptive_error.base_models_selection import (
    _select_unique_models,
    _slug_from_options,
)
from df_analyze.analysis.adaptive_error.base_models_runner import (
    run_base_model_analyses,
)
from df_analyze.analysis.adaptive_error.ensemble_config import (
    _build_ensemble_strategy_params,
    _build_strategy_labels,
)
from df_analyze.analysis.adaptive_error.ensemble_runner import (
    run_ensemble_analysis,
)
from df_analyze.analysis.adaptive_error.report import (
    write_cross_model_summaries,
    write_run_config,
)
from df_analyze.saving import windows_io_path

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
    from df_analyze.hypertune import EvaluationResults
    from df_analyze.preprocessing.prepare import PreparedData
    from df_analyze.saving import ProgramDirs


def _get_seed(options: Any, default: int = SEED) -> int:
    for attr in ("seed", "random_state", "rng_seed"):
        val = getattr(options, attr, None)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return int(default)


def run_adaptive_error_analysis(
    prep_train: PreparedData,
    prep_test: PreparedData,
    eval_results: Optional[EvaluationResults],
    options: ProgramOptions,
    prog_dirs: ProgramDirs,
    no_preds: bool = False,
    base_dir: Optional[Path] = None,
) -> None:
    if isinstance(prep_train.y, DataFrame) and prep_train.y.shape[1] > 1:
        raise RuntimeError(
            "Adaptive error analysis currently supports one target at a time. "
        )
    if not options.is_classification:
        warn("Adaptive error analysis is classification only for now; skipping.")
        return
    if eval_results is None or not getattr(eval_results, "results", None):
        warn("No tuned results available; skipping adaptive error analysis.")
        return

    if base_dir is None:
        results_dir = getattr(prog_dirs, "results", None)
        if results_dir is not None:
            base_dir = Path(results_dir) / "adaptive_error"
    if base_dir is None:
        legacy_base_dir = getattr(prog_dirs, "adaptive_error", None)
        if legacy_base_dir is not None:
            base_dir = Path(legacy_base_dir)
    if base_dir is None:
        root_dir = getattr(prog_dirs, "root", None)
        if root_dir is None:
            warn("No output root directory available; skipping adaptive error analysis.")
            return
        base_dir = Path(root_dir) / "results" / "adaptive_error"
    windows_io_path(base_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = base_dir / "plots"
    tables_dir = base_dir / "tables"
    preds_dir = base_dir / "predictions"
    for d in (plots_dir, tables_dir, preds_dir):
        windows_io_path(d).mkdir(parents=True, exist_ok=True)

    # the global `--no-preds` flag
    no_preds = bool(no_preds) or bool(getattr(options, "no_preds", False))
    seed = _get_seed(options)
    # exclude `dummy` before applying `aer_top_k`.
    top_results = _select_unique_models(
        eval_results.results,
        metric_pref=options.htune_cls_metric,
        options=options,
        top_k=0,
    )
    if not top_results:
        warn("No tuned results available; skipping adaptive error analysis.")
        return
    # compute holdout test accuracy
    slugs_all = [_slug_from_options(res.model_cls, options) for res in top_results]
    filtered = [
        (res, slug)
        for res, slug in zip(top_results, slugs_all)
        if str(slug).lower() != "dummy"
    ]
    if len(filtered) != len(top_results):
        warn("Skipping dummy model in adaptive error analysis.")
    if not filtered:
        warn("Only dummy model available; skipping adaptive error analysis.")
        return

    try:
        aer_top_k = int(getattr(options, "aer_top_k", 0) or 0)
    except (TypeError, ValueError):
        aer_top_k = 0
    if aer_top_k > 0:
        filtered = filtered[: min(aer_top_k, len(filtered))]

    top_results, slugs = zip(*filtered)
    top_results = list(top_results)
    slugs = list(slugs)

    base_result = run_base_model_analyses(
        top_results=top_results,
        slugs=slugs,
        prep_train=prep_train,
        prep_test=prep_test,
        options=options,
        base_dir=base_dir,
        no_preds=no_preds,
        seed=seed,
    )

    write_cross_model_summaries(
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        preds_dir=preds_dir,
        base_test=base_result.base_test,
        slugs=slugs,
        compare_tests=base_result.compare_tests,
        compare_bins=base_result.compare_bins,
        compare_test_bins=base_result.compare_test_bins,
        compare_metrics=base_result.compare_metrics,
        model_rows=base_result.model_rows,
        no_preds=no_preds,
    )

    strategy_labels = _build_strategy_labels(options)
    strategy_params = _build_ensemble_strategy_params(options)
    write_run_config(
        base_dir=base_dir,
        options=options,
        seed=seed,
        best_slug=base_result.best_slug,
        model_run_info=base_result.model_run_info,
        no_preds=no_preds,
        strategy_labels=strategy_labels,
        strategy_params=strategy_params,
    )

    run_ensemble_analysis(
        base_dir=base_dir,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        prep_train=prep_train,
        prep_test=prep_test,
        options=options,
        no_preds=no_preds,
        ensemble_models=base_result.ensemble_models,
        best_slug=base_result.best_slug,
        best_curve_test=base_result.best_curve_test,
        seed=seed,
    )

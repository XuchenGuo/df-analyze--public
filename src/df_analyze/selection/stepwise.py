from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Type, Union

from joblib import Parallel, delayed

if TYPE_CHECKING:
    from df_analyze.cli.cli import ProgramOptions
from dataclasses import dataclass
from math import ceil
from time import perf_counter

import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm

from df_analyze._constants import DEFAULT_N_STEPWISE_SELECT
from df_analyze.cli.cli import ProgramOptions
from df_analyze.enumerables import Scorer, WrapperSelectionModel
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.knn import KNNClassifier, KNNRegressor
from df_analyze.models.lgbm import LightGBMClassifier, LightGBMRegressor
from df_analyze.models.linear import ElasticNetRegressor, SGDClassifierSelector
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.runtime.hardware import RuntimeComponent, RuntimePolicy, get_runtime
from df_analyze.splitting import OmniKFold


@dataclass
class RedundantFeatures:
    best: str
    best_score: float
    features: list[str]
    scores: list[float]
    metric: str

    def n_feat(self) -> int:
        return len(self.features)

    def to_markdown_section(self, iteration: int, is_cls: bool) -> str:
        lines = []
        if iteration == 0:
            lines.append("# Redundant Stepwise Selection Results\n\n")
            lines.append(f"Metric: {self.metric}\n\n")

        lines.append(f"* {self.best} ({self.metric}={self.best_score:0.5f}) - ")
        lines.append(f"[Iteration {iteration: >3d}]\n\n")
        if len(self.features) <= 1:  # best feature always included...
            return "".join(lines)[:-1]  # ignore last \n

        df = DataFrame(
            data=self.scores,
            index=Series(self.features, name="redundants"),
            columns=["score"],
        ).sort_values(by="score", ascending=not is_cls)
        lines.append(df.to_markdown(tablefmt="simple", floatfmt="0.4f", index=True))
        lines.append("\n\n")
        return "".join(lines)


@dataclass
class WrapperScoreResult:
    score: float
    audit: list[dict[str, object]]


def _largest_fold_workload(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, int]:
    if not splits:
        raise RuntimeError(
            "Wrapper selection could not create any cross-validation folds."
        )
    train_idx, query_idx = max(
        splits,
        key=lambda split: len(split[0]) * len(split[1]),
    )
    return len(train_idx), len(query_idx)


def _wrapper_cv_splits(
    y: Series,
    groups: Optional[Series],
    *,
    is_classification: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = OmniKFold(
        n_splits=5,
        is_classification=is_classification,
        grouped=groups is not None,
        labels=None,
        warn_on_fallback=False,
        allow_group_fallback=False,
        df_analyze_phase="Tuning CV Score",
    )
    splits = splitter.split(y.to_frame(), y.copy(), groups)[0]
    if not splits:
        raise RuntimeError(
            "Wrapper selection could not create any cross-validation folds."
        )
    return splits


def _wrapper_audit_record(
    *,
    model_name: str,
    candidate: str,
    is_forward: bool,
    runtime: RuntimePolicy,
    component: RuntimeComponent,
    attempt: int,
    stage: str,
    error: Optional[BaseException] = None,
) -> dict[str, object]:
    direction = "forward" if is_forward else "backward"
    record: dict[str, object] = {
        "fold": None,
        "model": model_name,
        "selection": f"wrapper-{direction}:{candidate}",
        "phase": "wrapper",
        "attempt": attempt,
        "stage": stage,
        **runtime.decision_for(component).to_dict(),
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    return record


# def get_dfanalyze_score(
#     model_cls: Type[DfAnalyzeModel],
#     X: DataFrame,
#     y: Series,
#     metric: Scorer,
#     selection_idx: ndarray,
#     feature_idx: int,
#     is_forward: bool,
#     test: bool,
# ) -> float:
#     candidate_idx = selection_idx.copy()
#     candidate_idx[feature_idx] = True
#     if not is_forward:
#         candidate_idx = ~candidate_idx

#     X_new = X.loc[:, candidate_idx]
#     model = model_cls()
#     return model.cv_score(X_new, y, test=test, metric=metric)


def get_dfanalyze_score(
    model_cls: Type[DfAnalyzeModel],
    X: DataFrame,
    y: Series,
    g: Optional[Series],
    metric: Scorer,
    selected: set[str],
    candidate: str,
    is_forward: bool,
    test: bool,
    runtime: Optional[RuntimePolicy] = None,
    splits: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
) -> WrapperScoreResult:
    includes = selected.copy()
    includes.add(candidate)
    includes = sorted(includes)
    X_new = X.loc[:, includes] if is_forward else X.drop(columns=includes)
    X_new = X_new.copy()
    g = None if g is None else g.copy()
    base_runtime = runtime or get_runtime("cpu")
    if splits is None:
        n_train = len(X_new)
        n_queries = ceil(len(X_new) / 5)
    else:
        n_train, n_queries = _largest_fold_workload(splits)
    actual_runtime = base_runtime.for_task(
        n_train,
        X_new.shape[1],
        n_queries=n_queries,
    )
    component = getattr(model_cls, "runtime_component", RuntimeComponent.Sklearn)
    decision = actual_runtime.decision_for(component)
    if decision.resolved == "cpu" and model_cls in (KNNClassifier, KNNRegressor):
        # CPU wrapper candidates parallelize at the outer candidate level.
        # Keep each sklearn KNN fold single-threaded to avoid nested pools.
        model = model_cls(model_args={"n_jobs": 1})
    else:
        model = model_cls()
    model.set_runtime(actual_runtime)
    model_name = getattr(model, "shortname", model_cls.__name__)
    audit: list[dict[str, object]] = []
    try:
        score = model.cv_score(
            X_new,
            y.copy(),
            g,
            test=test,
            metric=metric,
            splits=splits,
        )
        audit.append(
            _wrapper_audit_record(
                model_name=model_name,
                candidate=candidate,
                is_forward=is_forward,
                runtime=actual_runtime,
                component=component,
                attempt=1,
                stage="completed",
            )
        )
        return WrapperScoreResult(score=score, audit=audit)
    except Exception as error:
        audit.append(
            _wrapper_audit_record(
                model_name=model_name,
                candidate=candidate,
                is_forward=is_forward,
                runtime=actual_runtime,
                component=component,
                attempt=1,
                stage="failed",
                error=error,
            )
        )
        raise


def n_feat_int(prepared: PreparedData, n_features: Union[int, float, None]) -> int:
    n_feat = prepared.X.shape[1]
    if n_features is None:
        return min(n_feat - 1, DEFAULT_N_STEPWISE_SELECT)
    if isinstance(n_features, float):
        return min(n_feat - 1, ceil(n_features * n_feat))
    return min(n_feat - 1, n_features)


class StepwiseSelector:
    def __init__(
        self,
        prep_train: PreparedData,
        options: ProgramOptions,
        n_features: Union[int, float, None] = None,
        direction: Literal["forward", "backward"] = "forward",
        test: bool = False,
    ) -> None:
        self.is_forward = direction == "forward"
        self.n_features: int = n_feat_int(prep_train, n_features)
        self.total_feats: int = prep_train.X.shape[1]
        self.prepared = prep_train
        self.options = options
        self.model = options.wrapper_model
        self.redundant = options.redundant_selection
        self.test = test

        # selection_idx is True for selected/excluded features in forward/backward select
        self.selection_idx = np.zeros(shape=self.total_feats, dtype=bool)
        self.scores: dict[str, float] = {}
        self.remaining: list[str] = prep_train.X.columns.to_list()
        self.ordered_scores: list[tuple[str, float]] = []
        self.candidate_scores: dict[str, float] = {}
        self.redundant_results: list[RedundantFeatures] = []
        self.redundant_early_stop: bool = False
        self.selected: set[str] = set()
        self.to_consider: set[str] = set(self.prepared.X.columns.tolist())
        self.cv_splits: Optional[list[tuple[np.ndarray, np.ndarray]]] = None

        self.n_iterations = (
            self.n_features if self.is_forward else self.total_feats - self.n_features
        )

    def _ensure_cv_splits(
        self,
    ) -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
        if self.test:
            return None
        if self.cv_splits is None:
            self.cv_splits = _wrapper_cv_splits(
                self.prepared.y,
                self.prepared.groups,
                is_classification=self.prepared.is_classification,
            )
        return self.cv_splits

    def _candidate_n_jobs(self) -> int:
        if self.options.wrapper_model is WrapperSelectionModel.KNN:
            if self.is_forward:
                n_features = len(self.selected) + 1
            else:
                n_features = self.total_feats - (len(self.selected) + 1)
            splits = self._ensure_cv_splits()
            if splits is None:
                n_train = len(self.prepared.X)
                n_queries = ceil(len(self.prepared.X) / 5)
            else:
                n_train, n_queries = _largest_fold_workload(splits)
            runtime = self.options.runtime.for_task(
                n_train,
                max(1, n_features),
                n_queries=n_queries,
            )
            return runtime.tuning_jobs(RuntimeComponent.KNN, -1)
        if self.options.wrapper_model is WrapperSelectionModel.LGBM:
            # LightGBM already parallelizes each fit across CPU cores. Running
            # candidate fits in parallel as well causes severe nested
            # oversubscription, especially on Windows.
            return 1
        return -1

    def _record_wrapper_audit(
        self,
        outcomes: Sequence[WrapperScoreResult],
        selected_idx: int,
    ) -> None:
        records: list[dict[str, object]] = []
        for idx, outcome in enumerate(outcomes):
            failed = any(record.get("stage") == "failed" for record in outcome.audit)
            if idx == selected_idx or failed:
                records.extend(dict(record) for record in outcome.audit)
        fold = getattr(self.options, "_runtime_current_fold", None)
        for record in records:
            record["fold"] = fold
        audit = getattr(self.options, "_runtime_model_audit", [])
        audit.extend(records)
        self.options._runtime_model_audit = audit

    def fit(self) -> None:
        ddesc = "Forward" if self.is_forward else "Backward"
        for _ in tqdm(
            range(self.n_iterations),
            total=self.n_iterations,  # type: ignore
            desc=f"{ddesc} feature selection: ",
            leave=True,
            position=0,
        ):  # type: ignore
            if not self.is_forward and len(self.to_consider) <= self.n_features:
                break
            if len(self.to_consider) == 0:
                self.redundant_early_stop = True
                break

            if not self.redundant:
                selected, score = self._get_best_new_feature()
            else:
                # TODO: save in self.selected the ordered features and scores
                results = self._get_best_new_features()
                selected = results.best
                score = results.best_score
                if not self.is_forward:
                    max_remove = len(self.to_consider) - self.n_features
                    ranked = sorted(
                        zip(results.features, results.scores),
                        key=lambda item: (item[0] == results.best, item[1]),
                        reverse=True,
                    )
                    removals = ranked[:max_remove]
                    results.features = [feature for feature, _ in removals]
                    results.scores = [value for _, value in removals]
                self.redundant_results.append(results)
                self.selected.update(results.features)
                self.to_consider.difference_update(results.features)

            self.scores[selected] = score
            self.ordered_scores.append((selected, score))
            self.selected.add(selected)
            self.to_consider.discard(selected)

        if not self.is_forward:
            self.selected = self.to_consider

    def estimate_runtime(self) -> float:
        if self.is_forward:
            # better approximation than using the first iteration
            orig = self.selected.copy()
            half = ceil(self.n_features / 2)
            self.selected = set(
                np.random.choice(
                    self.prepared.X.columns.tolist(), replace=False, size=half
                ).tolist()
            )
        start = perf_counter()
        self._get_best_new_feature()
        elapsed = perf_counter() - start
        if self.is_forward:
            self.selected = orig  # type: ignore

        total = self.n_iterations * elapsed
        return round(total / 60, 1)

    def _get_best_new_features(self) -> RedundantFeatures:
        """Return the set of selected features via the greedy method"""
        model_enum = self.options.wrapper_model
        is_cls = self.prepared.is_classification
        metric = (
            self.options.htune_cls_metric if is_cls else self.options.htune_reg_metric
        )
        if model_enum is WrapperSelectionModel.LGBM:
            model_cls = LightGBMClassifier if is_cls else LightGBMRegressor
        elif model_enum is WrapperSelectionModel.KNN:
            model_cls = KNNClassifier if is_cls else KNNRegressor
        else:
            model_cls = SGDClassifierSelector if is_cls else ElasticNetRegressor

        # loop only over un-flagged features
        candidates = list(self.to_consider.copy())
        splits = self._ensure_cv_splits()
        outcomes: list[WrapperScoreResult] = Parallel(n_jobs=self._candidate_n_jobs())(
            delayed(get_dfanalyze_score)(  # type: ignore
                model_cls=model_cls,
                X=self.prepared.X,
                y=self.prepared.y,
                g=self.prepared.groups,
                metric=metric,
                selected=self.selected,
                candidate=candidate,
                is_forward=self.is_forward,
                test=self.test,
                runtime=self.options.runtime,
                splits=splits,
            )
            for candidate in tqdm(
                candidates,
                total=len(candidates),
                desc="Getting best new feature",
                position=1,
            )
        )
        all_scores = [outcome.score for outcome in outcomes]
        scores = np.array(all_scores)
        feat_names = candidates
        self.candidate_scores = dict(zip(feat_names, scores.tolist()))
        # Now remember redundant selection can stop early, so

        best_idx = np.argmax(scores)
        self._record_wrapper_audit(outcomes, int(best_idx))
        best_score = scores[best_idx]
        best = feat_names[best_idx]
        # scores are such that higher is always better (negation already
        # applied to ensure this), and we take the abs of user-supplied
        # redundancy threshold, so we can just blindly subtract here
        selected_idx = scores >= (best_score - self.options.redundant_threshold)
        selected = np.array(feat_names)[selected_idx]
        scores_selected = scores[selected_idx]
        idx_sort = np.argsort(scores_selected)
        selected = selected[idx_sort].tolist()
        scores_selected = scores_selected[idx_sort].tolist()

        return RedundantFeatures(
            best=best,
            best_score=best_score,
            features=selected,
            scores=scores_selected,
            metric=metric.name,
        )

    def _get_best_new_feature(self) -> tuple[str, float]:
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        model_enum = self.options.wrapper_model
        is_cls = self.prepared.is_classification
        metric = (
            self.options.htune_cls_metric if is_cls else self.options.htune_reg_metric
        )
        if model_enum is WrapperSelectionModel.LGBM:
            model_cls = LightGBMClassifier if is_cls else LightGBMRegressor
        elif model_enum is WrapperSelectionModel.KNN:
            model_cls = KNNClassifier if is_cls else KNNRegressor
        else:
            model_cls = SGDClassifierSelector if is_cls else ElasticNetRegressor

        candidates = list(self.to_consider.copy())
        splits = self._ensure_cv_splits()
        outcomes: list[WrapperScoreResult] = Parallel(n_jobs=self._candidate_n_jobs())(
            delayed(get_dfanalyze_score)(  # type: ignore
                model_cls=model_cls,
                X=self.prepared.X,
                y=self.prepared.y,
                g=self.prepared.groups,
                metric=metric,
                selected=self.selected,
                candidate=candidate,
                is_forward=self.is_forward,
                test=self.test,
                runtime=self.options.runtime,
                splits=splits,
            )
            for candidate in tqdm(
                candidates,
                total=len(candidates),
                desc="Getting best new feature",
                position=1,
            )
        )
        scores = [outcome.score for outcome in outcomes]
        self.candidate_scores = dict(zip(candidates, scores))

        idx = np.argmax(scores)
        self._record_wrapper_audit(outcomes, int(idx))
        selected = candidates[idx]
        score = scores[idx]
        return selected, score


def stepwise_select(
    prep_train: PreparedData,
    options: ProgramOptions,
    test: bool = False,
) -> Optional[tuple[list[str], dict[str, float], list[RedundantFeatures], bool]]:
    if options.wrapper_select is None:
        return
    selector = StepwiseSelector(
        prep_train=prep_train,
        options=options,
        n_features=options.n_feat_wrapper,
        direction=options.wrapper_select.direction(),
        test=test,
    )
    selector.fit()

    if selector.is_forward:
        selected_feats = [fname for fname, _ in selector.ordered_scores]
        scores = dict(selector.ordered_scores)
    else:
        selected_feats = [
            fname for fname in prep_train.X.columns if fname in selector.selected
        ]
        scores = {
            fname: selector.candidate_scores[fname]
            for fname in selected_feats
            if fname in selector.candidate_scores
        }

    return (
        selected_feats,
        scores,
        selector.redundant_results,
        selector.redundant_early_stop,
    )

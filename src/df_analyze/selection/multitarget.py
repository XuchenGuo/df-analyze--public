# ref: Rank aggregation via Borda count: https://en.wikipedia.org/wiki/Borda_count

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pandas import Series

from df_analyze.enumerables import EmbedSelectionModel
from df_analyze.selection.embedded import EmbedSelected
from df_analyze.selection.filter import FilterSelected
from df_analyze.selection.models import ModelSelected
from df_analyze.selection.wrapper import WrapperSelected


def _feature_counts(feature_lists: Iterable[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for feats in feature_lists:
        counts.update(feats)
    return counts


def _sorted_features(counts: Counter[str], n_max: Optional[int] = None) -> list[str]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    feats = [name for name, _ in ordered]
    if n_max is not None:
        return feats[:n_max]
    return feats


def _resolve_target_keys(
    n_targets: int,
    target_names: Optional[Iterable[str]],
) -> tuple[list[str], Optional[list[str]]]:
    if target_names is None:
        return [f"t{idx}" for idx in range(n_targets)], None
    names = [str(name) for name in target_names]
    if len(names) != n_targets:
        raise ValueError(
            "Target name count does not match per-target selection result count."
        )
    return names, names


def aggregate_feature_lists(
    lists: Iterable[list[str]], n_max: Optional[int] = None
) -> list[str]:
    counts = _feature_counts(lists)
    if not counts:
        return []
    return _sorted_features(counts, n_max=n_max)


def _support_counts(score_by_target: dict[str, Series]) -> Series:
    if not score_by_target:
        return Series(dtype=float)
    feats: set[str] = set()
    for scores in score_by_target.values():
        feats.update(scores.index.to_list())
    if not feats:
        return Series(dtype=float)
    support = Series(0, index=sorted(feats), dtype=float)
    for scores in score_by_target.values():
        support.loc[scores.index] += 1
    return support


def _rank_aggregate(
    score_by_target: dict[str, Series],
    higher_is_better: bool,
    alpha: float = 1.0,
    n_targets: Optional[int] = None,
) -> Series:
    if not score_by_target:
        return Series(dtype=float)
    feats: set[str] = set()
    for scores in score_by_target.values():
        feats.update(scores.index.to_list())
    if not feats:
        return Series(dtype=float)
    all_features = sorted(feats)
    points = pd.DataFrame(index=all_features)
    for target, scores in score_by_target.items():
        if scores.empty:
            continue
        ranks = scores.rank(ascending=not higher_is_better, method="average")
        pts = ((len(ranks) - ranks + 1) / len(ranks)).astype(float).fillna(0.0)
        points[target] = pts.reindex(all_features, fill_value=0.0)
    if points.empty:
        return Series(dtype=float)
    support = (points > 0).sum(axis=1)
    denom = max(1, n_targets if n_targets is not None else len(score_by_target))
    stability = (support / denom).astype(float) ** float(alpha)
    agg_points = points.sum(axis=1)
    final = agg_points * stability
    return final.sort_values(ascending=False)


def _resolve_top_k(top_k: Optional[int]) -> Optional[int]:
    if top_k is None or top_k <= 0:
        return None
    return int(top_k)


def _apply_min_support(
    scores: Series, support: Series, min_support: float, n_targets: int
) -> Series:
    if scores.empty:
        return scores
    if min_support <= 0:
        return scores
    min_count = max(1, int(ceil(min_support * n_targets)))
    keep = support >= min_count
    if keep.empty:
        return scores.iloc[0:0]
    keep_idx = keep[keep].index
    return scores.loc[scores.index.intersection(keep_idx)]


def _metric_higher_is_better(metric_cls: type, metric_name: Optional[str]) -> bool:
    if metric_name is None:
        return True
    return metric_cls(metric_name).higher_is_better()


def _mean_scores(
    score_by_target: dict[str, Series],
    selected: list[str],
    name: Optional[str],
    higher_is_better: bool,
) -> Optional[Series]:
    if not score_by_target or not selected:
        return None
    df = pd.concat(score_by_target.values(), axis=1)
    mean_scores = df.mean(axis=1, skipna=True)
    mean_scores = mean_scores.reindex(selected).dropna()
    if name is not None:
        mean_scores.name = name
    mean_scores = mean_scores.sort_values(ascending=not higher_is_better)
    return mean_scores


def _normalize_ranked_scores(scores: Series) -> Series:
    if scores.empty:
        return scores
    max_val = float(scores.max())
    if not np.isfinite(max_val) or max_val <= 0:
        return scores
    return scores / max_val


def aggregate_filter_selected(
    per_target_selected: Iterable[FilterSelected],
    method: str,
    is_cls: bool,
    strategy: str = "borda",
    alpha: float = 1.0,
    min_support: float = 0.2,
    top_k: Optional[int] = None,
    target_names: Optional[Iterable[str]] = None,
) -> FilterSelected:
    per_target = list(per_target_selected)
    if not per_target:
        return FilterSelected(
            selected=[],
            cont_scores=None,
            cat_scores=None,
            method=method,
            is_classification=is_cls,
        )

    cont_scores_by_target: dict[str, Series] = {}
    cat_scores_by_target: dict[str, Series] = {}
    cont_name = None
    cat_name = None
    target_keys, _ = _resolve_target_keys(len(per_target), target_names)
    for idx, sel in enumerate(per_target):
        key = target_keys[idx]
        if sel.cont_scores is not None and not sel.cont_scores.empty:
            cont_scores_by_target[key] = sel.cont_scores
            if cont_name is None:
                cont_name = sel.cont_scores.name
        if sel.cat_scores is not None and not sel.cat_scores.empty:
            cat_scores_by_target[key] = sel.cat_scores
            if cat_name is None:
                cat_name = sel.cat_scores.name

    cont_cls, cat_cls = FilterSelected.get_cont_cat_metrics(
        method=method, is_classification=is_cls
    )
    cont_higher = _metric_higher_is_better(cont_cls, cont_name)
    cat_higher = _metric_higher_is_better(cat_cls, cat_name)
    n_targets = len(per_target)
    top_k_val = _resolve_top_k(top_k)

    use_freq = strategy == "freq" or (
        len(cont_scores_by_target) == 0 and len(cat_scores_by_target) == 0
    )
    if use_freq:
        counts = _feature_counts([sel.selected for sel in per_target])
        ranked = Series(counts, dtype=float).sort_values(ascending=False)
        ranked = _apply_min_support(ranked, ranked, min_support, n_targets)
        if top_k_val is not None:
            ranked = ranked.head(top_k_val)
        selected = ranked.index.to_list()
        cont_feats = {
            feat for scores in cont_scores_by_target.values() for feat in scores.index
        }
        cat_feats = {
            feat for scores in cat_scores_by_target.values() for feat in scores.index
        }
        cont_selected = [feat for feat in selected if feat in cont_feats]
        cat_selected = [feat for feat in selected if feat in cat_feats]
    else:
        cont_ranked = _rank_aggregate(
            cont_scores_by_target,
            higher_is_better=cont_higher,
            alpha=alpha,
            n_targets=n_targets,
        )
        cat_ranked = _rank_aggregate(
            cat_scores_by_target,
            higher_is_better=cat_higher,
            alpha=alpha,
            n_targets=n_targets,
        )
        cont_support = _support_counts(cont_scores_by_target)
        cat_support = _support_counts(cat_scores_by_target)
        cont_ranked = _apply_min_support(
            cont_ranked, cont_support, min_support, n_targets
        )
        cat_ranked = _apply_min_support(cat_ranked, cat_support, min_support, n_targets)
        cont_ranked = _normalize_ranked_scores(cont_ranked)
        cat_ranked = _normalize_ranked_scores(cat_ranked)
        combined = pd.concat([cont_ranked, cat_ranked]).sort_values(ascending=False)
        if top_k_val is not None:
            combined = combined.head(top_k_val)
        selected = combined.index.to_list()
        cont_selected = [feat for feat in selected if feat in cont_ranked.index]
        cat_selected = [feat for feat in selected if feat in cat_ranked.index]

    cont_scores = _mean_scores(
        cont_scores_by_target, cont_selected, cont_name, cont_higher
    )
    cat_scores = _mean_scores(cat_scores_by_target, cat_selected, cat_name, cat_higher)

    return FilterSelected(
        selected=selected,
        cont_scores=cont_scores,
        cat_scores=cat_scores,
        method=method,
        is_classification=is_cls,
    )


def aggregate_embed_selected(
    per_target: Iterable[list[EmbedSelected]],
    strategy: str = "borda",
    alpha: float = 1.0,
    min_support: float = 0.2,
    top_k: Optional[int] = None,
    target_names: Optional[Iterable[str]] = None,
) -> list[EmbedSelected]:
    per_target_list = list(per_target)
    if not per_target_list:
        return []
    n_targets = len(per_target_list)
    target_keys, _ = _resolve_target_keys(n_targets, target_names)
    score_maps: dict[EmbedSelectionModel, dict[str, Series]] = defaultdict(dict)
    selected_by_model: dict[EmbedSelectionModel, list[list[str]]] = defaultdict(list)
    is_cls_by_model: dict[EmbedSelectionModel, bool] = {}

    for idx, embeds in enumerate(per_target_list):
        if embeds is None:
            continue
        target_key = target_keys[idx]
        for embed in embeds:
            scores = Series(embed.scores, dtype=float)
            if embed.model is EmbedSelectionModel.Linear:
                scores = scores.abs()
            score_maps[embed.model][target_key] = scores
            selected_by_model[embed.model].append(embed.selected)
            is_cls_by_model.setdefault(embed.model, embed.is_classification)

    results: list[EmbedSelected] = []
    for model, score_by_target in score_maps.items():
        if strategy == "freq":
            counts = _feature_counts(selected_by_model.get(model, []))
            ranked = Series(counts, dtype=float).sort_values(ascending=False)
            ranked = _apply_min_support(ranked, ranked, min_support, n_targets)
        else:
            ranked = _rank_aggregate(
                score_by_target, higher_is_better=True, alpha=alpha, n_targets=n_targets
            )
            support = _support_counts(score_by_target)
            ranked = _apply_min_support(ranked, support, min_support, n_targets)
        top_k_val = _resolve_top_k(top_k)
        if top_k_val is not None:
            ranked = ranked.head(top_k_val)
        selected = ranked.index.to_list()
        mean_scores = _mean_scores(score_by_target, selected, None, True)
        scores_dict: dict[str, float] = {}
        if mean_scores is not None:
            for feat in mean_scores.index:
                scores_dict[feat] = float(mean_scores.loc[feat])
        if selected:
            results.append(
                EmbedSelected(
                    model=model,
                    selected=selected,
                    scores=scores_dict,
                    is_classification=is_cls_by_model.get(model, False),
                )
            )
    return results


def aggregate_wrapper_selected(
    per_target: Iterable[Optional[WrapperSelected]],
    strategy: str = "borda",
    alpha: float = 1.0,
    min_support: float = 0.2,
    top_k: Optional[int] = None,
    target_names: Optional[Iterable[str]] = None,
) -> Optional[WrapperSelected]:
    per_target_list = list(per_target)
    if not per_target_list:
        return None
    n_targets = len(per_target_list)
    target_keys, _ = _resolve_target_keys(n_targets, target_names)
    score_by_target: dict[str, Series] = {}
    selected_by_target: list[list[str]] = []
    first: Optional[WrapperSelected] = None
    early_stop = False
    for idx, selected in enumerate(per_target_list):
        if selected is None:
            continue
        if first is None:
            first = selected
        early_stop = early_stop or selected.early_stop
        score_by_target[target_keys[idx]] = Series(selected.scores, dtype=float)
        selected_by_target.append(selected.selected)
    if first is None or not score_by_target:
        return None

    # cv_score orients wrapper scores so larger values are always better.
    higher_is_better = True
    if strategy == "freq":
        counts = _feature_counts(selected_by_target)
        ranked = Series(counts, dtype=float).sort_values(ascending=False)
        ranked = _apply_min_support(ranked, ranked, min_support, n_targets)
    else:
        ranked = _rank_aggregate(
            score_by_target,
            higher_is_better=higher_is_better,
            alpha=alpha,
            n_targets=n_targets,
        )
        support = _support_counts(score_by_target)
        ranked = _apply_min_support(ranked, support, min_support, n_targets)
    top_k_val = _resolve_top_k(top_k)
    if top_k_val is not None:
        ranked = ranked.head(top_k_val)
    selected = ranked.index.to_list()
    mean_scores = _mean_scores(
        score_by_target, selected, None, higher_is_better=higher_is_better
    )
    scores_dict: dict[str, float] = {}
    if mean_scores is not None:
        for feat in mean_scores.index:
            scores_dict[feat] = float(mean_scores.loc[feat])

    return WrapperSelected(
        method=first.method,
        model=first.model,
        selected=selected,
        scores=scores_dict,
        redundants=[],
        early_stop=early_stop,
        is_classification=first.is_classification,
    )


def aggregate_model_selected(
    per_target_model_selected: Iterable[ModelSelected],
    is_cls: bool,
    strategy: str = "borda",
    alpha: float = 1.0,
    min_support: float = 0.2,
    top_k: Optional[int] = None,
    target_names: Optional[Iterable[str]] = None,
) -> ModelSelected:
    per_target = list(per_target_model_selected)
    per_target_embeds = [selected.embed_selected or [] for selected in per_target]
    per_target_wrap = [selected.wrap_selected for selected in per_target]

    embed_selected = aggregate_embed_selected(
        per_target_embeds,
        strategy=strategy,
        alpha=alpha,
        min_support=min_support,
        top_k=top_k,
        target_names=target_names,
    )
    if len(embed_selected) == 0:
        embed_selected = None
    wrap_selected = aggregate_wrapper_selected(
        per_target_wrap,
        strategy=strategy,
        alpha=alpha,
        min_support=min_support,
        top_k=top_k,
        target_names=target_names,
    )

    return ModelSelected(
        embed_selected=embed_selected,
        wrap_selected=wrap_selected,
    )

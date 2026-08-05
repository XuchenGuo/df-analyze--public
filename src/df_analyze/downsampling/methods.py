from __future__ import annotations

from time import perf_counter
from typing import Any, Optional, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.random_projection import SparseRandomProjection

from df_analyze._constants import (
    DIRECT_DOWNSAMPLE_MAX_FEATURES,
    DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
    DOWNSAMPLE_DENSE_PEAK_FACTOR,
    DOWNSAMPLE_MAX_CHUNK_BYTES,
    DOWNSAMPLE_MAX_DENSE_BYTES,
    DOWNSAMPLE_SCORE_LIMIT,
    LARGE_FEATURE_THRESHOLD,
    N_FEAT_DOWNSAMPLE_DEFAULT,
    SEED,
)
from df_analyze.downsampling.base import (
    auto_chunk_size,
    projection_n_components,
    resolve_n_features,
)
from df_analyze.downsampling.chunked import (
    aggregate_target_scores,
    chunked_f_test_scores,
    chunked_range_normalized_variance_scores,
    chunked_variance_scores,
)
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.ranking import (
    descending_rank_percentiles,
    fractional_top_k_votes,
    top_k_indices,
)
from df_analyze.downsampling.screening import (
    resolve_screening_split,
)
from df_analyze.enumerables import FeatureDownsampleMethod
from df_analyze.preprocessing.prepare import PreparedData

INDEXED_SAFE_METHODS = {
    FeatureDownsampleMethod.None_,
    FeatureDownsampleMethod.Auto,
    FeatureDownsampleMethod.Random,
    FeatureDownsampleMethod.Variance,
    FeatureDownsampleMethod.NormalizedVariance,
    FeatureDownsampleMethod.FTest,
    FeatureDownsampleMethod.RankEnsemble,
    FeatureDownsampleMethod.SelectorEnsemble,
    FeatureDownsampleMethod.StableRank,
}

STABLE_RANK_REPEATS = 20
STABLE_RANK_LARGE_REPEATS = 10
STABLE_RANK_SUBSAMPLE = 0.75

DIRECT_METHODS = {
    FeatureDownsampleMethod.MutualInfo,
    FeatureDownsampleMethod.Linear,
    FeatureDownsampleMethod.LGBM,
    FeatureDownsampleMethod.SVD,
    FeatureDownsampleMethod.SparseRandomProjection,
}


def _method(value: Any) -> FeatureDownsampleMethod:
    if value is None:
        return FeatureDownsampleMethod.None_
    if isinstance(value, FeatureDownsampleMethod):
        return value
    return FeatureDownsampleMethod(str(value))


def _target_columns(y: Series | DataFrame) -> list[Series]:
    if isinstance(y, DataFrame):
        return [y[col] for col in y.columns]
    return [y]


def _rank_indices(
    scores: NDArray[np.float64],
    n_select: int,
    threshold: Optional[float] = None,
    *,
    seed: int = SEED,
    tie_keys: Optional[Sequence[str]] = None,
) -> NDArray[np.int_]:
    return top_k_indices(
        scores,
        n_select,
        threshold,
        seed=seed,
        tie_keys=tie_keys,
    )


def _rank_score(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    return descending_rank_percentiles(scores)


def _effective_chunk_size(X: Any, requested: int) -> int:
    dtype = getattr(X, "dtype", None)
    itemsize = getattr(dtype, "itemsize", 8)
    return auto_chunk_size(
        n_rows=X.shape[0],
        itemsize=itemsize,
        max_chunk_bytes=DOWNSAMPLE_MAX_CHUNK_BYTES,
        requested=requested,
    )


def _direct_peak_bytes(n_rows: int, n_features: int) -> int:
    return n_rows * n_features * 8 * DOWNSAMPLE_DENSE_PEAK_FACTOR


def resolve_feature_downsample_method(
    requested: FeatureDownsampleMethod | str | None,
    n_samples: int,
    n_features: int,
    n_features_out: int,
    is_classification: bool,
    y: Optional[Series | DataFrame] = None,
) -> FeatureDownsampleMethod:
    method = _method(requested)
    if method is FeatureDownsampleMethod.None_ or n_features <= n_features_out:
        return FeatureDownsampleMethod.None_
    if method is not FeatureDownsampleMethod.Auto:
        return method
    if y is None or n_samples < 2:
        return FeatureDownsampleMethod.NormalizedVariance
    targets = _target_columns(y)
    if any(target.nunique(dropna=True) < 2 for target in targets):
        return FeatureDownsampleMethod.NormalizedVariance
    if n_features >= LARGE_FEATURE_THRESHOLD or n_features / max(n_samples, 1) >= 1_000:
        # With many more features than rows, repeat the supervised ranking
        # instead of mixing it equally with an unrelated variance ranking.
        return FeatureDownsampleMethod.StableRank
    return FeatureDownsampleMethod.FTest


def _auto_reason(
    requested: FeatureDownsampleMethod,
    resolved: FeatureDownsampleMethod,
    n_samples: int,
    n_features: int,
    n_select: int,
) -> Optional[str]:
    if requested is not FeatureDownsampleMethod.Auto:
        return None
    if resolved is FeatureDownsampleMethod.None_:
        return (
            "auto resolved to none because the requested feature count is greater "
            "than or equal to the input feature count."
        )
    if resolved is FeatureDownsampleMethod.NormalizedVariance:
        return (
            "auto resolved to normalized-variance because no target was available "
            "or the target could not support valid supervised screening."
        )
    if resolved is FeatureDownsampleMethod.StableRank:
        ratio = n_features / max(n_samples, 1)
        return (
            "auto resolved to stable-rank because the feature space is large "
            f"relative to the sample size (p/n={ratio:0.1f}); repeated "
            "subsample F-test rankings reduce sensitivity to one screening "
            "sample without mixing the supervised effect score with an "
            "untargeted variance heuristic."
        )
    return (
        "auto resolved to f-test because an informative target was available and "
        "the feature space was within the scalable first-pass range."
    )


def _result_strategy(
    resolved: FeatureDownsampleMethod,
    y: Optional[Series | DataFrame],
    n_features: int,
    full_ensemble: bool = False,
    ensemble_members: Optional[Sequence[str]] = None,
    large_feature_mode: bool = False,
) -> dict[str, Any]:
    aggregation = (
        "not_targeted"
        if resolved
        in {
            FeatureDownsampleMethod.None_,
            FeatureDownsampleMethod.Random,
            FeatureDownsampleMethod.Variance,
            FeatureDownsampleMethod.NormalizedVariance,
        }
        else (
            "mean_rank_percentile_across_targets"
            if isinstance(y, DataFrame) and y.shape[1] > 1
            else "single_target"
        )
    )
    details: dict[str, Any] = {
        "score_aggregation": aggregation,
        "large_feature_mode": large_feature_mode,
    }
    if resolved is FeatureDownsampleMethod.RankEnsemble:
        details.update(
            strategy="mean_rank_percentile",
            ensemble_members=(
                list(ensemble_members)
                if ensemble_members is not None
                else [
                    "range-normalized-variance",
                    "f-test",
                    "mutual-info",
                    "linear",
                    "lgbm",
                ]
                if full_ensemble
                else ["range-normalized-variance", "f-test"]
            ),
        )
    elif resolved is FeatureDownsampleMethod.SelectorEnsemble:
        details.update(
            strategy="top_k_vote_plus_rank_tiebreak",
            ensemble_members=(
                list(ensemble_members)
                if ensemble_members is not None
                else [
                    "range-normalized-variance",
                    "f-test",
                    "mutual-info",
                    "linear",
                    "lgbm",
                ]
                if full_ensemble
                else ["range-normalized-variance", "f-test"]
            ),
        )
    elif resolved is FeatureDownsampleMethod.StableRank:
        details.update(
            strategy="selection_frequency_then_mean_rank",
            ensemble_members=["f-test"],
            stability_repeats=(
                STABLE_RANK_LARGE_REPEATS
                if n_features >= LARGE_FEATURE_THRESHOLD
                else STABLE_RANK_REPEATS
            ),
            stability_subsample=STABLE_RANK_SUBSAMPLE,
        )
    return details


def _ensemble_scores(
    X: Any,
    y: Series | DataFrame,
    is_classification: bool,
    chunk_size: int,
    row_indices: Optional[NDArray[np.int_]],
    selector: bool,
    n_select: int,
    seed: int,
    allow_direct_members: bool,
) -> tuple[NDArray[np.float64], list[str]]:
    variance = chunked_range_normalized_variance_scores(X, chunk_size, row_indices)
    f_scores = chunked_f_test_scores(X, y, is_classification, chunk_size, row_indices)
    member_scores = [variance, f_scores]
    member_names = ["range-normalized-variance", "f-test"]
    direct_rows = X.shape[0] if row_indices is None else len(row_indices)
    direct_within_budget = (
        _direct_peak_bytes(direct_rows, X.shape[1]) <= DOWNSAMPLE_MAX_DENSE_BYTES
    )
    if (
        allow_direct_members
        and isinstance(X, DataFrame)
        and X.shape[1] <= DIRECT_DOWNSAMPLE_MAX_FEATURES
        and direct_within_budget
    ):
        X_direct = X if row_indices is None else X.iloc[row_indices]
        for method in (
            FeatureDownsampleMethod.MutualInfo,
            FeatureDownsampleMethod.Linear,
            FeatureDownsampleMethod.LGBM,
        ):
            try:
                scores = _aggregate_direct_scores(
                    X_direct, y, is_classification, method, seed
                )
                member_scores.append(scores)
                member_names.append(method.value)
            except (TypeError, ValueError, RuntimeError) as error:
                warn(
                    f"Skipping {method.value!r} in the feature ensemble because "
                    f"it could not produce scores: {error}"
                )
    ranked_members = [_rank_score(scores) for scores in member_scores]
    ranked = np.vstack(ranked_members)
    counts = np.sum(~np.isnan(ranked), axis=0)
    mean_rank = np.divide(
        np.nansum(ranked, axis=0),
        counts,
        out=np.full(ranked.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    if not selector:
        return mean_rank, member_names
    top = max(n_select, min(len(variance), n_select * 3))
    votes = np.zeros(len(variance), dtype=np.float64)
    for scores in member_scores:
        votes += fractional_top_k_votes(scores, top)
    return votes + mean_rank / 10.0, member_names


def _stable_rank_scores(
    X: Any,
    y: Series | DataFrame,
    is_classification: bool,
    chunk_size: int,
    row_indices: NDArray[np.int_],
    n_select: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    rank_quality_sums = np.zeros(X.shape[1], dtype=np.float64)
    rank_quality_counts = np.zeros(X.shape[1], dtype=np.int_)
    frequency = np.zeros(X.shape[1], dtype=np.float64)
    repeats = (
        STABLE_RANK_LARGE_REPEATS
        if X.shape[1] >= LARGE_FEATURE_THRESHOLD
        else STABLE_RANK_REPEATS
    )
    completed = 0
    for _ in range(repeats):
        size = max(2, int(np.ceil(STABLE_RANK_SUBSAMPLE * len(row_indices))))
        size = min(len(row_indices), size)
        if is_classification:
            sampled_local = _level_preserving_sample(y, size, rng)
        else:
            sampled_local = np.sort(
                rng.choice(len(row_indices), size=size, replace=False)
            )
        sampled_rows = row_indices[sampled_local]
        sampled_y = y.iloc[sampled_local]
        scores = chunked_f_test_scores(
            X, sampled_y, is_classification, chunk_size, sampled_rows
        )
        rank_quality = descending_rank_percentiles(scores)
        valid = ~np.isnan(rank_quality)
        if not valid.any():
            continue
        rank_quality_sums[valid] += rank_quality[valid]
        rank_quality_counts[valid] += 1
        frequency += fractional_top_k_votes(scores, n_select)
        completed += 1
    if completed == 0:
        return np.full(X.shape[1], np.nan, dtype=np.float64)
    mean_quality = np.divide(
        rank_quality_sums,
        rank_quality_counts,
        out=np.full(X.shape[1], np.nan, dtype=np.float64),
        where=rank_quality_counts > 0,
    )
    return frequency * (X.shape[1] + 1.0) + mean_quality


def _level_preserving_sample(
    y: Series | DataFrame,
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    required: set[int] = set()
    for target in _target_columns(y):
        values = target.reset_index(drop=True)
        for positions in values.groupby(values, dropna=False).indices.values():
            available = np.asarray(positions, dtype=int)
            required.update(
                rng.choice(
                    available,
                    size=min(2, len(available)),
                    replace=False,
                )
                .astype(int)
                .tolist()
            )

    sample_size = min(len(y), max(size, len(required)))
    remaining = np.setdiff1d(
        np.arange(len(y), dtype=int),
        np.fromiter(required, dtype=int),
        assume_unique=False,
    )
    n_fill = sample_size - len(required)
    if n_fill > 0:
        required.update(
            rng.choice(remaining, size=n_fill, replace=False).astype(int).tolist()
        )
    return np.asarray(sorted(required), dtype=int)


def _indexed_scores(
    method: FeatureDownsampleMethod,
    X: Any,
    y: Series | DataFrame,
    is_classification: bool,
    chunk_size: int,
    row_indices: Optional[NDArray[np.int_]],
    n_select: int,
    seed: int,
    allow_direct_ensemble_members: bool,
    ensemble_members: Optional[list[str]] = None,
) -> NDArray[np.float64]:
    if method is FeatureDownsampleMethod.Variance:
        return chunked_variance_scores(X, chunk_size, row_indices)
    if method is FeatureDownsampleMethod.NormalizedVariance:
        return chunked_range_normalized_variance_scores(X, chunk_size, row_indices)
    if method is FeatureDownsampleMethod.FTest:
        return chunked_f_test_scores(X, y, is_classification, chunk_size, row_indices)
    if method in {
        FeatureDownsampleMethod.RankEnsemble,
        FeatureDownsampleMethod.SelectorEnsemble,
    }:
        scores, actual_members = _ensemble_scores(
            X,
            y,
            is_classification,
            chunk_size,
            row_indices,
            selector=method is FeatureDownsampleMethod.SelectorEnsemble,
            n_select=n_select,
            seed=seed,
            allow_direct_members=allow_direct_ensemble_members,
        )
        if ensemble_members is not None:
            ensemble_members.extend(actual_members)
        return scores
    if method is FeatureDownsampleMethod.StableRank:
        if row_indices is None:
            row_indices = np.arange(X.shape[0], dtype=int)
        return _stable_rank_scores(
            X, y, is_classification, chunk_size, row_indices, n_select, seed
        )
    raise ValueError(f"Method {method.value!r} cannot score indexed columns.")


def _score_details(
    scores: Optional[NDArray[np.float64]],
    feature_names: Sequence[str],
    save_all: bool,
    seed: int,
) -> tuple[Optional[list[float]], Optional[list[int]], Optional[list[str]]]:
    if scores is None:
        return None, None, None
    # Keep only a bounded ranked summary here.  When requested, the complete
    # NumPy-backed vector is attached to the result and streamed by the saver.
    limit = min(DOWNSAMPLE_SCORE_LIMIT, len(scores))
    ranked = _rank_indices(
        scores,
        limit,
        seed=seed,
        tie_keys=feature_names,
    )
    return (
        [float(scores[idx]) for idx in ranked],
        ranked.astype(int).tolist(),
        [str(feature_names[idx]) for idx in ranked],
    )


def select_indexed_columns(
    X: Any,
    train_indices: NDArray[np.int_],
    y_train: Series | DataFrame,
    is_classification: bool,
    requested_method: FeatureDownsampleMethod | str | None,
    n_features_out: int | float = N_FEAT_DOWNSAMPLE_DEFAULT,
    chunk_size: int = DOWNSAMPLE_CHUNK_SIZE_DEFAULT,
    screening_positions: Optional[NDArray[np.int_]] = None,
    feature_names: Optional[Sequence[str]] = None,
    variance_threshold: Optional[float] = None,
    save_all_scores: bool = False,
    sparse_input: bool = False,
    input_format: str = "table",
    seed: int = SEED,
    large_feature_mode: bool = False,
    protected_indices: Optional[Sequence[int]] = None,
) -> tuple[NDArray[np.int_], FeatureDownsampleResult]:
    n_features = int(X.shape[1])
    n_select = resolve_n_features(n_features_out, n_features)
    requested = _method(requested_method)
    resolved = resolve_feature_downsample_method(
        requested,
        len(train_indices),
        n_features,
        n_select,
        is_classification,
        y_train,
    )
    if feature_names is None:
        names = (
            X.columns.astype(str).tolist()
            if isinstance(X, DataFrame)
            else [f"feature_{idx}" for idx in range(n_features)]
        )
    else:
        # Do not build millions of generated SVMlight names just to select a
        # small set of columns.
        names = feature_names
    if len(names) != n_features:
        raise ValueError("Feature names must match the input feature count.")
    raw_protected = [] if protected_indices is None else protected_indices
    protected = np.asarray(
        sorted(set(int(idx) for idx in raw_protected)),
        dtype=int,
    )
    if len(protected) > 0 and (int(protected[0]) < 0 or int(protected[-1]) >= n_features):
        raise ValueError("Protected feature indices must refer to source columns.")
    if len(protected) > n_select:
        raise ValueError(
            f"{len(protected):,} protected features exceed the requested output "
            f"count of {n_select:,}. Increase --n-feat-downsample."
        )
    n_ranked_select = n_select - len(protected)
    if (
        resolved is FeatureDownsampleMethod.None_
        and requested is FeatureDownsampleMethod.Variance
        and variance_threshold is not None
    ):
        resolved = FeatureDownsampleMethod.Variance
    if resolved not in INDEXED_SAFE_METHODS:
        raise ValueError(
            f"Feature downsampling method {resolved.value!r} requires a materialized "
            f"training matrix and is limited to {DIRECT_DOWNSAMPLE_MAX_FEATURES:,} "
            "features. Use variance, f-test, rank-ensemble, selector-ensemble, "
            "stable-rank, or random for indexed large-scale input."
        )

    started = perf_counter()
    scores: Optional[NDArray[np.float64]] = None
    actual_ensemble_members: list[str] = []
    notes: list[str] = []
    actual_chunk = _effective_chunk_size(X, chunk_size)
    if actual_chunk < int(chunk_size):
        budget_mib = DOWNSAMPLE_MAX_CHUNK_BYTES / 1024**2
        notes.append(
            f"Score chunk size was reduced from {int(chunk_size):,} to "
            f"{actual_chunk:,} to stay within the {budget_mib:g} MiB "
            "working-memory budget."
        )
    fit_rows = train_indices
    y_fit = y_train
    if screening_positions is not None:
        fit_rows = train_indices[screening_positions]
        y_fit = y_train.iloc[screening_positions]

    if resolved is FeatureDownsampleMethod.None_:
        selected = np.arange(n_features, dtype=int)
    elif resolved is FeatureDownsampleMethod.Random:
        variances = chunked_variance_scores(X, actual_chunk, train_indices)
        eligible = np.flatnonzero(np.isfinite(variances) & (variances > 0.0))
        if len(protected) > 0:
            eligible = np.setdiff1d(eligible, protected, assume_unique=True)
        if len(eligible) == 0:
            if n_ranked_select > 0:
                raise ValueError("No non-constant training features are available.")
            random_selected = np.array([], dtype=int)
        else:
            n_random = min(n_ranked_select, len(eligible))
            random_selected = np.sort(
                np.random.default_rng(seed).choice(eligible, n_random, replace=False)
            )
        selected = np.concatenate([protected, random_selected])
        if len(random_selected) < n_ranked_select:
            notes.append(
                "Random downsampling excluded constant or training-absent features, "
                f"leaving {len(random_selected):,} usable non-protected features."
            )
    elif n_ranked_select == 0:
        selected = protected.copy()
        notes.append(
            "The protected feature set filled the requested output count; no "
            "additional source features were ranked."
        )
        if save_all_scores:
            scores = _indexed_scores(
                resolved,
                X,
                y_fit,
                is_classification,
                actual_chunk,
                fit_rows,
                n_select,
                seed,
                not large_feature_mode,
                actual_ensemble_members,
            )
    else:
        scores = _indexed_scores(
            resolved,
            X,
            y_fit,
            is_classification,
            actual_chunk,
            fit_rows,
            n_select,
            seed,
            not large_feature_mode,
            actual_ensemble_members,
        )
        if resolved is FeatureDownsampleMethod.Variance:
            threshold = (
                variance_threshold
                if variance_threshold is not None
                else np.nextafter(0.0, np.inf)
            )
        elif resolved is FeatureDownsampleMethod.NormalizedVariance:
            threshold = np.nextafter(0.0, np.inf)
        else:
            threshold = None
        ranking_scores = scores
        if len(protected) > 0:
            ranking_scores = scores.copy()
            ranking_scores[protected] = -np.inf
        ranked_selected = _rank_indices(
            ranking_scores,
            n_ranked_select,
            threshold,
            seed=seed,
            tie_keys=names,
        )
        if len(ranked_selected) == 0:
            raise ValueError(
                "No features have usable downsampling scores or satisfy the requested "
                "score threshold."
            )
        selected = np.concatenate([protected, ranked_selected])

    score_values, score_indices, score_names = _score_details(
        scores, names, save_all_scores, seed
    )
    result = FeatureDownsampleResult(
        requested_method=requested.value,
        resolved_method=resolved.value,
        n_features_in=n_features,
        n_features_out=len(selected),
        selected_features=[str(names[idx]) for idx in selected],
        selected_indices=selected.astype(int).tolist(),
        protected_features=[str(names[idx]) for idx in protected],
        protected_indices=protected.astype(int).tolist(),
        scores=score_values,
        score_feature_indices=score_indices,
        score_feature_names=score_names,
        full_score_values=scores if save_all_scores else None,
        full_score_feature_names=names
        if save_all_scores and scores is not None
        else None,
        screening_rows=(
            [] if screening_positions is None else screening_positions.tolist()
        ),
        screening_samples=0 if screening_positions is None else len(fit_rows),
        tuning_samples=(
            len(y_train) - len(fit_rows)
            if screening_positions is not None
            else len(y_train)
        ),
        chunk_size=actual_chunk,
        sparse_input=sparse_input,
        input_format=input_format,
        fit_seconds=perf_counter() - started,
        notes=notes,
        auto_reason=_auto_reason(
            requested, resolved, len(train_indices), n_features, n_select
        ),
        **_result_strategy(
            resolved,
            y_train,
            n_features,
            full_ensemble=(
                isinstance(X, DataFrame) and n_features <= DIRECT_DOWNSAMPLE_MAX_FEATURES
            ),
            ensemble_members=(
                actual_ensemble_members
                if resolved
                in {
                    FeatureDownsampleMethod.RankEnsemble,
                    FeatureDownsampleMethod.SelectorEnsemble,
                }
                else None
            ),
            large_feature_mode=large_feature_mode,
        ),
    )
    return selected, result


def _aggregate_direct_scores(
    X: Any,
    y: Series | DataFrame,
    is_classification: bool,
    method: FeatureDownsampleMethod,
    seed: int = SEED,
) -> NDArray[np.float64]:
    target_scores: list[NDArray[np.float64]] = []
    for target in _target_columns(y):
        if method is FeatureDownsampleMethod.MutualInfo:
            scorer = mutual_info_classif if is_classification else mutual_info_regression
            values = scorer(X, target, random_state=seed)
        elif method is FeatureDownsampleMethod.Linear:
            if is_classification:
                model = SGDClassifier(loss="log_loss", random_state=seed, max_iter=2000)
            else:
                model = SGDRegressor(random_state=seed, max_iter=2000)
            model.fit(X, target)
            coef = np.asarray(model.coef_)
            values = np.max(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        elif method is FeatureDownsampleMethod.LGBM:
            model_cls = LGBMClassifier if is_classification else LGBMRegressor
            model = model_cls(
                n_estimators=50,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            )
            model.fit(X, target)
            values = model.feature_importances_
        else:
            raise ValueError(f"Unsupported direct scoring method: {method.value}")
        target_scores.append(np.asarray(values, dtype=np.float64))
    return aggregate_target_scores(target_scores)


def _downsample_frame(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: Series | DataFrame,
    is_classification: bool,
    requested_method: FeatureDownsampleMethod | str | None,
    n_features_out: int | float,
    chunk_size: int,
    screening_positions: Optional[NDArray[np.int_]],
    variance_threshold: Optional[float],
    save_all_scores: bool,
    seed: int,
) -> tuple[DataFrame, DataFrame, FeatureDownsampleResult]:
    n_features = X_train.shape[1]
    n_select = resolve_n_features(n_features_out, n_features)
    requested = _method(requested_method)
    resolved = resolve_feature_downsample_method(
        requested,
        len(X_train),
        n_features,
        n_select,
        is_classification,
        y_train,
    )
    if resolved in DIRECT_METHODS and n_features > DIRECT_DOWNSAMPLE_MAX_FEATURES:
        raise ValueError(
            f"Feature downsampling method {resolved.value!r} is limited to "
            f"{DIRECT_DOWNSAMPLE_MAX_FEATURES:,} materialized features."
        )
    fit_rows_count = (
        len(X_train) if screening_positions is None else len(screening_positions)
    )
    if (
        resolved in DIRECT_METHODS
        and _direct_peak_bytes(fit_rows_count, n_features) > DOWNSAMPLE_MAX_DENSE_BYTES
    ):
        gib = _direct_peak_bytes(fit_rows_count, n_features) / 1024**3
        raise MemoryError(
            f"Method {resolved.value!r} is estimated to require about {gib:.2f} "
            "GiB of peak dense fitting memory. Reduce the input size or use an "
            "indexed, column-chunked method such as normalized-variance, f-test, "
            "rank-ensemble, selector-ensemble, or stable-rank."
        )

    fit_started = perf_counter()
    X_fit = X_train if screening_positions is None else X_train.iloc[screening_positions]
    y_fit = y_train if screening_positions is None else y_train.iloc[screening_positions]
    names = X_train.columns.astype(str).tolist()
    scores: Optional[NDArray[np.float64]] = None
    projection = resolved in {
        FeatureDownsampleMethod.SVD,
        FeatureDownsampleMethod.SparseRandomProjection,
    }

    if projection:
        components = projection_n_components(n_features_out, len(X_fit), n_features)
        if resolved is FeatureDownsampleMethod.SVD:
            transformer = TruncatedSVD(n_components=components, random_state=seed)
            prefix = "svd"
        else:
            transformer = SparseRandomProjection(
                n_components=components, random_state=seed, dense_output=True
            )
            prefix = "sparse_rp"
        transformer.fit(X_fit)
        fit_seconds = perf_counter() - fit_started
        transform_started = perf_counter()
        columns = [f"{prefix}_{idx + 1}" for idx in range(components)]
        train_values = transformer.transform(X_train)
        test_values = transformer.transform(X_test)
        if sparse.issparse(train_values):
            train_values = train_values.toarray()
            test_values = test_values.toarray()
        train_out = pd.DataFrame(train_values, index=X_train.index, columns=columns)
        test_out = pd.DataFrame(test_values, index=X_test.index, columns=columns)
        selected_indices: list[int] = []
        selected_names = columns
    else:
        if resolved in INDEXED_SAFE_METHODS:
            selected, partial = select_indexed_columns(
                X_train,
                np.arange(len(X_train), dtype=int),
                y_train,
                is_classification,
                requested,
                n_features_out,
                chunk_size,
                screening_positions,
                names,
                variance_threshold,
                save_all_scores,
                seed=seed,
            )
            partial.tuning_rows = []
            transform_started = perf_counter()
            train_out = X_train.iloc[:, selected].copy()
            test_out = X_test.iloc[:, selected].copy()
            partial.transform_seconds = perf_counter() - transform_started
            return train_out, test_out, partial
        scores = _aggregate_direct_scores(X_fit, y_fit, is_classification, resolved, seed)
        selected = _rank_indices(
            scores,
            n_select,
            seed=seed,
            tie_keys=names,
        )
        if len(selected) == 0:
            raise ValueError("No features have usable downsampling scores.")
        fit_seconds = perf_counter() - fit_started
        transform_started = perf_counter()
        train_out = X_train.iloc[:, selected].copy()
        test_out = X_test.iloc[:, selected].copy()
        selected_indices = selected.astype(int).tolist()
        selected_names = [names[idx] for idx in selected]

    transform_seconds = perf_counter() - transform_started
    score_values, score_indices, score_names = _score_details(
        scores, names, save_all_scores, seed
    )
    result = FeatureDownsampleResult(
        requested_method=requested.value,
        resolved_method=resolved.value,
        n_features_in=n_features,
        n_features_out=train_out.shape[1],
        selected_features=selected_names,
        selected_indices=selected_indices,
        scores=score_values,
        score_feature_indices=score_indices,
        score_feature_names=score_names,
        full_score_values=scores if save_all_scores else None,
        full_score_feature_names=names
        if save_all_scores and scores is not None
        else None,
        projection=projection,
        fit_seconds=fit_seconds,
        transform_seconds=transform_seconds,
        auto_reason=_auto_reason(requested, resolved, len(X_train), n_features, n_select),
        **_result_strategy(resolved, y_train, n_features),
    )
    return train_out, test_out, result


def downsample_split(
    train: PreparedData,
    test: PreparedData,
    options: Any,
) -> tuple[PreparedData, PreparedData, FeatureDownsampleResult]:
    requested = _method(getattr(options, "feat_downsample", None))
    n_features_out = getattr(options, "n_feat_downsample", N_FEAT_DOWNSAMPLE_DEFAULT)
    seed = int(getattr(options, "seed", SEED))
    resolved = resolve_feature_downsample_method(
        requested,
        len(train.X),
        train.X.shape[1],
        resolve_n_features(n_features_out, train.X.shape[1]),
        train.is_classification,
        train.y,
    )
    effective, screening_rows, tuning_rows, fallback_note = resolve_screening_split(
        requested,
        resolved,
        train.y,
        getattr(options, "downsample_screening_fraction", 0.25),
        train.is_classification,
        train.groups,
        seed,
    )

    selection_request = requested if fallback_note is None else effective
    try:
        X_train, X_test, result = _downsample_frame(
            train.X,
            test.X,
            train.y,
            train.is_classification,
            selection_request,
            n_features_out,
            getattr(options, "downsample_chunk_size", DOWNSAMPLE_CHUNK_SIZE_DEFAULT),
            screening_rows,
            getattr(options, "downsample_variance_threshold", None),
            getattr(options, "downsample_save_scores", False),
            seed,
        )
    except ValueError as error:
        if (
            requested is not FeatureDownsampleMethod.Auto
            or screening_rows is None
            or "usable downsampling scores" not in str(error)
        ):
            raise
        fallback_note = (
            f"Auto feature downsampling fell back from {effective.value} to "
            "normalized-variance because supervised scoring produced no usable "
            f"feature scores: {error}"
        )
        warn(fallback_note, stacklevel=2)
        screening_rows = None
        tuning_rows = np.arange(len(train.y), dtype=int)
        X_train, X_test, result = _downsample_frame(
            train.X,
            test.X,
            train.y,
            train.is_classification,
            FeatureDownsampleMethod.NormalizedVariance,
            n_features_out,
            getattr(options, "downsample_chunk_size", DOWNSAMPLE_CHUNK_SIZE_DEFAULT),
            None,
            getattr(options, "downsample_variance_threshold", None),
            getattr(options, "downsample_save_scores", False),
            seed,
        )
    result.requested_method = requested.value
    if requested is FeatureDownsampleMethod.Auto and result.auto_reason is None:
        result.auto_reason = _auto_reason(
            requested,
            _method(result.resolved_method),
            len(train.X),
            train.X.shape[1],
            resolve_n_features(n_features_out, train.X.shape[1]),
        )
    if fallback_note is not None:
        result.notes.append(fallback_note)
    result.screening_rows = [] if screening_rows is None else screening_rows.tolist()
    result.tuning_rows = tuning_rows.tolist()
    result.screening_samples = 0 if screening_rows is None else len(screening_rows)
    result.tuning_samples = len(tuning_rows)
    if result.resolved_method == FeatureDownsampleMethod.None_.value:
        return train, test, result
    return (
        train.with_features(X_train, result.resolved_method),
        test.with_features(X_test, result.resolved_method),
        result,
    )

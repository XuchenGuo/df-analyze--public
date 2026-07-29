import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_selection import f_classif

import df_analyze.downsampling.large as large_module
import df_analyze.downsampling.methods as methods_module
from df_analyze.cli.cli import make_parser
from df_analyze.downsampling.base import resolve_n_features
from df_analyze.downsampling.chunked import (
    chunked_f_test_scores,
    chunked_range_normalized_variance_scores,
    chunked_variance_scores,
)
from df_analyze.downsampling.containers import FeatureDownsampleResult
from df_analyze.downsampling.large import large_table_prepared_splits
from df_analyze.downsampling.methods import (
    _level_preserving_sample,
    _rank_indices,
    downsample_split,
    resolve_feature_downsample_method,
    select_indexed_columns,
)
from df_analyze.downsampling.screening import screening_tuning_indices
from df_analyze.enumerables import FeatureDownsampleMethod, ValidationMethod
from df_analyze.preprocessing.prepare import PreparedData


def _frames(seed: int = 0) -> tuple[PreparedData, PreparedData]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(240, 60)), columns=[f"feature_{idx}" for idx in range(60)]
    )
    y = pd.Series(
        (2 * X["feature_3"] - X["feature_11"] + rng.normal(size=240) > 0).astype(
            int
        ),
        name="target",
    )
    train_X = X.iloc[:180].reset_index(drop=True)
    test_X = X.iloc[180:].reset_index(drop=True)
    train = PreparedData(
        train_X,
        y.iloc[:180].reset_index(drop=True),
        None,
        True,
        X_cont=train_X,
    )
    test = PreparedData(
        test_X,
        y.iloc[180:].reset_index(drop=True),
        None,
        True,
        X_cont=test_X,
    )
    return train, test


def _options(method: FeatureDownsampleMethod, n_features: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        feat_downsample=method,
        n_feat_downsample=n_features,
        downsample_chunk_size=13,
        downsample_screening_fraction=0.25,
        downsample_variance_threshold=None,
        downsample_save_scores=False,
    )


@pytest.mark.parametrize(
    ("requested", "total", "expected"),
    [(10, 100, 10), (1000, 100, 100), (0.1, 101, 11), (1.0, 30, 30)],
)
def test_resolve_n_features(requested, total, expected):
    assert resolve_n_features(requested, total) == expected


def test_cli_parses_downsample_count_as_an_integer():
    args = make_parser().parse_args(
        [
            "--df",
            "input.csv",
            "--target",
            "target",
            "--mode",
            "classify",
            "--feat-downsample",
            "f-test",
            "--n-feat-downsample",
            "10",
        ]
    )
    assert args.n_feat_downsample == 10


def test_chunked_scores_match_full_scores():
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(100, 31)))
    y = pd.Series(rng.integers(0, 2, size=100))
    assert np.allclose(
        chunked_variance_scores(X, 7), X.var(axis=0).to_numpy(), equal_nan=True
    )
    expected = f_classif(X, y)[0]
    assert np.allclose(
        chunked_f_test_scores(X, y, True, 7), expected, equal_nan=True
    )


def test_range_normalized_variance_is_scale_and_translation_invariant():
    rng = np.random.default_rng(120)
    X = pd.DataFrame(rng.normal(size=(100, 7)))
    transformed = X.copy()
    transformed[0] = transformed[0] * 1.0e9 + 3.0e12
    transformed[1] = transformed[1] * -0.001 - 40.0

    original = chunked_range_normalized_variance_scores(X, 3)
    changed = chunked_range_normalized_variance_scores(transformed, 3)

    assert np.allclose(original, changed, rtol=1e-10, atol=1e-12)


def test_range_normalized_variance_matches_sparse_input():
    X = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 4.0, 6.0],
            [2.0, 0.0, 0.0],
        ]
    )

    dense = chunked_range_normalized_variance_scores(X, 2)
    sparse_scores = chunked_range_normalized_variance_scores(
        sparse.csr_matrix(X), 2
    )

    assert np.allclose(dense, sparse_scores)


@pytest.mark.parametrize(
    "method",
    [
        FeatureDownsampleMethod.Auto,
        FeatureDownsampleMethod.Random,
        FeatureDownsampleMethod.Variance,
        FeatureDownsampleMethod.FTest,
        FeatureDownsampleMethod.MutualInfo,
        FeatureDownsampleMethod.Linear,
        FeatureDownsampleMethod.LGBM,
        FeatureDownsampleMethod.SVD,
        FeatureDownsampleMethod.SparseRandomProjection,
        FeatureDownsampleMethod.RankEnsemble,
        FeatureDownsampleMethod.SelectorEnsemble,
        FeatureDownsampleMethod.StableRank,
    ],
)
def test_all_downsample_methods(method):
    train, test = _frames()
    train_out, test_out, result = downsample_split(train, test, _options(method))
    assert train_out.X.shape == (180, 12)
    assert test_out.X.shape == (60, 12)
    assert result.n_features_in == 60
    assert result.n_features_out == 12
    assert train_out.X.columns.tolist() == test_out.X.columns.tolist()
    if method in {
        FeatureDownsampleMethod.RankEnsemble,
        FeatureDownsampleMethod.SelectorEnsemble,
    }:
        assert result.ensemble_members == [
            "range-normalized-variance",
            "f-test",
            "mutual-info",
            "linear",
            "lgbm",
        ]


def test_ensemble_records_only_members_that_produced_scores(monkeypatch):
    train, _ = _frames()

    def controlled_scores(X, y, is_classification, method):
        del y, is_classification
        if method is FeatureDownsampleMethod.Linear:
            raise ValueError("deliberate member failure")
        return np.arange(X.shape[1], dtype=np.float64)

    monkeypatch.setattr(methods_module, "_aggregate_direct_scores", controlled_scores)

    with pytest.warns(UserWarning, match="Skipping 'linear'"):
        _, result = select_indexed_columns(
            train.X,
            np.arange(len(train.X)),
            train.y,
            True,
            FeatureDownsampleMethod.RankEnsemble,
            12,
        )

    assert result.ensemble_members == [
        "range-normalized-variance",
        "f-test",
        "mutual-info",
        "lgbm",
    ]


def test_supervised_screening_is_disjoint_from_tuning():
    train, test = _frames()
    _, _, result = downsample_split(
        train, test, _options(FeatureDownsampleMethod.FTest)
    )
    assert set(result.screening_rows).isdisjoint(result.tuning_rows)
    assert sorted(result.screening_rows + result.tuning_rows) == list(range(180))


def test_rank_indices_keeps_positive_infinity():
    scores = np.array([np.nan, np.inf, 3.0, -np.inf, 2.0])

    selected = _rank_indices(scores, 3)

    assert selected.tolist() == [1, 2, 4]


def test_indexed_selection_rejects_all_invalid_scores():
    X = pd.DataFrame(np.ones((20, 12)))
    y = pd.Series([0, 1] * 10)

    with pytest.raises(ValueError, match="usable downsampling scores"):
        select_indexed_columns(
            X,
            np.arange(len(X)),
            y,
            True,
            FeatureDownsampleMethod.FTest,
            5,
        )


def test_auto_downsampling_falls_back_for_singleton_multitarget():
    rng = np.random.default_rng(44)
    X = pd.DataFrame(rng.normal(size=(40, 20)))
    y = pd.DataFrame(
        {
            "common": np.tile([0, 1], 20),
            "singleton": [1, *([0] * 39)],
        }
    )
    train = PreparedData(X, y, None, True, X_cont=X)
    test = PreparedData(X.iloc[:10].copy(), y.iloc[:10].copy(), None, True, X_cont=X.iloc[:10])

    with pytest.warns(UserWarning, match="fell back.*variance"):
        _, _, result = downsample_split(
            train, test, _options(FeatureDownsampleMethod.Auto, 5)
        )

    assert result.requested_method == FeatureDownsampleMethod.Auto.value
    assert result.resolved_method == FeatureDownsampleMethod.Variance.value
    assert result.screening_rows == []
    assert result.tuning_rows == list(range(len(train.X)))
    assert result.auto_reason is not None


def test_auto_uses_scalable_rank_ensemble_for_extreme_p_over_n():
    y = pd.Series([0, 1] * 5)

    resolved = resolve_feature_downsample_method(
        FeatureDownsampleMethod.Auto,
        n_samples=10,
        n_features=10_001,
        n_features_out=100,
        is_classification=True,
        y=y,
    )

    assert resolved is FeatureDownsampleMethod.RankEnsemble


def test_explicit_supervised_downsampling_rejects_singleton_multitarget():
    rng = np.random.default_rng(45)
    X = pd.DataFrame(rng.normal(size=(40, 20)))
    y = pd.DataFrame(
        {
            "common": np.tile([0, 1], 20),
            "singleton": [1, *([0] * 39)],
        }
    )
    train = PreparedData(X, y, None, True, X_cont=X)
    test = PreparedData(X.iloc[:10].copy(), y.iloc[:10].copy(), None, True, X_cont=X.iloc[:10])

    with pytest.raises(ValueError, match="screening and tuning subsets"):
        downsample_split(train, test, _options(FeatureDownsampleMethod.FTest, 5))


def test_small_classification_screening_is_feasible():
    y = pd.Series([0, 0, 1, 1, 2, 2])
    screening, tuning = screening_tuning_indices(y, 0.25, True)
    assert set(y.iloc[screening]) == {0, 1, 2}
    assert set(y.iloc[tuning]) == {0, 1, 2}


def test_grouped_screening_splits_whole_groups():
    y = pd.Series([0, 1] * 20)
    groups = pd.Series(np.repeat(np.arange(10), 4))
    screening, tuning = screening_tuning_indices(y, 0.25, True, groups)
    assert set(groups.iloc[screening]).isdisjoint(groups.iloc[tuning])


def test_grouped_screening_rejects_a_single_group():
    y = pd.Series([0, 1] * 10)
    groups = pd.Series(["only-group"] * len(y))

    with pytest.raises(ValueError, match="at least two distinct training groups"):
        screening_tuning_indices(y, 0.25, True, groups)


def test_auto_downsampling_falls_back_when_group_screening_is_impossible():
    train, test = _frames()
    train.groups = pd.Series(["only-group"] * len(train.X))

    _, _, result = downsample_split(
        train,
        test,
        _options(FeatureDownsampleMethod.Auto),
    )

    assert result.resolved_method == FeatureDownsampleMethod.Variance.value
    assert result.screening_rows == []
    assert result.tuning_rows == list(range(len(train.X)))
    assert any("at least two distinct training groups" in note for note in result.notes)


def test_explicit_supervised_downsampling_rejects_a_single_group():
    train, test = _frames()
    train.groups = pd.Series(["only-group"] * len(train.X))

    with pytest.raises(ValueError, match="at least two distinct training groups"):
        downsample_split(
            train,
            test,
            _options(FeatureDownsampleMethod.FTest),
        )


def test_indexed_selection_uses_compact_training_target():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(80, 25)))
    train_rows = np.arange(5, 75, 2)
    y = pd.Series((X.iloc[train_rows, 7] > 0).astype(int).to_numpy())
    selected, result = select_indexed_columns(
        X,
        train_rows,
        y,
        True,
        FeatureDownsampleMethod.FTest,
        5,
        screening_positions=np.arange(20),
    )
    assert len(selected) == 5
    assert result.n_features_out == 5


def test_test_only_signal_is_not_used_for_scoring():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(100, 20)))
    y = pd.Series(rng.integers(0, 2, size=80))
    X.iloc[:80, 19] = 0.0
    X.iloc[80:, 19] = np.arange(20)
    selected, _ = select_indexed_columns(
        X,
        np.arange(80),
        y,
        True,
        FeatureDownsampleMethod.FTest,
        5,
    )
    assert 19 not in selected


def test_stable_rank_handles_constant_features():
    train, test = _frames()
    train.X.iloc[:, 20:] = 1.0
    test.X.iloc[:, 20:] = 1.0
    out, _, result = downsample_split(
        train, test, _options(FeatureDownsampleMethod.StableRank)
    )
    assert out.X.shape[1] == 12
    assert result.strategy == "selection_frequency_then_mean_rank"
    assert result.stability_repeats == 20
    assert result.stability_subsample == pytest.approx(0.75)


def test_variance_threshold_is_enforced():
    train, test = _frames()
    options = _options(FeatureDownsampleMethod.Variance)
    options.downsample_variance_threshold = 1e6
    with pytest.raises(ValueError, match="score threshold"):
        downsample_split(train, test, options)


def test_large_table_path_selects_before_materializing():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(240, 200))
    frame = pd.DataFrame(values, columns=[f"x{idx}" for idx in range(200)])
    frame["target"] = (values[:, 17] > 0).astype(int)
    options = _options(FeatureDownsampleMethod.FTest, 15)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False
    train, test, result = large_table_prepared_splits(frame, options)[0]
    assert train.X.shape == (180, 15)
    assert test.X.shape == (60, 15)
    assert result.input_format == "table-large"


def test_large_table_regression_does_not_stratify_continuous_target():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(160, 50))
    frame = pd.DataFrame(values, columns=[f"x{idx}" for idx in range(50)])
    frame["target"] = values[:, 4] + rng.normal(size=160)
    options = _options(FeatureDownsampleMethod.FTest, 10)
    options.is_classification = False
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False
    train, test, _ = large_table_prepared_splits(frame, options)[0]
    assert train.X.shape == (120, 10)
    assert test.X.shape == (40, 10)


def test_large_table_keeps_numeric_binary_feature_name():
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        {
            "binary": np.tile([0, 1], 120),
            "x1": rng.normal(size=240),
            "x2": rng.normal(size=240),
            "x3": rng.normal(size=240),
            "x4": rng.normal(size=240),
        }
    )
    frame["target"] = np.tile([0, 1], 120)
    options = _options(FeatureDownsampleMethod.Variance, 5)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False

    train, test, result = large_table_prepared_splits(frame, options)[0]

    assert "binary" in train.X.columns
    assert "binary" in test.X.columns
    assert "binary" in result.selected_features
    assert not any(name.startswith("binary_") for name in train.X.columns)


@pytest.mark.parametrize("is_classification", [True, False])
def test_large_table_rejects_constant_targets(is_classification: bool):
    rng = np.random.default_rng(18)
    frame = pd.DataFrame(rng.normal(size=(80, 20)))
    frame["target"] = 1 if is_classification else 2.5
    options = _options(FeatureDownsampleMethod.Variance, 5)
    options.is_classification = is_classification
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False

    with pytest.raises(ValueError, match="constant"):
        large_table_prepared_splits(frame, options)


def test_large_table_multitarget_split_and_target_audits():
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(rng.normal(size=(120, 20)))
    frame["target_a"] = [1, *([0] * 119)]
    frame["target_b"] = [*([0] * 8), 1, *([0] * 111)]
    options = _options(FeatureDownsampleMethod.Variance, 5)
    options.is_classification = True
    options.target = "target_a"
    options.targets = ["target_a", "target_b"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False

    with pytest.warns(UserWarning, match="stratification"):
        train, test, _ = large_table_prepared_splits(frame, options)[0]

    assert train.X.shape == (90, 5)
    assert test.X.shape == (30, 5)
    assert train.info is not None
    assert train.info.multitarget_audit is not None
    assert train.info.multitarget_audit.target_names == ["target_a", "target_b"]
    assert train.info.split_audit is not None
    assert train.info.split_audit.used_targets == []
    assert set(train.info.split_audit.dropped_targets) == {"target_a", "target_b"}


def test_large_table_lodo_uses_every_partition_as_training_set():
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        rng.normal(size=(210, 12)), columns=[f"x{idx}" for idx in range(12)]
    )
    frame["target"] = np.arange(210) % 2
    options = _options(FeatureDownsampleMethod.Variance, 5)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False
    options.tests_method = ValidationMethod.LODO
    partitions = [np.arange(0, 60), np.arange(60, 130), np.arange(130, 210)]

    outputs = large_table_prepared_splits(
        frame, options, partitions[0], partitions[1:]
    )

    assert len(outputs) == 3
    assert [(len(train.X), len(test.X)) for train, test, _ in outputs] == [
        (60, 150),
        (70, 140),
        (80, 130),
    ]


def test_large_table_fits_target_encoder_on_training_partition():
    rng = np.random.default_rng(32)
    frame = pd.DataFrame(
        rng.normal(size=(90, 10)), columns=[f"x{idx}" for idx in range(10)]
    )
    frame["target"] = [0, 1] * 30 + [2] * 30
    options = _options(FeatureDownsampleMethod.Variance, 4)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.seed = 42
    options.assume_numeric_features = False
    options.tests_method = ValidationMethod.List

    with pytest.raises(ValueError, match="holdout set"):
        large_table_prepared_splits(
            frame, options, np.arange(60), [np.arange(60, 90)]
        )


def test_tiny_supervised_screening_does_not_reuse_rows():
    with pytest.raises(ValueError, match="at least four"):
        screening_tuning_indices(pd.Series([0, 1, 0]), 0.25, True)


def test_large_table_validation_respects_chunk_memory_limit(monkeypatch):
    frame = pd.DataFrame(np.ones((10, 9)))
    widths = []
    to_numpy = pd.DataFrame.to_numpy

    def record_width(self, *args, **kwargs):
        widths.append(self.shape[1])
        return to_numpy(self, *args, **kwargs)

    monkeypatch.setattr(large_module, "DOWNSAMPLE_MAX_CHUNK_BYTES", 480)
    monkeypatch.setattr(pd.DataFrame, "to_numpy", record_width)
    large_module._validate_predictors(frame, chunk_size=100)

    assert max(widths) <= 2


def test_large_table_rejects_nonfinite_predictors():
    frame = pd.DataFrame({"finite": [1.0, 2.0], "missing": [3.0, np.nan]})

    with pytest.raises(ValueError, match="NaN or infinite"):
        large_module._validate_predictors(frame, chunk_size=2)


def test_projection_result_has_a_selected_features_table():
    result = FeatureDownsampleResult(
        requested_method="svd",
        resolved_method="svd",
        n_features_in=100,
        n_features_out=3,
        selected_features=["svd_1", "svd_2", "svd_3"],
        projection=True,
    )
    frame = result.selected_frame()
    assert frame.shape == (3, 3)
    assert frame["feature_index"].isna().all()


def test_downsampling_json_replaces_nonfinite_scores_with_null():
    result = FeatureDownsampleResult(
        requested_method="f-test",
        resolved_method="f-test",
        n_features_in=3,
        n_features_out=2,
        selected_features=["a", "b"],
        scores=[np.inf, 2.0, np.nan],
    )

    payload = json.loads(result.to_json())

    assert payload["scores"] == [None, 2.0, None]
    assert "Infinity" not in result.to_json()
    assert "NaN" not in result.to_json()


def test_downsampling_json_keeps_large_scores_in_csv_only():
    result = FeatureDownsampleResult(
        requested_method="f-test",
        resolved_method="f-test",
        n_features_in=501,
        n_features_out=2,
        selected_features=["a", "b"],
        scores=[1.0] * 501,
        score_feature_indices=list(range(501)),
        score_feature_names=[f"feature_{idx}" for idx in range(501)],
    )

    payload = json.loads(result.to_json())

    assert payload["scores_saved_separately"]
    assert "scores" not in payload
    assert "score_feature_indices" not in payload
    assert "score_feature_names" not in payload
    assert len(result.scores_frame()) == 501


def test_stable_rank_sample_keeps_every_target_level():
    y = pd.DataFrame(
        {
            "target_a": [1, *([0] * 19)],
            "target_b": [0, 1, *([0] * 18)],
        }
    )

    sampled = _level_preserving_sample(y, 16, np.random.default_rng(42))

    assert len(sampled) == 16
    assert set(y.iloc[sampled]["target_a"]) == {0, 1}
    assert set(y.iloc[sampled]["target_b"]) == {0, 1}


def test_direct_method_is_rejected_for_indexed_large_input():
    X = np.zeros((10, 50_001), dtype=np.float32)
    y = pd.Series([0, 1] * 5)
    with pytest.raises(ValueError, match="materialized training matrix"):
        select_indexed_columns(
            X,
            np.arange(10),
            y,
            True,
            FeatureDownsampleMethod.MutualInfo,
            10,
        )

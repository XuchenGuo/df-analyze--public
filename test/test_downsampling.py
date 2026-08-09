import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_selection import f_classif

import df_analyze.downsampling.large as large_module
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
    _rank_indices,
    _rank_score,
    downsample_split,
    select_indexed_columns,
)
from df_analyze.downsampling.screening import screening_tuning_indices
from df_analyze.enumerables import FeatureDownsampleMethod, ValidationMethod
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.saving import ProgramDirs


def _frames(seed: int = 0) -> tuple[PreparedData, PreparedData]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(240, 60)), columns=[f"feature_{idx}" for idx in range(60)]
    )
    y = pd.Series(
        (2 * X["feature_3"] - X["feature_11"] + rng.normal(size=240) > 0).astype(int),
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
        downsample_save_scores=False,
        seed=42,
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
    assert np.allclose(chunked_f_test_scores(X, y, True, 7), expected, equal_nan=True)


def test_dense_screening_stays_column_chunk_bounded():
    class RecordingIndexer:
        def __init__(self, owner):
            self.owner = owner

        def __getitem__(self, key):
            self.owner.keys.append(key)
            return self.owner.frame.iloc[key]

    class RecordingFrame:
        def __init__(self, frame):
            self.frame = frame
            self.shape = frame.shape
            self.keys = []
            self.iloc = RecordingIndexer(self)

    rows = np.asarray([0, 2, 4, 6, 8], dtype=int)
    source = RecordingFrame(pd.DataFrame(np.arange(120).reshape(12, 10)))

    scores = chunked_variance_scores(source, 3, rows)

    assert len(scores) == 10
    assert len(source.keys) == 4
    assert all(isinstance(key, tuple) for key in source.keys)
    assert all(np.array_equal(key[0], rows) for key in source.keys)
    assert all(key[1].stop - key[1].start <= 3 for key in source.keys)


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
    sparse_scores = chunked_range_normalized_variance_scores(sparse.csr_matrix(X), 2)

    assert np.allclose(dense, sparse_scores)


@pytest.mark.parametrize(
    "method",
    [
        FeatureDownsampleMethod.NormalizedVariance,
        FeatureDownsampleMethod.FTest,
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


def test_supervised_screening_is_disjoint_from_tuning():
    train, test = _frames()
    _, _, result = downsample_split(train, test, _options(FeatureDownsampleMethod.FTest))
    assert set(result.screening_rows).isdisjoint(result.tuning_rows)
    assert sorted(result.screening_rows + result.tuning_rows) == list(range(180))


def test_rank_indices_keeps_positive_infinity():
    scores = np.array([np.nan, np.inf, 3.0, -np.inf, 2.0])

    selected = _rank_indices(scores, 3)

    assert selected.tolist() == [1, 2, 4]


def test_rank_scores_are_tie_aware_and_keep_invalid_scores_missing():
    tied = _rank_score(np.zeros(6))
    partial = _rank_score(np.array([10.0, np.nan, 9.0]))

    assert np.allclose(tied, tied[0])
    assert np.isnan(partial[1])
    assert partial[0] > partial[2]


def test_tied_selection_is_invariant_to_column_permutation():
    values = np.tile(np.arange(40, dtype=np.float64)[:, None], (1, 8))
    columns = [f"feature_{idx}" for idx in range(values.shape[1])]
    X = pd.DataFrame(values, columns=columns)
    y = pd.Series(np.arange(len(X)) % 2)
    rows = np.arange(len(X), dtype=int)
    selected, _ = select_indexed_columns(
        X,
        rows,
        y,
        True,
        FeatureDownsampleMethod.FTest,
        3,
        seed=77,
    )
    permutation = [6, 2, 7, 0, 5, 1, 4, 3]
    permuted = X.iloc[:, permutation]
    permuted_selected, _ = select_indexed_columns(
        permuted,
        rows,
        y,
        True,
        FeatureDownsampleMethod.FTest,
        3,
        seed=77,
    )

    first_names = {X.columns[idx] for idx in selected}
    permuted_names = {permuted.columns[idx] for idx in permuted_selected}
    assert first_names == permuted_names


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
    test = PreparedData(
        X.iloc[:10].copy(),
        y.iloc[:10].copy(),
        None,
        True,
        X_cont=X.iloc[:10],
    )

    with pytest.raises(ValueError, match="screening and tuning subsets"):
        downsample_split(train, test, _options(FeatureDownsampleMethod.FTest, 5))


def test_regression_screening_preserves_each_target_variation():
    n = 40
    y = pd.DataFrame(
        {
            "dense": np.arange(n, dtype=float),
            "sparse": np.r_[np.ones(10), np.zeros(n - 10)],
        }
    )

    screening, tuning = screening_tuning_indices(y, 0.25, False)

    for target in y.columns:
        assert y.iloc[screening][target].nunique() >= 2
        assert y.iloc[tuning][target].nunique() >= 2


def test_regression_screening_rejects_impossible_target_variation():
    n = 40
    y = pd.DataFrame(
        {
            "dense": np.arange(n, dtype=float),
            "sparse": np.r_[1.0, np.zeros(n - 1)],
        }
    )

    with pytest.raises(ValueError, match="screening and tuning subsets"):
        screening_tuning_indices(y, 0.25, False)


def test_single_regression_screening_retains_original_behavior():
    y = pd.Series(np.r_[1.0, np.zeros(39)], name="target")

    screening, tuning = screening_tuning_indices(y, 0.25, False)

    assert len(screening) == 10
    assert len(tuning) == 30
    assert set(screening).isdisjoint(tuning)


def test_small_classification_screening_is_feasible():
    y = pd.Series(np.repeat([0, 1, 2], 4))
    screening, tuning = screening_tuning_indices(y, 0.25, True)
    assert set(y.iloc[screening]) == {0, 1, 2}
    assert set(y.iloc[tuning]) == {0, 1, 2}
    assert y.iloc[screening].value_counts().min() >= 2
    assert y.iloc[tuning].value_counts().min() >= 2


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


def test_large_table_path_selects_before_materializing(monkeypatch):
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
    prepared_widths = []
    original_prepare = large_module.prepare_data

    def record_prepare(selected_frame, *args, **kwargs):
        prepared_widths.append(selected_frame.shape[1])
        return original_prepare(selected_frame, *args, **kwargs)

    monkeypatch.setattr(large_module, "prepare_data", record_prepare)
    train, test, result = large_table_prepared_splits(frame, options)[0]
    assert train.X.shape == (180, 15)
    assert test.X.shape == (60, 15)
    assert result.input_format == "table-large"
    assert result.large_feature_mode
    assert result.resolved_method == FeatureDownsampleMethod.FTest.value
    assert prepared_widths == [16]


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
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 5)
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
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 5)
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
    frame["target_a"] = [*([1] * 8), *([0] * 112)]
    frame["target_b"] = [*([0] * 8), *([1] * 8), *([0] * 104)]
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 5)
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
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 5)
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

    outputs = large_table_prepared_splits(frame, options, partitions[0], partitions[1:])

    assert len(outputs) == 3
    assert [(len(train.X), len(test.X)) for train, test, _ in outputs] == [
        (60, 150),
        (70, 140),
        (80, 130),
    ]


def test_large_table_list_materializes_only_the_current_split(monkeypatch):
    rng = np.random.default_rng(310)
    frame = pd.DataFrame(
        rng.normal(size=(150, 12)), columns=[f"x{idx}" for idx in range(12)]
    )
    frame["target"] = np.arange(150) % 2
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 5)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.assume_numeric_features = False
    options.tests_method = ValidationMethod.List
    prepared_sizes = []
    original_prepare = large_module.prepare_data

    def record_prepare(frame, *args, **kwargs):
        prepared_sizes.append(len(frame))
        return original_prepare(frame, *args, **kwargs)

    monkeypatch.setattr(large_module, "prepare_data", record_prepare)
    outputs = large_table_prepared_splits(
        frame,
        options,
        np.arange(0, 60),
        [np.arange(60, 100), np.arange(100, 150)],
    )

    assert len(outputs) == 2
    assert prepared_sizes == [100, 110]


def test_large_table_fits_target_encoder_on_training_partition():
    rng = np.random.default_rng(32)
    frame = pd.DataFrame(
        rng.normal(size=(90, 10)), columns=[f"x{idx}" for idx in range(10)]
    )
    frame["target"] = [0, 1] * 30 + [2] * 30
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 4)
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
        large_table_prepared_splits(frame, options, np.arange(60), [np.arange(60, 90)])


def test_tiny_supervised_screening_does_not_reuse_rows():
    with pytest.raises(ValueError, match="at least six"):
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


def test_large_table_normalizes_only_group_metadata(monkeypatch):
    rng = np.random.default_rng(808)
    frame = pd.DataFrame(
        rng.normal(size=(160, 40)), columns=[f"x{idx}" for idx in range(40)]
    )
    frame["site"] = np.repeat([f"site_{idx}" for idx in range(40)], 4)
    frame["target"] = np.arange(len(frame)) % 2
    options = _options(FeatureDownsampleMethod.NormalizedVariance, 8)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = "site"
    options.test_val_size = 0.25
    options.assume_numeric_features = False
    observed_shapes = []
    original = large_module.unify_nans

    def record_unify(values):
        observed_shapes.append(values.shape)
        return original(values)

    monkeypatch.setattr(large_module, "unify_nans", record_unify)

    large_table_prepared_splits(frame, options)

    assert observed_shapes
    assert all(width == 1 for _, width in observed_shapes)


def test_large_table_retains_protected_source_features():
    rng = np.random.default_rng(809)
    frame = pd.DataFrame(
        rng.normal(size=(160, 20)), columns=[f"x{idx}" for idx in range(20)]
    )
    frame["protected_prior"] = np.tile([0.0, 1.0], len(frame) // 2)
    frame["target"] = np.arange(len(frame)) % 2
    options = _options(FeatureDownsampleMethod.FTest, 5)
    options.is_classification = True
    options.target = "target"
    options.targets = ["target"]
    options.categoricals = []
    options.ordinals = []
    options.drops = []
    options.grouper = None
    options.test_val_size = 0.25
    options.assume_numeric_features = False
    options.downsample_protected_features = ["protected_prior"]

    train, test, result = large_table_prepared_splits(frame, options)[0]

    assert "protected_prior" in train.X
    assert "protected_prior" in test.X
    assert result.protected_features == ["protected_prior"]
    assert len(result.selected_features) == 5


def test_protected_feature_does_not_change_nonprotected_ranking():
    rng = np.random.default_rng(911)
    y = pd.Series(np.tile([0, 1], 60))
    X = pd.DataFrame(
        {
            "protected": y.to_numpy(dtype=float),
            "signal_a": y.to_numpy(dtype=float) + rng.normal(0, 0.05, len(y)),
            "signal_b": y.to_numpy(dtype=float) + rng.normal(0, 0.15, len(y)),
            "noise_a": rng.normal(size=len(y)),
            "noise_b": rng.normal(size=len(y)),
        }
    )
    rows = np.arange(len(X))

    selected, _ = select_indexed_columns(
        X,
        rows,
        y,
        True,
        FeatureDownsampleMethod.FTest,
        n_features_out=3,
        protected_indices=[0],
    )
    without_protected, _ = select_indexed_columns(
        X.drop(columns="protected"),
        rows,
        y,
        True,
        FeatureDownsampleMethod.FTest,
        n_features_out=2,
    )

    selected_names = [X.columns[index] for index in selected if index != 0]
    expected_names = [X.columns[1:][index] for index in without_protected]
    assert selected_names == expected_names


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


def test_sparse_index_metadata_without_scores_is_serializable():
    result = FeatureDownsampleResult(
        requested_method="normalized-variance",
        resolved_method="normalized-variance",
        n_features_in=3,
        n_features_out=1,
        selected_features=["feature_2"],
        selected_indices=[1],
        sparse_input=True,
        input_format="svmlight",
        source_index_base=1,
    )

    payload = json.loads(result.to_json())

    assert payload["source_index_base"] == 1
    assert payload["selected_source_indices"] == [2]
    assert "score_source_feature_indices" not in payload
    assert result.selected_frame()["source_feature_index"].tolist() == [2]


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


def test_full_score_export_keeps_invalid_features_in_bounded_chunks(tmp_path):
    values = np.asarray([2.0, np.nan, 0.0, -np.inf], dtype=np.float64)
    result = FeatureDownsampleResult(
        requested_method="f-test",
        resolved_method="f-test",
        n_features_in=4,
        n_features_out=1,
        selected_features=["a"],
        scores=[2.0, 0.0],
        score_feature_indices=[0, 2],
        score_feature_names=["a", "c"],
        full_score_values=values,
        full_score_feature_names=["a", "b", "c", "d"],
    )

    exported = pd.concat(list(result.iter_score_frames(chunk_size=2)))
    payload = json.loads(result.to_json())
    ProgramDirs(downsampling=tmp_path).save_downsampling(result)
    saved = pd.read_csv(tmp_path / "feature_scores.csv")

    assert exported["feature_index"].tolist() == [0, 1, 2, 3]
    assert exported["score_valid"].tolist() == [True, False, True, False]
    assert exported["invalid_reason"].tolist() == [
        "",
        "nan",
        "",
        "negative_infinity",
    ]
    assert payload["n_scores_saved"] == 4
    assert payload["invalid_score_count"] == 2
    assert saved["feature_index"].tolist() == [0, 1, 2, 3]
    assert saved["score_valid"].tolist() == [True, False, True, False]


def test_select_indexed_columns_attaches_complete_raw_score_vector():
    rng = np.random.default_rng(810)
    X = pd.DataFrame(
        {
            "signal": np.tile([0.0, 1.0], 40),
            "constant": np.ones(80),
            "noise_a": rng.normal(size=80),
            "noise_b": rng.normal(size=80),
        }
    )
    y = pd.Series(np.tile([0, 1], 40))

    _, result = select_indexed_columns(
        X,
        np.arange(len(X)),
        y,
        True,
        FeatureDownsampleMethod.FTest,
        n_features_out=2,
        save_all_scores=True,
    )

    assert result.full_score_values is not None
    assert len(result.full_score_values) == X.shape[1]
    exported = pd.concat(list(result.iter_score_frames(chunk_size=2)))
    assert exported["feature"].tolist() == list(X.columns)
    assert not exported.loc[exported["feature"] == "constant", "score_valid"].item()

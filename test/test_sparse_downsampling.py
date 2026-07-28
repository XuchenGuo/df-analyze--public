import bz2
import gzip
import io
import lzma
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file

from df_analyze.cli.cli import ArgumentError, get_options
from df_analyze.downsampling.sparse import (
    _encode_target,
    load_sparse_input,
    sparse_prepared_splits,
)
from df_analyze.enumerables import FeatureDownsampleMethod, ValidationMethod


def test_svmlight_cli_rejects_unsupported_combinations(tmp_path: Path):
    path = tmp_path / "input.svmlight"
    path.write_text("0 1:1\n1 2:1\n", encoding="utf-8")
    base = (
        f"--df {path} --outdir {tmp_path} --mode classify "
        "--feat-downsample variance"
    )

    with pytest.raises(ArgumentError, match="large-feature-mode"):
        get_options(f"{base} --large-feature-mode")
    with pytest.raises(ArgumentError, match="indexed feature downsampling"):
        get_options(f"{base} --feat-downsample mutual-info")
    with pytest.raises(ArgumentError, match="exactly one target"):
        get_options(f"{base} --targets first,second")
    with pytest.raises(ArgumentError, match="table column arguments"):
        get_options(f"{base} --categoricals feature")
    with pytest.raises(ArgumentError, match="requires.*feat-downsample"):
        get_options(f"--df {path} --mode classify")


def test_svmlight_index_base_and_downsample(tmp_path: Path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(240, 80))
    y = (X[:, 9] - X[:, 13] > 0).astype(int)
    path = tmp_path / "input.svmlight"
    dump_svmlight_file(sparse.csr_matrix(X), y, str(path), zero_based=False)

    with pytest.warns(UserWarning, match="ambiguous"):
        loaded = load_sparse_input([path], "auto")
    assert loaded.index_base == 1
    options = SimpleNamespace(
        datapath=path,
        test_paths=[],
        svmlight_index_base="auto",
        feat_downsample=FeatureDownsampleMethod.FTest,
        n_feat_downsample=10,
        downsample_chunk_size=11,
        downsample_screening_fraction=0.25,
        downsample_variance_threshold=None,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )
    train, test, result = sparse_prepared_splits(options)[0]
    assert train.X.shape == (180, 10)
    assert test.X.shape == (60, 10)
    assert result.sparse_input
    for name, index in zip(result.selected_features, result.selected_indices):
        assert name == f"feature_{index + 1}"


def test_external_sparse_targets_use_training_label_mapping():
    train, _, encoder = _encode_target(np.array([1.0, 2.0]), True, "target")
    test, _, _ = _encode_target(np.array([2.0]), True, "target", encoder)
    assert train.tolist() == [0, 1]
    assert test.tolist() == [1]


def test_sparse_lodo_trains_on_all_other_partitions(tmp_path: Path):
    paths = [tmp_path / f"part_{idx}.svmlight" for idx in range(3)]
    rng = np.random.default_rng(22)
    for path in paths:
        X = sparse.csr_matrix(rng.normal(size=(50, 12)))
        y = np.arange(50) % 2
        dump_svmlight_file(X, y, str(path), zero_based=True)
    options = SimpleNamespace(
        datapath=paths[0],
        test_paths=paths[1:],
        tests_method=ValidationMethod.LODO,
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.Variance,
        n_feat_downsample=4,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_variance_threshold=None,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    outputs = sparse_prepared_splits(options)

    assert len(outputs) == 3
    assert [(len(train.X), len(test.X)) for train, test, _ in outputs] == [
        (100, 50),
        (100, 50),
        (100, 50),
    ]


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
def test_compressed_svmlight_loading(tmp_path: Path, suffix, compress):
    buffer = io.BytesIO()
    dump_svmlight_file(sparse.eye(6), np.arange(6) % 2, buffer, zero_based=True)
    path = tmp_path / f"input.svm{suffix}"
    path.write_bytes(compress(buffer.getvalue()))
    loaded = load_sparse_input([path], "auto")
    assert loaded.index_base == 0
    assert loaded.matrices[0].shape == (6, 6)


def test_svmlight_auto_index_base_scans_the_complete_input(tmp_path: Path):
    path = tmp_path / "late-zero.svmlight"
    rows = ["0 1:1"] * 1000 + ["1 0:1"]
    path.write_text("\n".join(rows), encoding="utf-8")

    loaded = load_sparse_input([path], "auto")

    assert loaded.index_base == 0


def test_sparse_path_rejects_constant_training_target(tmp_path: Path):
    path = tmp_path / "constant.svmlight"
    dump_svmlight_file(sparse.eye(20), np.ones(20), str(path), zero_based=True)
    options = SimpleNamespace(
        datapath=path,
        test_paths=[],
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.Variance,
        n_feat_downsample=5,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_variance_threshold=None,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    with pytest.raises(ValueError, match="constant"):
        sparse_prepared_splits(options)


@pytest.mark.parametrize(
    ("external_row", "message"),
    [("0 1:nan", "predictors"), ("inf 1:1", "target")],
)
def test_sparse_path_validates_external_values(
    tmp_path: Path, external_row: str, message: str
):
    train_path = tmp_path / "train.svmlight"
    test_path = tmp_path / "test.svmlight"
    dump_svmlight_file(
        sparse.eye(20), np.arange(20) % 2, str(train_path), zero_based=False
    )
    test_path.write_text(f"{external_row}\n1 2:1\n", encoding="utf-8")
    options = SimpleNamespace(
        datapath=train_path,
        test_paths=[test_path],
        svmlight_index_base="one",
        feat_downsample=FeatureDownsampleMethod.Variance,
        n_feat_downsample=2,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_variance_threshold=None,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    with pytest.raises(ValueError, match=message):
        sparse_prepared_splits(options)

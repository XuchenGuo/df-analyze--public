import bz2
import gzip
import io
import lzma
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file

import df_analyze.downsampling.sparse as sparse_module
from df_analyze.cli.cli import (
    ArgumentError,
    get_options,
    looks_like_svmlight_path,
)
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
        "--feat-downsample normalized-variance"
    )

    with pytest.raises(ArgumentError, match="large-feature-mode"):
        get_options(f"{base} --large-feature-mode")
    with pytest.raises(SystemExit):
        get_options(f"{base} --feat-downsample mutual-info")
    with pytest.raises(ArgumentError, match="exactly one target"):
        get_options(f"{base} --targets first,second")
    with pytest.raises(ArgumentError, match="table column arguments"):
        get_options(f"{base} --categoricals feature")
    with pytest.raises(ArgumentError, match="requires.*feat-downsample"):
        get_options(f"--df {path} --mode classify")


def test_svmlight_cli_accepts_clinical_and_feature_sidecars(tmp_path: Path):
    sparse_path = tmp_path / "input.svmlight"
    metadata_path = tmp_path / "participants.tsv"
    feature_map_path = tmp_path / "features.csv"
    sparse_path.write_text("0 0:1\n1 1:1\n", encoding="utf-8")
    metadata_path.write_text(
        "diagnosis\tsex\tparticipant_id\ncontrol\tF\tsub-1\ncase\tM\tsub-2\n",
        encoding="utf-8",
    )
    feature_map_path.write_text(
        "feature_index,feature_name\n0,crush_0\n1,crush_1\n",
        encoding="utf-8",
    )

    options = get_options(
        f"--df {sparse_path} --outdir {tmp_path} --target diagnosis "
        "--mode classify --feat-downsample normalized-variance "
        f"--svmlight-metadata {metadata_path} "
        f"--svmlight-feature-map {feature_map_path} "
        "--svmlight-sample-id-column participant_id "
        "--categoricals sex --drops participant_id "
        "--downsample-protected-features crush_0"
    )

    assert options.svmlight_metadata == [metadata_path.resolve()]
    assert options.svmlight_feature_map == feature_map_path.resolve()
    assert options.svmlight_sample_id_column == "participant_id"
    assert options.downsample_protected_features == ["crush_0"]
    assert options.categoricals == ["sex"]
    assert options.drops == ["participant_id"]


def test_extensionless_svmlight_auto_detection_supports_external_regression(
    tmp_path: Path,
):
    train_path = tmp_path / "log1p.E2006.train"
    test_path = tmp_path / "log1p.E2006.test"
    train_path.write_text(
        "-3.5 1:0.5 2:1.5 4272227:0.25\n-2.5 1:1.0 3:2.0 4272227:0.50\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "-3.0 1:0.25 2:0.75 4272227:0.10\n",
        encoding="utf-8",
    )

    options = get_options(
        f"--df-train {train_path} --df-tests {test_path} "
        "--df-tests-method list --target target --mode regress "
        f"--feat-downsample normalized-variance --outdir {tmp_path}"
    )

    assert looks_like_svmlight_path(train_path)
    assert looks_like_svmlight_path(test_path)
    assert options.uses_svmlight_input()
    assert options.datapath == train_path.resolve()
    assert options.test_paths == [test_path.resolve()]


def test_known_table_suffix_overrides_svmlight_content_sniff(tmp_path: Path):
    path = tmp_path / "looks_sparse.csv"
    path.write_text("0 1:1 2:2\n", encoding="utf-8")

    assert not looks_like_svmlight_path(path)


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
def test_nonstandard_compressed_svmlight_names_are_content_detected(
    tmp_path: Path, suffix, compress
):
    path = tmp_path / f"log1p.E2006.train{suffix}"
    path.write_bytes(compress(b"-3.5 1:0.5 2:1.5 4272227:0.25\n"))

    assert looks_like_svmlight_path(path)


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
    assert result.resolved_method == FeatureDownsampleMethod.FTest.value
    assert result.source_index_base == 1
    for name, index in zip(result.selected_features, result.selected_indices):
        assert name == f"feature_{index + 1}"
    selected = result.selected_frame()
    assert selected["source_feature_index"].tolist() == [
        index + 1 for index in result.selected_indices
    ]
    scores = result.scores_frame()
    assert scores is not None
    assert scores["source_feature_index"].tolist() == [
        index + 1 for index in result.score_feature_indices or []
    ]
    payload = result.to_dict()
    assert payload["source_index_base"] == 1
    assert payload["selected_source_indices"] == [
        index + 1 for index in result.selected_indices
    ]


def test_sparse_external_reports_all_timing_and_split_shape_phases(tmp_path: Path):
    rng = np.random.default_rng(73)
    train_path = tmp_path / "train.svmlight"
    test_path = tmp_path / "test.svmlight"
    dump_svmlight_file(
        sparse.csr_matrix(rng.normal(size=(60, 10))),
        np.arange(60) % 2,
        str(train_path),
        zero_based=True,
    )
    dump_svmlight_file(
        sparse.csr_matrix(rng.normal(size=(50, 10))),
        np.arange(50) % 2,
        str(test_path),
        zero_based=True,
    )
    options = SimpleNamespace(
        datapath=train_path,
        test_paths=[test_path],
        tests_method=ValidationMethod.List,
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=4,
        downsample_chunk_size=5,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    train, test, result = sparse_prepared_splits(options)[0]
    payload = result.to_dict()
    downsample_report = result.to_markdown()
    preparation_report = train.to_markdown()

    assert result.input_scan_seconds > 0.0
    assert result.train_load_seconds > 0.0
    assert result.test_load_seconds > 0.0
    assert result.fit_seconds >= 0.0
    assert result.transform_seconds >= 0.0
    assert result.total_seconds == pytest.approx(
        result.input_scan_seconds
        + result.train_load_seconds
        + result.fit_seconds
        + result.test_load_seconds
        + result.transform_seconds
    )
    assert payload["total_seconds"] == pytest.approx(result.total_seconds)
    assert "Input layout/target scan time" in downsample_report
    assert "Sparse training matrix load time" in downsample_report
    assert "Sparse test matrix load time" in downsample_report
    assert "not include the source CSR matrix" in downsample_report
    assert preparation_report is not None
    assert "Training input shape:       60 samples × 10 features" in preparation_report
    assert "Training final shape:       60 samples × 4 features" in preparation_report
    assert "External test input shape:  50 samples × 10 features" in preparation_report
    assert "External test final shape:  50 samples × 4 features" in preparation_report
    assert "Data original shape:" not in preparation_report
    assert train.info is not None
    assert test.info is not None
    assert train.info.runtimes["feature downsampling"] == pytest.approx(
        result.total_seconds
    )
    assert test.info.runtimes["feature downsampling"] == pytest.approx(
        result.total_seconds
    )


def test_external_sparse_targets_use_training_label_mapping():
    train, _, encoder = _encode_target(np.array([1.0, 2.0]), True, "target")
    test, _, _ = _encode_target(np.array([2.0]), True, "target", encoder)
    assert train.tolist() == [0, 1]
    assert test.tolist() == [1]


def test_sparse_lodo_uses_each_partition_as_training_set(tmp_path: Path, monkeypatch):
    paths = [tmp_path / f"part_{idx}.svmlight" for idx in range(3)]
    rng = np.random.default_rng(22)
    for path, n_rows in zip(paths, [50, 60, 70]):
        X = sparse.csr_matrix(rng.normal(size=(n_rows, 12)))
        y = np.arange(n_rows) % 2
        dump_svmlight_file(X, y, str(path), zero_based=True)
    options = SimpleNamespace(
        datapath=paths[0],
        test_paths=paths[1:],
        tests_method=ValidationMethod.LODO,
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=4,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    original_loader = sparse_module._load_sparse_matrix
    matrix_refs = []
    peak_live_matrices = 0

    def recording_loader(*args, **kwargs):
        nonlocal peak_live_matrices
        live_before = sum(reference() is not None for reference in matrix_refs)
        matrix = original_loader(*args, **kwargs)
        matrix_refs.append(weakref.ref(matrix))
        peak_live_matrices = max(peak_live_matrices, live_before + 1)
        return matrix

    monkeypatch.setattr(sparse_module, "_load_sparse_matrix", recording_loader)
    outputs = sparse_prepared_splits(options)

    assert len(outputs) == 3
    assert [(len(train.X), len(test.X)) for train, test, _ in outputs] == [
        (50, 130),
        (60, 120),
        (70, 110),
    ]
    assert outputs[0][2].input_scan_seconds > 0.0
    assert [result.input_scan_seconds for _, _, result in outputs[1:]] == [0.0, 0.0]
    assert all(
        any("Reused the input layout" in note for note in result.notes)
        for _, _, result in outputs[1:]
    )
    assert peak_live_matrices <= 2


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


def test_svmlight_auto_index_base_ignores_comment_tokens(tmp_path: Path):
    path = tmp_path / "commented-one-based.svmlight"
    path.write_text("0 1:1 # comment 0:99\n1 2:1\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="ambiguous"):
        loaded = load_sparse_input([path], "auto")

    assert loaded.index_base == 1
    assert loaded.matrices[0].shape == (2, 2)


def test_sparse_path_rejects_constant_training_target(tmp_path: Path):
    path = tmp_path / "constant.svmlight"
    dump_svmlight_file(sparse.eye(20), np.ones(20), str(path), zero_based=True)
    options = SimpleNamespace(
        datapath=path,
        test_paths=[],
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=5,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
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
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=2,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    with pytest.raises(ValueError, match=message):
        sparse_prepared_splits(options)


def test_sparse_scaling_preserves_rare_nonzero_features(tmp_path: Path):
    train_path = tmp_path / "rare-train.svmlight"
    test_path = tmp_path / "rare-test.svmlight"
    train_values = np.zeros((100, 2), dtype=np.float64)
    train_values[:4, 0] = 1.0
    test_values = np.zeros((50, 2), dtype=np.float64)
    test_values[:2, 0] = 1.0
    dump_svmlight_file(
        sparse.csr_matrix(train_values),
        np.arange(100) % 2,
        str(train_path),
        zero_based=True,
    )
    dump_svmlight_file(
        sparse.csr_matrix(test_values),
        np.arange(50) % 2,
        str(test_path),
        zero_based=True,
    )
    options = SimpleNamespace(
        datapath=train_path,
        test_paths=[test_path],
        tests_method=ValidationMethod.List,
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=1,
        downsample_chunk_size=10,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    train, test, result = sparse_prepared_splits(options)[0]

    assert result.selected_indices == [0]
    assert np.count_nonzero(train.X.to_numpy()) == 4
    assert np.count_nonzero(test.X.to_numpy()) == 2


def test_sparse_clinical_sidecar_feature_map_and_protection(tmp_path: Path):
    rng = np.random.default_rng(901)
    values = rng.normal(size=(240, 14))
    target = np.arange(len(values)) % 2
    sparse_path = tmp_path / "crush.svmlight"
    metadata_path = tmp_path / "participants.tsv"
    feature_map_path = tmp_path / "feature_map.csv"
    dump_svmlight_file(
        sparse.csr_matrix(values),
        target,
        str(sparse_path),
        zero_based=True,
    )
    sparse_lines = sparse_path.read_text(encoding="utf-8").splitlines()
    sparse_path.write_text(
        "\n".join(
            f"{line} # participant_id=sub-{idx:04d}"
            for idx, line in enumerate(sparse_lines)
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = pd.DataFrame(
        {
            "participant_id": [f"sub-{idx:04d}" for idx in range(len(target))],
            "diagnosis": np.where(target == 1, "case", "control"),
            "age": rng.integers(18, 70, size=len(target)),
            "sex": np.where(np.arange(len(target)) % 2, "F", "M"),
        }
    )
    metadata.to_csv(metadata_path, sep="\t", index=False)
    feature_map = pd.DataFrame(
        {
            "feature_index": np.arange(values.shape[1]),
            "feature_name": [f"crush_{idx}" for idx in range(values.shape[1])],
            "protected": [idx == 7 for idx in range(values.shape[1])],
        }
    )
    feature_map.to_csv(feature_map_path, index=False)
    options = SimpleNamespace(
        datapath=sparse_path,
        test_paths=[],
        svmlight_index_base="zero",
        svmlight_metadata=[metadata_path],
        svmlight_feature_map=feature_map_path,
        svmlight_sample_id_column="participant_id",
        downsample_protected_features=[],
        feat_downsample=FeatureDownsampleMethod.FTest,
        n_feat_downsample=4,
        downsample_chunk_size=5,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="diagnosis",
        grouper=None,
        categoricals=["sex"],
        ordinals=[],
        drops=["participant_id"],
        separator=",",
        test_val_size=0.25,
        seed=42,
    )

    train, test, result = sparse_prepared_splits(options)[0]

    assert result.protected_features == ["crush_7"]
    assert "crush_7" in train.X
    assert "age" in train.X
    assert any(name.startswith("sex_") for name in train.X)
    assert list(train.X.columns) == list(test.X.columns)
    assert any("clinical metadata" in note for note in result.notes)


def test_sparse_clinical_sidecar_detects_row_misalignment(tmp_path: Path):
    values = sparse.eye(80, 10, format="csr")
    target = np.arange(80) % 2
    sparse_path = tmp_path / "misaligned.svmlight"
    metadata_path = tmp_path / "misaligned.csv"
    dump_svmlight_file(values, target, str(sparse_path), zero_based=True)
    misaligned = target.copy()
    misaligned[0] = 1
    pd.DataFrame(
        {
            "diagnosis": misaligned,
            "age": np.arange(80),
        }
    ).to_csv(metadata_path, index=False)
    options = SimpleNamespace(
        datapath=sparse_path,
        test_paths=[],
        svmlight_index_base="zero",
        svmlight_metadata=[metadata_path],
        svmlight_feature_map=None,
        downsample_protected_features=[],
        feat_downsample=FeatureDownsampleMethod.NormalizedVariance,
        n_feat_downsample=3,
        downsample_chunk_size=5,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="diagnosis",
        grouper=None,
        categoricals=[],
        ordinals=[],
        drops=[],
        separator=",",
        test_val_size=0.25,
        seed=42,
    )

    with pytest.raises(ValueError, match="one-to-one label correspondence"):
        sparse_prepared_splits(options)


def test_sparse_million_feature_smoke(tmp_path: Path):
    rng = np.random.default_rng(902)
    n_rows, n_features = 200, 1_000_000
    rows = np.repeat(np.arange(n_rows), 20)
    columns = rng.integers(0, n_features, size=len(rows))
    columns[-1] = n_features - 1
    data = rng.normal(size=len(rows))
    matrix = sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(n_rows, n_features),
    )
    target = np.arange(n_rows) % 2
    path = tmp_path / "million.svmlight"
    dump_svmlight_file(matrix, target, str(path), zero_based=True)
    options = SimpleNamespace(
        datapath=path,
        test_paths=[],
        svmlight_index_base="zero",
        feat_downsample=FeatureDownsampleMethod.FTest,
        n_feat_downsample=20,
        downsample_chunk_size=100_000,
        downsample_screening_fraction=0.25,
        downsample_save_scores=False,
        is_classification=True,
        target="target",
        test_val_size=0.25,
        seed=42,
    )

    train, test, result = sparse_prepared_splits(options)[0]

    assert result.n_features_in == n_features
    assert train.X.shape == (150, 20)
    assert test.X.shape == (50, 20)

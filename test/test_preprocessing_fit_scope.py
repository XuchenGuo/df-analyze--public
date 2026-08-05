from types import SimpleNamespace

import numpy as np
import pandas as pd

from df_analyze.preprocessing.cleaning import encode_target, encode_targets, reindex
from df_analyze._main import _run
from df_analyze.cli.cli import get_options
from df_analyze.hypertune import _get_splits
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data, usable_training_indices
from df_analyze.selection.models import model_select_features


def test_preprocessing_statistics_are_fit_on_training_rows() -> None:
    n_train = 80
    n_test = 40
    n_rows = n_train + n_test
    df = pd.DataFrame(
        {
            "x": np.r_[np.tile(np.arange(10, dtype=float), 8), np.full(n_test, 100.0)],
            "test_nan": np.r_[
                np.tile(np.arange(4, dtype=float), 20), np.full(n_test, np.nan)
            ],
            "cat": np.r_[np.tile(["a", "b"], 40), np.full(n_test, "unseen")],
            "target": np.tile([0, 1], n_rows // 2),
        }
    )
    idx_train = np.arange(n_train)
    idx_test = np.arange(n_train, n_rows)
    _, inspection = inspect_data(
        df.iloc[idx_train],
        target="target",
        categoricals=["cat"],
        ordinals=["x", "test_nan"],
        _warn=False,
    )

    prepared = prepare_data(
        df,
        target="target",
        grouper=None,
        results=inspection,
        is_classification=True,
        ix_train=idx_train,
        ix_tests=[idx_test],
        _warn=False,
    )

    train, test = next(prepared.get_splits())
    assert train.X_cont is not None
    assert test.X_cont is not None
    assert train.X_cont["x"].max() == 1.0
    assert test.X_cont["x"].min() > 1.0
    assert train.X["x"].max() == 1.0
    assert test.X["x"].min() > 1.0
    assert "test_nan_NAN" not in prepared.X.columns
    assert not any("unseen" in str(column) for column in prepared.X.columns)
    assert np.isfinite(prepared.X.to_numpy()).all()


def test_feat_select_none_skips_model_selectors(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("selector should not run")

    monkeypatch.setattr("df_analyze.selection.models.embed_select_features", fail)
    monkeypatch.setattr("df_analyze.selection.models.wrap_select_features", fail)
    options = SimpleNamespace(
        feat_select=(),
        embed_select=("linear",),
        wrapper_select=("step-up",),
    )

    selected = model_select_features(SimpleNamespace(), options)

    assert selected.embed_selected is None
    assert selected.wrap_selected is None


def test_reindex_preserves_interleaved_partitions() -> None:
    keep = np.array([True, False, True, True, False, True])
    train = np.array([0, 3, 5])
    tests = [np.array([1, 2, 4])]

    train_new, tests_new = reindex(keep, train, tests)

    assert np.array_equal(train_new, np.array([0, 2, 3]))
    assert len(tests_new) == 1
    assert np.array_equal(tests_new[0], np.array([1]))


def test_training_target_cleanup_does_not_use_holdout_counts() -> None:
    target = pd.Series(
        ["a"] * 40
        + ["b"] * 40
        + ["c"] * 5
        + ["a"] * 20
        + ["b"] * 20
        + ["c"] * 30,
        name="target",
    )
    frame = pd.DataFrame({"feature": np.arange(len(target)), "target": target})
    idx_train = np.arange(85)
    idx_test = np.arange(85, len(target))

    _, encoded, labels, train_new, tests_new = encode_target(
        frame,
        target,
        idx_train,
        [idx_test],
        _warn=False,
    )

    assert set(labels.values()) == {"a", "b"}
    assert encoded.nunique() == 2
    assert train_new is not None and len(train_new) == 80
    assert tests_new is not None and len(tests_new[0]) == 40


def test_multitarget_holdout_only_label_is_rejected() -> None:
    train_rows = 40
    frame = pd.DataFrame(
        {
            "feature": np.arange(60),
            "first": np.r_[np.tile(["a", "b"], 20), np.full(20, "held")],
            "second": np.tile(["x", "y"], 30),
        }
    )

    with np.testing.assert_raises_regex(ValueError, "not present in its training"):
        encode_targets(
            frame,
            ["first", "second"],
            np.arange(train_rows),
            [np.arange(train_rows, len(frame))],
            _warn=False,
        )


def test_selected_feature_resolution_uses_exact_names() -> None:
    X = pd.DataFrame(
        {
            "age": [1.0, 2.0],
            "stage": [3.0, 4.0],
            "age2": [5.0, 6.0],
        }
    )
    prepared = SimpleNamespace(X=X, feature_lineage={column: column for column in X})

    X_train, X_test = _get_splits(prepared, prepared, "filter", ["age"])

    assert X_train.columns.tolist() == ["age"]
    assert X_test.columns.tolist() == ["age"]


def test_selected_feature_resolution_expands_declared_lineage() -> None:
    X = pd.DataFrame(
        {
            "cat_blue": [1.0, 0.0],
            "cat_red": [0.0, 1.0],
            "category_count": [2.0, 3.0],
        }
    )
    prepared = SimpleNamespace(
        X=X,
        feature_lineage={
            "cat_blue": "cat",
            "cat_red": "cat",
            "category_count": "category_count",
        },
    )

    X_train, _ = _get_splits(prepared, prepared, "filter", ["cat"])

    assert X_train.columns.tolist() == ["cat_blue", "cat_red"]


def test_inspection_rows_exclude_missing_training_targets() -> None:
    frame = pd.DataFrame(
        {
            "feature": np.arange(60),
            "target": np.r_[np.tile([0, 1], 25), np.full(10, np.nan)],
        }
    )

    rows = usable_training_indices(
        frame,
        target="target",
        is_classification=False,
        indices=np.arange(len(frame)),
    )

    assert np.array_equal(rows, np.arange(50))


def test_standard_cli_path_runs_without_feature_selection(tmp_path) -> None:
    n_rows = 240
    df = pd.DataFrame(
        {
            "x": np.tile(np.arange(12, dtype=float), n_rows // 12),
            "cat": np.tile(["a", "b"], n_rows // 2),
            "target": np.tile([0, 1], n_rows // 2),
        }
    )
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "results"
    df.to_csv(data_path, index=False)
    options = get_options(
        f"--df {data_path} --target target --mode classify "
        "--classifiers dummy --feat-select none --no-preds "
        "--htune-trials 1 --test-val-size 0.4 --device cpu "
        f"--outdir {output_path}"
    )

    _run(options)

    tables = list(output_path.rglob("*final_performances*.csv"))
    assert tables


def test_lodo_refits_preprocessing_for_each_partition(tmp_path) -> None:
    paths = []
    for idx, level in enumerate(["first", "second", "third"]):
        frame = pd.DataFrame(
            {
                "x": np.tile(np.arange(10, dtype=float), 10) + idx,
                "cat": np.full(100, level),
                "target": np.tile([0, 1], 50),
            }
        )
        path = tmp_path / f"partition_{idx}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    output_path = tmp_path / "lodo_results"
    options = get_options(
        f"--df-train {paths[0]} --df-tests {paths[1]},{paths[2]} "
        "--df-tests-method lodo --target target --mode classify "
        "--classifiers dummy --feat-select none --no-preds "
        "--htune-trials 1 --device cpu "
        f"--outdir {output_path}"
    )

    _run(options)

    tables = [
        path
        for path in output_path.rglob("*final_performances*.csv")
        if "per_target" not in path.stem
    ]
    assert len(tables) == 3
    inspection_root = options.program_dirs.inspection
    assert inspection_root is not None
    inspection_dirs = list(inspection_root.glob("test*"))
    assert len(inspection_dirs) == 3

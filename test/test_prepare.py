from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from df_analyze.enumerables import ValidationMethod
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import (
    PreparedData,
    _build_multitarget_audit,
    _ensure_target_levels_in_training,
    prepare_data,
    raw_train_test_indices,
)
from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)


def test_multitarget_split_rejects_level_that_cannot_cover_holdout() -> None:
    n = 40
    df = DataFrame(
        {
            "feature": np.arange(n),
            "target_a": [1, *([0] * (n - 1))],
            "target_b": [0, 1, *([0] * (n - 2))],
        }
    )

    with pytest.raises(ValueError, match="containing every multi-target"):
        raw_train_test_indices(
            df,
            ["target_a", "target_b"],
            grouper=None,
            is_classification=True,
            test_size=0.8,
            seed=42,
        )


def test_multitarget_split_retries_until_holdout_covers_every_level() -> None:
    n = 80
    df = DataFrame(
        {
            "feature": np.arange(n),
            "target_a": np.tile([0, 1], n // 2),
            "target_b": [2, 2, *np.tile([0, 1], (n - 2) // 2)],
        }
    )

    train, test, _ = raw_train_test_indices(
        df,
        ["target_a", "target_b"],
        grouper=None,
        is_classification=True,
        test_size=0.5,
        seed=42,
    )

    for target in ("target_a", "target_b"):
        assert set(df.iloc[train][target]) == set(df[target])
        assert set(df.iloc[test][target]) == set(df[target])


def test_grouped_multitarget_split_retries_with_distinct_seeds() -> None:
    n_groups = 12
    group_size = 3
    groups = np.repeat(np.arange(n_groups), group_size)
    df = DataFrame(
        {
            "feature": np.arange(n_groups * group_size),
            "group": groups,
            "target_a": 0,
            "target_b": 0,
        }
    )
    df.loc[df["group"].isin([2, 9]), "target_a"] = 1
    df.loc[df["group"].isin([3, 11]), "target_b"] = 1

    train, test, _ = raw_train_test_indices(
        df,
        ["target_a", "target_b"],
        grouper="group",
        is_classification=True,
        test_size=0.25,
        seed=0,
    )

    assert set(df.iloc[train]["group"]).isdisjoint(df.iloc[test]["group"])
    for target in ("target_a", "target_b"):
        assert set(df.iloc[train][target]) == set(df[target])
        assert set(df.iloc[test][target]) == set(df[target])


def test_prepared_multitarget_split_rejects_impossible_level_coverage() -> None:
    n = 60
    prepared = PreparedData(
        X=DataFrame({"feature": np.arange(n, dtype=float)}),
        y=DataFrame(
            {
                "target_a": np.tile([0, 1], n // 2),
                "target_b": [1, *([0] * (n - 1))],
            }
        ),
        groups=None,
        is_classification=True,
    )

    with pytest.raises(ValueError, match="containing every multi-target"):
        prepared.split(train_size=0.6, seed=0)


def test_prepared_multitarget_regression_split_rejects_constant_partition() -> None:
    n = 40
    prepared = PreparedData(
        X=DataFrame({"feature": np.arange(n, dtype=float)}),
        y=DataFrame(
            {
                "target_a": np.arange(n, dtype=float),
                "target_b": [1.0, *([0.0] * (n - 1))],
            }
        ),
        groups=None,
        is_classification=False,
    )

    with pytest.raises(
        ValueError,
        match="every target varies in both training and holdout",
    ):
        prepared.split(train_size=0.6, seed=0)


def test_multitarget_regression_split_retries_until_every_target_varies() -> None:
    n = 40
    df = DataFrame(
        {
            "feature": np.arange(n, dtype=float),
            "target_a": np.arange(n, dtype=float),
            "target_b": np.r_[np.ones(2), np.zeros(n - 2)],
        }
    )

    train, test, _ = raw_train_test_indices(
        df,
        ["target_a", "target_b"],
        grouper=None,
        is_classification=False,
        test_size=0.4,
        seed=0,
    )

    for target in ("target_a", "target_b"):
        assert df.iloc[train][target].nunique() >= 2
        assert df.iloc[test][target].nunique() >= 2


def test_multitarget_training_coverage_moves_whole_groups() -> None:
    y = DataFrame(
        {
            "target_a": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "target_b": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    groups = pd.Series(np.repeat(np.arange(5), 2))

    train, test, moved = _ensure_target_levels_in_training(
        y,
        np.array([2, 3, 4, 5]),
        np.array([0, 1, 6, 7, 8, 9]),
        groups,
    )

    assert moved == 4
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    for target in y.columns:
        assert set(y.iloc[train][target]) == set(y[target])


def test_lodo_preserves_public_train_partition_contract() -> None:
    n_rows = 18
    prepared = PreparedData(
        X=DataFrame({"feature": np.arange(n_rows)}),
        y=pd.Series(np.arange(n_rows, dtype=float), name="target"),
        groups=None,
        is_classification=False,
        ix_train=np.arange(0, 4),
        ix_tests=[np.arange(4, 10), np.arange(10, 18)],
        tests_method=ValidationMethod.LODO,
    )

    splits = list(prepared.get_splits())

    assert [(len(train.X), len(test.X)) for train, test in splits] == [
        (4, 14),
        (6, 12),
        (8, 10),
    ]
    assert [train.X["feature"].tolist() for train, _ in splits] == [
        list(range(0, 4)),
        list(range(4, 10)),
        list(range(10, 18)),
    ]


def test_multitarget_preparation_report_contains_audit() -> None:
    n = 49
    df = DataFrame(
        {
            "feature": np.linspace(0.0, 1.0, n),
            "target_a": [0] * 45 + [1] * 4,
            "target_b": np.tile([0, 1], 25)[:n],
        }
    )
    target_cols = ["target_a", "target_b"]
    inspected, inspection = inspect_data(df, target_cols, _warn=False)

    with pytest.warns(UserWarning, match="undersampled levels"):
        prepared = prepare_data(
            inspected,
            target_cols,
            grouper=None,
            results=inspection,
            is_classification=True,
            _warn=False,
        )

    assert prepared.info is not None
    assert prepared.info.multitarget_audit is not None
    assert prepared.info.multitarget_audit.n_original_rows == n
    assert prepared.info.multitarget_audit.n_final_rows == n
    assert "Multi-Target Target Audit" in prepared.to_markdown()


def test_multitarget_audit_records_missingness_patterns() -> None:
    raw = DataFrame(
        {
            "a": [1.0, np.nan, np.nan, 1.0],
            "b": [1.0, 1.0, np.nan, np.nan],
        }
    )
    complete = DataFrame({"a": [1.0], "b": [1.0]})

    audit = _build_multitarget_audit(
        raw, complete, labels=None, is_classification=False
    )

    assert audit.missing_patterns == {"a": 1, "a, b": 1, "b": 1}
    assert "Missing-Target Patterns" in audit.to_markdown()


def do_prepare(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset

    try:
        prepared = ds.prepared(load_cached=False, force=True)
        X = prepared.X
        y = prepared.y
        check_X_y(X, y, y_numeric=True)

        assert prepared.X_cont is not None
        assert prepared.X_cat is not None

        if not prepared.X_cont.empty:
            check_X_y(prepared.X_cont, y, y_numeric=True)
        assert prepared.X_cont.shape[0] == prepared.X_cat.shape[0] == len(y)
        lens = np.array([len(X), len(y), len(prepared.X_cat), len(prepared.X_cont)])  # type: ignore
        assert np.all(lens == lens[0]), "Lengths of returned cardinality splits differ"

        assert prepared.inspection is not None, "Missing inspection data on PreparedData"
        cats = prepared.inspection.cats
        conts = prepared.inspection.conts
        if len(cats.cols.intersection(conts.cols)) > 0:
            raise RuntimeError("Inspection categorical and continuous overlap")

        X_cont, X_cat = prepared.X_cont, prepared.X_cat
        cats = set(X_cat.columns.to_list())
        conts = set(X_cont.columns.to_list())
        if len(cats.intersection(conts)) > 0:
            raise RuntimeError("Returned X_cat and X_cont overlap")

    except ValueError as e:
        if dsname in ["credit-approval_reproduced"]:
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


def do_prep_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset

    try:
        ds.inspect(load_cached=True)
        ds.prepared(load_cached=True)
    except ValueError as e:
        if dsname == "credit-approval_reproduced":
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


@fast_ds
@pytest.mark.regen
def test_prepare_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@med_ds
@pytest.mark.regen
def test_prepare_med(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@slow_ds
@pytest.mark.regen
def test_prepare_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prepare(dataset)


@fast_ds
@pytest.mark.cached
def test_prep_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


@med_ds
@pytest.mark.cached
def test_prep_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


@slow_ds
@pytest.mark.cached
def test_prep_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prep_cached(dataset)


if __name__ == "__main__":
    tinfos = []
    for dsname, ds in tqdm(FAST_INSPECTION):
        if dsname != "abalone":
            continue
        df = ds.load()
        cats = ds.categoricals

        try:
            prepared = ds.prepared(load_cached=False)
            if prepared.inspection is None:
                raise ValueError("Missing inspection data on PreparedData")

            cats = prepared.inspection.cats
            conts = prepared.inspection.conts
            if len(cats.cols.intersection(conts.cols)) > 0:
                raise RuntimeError("Inspection categorical and continuous overlap")

            X_cont, X_cat = prepared.X_cont, prepared.X_cat
            if X_cont is None:
                raise ValueError("Missing X_cont!")
            if X_cat is None:
                raise ValueError("Missing X_cat!")

            cats = set(X_cat.columns.to_list())
            conts = set(X_cont.columns.to_list())
            if len(cats.intersection(conts)) > 0:
                raise RuntimeError("Returned X_cat and X_cont overlap")

            funcs, times = [*zip(*prepared.info["runtimes"].items())]
            tinfo = DataFrame(data=[times], columns=funcs, index=[dsname])
            tinfos.append(tinfo)

        except ValueError as e:
            if dsname in ["credit-approval_reduced", "credit-approval_reproduced"]:
                message = str(e)
                assert "is constant" in message
            else:
                raise ValueError(f"Error for {dsname}") from e
    df = pd.concat(tinfos, axis=0, ignore_index=False)
    out = ROOT / "perf_times.parquet"
    df.to_parquet(out)
    print(f"Saved timings to {out}")
    print(df.max().sort_values(ascending=False))

"""
>>> df.max().sort_values(ascending=False)  # FAST
encode_categoricals        0.427563
unify_nans                 0.162584
deflate_categoricals       0.041391
inspect_target             0.026867
convert_categoricals       0.022642
drop_target_nans           0.018817
encode_target              0.007125
clean_regression_target    0.006605
drop_unusable              0.003869

>>> df.max().sort_values(ascending=False)  # MEDIUM
encode_categoricals        1.712539
unify_nans                 1.679253
inspect_target             0.102524
convert_categoricals       0.057981
encode_target              0.050567
deflate_categoricals       0.045729
drop_target_nans           0.034897
drop_unusable              0.007253
clean_regression_target    0.006596

>>> df.max().sort_values(ascending=False)  # SLOW
encode_categoricals        36.807302
unify_nans                 18.126168
encode_target               1.710700
deflate_categoricals        0.603854
convert_categoricals        0.364583
drop_target_nans            0.242952
clean_regression_target     0.110536
inspect_target              0.055626
drop_unusable               0.043274
"""

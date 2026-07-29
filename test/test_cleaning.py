from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from shutil import get_terminal_size
from sys import stderr

import numpy as np
import pytest
from _pytest.capture import CaptureFixture
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler

from df_analyze._constants import N_CAT_LEVEL_MIN
from df_analyze.enumerables import NanHandling
from df_analyze.preprocessing.cleaning import (
    clean_regression_target,
    clean_regression_targets,
    deflate_categoricals,
    encode_categoricals,
    encode_targets,
    handle_continuous_nans,
)
from df_analyze.preprocessing.inspection.inspection import (
    get_unq_counts,
)
from df_analyze.testing.datasets import (
    TEST_DATASETS,
    TestDataset,
    all_ds,
    fast_ds,
    med_ds,
)


def test_encode_targets_keeps_low_support_rows() -> None:
    n = 30
    df = DataFrame(
        {
            "feature": np.arange(n),
            "target_a": [0] * 28 + [1, 1],
            "target_b": np.tile([0, 1], n // 2),
        }
    )

    cleaned, y, labels, _, _ = encode_targets(df, ["target_a", "target_b"], None, None)

    assert len(cleaned) == n
    assert len(y) == n
    assert list(y.columns) == ["target_a", "target_b"]
    assert labels["target_a"] == {0: "0", 1: "1"}


def test_clean_regression_targets_preserves_units() -> None:
    df = DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target_a": [10.0, 20.0, "NA", 40.0],
            "target_b": [100.0, 200.0, 300.0, 400.0],
        }
    )

    cleaned, y, _, _ = clean_regression_targets(df, ["target_a", "target_b"], None, None)

    assert cleaned["feature"].tolist() == [1, 2, 4]
    assert y["target_a"].tolist() == [10.0, 20.0, 40.0]
    assert y["target_b"].tolist() == [100.0, 200.0, 400.0]


def test_clean_regression_target_uses_original_robust_scaling() -> None:
    df = DataFrame({"feature": [1, 2, 3], "target": [10.0, "NA", 30.0]})

    cleaned, y, _, _ = clean_regression_target(df, df["target"], None, None)

    assert cleaned["feature"].tolist() == [1, 3]
    expected = (
        RobustScaler(quantile_range=(2.5, 97.5))
        .fit_transform(np.array([[10.0], [30.0]]))
        .ravel()
    )
    np.testing.assert_allclose(y.to_numpy(), expected)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_clean_regression_target_rejects_infinite_values(value: float) -> None:
    df = DataFrame({"feature": [1, 2, 3], "target": [10.0, value, 30.0]})

    with pytest.raises(ValueError, match="contains infinite values"):
        clean_regression_target(df, df["target"], None, None)


def test_clean_regression_targets_rejects_infinite_values() -> None:
    df = DataFrame(
        {
            "feature": [1, 2, 3],
            "target_a": [10.0, 20.0, 30.0],
            "target_b": [1.0, np.inf, 3.0],
        }
    )

    with pytest.raises(ValueError, match="'target_b' contains infinite values"):
        clean_regression_targets(df, ["target_a", "target_b"], None, None)


def no_cats(df: DataFrame, target: str) -> bool:
    return (
        df.drop(columns=target)
        .select_dtypes(include=["object", "string[python]"])
        .shape[1]
        == 0
    )


@fast_ds
@pytest.mark.cached
def test_na_handling(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    results = ds.inspect(load_cached=True)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cats = df.loc[:, cats]
    # remember NaNs in target cause dropping in handle_cont_nans below
    # X_cats = X_cats[~X_cats["target"].isna()].drop(columns="target")
    cat_nan_idx = X_cats.isna().to_numpy()

    for nans in [NanHandling.Mean, NanHandling.Median]:
        dfc = handle_continuous_nans(
            df, target="target", grouper=None, results=results, nans=nans
        )[0]
        clean = dfc.drop(columns=["target", *cats])
        assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

        # Check that categorical and target NaNs unaffected
        X_cat_clean = dfc.loc[:, cats]
        cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
        np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)

    for nans in [NanHandling.Drop]:
        try:
            dfc = handle_continuous_nans(
                df, target="target", grouper=None, results=results, nans=nans
            )[0]
            clean = dfc.drop(columns=["target", *cats])
            assert clean.isna().sum().sum() == 0, f"NaNs remaining in data {dsname}"
        except RuntimeError as e:
            if dsname not in [
                "dermatology",
                "colic",
                "colleges",
                "Traffic_violations",
                "hypothyroid",
                "Midwest_Survey_nominal",
            ]:
                raise e


@all_ds
@pytest.mark.cached
@pytest.mark.fast
def test_multivariate_interpolate(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    dsname, ds = dataset
    if dsname in ["community_crime", "news_popularity"]:
        return  # extremely slow
    if dsname in [
        "analcatdata_marketing",
        "analcatdata_reviewer",
        "internet_usage",
        "ipums_la_97-small",
        "kdd_internet_usage",
        "Mercedes_Benz_Greener_Manufacturing",
        "Midwest_Survey_nominal",
        "Midwest_Survey",
        "Midwest_survey2",
        "ozone_level",
        "primary-tumor",
        "soybean",
        "vote",
    ]:
        return  # nothing to impute, no continuous

    df = ds.load()
    results = ds.inspect(load_cached=True)
    cats = [*results.cats.infos.keys(), *results.binaries.infos.keys()]
    X_cats = df.loc[:, cats]
    if not X_cats.empty:
        n_cont = df.shape[1] - X_cats.shape[1]
    else:
        n_cont = df.shape[1] - 1
    cat_nan_idx = X_cats.isna().to_numpy()
    if n_cont > 20:
        return

    results = ds.inspect(load_cached=True)
    with pytest.warns(UserWarning, match="Using experimental multivariate"):
        dfc = handle_continuous_nans(
            df,
            target="target",
            grouper=None,
            results=results,
            nans=NanHandling.Impute,
        )[0]

    clean = dfc.drop(columns=["target", *cats])
    assert clean.isna().sum().sum() == 0, f"NaNs remaning in data {dsname}"

    # Check that categorical and target NaNs unaffected
    X_cat_clean = dfc.loc[:, cats]
    cat_nan_idx_clean = X_cat_clean.isna().to_numpy()
    np.testing.assert_equal(cat_nan_idx, cat_nan_idx_clean)


def do_encode(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    results = ds.inspect(load_cached=True)
    try:
        enc = encode_categoricals(df, target="target", grouper=None, results=results)[0]
    except TypeError as e:
        if dsname == "community_crime" and (
            "Cannot automatically determine the cardinality" in str(e)
        ):
            return
        raise e
    except Exception as e:
        raise ValueError(f"Could not encode categoricals for data: {dsname}") from e
    assert no_cats(enc, target="target"), f"Found categoricals remaining for {dsname}"
    enc = enc.drop(columns="target", errors="ignore")
    has_const = enc.apply(lambda col: len(np.unique(col.apply(str))) == 1).any()
    if has_const:
        consts = enc.columns[enc.apply(lambda col: len(np.unique(col.apply(str))) == 1)]
        raise ValueError(f"Encoding created constant columns: {consts}")


def do_deflate(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    results = ds.inspect(load_cached=True)
    try:
        deflated = deflate_categoricals(
            df.drop(columns="target", errors="ignore"),
            grouper=None,
            results=results,
            _warn=False,
        )
        cols = [info.col for info in results.inflation]
        deflate_cols = sorted(set(deflated.columns).intersection(cols))
        deflate_names = ["DEFLATED", "DEFLATED_OTHER", "DEFLATED_OTHER_DFANALYZE"]
        for col in deflate_cols:
            series = deflated[col].dropna()
            ix_deflated = series.isin(deflate_names)
            series = series[~ix_deflated]
            unqs, cnts = np.unique(series, return_counts=True)
            if len(cnts) <= 0:
                continue
            ix = np.argmin(cnts)
            unq = unqs[ix]
            if np.min(cnts) < N_CAT_LEVEL_MIN:
                raise ValueError(
                    f"Column `{col}` appears to been deflated incorrectly: "
                    f"value `{unq}` has less than {N_CAT_LEVEL_MIN} samples. "
                )
        report = results.short_report()
        print(report)
    except Exception as e:
        raise ValueError(f"Got error deflating dataset: {dsname}") from e


@fast_ds
@pytest.mark.cached
def test_encoding_fast(dataset: tuple[str, TestDataset]) -> None:
    do_encode(dataset)


@med_ds
@pytest.mark.cached
def test_encoding_med(dataset: tuple[str, TestDataset]) -> None:
    do_encode(dataset)


@fast_ds
@pytest.mark.cached
def test_deflate_fast(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    # with capsys.disabled():
    do_deflate(dataset)


@med_ds
@pytest.mark.cached
def test_deflate_med(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    # with capsys.disabled():
    do_deflate(dataset)


# @slow_ds
# @pytest.mark.cached
# def test_encoding_slow(dataset: tuple[str, TestDataset]) -> None:
#     do_encode(dataset)


if __name__ == "__main__":
    for dsname, ds in TEST_DATASETS.items():
        # if dsname != "forest_fires":
        #     continue
        df = ds.load()
        X = df.drop(columns="target")
        dtypes = ["object", "string[python]"]
        unqs = get_unq_counts(df, "target")
        str_cols = X.select_dtypes(include=dtypes).columns.tolist()

        # print(f"Inspecting string/object columns of {dsname}")
        # inspect_str_columns(df, str_cols=str_cols, categoricals=ds.categoricals)
        w = get_terminal_size((81, 24))[0]
        print("#" * w, file=stderr)
        print(f"Checking {dsname}", file=stderr)
        results = ds.inspect(load_cached=True)
        encode_categoricals(df=df, target="target", grouper=None, results=results)
        print("#" * w, file=stderr)
        # input("Continue?")

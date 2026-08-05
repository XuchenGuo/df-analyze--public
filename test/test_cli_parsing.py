from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import EnumMeta
from tempfile import TemporaryDirectory
from typing import Type, Union

import pytest
from cli_test_helpers import ArgvContext

from df_analyze._constants import SEED
from df_analyze.analysis.univariate.associate import (
    CatClsStats,
    CatRegStats,
    ContClsStats,
    ContRegStats,
)
from df_analyze.cli.cli import (
    ProgramOptions,
    Verbosity,
    _split_cli_args,
    get_options,
    random_cli_args,
)
from df_analyze.enumerables import (
    ClsScore,
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    NanHandling,
    Normalization,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.testing.datasets import ALL_DATASETS, TEST_DATASETS, TestDataset, all_ds

requires_test_data = pytest.mark.skipif(
    not TEST_DATASETS, reason="Optional parquet test datasets are not available."
)
PATH = next(iter(TEST_DATASETS.values())).datapath if TEST_DATASETS else Path()


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_classifiers() -> None:
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--categoricals",
        "one,two,three",  # spaces are allowed but count in names; commas separate
    ):
        # opts = get_options(f"--df {PATH} --categoricals one two three")
        opts = get_options()
        assert opts.categoricals == sorted(["one", "two", "three"])


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_quoted_classifiers() -> None:
    # NOTE: can also just confirm manually that this does work to allow
    # column names with spaces in them.
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--categoricals",
        "a one,a two",
        "--verbosity",
        "0",
    ):
        opts = get_options()
    assert opts.categoricals == ["a one", "a two"]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_programmatic_args_preserve_quotes_and_windows_paths() -> None:
    args = (
        r'--df "C:\Users\Jane Doe\data.csv" '
        r'--target "Shell weight" '
        r'--categoricals "CO2 CosIR Value" "MOX 2"'
    )

    assert _split_cli_args(args) == [
        "--df",
        r"C:\Users\Jane Doe\data.csv",
        "--target",
        "Shell weight",
        "--categoricals",
        "CO2 CosIR Value,MOX 2",
    ]


@pytest.mark.fast
def test_programmatic_args_preserve_apostrophes_in_column_names() -> None:
    assert _split_cli_args(
        "--target mother's_qualification --verbosity 0"
    ) == ["--target", "mother's_qualification", "--verbosity", "0"]
    assert _split_cli_args(
        "--target mother's_qualification --categoricals father's_occupation"
    ) == [
        "--target",
        "mother's_qualification",
        "--categoricals",
        "father's_occupation",
    ]


@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_spreadsheet_args_preserve_quoted_columns(tmp_path: Path) -> None:
    spreadsheet = tmp_path / "data with spaces.csv"
    spreadsheet.write_text(
        '--target "Shell weight"\n'
        '--categoricals "CO2 CosIR Value" "MOX 2"\n'
        "\n"
        '"Shell weight","CO2 CosIR Value","MOX 2"\n'
        "0,1,2\n",
        encoding="utf-8",
    )

    opts = get_options(f'--spreadsheet "{spreadsheet}" --verbosity 0')

    assert opts.target == "Shell weight"
    assert opts.categoricals == ["CO2 CosIR Value", "MOX 2"]


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_int_seed() -> None:
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--seed",
        "12345",  # spaces are allowed but count in names; commas separate
    ):
        # opts = get_options(f"--df {PATH} --categoricals one two three")
        opts = get_options()
        assert opts.seed == 12345


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_unseeded() -> None:
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
    ):
        # opts = get_options(f"--df {PATH} --categoricals one two three")
        opts = get_options()
        assert opts.seed == SEED


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_default_seeded() -> None:
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--seed",
        "default",  # spaces are allowed but count in names; commas separate
    ):
        # opts = get_options(f"--df {PATH} --categoricals one two three")
        opts = get_options()
        assert opts.seed == SEED


@requires_test_data
@pytest.mark.fast
@pytest.mark.filterwarnings("ignore:.*not have write permissions.*")
def test_random_seeded() -> None:
    with ArgvContext(
        "df-analyze.py",
        "--df",
        f"{PATH}",
        "--seed",
        "random",  # spaces are allowed but count in names; commas separate
    ):
        # opts = get_options(f"--df {PATH} --categoricals one two three")
        opts = get_options()
        assert isinstance(opts.seed, int)


@all_ds
def test_random_types(dataset: tuple[str, TestDataset]) -> None:
    typings = {
        "datapath": (Path,),
        "target": (str,),
        "categoricals": (list, str),
        "ordinals": (list, str),
        "drops": (list, str),
        "nan_handling": (NanHandling,),
        "norm": (Normalization,),
        "wrapper_model": (WrapperSelectionModel,),
        "filter_assoc_cont_cls": (ContClsStats,),
        "filter_assoc_cat_cls": (CatClsStats,),
        "filter_assoc_cont_reg": (ContRegStats,),
        "filter_assoc_cat_reg": (CatRegStats,),
        "filter_pred_cls_score": (ClsScore,),
        "filter_pred_reg_score": (RegScore,),
        "is_classification": (bool,),
        "htune_trials": (int,),
        "outdir": (Path,),
        "is_spreadsheet": (bool,),
        "separator": (str,),
        "verbosity": (Verbosity,),
        "no_warn_explosion": (bool,),
        "no_preds": (bool,),
    }
    comparables: dict[str, tuple[Union[EnumMeta, Type[None]], ...]] = {
        "feat_select": (FeatureSelection,),
        "embed_select": (EmbedSelectionModel, type(None)),
        "classifiers": (DfAnalyzeClassifier,),
        "regressors": (DfAnalyzeRegressor,),
    }
    nullables = {
        "wrapper_select": (WrapperSelection,),
        "n_feat_wrapper": (int, float),
    }
    numerics = {
        "n_filter_cont": (int, float),
        "n_filter_cat": (int, float),
        "n_feat_filter": (int, float),
        "test_val_size": (int, float),
    }
    all_attrs = sorted(
        set([*typings.keys(), *comparables.keys(), *nullables.keys(), *numerics.keys()])
    )
    dsname, ds = dataset
    with TemporaryDirectory() as tempfile:
        for _ in range(50):
            opts = ProgramOptions.random(ds, outdir=Path(tempfile))
            for attr in all_attrs:
                if not hasattr(opts, attr):
                    raise ValueError(f"ProgramOptions missing attribute: {attr}")

                value = getattr(opts, attr)
                if attr in comparables:
                    if not isinstance(value, tuple):
                        raise TypeError(
                            f"Expected tuple for type of ProgramOptions.{attr}. "
                            f"Got '{type(value)}' instead."
                        )
                    expected = comparables[attr]
                    values = value
                    for val in values:
                        if not isinstance(val, expected):
                            raise TypeError(
                                f"Expected ProgramOptions.{attr} to be instance of "
                                f"{expected}. Got '{type(val)}' instead."
                            )

                elif attr in nullables:
                    expected = (type(None), *nullables[attr])
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Expected one of {expected} for type of ProgramOptions.{attr}. "
                            f"Got '{type(value)}' instead."
                        )

                elif attr in numerics:
                    expected = numerics[attr]
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Expected one of {expected} for type of ProgramOptions.{attr}. "
                            f"Got '{type(value)}' instead."
                        )
                elif attr in typings:
                    expected = typings[attr]
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Expected one of {expected} for type of ProgramOptions.{attr}. "
                            f"Got '{type(value)}' instead."
                        )
                else:
                    raise RuntimeError(
                        f"Impossible! (Attribute {attr} is missing on `ProgramOptions`."
                    )


if __name__ == "__main__":
    for dsname, ds in ALL_DATASETS:
        with TemporaryDirectory() as tempdir:
            args = random_cli_args(ds, Path(tempdir), spreadsheet=True)
            print(args)
            args = random_cli_args(ds, Path(tempdir), spreadsheet=False)
            print(args)

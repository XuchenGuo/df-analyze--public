from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
import random
from collections.abc import Generator
from io import StringIO
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from pytest import CaptureFixture
from tqdm import tqdm

from df_analyze.analysis.univariate.associate import (
    target_associations,
)
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import get_options, random_cli_args
from df_analyze.enumerables import (
    ValidationMethod,
    WrapperSelection,
)
from df_analyze.hypertune import evaluate_tuned
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data
from df_analyze.selection.filter import filter_select_features
from df_analyze.selection.models import model_select_features
from df_analyze.testing.datasets import (
    FASTEST,
    TestDataset,
    fast_ds,
    sparse_snplike_data,
    turbo_ds,
)


@pytest.fixture
def deterministic_cli_randomness() -> Generator[None, None, None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(42)
        np.random.seed(42)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sanity_check_prediction_outputs(ds: TestDataset, preds: list[dict]) -> None:
    for pred in preds:
        probs_test = pred["probs_test"]
        probs_train = pred["probs_train"]

        preds_test = pred["preds_test"]
        preds_train = pred["preds_train"]

        assert preds_test is not None, "Null / missing preds"
        assert preds_train is not None, "Null / missing preds"

        if ds.is_classification:
            assert probs_test is not None, "Null / missing probs"
            assert probs_train is not None, "Null / missing probs"

            assert len(probs_test) == len(preds_test), "Test probs and preds mismatch"


def do_random_spreadsheet(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    with TemporaryDirectory() as tempdir:
        for _ in range(10):
            args, tempfile = random_cli_args(
                ds, Path(tempdir), spreadsheet=True, multitest=False, random_target=True
            )[:-1]
            buf = StringIO()
            ds.load().to_csv(buf, index=False)
            table = buf.getvalue()
            header = "\n".join(args) + "\n\n"
            content = header + table
            tempfile.write_text(content)
            options = get_options(" ".join(args))
            target = options.target
            categoricals = options.categoricals
            ordinals = options.ordinals
            drops = options.drops

            df = options.load_df()
            df, renames = sanitize_names(df, target)
            categoricals = renames.rename_columns(categoricals)
            ordinals = renames.rename_columns(ordinals)
            drops = renames.rename_columns(drops)

            inspect_data(
                df=df,
                target=target,
                grouper=None,
                categoricals=categoricals,
                ordinals=ordinals,
                _warn=True,
            )


# undersampled target levels in some random subsampled test sets due to bad
# target distribution, hard to get enough samples with just 200, easier to
# just ignore than try to handle these by increasing n for random test sets
SKIPS = [
    "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc",
    "Midwest_Survey",
    "Midwest_survey2",
    "Midwest_Survey_nominal",
]


def do_random_full_single(dataset: tuple[str, TestDataset]) -> None:
    """
    Generate multiple test files (say, 1-3) and test various logic related to them. This
    is essentially an integration test.
    """
    dsname, ds = dataset
    if dsname in SKIPS:
        return

    with TemporaryDirectory() as tempdir:
        for _ in range(10):
            args, temp_train_path, temp_testpaths = random_cli_args(
                ds,
                Path(tempdir),
                spreadsheet=False,
                multitest=False,
                random_target=False,
                n_tests_min=1,
                n_tests_max=2,
            )
            df = ds.load()
            df.to_parquet(temp_train_path)

            options = get_options(" ".join(args))
            options.tests_method = ValidationMethod.List
            target = options.target
            categoricals = options.categoricals
            ordinals = options.ordinals
            drops = options.drops
            grouper = options.grouper
            is_cls = options.is_classification
            test_size = options.test_val_size
            prog_dirs = options.program_dirs

            df = options.load_df()
            merges = options.merged_df()
            assert merges is None

            df, renames = sanitize_names(df, target)
            categoricals = renames.rename_columns(categoricals)
            ordinals = renames.rename_columns(ordinals)
            drops = renames.rename_columns(drops)

            df, inspection = inspect_data(
                df=df,
                target=target,
                grouper=None,
                categoricals=categoricals,
                ordinals=ordinals,
                _warn=True,
            )

            prepared = prepare_data(
                df, target, grouper, inspection, is_cls, None, None, options.tests_method
            )
            prog_dirs.save_prepared_raw(prepared)
            prog_dirs.save_prep_report(prepared.to_markdown())
            prog_dirs = options.program_dirs

            # Now test saving

            prep_splits = prepared.get_splits(test_size=test_size)
            for fold_idx, (prep_train, prep_test) in enumerate(prep_splits):
                # prep_train, prep_test = prepared.split()
                fold_idx = None

                associations = target_associations(prep_train)
                prog_dirs.save_univariate_assocs(associations, fold_idx)
                prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

                predictions = univariate_predictions(prep_train, is_cls)
                prog_dirs.save_univariate_preds(predictions, fold_idx)
                prog_dirs.save_pred_report(predictions.to_markdown(), fold_idx)

                # select features via filter methods first
                assoc_filtered, pred_filtered = filter_select_features(
                    prep_train, associations, predictions, options
                )
                prog_dirs.save_filter_report(assoc_filtered, fold_idx)
                prog_dirs.save_filter_report(pred_filtered, fold_idx)

                # TODO: make embedded and wrapper selection mutually exclusive. Only two
                # phases of feature selection: filter selection, and model-based
                # selection, wher model-based selection means either embedded or wrapper
                # (stepup, stepdown) methods.
                selected = model_select_features(prep_train, options)
                prog_dirs.save_model_selection_reports(selected, fold_idx)
                prog_dirs.save_model_selection_data(selected, fold_idx)

                silence_spam()
                eval_results = evaluate_tuned(
                    prepared=prepared,
                    prep_train=prep_train,
                    prep_test=prep_test,
                    assoc_filtered=assoc_filtered,
                    pred_filtered=pred_filtered,
                    model_selected=selected,
                    options=options,
                )
                prog_dirs.save_eval_report(eval_results, fold_idx)
                prog_dirs.save_eval_tables(eval_results, fold_idx)
                prog_dirs.save_eval_data(eval_results, fold_idx)
                print("Finished fold")

                assert prog_dirs.results is not None
                preds_path = prog_dirs.results / "prediction_results.json"
                preds = json.loads(preds_path.read_text())
                try:
                    sanity_check_prediction_outputs(ds, preds["predictions"])
                except AssertionError as e:
                    print()
                    raise e

            # Now inspect files
            report_path = prog_dirs.make_summary_report()
            print(report_path.read_text())


def do_random_multitests(dataset: tuple[str, TestDataset]) -> None:
    """
    Generate multiple test files (say, 1-3) and test various logic related to them. This
    is essentially an integration test.
    """
    dsname, ds = dataset
    if dsname in SKIPS:
        return

    with TemporaryDirectory() as tempdir:
        for _ in range(3):
            args, temp_train_path, temp_testpaths = random_cli_args(
                ds, Path(tempdir), spreadsheet=False, multitest=True, random_target=False
            )

            options = get_options(" ".join(args))
            target = options.target
            categoricals = options.categoricals
            ordinals = options.ordinals
            drops = options.drops
            grouper = options.grouper
            is_cls = options.is_classification
            method = options.tests_method
            test_size = options.test_val_size

            ds.to_multitest(temp_train_path, temp_testpaths)

            merges = options.merged_df()
            assert merges is not None, "Failure loading or making merged df"
            merged_df, ix_train, ix_tests = merges

            merged_df, renames = sanitize_names(merged_df, target)
            categoricals = renames.rename_columns(categoricals)
            ordinals = renames.rename_columns(ordinals)
            drops = renames.rename_columns(drops)

            merged_df, inspection = inspect_data(
                df=merged_df,
                target=target,
                grouper=None,
                categoricals=categoricals,
                ordinals=ordinals,
                _warn=True,
            )

            prepared = prepare_data(
                merged_df, target, grouper, inspection, is_cls, ix_train, ix_tests, method
            )

            # Now test saving
            prog_dirs = options.program_dirs

            prep_splits = prepared.get_splits(test_size=test_size)
            for fold_idx, (prep_train, prep_test) in enumerate(prep_splits):
                # prep_train, prep_test = prepared.split()
                if merged_df is None:
                    fold_idx = None

                associations = target_associations(prep_train)
                prog_dirs.save_univariate_assocs(associations, fold_idx)
                prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

            # Now we need to check that we actually have the files
            n_test = len(temp_testpaths)
            assert prog_dirs.associations is not None

            if not inspection.cats.is_empty:
                pqs = sorted(prog_dirs.associations.rglob("*categorical*.parquet"))
                assert len(pqs) == n_test
                csvs = sorted(prog_dirs.associations.rglob("*categorical*.csv"))
                assert len(csvs) == n_test

            if not inspection.conts.is_empty:
                pqs = sorted(prog_dirs.associations.rglob("*continuous*.parquet"))
                assert len(pqs) == n_test
                csvs = sorted(prog_dirs.associations.rglob("*continuous*.csv"))
                assert len(csvs) == n_test

            mds = sorted(prog_dirs.associations.rglob("*.md"))
            assert len(mds) == n_test


def do_random_full_multitests(dataset: tuple[str, TestDataset]) -> None:
    """
    Generate multiple test files (say, 1-3) and test various logic related to them. This
    is essentially an integration test.
    """
    dsname, ds = dataset
    if dsname in SKIPS:
        return

    with TemporaryDirectory() as tempdir:
        for method in [ValidationMethod.List, ValidationMethod.LODO]:
            args, temp_train_path, temp_testpaths = random_cli_args(
                ds,
                Path(tempdir),
                spreadsheet=False,
                multitest=True,
                random_target=False,
                n_tests_min=2 if method is ValidationMethod.List else 1,
                n_tests_max=3 if method is ValidationMethod.List else 2,
            )

            options = get_options(" ".join(args))
            options.tests_method = method
            target = options.target
            categoricals = options.categoricals
            ordinals = options.ordinals
            drops = options.drops
            grouper = options.grouper
            is_cls = options.is_classification
            test_size = options.test_val_size
            prog_dirs = options.program_dirs

            ds.to_multitest(temp_train_path, temp_testpaths)

            merges = options.merged_df()
            assert merges is not None, "Failure loading or making merged df"
            merged_df, ix_train, ix_tests = merges

            merged_df, renames = sanitize_names(merged_df, target)
            categoricals = renames.rename_columns(categoricals)
            ordinals = renames.rename_columns(ordinals)
            drops = renames.rename_columns(drops)

            merged_df, inspection = inspect_data(
                df=merged_df,
                target=target,
                grouper=None,
                categoricals=categoricals,
                ordinals=ordinals,
                _warn=True,
            )

            prepared = prepare_data(
                merged_df, target, grouper, inspection, is_cls, ix_train, ix_tests, method
            )
            prog_dirs.save_prepared_raw(prepared)
            prog_dirs.save_prep_report(prepared.to_markdown())
            prog_dirs = options.program_dirs

            # Now test saving

            prep_splits = prepared.get_splits(test_size=test_size)
            for fold_idx, (prep_train, prep_test) in enumerate(prep_splits):
                # prep_train, prep_test = prepared.split()
                if merged_df is None:
                    fold_idx = None

                associations = target_associations(prep_train)
                prog_dirs.save_univariate_assocs(associations, fold_idx)
                prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

                predictions = univariate_predictions(prep_train, is_cls)
                prog_dirs.save_univariate_preds(predictions, fold_idx)
                prog_dirs.save_pred_report(predictions.to_markdown(), fold_idx)

                # select features via filter methods first
                assoc_filtered, pred_filtered = filter_select_features(
                    prep_train, associations, predictions, options
                )
                prog_dirs.save_filter_report(assoc_filtered, fold_idx)
                prog_dirs.save_filter_report(pred_filtered, fold_idx)

                # TODO: make embedded and wrapper selection mutually exclusive. Only two
                # phases of feature selection: filter selection, and model-based
                # selection, wher model-based selection means either embedded or wrapper
                # (stepup, stepdown) methods.
                selected = model_select_features(prep_train, options)
                prog_dirs.save_model_selection_reports(selected, fold_idx)
                prog_dirs.save_model_selection_data(selected, fold_idx)

                silence_spam()
                eval_results = evaluate_tuned(
                    prepared=prepared,
                    prep_train=prep_train,
                    prep_test=prep_test,
                    assoc_filtered=assoc_filtered,
                    pred_filtered=pred_filtered,
                    model_selected=selected,
                    options=options,
                )
                prog_dirs.save_eval_report(eval_results, fold_idx)
                prog_dirs.save_eval_tables(eval_results, fold_idx)
                prog_dirs.save_eval_data(eval_results, fold_idx)
                print("Finished fold")

            # Now inspect files
            report_path = prog_dirs.make_summary_report()
            print(report_path.read_text())


@fast_ds
def test_rand_spreadsheet(
    dataset: tuple[str, TestDataset], deterministic_cli_randomness
) -> None:
    do_random_spreadsheet(dataset)


@fast_ds
def test_rand_multitest(
    dataset: tuple[str, TestDataset], deterministic_cli_randomness
) -> None:
    do_random_multitests(dataset)


@turbo_ds
def test_rand_full_multitest(
    dataset: tuple[str, TestDataset], deterministic_cli_randomness
) -> None:
    do_random_full_multitests(dataset)


@turbo_ds
def test_rand_full_single(
    dataset: tuple[str, TestDataset], deterministic_cli_randomness
) -> None:
    do_random_full_single(dataset)


def summarize_sparse(X: DataFrame, y: Series, rev: bool = False) -> None:
    n_const = (X.var(skipna=True, axis=0) <= 0).sum()
    df = pd.get_dummies(X.astype("category"), dummy_na=True, drop_first=False).astype(int)

    distractors = df.filter(regex="x.*")
    predictive = df.filter(regex="v.*")
    discrim = df.filter(regex="d.*")

    r_dist = distractors.corrwith(y).abs().max()
    r_pred = predictive.corrwith(y).abs().max()
    r_disc = discrim.corrwith(y).abs().max()

    lead = "[REV]" if rev else "     "
    print(f"{lead} n_const: {n_const}, ", end="")
    print(f"Distract_r: {r_dist:0.3f}, Pred_r: {r_pred:0.3f}, Discrim_r: {r_disc:0.3f}")


def rand_sparse_data(N_samples: int = 1500) -> tuple[DataFrame, int]:
    for _ in range(20):
        seed = np.random.randint(1, 10**8)
        rng = np.random.default_rng(seed)
        try:
            X, y = sparse_snplike_data(
                mode="classify",
                N=N_samples,
                n_allele=4,
                n_distractor_snps=rng.choice([50, 100]),  # dimensionality of D
                n_predictive_snps=rng.integers(2, 7),  # dimensionality of V'
                n_discrim_snps=rng.integers(3, 7),  # dimensionality of H
                n_predictive_variants=rng.choice([*range(1, 10)]),
                modal_allele_frequency=0.9,
                p_target_if_predictive_variant=rng.uniform(0.60, 0.95),
                freq_target=0.4,  # percent of samples where target is 1
                p_nan=0.01,  # percent chance an allele is NaN
                rev=rng.choice([True, False]),
                rng=rng,
            )
            df = pd.concat([X, y], axis=1)
            return df, seed

        except ValueError as e:
            if "Couldn't generate" not in str(e):
                raise e
    raise ValueError("Couldn't generate sparse data.")


def inspect_sparse_data(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        print("")
        # for n_related in [3, 5, 10, 20]:
        errs = []
        errs_rev = []
        infos = []
        for _ in tqdm(range(1000), disable=False, ncols=80, colour="blue"):
            seed = np.random.randint(1, 10**8)
            rng = np.random.default_rng(seed)
            # X_train, X_test, y_train, y_test = sparse_snplike_data(
            n_dist = rng.choice([50, 100, 200])
            n_pred = rng.integers(2, 7)
            n_disc = rng.integers(3, 7)
            n_var = rng.choice([*range(1, 10)])  # should be less than 8 generally
            p_targ = rng.uniform(0.60, 0.95)
            rev = rng.choice([True, False])
            info = {
                "n_dist": n_dist,
                "n_pred": n_pred,
                "n_disc": n_disc,
                "n_var": n_var,
                "p_targ": p_targ,
                "rev": rev,
                "seed": seed,
                "n_const": np.nan,
                "r_dist": np.nan,
                "r_pred": np.nan,
                "r_disc": np.nan,
                "fail": 1,
            }
            try:
                X, y = sparse_snplike_data(
                    mode="classify",
                    N=1500,
                    n_allele=4,
                    n_distractor_snps=n_dist,  # dimensionality of D
                    n_predictive_snps=n_pred,  # dimensionality of V'
                    n_discrim_snps=n_disc,  # dimensionality of H
                    n_predictive_variants=n_var,  # number of unique variants in V
                    modal_allele_frequency=0.9,
                    p_target_if_predictive_variant=p_targ,
                    freq_target=0.2,  # percent of samples where target is 1
                    p_nan=0.01,  # percent chance an allele is NaN
                    rng=rng,
                    rev=rev,
                )

                df = pd.get_dummies(
                    X.astype("category"), dummy_na=True, drop_first=False
                ).astype(int)
                distractors = df.filter(regex="x.*")
                predictive = df.filter(regex="v.*")
                discrim = df.filter(regex="d.*")
                info["n_const"] = (X.var(skipna=True, axis=0) <= 0).sum()
                info["r_dist"] = distractors.corrwith(y).abs().max()
                info["r_pred"] = predictive.corrwith(y).abs().max()
                info["r_disc"] = discrim.corrwith(y).abs().max()
                info["fail"] = 0

            except ValueError:
                append = errs_rev if rev else errs
                append.append(
                    f"FAIL: n_distract={n_dist:3d}, n_predictive={n_pred}, n_discrim={n_disc}, "
                    f"n_variants={n_var}, p_target={p_targ:0.2f}"
                )
            infos.append(info)
        info = pd.concat(
            [DataFrame(inf, index=[0]) for inf in infos], axis=0, ignore_index=True
        )

        pd.options.display.max_columns = 20
        pd.options.display.max_info_columns = 20
        pd.options.display.width = 400

        corrs = info.corr(method="spearman").round(3)
        print("All cases")
        print(corrs)

        corrs = info.loc[info["fail"] == 0].corr(method="spearman").round(3)
        print("Success cases")
        print(corrs)

        corrs = info.loc[info["fail"] == 1].corr(method="spearman").round(3)
        print("Fail cases")
        print(corrs)

        for err in errs:
            print(err)
        for err in errs_rev:
            print(err)
        print("N_fail: ", len(errs))
        print("N_fail_reversed: ", len(errs_rev))


def test_sparse_data(capsys: CaptureFixture) -> None:
    classification_datasets = [ds for _, ds in FASTEST if ds.is_classification]
    if not classification_datasets:
        pytest.skip("Bundled classification test datasets are not available")
    ds = classification_datasets[0]
    with capsys.disabled():
        with TemporaryDirectory() as tempdir:
            for _ in range(10):
                args, temp_train_path, temp_testpaths = random_cli_args(
                    ds,
                    Path(tempdir),
                    spreadsheet=False,
                    multitest=False,
                    random_target=False,
                    n_tests_min=1,
                    n_tests_max=2,
                )
                df, seed = rand_sparse_data(N_samples=400)
                df.to_parquet(temp_train_path)
                rng = np.random.default_rng()

                options = get_options(" ".join(args))
                options.tests_method = ValidationMethod.List
                if options.wrapper_select is WrapperSelection.StepDown:
                    options.wrapper_select = WrapperSelection.StepUp
                target = "target"
                categoricals = [*df.columns][:-1]
                ordinals = []
                drops = options.drops
                grouper = None
                is_cls = True
                test_size = 0.4
                options.n_feat_wrapper = rng.integers(1, 6)
                prog_dirs = options.program_dirs

                df = options.load_df()
                merges = options.merged_df()
                assert merges is None

                df, renames = sanitize_names(df, target)
                categoricals = renames.rename_columns(categoricals)
                ordinals = renames.rename_columns(ordinals)
                drops = renames.rename_columns(drops)

                df, inspection = inspect_data(
                    df=df,
                    target=target,
                    grouper=None,
                    categoricals=categoricals,
                    ordinals=ordinals,
                    _warn=True,
                )

                prepared = prepare_data(
                    df,
                    target,
                    grouper,
                    inspection,
                    is_cls,
                    None,
                    None,
                    options.tests_method,
                )
                prog_dirs.save_prepared_raw(prepared)
                prog_dirs.save_prep_report(prepared.to_markdown())
                prog_dirs = options.program_dirs

                # Now test saving

                prep_splits = prepared.get_splits(test_size=test_size)
                for fold_idx, (prep_train, prep_test) in enumerate(prep_splits):
                    # prep_train, prep_test = prepared.split()
                    fold_idx = None

                    associations = target_associations(prep_train)
                    prog_dirs.save_univariate_assocs(associations, fold_idx)
                    prog_dirs.save_assoc_report(associations.to_markdown(), fold_idx)

                    predictions = univariate_predictions(prep_train, is_cls)
                    prog_dirs.save_univariate_preds(predictions, fold_idx)
                    prog_dirs.save_pred_report(predictions.to_markdown(), fold_idx)

                    # select features via filter methods first
                    assoc_filtered, pred_filtered = filter_select_features(
                        prep_train, associations, predictions, options
                    )
                    prog_dirs.save_filter_report(assoc_filtered, fold_idx)
                    prog_dirs.save_filter_report(pred_filtered, fold_idx)

                    # TODO: make embedded and wrapper selection mutually exclusive. Only two
                    # phases of feature selection: filter selection, and model-based
                    # selection, wher model-based selection means either embedded or wrapper
                    # (stepup, stepdown) methods.
                    selected = model_select_features(prep_train, options)
                    prog_dirs.save_model_selection_reports(selected, fold_idx)
                    prog_dirs.save_model_selection_data(selected, fold_idx)

                    silence_spam()
                    eval_results = evaluate_tuned(
                        prepared=prepared,
                        prep_train=prep_train,
                        prep_test=prep_test,
                        assoc_filtered=assoc_filtered,
                        pred_filtered=pred_filtered,
                        model_selected=selected,
                        options=options,
                    )
                    prog_dirs.save_eval_report(eval_results, fold_idx)
                    prog_dirs.save_eval_tables(eval_results, fold_idx)
                    prog_dirs.save_eval_data(eval_results, fold_idx)
                    print("Finished fold")

                    assert prog_dirs.results is not None
                    preds_path = prog_dirs.results / "prediction_results.json"
                    preds = json.loads(preds_path.read_text())
                    try:
                        sanity_check_prediction_outputs(ds, preds["predictions"])
                    except AssertionError as e:
                        print(f"Failure with seed: {seed}")
                        raise e

                # Now inspect files
                report_path = prog_dirs.make_summary_report()
                print(report_path.read_text())

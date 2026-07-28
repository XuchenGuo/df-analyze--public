from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from random import randint
from time import perf_counter
from typing import Any, Callable, Optional

from _pytest.capture import CaptureFixture

from df_analyze._constants import ROOT
from df_analyze.analysis.univariate.associate import CatAssociation, ContAssociation
from df_analyze.cli.cli import ProgramOptions
from df_analyze.enumerables import (
    ClsScore,
    EmbedSelectionModel,
    RegScore,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.nonsense import silence_spam
from df_analyze.selection.embedded import embed_select_features
from df_analyze.selection.filter import (
    FilterSelected,
    filter_by_univariate_associations,
    filter_by_univariate_predictions,
)
from df_analyze.selection.multitarget import aggregate_wrapper_selected
from df_analyze.selection.stepwise import (
    StepwiseSelector,
    stepwise_select,
)
from df_analyze.selection.wrapper import WrapperSelected
from df_analyze.testing.datasets import (
    ALL_DATASETS,
    TestDataset,
    all_ds,
    fast_ds,
    med_ds,
    slow_ds,
)

DATA = ROOT / "data/banking/bank.json"
RUNTIMES = ROOT / "runtimes"
RUNTIMES.mkdir(exist_ok=True)


def test_multitarget_wrapper_aggregation_uses_oriented_scores() -> None:
    first = WrapperSelected(
        method=WrapperSelection.StepUp,
        model=WrapperSelectionModel.Linear,
        selected=["feature_a", "feature_b"],
        scores={"feature_a": -0.2, "feature_b": -0.8},
        redundants=[],
        early_stop=False,
        is_classification=False,
    )
    second = WrapperSelected(
        method=WrapperSelection.StepUp,
        model=WrapperSelectionModel.Linear,
        selected=["feature_a", "feature_b"],
        scores={"feature_a": -0.1, "feature_b": -0.7},
        redundants=[],
        early_stop=False,
        is_classification=False,
    )

    selected = aggregate_wrapper_selected(
        [first, second], target_names=["a", "b"], top_k=1
    )

    assert selected is not None
    assert selected.selected == ["feature_a"]


def test_multitarget_aggregation_without_top_k_keeps_union() -> None:
    first = WrapperSelected(
        method=WrapperSelection.StepUp,
        model=WrapperSelectionModel.Linear,
        selected=["a", "b"],
        scores={"a": 2.0, "b": 1.0},
        redundants=[],
        early_stop=False,
        is_classification=True,
    )
    second = WrapperSelected(
        method=WrapperSelection.StepUp,
        model=WrapperSelectionModel.Linear,
        selected=["c", "d"],
        scores={"c": 2.0, "d": 1.0},
        redundants=[],
        early_stop=False,
        is_classification=True,
    )

    selected = aggregate_wrapper_selected(
        [first, second], target_names=["a", "b"], top_k=None
    )

    assert selected is not None
    assert set(selected.selected) == {"a", "b", "c", "d"}


def do_association_select(dataset: tuple[str, TestDataset]) -> FilterSelected:
    dsname, ds = dataset
    assocs = ds.associations(load_cached=True)
    prepared = ds.prepared(load_cached=True)
    for _ in range(25):
        options = ProgramOptions.random(ds)
        filtered = filter_by_univariate_associations(
            prepared=prepared,
            associations=assocs,
            cont_metric=ContAssociation.random(),
            cat_metric=CatAssociation.random(),
            n_cont=options.n_filter_cont,
            n_cat=options.n_filter_cat,
            n_total=options.n_feat_filter,
        )
        print(filtered.cont_scores)
        print(filtered.cat_scores)
        assert len(filtered.selected) > 0
    return filtered  # type: ignore


def do_predict_select(dataset: tuple[str, TestDataset]) -> Optional[FilterSelected]:
    dsname, ds = dataset
    if dsname == "internet_usage":  # undersampled levels in target
        return
    predictions = ds.predictions(load_cached=True)
    prepared = ds.prepared(load_cached=True)
    for _ in range(25):
        options = ProgramOptions.random(ds)
        filtered = filter_by_univariate_predictions(
            prepared=prepared,
            predictions=predictions,
            cont_metric=RegScore.random(),
            cat_metric=ClsScore.random(),
            n_cont=options.n_filter_cont,
            n_cat=options.n_filter_cat,
        )
        print(filtered.cont_scores)
        print(filtered.cat_scores)
        assert len(filtered.selected) > 0
    return filtered  # pyright: ignore


def do_embed_select(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname == "internet_usage":  # undersampled levels in target
        return
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]

    options = ProgramOptions.random(ds)
    options.embed_select = (EmbedSelectionModel.LGBM,)

    selecteds = embed_select_features(prep_train=prep_train, options=options)
    for selected in selecteds:
        if selected is None or (selected.selected is None):
            raise ValueError("Impossible!")
        assert len(selected.selected) > 0


def estimate_select(
    dataset: tuple[str, TestDataset],
    file: Path,
    forward: bool,
    model: WrapperSelectionModel,
    subsample: bool = True,
) -> float:
    dsname, ds = dataset
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0] if subsample else prepared
    n, p = prep_train.X.shape
    m = 10

    options = ProgramOptions.random(ds)
    options.wrapper_model = model
    options.wrapper_select = (
        WrapperSelection.StepUp if forward else WrapperSelection.StepDown
    )
    options.n_feat_wrapper = m if forward else p - m
    if (p <= m) and forward:
        return float("nan")
    if (p - m <= 0) and (not forward):
        return float("nan")

    selector = StepwiseSelector(
        prep_train=prep_train,
        options=options,
        n_features=options.n_feat_wrapper,
        direction=options.wrapper_select.direction(),
    )
    minutes = selector.estimate_runtime()
    N = prepared.X.shape[0]
    needs_header = False
    if file.exists():
        with open(file, "r") as handle:
            if handle.readline().strip() == "":
                needs_header = True
    else:
        needs_header = True

    with open(file, "a") as handle:
        if needs_header:
            handle.write(
                f"{'dsname':>40}  {'N':>6}  {'N_sub':>6}  {'p':>5}  {'n_iter':>6}  {'minutes':>4}\n"
            )
        handle.write(f"{dsname:>40}  {N:>6d}  {n:>6d}  {p:5d}  {m:>6d}  {minutes:3.1f}\n")
        handle.flush()

    return minutes


def estimate_linear_forward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"linear_forward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.Linear
    estimate_select(dataset, file=file, forward=True, model=model, subsample=subsample)


def estimate_linear_backward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"linear_backward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.Linear
    estimate_select(dataset, file=file, forward=False, model=model, subsample=subsample)


def estimate_lgbm_forward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"lgbm_forward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.LGBM
    estimate_select(dataset, file=file, forward=True, model=model, subsample=subsample)


def estimate_lgbm_backward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"lgbm_backward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.LGBM
    estimate_select(dataset, file=file, forward=False, model=model, subsample=subsample)


def estimate_knn_forward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"knn_forward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.KNN
    estimate_select(dataset, file=file, forward=True, model=model, subsample=subsample)


def estimate_knn_backward_select(
    dataset: tuple[str, TestDataset],
    subsample: bool = True,
) -> None:
    extra = "_no_subsample" if not subsample else ""
    file = RUNTIMES / f"knn_backward_select_runtime_estimates{extra}.txt"
    model = WrapperSelectionModel.KNN
    estimate_select(dataset, file=file, forward=False, model=model, subsample=subsample)


def do_redundant_report(dataset: tuple[str, TestDataset], capsys: CaptureFixture) -> None:
    dsname, ds = dataset
    if dsname == "internet_usage":  # undersampled target
        return
    selected = WrapperSelected.random(ds)
    report = selected.to_markdown()
    with capsys.disabled():
        print(report)


@all_ds
def test_redundant_report(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    do_redundant_report(dataset=dataset, capsys=capsys)


def do_forward_select(
    dataset: tuple[str, TestDataset],
    linear: bool,
    redundant: bool = False,
    test: bool = False,
) -> None:
    silence_spam()
    dsname, ds = dataset
    if dsname == "internet_usage":  # undersampled target
        return
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    options.redundant_selection = redundant
    options.wrapper_model = (
        WrapperSelectionModel.Linear if linear else WrapperSelectionModel.LGBM
    )
    options.wrapper_select = WrapperSelection.StepUp
    options.n_feat_wrapper = 10
    if prepared.X.shape[1] <= 10:
        return
    results = stepwise_select(prep_train=prep_train, options=options, test=test)
    if results is None:
        raise ValueError("Impossible")
    selected, scores, redundants, early_stop = results
    if (len(selected) != options.n_feat_wrapper) and (not early_stop):
        raise ValueError("Did not select correct number of features")

    # check that redundant sets are disjoint
    if len(redundants) > 1:
        for i in range(len(redundants) - 1):
            r1 = redundants[i].features
            r2 = redundants[i + 1].features
            assert len(set(r1).intersection(r2)) == 0

    for fname, score in scores.items():
        print(f"{fname:>30}  {round(score, 3)}")


def do_forward_pseudo_select(dataset: tuple[str, TestDataset]) -> None:
    """Use the `test=True` option to get random scores instead

    NOTE: We don't need to test both linear or not in this case, as the
    model is not used.
    """
    do_forward_select(dataset=dataset, linear=True, test=True)


def do_forward_pseudo_select_redundant(dataset: tuple[str, TestDataset]) -> None:
    """Use the `test=True` option to get random scores instead

    NOTE: We don't need to test both linear or not in this case, as the
    model is not used.
    """
    do_forward_select(dataset=dataset, linear=True, test=True, redundant=True)


def do_linear_forward_select(dataset: tuple[str, TestDataset]) -> None:
    do_forward_select(dataset, linear=True)


def do_lgbm_forward_select(dataset: tuple[str, TestDataset]) -> None:
    do_forward_select(dataset, linear=False)


def do_backward_select(
    dataset: tuple[str, TestDataset],
    linear: bool,
    redundant: bool = False,
    test: bool = False,
) -> None:
    silence_spam()
    dsname, ds = dataset
    if dsname == "internet_usage":  # undersampled target
        return
    prepared = ds.prepared(load_cached=True)
    prep_train = prepared.representative_subsample()[0]
    options = ProgramOptions.random(ds)
    options.redundant_selection = redundant
    options.wrapper_model = (
        WrapperSelectionModel.Linear if linear else WrapperSelectionModel.LGBM
    )
    options.wrapper_select = WrapperSelection.StepDown
    p = prepared.X.shape[1]
    pmin = max(1, p - 10)
    pmax = p - 1
    p_select = randint(pmin, pmax)
    options.n_feat_wrapper = p_select
    if options.n_feat_wrapper <= 0:
        return
    results = stepwise_select(prep_train=prep_train, options=options, test=test)
    if results is None:
        raise ValueError("Impossible")
    selected, scores, redundants, early_stop = results
    if (len(selected) > p_select) and (not early_stop):
        raise ValueError(f"Expected {p_select} to be selected: got {len(selected)}")

    # check that redundant sets are disjoint
    if len(redundants) > 1:
        for i in range(len(redundants) - 1):
            r1 = redundants[i].features
            r2 = redundants[i + 1].features
            assert len(set(r1).intersection(r2)) == 0

    for fname, score in scores.items():
        print(f"{fname:>30}  {round(score, 3)}")


def do_backward_pseudo_select(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset=dataset, linear=True, test=True)


def do_backward_pseudo_select_redundant(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset=dataset, linear=True, test=True, redundant=True)


def do_linear_backward_select(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset, linear=True)


def do_lgbm_backward_select(dataset: tuple[str, TestDataset]) -> None:
    do_backward_select(dataset, linear=False)


def do_logged(
    f: Callable[[tuple[str, TestDataset]], Any],
    file: Path,
    dataset: tuple[str, TestDataset],
) -> Any:
    start = perf_counter()
    f(dataset)
    elapsed = perf_counter() - start
    elapsed /= 60
    elapsed = round(elapsed, 1)

    dsname, ds = dataset
    prep = ds.prepared(load_cached=True)
    prep_train = prep.representative_subsample()[0]
    m = 10
    N = prep.X.shape[0]
    n, p = prep_train.X.shape
    needs_header = False

    if file.exists():
        with open(file, "r") as handle:
            if handle.readline().strip() == "":
                needs_header = True
    else:
        needs_header = True

    with open(file, "a") as handle:
        if needs_header:
            handle.write(
                f"{'dsname':>40}  {'N':>6}  {'N_sub':>6}  {'p':>5}  {'n_iter':>6}  {'minutes':>4}\n"
            )
        handle.write(f"{dsname:>40}  {N:>6d}  {n:>6d}  {p:5d}  {m:>6d}  {elapsed:3.1f}\n")


@fast_ds
def test_associate_select_fast(dataset: tuple[str, TestDataset]) -> None:
    dsname = dataset[0]
    if dsname in ["credit-approval_reproduced"]:
        return
    do_association_select(dataset)


@med_ds
def test_associate_select_med(dataset: tuple[str, TestDataset]) -> None:
    do_association_select(dataset)


@slow_ds
def test_associate_select_slow(dataset: tuple[str, TestDataset]) -> None:
    do_association_select(dataset)


@fast_ds
def test_predict_select_fast(dataset: tuple[str, TestDataset]) -> None:
    dsname = dataset[0]
    if dsname in ["credit-approval_reproduced"]:
        return
    do_predict_select(dataset)


@med_ds
def test_predict_select_med(dataset: tuple[str, TestDataset]) -> None:
    do_predict_select(dataset)


@slow_ds
def test_predict_select_slow(dataset: tuple[str, TestDataset]) -> None:
    do_predict_select(dataset)


@fast_ds
def test_embed_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "lgbm_embed_fast_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_embed_select, file, dataset)


@med_ds
def test_embed_select_med(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "lgbm_embed_med_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_embed_select, file, dataset)


@fast_ds
def test_linear_forward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "linear_forward_select_fast_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_linear_forward_select, file, dataset)


@slow_ds
def test_linear_forward_select_slow(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "linear_forward_select_slow_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_linear_forward_select, file, dataset)


@fast_ds
def test_lgbm_forward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "lgbm_forward_select_fast_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_lgbm_forward_select, file, dataset)


@fast_ds
def test_linear_backward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "linear_backward_select_fast_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_linear_backward_select, file, dataset)


@fast_ds
def test_lgbm_backward_select_fast(
    dataset: tuple[str, TestDataset], capsys: CaptureFixture
) -> None:
    file = RUNTIMES / "lgbm_backward_select_fast_runtimes.txt"
    # with capsys.disabled():
    do_logged(do_lgbm_backward_select, file, dataset)


@all_ds
def test_pseudo_forward_select_fast(dataset: tuple[str, TestDataset]) -> None:
    do_forward_pseudo_select(dataset)


@all_ds
def test_pseudo_forward_redundant_select_fast(dataset: tuple[str, TestDataset]) -> None:
    do_forward_pseudo_select_redundant(dataset)


@all_ds
def test_pseudo_backward_select_fast(dataset: tuple[str, TestDataset]) -> None:
    do_backward_pseudo_select(dataset)


if __name__ == "__main__":
    for dsname, ds in ALL_DATASETS:
        print(dsname)
        # do_forward_select((dsname, ds))
        try:
            # estimate_linear_backward_select((dsname, ds))
            estimate_linear_forward_select((dsname, ds), subsample=False)
            # estimate_lgbm_forward_select((dsname, ds))
            # estimate_lgbm_backward_select((dsname, ds))
            # estimate_knn_forward_select((dsname, ds))
            # estimate_knn_backward_select((dsname, ds))
        except Exception as e:
            print(e)

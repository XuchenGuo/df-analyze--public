from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import json
import sys
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Tuple,
)
from unittest.mock import patch

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from df_analyze.cli.cli import ProgramOptions
from df_analyze.enumerables import DfAnalyzeClassifier, DfAnalyzeRegressor
from df_analyze.hypertune import EvaluationResults
from df_analyze.preprocessing.inspection.inspection import InspectionResults
from df_analyze.saving import ProgramDirs, windows_io_path
from df_analyze.selection.embedded import EmbedSelected
from df_analyze.selection.wrapper import WrapperSelected
from df_analyze.testing.datasets import TestDataset, fast_ds


def test_target_markdown_reports_resolved_final_cv_folds() -> None:
    scores = DataFrame(
        {
            "model": ["dummy"],
            "selection": ["none"],
            "metric": ["acc"],
            "trainset": [0.8],
            "holdout": [0.7],
            "cv_mean": [0.6],
            "final_cv_folds": [3],
        }
    )

    adaptive_report = ProgramDirs()._target_markdown(
        scores, target_name="target", is_classification=True
    )
    assert "## 3-fold performance on holdout set" in adaptive_report

    with np.testing.assert_raises_regex(ValueError, "final_cv_folds"):
        ProgramDirs()._target_markdown(
            scores.drop(columns="final_cv_folds"),
            target_name="target",
            is_classification=True,
        )

    scores["positive_class"] = "case"
    positive_report = ProgramDirs()._target_markdown(
        scores, target_name="target", is_classification=True
    )
    assert "original label `case`" in positive_report


def test_safe_target_names_are_collision_resistant() -> None:
    dirs = ProgramDirs()

    first = dirs._safe_target_name("a/b")
    second = dirs._safe_target_name("a?b")

    assert first != second
    assert dirs._safe_target_name("a/b") == first
    assert first.casefold() != second.casefold()
    assert dirs._safe_target_name("Outcome").casefold() != dirs._safe_target_name(
        "outcome"
    ).casefold()


@fast_ds
def test_random_options(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        ProgramOptions.random(ds)


def _local_test_dataset(root: Path, *, classification: bool) -> TestDataset:
    mode = "classification" if classification else "regression"
    dataset_root = root / mode / "local"
    dataset_root.mkdir(parents=True)
    DataFrame(
        {
            "feature_name": ["feature", "target"],
            "type": ["continuous", "continuous"],
        }
    ).to_csv(dataset_root / "types.csv", index=False)
    target = [0, 1, 0, 1] if classification else [0.1, 0.4, 0.8, 1.2]
    DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": target}).to_parquet(
        dataset_root / "local.parquet", index=False
    )
    return TestDataset(dataset_root)


def test_random_options_falls_back_when_only_tabpfn_was_drawn(
    tmp_path: Path,
) -> None:
    cls_ds = _local_test_dataset(tmp_path / "classification-data", classification=True)
    reg_ds = _local_test_dataset(tmp_path / "regression-data", classification=False)

    with (
        patch.object(
            DfAnalyzeClassifier,
            "random_n",
            return_value=(DfAnalyzeClassifier.TabPFN,),
        ),
        patch.object(
            DfAnalyzeRegressor,
            "random_n",
            return_value=(DfAnalyzeRegressor.TabPFN,),
        ),
    ):
        cls_options = ProgramOptions.random(cls_ds)
        reg_options = ProgramOptions.random(reg_ds)

    assert cls_options.classifiers == (DfAnalyzeClassifier.Dummy,)
    assert reg_options.regressors == (DfAnalyzeRegressor.Dummy,)


@fast_ds
def test_hashing(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    for _ in range(100):
        with TemporaryDirectory() as tempdir:
            outdir = Path(tempdir)
            opts1 = ProgramOptions.random(ds, outdir=outdir)
            opts2 = ProgramOptions.random(ds, outdir=outdir)
            opts11 = deepcopy(opts1)
            opts22 = deepcopy(opts2)
            hsh1 = opts1.hash()
            hsh2 = opts2.hash()
            hsh11 = opts11.hash()
            hsh22 = opts22.hash()
            assert hsh1 != hsh2
            assert hsh1 == hsh11
            assert hsh2 == hsh22


def are_equal(obj1: Any, obj2: Any) -> bool:
    if is_dataclass(obj1) and is_dataclass(obj2):
        if obj1.__dict__.keys() != obj2.__dict__.keys():
            return False

        for key in obj1.__dict__.keys():
            # not sure why, but this appears when you add a mixin to the Enum
            if key == "__objclass__":
                continue
            attr1 = obj1.__dict__[key]
            attr2 = obj2.__dict__[key]
            try:
                eq = are_equal(attr1, attr2)
                if not eq:
                    return False
            except RecursionError:
                raise ValueError(
                    f"Failed to compare objects on key={key} due to infinite recursion. Compared attribute types:\n"
                    f"attr1={type(attr1)}, attr2={type(attr2)}"
                )
        return True
    elif isinstance(obj1, ndarray) and isinstance(obj2, ndarray):
        return (obj1.ravel().round(8) == obj2.ravel().round(8)).all()
    elif isinstance(obj1, (DataFrame, Series)) and isinstance(obj2, (DataFrame, Series)):
        return np.all((obj1.round(8).values == obj2.round(8).values)).item()
    else:
        return obj1 == obj2


@fast_ds
def test_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    try:
        with TemporaryDirectory() as tempdir:
            for _ in range(10):
                outdir = Path(tempdir)
                opts = ProgramOptions.random(ds, outdir=outdir)
                opts.to_json()
                assert outdir.exists()
                assert opts.program_dirs.options is not None
                assert opts.program_dirs.options.exists()
                opts2 = ProgramOptions.from_json(opts.program_dirs.root)  # type: ignore
                for attr in opts.__dict__:
                    attr1 = getattr(opts, attr)
                    attr2 = getattr(opts2, attr)
                    if not are_equal(attr1, attr2):
                        raise ValueError(
                            f"Saved attribute {attr}: {attr2} does not equal initial "
                            f"value of: {attr1}"
                        )
    except Exception as e:
        raise e


@fast_ds
def test_inspection_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    with TemporaryDirectory() as tempdir:
        inspections = ds.inspect(load_cached=True, force=False)
        outdir = Path(tempdir)
        outfile = outdir / "inspections.json"
        inspections.to_json(outfile)
        loaded = InspectionResults.from_json(outfile)
        assert isinstance(loaded, InspectionResults)
        for attr in inspections.__dict__:
            attr1 = getattr(inspections, attr)
            attr2 = getattr(loaded, attr)
            if not are_equal(attr1, attr2):
                raise ValueError(
                    f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
                )


@fast_ds
def test_embed_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    selected = EmbedSelected.random(ds)
    outdir = Path(tempdir.name)
    outfile = outdir / "embed.json"
    outfile.write_text(selected.to_json(), encoding="utf-8")
    loaded = EmbedSelected.from_json(outfile)
    assert isinstance(loaded, EmbedSelected)
    for attr in selected.__dict__:
        attr1 = getattr(selected, attr)
        attr2 = getattr(loaded, attr)
        if not are_equal(attr1, attr2):
            raise ValueError(
                f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
            )


@fast_ds
def test_wrap_json(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    tempdir = TemporaryDirectory()
    selected = WrapperSelected.random(ds)
    outdir = Path(tempdir.name)
    outfile = outdir / "wrap.json"
    outfile.write_text(selected.to_json(), encoding="utf-8")
    loaded = WrapperSelected.from_json(outfile)
    assert isinstance(loaded, WrapperSelected)
    for attr in selected.__dict__:
        attr1 = getattr(selected, attr)
        attr2 = getattr(loaded, attr)
        if not are_equal(attr1, attr2):
            raise ValueError(
                f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
            )


@fast_ds
def test_eval_save(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc", "internet_usage"]:
        return  # defective target
    tempdir = TemporaryDirectory()
    outdir = Path(tempdir.name)
    try:
        for _ in range(10):
            options = ProgramOptions.random(ds, outdir=outdir)
            selected = EvaluationResults.random(ds, options)

            selected.save(root=outdir, fold_idx=None)
            loaded = EvaluationResults.load(root=outdir)

            assert isinstance(loaded, EvaluationResults)
            for attr in selected.__dict__:
                if attr == "results":  # broken jsonpickle again...
                    continue
                attr1 = getattr(selected, attr)
                attr2 = getattr(loaded, attr)
                if not are_equal(attr1, attr2):
                    raise ValueError(
                        f"Saved attribute {attr}: {attr2} does not equal initial value of: {attr1}"
                    )
    except Exception as e:
        raise ValueError(f"Got error saving eval results for data {dsname}:\n{e}")
    finally:
        tempdir.cleanup()


@fast_ds
def test_eval_preds_save(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc", "internet_usage"]:
        return  # defective target
    tempdir = TemporaryDirectory()
    outdir = Path(tempdir.name)
    try:
        for _ in range(10):
            options = ProgramOptions.random(ds, outdir=outdir)
            selected = EvaluationResults.random(ds, options)
            for idx, result in enumerate(selected.results):
                result.per_target_tuning_scores = {
                    "target_a": 0.25 + (idx / 1000)
                }
            preds = [result.to_preds() for result in selected.results]

            selected.save(root=outdir, fold_idx=None)
            loaded_preds = EvaluationResults.load_preds(root=outdir)

            for i, pred in enumerate(preds):
                for key in pred.__dict__.keys():
                    original = pred.__dict__[key]
                    loaded = loaded_preds[i].__dict__[key]
                    if key == "params":
                        if str(original) == str(loaded):
                            continue
                    if not are_equal(original, loaded):
                        raise ValueError(
                            f"Saved values at key: '{key}' are not equal after loading:\n"
                            "Before:\n"
                            f"{original}\n"
                            "Loaded:\n"
                            f"{loaded}"
                        )

                probs_test = pred.probs_test
                probs_test_loaded = loaded_preds[i].__dict__["probs_test"]
                probs_train = pred.probs_train
                probs_train_loaded = loaded_preds[i].__dict__["probs_train"]

                preds_test = pred.preds_test
                preds_test_loaded = loaded_preds[i].__dict__["preds_test"]
                preds_train = pred.preds_train
                preds_train_loaded = loaded_preds[i].__dict__["preds_train"]

                assert preds_test is not None, "Null / missing preds"
                assert preds_train is not None, "Null / missing preds"
                assert preds_test_loaded is not None, "Null / missing preds"
                assert preds_train_loaded is not None, "Null / missing preds"

                if ds.is_classification:
                    assert probs_test is not None, "Null / missing probs"
                    assert probs_train is not None, "Null / missing probs"
                    assert probs_test_loaded is not None, "Null / missing probs"
                    assert probs_train_loaded is not None, "Null / missing probs"

                    assert len(probs_test) == len(preds_test), (
                        "Test probs and preds mismatch"
                    )
                    assert len(probs_test_loaded) == len(preds_test_loaded), (
                        "Loaded test probs and preds mismatch"
                    )

            # Check predictions load as plain json for use outside of df-analyze
            json_str = (outdir / "prediction_results.json").read_text()
            try:
                json.loads(json_str)
            except Exception as e:
                raise ValueError("Could not load predictions as plain json.") from e

    except Exception as e:
        raise ValueError(
            f"Got error saving prediction results for data {dsname}:\n{e}"
        ) from e
    finally:
        tempdir.cleanup()


@fast_ds
def test_descs_save(dataset: Tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc", "internet_usage"]:
        return  # defective target
    tempdir = TemporaryDirectory()
    outdir = Path(tempdir.name)

    try:
        options = ProgramOptions.random(ds, outdir=outdir)
        preds = ds.prepared(load_cached=True)
        df_cont, df_cat, df_targ = preds.describe_features()
        options.program_dirs.save_feature_descriptions(df_cont, df_cat, df_targ)

    except Exception as e:
        raise ValueError(
            f"Got error saving prediction results for data {dsname}:\n{e}"
        ) from e
    finally:
        tempdir.cleanup()


def test_target_feature_descriptions_support_long_paths(tmp_path: Path) -> None:
    base = tmp_path / ("long-output-component-" * 8)
    windows_io_path(base).mkdir(parents=True)
    dirs = ProgramDirs(descriptions=base)
    frame = DataFrame({"value": [1.0]})

    dirs.save_feature_descriptions(
        frame,
        frame,
        frame,
        target_name="scholarship_holder",
    )

    target_dir = base / "scholarship_holder"
    assert windows_io_path(target_dir / "continuous_features.csv").is_file()
    assert windows_io_path(target_dir / "categorical_features.csv").is_file()
    assert windows_io_path(target_dir / "target.csv").is_file()

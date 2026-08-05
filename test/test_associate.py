from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from df_analyze._constants import TEST_RESULTS
from df_analyze.analysis.metrics import auroc as directionless_feature_auroc
from df_analyze.analysis.univariate.associate import AssocResults
from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)

logging.captureWarnings(capture=True)
logger = logging.getLogger("py.warnings")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())


def test_univariate_auroc_is_explicitly_directionless() -> None:
    y = np.asarray([0, 0, 1, 1])
    decreasing_feature = np.asarray([0.9, 0.8, 0.2, 0.1])

    assert directionless_feature_auroc(decreasing_feature, y) == pytest.approx(1.0)


def do_associate(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:  # const targets
        return
    ds.associations(load_cached=False, force=True)


def do_associate_cached(dataset: tuple[str, TestDataset]) -> Optional[AssocResults]:
    dsname, ds = dataset
    if dsname in ["credit-approval_reproduced"]:  # const targets
        return
    assocs = ds.associations(load_cached=True)
    outdir = TEST_RESULTS / dsname
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / "assoc_tables.md"
    assocs.to_markdown(outfile)
    return assocs


@pytest.mark.regen
@fast_ds
def test_associate_fast(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@pytest.mark.regen
@med_ds
def test_associate_med(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@pytest.mark.regen
@slow_ds
def test_associate_slow(dataset: tuple[str, TestDataset]) -> None:
    do_associate(dataset)


@pytest.mark.cached
@fast_ds
def test_associate_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


@pytest.mark.cached
@med_ds
def test_associate_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


@pytest.mark.cached
@slow_ds
def test_associate_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_associate_cached(dataset)


if __name__ == "__main__":
    for dsname, ds in FAST_INSPECTION:
        if dsname in ["credit-approval_reproduced"]:  # const targets
            continue
        try:
            assocs = ds.associations(load_cached=True, force=False)
            if assocs is None:
                continue
            # print("Continuous associations with target: 'target'")
            # print(assocs.conts)
            # print("Categorical associations with target: 'target'")
            # print(assocs.cats)
            if assocs.cats is not None and len(assocs.cats) > 1:
                cols = ["cramer_v", "mut_info", "H"]
                cols = sorted(set(cols).intersection(assocs.cats.columns))

                corrs = assocs.cats.loc[:, cols].corr()
                corrs = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
                corrs = corrs.stack().reset_index()
                corrs.columns = ["metric1", "metric2", "correlation"]
                corrs["correlation"] = corrs["correlation"].abs()
                print(corrs.sort_values(by="correlation", ascending=False))
        except Exception as e:
            raise ValueError(f"Got error for data: {dsname}") from e

import os
from itertools import islice
from typing import Optional

from pandas import Series
from sklearn.preprocessing import KBinsDiscretizer
from torch import Tensor


def batched(iterable, n):
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(tuple(islice(iterator, n))):
        yield batch


def get_n_test_samples(n_samples: Optional[int] = None) -> int:
    ON_CLUSTER = os.environ.get("CC_CLUSTER") is not None
    if n_samples is None:
        N = 256 if ON_CLUSTER else 32
    else:
        N = n_samples
    return N


def get_reg_stratify(y: Series) -> Series:
    yy = y.to_numpy().reshape(-1, 1)
    kb = KBinsDiscretizer(
        n_bins=5,
        encode="ordinal",
        quantile_method="linear",
    )
    strat = kb.fit_transform(yy)
    strat = strat.ravel()
    strat = Series(name=y.name, data=strat)
    return strat


def avg_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

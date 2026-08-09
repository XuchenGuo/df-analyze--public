from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
ROOT2 = Path(__file__).resolve().parent.parent / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(ROOT2))  # isort: skip
# fmt: on

import numpy as np
from pandas import Series

from df_analyze.splitting import OmniKFold


def test_omni_kfold_is_reproducible_with_group_fallback() -> None:
    y = Series(np.tile([0, 1], 40), name="target")
    groups = Series(np.repeat(np.arange(4), 20), name="group")

    def split_once(seed: int):
        splitter = OmniKFold(
            n_splits=5,
            is_classification=True,
            grouped=True,
            shuffle=True,
            seed=seed,
            warn_on_fallback=False,
            allow_group_fallback=True,
        )
        splits, used_fallback = splitter.split(y.to_frame(), y, groups)
        return splitter, splits, used_fallback

    for seed in (0, 42, 2**32 - 2):
        first, first_splits, first_fallback = split_once(seed)
        second, second_splits, second_fallback = split_once(seed)

        assert first_fallback and second_fallback
        assert first.effective_n_splits == second.effective_n_splits == 4
        assert len(first_splits) == len(second_splits) == 4
        for (train_a, test_a), (train_b, test_b) in zip(
            first_splits, second_splits
        ):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
            assert set(groups.iloc[train_a]).isdisjoint(groups.iloc[test_a])

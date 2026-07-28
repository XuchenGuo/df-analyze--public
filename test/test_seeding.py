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
import pandas as pd
from pandas import DataFrame
from pytest import CaptureFixture
from tqdm import tqdm

from df_analyze.splitting import OmniKFold
from df_analyze.testing.datasets import (
    random_grouped_data,
)


def test_omni_kfold(capsys: CaptureFixture) -> None:
    did_error = False
    okf_splits: list[tuple[np.ndarray, np.ndarray]]
    rows: list[DataFrame]  # https://github.com/pandas-dev/pandas-stubs/issues/902

    with capsys.disabled():
        i = 0
        attempts = 0
        rows = []
        N_ITER = 25
        pbar = tqdm(total=N_ITER)
        n_errors = 0
        while i < N_ITER:
            rng = np.random.default_rng()
            seed = rng.integers(0, 2**32 - 1)
            rng = np.random.default_rng(seed=seed)

            n_samp = rng.integers(1000, 2000)
            n_cls = rng.choice([2, 3])
            n_grp = rng.choice([2, 3])
            y, g_rand = random_grouped_data(
                n_cls=n_cls,
                n_grp=n_grp,
                n_samp=n_samp,
                n_min_per_targ_cls=100,
                n_min_per_g=100,
                degenerate=False,
            )
            g_id = g_rand.copy()
            g_id[:] = np.arange(len(g_rand))

            try:
                for g, degen in [
                    (g_rand, "random"),
                    (g_id, "ids"),
                ]:
                    okf = OmniKFold(
                        n_splits=5,
                        is_classification=True,
                        grouped=True,
                        labels=None,
                        shuffle=False,
                        seed=seed,
                        warn_on_fallback=False,
                        allow_group_fallback=True,
                    )
                    okf2 = OmniKFold(
                        n_splits=5,
                        is_classification=True,
                        grouped=True,
                        labels=None,
                        shuffle=False,
                        seed=seed,
                        warn_on_fallback=False,
                        allow_group_fallback=True,
                    )

                    okf_splits, fails = okf.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    okf_splits2, fails2 = okf2.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    for k in range(len(okf_splits)):
                        is_test = True
                        item = 1 if is_test else 0
                        okf_ix_train = okf_splits[k][item]
                        okf_ix_train2 = okf_splits2[k][item]

                        if not np.array_equal(okf_ix_train, okf_ix_train2):
                            did_error = True
                            row = DataFrame(
                                {
                                    "degen": degen,
                                    "seed": seed,
                                    "fallback": any([fails, fails2]),
                                },
                                index=[n_errors],
                            )
                            rows.append(row)
                            i += 1
                            n_errors += 1
                            pbar.update()
                            continue

            except RuntimeError as e:
                if attempts > 50:
                    raise RuntimeError(
                        f"Couldn't generate splittable data for seed: {seed}"
                    ) from e
                attempts += 1
                continue
            attempts = 0
            i += 1
            pbar.update()
        pbar.close()
    if not did_error:
        return

    if len(rows) == 0:
        raise RuntimeError("Couldn't generate any splittable data")

    with capsys.disabled():
        df = pd.concat(rows, axis=0, ignore_index=False)

        pd.options.display.max_rows = N_ITER
        pd.options.display.width = 300
        pd.options.display.max_columns = 20
        print(df)
        print(n_errors)

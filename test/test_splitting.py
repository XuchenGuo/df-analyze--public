from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
ROOT2 = Path(__file__).resolve().parent.parent / "src"  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
sys.path.append(str(ROOT2))  # isort: skip
# fmt: on


import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from pytest import CaptureFixture
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from df_analyze.splitting import (
    ApproximateStratifiedGroupSplit,
    OmniKFold,
    regression_split_label,
    resolve_final_cv_folds,
    validate_multitarget_model_cv_support,
    validate_multitarget_cv_support,
    validate_multitarget_holdout_coverage,
    validate_multitarget_regression_support,
    y_split_label_info,
)
from df_analyze.testing.datasets import (
    TestDataset,
    fast_ds,
    med_ds,
    random_grouped_data,
    slow_ds,
)


def test_multitarget_split_label_reports_fallback() -> None:
    y = DataFrame(
        {
            "target_a": [0] * 40 + [1] * 40,
            "target_b": np.tile([0, 1, 2, 3], 20),
        }
    )

    label, info = y_split_label_info(y, min_count=20, warn_on_fallback=False)

    assert info.initial_targets == ["target_a", "target_b"]
    assert info.used_targets == ["target_b"]
    assert info.dropped_targets == ["target_a"]
    assert info.fallback_used
    assert label.nunique() == 4
    assert "Multi-Target Split Audit" in info.to_markdown()


def test_multitarget_split_label_has_no_delimiter_collisions() -> None:
    y = DataFrame(
        {
            "target_a": ["x__y", "x__y", "x", "x"],
            "target_b": ["z", "z", "y__z", "y__z"],
        }
    )

    label, info = y_split_label_info(y, min_count=2, warn_on_fallback=False)

    assert label.nunique() == 2
    assert label.value_counts().tolist() == [2, 2]
    assert info.initial_n_unique_combinations == 2
    assert info.initial_min_combination_count == 2
    assert info.used_targets == ["target_a", "target_b"]
    assert not info.fallback_used


def test_multitarget_split_fallback_keeps_most_informative_combination() -> None:
    target_a = np.repeat([0, 1], 60)
    target_c = np.tile(np.repeat([0, 1, 2], 20), 2)
    target_b = target_a.copy()
    target_b[[0, 60]] = 1 - target_b[[0, 60]]
    y = DataFrame({"target_a": target_a, "target_b": target_b, "target_c": target_c})

    _, info = y_split_label_info(y, min_count=20, warn_on_fallback=False)

    assert info.dropped_targets == ["target_b"]
    assert info.used_targets == ["target_a", "target_c"]
    assert info.final_min_combination_count == 20


def test_multitarget_split_falls_back_when_every_target_has_a_singleton() -> None:
    y = DataFrame(
        {
            "target_a": [1, *([0] * 79)],
            "target_b": [0, 1, *([0] * 78)],
        }
    )

    label, info = y_split_label_info(y, min_count=20, warn_on_fallback=False)

    assert label.nunique() == 1
    assert any(label.attrs.values())
    assert info.used_targets == []
    assert set(info.dropped_targets) == {"target_a", "target_b"}
    assert "without stratification" in info.reason


def test_multitarget_cv_support_fails_before_model_tuning() -> None:
    y = DataFrame(
        {
            "target_a": [1, *([0] * 39)],
            "target_b": np.tile([0, 1], 20),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"'target_a' level 1 has 1 samples",
    ):
        validate_multitarget_cv_support(y, n_splits=5)


def test_multitarget_cv_support_accepts_feasible_levels() -> None:
    y = DataFrame(
        {
            "target_a": np.tile([0, 1], 20),
            "target_b": np.tile([0, 1, 2, 3], 10),
        }
    )

    validate_multitarget_cv_support(y, n_splits=5)


def test_internal_multitarget_holdout_must_cover_training_levels() -> None:
    train = DataFrame(
        {
            "a": np.tile([0, 1], 20),
            "b": np.tile([0, 1, 2, 0], 10),
        }
    )
    holdout = DataFrame(
        {
            "a": np.tile([0, 1], 10),
            "b": np.tile([0, 1], 10),
        }
    )

    with pytest.raises(ValueError, match="missing level"):
        validate_multitarget_holdout_coverage(train, holdout, external=False)


def test_external_multitarget_holdout_missing_level_warns() -> None:
    train = DataFrame({"a": np.tile([0, 1, 2], 10)})
    holdout = DataFrame({"a": np.tile([0, 1], 10)})

    with pytest.warns(UserWarning, match="supplied externally"):
        validate_multitarget_holdout_coverage(train, holdout, external=True)


def test_multitarget_regression_rejects_constant_partition_target() -> None:
    y = DataFrame({"varying": np.arange(20), "constant": np.zeros(20)})

    with pytest.raises(ValueError, match="'constant' is constant"):
        validate_multitarget_regression_support(y, phase="test partition")


def test_multitarget_model_cv_support_uses_each_declared_fold_design() -> None:
    class ThreeFoldModel:
        shortname = "three-fold"
        tuning_cv_folds = 3

    class FiveFoldModel:
        shortname = "five-fold"
        tuning_cv_folds = 5

    y = DataFrame(
        {
            "target_a": np.repeat([0, 1], 11),
            "target_b": np.tile([0, 1], 11),
        }
    )

    # Eleven examples per level support the five-fold design (minimum 10)
    # but not the three-fold design (minimum 12).
    validate_multitarget_model_cv_support(y, [FiveFoldModel])
    with pytest.raises(
        ValueError,
        match=r"three-fold.*with 3 folds.*at least 12",
    ):
        validate_multitarget_model_cv_support(
            y,
            [FiveFoldModel, ThreeFoldModel],
        )


def test_multitarget_model_cv_support_skips_non_kfold_tuners() -> None:
    class HoldoutTunedModel:
        shortname = "holdout"
        tuning_cv_folds = None

    y = DataFrame(
        {
            "target_a": [0, 1],
            "target_b": [0, 1],
        }
    )

    validate_multitarget_model_cv_support(y, [HoldoutTunedModel])


def test_multitarget_cv_reseeds_to_preserve_every_target_level_per_fold() -> None:
    y = DataFrame(
        {
            "target_a": np.r_[
                np.ones(10, dtype=int),
                np.zeros(90, dtype=int),
            ],
            "target_b": np.repeat(np.arange(5), 20),
        }
    )
    proxy, info = y_split_label_info(y, min_count=20, warn_on_fallback=False)
    assert info.used_targets == ["target_b"]
    assert info.dropped_targets == ["target_a"]

    splitter = OmniKFold(
        n_splits=5,
        is_classification=True,
        seed=42,
        warn_on_fallback=False,
        df_analyze_phase="multi-target fold-support test",
    )
    splits, failed = splitter.split(
        y,
        proxy,
        multitarget_y=y,
    )

    assert not failed
    for train, test in splits:
        for target in y.columns:
            assert y.iloc[train][target].value_counts().min() >= 8
            assert y.iloc[test][target].value_counts().min() >= 2


def test_multitarget_regression_cv_reseeds_to_preserve_target_variation() -> None:
    n = 50
    y = DataFrame(
        {
            "dense": np.arange(n, dtype=float),
            "sparse": np.r_[np.ones(10), np.zeros(n - 10)],
        }
    )
    splitter = OmniKFold(
        n_splits=5,
        is_classification=False,
        seed=42,
        warn_on_fallback=False,
        df_analyze_phase="multi-target regression fold-support test",
    )

    splits, failed = splitter.split(
        y,
        y["dense"],
        multitarget_y=y,
    )

    assert not failed
    for train, validation in splits:
        for target in y.columns:
            assert y.iloc[train][target].nunique() >= 2
            assert y.iloc[validation][target].nunique() >= 2


def test_multitarget_regression_cv_rejects_impossible_target_variation() -> None:
    n = 50
    y = DataFrame(
        {
            "dense": np.arange(n, dtype=float),
            "sparse": np.r_[np.ones(4), np.zeros(n - 4)],
        }
    )
    splitter = OmniKFold(
        n_splits=5,
        is_classification=False,
        seed=42,
        warn_on_fallback=False,
        df_analyze_phase="multi-target regression fold-support test",
    )

    with pytest.raises(
        RuntimeError,
        match="every multi-target regression target varies",
    ):
        splitter.split(y, y["dense"], multitarget_y=y)


def test_grouped_multitarget_cv_rejects_impossible_per_fold_support() -> None:
    y = DataFrame(
        {
            "target_a": np.r_[
                np.ones(10, dtype=int),
                np.zeros(90, dtype=int),
            ],
            "target_b": np.tile([0, 1], 50),
        }
    )
    groups = Series(np.repeat(np.arange(10), 10))
    proxy, _ = y_split_label_info(y, min_count=20, warn_on_fallback=False)
    splitter = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        shuffle=True,
        seed=42,
        warn_on_fallback=False,
        allow_group_fallback=False,
        df_analyze_phase="grouped multi-target fold-support test",
    )

    with pytest.raises(
        RuntimeError,
        match="every level of every multi-target classification target",
    ):
        splitter.split(y, proxy, groups, multitarget_y=y)


def test_final_cv_folds_respect_group_capacity() -> None:
    y = Series(np.arange(20))

    assert resolve_final_cv_folds(y) == 5
    assert resolve_final_cv_folds(y, Series(np.repeat(np.arange(4), 5))) == 4

    with pytest.raises(ValueError, match="at least two groups; found 1"):
        resolve_final_cv_folds(y, Series(np.zeros(len(y), dtype=int)))


def test_multitarget_unstratified_fallback_keeps_groups_disjoint() -> None:
    y = DataFrame(
        {
            "target_a": [1, *([0] * 79)],
            "target_b": [0, 1, *([0] * 78)],
        }
    )
    groups = Series(np.repeat(np.arange(10), 8))
    label, _ = y_split_label_info(y, min_count=20, warn_on_fallback=False)
    splitter = ApproximateStratifiedGroupSplit(
        train_size=0.75,
        is_classification=True,
        grouped=True,
        seed=42,
        warn_on_fallback=False,
        warn_on_large_size_diff=False,
    )

    (train, test), failed = splitter.split(label.to_frame(), label, groups)

    assert not failed
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])


def test_multitarget_regression_split_label_is_scale_invariant() -> None:
    y = DataFrame(
        {
            "small": [0.0, 1.0, 2.0, 3.0],
            "large": [30_000.0, 20_000.0, 10_000.0, 0.0],
        }
    )

    label = regression_split_label(y)
    scaled = regression_split_label(y.assign(large=y["large"] * 1_000_000))

    np.testing.assert_allclose(label, scaled)


def test_grouped_safe_fallback_rejects_impossible_data() -> None:
    g = Series(np.concatenate([np.zeros(50), np.ones(50)]))
    y = Series(np.concatenate([np.ones(45), np.zeros(5), np.zeros(45), np.ones(5)]))
    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
        allow_group_fallback=True,
    )
    with pytest.raises(RuntimeError, match="group-disjoint"):
        kf.split(y.to_frame(), y, g)

    y = Series(np.concatenate([range(5) for _ in range(5)]))
    kf = OmniKFold(
        n_splits=5, is_classification=True, grouped=False, warn_on_fallback=False
    )
    with pytest.raises(RuntimeError, match="Attempted to split target"):
        kf.split(y.to_frame(), y, g)


def test_grouped_split_does_not_fall_back_by_default() -> None:
    g = Series(np.concatenate([np.zeros(50), np.ones(50)]))
    y = Series(np.concatenate([np.ones(45), np.zeros(5), np.zeros(45), np.ones(5)]))
    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
    )

    with pytest.raises(RuntimeError, match="group-disjoint"):
        kf.split(y.to_frame(), y, g)


def test_single_group_is_rejected_without_rowwise_fallback() -> None:
    groups = Series(np.zeros(40, dtype=int))
    target = Series(np.tile([0, 1], 20))
    splitter = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
        allow_group_fallback=True,
    )

    with pytest.raises(RuntimeError, match="group-disjoint"):
        splitter.split(target.to_frame(), target, groups)


def test_grouped_holdout_split_does_not_fall_back_by_default() -> None:
    groups = Series(np.concatenate([np.zeros(50), np.ones(50)]))
    target = Series(np.concatenate([np.ones(45), np.zeros(5), np.zeros(45), np.ones(5)]))
    splitter = ApproximateStratifiedGroupSplit(
        train_size=0.8,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
        warn_on_large_size_diff=False,
    )

    with pytest.raises(RuntimeError, match="group-disjoint"):
        splitter.split(target.to_frame(), target, groups)


@pytest.mark.parametrize("n_groups", [2, 3, 4])
def test_grouped_safe_fallback_reduces_folds_without_group_overlap(
    n_groups: int,
) -> None:
    """
    A requested five-fold split is impossible with four groups. The opt-in
    fallback may reduce the fold count, but it must retain group disjointness.
    """
    g = Series(np.repeat(np.arange(n_groups), 20))
    y = Series(np.tile([0, 1], n_groups * 10))
    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        warn_on_fallback=True,
        allow_group_fallback=True,
    )
    with pytest.warns(
        UserWarning,
        match="Group membership remains disjoint",
    ):
        splits, failed = kf.split(y.to_frame(), y, g)
    assert failed
    assert len(splits) == n_groups
    assert kf.effective_n_splits == n_groups
    assert kf.fallback_strategy == f"StratifiedGroupKFold with {n_groups} folds"
    for train, test in splits:
        assert set(g.iloc[train]).isdisjoint(g.iloc[test])

    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
        allow_group_fallback=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        splits, failed = kf.split(y.to_frame(), y, g)
    assert failed
    assert len(splits) == n_groups
    for train, test in splits:
        assert set(g.iloc[train]).isdisjoint(g.iloc[test])


def test_grouped_holdout_safe_fallback_preserves_groups() -> None:
    groups = Series(np.repeat(np.arange(4), 10))
    target = Series(np.tile([0, 1], 20))
    splitter = ApproximateStratifiedGroupSplit(
        train_size=0.8,
        is_classification=True,
        grouped=True,
        warn_on_fallback=False,
        allow_group_fallback=True,
        warn_on_large_size_diff=False,
    )

    (train, test), failed = splitter.split(target.to_frame(), target, groups)

    assert failed
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])


def test_fail_labeled_error_message() -> None:
    """
    Test that the 'labels' argument gets handled correctly when informing user
    of a splitting failure.
    """
    y = Series(np.concatenate([range(5) for _ in range(5)]))
    labels = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
    }
    kf = OmniKFold(
        n_splits=5,
        is_classification=True,
        grouped=False,
        labels=labels,
        warn_on_fallback=False,
    )
    regex = re.compile(
        r"Attempted to split target.*zero.*one.*two.*three.*four", flags=re.DOTALL
    )
    with pytest.raises(RuntimeError, match=regex):
        kf.split(y.to_frame(), y, g_train=None)


def test_approximate_split(capsys: CaptureFixture) -> None:
    rng = np.random.default_rng()
    rows = []
    i = 0
    attempts = 0
    while i < 500:
        n_samp = int(rng.integers(200, 2000))
        n_cls = int(rng.integers(2, 5))
        n_grp = int(rng.integers(2, 10))
        g, y = random_grouped_data(
            n_cls=n_cls, n_grp=n_grp, n_samp=n_samp, n_min_per_targ_cls=20
        )
        train_size = rng.uniform(0.5, 0.9)
        ss = ApproximateStratifiedGroupSplit(
            train_size=train_size,
            is_classification=True,
            grouped=True,
            warn_on_fallback=False,
            warn_on_large_size_diff=False,
            df_analyze_phase="Initial holdout split",
        )
        try:
            (ix_train, ix_test), group_fail = ss.split(y.to_frame(), y, g)
        except RuntimeError as e:
            if attempts > 50:
                raise RuntimeError("Couldn't generate splittable data") from e
            attempts += 1
            continue
        attempts = 0
        desired = ss.desired_train_size
        achieved = len(ix_train) / len(y)
        row = DataFrame(
            {
                "N": n_samp,
                "c": n_cls,
                "g": n_grp,
                "grp_fail": group_fail,
                "train": train_size,
                "actual": achieved,
                "diff": achieved - desired,
            },
            index=[i],
        )
        rows.append(row)
        i += 1
    df = pd.concat(rows, axis=0, ignore_index=False)
    with capsys.disabled():
        print("")
        with pd.option_context("display.max_rows", 50):
            print(df)
            print("Mean diff:", df["diff"].mean(), "P(grp_fail):", df["grp_fail"].mean())


def test_degenerate_group_splitting(capsys: CaptureFixture) -> None:
    did_error = False
    okf_splits: list[tuple[np.ndarray, np.ndarray]]
    rows: list[DataFrame]  # https://github.com/pandas-dev/pandas-stubs/issues/902

    with capsys.disabled():
        i = 0
        attempts = 0
        rows = []
        N_ITER = 50
        pbar = tqdm(total=N_ITER)
        n_errors = 0
        while i < N_ITER:
            rng = np.random.default_rng()
            seed = rng.integers(0, 2**32 - 1)
            rng = np.random.default_rng(seed=seed)

            n_samp = int(rng.integers(100, 20000))
            n_cls = int(rng.integers(2, 3))
            n_grp = int(rng.integers(2, 10))
            y, g = random_grouped_data(
                n_cls=n_cls, n_grp=n_grp, n_samp=n_samp, n_min_per_targ_cls=20
            )
            g_id = g.copy()
            g_id[:] = np.arange(len(g))

            okf = OmniKFold(
                n_splits=5,
                is_classification=True,
                grouped=True,
                labels=None,
                shuffle=False,
                seed=seed,
                warn_on_fallback=True,
                allow_group_fallback=True,
            )
            okf2 = OmniKFold(
                n_splits=5,
                is_classification=True,
                grouped=True,
                labels=None,
                shuffle=False,
                seed=seed,
                warn_on_fallback=True,
                allow_group_fallback=True,
            )

            assert okf.kf is StratifiedGroupKFold
            sgkf = okf.kf(n_splits=5, shuffle=False)
            skf = StratifiedKFold(n_splits=5, shuffle=False)

            try:
                for g, degen in [(g_id, "ids")]:
                    okf_splits, fails = okf.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    okf_splits2, fails2 = okf2.split(
                        X_train=y.to_frame(), y_train=y, g_train=g
                    )
                    sgkf_splits = [*sgkf.split(X=y.to_frame(), y=y, groups=g)]
                    skf_splits = [*skf.split(X=y.to_frame(), y=y)]
                    for k in range(len(okf_splits)):
                        is_test = True
                        item = 1 if is_test else 0
                        okf_ix_train = okf_splits[k][item]
                        okf_ix_train2 = okf_splits2[k][item]
                        skf_ix_train = skf_splits[k][item]
                        sgkf_ix_train = sgkf_splits[k][item]

                        if not np.array_equal(
                            okf_ix_train, skf_ix_train
                        ) or not np.array_equal(okf_ix_train, okf_ix_train2):
                            did_error = True
                            y_unqs_okf, y_cnts_okf = np.unique(
                                y[okf_ix_train], return_counts=True
                            )
                            y_unqs_skf, y_cnts_skf = np.unique(
                                y[skf_ix_train], return_counts=True
                            )
                            y_unqs_sgkf, y_cnts_sgkf = np.unique(
                                y[sgkf_ix_train], return_counts=True
                            )
                            y_unqs, y_cnts = np.unique(y, return_counts=True)

                            y_rats = y_cnts / y_cnts.sum()
                            okf_rats = y_cnts_okf / y_cnts_okf.sum()
                            skf_rats = y_cnts_skf / y_cnts_skf.sum()
                            sgkf_rats = y_cnts_sgkf / y_cnts_sgkf.sum()

                            info = {
                                "y_train classes                   ": y_rats,
                                "OmniKfold train classes:          ": okf_rats,
                                "StratifiedGroupKfold train classes": sgkf_rats,
                                "StratifiedKfold train classes     ": skf_rats,
                            }
                            full_infos = []
                            for line, ratios in info.items():
                                full_infos.append(f"{line}{ratios.round(4)}")

                            # just make sure different split issues aren't due to counts
                            okf_n = len(okf_ix_train)
                            skf_n = len(skf_ix_train)
                            sgkf_n = len(sgkf_ix_train)
                            n_tr_max = max(okf_n, skf_n, sgkf_n)
                            max_size_diff = max(1, int(np.ceil(n_tr_max * 0.01)))
                            assert abs(okf_n - skf_n) <= max_size_diff
                            assert abs(okf_n - sgkf_n) <= max_size_diff
                            assert abs(sgkf_n - skf_n) <= max_size_diff
                            n_tr_split = n_tr_max

                            okf_skf_overlap = set(okf_ix_train).intersection(skf_ix_train)
                            okf_sgkf_overlap = set(okf_ix_train).intersection(
                                sgkf_ix_train
                            )
                            skf_sgkf_overlap = set(skf_ix_train).intersection(
                                sgkf_ix_train
                            )

                            # full_infos.append(
                            #     f"OmniKFold/StratifiedKFold            ({okf_n}/{skf_n}) n_overlap: {len(okf_skf_overlap)}"
                            # )
                            # full_infos.append(
                            #     f"OmniKFold/StratifiedGroupKFold       ({okf_n}/{sgkf_n}) n_overlap: {len(okf_sgkf_overlap)}"
                            # )
                            # full_infos.append(
                            #     f"StratifiedKFold/StratifiedGroupKFold ({skf_n}/{sgkf_n}) n_overlap: {len(skf_sgkf_overlap)}"
                            # )

                            # full_info = "\n".join(full_infos)
                            row = DataFrame(
                                {
                                    "degen": degen,
                                    "seed": seed,
                                    "fallback": fails,
                                    "n_samp": len(y),
                                    "n_cls": n_cls,
                                    "p_cls_min": np.min(y_rats),
                                    "p_cls_max": np.max(y_rats),
                                    "okf_cls_min": np.min(okf_rats),
                                    "skf_cls_min": np.min(skf_rats),
                                    "sgkf_cls_min": np.min(sgkf_rats),
                                    "okf_cls_max": np.max(okf_rats),
                                    "skf_cls_max": np.max(skf_rats),
                                    "sgkf_cls_max": np.max(sgkf_rats),
                                    "p(O-kf/S-kf)": len(okf_skf_overlap) / n_tr_split,
                                    "p(S-kf/SG-kf)": len(skf_sgkf_overlap) / n_tr_split,
                                    "p(O-kf/SG-kf)": len(okf_sgkf_overlap) / n_tr_split,
                                },
                                index=[n_errors],
                            )
                            rows.append(row)
                            i += 1
                            n_errors += 1
                            pbar.update()
                            continue

                            # raise ValueError(
                            #     f"Splits don't match for fold {k} when grouper is {degen}:\n"
                            #     f"{full_info}"
                            # )
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


def do_prep_split_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname in ["internet_usage", "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc"]:
        with pytest.raises(
            ValueError, match=r".*Target 'target' has undersampled levels.*"
        ):
            prep = ds.prepared(load_cached=True)
            prep.split(train_size=0.6)
        return

    try:
        prep = ds.prepared(load_cached=True)
        prep.split(train_size=0.6)
    except ValueError as e:
        if dsname == "credit-approval_reproduced":
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e


@fast_ds
@pytest.mark.cached
def test_prep_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


@med_ds
@pytest.mark.cached
def test_prep_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


@slow_ds
@pytest.mark.cached
def test_prep_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_prep_split_cached(dataset)


def visual_sanity_check() -> None:
    matplotlib.use("QtAgg")
    clses_grps = []
    for _ in range(20):
        for n_cls in np.random.randint(2, 6, 3):
            for n_grp in np.random.randint(2, 6, 3):
                clses_grps.append((n_cls, n_grp))
    np.random.default_rng().shuffle(clses_grps)

    for n_cls, n_grp in clses_grps:
        y, g = random_grouped_data(
            n_grp=n_grp,
            n_cls=n_cls,
            n_samp=200,
            n_min_per_g=2,
            n_min_per_targ_cls=20,
        )
        fig, axes = plt.subplots(ncols=2)
        axes[0].hist(g, bins=len(g.unique()) * 2, color="black")
        axes[0].set_title(f"Groups={n_grp}")

        axes[1].hist(y, bins=len(y.unique()) * 2, color="black")
        axes[1].set_title(f"Target={n_cls}")
        plt.show()


if __name__ == "__main__":
    visual_sanity_check()

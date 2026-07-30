from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import pickle
from itertools import product
from math import ceil
from typing import Literal, cast
from warnings import catch_warnings, filterwarnings

import jsonpickle
import numpy as np
import pandas as pd
import pytest
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from scipy.special import softmax
from sklearn.model_selection import train_test_split as tt_split
from sklearn.preprocessing import KBinsDiscretizer

from df_analyze._constants import TESTDATA
from df_analyze.analysis.univariate.associate import AssocResults, target_associations
from df_analyze.analysis.univariate.predict.predict import (
    PredResults,
    univariate_predictions,
)
from df_analyze.enumerables import NanHandling, ValidationMethod
from df_analyze.preprocessing.cleaning import (
    clean_regression_target,
    drop_unusable,
    encode_categoricals,
    encode_target,
    handle_continuous_nans,
    normalize,
)
from df_analyze.preprocessing.inspection.inspection import (
    InspectionResults,
    inspect_data,
)
from df_analyze.preprocessing.prepare import PreparedData, prepare_data

CLASSIFICATIONS = TESTDATA / "classification"
REGRESSIONS = TESTDATA / "regression"
ALL = sorted(list(CLASSIFICATIONS.glob("*")) + list(REGRESSIONS.glob("*")))

TEST_CACHE = TESTDATA / "__TEST_CACHE__"
TEST_CACHE.mkdir(exist_ok=True, parents=True)


class TestDataset:
    __test__ = False

    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_classification: bool = root.parent.name == "classification"
        self.datapath = root / f"{root.name}.parquet"
        self.dsname = root.name
        self.types = root / "types.csv"
        df = pd.read_csv(self.types)
        dfc = df.loc[df["type"] != "categorical"]
        dfc = dfc.loc[dfc["feature_name"] != "target"]
        df = df.loc[df["type"] == "categorical"]
        df = df.loc[df["feature_name"] != "target"]
        self.categoricals = df["feature_name"].to_list()
        self.continuous = dfc["feature_name"].to_list()
        self.is_multiclass = False

        df = self.load()
        if self.is_classification:
            num_classes = len(np.unique(df["target"].astype(str)))
            self.is_multiclass = num_classes > 2
        self.shape = df.shape

        self.inspect_cachefile = TEST_CACHE / f"{self.dsname}_inspect.json"
        self.prep_cachefile = TEST_CACHE / f"{self.dsname}_prepare.pickle"

        self.assoc_cachedir = TEST_CACHE / f"{self.dsname}_associate"
        self.preds_cachedir = TEST_CACHE / f"{self.dsname}_preds"
        if not self.assoc_cachedir.exists():
            self.assoc_cachedir.mkdir(exist_ok=True, parents=True)
        if not self.preds_cachedir.exists():
            self.preds_cachedir.mkdir(exist_ok=True, parents=True)

    def inspect(self, load_cached: bool = True, force: bool = False) -> InspectionResults:
        if load_cached and force:
            raise ValueError("Cannot both use and overwrite cache, this is pointless.")

        if load_cached and self.inspect_cachefile.exists() and not force:
            results = jsonpickle.decode(self.inspect_cachefile.read_text())
            if not isinstance(results, InspectionResults):
                raise ValueError("Garbage jsonpickle failed again.")
            return cast(InspectionResults, results)

        df = self.load()
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            df, results = inspect_data(
                df=df,
                target="target",
                grouper=None,
                categoricals=self.categoricals,
                ordinals=[],
                _warn=False,
            )
        if force:
            enc = str(jsonpickle.encode(results, keys=True))
            self.inspect_cachefile.write_text(enc)
            saved = jsonpickle.decode(self.inspect_cachefile.read_text())
            if not isinstance(saved, InspectionResults):
                raise ValueError("Garbage jsonpickle failed again.")
            return results

        if not self.inspect_cachefile.exists():
            enc = str(jsonpickle.encode(results, keys=True))
            self.inspect_cachefile.write_text(enc)
            saved = jsonpickle.decode(self.inspect_cachefile.read_text())
            if not isinstance(saved, InspectionResults):
                raise ValueError("Garbage jsonpickle failed again.")
        return results

    def prepared(self, load_cached: bool = True, force: bool = False) -> PreparedData:
        if load_cached and self.prep_cachefile.exists() and (not force):
            # results = jsonpickle.decode(self.prep_cachefile.read_text())
            try:
                with open(self.prep_cachefile, "rb") as handle:
                    results = pickle.load(handle)
                return cast(PreparedData, results)
            except ModuleNotFoundError:
                pass  # probably the pickle predates the refactor, so we need to rebuild...

        df = self.load()
        inspect = self.inspect(load_cached=not force)
        grouper = None
        prep = prepare_data(
            df=df,
            target="target",
            grouper=grouper,
            results=inspect,
            is_classification=self.is_classification,
            ix_train=None,
            ix_tests=[],
            tests_method=ValidationMethod.List,
        )

        if force:
            # enc = str(jsonpickle.encode(prep))
            # self.prep_cachefile.write_text(enc)
            with open(self.prep_cachefile, "wb") as handle:
                pickle.dump(prep, handle)
            return prep

        if not self.prep_cachefile.exists():
            # enc = str(jsonpickle.encode(prep))
            # self.prep_cachefile.write_text(enc)
            with open(self.prep_cachefile, "wb") as handle:
                pickle.dump(prep, handle)
        return prep

    def associations(self, load_cached: bool = True, force: bool = False) -> AssocResults:
        if load_cached and AssocResults.is_saved(self.assoc_cachedir) and (not force):
            return AssocResults.load(self.assoc_cachedir, self.is_classification)

        prep = self.prepared(load_cached=True)
        assocs = target_associations(prep)

        if force:
            assocs.save_raw(self.assoc_cachedir, None)
            return assocs

        if not AssocResults.is_saved(self.assoc_cachedir):
            assocs.save_raw(self.assoc_cachedir, None)

        return assocs

    def predictions(self, load_cached: bool = True, force: bool = False) -> PredResults:
        if load_cached and PredResults.is_saved(self.preds_cachedir) and (not force):
            return PredResults.load(self.preds_cachedir, self.is_classification)

        prep = self.prepared(load_cached=True)
        preds = univariate_predictions(prep, self.is_classification)

        if force:
            preds.save_raw(self.preds_cachedir, None)
            return preds

        if not preds.is_saved(self.preds_cachedir):
            preds.save_raw(self.preds_cachedir, None)

        return preds

    def load(self) -> DataFrame:
        return pd.read_parquet(self.datapath)

    def train_test_split(
        self, test_size: float = 0.2
    ) -> tuple[DataFrame, DataFrame, Series, Series, int]:
        df = self.load()
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            results = self.inspect(load_cached=True)
            df = drop_unusable(df, results, target="target")
            df, X_cont, nan_ind = handle_continuous_nans(
                df,
                target="target",
                grouper=None,
                results=results,
                nans=NanHandling.Median,
            )
            df = encode_categoricals(
                df,
                target="target",
                grouper=None,
                results=results,
            )[0]
            df = normalize(df, "target")
            df = df.copy(deep=True)

            X = df.drop(columns="target")
            y = df["target"]

            if self.is_classification:
                X, y, labels = encode_target(X, y, ix_train=None, ix_tests=None)[:3]
            else:
                X, y = clean_regression_target(X, y, ix_train=None, ix_tests=None)[:2]

            strat = y
            if not self.is_classification:
                yy = y.to_numpy().reshape(-1, 1)
                strat = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(yy)
                strat = strat.ravel()
        X_tr, X_test, y_tr, y_test = tt_split(X, y, test_size=test_size, stratify=strat)
        num_classes = len(np.unique(y)) if self.is_classification else 1
        return X_tr, X_test, y_tr, y_test, num_classes

    def to_multitest(self, temp_train_path: Path, temp_testpaths: list[Path]) -> None:
        df_base = self.load()
        n_test = len(temp_testpaths)
        # We have to be a bit careful here because subsampling can create test
        # sets with rare classes
        target = df_base["target"].astype(str)
        unqs, unq_ixs, labs, cnts = np.unique(
            target, return_counts=True, return_inverse=True, return_index=True
        )
        freqs = cnts / cnts.sum()
        C = len(unqs)
        # given C classes with frequencies (f_1, ..., f_C), uniform sampling will
        # produce a test set with the same expected frequencies. We want instead
        # a balanced subsample, i.e. the frequencies to be (1/C, ..., 1/C). So
        # letting f_i * a_i = 1 / C, a_i = 1/ (f_i, C_i)
        cls_weights = 1 / (freqs * C)
        weights = target.copy()
        remap = {
            target[unq_ix]: cls_weight for unq_ix, cls_weight in zip(unq_ixs, cls_weights)
        }
        weights = weights.map(lambda x: remap[x])
        weights /= weights.sum()

        df_tests = [
            df_base.sample(n=200, replace=True, weights=weights) for _ in range(n_test)
        ]

        df_base.to_parquet(temp_train_path)
        for df_test, testpath in zip(df_tests, temp_testpaths):
            df_test.to_parquet(testpath)

    @staticmethod
    def from_name(name: str) -> TestDataset:
        if name in TEST_DATASETS:
            return TEST_DATASETS[name]
        raise KeyError(f"Dataset with name: {name} not found in current test datasets.")

    __test__ = False  # https://stackoverflow.com/a/59888230


def fake_data(
    mode: Literal["classify", "regress"],
    N: int = 100,
    C: int = 5,
    noise: float = 1.0,
    num_classes: int = 2,
    rng: Generator | None = None,
) -> tuple[DataFrame, DataFrame, Series, Series]:
    if rng is None:
        rng = np.random.default_rng()
    if C == 0:
        # or concat logic is tedious
        raise ValueError("Must have C > 0")

    X_cont_tr = rng.uniform(0, 1, [N, C])
    X_cont_test = rng.uniform(0, 1, [N, C])

    cat_sizes = rng.integers(2, 20, C)
    cats_tr = [rng.integers(0, c, N) for c in cat_sizes]
    cats_test = [rng.integers(0, c, N) for c in cat_sizes]

    X_cat_tr = np.empty([N, C])
    for i, cat in enumerate(cats_tr):
        X_cat_tr[:, i] = cat

    X_cat_test = np.empty([N, C])
    for i, cat in enumerate(cats_test):
        X_cat_test[:, i] = cat

    df_cat_tr = pd.get_dummies(DataFrame(X_cat_tr))
    df_cat_test = pd.get_dummies(DataFrame(X_cat_test))

    df_cont_tr = DataFrame(X_cont_tr)
    df_cont_test = DataFrame(X_cont_test)

    df_tr = pd.concat([df_cont_tr, df_cat_tr], axis=1)
    df_test = pd.concat([df_cont_test, df_cat_test], axis=1)

    cols = [f"f{i}" for i in range(df_tr.shape[1])]
    df_tr.columns = cols
    df_test.columns = cols

    weights = np.random.uniform(0, 1, 2 * C)  # bias to positive
    y_tr = np.dot(df_tr.values, weights) + np.random.normal(0, noise, N)
    y_test = np.dot(df_test.values, weights) + np.random.normal(0, noise, N)

    if mode == "classify":
        encoder = KBinsDiscretizer(
            n_bins=num_classes,
            encode="ordinal",
            strategy="quantile",
            quantile_method="linear",
        )
        encoder.fit(np.concatenate([y_tr.ravel(), y_test.ravel()]).reshape(-1, 1))
        y_tr = encoder.transform(y_tr.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

    target_tr = Series(np.asarray(y_tr).ravel(), name="target")
    target_test = Series(np.asarray(y_test).ravel(), name="target")
    if mode == "classify":
        if len(target_tr.unique()) == 1:
            raise ValueError("Generated training classification target is constant")
        if len(target_test.unique()) == 1:
            raise ValueError("Generated testing classification target is constant")

    return df_tr, df_test, target_tr, target_test


def random_grouped_data(
    n_grp: int,
    n_cls: int,
    n_samp: int = 200,
    n_min_per_g: int = 2,
    n_min_per_targ_cls: int = 20,
    degenerate: bool = True,
) -> tuple[Series, Series]:
    g = []
    for i in range(n_grp):
        for _ in range(n_min_per_g):
            g.append(i)

    y = []
    for i in range(n_cls):
        for _ in range(n_min_per_targ_cls):
            y.append(i)

    n_grp_remain = n_samp - len(g)
    n_cls_remain = n_samp - len(y)

    rng = np.random.default_rng()
    if degenerate:
        p_cls = rng.standard_exponential(size=n_cls)
        p_grp = rng.standard_exponential(size=n_grp)
    else:
        p_cls = rng.triangular(left=1, mode=3, right=3, size=n_cls)
        p_grp = rng.triangular(left=1, mode=3, right=3, size=n_grp)

    p_cls = p_cls / p_cls.sum()
    p_grp = p_grp / p_grp.sum()

    grp_labels = [*range(n_grp)]
    cls_labels = [*range(n_cls)]
    y = y + rng.choice(cls_labels, size=n_cls_remain, replace=True, p=p_cls).tolist()
    g = g + rng.choice(grp_labels, size=n_grp_remain, replace=True, p=p_grp).tolist()

    rng.shuffle(y)
    rng.shuffle(g)

    y_cnts = np.unique(y, return_counts=True)[1]
    if len(y_cnts) != n_cls or y_cnts.min() < n_min_per_targ_cls:
        raise RuntimeError("Impossible!")

    g_cnts = np.unique(g, return_counts=True)[1]
    if len(g_cnts) != n_grp or g_cnts.min() < n_min_per_g:
        raise RuntimeError("Impossible!")
    return Series(name="y", data=y), Series(name="g", data=g)


def sparse_snplike_data(
    mode: Literal["classify", "regress"] = "classify",
    N: int = 1000,
    n_allele: int = 4,
    n_distractor_snps: int = 50,  # dimensionality of D
    n_predictive_snps: int = 3,  # dimensionality of V'
    n_discrim_snps: int = 2,  # dimensionality of H
    n_predictive_variants: int = 5,  # number of unique variants in V
    modal_allele_frequency: float = 0.8,
    p_target_if_predictive_variant: float = 0.8,
    freq_target: float = 0.2,  # percent of samples where target is 1
    p_nan: float = 0.01,  # percent chance an allele is NaN
    rev: bool = True,  # strongly increases correlation between pred variants and target
    rng: Generator | None = None,
) -> tuple[DataFrame, Series]:
    """Simulate SNP-like data.

    Parameters
    ----------
    mode: Literal["classify", "regress"]
        Whether to generate classification or regression task

    N: int = 500
        Number of samples

    n_snp: int = 100
        Number of features with n_poly classes (and maybe NaN)

    n_allele: int = 4
        Number of alleles (classes) per polymorphism (snp feature)

    n_distractor_snps: int = 50
        Dimensionality of D, i.e. how many distractor SNP features. See notes
        for details on D.

    n_predictive_snps: int = 3
        Dimensionality of V', i.e., how many SNPs are predictive of y=1 with
        probability `p_target_if_predictive_variant`. See notes for definition
        of V'. Larger values of this argument relative to n_discrim_snps should
        lead to more false positive predictions.

    n_discrim_snps: int = 2
        Dimensionality of H, i.e. how many SNPs distinguish predictive variants
        as actually predictive. Larger values of this relative to
        n_predictive_snps

    n_predictive_variants: int = 5
        Number of unique variants in V. Larger values make prediction more
        difficult by making the



    modal_allele_frequency: float = 0.8,

    p_target_if_predictive_variant: float = 0.8,

    freq_target_min: float = 0.2, # percent of samples where target is 1

    p_nan: float = 0.01
        Percent chance an allele is NaN

    freq_target_min: float = 0.2
        Minimum number of samples from the target class of interest in the
        binary classification case.

    rng: Generator | None = None
        RNG for reproducing data during testing.


    Notes
    -----

    # Feature Generation

    If SNP features are generated as following: if m=modal_frequency, and rng
    is a NumPy generator, then each SNP, f, will be generated via:

    ```
    snp_features = []
    for _ in range(n_snp):
        if n_poly > 1:
            n_r = n_allele - 1  # number of non-dominant alleles remaining
            p_r = (1 - m - p_nan)  # probability for other alleles
            p_remains = sorted(softmax(rng._exponential(2, n_r))) * p_r
            ps = [m, p_nan, *p_remains]
        else:
            ps = [m, p_nan]

        f = rng.choice(n_poly + 1, size=N, p=ps, replace=True)
        snp_features.append(f)
    ```

    Most of the time, if m=0.8 and p_nan is small, this will result in the
    second most common allele frequency being about 8-15%.

    # Target Generation

    For classification, the assumptive model will be that a specific combination
    of alleles is predictive of the target class `effect_size` percent of the time.


    Why not just generate much more samples than needed, select the variants
    from the related features as needed until reaching the desired number, and
    then discard un-needed samples to get a final exact n? This seems far more
    straightforward.

    It also makes sense to have a "hidden" discriminative feature that is
    necessary to make a variant predictive. This can be generated exactly,
    i.e. we have our n predictive variants (v is dimensionality of the
    predictive variants):

                            V1 = (i_11, i_12, ..., i_1v)    h_1
                            V2 = (i_21, i_22, ..., i_2v)    h_2
                                         ...                ...
                            Vn = (i_n1, i_n2, ..., i_nv)    h_n

    Randomly sample m = freq_target_min * N of these to get e.g. m samples

               V_i1, V_i2, ..., V_im,  ik in {1, 2, ..., n} for all k

    Now, randomly sample from

            p_r = (1 - freq_target_min - p_nan)
            p_remains = sorted(softmax(rng._exponential(2, N))) * p_r
            ps = [freq_target_min, p_nan, *p_remains]
            h = rng.choice(n_allele + 1, size=N, p=ps, replace=True)

    On expectation,

    They key is: we need to generate *rows* for the variants, and only
    h, the discriminator, is generated as a column. We can stil generate
    the rows by our methods above, and than sampling otherwise. Basically, we
    have blocks


                           <-  V' -> h   y

                     |     |       | 0 | 1 |
                     |     |   V   | 0 | 1 |
                     |     |       | 1 | 0 |
                     |     |_______| 0 | 1 |
                     |     |       | 0 | 0 |
                     |  D  |   V^  | 0 | 0 |
                     |     |       | 1 | 0 |
                     |     |       | 1 | 0 |
                     |     |       | 0 | 0 |
                     |     |       | 1 | 0 |

               variants in V and V^ are mutually exclusive


    - D, the distractors, can be generated column-wise, as above and already
      implemented

    - V and V^ can also be generated in the same manner as X, but then we
      want to look at all the resulting unique rows, and randomly sample m
      of them for V, and N - m of them for V^. Or do we want this, actually?

      I.e. perhaps we want to allow predictive variants to appear in V^ as
      well? No, then this breaks the control over the proportion. But how can
      we ensure V' remains SNP-like? Maybe a greedy approach.

    - h is sampled as rng.binomial(n=1, p=target_freq_min, [N])

    - y = 1 only if: (v in V) AND (h == 1)

    - h can optionally be included in X (replace one column of X with h) to
      act as a distractor

    - h could also optionally be multidimensional with dimensionality H, if
      each h is sampled instead independently as

                      p =  target_freq_min**(1/H)
                      rng.binomial(n=1, p=p, [N])

      in fact, if we want to include h in X, we should do:

          p_r = (1 - freq_target_min - p_nan)
          p_remains = sorted(softmax(rng._exponential(2, N))) * p_r
          ps = [freq_target_min**(1/H), p_nan, *p_remains]
          hs = rng.choice(n_allele + 1, size=[N, H], p=ps, replace=True)


                              V'  <----- h ------>  y
                                  h1 h2 h3  ... hH
                     |     |     | 0  0  0  ...  0| 1 |
                     |     |     | 0  0  0  ...  0| 1 |
                     |     |  V  | 0  0  0  ...  0| 1 |
                     |     |     | 0  1  0  ...  0| 0 |
                     |     |     | 1  0  0  ...  0| 0 |
                     |     |_____| 0  0  1  ...  0| 0 |
                     |     |     | 0  0  0  ...  1| 0 |
                     |  D  |  V^ | 0  1  0  ...  0| 0 |
                     |     |     | 0  0  0  ...  0| 0 |
                     |     |     | 0  0  1  ...  1| 0 |
                     |     |     | 0  0  0  ...  0| 0 |
                     |     |     | 0  0  0  ...  1| 0 |

                     P(h = (0, 0, ..., 0)) = target_freq_min




    Also, suppose V is our predictive set of variants, and they predict the
    target y = 1 with probability p. I.e.

    P( y = 1 | V ) = p

    By law of total probability, we have, if A is y=1

    P(A) = P(A|V) * P(V) + P( A | V^ ) * P(V^)
         =    p   * P(V) + P( A | V^ ) * P(V^)






    """
    m = modal_allele_frequency
    if n_allele <= 1:
        raise ValueError("Got n_allele <=1. Can't have constant SNPs")
    if m < 0.5:
        raise ValueError("Must have modal_frequency >= 0.5")

    if n_predictive_snps < 2:
        raise ValueError(
            "Must have more than one predictive SNP. Data generation can also "
            "fail <10%% of the time when n_predictive_snps == 2, often when "
            "also simultaneously n_predictive_variants >= 4"
        )

    rng = rng or np.random.default_rng()

    def make_features(n: int, label: str) -> DataFrame:
        names = []
        snp_features = []
        for i in range(n):
            n_r = n_allele - 1  # number of non-dominant polymorphisms remaining
            p_r = 1 - m - p_nan  # probability for other alleles
            p_remains = sorted(softmax(rng.exponential(2, n_r)) * p_r)
            ps = np.asarray([m, *p_remains, p_nan])

            f = rng.choice(
                np.asarray([*np.arange(n_allele), np.nan]), size=N, p=ps, replace=True
            )
            snp_features.append(f)
            names.append(f"{label}{i:03d}")
        X = np.stack(snp_features, axis=1)
        X = DataFrame(X, columns=names)
        return X

    def make_V_prime() -> DataFrame:
        names = [f"v{i:03d}" for i in range(n_predictive_snps)]
        N_max = ceil(freq_target * N / p_target_if_predictive_variant)

        def _make_V_prime() -> tuple[DataFrame, ndarray]:
            # maybe the best way to handle this is just by rejection sampling
            v = n_predictive_variants
            variants = []
            all_variants = product([*range(n_allele)], repeat=n_predictive_snps)
            for variant in all_variants:
                variants.append(np.array(variant))
                if len(variants) >= v:
                    break
            del all_variants

            p_vs = np.sort(softmax(rng.exponential(1.5, v)))
            V_preds = rng.choice(variants, size=N_max, p=p_vs)

            m = modal_allele_frequency
            n_r = n_allele - 1  # number of non-dominant polymorphisms remaining
            p_r = 1 - m - p_nan  # probability for other alleles
            # p_remains = sorted(softmax(rng.exponential(2, n_r)) * p_r)
            p_remains = sorted(softmax(rng.exponential(1.3, n_r)) * p_r)
            if rev:
                p_remains = reversed(p_remains)
                ps = [*p_remains, m, p_nan]
            else:
                ps = [m, *p_remains, p_nan]
            # make plenty to ensure we have enough after removing pred variants
            V_large = rng.choice(
                np.asarray([*np.arange(n_allele), np.nan]),
                size=[N * 5, n_predictive_snps],
                p=ps,
                replace=True,
            )

            is_predictives = []
            for variant in variants:
                is_predictive = np.apply_along_axis(
                    lambda x: np.equal(x, variant).all(), arr=V_large, axis=1
                ).reshape(-1, 1)
                is_predictives.append(is_predictive)
            removes = np.any(np.concatenate(is_predictives, axis=1), axis=1)
            V_clean = V_large[~removes]
            V_clean = rng.choice(V_clean, size=N - V_preds.shape[0], replace=False)

            V_prime = np.concatenate([V_preds, V_clean], axis=0)
            V_prime = DataFrame(V_prime, columns=names)
            return V_prime, p_vs

        N_attempts = 50
        for _ in range(N_attempts):
            try:
                V, ps = _make_V_prime()
            except ValueError as e:
                if "larger sample than population" in str(e):
                    continue
                else:
                    raise e

            ccs = V.iloc[:N_max].value_counts().reset_index()
            n_var = ccs.shape[0]
            if n_var == n_predictive_variants:
                return V

        raise ValueError("Couldn't generate predictive features to spec")

    def make_h() -> tuple[DataFrame, Series]:
        H = n_discrim_snps
        names = [f"d{i:02d}" for i in range(H)]
        N_target = freq_target * N
        N_max = ceil(N_target / p_target_if_predictive_variant)

        def _make_h() -> DataFrame:
            H = n_discrim_snps
            n_r = n_allele - 1

            p_h = p_target_if_predictive_variant ** (1 / H)
            p_r = 1 - p_h - p_nan
            p_remains = sorted(softmax(rng.exponential(2, n_r)) * p_r)
            ps = [p_h, *p_remains, p_nan]
            hs = [
                rng.choice(n_allele + 1, size=[N_max], p=ps, replace=True)
                for _ in range(H)
            ]
            hs = np.stack(hs, axis=1)
            h = DataFrame(hs, columns=names)
            return h

        N_attempts = 10
        for _ in range(N_attempts):
            h = _make_h()
            ands = h["d00"] == 0
            for i in range(1, h.shape[1]):
                ands &= h[f"d{i:02d}"] == 0
            n_target = ands.sum()
            if abs(N_target - n_target) / N <= 0.01:
                remain = np.zeros([N - len(h), h.shape[1]], dtype=np.int64)
                remain = DataFrame(remain, columns=h.columns)
                h = pd.concat([h, remain], axis=0, ignore_index=True)
                return h, ands

        raise ValueError("Couldn't generate discriminative features to spec")

    # Make distractors
    D = make_features(n=n_distractor_snps, label="x")
    V = make_V_prime()
    h, ands = make_h()
    y = ands.astype(np.int64)
    y = Series(np.concatenate([y, np.zeros(N - len(y), dtype=np.int64)]), name="target")
    X = pd.concat([D, V, h], axis=1)
    return X, y


AVAILABLE = [
    path
    for path in ALL
    if (path / f"{path.name}.parquet").is_file()
    and (path / "types.csv").is_file()
]
__UNSORTED: list[tuple[str, TestDataset]] = [
    (path.name, TestDataset(path)) for path in AVAILABLE
]

TEST_DATASETS: dict[str, TestDataset] = dict(
    sorted(__UNSORTED, key=lambda pair: pair[1].shape[0])
)
if "credit-approval_reproduced" in TEST_DATASETS:
    TEST_DATASETS.pop("credit-approval_reproduced")  # constant target

INSPECTION_TIMES = {
    "KDD98": 68.49440933300002,
    "KDDCup09_appetency": 29.841349791,
    "KDDCup09_churn": 24.500925083,
    "Traffic_violations": 7.761734875000002,
    "okcupid-stem": 7.070780333000002,
    "kick": 5.860810333000003,
    "nomao": 5.264413917000006,
    "adult": 4.739781624999999,
    "news_popularity": 3.348081999999998,
    "jungle_chess_2pcs_endgame_complete": 3.2254894579999984,
    "OnlineNewsPopularity": 3.165630499999999,
    "mushrooms": 3.1523351669999897,
    "Mercedes_Benz_Greener_Manufacturing": 2.320836,
    "adult-census": 2.0808727919999974,
    "kdd_internet_usage": 1.856913500000001,
    "SpeedDating": 1.7916436250000025,
    "internet_usage": 1.7481538749999999,
    "fps_benchmark": 1.5828882909999997,
    "bank-marketing": 1.5273100420000034,
    "cholesterol": 1.4553109579999997,
    "ames_housing": 1.2510204170000012,
    "community_crime": 1.2,  # guess
    "ipums_la_97-small": 1.0387160410000007,
    "ipums_la_99-small": 0.9914642499999999,
    "ipums_la_98-small": 0.9687000410000017,
    "Insurance": 0.9027092499999974,
    "colleges": 0.8229736249999995,
    "cleveland": 0.7004754589999997,
    "jasmine": 0.6840260829999991,
    "ozone_level": 0.6536169579999989,
    "house_prices_nominal": 0.6510138750000003,
    "BNG(lowbwt)": 0.46148729200000105,
    "health_insurance": 0.4552565830000006,
    "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc": 0.3985209169999999,
    "jungle_chess_2pcs_endgame_rat_elephant": 0.35591020899999926,
    "jungle_chess_2pcs_endgame_rat_lion": 0.35248429099999967,
    "jungle_chess_2pcs_endgame_rat_panther": 0.3501694999999998,
    "hypothyroid": 0.3442913749999992,
    "Midwest_Survey_nominal": 0.2911310829999998,
    "KDD": 0.28601433400000076,
    "telco-customer-churn": 0.28339487499999905,
    "Midwest_Survey": 0.27862320800000084,
    "ada_prior": 0.2538542500000016,
    "Midwest_survey2": 0.24950150000000093,
    "shrutime": 0.22340208299999986,
    "arrhythmia": 0.22146737500000047,
    "Kaggle_bike_sharing_demand_challange": 0.21438554099999863,
    "soybean": 0.16840979200000028,
    "student_dropout": 0.15945937499999907,
    "cylinder-bands": 0.11440770899999997,
    "credit_approval": 0.09032287500000002,
    "vote": 0.09019129200000009,
    "dresses-sales": 0.08566837500000002,
    "credit-approval_reproduced": 0.07974441699999968,
    "colic": 0.07908112500000009,
    "primary-tumor": 0.07623075000000057,
    "analcatdata_marketing": 0.07507525000000026,
    "abalone": 0.07501758300000105,
    "wine_quality": 0.07421587500000015,
    "student_performance_por": 0.07353629099999992,
    "pbcseq": 0.06485062500000005,
    "water-treatment": 0.059749041999999974,
    "heart-c": 0.058186541000000425,
    "solar_flare": 0.05271508300000072,
    "dermatology": 0.0504779580000001,
    "cps_85_wages": 0.04649529100000027,
    "analcatdata_reviewer": 0.045098624999999615,
    "elder": 0.0446324580000006,
    "cmc": 0.041142250000000935,
    "pbc": 0.04081912499999962,
    "forest_fires": 0.03245541599999946,
}

FAST_INSPECTION: list[tuple[str, TestDataset]] = []
MEDIUM_INSPECTION: list[tuple[str, TestDataset]] = []
SLOW_INSPECTION: list[tuple[str, TestDataset]] = []
ALL_DATASETS: list[tuple[str, TestDataset]] = []
for dsname, ds in TEST_DATASETS.items():
    ALL_DATASETS.append((dsname, ds))
    if dsname in INSPECTION_TIMES:
        runtime = INSPECTION_TIMES[dsname]
        if runtime < 1.0:
            FAST_INSPECTION.append((dsname, ds))
        elif runtime < 5.0:
            MEDIUM_INSPECTION.append((dsname, ds))
        else:
            SLOW_INSPECTION.append((dsname, ds))

FAST_INSPECTION = sorted(FAST_INSPECTION, key=lambda d: str(d[0]).lower())
MEDIUM_INSPECTION = sorted(MEDIUM_INSPECTION, key=lambda d: str(d[0]).lower())
SLOW_INSPECTION = sorted(SLOW_INSPECTION, key=lambda d: str(d[0]).lower())
ALL_DATASETS = sorted(ALL_DATASETS, key=lambda d: str(d[0]).lower())
DATASET_LIST = FAST_INSPECTION + MEDIUM_INSPECTION + SLOW_INSPECTION

# Keep optional-data test groups executable when only a subset of the
# repository's developer datasets is installed.
if not SLOW_INSPECTION and DATASET_LIST:
    SLOW_INSPECTION = [
        max(
            DATASET_LIST,
            key=lambda pair: (
                INSPECTION_TIMES.get(pair[0], 0.0),
                pair[1].shape[0] * pair[1].shape[1],
            ),
        )
    ]

FASTEST = sorted(
    DATASET_LIST,
    key=lambda pair: (
        INSPECTION_TIMES.get(pair[0], float("inf")),
        pair[1].shape[0] * pair[1].shape[1],
    ),
)[:3]


# https://stackoverflow.com/a/5409569
def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


all_ds = pytest.mark.parametrize(
    "dataset",
    [*TEST_DATASETS.items()],
    ids=lambda pair: str(pair[0]),
)
turbo_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        FASTEST,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.turbo,
)

fast_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        FAST_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.fast,
)
med_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        MEDIUM_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.med,
)
slow_ds = composed(
    pytest.mark.parametrize(
        "dataset",
        SLOW_INSPECTION,
        ids=lambda pair: str(pair[0]),
    ),
    pytest.mark.slow,
)

if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC

    from df_analyze._constants import TEMPLATES

    out = TEMPLATES / "binary_classification.csv"
    X, _, y, _ = fake_data("classify", N=300, C=25)
    print(cross_val_score(SVC(), X, y))
    print(X)
    df = pd.concat([X, y], axis=1)
    df.to_csv(out, index=False)

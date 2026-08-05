from typing import (
    Union,
)

import numpy as np
from numpy import ndarray
from pandas import Series
from scipy.stats import entropy
from scipy.stats.contingency import association, crosstab
from sklearn.metrics import roc_auc_score


def cramer_v(y1: Union[ndarray, Series], y2: Union[ndarray, Series]) -> float:
    if len(np.unique(y1)) == 1 or len(np.unique(y2)) == 1:
        # can't correlate constants...
        return float("nan")
    ct = crosstab(y1, y2)[1]
    return float(association(observed=ct, method="cramer", correction=False))


def cohens_d(g0: Union[ndarray, Series], g1: Union[ndarray, Series]) -> float:
    m0 = np.nanmean(g0)
    m1 = np.nanmean(g1)
    s0 = np.nanstd(g0, ddof=1)
    s1 = np.nanstd(g1, ddof=1)
    n0 = len(g0) - 1
    n1 = len(g1) - 1
    s_pool = np.sqrt((n0 * s0**2 + n1 * s1**2) / (n0 + n1))
    if s_pool == 0:
        return np.nan
    return float(m1 - m0) / s_pool


def auroc(x: Union[ndarray, Series], y_bin: Union[ndarray, Series]) -> float:
    """Return directionless one-feature discrimination strength.

    This is an association statistic, not a model-performance AUROC. A feature
    that ranks a class perfectly in the decreasing direction is just as
    informative as one that ranks it in the increasing direction.
    """

    raw = float(roc_auc_score(y_bin, x, multi_class="raise"))
    return max(raw, 1.0 - raw)


def relative_entropy(x: Union[ndarray, Series], y: Union[ndarray, Series]) -> float:
    """Returns the relative entropy (KL divergence) of x and y
    i.e. KL(P, Q) for P = x, Q = y

    Notes
    -----
    Probabilities / pdfs for the discrete variables are estimated very naively
    by simply computing the frequencies of each unique value.

    Requires x and y to have same number of classes...
    """

    y_cnts = np.unique(y, return_counts=True)[1]
    x_cnts = np.unique(x, return_counts=True)[1]

    p_y = y_cnts / y_cnts.sum()
    p_x = x_cnts / x_cnts.sum()

    return entropy(p_x, p_y, base=2.0).item()

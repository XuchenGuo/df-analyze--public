# ref: Wilson / Clopperâ€“Pearson confidence intervals: https://www.statsmodels.org/v0.14.0/generated/statsmodels.stats.proportion.proportion_confint.html

from __future__ import annotations

import numpy as np


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)

    phat = k / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) / n) + (z**2) / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))

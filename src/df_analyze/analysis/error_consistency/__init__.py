"""Compare errors from repeated model fits on one shared holdout.

Each configuration produces K x R fitted models. This package compares their
classification errors or regression residuals.
"""

from df_analyze.analysis.error_consistency.classification import (
    compute_classification_ec,
)
from df_analyze.analysis.error_consistency.regression import compute_regression_ec

__all__ = ["compute_classification_ec", "compute_regression_ec"]

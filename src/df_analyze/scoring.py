import warnings
from typing import Any, Callable, Union
from warnings import warn

import numpy as np
from numpy import ndarray
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    multilabel_confusion_matrix,
    r2_score,
    roc_auc_score,
)
from sklearn.utils import assert_all_finite


def consistent_confusion_matrix(
    y_true: Union[Series, ndarray], y_pred: Union[ndarray, Series]
) -> ndarray:
    """This ensures that in the binary case, we are getting entries of the confusion
    matrix that correspond to "1" as the label for the class we are trying to predict
    """
    labs = np.unique(np.concatenate([y_true, y_pred], axis=0))
    if len(labs) == 2:
        labs = [1]
    return multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labs)


def sensitivity(y_true: Union[Series, ndarray], y_pred: Union[ndarray, Series]) -> float:
    mat = consistent_confusion_matrix(y_true=y_true, y_pred=y_pred)
    if mat.shape[0] == 2:  # binary case:
        mat = mat[0].reshape(1, *mat.shape[1:])
    tps = mat[:, 1, 1]
    fns = mat[:, 1, 0]

    ps = tps + fns  # all actual positives
    if np.all(ps == 0):
        return float("nan")
    idx = ps > 0
    sens = tps[idx] / ps[idx]
    return sens.mean()


def specificity(y_true: Union[Series, ndarray], y_pred: Union[ndarray, Series]) -> float:
    mat = consistent_confusion_matrix(y_true=y_true, y_pred=y_pred)
    if mat.shape[0] == 2:  # binary case:
        mat = mat[0].reshape(1, *mat.shape[1:])
    tns = mat[:, 0, 0]
    fps = mat[:, 0, 1]
    ns = tns + fps  # all actual negatives
    if np.all(ns == 0):
        return float("nan")
    idx = ns > 0
    specs = tns[idx] / ns[idx]
    return specs.mean()


def ppv(y_true: Union[Series, ndarray], y_pred: Union[ndarray, Series]) -> float:
    mat = consistent_confusion_matrix(y_true=y_true, y_pred=y_pred)
    if mat.shape[0] == 2:  # binary case:
        mat = mat[0].reshape(1, *mat.shape[1:])
    tps = mat[:, 1, 1]  # shape is (n_classes,)
    fps = mat[:, 0, 1]
    denom = tps + fps
    if np.all(denom == 0):
        return float("nan")
    idx = denom > 0
    value = tps[idx] / denom[idx]
    return value.mean()


def npv(y_true: Union[Series, ndarray], y_pred: Union[ndarray, Series]) -> float:
    mat = consistent_confusion_matrix(y_true=y_true, y_pred=y_pred)
    if mat.shape[0] == 2:  # binary case:
        mat = mat[0].reshape(1, *mat.shape[1:])
    tns = mat[:, 0, 0]
    fns = mat[:, 1, 0]
    denom = tns + fns
    if np.all(denom == 0):
        return float("nan")
    idx = denom > 0
    value = tns[idx] / denom[idx]
    return value.mean()


def silent_scorer(f: Callable) -> Callable:
    def silent(*args, **kwargs) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Scoring failed", category=UserWarning
            )
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                if "Number of classes in y_true not equal" in str(e):
                    return np.nan
                if "Input contains NaN" in str(e):
                    return np.nan

    return silent


def robust_auroc_score(
    y_true: Union[Series, ndarray], y_prob: ndarray, *args, **kwargs
) -> float:
    if not isinstance(y_true, Series):
        y_true = Series(y_true)

    try:
        assert_all_finite(y_true)
        assert_all_finite(y_prob)
    except ValueError:
        warn(
            "Found non-finite values (infinity or NaN) in predicted "
            "probabilities. This will cause errors when computing the "
            "AUROC performance. Attempting to drop NaN or infinite "
            "samples as a workaround."
        )
        idx_keep = ~np.isnan(y_prob).any(axis=1)
        idx_keep = idx_keep & np.isfinite(y_prob).any(axis=1)
        y_true = y_true.loc[idx_keep]  # type: ignore
        y_prob = y_prob[idx_keep]

    try:
        assert_all_finite(y_true)
        assert_all_finite(y_prob)
    except ValueError:
        warn(
            "Could not remove non-finite values from predicted "
            "probabilities. This might happen with a badly tuned model "
            "(e.g. SGDClassifier or SGDRegressor) which might output NaN or "
            "infinity due to internal sigmoids or interactions with a bad "
            "learning rate. Will return AUROC=0.5 in this case."
        )
        return 0.5

    try:
        if y_prob.ndim == 1:
            # y_prob = np.stack([y_prob, 1 - y_prob], axis=1)
            y_prob = y_prob.reshape(-1, 1)
        # if y_prob.shape[1] == 2:
        #     y_prob = y_prob[:, 1].reshape(-1, 1)
        raw = float(roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr"))

    except Exception as e:
        message = str(e)
        undefined_fold_messages = (
            "Number of classes in y_true not equal to the number of columns",
            "Only one class present in y_true",
        )
        if isinstance(e, ValueError) and any(
            fragment in message for fragment in undefined_fold_messages
        ):
            # A resampling fold that omits a class has no well-defined macro
            # AUROC for the full probability matrix. Returning chance (0.5)
            # would turn an undefined statistic into an apparently valid score.
            return float("nan")
        idx = np.random.permutation(len(y_true))[:20]
        yt = y_true.iloc[idx]  # type: ignore
        yp = y_prob[idx]
        warn(
            "Could not compute AUROC for given inputs:\n"
            f"y_true: {type(y_true)} shape={y_true.shape}\n{yt}\n"
            f"y_prob: {type(y_prob)} shape={y_prob.shape}\n{yp}\n"
            f"Details:\n{message}"
        )
        return 0.5

    # AUROC values below chance are meaningful: they indicate that the model's
    # ranking is systematically reversed. Reflecting them around 0.5 turns a
    # completely wrong ranking (AUROC=0) into an apparently perfect one.
    return raw


accuracy_scorer = make_scorer(accuracy_score)
auc_scorer = make_scorer(
    # silent_scorer(roc_auc_score),
    silent_scorer(robust_auroc_score),
    response_method="predict_proba",
    # multi_class="ovr",
    # average="macro",
)
sensitivity_scorer = make_scorer(sensitivity)
specificity_scorer = make_scorer(specificity)
f1_scorer = make_scorer(f1_score, average="macro")
bal_acc_scorer = make_scorer(balanced_accuracy_score)

mae_scorer = make_scorer(mean_absolute_error)
mse_scorer = make_scorer(mean_squared_error)
mdae_scorer = make_scorer(median_absolute_error)
mape_scorer = make_scorer(mean_absolute_percentage_error)
r2_scorer = make_scorer(r2_score)
expl_var_scorer = make_scorer(explained_variance_score)

CLASSIFIER_TEST_SCORERS = dict(
    acc=accuracy_scorer,
    auroc=auc_scorer,
    sens=sensitivity_scorer,
    spec=specificity_scorer,
    f1=f1_scorer,
    bal_acc=bal_acc_scorer,
)
REGRESSION_TEST_SCORERS = {
    "mae": mae_scorer,
    "msqe": mse_scorer,
    "mdae": mdae_scorer,
    # "MAPE": mape_scorer,
    "r2": r2_scorer,
    "var-exp": expl_var_scorer,
}

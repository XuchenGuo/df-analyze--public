from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy import ndarray
from optuna.study import Study
from optuna.study import StudyDirection as Direction
from pytest import CaptureFixture

from df_analyze.enumerables import ClassifierScorer, RegressorScorer, Scorer
from df_analyze.models.base import DfAnalyzeModel
from df_analyze.models.knn import KNNClassifier, KNNRegressor
from df_analyze.models.lgbm import (
    LightGBMClassifier,
    LightGBMRegressor,
    LightGBMRFClassifier,
    LightGBMRFRegressor,
)
from df_analyze.models.linear import (
    ElasticNetRegressor,
    LRClassifier,
    SGDClassifier,
    SGDRegressor,
)
from df_analyze.models.svm import SVMClassifier, SVMRegressor
from df_analyze.scoring import robust_auroc_score
from df_analyze.testing.datasets import fake_data


def check_optuna_tune(
    model: DfAnalyzeModel, metric: Scorer
) -> tuple[float, Optional[ndarray]]:
    mode = "classify" if model.is_classifier else "regress"
    X_tr, X_test, y_tr, y_test = fake_data(mode)
    is_cls = model.is_classifier
    metric = ClassifierScorer.default() if is_cls else RegressorScorer.default()  # type: ignore
    study: Study = model.htune_optuna(
        X_train=X_tr,
        y_train=y_tr,
        g_train=None,
        metric=metric,  # type: ignore
        n_trials=8,
        n_jobs=4,
    )
    higher_better = metric.higher_is_better()
    direction = Direction.MAXIMIZE if higher_better else Direction.MINIMIZE
    assert study.direction == direction
    summaries = study.trials_dataframe()
    summaries = summaries.loc[summaries["state"] == "COMPLETE"]
    score = np.round(study.best_value, 8)
    scores = np.round(summaries["value"].values, 8)
    if higher_better:
        if not np.all(score >= scores):
            raise ValueError(
                f"score: {score} is less than one of: {scores.tolist()}.\n"
                f"Trials:\n{summaries}"
            )
    else:
        if not np.all(score <= scores):
            raise ValueError(
                f"score: {score} is greater than one of: {scores.tolist()}.\n"
                f"Trials:\n{summaries}"
            )

    overrides = study.best_params
    model.refit_tuned(X_tr, y_tr, tuned_args=overrides)
    score = model.tuned_scores(X_test, y_test)
    if model.is_classifier:
        probs = model.predict_proba(X_test)
        return score, probs
    return score, None


CLS_SCORERS = [*ClassifierScorer]
REG_SCORERS = [*RegressorScorer]


@pytest.mark.fast
@pytest.mark.parametrize(
    ("metric", "y_pred"),
    [
        (ClassifierScorer.PPV, np.zeros(4, dtype=int)),
        (ClassifierScorer.NPV, np.ones(4, dtype=int)),
    ],
)
def test_undefined_predictive_value_is_zero_only_for_tuning(
    metric: ClassifierScorer, y_pred: ndarray
) -> None:
    y_true = np.asarray([0, 1, 0, 1])

    assert metric.tuning_score(y_true, y_pred) == 0.0
    assert np.isnan(ClassifierScorer.get_scores(y_true, y_pred, None)[metric.value])


@pytest.mark.fast
def test_multiclass_auroc_is_nan_when_a_fold_omits_a_class() -> None:
    y_true = np.asarray([0, 1, 0, 1])
    y_prob = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        score = robust_auroc_score(y_true, y_prob)

    assert np.isnan(score)
    assert len(recorded) == 0


@pytest.mark.fast
def test_binary_auroc_does_not_reflect_below_chance_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    reversed_prob = np.asarray([[0.9], [0.8], [0.2], [0.1]])

    assert robust_auroc_score(y_true, reversed_prob) == pytest.approx(0.0)


@pytest.mark.fast
class TestLinear:
    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_lin_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = LRClassifier()
            check_optuna_tune(model, metric=metric)
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_lin_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = ElasticNetRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)


@pytest.mark.fast
class TestSGDLinear:
    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_sgd_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        logging.captureWarnings(capture=True)
        logger = logging.getLogger("py.warnings")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
        try:
            model = SGDClassifier()
            check_optuna_tune(model, metric=metric)
        except Exception as e:
            raise e
        finally:
            logging.captureWarnings(capture=False)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_sgd_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = SGDRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)


@pytest.mark.fast
class TestKNN:
    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_knn_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = KNNClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_knn_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = KNNRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)


@pytest.mark.fast
class TestSVM:
    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_svm_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = SVMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_svm_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = SVMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)


@pytest.mark.fast
class TestLightGBM:
    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_lgbm_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = LightGBMClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_lgbm_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = LightGBMRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)

    @pytest.mark.parametrize("metric", CLS_SCORERS)
    def test_lgbm_rf_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = LightGBMRFClassifier()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)

    @pytest.mark.parametrize("metric", REG_SCORERS)
    def test_lgbm_rf_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
        model = LightGBMRFRegressor()
        # with capsys.disabled():
        check_optuna_tune(model, metric=metric)


# TODO: Figure out what is causing segmentation fault...
# @pytest.mark.fast
# class TestMLP:
#     @pytest.mark.parametrize("metric", CLS_SCORERS)
#     def test_mlp_cls_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
#         model = MLPEstimator(num_classes=2)
#         # with capsys.disabled():
#         check_optuna_tune(model, metric=metric)

#     @pytest.mark.parametrize("metric", REG_SCORERS)
#     def test_mlp_reg_tune(self, metric: Scorer, capsys: CaptureFixture) -> None:
#         model = MLPEstimator(num_classes=1)
#         # with capsys.disabled():
#         check_optuna_tune(model, metric=metric)

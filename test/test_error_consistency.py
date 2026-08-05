from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import ExtraTreesClassifier

from df_analyze.analysis.adaptive_error.base_models_runner import (
    _init_model_output_dirs,
)
from df_analyze.analysis.adaptive_error.confidence_metrics import (
    tree_leaf_support_conf,
    tree_vote_agreement_conf,
)
from df_analyze.analysis.error_consistency.backend import (
    ECBackendDecision,
    resolve_ec_backend,
)
from df_analyze.analysis.error_consistency.checkpoint import configuration_fingerprint
from df_analyze.analysis.error_consistency.classification import (
    _pair_counts,
    compute_classification_ec,
)
from df_analyze.analysis.error_consistency.containers import finite_summary
from df_analyze.analysis.error_consistency.diagnostics import (
    classification_sample_diagnostics,
    compute_model_ec_ranking,
    regression_sample_diagnostics,
)
from df_analyze.analysis.error_consistency.profiles import (
    REGRESSION_PAPER_METHODS,
)
from df_analyze.analysis.error_consistency.regression import (
    _torch_pairwise,
    compute_regression_ec,
    normalize_regression_method,
    regression_pairwise_consistency,
)
from df_analyze.analysis.error_consistency.risk_stability import (
    risk_stability_summary,
    write_risk_stability_report,
)
from df_analyze.analysis.error_consistency.runner import (
    _init_model,
    _seed_trial_model,
    _trial_model_seed,
    detail_output_dir,
    run_error_consistency_analysis,
)
from df_analyze.analysis.error_consistency.writer import _write_detail_plots
from df_analyze.cli.cli import ArgumentError, get_options
from df_analyze.enumerables import ClassifierScorer, RegressorScorer
from df_analyze.hypertune import EvaluationResults, HtuneResult
from df_analyze.models.dummy import DummyClassifier, DummyRegressor
from df_analyze.models.mlp import MLPEstimator
from df_analyze.multitarget import _eval_results_for_target
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.runtime.hardware import (
    CudaConfigurationError,
    DeviceIntent,
    HardwareCapabilities,
    RuntimePolicy,
)
from df_analyze.saving import windows_io_path
from df_analyze.splitting import OmniKFold


@pytest.mark.fast
def test_published_regression_metrics_and_zero_conventions() -> None:
    ratio = regression_pairwise_consistency(
        np.asarray([0.0, 0.002, -2.0]),
        np.asarray([0.0, 0.003, 4.0]),
        "ratio",
    )
    ratio_diff = regression_pairwise_consistency(
        np.asarray([0.0, 0.002, -2.0]),
        np.asarray([0.0, 0.003, 4.0]),
        "ratio_diff",
    )
    ratio_sign = regression_pairwise_consistency(
        np.asarray([0.0, 0.002, -2.0]),
        np.asarray([0.0, 0.003, 4.0]),
        "ratio_sign",
    )
    ratio_diff_sign = regression_pairwise_consistency(
        np.asarray([0.0, 0.002, -2.0]),
        np.asarray([0.0, 0.003, 4.0]),
        "ratio_diff_sign",
    )

    assert ratio.tolist() == pytest.approx([1.0, 2.0 / 3.0, 0.5])
    assert ratio_diff.tolist() == pytest.approx([0.0, 0.2, 1.0 / 3.0])
    assert ratio_sign.tolist() == pytest.approx([1.0, 2.0 / 3.0, -0.5])
    assert ratio_diff_sign.tolist() == pytest.approx([0.0, 0.2, -1.0 / 3.0])


@pytest.mark.fast
def test_ratio_diff_sign_magnitude_preferred_alias_is_backward_compatible() -> None:
    assert normalize_regression_method("ratio_diff_sign_magnitude") == (
        "ratio_diff_sign_magnitude"
    )
    assert normalize_regression_method("ratio-diff-sign-magnitude") == (
        "ratio_diff_sign_magnitude"
    )
    assert normalize_regression_method("ratio_diff_sign_reference") == (
        "ratio_diff_sign_reference"
    )

    residuals = np.asarray([[1.0, 1.0], [2.0, -2.0]])
    preferred = compute_regression_ec(
        residuals,
        methods=["ratio_diff_sign_magnitude"],
    )[0]
    compatibility = compute_regression_ec(
        residuals,
        methods=["ratio_diff_sign"],
    )[0]
    reference = compute_regression_ec(
        residuals,
        methods=["ratio_diff_sign_reference"],
    )[0]

    assert preferred.info.name == "ratio_diff_sign_magnitude"
    assert compatibility.info.name == "ratio_diff_sign"
    assert preferred.summary()["ec_mean"] == compatibility.summary()["ec_mean"]
    assert preferred.summary()["preferred_method_name"] == ("ratio_diff_sign_magnitude")
    assert preferred.summary()["compatibility_method_name"] == "ratio_diff_sign"
    assert compatibility.summary()["legacy_ambiguous_method_name"] is True
    assert reference.summary()["ec_mean"] == pytest.approx(0.0)
    assert reference.summary()["ec_signed_mean"] == pytest.approx(0.0)
    assert reference.summary()["primary_aggregation"] == ("mean_signed_ratio_difference")
    assert reference.summary()["ranking_supported"] is False
    assert reference.summary()["ranking_rule"] == "not_ranked"


@pytest.mark.fast
def test_regression_ratio_epsilon_preserves_equal_residuals() -> None:
    first = np.asarray([1e-12, 2.0, -3.0])
    second = first.copy()

    ratio = regression_pairwise_consistency(first, second, "ratio", epsilon=1e-6)
    ratio_sign = regression_pairwise_consistency(
        first, second, "ratio_sign", epsilon=1e-6
    )

    assert ratio.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert ratio_sign.tolist() == pytest.approx([1.0, 1.0, 1.0])


@pytest.mark.fast
def test_regression_ratio_epsilon_preserves_zero_endpoints() -> None:
    first = np.asarray([0.0, 0.0, 2.0, -3.0])
    second = np.asarray([0.0, 2.0, 0.0, -3.0])

    ratio = regression_pairwise_consistency(first, second, "ratio", epsilon=1e-6)
    ratio_sign = regression_pairwise_consistency(
        first, second, "ratio_sign", epsilon=1e-6
    )

    assert ratio.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert ratio_sign.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


@pytest.mark.fast
@pytest.mark.parametrize("method", ["ratio", "ratio_sign"])
def test_regression_ratio_epsilon_stabilizes_small_nonzero_residuals(
    method: str,
) -> None:
    first = np.asarray([1e-12, -1e-12, 1.0])
    second = np.asarray([2e-12, -2e-12, 2.0])
    epsilon = 1e-6

    observed = regression_pairwise_consistency(first, second, method, epsilon=epsilon)
    expected_magnitude = (np.abs(first) + epsilon) / (np.abs(second) + epsilon)
    expected = (
        expected_magnitude
        if method == "ratio"
        else np.sign(first * second) * expected_magnitude
    )

    assert observed.tolist() == pytest.approx(expected.tolist())
    assert observed[0] > 0.99
    assert observed[2] == pytest.approx((1.0 + epsilon) / (2.0 + epsilon))


@pytest.mark.fast
@pytest.mark.parametrize("method", ["ratio", "ratio_sign"])
def test_regression_ratio_epsilon_cpu_torch_parity(method: str) -> None:
    first = np.asarray([0.0, 0.0, 1e-12, -1e-12, 1.0, -1.0])
    second = np.asarray([0.0, 2.0, 2e-12, -2e-12, 2.0, 2.0])
    epsilon = 1e-6

    expected = regression_pairwise_consistency(first, second, method, epsilon=epsilon)
    observed = _torch_pairwise(
        torch.as_tensor(first, dtype=torch.float64),
        torch.as_tensor(second, dtype=torch.float64),
        method,
        epsilon,
    )

    assert observed.cpu().numpy().tolist() == pytest.approx(expected.tolist())


@pytest.mark.fast
def test_all_regression_methods_numpy_torch_parity() -> None:
    first = np.asarray([0.0, 2.0, -3.0, 4.0, -1.0])
    second = np.asarray([0.0, 1.0, -6.0, -2.0, 3.0])
    first_t = torch.as_tensor(first, dtype=torch.float64).reshape(1, -1)
    second_t = torch.as_tensor(second, dtype=torch.float64).reshape(1, -1)
    for method in REGRESSION_PAPER_METHODS:
        expected = regression_pairwise_consistency(first, second, method)
        observed = _torch_pairwise(first_t, second_t, method, epsilon=0.0)
        np.testing.assert_allclose(
            observed.detach().cpu().numpy().ravel(),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.fast
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="physical CUDA parity requires a CUDA device"
)
def test_physical_cuda_pairwise_parity() -> None:
    errors = np.asarray(
        [
            [False, True, False, True, False],
            [False, True, True, False, False],
            [True, False, True, False, True],
        ],
        dtype=bool,
    )
    pairs = [(0, 1), (0, 2), (1, 2)]
    expected_intersections, expected_unions = _pair_counts(errors, pairs, False)
    observed_intersections, observed_unions = _pair_counts(errors, pairs, True)
    np.testing.assert_array_equal(observed_intersections, expected_intersections)
    np.testing.assert_array_equal(observed_unions, expected_unions)

    first = np.asarray([0.0, 2.0, -3.0, 4.0, -1.0])
    second = np.asarray([0.0, 1.0, -6.0, -2.0, 3.0])
    first_t = torch.as_tensor(first, dtype=torch.float64, device="cuda").reshape(1, -1)
    second_t = torch.as_tensor(second, dtype=torch.float64, device="cuda").reshape(1, -1)
    for method in REGRESSION_PAPER_METHODS:
        expected = regression_pairwise_consistency(first, second, method)
        observed = _torch_pairwise(first_t, second_t, method, epsilon=0.0)
        np.testing.assert_allclose(
            observed.detach().cpu().numpy().ravel(),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.fast
def test_ec_model_seeds_are_reproducible_and_mode_specific() -> None:
    varied = [
        _trial_model_seed(13, repetition, fold, "vary")
        for repetition in range(2)
        for fold in range(3)
    ]
    assert len(set(varied)) == 6
    assert varied == [
        _trial_model_seed(13, repetition, fold, "vary")
        for repetition in range(2)
        for fold in range(3)
    ]
    assert {
        _trial_model_seed(13, repetition, fold, "fixed")
        for repetition in range(2)
        for fold in range(3)
    } == {13}


@pytest.mark.fast
def test_ec_model_seeding_uses_only_one_estimator_seed_alias() -> None:
    class AliasEstimator:
        def __init__(
            self,
            random_seed: int | None = None,
            random_state: int | None = None,
        ) -> None:
            self.random_seed = random_seed
            self.random_state = random_state

    class AliasModel:
        shortname = "alias-model"

        def __init__(self) -> None:
            self.fixed_args: dict[str, object] = {}
            self.default_args = {"random_seed": 13}
            self.model_args: dict[str, object] = {}

        @staticmethod
        def model_cls_args(full_args):
            return AliasEstimator, full_args

    args, seeded = _seed_trial_model(AliasModel(), {}, 41)

    assert args == {"random_seed": 41}
    assert seeded == "random_seed"


@pytest.mark.fast
@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_ec_detail_plot_supports_a_long_windows_output_path(tmp_path) -> None:
    padding = max(1, 210 - len(str(tmp_path.resolve())))
    detail = tmp_path / ("x" * padding)
    pairwise = pd.DataFrame(
        {
            "pair_mean": [0.25, 0.75],
            "ec_method": ["classification_iou", "classification_iou"],
        }
    )

    _write_detail_plots(detail, pairwise)

    legacy_plot = detail / "plots" / "pairwise_ec_distribution_classification_iou.png"
    plot = detail / "plots" / "pairwise_ec_classification_iou.png"
    assert len(str(legacy_plot.resolve())) > 260
    assert len(str(plot.resolve())) < 260
    assert plot.exists()


@pytest.mark.fast
@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_adaptive_error_model_outputs_support_a_long_windows_path(tmp_path) -> None:
    padding = max(1, 220 - len(str(tmp_path.resolve())))
    model_dir = (
        tmp_path / ("x" * padding) / "results" / "adaptive_error" / "models" / "et"
    )
    output_dirs = _init_model_output_dirs(model_dir)
    metadata = model_dir / "metadata" / "confidence_metric_selection.json"

    (output_dirs.meta / metadata.name).write_text("{}", encoding="utf-8")

    assert len(str(metadata.resolve())) > 260
    assert windows_io_path(metadata).read_text(encoding="utf-8") == "{}"


@pytest.mark.fast
def test_tree_vote_agreement_avoids_sklearn_feature_name_warnings() -> None:
    X = pd.DataFrame(
        {
            "first": [0.0, 0.1, 0.9, 1.0],
            "second": [1.0, 0.9, 0.1, 0.0],
        }
    )
    y = np.asarray([0, 0, 1, 1])
    estimator = ExtraTreesClassifier(n_estimators=5, random_state=0).fit(X, y)
    y_pred = estimator.predict(X)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        agreement = tree_vote_agreement_conf(estimator, X, y_pred)
        leaf_support = tree_leaf_support_conf(estimator, X, n_train=len(X))

    assert agreement is not None
    assert agreement.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert leaf_support is not None
    assert leaf_support.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


@pytest.mark.fast
def test_ec_kan_seed_is_forwarded_to_the_skorch_module() -> None:
    model = SimpleNamespace(
        shortname="kan", fixed_args={}, default_args={}, model_args={}
    )

    varied_seed = _trial_model_seed(13, 1, 1, "vary")
    varied_args, varied_parameters = _seed_trial_model(model, {}, varied_seed)
    fixed_args, fixed_parameters = _seed_trial_model(model, {}, 13)

    assert varied_args["module__seed"] == varied_seed
    assert fixed_args["module__seed"] == 13
    assert varied_parameters == fixed_parameters == "module__seed"


@pytest.mark.fast
def test_intersection_union_regression_definitions() -> None:
    first = np.asarray([2.0, -2.0, 1.0])
    second = np.asarray([4.0, -1.0, -3.0])

    assert regression_pairwise_consistency(
        first, second, "intersection_union_sample"
    ).tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert regression_pairwise_consistency(
        first, second, "intersection_union_all"
    ).item() == pytest.approx(0.3)
    assert regression_pairwise_consistency(
        first, second, "intersection_union_distance"
    ).tolist() == pytest.approx([2.0, 1.0, 4.0])


@pytest.mark.fast
def test_regression_computation_returns_all_metrics_and_sample_sd() -> None:
    residuals = np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    computations = compute_regression_ec(residuals, row_ids=[10, 20, 30])

    assert len(computations) == 7
    assert all(computation.matrix.shape == (3, 3) for computation in computations)
    assert finite_summary(np.asarray([1.0, 2.0, 3.0]))["ec_sd"] == pytest.approx(1.0)
    assert {
        computation.info.name
        for computation in computations
        if computation.info.optimal_value == 0.0
    } == {
        "ratio_diff",
        "ratio_diff_sign_magnitude",
        "intersection_union_distance",
    }
    for computation in computations:
        summary = computation.summary()
        assert summary["n_model_pairs"] == 3
        assert summary["scientific_status"] == "experimental_descriptive_diagnostic"
        assert summary["inferential_status"] == (
            "descriptive_only_not_confidence_interval"
        )
        assert summary["reference_url"] is None
        assert summary["paper_equation"] is None
        expected = 3 if computation.info.name == "intersection_union_all" else 9
        assert summary["n_pair_sample_values"] == expected
        if computation.samplewise is not None:
            assert computation.samplewise["row_id"].tolist() == [10, 20, 30]
    ratio_diff_sign = next(
        computation
        for computation in computations
        if computation.info.name == "ratio_diff_sign_magnitude"
    )
    assert ratio_diff_sign.summary()["optimization_direction"] == "minimize"


@pytest.mark.fast
def test_regression_streaming_summaries_match_explicit_values() -> None:
    residuals = np.asarray([[0.0, 1.0, -2.0], [0.0, 2.0, 4.0], [1.0, -2.0, 3.0]])
    for computation in compute_regression_ec(residuals):
        method = computation.info.name
        explicit = np.concatenate(
            [
                regression_pairwise_consistency(residuals[i], residuals[j], method)
                for i, j in [(0, 1), (0, 2), (1, 2)]
            ]
        )
        summary = computation.summary()
        primary = explicit
        assert summary["ec_mean"] == pytest.approx(primary.mean())
        assert summary["ec_sd"] == pytest.approx(primary.std(ddof=1))
        assert summary["ec_min"] == pytest.approx(primary.min())
        assert summary["ec_max"] == pytest.approx(primary.max())
        assert summary["n_comparisons"] == explicit.size
        if method == "ratio_diff_sign_magnitude":
            signed = np.concatenate(
                [
                    regression_pairwise_consistency(
                        residuals[i],
                        residuals[j],
                        "ratio_diff_sign_reference",
                    )
                    for i, j in [(0, 1), (0, 2), (1, 2)]
                ]
            )
            assert summary["ec_signed_mean"] == pytest.approx(signed.mean())
            assert summary["primary_aggregation"] == ("mean_unsigned_ratio_difference")
        if method == "intersection_union_all":
            assert np.isnan(summary["EC_scalar_sd"])


@pytest.mark.fast
def test_summary_detail_preserves_metric_values_without_detail_frames() -> None:
    residuals = np.asarray(
        [[0.0, 1.0, -2.0], [0.0, 2.0, 4.0], [1.0, -2.0, 3.0]]
    )
    full = compute_regression_ec(residuals, output_detail="full")
    compact = compute_regression_ec(residuals, output_detail="summary")

    for expected, observed in zip(full, compact):
        expected_summary = expected.summary()
        observed_summary = observed.summary()
        for field in (
            "ec_mean",
            "ec_sd",
            "ec_min",
            "ec_max",
            "ec_model_pair_sd",
            "ec_sample_profile_sd",
            "n_comparisons",
        ):
            assert observed_summary[field] == pytest.approx(
                expected_summary[field], nan_ok=True
            )
        assert observed.pairwise.empty
        assert observed.samplewise is None

    classification = compute_classification_ec(
        np.asarray([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),
        np.asarray([0, 1, 1]),
        empty_unions="1",
        output_detail="summary",
    )
    assert classification.pairwise.empty
    assert len(classification.leave_one_model_out) == 3


@pytest.mark.fast
def test_ratio_diff_sign_primary_summary_cannot_cancel_opposite_directions() -> None:
    residuals = np.asarray([[1.0, 1.0], [2.0, -2.0]])

    computation = compute_regression_ec(residuals, methods=["ratio_diff_sign"])[0]
    summary = computation.summary()

    assert summary["ec_mean"] == pytest.approx(1.0 / 3.0)
    assert summary["ec_signed_mean"] == pytest.approx(0.0)
    assert computation.pairwise["pair_mean"].item() == pytest.approx(1.0 / 3.0)
    assert computation.pairwise["pair_signed_mean"].item() == pytest.approx(0.0)
    assert computation.samplewise["ec_mean"].tolist() == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0]
    )
    assert computation.samplewise["ec_signed_mean"].tolist() == pytest.approx(
        [1.0 / 3.0, -1.0 / 3.0]
    )


@pytest.mark.fast
def test_ratio_diff_sign_primary_preserves_zero_nonzero_difference() -> None:
    residuals = np.asarray([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]])

    computation = compute_regression_ec(residuals, methods=["ratio_diff_sign"])[0]
    summary = computation.summary()

    assert regression_pairwise_consistency(
        residuals[0], residuals[1], "ratio_diff_sign"
    ).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert summary["ec_mean"] == pytest.approx(2.0 / 3.0)
    assert summary["ec_signed_mean"] == pytest.approx(0.0)
    assert computation.pairwise["pair_mean"].item() == pytest.approx(2.0 / 3.0)
    assert computation.samplewise["ec_mean"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert computation.samplewise["ec_signed_mean"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )


@pytest.mark.fast
def test_classification_iou_and_empty_union_policy() -> None:
    truth = np.asarray([0, 1, 1, 0])
    predictions = np.asarray([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    computation = compute_classification_ec(predictions, truth)

    assert computation.matrix[0, 1] == pytest.approx(1.0 / 3.0)
    assert computation.matrix[0, 2] == pytest.approx(1.0 / 3.0)
    assert computation.matrix[1, 2] == pytest.approx(0.0)
    summary = computation.summary()
    assert summary["scientific_status"] == "published_classification_error_iou"
    assert summary["paper_equation"] == "1"
    assert summary["reference_url"] == ("https://doi.org/10.3390/diagnostics13071315")
    assert summary["inferential_status"] == ("descriptive_only_not_confidence_interval")
    perfect = compute_classification_ec(
        np.vstack([truth, truth]), truth, empty_unions="1"
    )
    assert perfect.values.tolist() == [1.0]
    assert np.isnan(perfect.summary()["leave_one_out_mean"])

    with pytest.warns(UserWarning, match="empty error unions"):
        perfect_default = compute_classification_ec(np.vstack([truth, truth]), truth)
    perfect_summary = perfect_default.summary()
    assert np.isnan(perfect_default.values).all()
    assert np.isnan(perfect_default.matrix).all()
    assert perfect_summary["empty_union_policy"] == "warn"
    assert perfect_summary["empty_union_pairs"] == 1
    assert perfect_summary["n_model_pairs"] == 1
    assert perfect_summary["n_valid_model_pairs"] == 0
    assert perfect_summary["n_comparisons"] == 0

    perfect_nan = compute_classification_ec(
        np.vstack([truth, truth]), truth, empty_unions="nan"
    )
    assert np.isnan(np.diag(perfect_nan.matrix)).all()


@pytest.mark.fast
def test_classification_total_and_leave_one_out_are_set_iou() -> None:
    truth = np.zeros(5, dtype=int)
    predictions = np.zeros((4, 5), dtype=int)
    predictions[:, 0] = 1
    predictions[0, 1] = 1
    predictions[1, 2] = 1
    predictions[2, 3] = 1
    predictions[3, 4] = 1

    computation = compute_classification_ec(predictions, truth)
    summary = computation.summary()

    assert summary["ec_mean"] == pytest.approx(1.0 / 3.0)
    assert summary["total_consistency"] == pytest.approx(1.0 / 5.0)
    assert summary["leave_one_out_mean"] == pytest.approx(1.0 / 4.0)
    assert summary["leave_one_model_out_mean"] == pytest.approx(1.0 / 4.0)
    assert summary["n_leave_one_model_out"] == 4
    assert summary["total_error_intersection"] == 1
    assert summary["total_error_union"] == 5
    leave_one_model_out = computation.leave_one_model_out
    assert leave_one_model_out is not None
    assert leave_one_model_out["model_removed"].tolist() == [0, 1, 2, 3]
    assert leave_one_model_out["consistency"].tolist() == pytest.approx(
        [0.25, 0.25, 0.25, 0.25]
    )


@pytest.mark.fast
def test_leave_one_model_out_keeps_undefined_rows_for_audit() -> None:
    truth = np.asarray([0, 1, 0])
    computation = compute_classification_ec(
        np.vstack([truth, truth, truth]),
        truth,
        empty_unions="drop",
    )

    leave_one_model_out = computation.leave_one_model_out
    assert leave_one_model_out is not None
    assert len(leave_one_model_out) == 3
    assert leave_one_model_out["empty_union"].all()
    assert not leave_one_model_out["included_in_summary"].any()
    assert leave_one_model_out["consistency"].isna().all()
    assert computation.summary()["n_leave_one_model_out"] == 3
    assert np.isnan(computation.summary()["leave_one_model_out_mean"])


@pytest.mark.fast
def test_regression_model_rebuild_keeps_single_output() -> None:
    result = SimpleNamespace(model_cls=MLPEstimator, model=MLPEstimator(num_classes=1))

    regression = _init_model(result, pd.Series([0.1, 0.2, 0.3]), False)
    classification = _init_model(result, pd.Series([0, 1, 2]), True)

    assert regression.fixed_args["module__num_classes"] == 1
    assert regression.is_classifier is False
    assert classification.fixed_args["module__num_classes"] == 3
    assert classification.is_classifier is True


@pytest.mark.fast
def test_performance_rank_is_computed_within_ec_method() -> None:
    summary = pd.DataFrame(
        {
            "target": ["target"] * 4,
            "model": ["a", "b", "a", "b"],
            "selection": ["none"] * 4,
            "embed_selector": ["none"] * 4,
            "ec_method": ["ratio", "ratio", "ratio_diff", "ratio_diff"],
            "ec_mean": [0.8, 0.7, 0.2, 0.3],
            "optimal_value": [1.0, 1.0, 0.0, 0.0],
        }
    )
    performance = pd.DataFrame(
        {
            "target": ["target", "target"],
            "model": ["a", "b"],
            "selection": ["none", "none"],
            "embed_selector": ["none", "none"],
            "metric": ["acc", "acc"],
            "ec_trial_mean": [0.9, 0.8],
        }
    )

    ranking = compute_model_ec_ranking(summary, performance)

    assert ranking.groupby("ec_method")["rank_by_performance"].apply(list).to_dict() == {
        "ratio": [1.0, 2.0],
        "ratio_diff": [1.0, 2.0],
    }


@pytest.mark.fast
def test_msqe_performance_ranking_minimizes_error() -> None:
    summary = pd.DataFrame(
        {
            "target": ["target", "target"],
            "model": ["a", "b"],
            "selection": ["none", "none"],
            "embed_selector": ["none", "none"],
            "ec_method": ["ratio", "ratio"],
            "ec_mean": [0.8, 0.7],
            "optimal_value": [1.0, 1.0],
        }
    )
    performance = pd.DataFrame(
        {
            "target": ["target", "target"],
            "model": ["a", "b"],
            "selection": ["none", "none"],
            "embed_selector": ["none", "none"],
            "metric": ["msqe", "msqe"],
            "ec_trial_mean": [1.0, 4.0],
        }
    )

    ranking = compute_model_ec_ranking(summary, performance)

    assert ranking["rank_by_performance"].tolist() == [1.0, 2.0]


@pytest.mark.fast
def test_signed_reference_method_is_not_ranked() -> None:
    computation = compute_regression_ec(
        np.asarray([[1.0, 1.0], [2.0, -2.0]]),
        methods=["ratio_diff_sign_reference"],
    )[0]
    summary = pd.DataFrame(
        [
            {
                "target": "target",
                "model": "a",
                "selection": "none",
                "embed_selector": "none",
                **computation.summary(),
            }
        ]
    )
    performance = pd.DataFrame(
        [
            {
                "target": "target",
                "model": "a",
                "selection": "none",
                "embed_selector": "none",
                "metric": "mae",
                "ec_trial_mean": 1.0,
            }
        ]
    )

    ranking = compute_model_ec_ranking(summary, performance)

    assert ranking["stability_distance"].isna().all()
    assert ranking["rank_by_stability"].isna().all()
    assert ranking["rank_by_performance"].isna().all()


@pytest.mark.fast
@pytest.mark.parametrize("is_classification", [False, True])
def test_grouped_kfold_repetitions_shuffle_groups(is_classification) -> None:
    n_groups = 18
    group_size = 10
    groups = pd.Series(np.repeat(np.arange(n_groups), group_size))
    X = pd.DataFrame({"x": np.arange(len(groups), dtype=float)})
    if is_classification:
        y = pd.Series(np.tile([0, 1], len(groups) // 2))
    else:
        y = pd.Series(np.arange(len(groups), dtype=float))

    signatures = []
    for seed in (11, 12, 13):
        splitter = OmniKFold(
            n_splits=3,
            is_classification=is_classification,
            grouped=True,
            shuffle=True,
            seed=seed,
            warn_on_fallback=False,
        )
        splits, fallback = splitter.split(X, y, groups)
        assert not fallback
        signature = tuple(
            sorted(
                tuple(sorted(groups.iloc[validation].unique()))
                for _, validation in splits
            )
        )
        signatures.append(signature)
        for train, validation in splits:
            assert set(groups.iloc[train]).isdisjoint(groups.iloc[validation])

    assert len(set(signatures)) == len(signatures)


@pytest.mark.fast
def test_sample_and_group_diagnostics() -> None:
    cls = classification_sample_diagnostics(
        np.asarray([[False, True], [True, True]]),
        row_ids=[10, 11],
        groups=pd.Series(["a", "b"]),
    )
    reg = regression_sample_diagnostics(
        np.asarray([[1.0, 2.0], [1.5, 3.0]]),
        row_ids=[10, 11],
        groups=pd.Series(["a", "b"]),
    )

    assert cls["sample_diagnostics.csv"]["error_rate"].tolist() == [0.5, 1.0]
    assert "group_diagnostics.csv" in cls
    assert "mean_abs_residual_mean" in reg["group_diagnostics.csv"]


@pytest.mark.fast
def test_backend_selection_respects_runtime_policy_and_work_size() -> None:
    capabilities = HardwareCapabilities(True, False, False, False)
    cuda = SimpleNamespace(runtime=RuntimePolicy(DeviceIntent.CUDA, capabilities))
    auto = SimpleNamespace(runtime=RuntimePolicy(DeviceIntent.Auto, capabilities))
    cpu = SimpleNamespace(runtime=RuntimePolicy(DeviceIntent.CPU, capabilities))

    assert resolve_ec_backend(cuda, 3, 10).resolved == "torch_cuda"
    assert resolve_ec_backend(auto, 3, 10).resolved == "numpy"
    assert resolve_ec_backend(auto, 400, 100).resolved == "torch_cuda"
    assert resolve_ec_backend(cpu, 400, 100).resolved == "numpy"


@pytest.mark.fast
def test_backend_reports_unavailable_cuda_fallback() -> None:
    unavailable = HardwareCapabilities(False, False, False, False)
    explicit = SimpleNamespace(runtime=RuntimePolicy(DeviceIntent.CUDA, unavailable))
    automatic = SimpleNamespace(runtime=RuntimePolicy(DeviceIntent.Auto, unavailable))

    with pytest.raises(CudaConfigurationError, match="error consistency"):
        resolve_ec_backend(explicit, 400, 100)
    assert resolve_ec_backend(automatic, 400, 100).reason == "cuda_unavailable"


@pytest.mark.fast
def test_classification_cuda_error_falls_back_to_numpy(monkeypatch) -> None:
    from df_analyze.analysis.error_consistency import classification

    original = classification._pair_counts

    def fail_cuda(errors, pairs, use_cuda):
        if use_cuda:
            raise RuntimeError("simulated CUDA failure")
        return original(errors, pairs, use_cuda)

    monkeypatch.setattr(classification, "_pair_counts", fail_cuda)
    backend = ECBackendDecision("auto", "torch_cuda", "size_threshold", 10)

    with pytest.warns(UserWarning, match="falling back to numpy"):
        computation = compute_classification_ec(
            np.asarray([[0, 1], [1, 1]]),
            np.asarray([0, 0]),
            backend=backend,
        )

    assert computation.summary()["ec_backend_resolved"] == "numpy"
    assert computation.summary()["ec_backend_reason"] == "torch_cuda_error:RuntimeError"

    strict = ECBackendDecision("cuda", "torch_cuda", "device_cuda", 10)
    with pytest.raises(RuntimeError, match="strict"):
        compute_classification_ec(
            np.asarray([[0, 1], [1, 1]]),
            np.asarray([0, 0]),
            backend=strict,
        )


def _options() -> SimpleNamespace:
    return SimpleNamespace(
        ec_folds=2,
        ec_repetitions=2,
        ec_model_seed_mode="vary",
        ec_methods=None,
        ec_holdout_role="validation",
        ec_output_detail="full",
        ec_resume=False,
        ec_checkpoint_every=5,
        ec_save_predictions=True,
        ec_empty_unions="1",
        ec_epsilon=0.0,
        seed=13,
        device=DeviceIntent.CPU,
    )


def _runner_case(is_classification: bool):
    n_samples = 120 if is_classification else 12
    X = pd.DataFrame(
        {
            "x0": np.arange(n_samples, dtype=float),
            "x1": np.arange(n_samples, dtype=float) * 2,
        }
    )
    if is_classification:
        y = pd.Series([0, 1] * (n_samples // 2), name="target")
        model_cls = DummyClassifier
        metric = ClassifierScorer.Accuracy
        params = {"strategy": "most_frequent"}
        split = 80
        metric_name = "acc"
    else:
        y = pd.Series(np.arange(n_samples, dtype=float), name="target")
        model_cls = DummyRegressor
        metric = RegressorScorer.MAE
        params = {"strategy": "mean"}
        split = 8
        metric_name = "mae"
    prep_train = PreparedData(
        X.iloc[:split], y.iloc[:split], groups=None, is_classification=is_classification
    )
    prep_test = PreparedData(
        X.iloc[split:], y.iloc[split:], groups=None, is_classification=is_classification
    )
    model = model_cls()
    result = HtuneResult(
        selection="none",
        selected_cols=["x0", "x1"],
        embed_select_model=None,
        model_cls=model_cls,
        model=model,
        params=params,
        metric=metric,
        score=0.5,
        preds_test=pd.Series(dtype=float),
        preds_train=pd.Series(dtype=float),
        probs_test=None,
        probs_train=None,
    )
    eval_results = EvaluationResults(
        df=pd.DataFrame(
            [
                {
                    "model": "dummy",
                    "selection": "none",
                    "embed_selector": "none",
                    "metric": metric_name,
                    "trainset": 0.5,
                    "holdout": 0.5,
                    "5-fold": 0.5,
                }
            ]
        ),
        X_train=prep_train.X,
        y_train=prep_train.y,
        X_test=prep_test.X,
        y_test=prep_test.y,
        results=[result],
        is_classification=is_classification,
    )
    return prep_train, prep_test, eval_results


@pytest.mark.fast
@pytest.mark.parametrize("is_classification", [False, True])
def test_repeated_kfold_runner_uses_common_holdout(tmp_path, is_classification) -> None:
    prep_train, prep_test, eval_results = _runner_case(is_classification)
    output_dir = tmp_path / "error_consistency"
    result = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        _options(),
        base_dir=output_dir,
    )

    assert set(result.summary["n_ec_models"]) == {4}
    assert set(result.trial_design["n_validation"]) == {len(prep_train.X) // 2}
    assert not result.trial_design["split_fallback"].any()
    assert not result.summary["group_split_fallback_used"].any()
    assert result.trial_design["group_overlap"].isna().all()
    assert result.trial_design["partition_signature"].nunique() == 2
    assert set(result.summary["n_unique_partitions"]) == {2}
    assert result.trial_design["model_seed"].nunique() == 4
    assert set(result.trial_design["model_seed_mode"]) == {"vary"}
    assert len(result.fold_assignments) == 2 * len(prep_train.X)
    for _, assignments in result.fold_assignments.groupby("repetition"):
        assert sorted(assignments["train_position"]) == list(range(len(prep_train.X)))
    assert set(result.summary["n_within_repetition_pairs"]) == {2}
    assert set(result.summary["n_between_repetition_pairs"]) == {4}
    assert set(result.summary["n_repetition_estimates"]) == {2}
    expected_status = (
        "published_classification_error_iou"
        if is_classification
        else "experimental_descriptive_diagnostic"
    )
    assert set(result.summary["scientific_status"]) == {expected_status}
    assert set(result.summary["inferential_status"]) == {
        "descriptive_only_not_confidence_interval"
    }
    evidence_scope = result.metadata["method_evidence_scope"]
    assert (
        "Equation 1 of Levman" in evidence_scope
        if is_classification
        else "Regression EC methods compare residuals" in evidence_scope
    )
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "fold_assignments.csv").exists()
    guard = pd.read_csv(output_dir / "selection_guard.csv")
    assert guard["selection_outputs_enabled"].item()
    generated_readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## Read these files first" in generated_readme
    assert "`trial_failures.csv`" in generated_readme
    detail = output_dir / "target" / "dummy" / "none_none"
    assert (detail / "fold_assignments.csv").exists()
    pairwise = pd.read_csv(detail / "pairwise_values.csv")
    n_methods = 1 if is_classification else 7
    assert pairwise["pair_scope"].value_counts().to_dict() == {
        "between_repetition": 4 * n_methods,
        "within_repetition": 2 * n_methods,
    }
    assert (detail / "trial_predictions.csv").exists()
    saved = pd.read_csv(detail / "trial_predictions.csv")
    assert saved["row_id"].tolist() == prep_test.X.index.tolist()
    assert saved["holdout_position"].tolist() == list(range(len(prep_test.X)))
    assert saved["y_true"].tolist() == prep_test.y.tolist()
    assert saved.shape[1] == 7
    residual_or_error = pd.read_csv(detail / "residual_or_error_matrix.csv")
    assert residual_or_error["y_true"].tolist() == prep_test.y.tolist()
    if is_classification:
        leave_one_model_out = pd.read_csv(detail / "leave_one_model_out.csv")
        assert leave_one_model_out["model_removed"].tolist() == [0, 1, 2, 3]
    if not is_classification:
        sample_ec = pd.read_csv(detail / "sample_ec_ratio.csv")
        assert sample_ec["row_id"].tolist() == prep_test.X.index.tolist()
        assert (detail / "plots" / "pairwise_ec_ratio.png").exists()
        assert (output_dir / "plots" / "ec_ratio_vs_mae.png").exists()


@pytest.mark.fast
@pytest.mark.parametrize("is_classification", [False, True])
def test_reference_pipeline_golden_outputs(tmp_path, is_classification) -> None:
    golden = json.loads(
        (
            Path(__file__).parent
            / "data"
            / "error_consistency"
            / "reference_pipeline_golden.json"
        ).read_text(encoding="utf-8")
    )
    expected = golden["classification" if is_classification else "regression"]
    prep_train, prep_test, eval_results = _runner_case(is_classification)
    options = _options()
    if not is_classification:
        options.ec_methods = REGRESSION_PAPER_METHODS
    output_dir = tmp_path / "error_consistency"

    result = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        options,
        base_dir=output_dir,
    )
    design = result.trial_design.sort_values("model_index")
    assert design["model_seed"].astype(int).tolist() == expected["trial_model_seeds"]
    assert (
        design.groupby("repetition")["partition_signature"].first().tolist()
        == (expected["partition_signatures"])
    )

    detail = output_dir / "target" / "dummy" / "none_none"
    saved = pd.read_csv(detail / "trial_predictions.csv")
    if is_classification:
        model_columns = saved.filter(regex=r"^model_\d+$")
        assert len(saved) == expected["n_holdout_rows"]
        assert np.unique(model_columns.to_numpy()).tolist() == [
            expected["prediction_value"]
        ]
        summary = result.summary.iloc[0]
        assert summary["ec_mean"] == pytest.approx(expected["ec_mean"])
        assert summary["total_error_intersection"] == expected["total_error_intersection"]
        assert summary["total_error_union"] == expected["total_error_union"]
        loo = pd.read_csv(detail / "leave_one_model_out.csv")
        assert loo["consistency"].tolist() == pytest.approx(
            expected["leave_one_model_out"]
        )
    else:
        assert saved["y_true"].tolist() == pytest.approx(expected["y_true"])
        observed_predictions = saved.filter(regex=r"^model_\d+$").to_numpy().T
        np.testing.assert_allclose(
            observed_predictions,
            np.asarray(expected["predictions_by_model"]),
        )
        mae_scores = result.trial_scores[result.trial_scores["metric"] == "mae"]
        assert mae_scores.sort_values("model_index")["score"].tolist() == pytest.approx(
            expected["trial_mae"]
        )
        observed = result.summary.set_index("ec_method")
        assert set(observed.index) == set(expected["summary"])
        for method, fields in expected["summary"].items():
            for field, value in fields.items():
                assert observed.loc[method, field] == pytest.approx(value)


@pytest.mark.fast
def test_summary_output_detail_keeps_audit_outputs_without_large_details(
    tmp_path,
) -> None:
    prep_train, prep_test, eval_results = _runner_case(True)
    options = _options()
    options.ec_output_detail = "summary"
    options.ec_save_predictions = False
    output_dir = tmp_path / "error_consistency"

    result = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        options,
        base_dir=output_dir,
    )

    detail = output_dir / "target" / "dummy" / "none_none"
    assert not (detail / "pairwise_values.csv").exists()
    assert not (detail / "sample_diagnostics.csv").exists()
    assert not (detail / "trial_predictions.csv").exists()
    assert (detail / "leave_one_model_out.csv").exists()
    assert set(result.summary["n_within_repetition_pairs"]) == {2}
    assert set(result.summary["n_between_repetition_pairs"]) == {4}
    failures = pd.read_csv(output_dir / "trial_failures.csv")
    assert failures.empty
    manifest = json.loads(
        (output_dir / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "1.0"
    assert manifest["method_implementation_version"]
    assert manifest["resolved_ec_settings"]["output_detail"] == "summary"
    assert manifest["audit_files"]["trial_failures"] == "trial_failures.csv"


@pytest.mark.fast
def test_checkpoint_fingerprint_includes_model_class_and_tuned_parameters() -> None:
    class AlternateDummyRegressor(DummyRegressor):
        pass

    prep_train, prep_test, eval_results = _runner_case(False)
    tuned = eval_results.results[0]
    common = {
        "identity": {
            "target": "target",
            "model": "dummy",
            "selection": "none",
            "embed_selector": "none",
        },
        "options": _options(),
        "X_train": prep_train.X,
        "y_train": prep_train.y,
        "X_holdout": prep_test.X,
        "y_holdout": prep_test.y,
        "is_classification": False,
    }

    baseline, payload = configuration_fingerprint(
        **common,
        model_class=tuned.model_cls,
        tuned_parameters={"strategy": "mean", "constant": np.float64(1.0)},
    )
    changed_parameters, _ = configuration_fingerprint(
        **common,
        model_class=tuned.model_cls,
        tuned_parameters={"strategy": "median", "constant": np.float64(1.0)},
    )
    changed_class, _ = configuration_fingerprint(
        **common,
        model_class=AlternateDummyRegressor,
        tuned_parameters={"strategy": "mean", "constant": np.float64(1.0)},
    )

    assert baseline != changed_parameters
    assert baseline != changed_class
    assert payload["model_class"].endswith(".DummyRegressor")
    assert payload["tuned_parameters"]["constant"] == 1.0


@pytest.mark.fast
def test_complete_checkpoint_resume_does_not_refit_models(tmp_path) -> None:
    class CountingDummyRegressor(DummyRegressor):
        refit_calls = 0

        def refit_tuned(self, X, y, g=None, tuned_args=None) -> None:
            type(self).refit_calls += 1
            return super().refit_tuned(X, y, g=g, tuned_args=tuned_args)

    prep_train, prep_test, eval_results = _runner_case(False)
    tuned = eval_results.results[0]
    tuned.model_cls = CountingDummyRegressor
    tuned.model = CountingDummyRegressor()
    options = _options()
    output_dir = tmp_path / "error_consistency"

    first = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        options,
        base_dir=output_dir,
    )
    assert CountingDummyRegressor.refit_calls == 4

    options.ec_resume = True
    second = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        options,
        base_dir=output_dir,
    )
    assert CountingDummyRegressor.refit_calls == 4
    assert second.metadata["n_resumed_complete_configurations"] == 1
    assert first.summary["ec_method"].tolist() == second.summary["ec_method"].tolist()
    np.testing.assert_allclose(
        first.summary[["ec_mean", "ec_sd", "ec_model_pair_sd"]].to_numpy(float),
        second.summary[["ec_mean", "ec_sd", "ec_model_pair_sd"]].to_numpy(float),
        equal_nan=True,
    )


@pytest.mark.fast
def test_partial_checkpoint_resumes_after_completed_repetition(
    tmp_path, monkeypatch
) -> None:
    from df_analyze.analysis.error_consistency import runner

    class CountingDummyRegressor(DummyRegressor):
        refit_calls = 0

        def refit_tuned(self, X, y, g=None, tuned_args=None) -> None:
            type(self).refit_calls += 1
            return super().refit_tuned(X, y, g=g, tuned_args=tuned_args)

    prep_train, prep_test, eval_results = _runner_case(False)
    tuned = eval_results.results[0]
    tuned.model_cls = CountingDummyRegressor
    tuned.model = CountingDummyRegressor()
    options = _options()
    options.ec_checkpoint_every = 1
    output_dir = tmp_path / "error_consistency"
    original_save = runner.save_partial_checkpoint

    def interrupt_after_first(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs["completed_repetitions"] == 1:
            raise RuntimeError("simulated interruption after checkpoint")

    monkeypatch.setattr(runner, "save_partial_checkpoint", interrupt_after_first)
    with pytest.raises(RuntimeError, match="did not complete"):
        run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            options,
            base_dir=output_dir,
        )
    assert CountingDummyRegressor.refit_calls == 2

    monkeypatch.setattr(runner, "save_partial_checkpoint", original_save)
    options.ec_resume = True
    resumed = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        options,
        base_dir=output_dir,
    )
    assert CountingDummyRegressor.refit_calls == 4
    assert resumed.metadata["n_resumed_partial_configurations"] == 1
    assert set(resumed.summary["n_ec_models"]) == {4}
    assert len(resumed.trial_design) == 4


@pytest.mark.fast
def test_failed_trial_is_recorded_and_remaining_trials_are_analyzed(
    tmp_path,
) -> None:
    class FlakyDummyRegressor(DummyRegressor):
        refit_calls = 0

        def refit_tuned(self, X, y, g=None, tuned_args=None) -> None:
            type(self).refit_calls += 1
            if type(self).refit_calls == 2:
                raise RuntimeError("intentional fold failure")
            return super().refit_tuned(X, y, g=g, tuned_args=tuned_args)

    prep_train, prep_test, eval_results = _runner_case(False)
    tuned = eval_results.results[0]
    tuned.model_cls = FlakyDummyRegressor
    tuned.model = FlakyDummyRegressor()
    output_dir = tmp_path / "error_consistency"

    with pytest.warns(UserWarning, match="trial failed and was recorded"):
        result = run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            _options(),
            base_dir=output_dir,
        )

    assert set(result.summary["n_ec_models"]) == {3}
    assert set(result.summary["n_failed_trials"]) == {1}
    assert result.trial_design["status"].value_counts().to_dict() == {
        "success": 3,
        "failed": 1,
    }
    failures = pd.read_csv(output_dir / "trial_failures.csv")
    assert len(failures) == 1
    assert failures["error_type"].item() == "RuntimeError"
    assert "intentional fold failure" in failures["reason"].item()


@pytest.mark.fast
def test_final_test_holdout_disables_selection_outputs(tmp_path) -> None:
    prep_train, prep_test, eval_results = _runner_case(True)
    options = _options()
    options.ec_holdout_role = "test"
    output_dir = tmp_path / "error_consistency"

    with pytest.warns(UserWarning, match="final test data"):
        result = run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            options,
            base_dir=output_dir,
        )

    assert result.metadata["holdout_role"] == "test"
    assert result.metadata["selection_outputs_enabled"] is False
    assert pd.read_csv(output_dir / "model_ec_ranking.csv").empty
    assert pd.read_csv(output_dir / "correlation_summary.csv").empty
    guard = pd.read_csv(output_dir / "selection_guard.csv")
    assert guard["holdout_role"].item() == "test"
    assert not bool(guard["selection_outputs_enabled"].item())
    assert not (output_dir / "plots" / "ec_classification_iou_vs_acc.png").exists()


@pytest.mark.fast
def test_ec_runner_restores_global_random_state(tmp_path) -> None:
    prep_train, prep_test, eval_results = _runner_case(True)
    random.seed(101)
    np.random.seed(202)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    expected_python = random.random()
    expected_numpy = np.random.random()
    random.setstate(python_state)
    np.random.set_state(numpy_state)

    run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        _options(),
        base_dir=tmp_path / "error_consistency",
    )

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


@pytest.mark.fast
def test_cpu_ec_does_not_probe_cuda(tmp_path, monkeypatch) -> None:
    def unexpected_cuda_probe():
        raise AssertionError("--device cpu must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_probe)
    prep_train, prep_test, eval_results = _runner_case(True)

    result = run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        _options(),
        base_dir=tmp_path / "error_consistency",
    )

    assert not result.summary.empty


@pytest.mark.fast
def test_ec_runner_releases_fold_model_before_cleanup(tmp_path) -> None:
    class CleanupAwareDummy(DummyClassifier):
        cleanup_states = []

        def _cleanup_after_fold(self) -> None:
            self.cleanup_states.append(self.tuned_model is None and self.model is None)

    prep_train, prep_test, eval_results = _runner_case(True)
    result = eval_results.results[0]
    result.model_cls = CleanupAwareDummy
    result.model = CleanupAwareDummy()

    run_error_consistency_analysis(
        prep_train,
        prep_test,
        eval_results,
        _options(),
        base_dir=tmp_path / "error_consistency",
    )

    assert CleanupAwareDummy.cleanup_states == [True] * 4


@pytest.mark.fast
def test_ec_runner_metadata_records_cuda_fallback(tmp_path, monkeypatch) -> None:
    from df_analyze.analysis.error_consistency import classification, runner

    original = classification._pair_counts

    def fail_cuda(errors, pairs, use_cuda):
        if use_cuda:
            raise RuntimeError("simulated CUDA failure")
        return original(errors, pairs, use_cuda)

    monkeypatch.setattr(classification, "_pair_counts", fail_cuda)
    monkeypatch.setattr(
        runner,
        "resolve_ec_backend",
        lambda *args, **kwargs: ECBackendDecision(
            "auto", "torch_cuda", "size_threshold", 10
        ),
    )
    prep_train, prep_test, eval_results = _runner_case(True)

    with pytest.warns(UserWarning, match="falling back to numpy"):
        result = run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            _options(),
            base_dir=tmp_path / "error_consistency",
        )

    assert result.metadata["ec_backends"] == [
        {
            "ec_backend_requested": "auto",
            "ec_backend_resolved": "numpy",
            "ec_backend_reason": "torch_cuda_error:RuntimeError",
            "ec_backend_work_items": 10,
        }
    ]


@pytest.mark.fast
def test_ec_runner_does_not_swallow_strict_cuda_failure(
    tmp_path, monkeypatch
) -> None:
    from df_analyze.analysis.error_consistency import classification, runner

    def fail_cuda(errors, pairs, use_cuda):
        if use_cuda:
            raise RuntimeError("simulated CUDA failure")
        return _pair_counts(errors, pairs, use_cuda)

    monkeypatch.setattr(classification, "_pair_counts", fail_cuda)
    monkeypatch.setattr(
        runner,
        "resolve_ec_backend",
        lambda *args, **kwargs: ECBackendDecision(
            "cuda", "torch_cuda", "device_cuda", 10
        ),
    )
    prep_train, prep_test, eval_results = _runner_case(True)
    options = _options()
    options.device = DeviceIntent.CUDA
    options.runtime = RuntimePolicy(
        DeviceIntent.CUDA,
        HardwareCapabilities(
            torch_cuda=True,
            torch_mps=False,
            catboost_cuda=False,
            xgboost_cuda=False,
        ),
    )

    with pytest.raises(RuntimeError, match="strict"):
        run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            options,
            base_dir=tmp_path / "error_consistency",
        )


@pytest.mark.fast
def test_multitarget_ec_reconstruction_keeps_model_args() -> None:
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    X_test = pd.DataFrame({"feature": [4.0, 5.0]})
    y_train = pd.DataFrame({"target_a": [0, 1, 0, 1], "target_b": [1, 0, 1, 0]})
    y_test = pd.DataFrame({"target_a": [0, 1], "target_b": [1, 0]})
    model_args = {"strategy": "uniform", "random_state": 41}
    source_model = DummyClassifier(model_args=model_args)
    tuned = HtuneResult(
        selection="none",
        selected_cols=["feature"],
        embed_select_model=None,
        model_cls=DummyClassifier,
        model=source_model,
        params={"strategy": "most_frequent"},
        metric=ClassifierScorer.Accuracy,
        score=0.5,
        preds_test=y_test.copy(),
        preds_train=y_train.copy(),
        probs_test={
            target: np.column_stack([1 - y_test[target], y_test[target]])
            for target in y_test
        },
        probs_train={
            target: np.column_stack([1 - y_train[target], y_train[target]])
            for target in y_train
        },
        per_target_tuning_scores={"target_a": 0.72, "target_b": 0.44},
    )
    evaluation = EvaluationResults(
        df=pd.DataFrame(
            {
                "model": ["dummy"],
                "selection": ["none"],
                "embed_selector": ["none"],
                "metric": ["acc"],
                "trainset": [0.5],
                "holdout": [0.5],
                "5-fold": [0.5],
            }
        ),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        results=[tuned],
        is_classification=True,
    )

    target_eval = _eval_results_for_target(
        evaluation,
        SimpleNamespace(X=X_train, y=y_train["target_a"]),
        SimpleNamespace(X=X_test, y=y_test["target_a"]),
        "target_a",
    )

    assert target_eval.results[0].model.model_args == model_args
    assert target_eval.results[0].score == pytest.approx(0.72)


@pytest.mark.fast
def test_error_consistency_cli_options(tmp_path) -> None:
    path = tmp_path / "input.csv"
    pd.DataFrame({"x": [0, 1], "target": [0, 1]}).to_csv(path, index=False)
    options = get_options(
        f"--df {path} --outdir {tmp_path} --error-consistency "
        "--ec-folds 3 --ec-repetitions 2 --ec-methods ratio ratio-diff "
        "--ec-save-predictions --ec-empty-unions nan --ec-epsilon 1e-8"
        " --ec-recurrence-threshold 0.65 --ec-model-seed-mode fixed"
        " --ec-holdout-role validation"
    )

    assert options.error_consistency is True
    assert options.ec_folds == 3
    assert options.ec_repetitions == 2
    assert options.ec_model_seed_mode == "fixed"
    assert options.ec_methods == ("ratio", "ratio_diff")
    assert options.ec_holdout_role == "validation"
    assert options.ec_save_predictions is True
    assert options.ec_empty_unions == "nan"
    assert options.ec_epsilon == pytest.approx(1e-8)
    assert options.ec_recurrence_threshold == pytest.approx(0.65)

    with pytest.raises(ArgumentError, match="ec-folds"):
        get_options(f"--df {path} --outdir {tmp_path} --ec-folds 1")
    with pytest.raises(ArgumentError, match="ec-recurrence-threshold"):
        get_options(f"--df {path} --outdir {tmp_path} --ec-recurrence-threshold 1.1")

    defaults = get_options(f"--df {path} --outdir {tmp_path}")
    assert defaults.error_consistency is False
    assert defaults.ec_repetitions == 5
    assert defaults.ec_model_seed_mode == "vary"
    assert defaults.ec_empty_unions == "warn"
    assert defaults.ec_holdout_role == "test"
    assert defaults.ec_profile == "none"
    assert defaults.ec_output_detail == "full"
    assert defaults.ec_resume is False
    assert defaults.ec_checkpoint_every == 5

    explicit = get_options(
        f"--df {path} --outdir {tmp_path} --mode regress "
        "--ec-methods ratio_diff_sign_magnitude ratio_diff_sign_reference"
    )
    assert explicit.ec_methods == (
        "ratio_diff_sign_magnitude",
        "ratio_diff_sign_reference",
    )
    with pytest.warns(UserWarning, match="ambiguous legacy EC method"):
        legacy = get_options(
            f"--df {path} --outdir {tmp_path} --mode regress --ec-methods ratio_diff_sign"
        )
    assert legacy.ec_methods == ("ratio_diff_sign",)


@pytest.mark.fast
def test_ec_paper_profiles_apply_defaults_and_preserve_explicit_overrides(
    tmp_path,
) -> None:
    path = tmp_path / "input.csv"
    pd.DataFrame({"x": [0, 1, 2, 3], "target": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        path, index=False
    )

    regression = get_options(
        f"--df {path} --outdir {tmp_path} --mode regress --ec-profile regression-paper"
    )
    assert regression.error_consistency is True
    assert regression.test_val_size == pytest.approx(0.2)
    assert regression.ec_folds == 5
    assert regression.ec_repetitions == 50
    assert regression.ec_model_seed_mode == "fixed"
    assert regression.ec_methods == REGRESSION_PAPER_METHODS
    assert regression.ec_holdout_role == "test"
    assert regression.ec_profile_overrides == {}
    resume_same_run = get_options(
        f"--df {path} --outdir {tmp_path} --mode regress "
        "--ec-profile regression-paper --ec-resume --ec-checkpoint-every 2"
    )
    assert resume_same_run.hash() == regression.hash()

    overridden = get_options(
        f"--df {path} --outdir {tmp_path} --mode regress "
        "--ec-profile regression-paper --test-val-size 0.3 "
        "--ec-repetitions 3 --ec-methods ratio --ec-holdout-role validation "
        "--ec-output-detail summary --ec-resume --ec-checkpoint-every 2"
    )
    assert overridden.test_val_size == pytest.approx(0.3)
    assert overridden.ec_repetitions == 3
    assert overridden.ec_methods == ("ratio",)
    assert overridden.ec_holdout_role == "validation"
    assert overridden.ec_output_detail == "summary"
    assert overridden.ec_resume is True
    assert overridden.ec_checkpoint_every == 2
    assert overridden.ec_profile_overrides == {
        "test_val_size": 0.3,
        "ec_repetitions": 3,
        "ec_methods": ["ratio"],
        "ec_holdout_role": "validation",
    }

    classification = get_options(
        f"--df {path} --outdir {tmp_path} --mode classify "
        "--ec-profile classification-paper"
    )
    assert classification.error_consistency is True
    assert classification.test_val_size == pytest.approx(0.2)
    assert classification.ec_folds == 5
    assert classification.ec_repetitions == 10
    assert classification.ec_model_seed_mode == "fixed"

    with pytest.raises(ArgumentError, match="requires regression mode"):
        get_options(
            f"--df {path} --outdir {tmp_path} --mode classify "
            "--ec-profile regression-paper"
        )


@pytest.mark.fast
def test_runner_raises_when_every_configuration_fails(tmp_path) -> None:
    prep_train, prep_test, eval_results = _runner_case(False)
    eval_results.results[0].score = np.nan
    output_dir = tmp_path / "error_consistency"

    with pytest.raises(RuntimeError, match="did not complete"):
        run_error_consistency_analysis(
            prep_train,
            prep_test,
            eval_results,
            _options(),
            base_dir=output_dir,
        )

    metadata = (output_dir / "metadata.json").read_text(encoding="utf-8")
    assert "finite score" in metadata


@pytest.mark.fast
def test_risk_stability_report_joins_by_holdout_row(tmp_path) -> None:
    aer_base = tmp_path / "adaptive_error"
    ec_base = tmp_path / "error_consistency"
    aer_predictions = aer_base / "models" / "dummyclassifier" / "predictions"
    target_name = "target / special"
    ec_details = detail_output_dir(
        ec_base,
        {
            "target": target_name,
            "model": "dummy",
            "selection": "none",
            "embed_selector": "none",
        },
    )
    aer_predictions.mkdir(parents=True)
    ec_details.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13],
            "correct": [1, 0, 1, 0],
            "aer": [0.05, 0.4, 0.3, 0.1],
        }
    ).to_csv(aer_predictions / "test_per_sample.csv", index=False)
    pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13],
            "error_rate": [0.0, 0.75, 0.25, 1.0],
            "instability": [0.0, 0.75, 0.75, 0.0],
        }
    ).to_csv(ec_details / "sample_diagnostics.csv", index=False)
    _, _, eval_results = _runner_case(True)
    options = SimpleNamespace(
        htune_cls_metric=ClassifierScorer.Accuracy,
        is_classification=True,
        classifiers=(),
        aer_top_k=0,
        aer_target_error=0.2,
        ec_recurrence_threshold=0.5,
    )

    report = write_risk_stability_report(
        target_name, eval_results, options, aer_base, ec_base
    )
    summary = risk_stability_summary(report)

    assert set(report["risk_stability_quadrant"]) == {
        "low_aer_low_recurrence",
        "high_aer_high_recurrence",
        "high_aer_low_recurrence",
        "low_aer_high_recurrence",
    }
    assert int(summary["n_samples"].sum()) == 4
    assert set(report["ec_recurrence_threshold"]) == {0.5}
    assert set(report["threshold_basis"]) == {"user_configurable_heuristic"}
    assert set(summary["ec_recurrence_threshold"]) == {0.5}
    assert (aer_base / "tables" / "risk_stability_summary.csv").exists()


@pytest.mark.fast
def test_risk_stability_empty_join_writes_readable_outputs(tmp_path) -> None:
    aer_base = tmp_path / "adaptive_error"
    ec_base = tmp_path / "error_consistency"
    aer_predictions = aer_base / "models" / "dummyclassifier" / "predictions"
    ec_details = ec_base / "target" / "dummy" / "none_none"
    aer_predictions.mkdir(parents=True)
    ec_details.mkdir(parents=True)
    pd.DataFrame({"row_id": [1], "correct": [1], "aer": [0.1]}).to_csv(
        aer_predictions / "test_per_sample.csv", index=False
    )
    pd.DataFrame({"row_id": [2], "error_rate": [0.0], "instability": [0.0]}).to_csv(
        ec_details / "sample_diagnostics.csv", index=False
    )
    _, _, eval_results = _runner_case(True)
    options = SimpleNamespace(
        htune_cls_metric=ClassifierScorer.Accuracy,
        is_classification=True,
        classifiers=(),
        aer_top_k=0,
        aer_target_error=0.2,
        ec_recurrence_threshold=0.5,
    )

    report = write_risk_stability_report(
        "target", eval_results, options, aer_base, ec_base
    )
    report_file = pd.read_csv(aer_base / "tables" / "risk_stability_report.csv")
    summary_file = pd.read_csv(aer_base / "tables" / "risk_stability_summary.csv")
    skipped_file = pd.read_csv(aer_base / "tables" / "risk_stability_skipped.csv")

    assert report.empty and report_file.empty and summary_file.empty
    assert {"row_id", "aer", "error_rate", "risk_stability_quadrant"}.issubset(
        report_file.columns
    )
    assert {"n_samples", "sample_fraction", "mean_ec_error_rate"}.issubset(
        summary_file.columns
    )
    assert skipped_file["reason"].str.contains("no matching holdout rows").all()
    assert report.attrs["skipped_configurations"] == skipped_file.to_dict("records")


@pytest.mark.fast
def test_risk_stability_summary_keeps_model_denominators_separate() -> None:
    report = pd.DataFrame(
        {
            "target": ["target"] * 4,
            "model": ["a", "a", "b", "b"],
            "selection": ["none"] * 4,
            "risk_stability_quadrant": ["high", "low", "high", "high"],
            "correct": [0, 1, 0, 1],
            "aer": [0.5, 0.1, 0.4, 0.3],
            "error_rate": [0.8, 0.2, 0.7, 0.6],
            "instability": [0.2, 0.3, 0.4, 0.5],
        }
    )

    summary = risk_stability_summary(report)

    fractions = summary.groupby("model")["sample_fraction"].sum()
    assert fractions.to_dict() == pytest.approx({"a": 1.0, "b": 1.0})
    assert summary.loc[
        (summary["model"] == "b") & (summary["risk_stability_quadrant"] == "high"),
        "sample_fraction",
    ].item() == pytest.approx(1.0)

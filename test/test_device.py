from __future__ import annotations

from types import SimpleNamespace

import jsonpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torch.nn import Dropout, Identity

from df_analyze import _main
from df_analyze import hypertune as hypertune_module
from df_analyze.analysis.adaptive_error import oof as adaptive_error_oof
from df_analyze.analysis.adaptive_error import test_stage as adaptive_error_test_stage
from df_analyze.analysis.adaptive_error.oof import build_oof_for_result
from df_analyze.cli import cli
from df_analyze.cli.cli import ProgramOptions, make_parser
from df_analyze.embedding import download as embedding_download
from df_analyze.embedding import embed
from df_analyze.embedding.cli import EmbeddingModality
from df_analyze.enumerables import (
    ClassifierScorer,
    DfAnalyzeClassifier,
    FeatureSelection,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.hypertune import EvaluationResults, evaluate_tuned
from df_analyze.models.catboost import CatBoostClassifier
from df_analyze.models.gandalf import GandalfEstimator, _dropout_layer
from df_analyze.models.kan import KANEstimator
from df_analyze.models.knn import (
    KNNClassifier,
    TorchKNNClassifier,
    TorchKNNRegressor,
)
from df_analyze.models.mlp import MLPEstimator
from df_analyze.models.tabpfn import TabPFNClassifierV3
from df_analyze.models.xgboost import XGBoostClassifier
from df_analyze.preprocessing.prepare import PreparedData
from df_analyze.runtime import hardware
from df_analyze.runtime.hardware import (
    DeviceIntent,
    HardwareCapabilities,
    RuntimeComponent,
    RuntimePolicy,
)
from df_analyze.selection import models as selection_models
from df_analyze.selection.models import ModelSelected
from df_analyze.selection.stepwise import get_dfanalyze_score


def runtime(
    intent: DeviceIntent = DeviceIntent.Auto,
    *,
    torch_cuda: bool = False,
    torch_mps: bool = False,
    catboost_cuda: bool = False,
    xgboost_cuda: bool = False,
) -> RuntimePolicy:
    return RuntimePolicy(
        intent,
        HardwareCapabilities(
            torch_cuda,
            torch_mps,
            catboost_cuda,
            xgboost_cuda,
        ),
    )


@pytest.mark.parametrize("probability", [-0.1, 1.0, np.nan, np.inf])
def test_gandalf_dropout_rejects_invalid_probabilities(probability: float) -> None:
    with pytest.raises(ValueError, match="0 <= probability < 1"):
        _dropout_layer(probability, "test dropout")


def test_gandalf_dropout_uses_identity_only_at_zero() -> None:
    assert isinstance(_dropout_layer(0.0, "test dropout"), Identity)
    layer = _dropout_layer(0.25, "test dropout")
    assert isinstance(layer, Dropout)
    assert layer.p == pytest.approx(0.25)


@pytest.mark.fast
def test_device_parser_and_intent() -> None:
    args = make_parser().parse_known_args(["--device", "CPU"])[0]

    assert args.device == "cpu"
    assert DeviceIntent.from_arg("cuda") is DeviceIntent.CUDA
    assert DeviceIntent.choices() == ["auto", "cpu", "cuda"]


@pytest.mark.fast
def test_runtime_routes_each_backend_independently() -> None:
    policy = runtime(
        torch_mps=True, catboost_cuda=True, xgboost_cuda=True
    ).with_workload(10_000, 100)

    assert policy.device_for(RuntimeComponent.CatBoost) == "cuda"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cuda"
    assert policy.device_for(RuntimeComponent.MLP) == "cpu"
    assert policy.device_for(RuntimeComponent.Gandalf) == "mps"
    assert policy.device_for(RuntimeComponent.LightGBM) == "cpu"


@pytest.mark.fast
def test_knn_workload_uses_actual_query_rows() -> None:
    below = runtime(torch_cuda=True).with_workload(
        10_000,
        100,
        n_queries=10,
    )
    at_threshold = runtime(torch_cuda=True).with_workload(
        10_000,
        100,
        n_queries=20,
    )

    below_decision = below.decision_for(RuntimeComponent.KNN)
    threshold_decision = at_threshold.decision_for(RuntimeComponent.KNN)
    assert below_decision.resolved == "cpu"
    assert below_decision.work_items == 10_000_000
    assert below_decision.n_queries == 10
    assert threshold_decision.resolved == "cuda"
    assert threshold_decision.work_items == 20_000_000


@pytest.mark.fast
def test_cpu_policy_does_not_probe_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected():
        raise AssertionError("CPU policy must not probe accelerator backends")

    monkeypatch.setattr(hardware, "_torch_capabilities", unexpected)
    monkeypatch.setattr(hardware, "_catboost_cuda_available", unexpected)
    policy = hardware.get_runtime(DeviceIntent.CPU)

    assert policy.device_for(RuntimeComponent.MLP) == "cpu"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cpu"


@pytest.mark.fast
def test_runtime_fallback_state_is_per_run() -> None:
    first = hardware.get_runtime(DeviceIntent.CPU).with_workload(100, 10)
    first.record_cpu_fallback(RuntimeComponent.KNN, "first_run_fallback")
    second = hardware.get_runtime(DeviceIntent.CPU).with_workload(100, 10)

    assert first.decision_for(RuntimeComponent.KNN).reason == "first_run_fallback"
    assert second.decision_for(RuntimeComponent.KNN).reason == "explicit_cpu"


@pytest.mark.fast
def test_runtime_fallback_is_isolated_per_model_task() -> None:
    base = runtime(DeviceIntent.Auto, xgboost_cuda=True)
    first = base.for_task(5_000, 100)
    second = base.for_task(5_000, 100)
    first.record_cpu_fallback(RuntimeComponent.XGBoost, "cuda_runtime_fallback:test")

    assert first.device_for(RuntimeComponent.XGBoost) == "cpu"
    assert second.device_for(RuntimeComponent.XGBoost) == "cuda"


@pytest.mark.fast
def test_explicit_cuda_is_strict_only_for_supported_components() -> None:
    policy = runtime(DeviceIntent.CUDA)

    with pytest.raises(hardware.CudaConfigurationError, match="PyTorch CUDA"):
        policy.device_for(RuntimeComponent.MLP)
    assert policy.decision_for(RuntimeComponent.MLP).resolved == "unavailable"
    assert policy.device_for(RuntimeComponent.LightGBM) == "cpu"
    assert policy.device_for(RuntimeComponent.Sklearn) == "cpu"


@pytest.mark.fast
def test_cuda_validation_allows_mixed_models_but_rejects_missing_backend() -> None:
    components = {
        "xgb": RuntimeComponent.XGBoost,
        "svm": RuntimeComponent.Sklearn,
        "preprocessing": RuntimeComponent.Preprocessing,
    }
    available = runtime(DeviceIntent.CUDA, xgboost_cuda=True)
    hardware.validate_cuda_request(available, components)
    plan = hardware.format_device_plan(available, components)
    assert "CUDA: xgb" in plan
    assert "CPU:  svm, preprocessing" in plan

    unavailable = runtime(DeviceIntent.CUDA)
    with pytest.raises(
        hardware.CudaConfigurationError,
        match="XGBoost CUDA is unavailable",
    ):
        hardware.validate_cuda_request(unavailable, components)


@pytest.mark.fast
def test_cuda_validation_rejects_cpu_only_selection() -> None:
    components = {
        "svm": RuntimeComponent.Sklearn,
        "preprocessing": RuntimeComponent.Preprocessing,
    }
    with pytest.raises(
        hardware.CudaConfigurationError,
        match="none of the selected models",
    ):
        hardware.validate_cuda_request(runtime(DeviceIntent.CUDA), components)


@pytest.mark.fast
def test_cuda_device_plan_reports_visibility_without_wrong_physical_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    hardware.cuda_device_name.cache_clear()
    try:
        plan = hardware.format_device_plan(
            runtime(DeviceIntent.CUDA, xgboost_cuda=True),
            {"xgb": RuntimeComponent.XGBoost},
        )
    finally:
        hardware.cuda_device_name.cache_clear()

    assert "CUDA_VISIBLE_DEVICES=2,3 (backend device index 0)" in plan


@pytest.mark.fast
def test_error_consistency_is_a_cuda_capable_planned_component() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.SVM,),
        regressors=(),
        wrapper_select=None,
        wrapper_model=WrapperSelectionModel.Linear,
        error_consistency=True,
        runtime=runtime(DeviceIntent.CUDA, torch_cuda=True),
    )

    components = _main._runtime_components(options)
    hardware.validate_cuda_request(options.runtime, components)

    assert (
        components["error-consistency"]
        is RuntimeComponent.ErrorConsistency
    )
    assert _main._resolved_devices(options)["error-consistency"] == "cuda"


@pytest.mark.fast
def test_model_adapters_receive_runtime_policy() -> None:
    torch_policy = runtime(DeviceIntent.CUDA, torch_cuda=True)
    catboost_policy = runtime(DeviceIntent.CUDA, catboost_cuda=True)
    xgboost_policy = runtime(DeviceIntent.CUDA, xgboost_cuda=True)

    mlp = MLPEstimator(num_classes=2).set_runtime(torch_policy)
    kan = KANEstimator(num_classes=2).set_runtime(torch_policy)
    gandalf = GandalfEstimator(num_classes=2).set_runtime(torch_policy)
    knn = KNNClassifier().set_runtime(torch_policy)
    catboost = CatBoostClassifier().set_runtime(catboost_policy)
    xgboost = XGBoostClassifier().set_runtime(xgboost_policy)

    knn_cls, knn_args = knn.model_cls_args(
        {"n_neighbors": 3, "metric": "l2", "n_jobs": -1}
    )
    catboost_args = {}
    catboost._maybe_use_gpu(catboost_args)
    _, xgboost_args = xgboost.model_cls_args({"tree_method": "gpu_hist"})

    assert mlp.fixed_args["device"] == "cuda"
    assert kan.fixed_args["device"] == "cuda"
    assert gandalf._trainer_hardware() == ("gpu", 1)
    assert knn_cls is TorchKNNClassifier
    assert knn_args == {
        "n_neighbors": 3,
        "metric": "l2",
        "device": "cuda",
        "runtime": torch_policy,
    }
    assert catboost_args == {"task_type": "GPU", "devices": "0"}
    assert xgboost_args["device"] == "cuda"
    assert xgboost_args["tree_method"] == "hist"


@pytest.mark.fast
def test_mlp_and_kan_runtime_device_cannot_be_overridden_by_model_args() -> None:
    cpu_policy = runtime(DeviceIntent.CPU)
    cuda_policy = runtime(DeviceIntent.CUDA, torch_cuda=True)

    mlp_cpu = MLPEstimator(num_classes=2, model_args={"device": "cuda"}).set_runtime(
        cpu_policy
    )
    kan_cpu = KANEstimator(num_classes=2, model_args={"device": "cuda"}).set_runtime(
        cpu_policy
    )
    mlp_cuda = MLPEstimator(num_classes=2, model_args={"device": "cpu"}).set_runtime(
        cuda_policy
    )

    assert mlp_cpu._runtime_model_args(mlp_cpu.model_args)["device"] == "cpu"
    assert kan_cpu._runtime_model_args(kan_cpu.model_args)["device"] == "cpu"
    assert mlp_cuda._runtime_model_args(mlp_cuda.model_args)["device"] == "cuda"


@pytest.mark.fast
def test_strict_cuda_feature_selection_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cuda(**_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(selection_models, "wrap_select_features", fail_cuda)
    options = SimpleNamespace(
        feat_select=(FeatureSelection.Wrapper,),
        wrapper_select=object(),
        runtime=runtime(DeviceIntent.CUDA, torch_cuda=True),
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        selection_models.model_select_features(
            prep_train=SimpleNamespace(),
            options=options,
        )


@pytest.mark.fast
def test_auto_keeps_small_traditional_workloads_on_cpu() -> None:
    policy = runtime(
        torch_cuda=True, catboost_cuda=True, xgboost_cuda=True
    ).with_workload(400, 20)

    assert policy.device_for(RuntimeComponent.KNN) == "cpu"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cpu"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cpu"
    decision = policy.decision_for(RuntimeComponent.CatBoost)
    assert decision.reason == "auto_workload_below_threshold"
    assert decision.work_items == 8_000
    assert decision.threshold == 1_000_000


@pytest.mark.fast
def test_auto_small_traditional_workloads_do_not_probe_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected():
        raise AssertionError("Small auto workloads must not probe CUDA backends")

    monkeypatch.setattr(hardware, "_torch_capabilities", unexpected)
    monkeypatch.setattr(hardware, "_catboost_cuda_available", unexpected)
    monkeypatch.setattr(hardware, "_xgboost_cuda_available", unexpected)
    policy = RuntimePolicy(
        DeviceIntent.Auto,
        HardwareCapabilities(None, False, None, None),
    ).with_workload(400, 20)

    assert policy.device_for(RuntimeComponent.KNN) == "cpu"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cpu"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cpu"


@pytest.mark.fast
def test_auto_uses_cuda_for_large_traditional_workloads() -> None:
    policy = runtime(
        torch_cuda=True, catboost_cuda=True, xgboost_cuda=True
    ).with_workload(10_000, 100)

    assert policy.device_for(RuntimeComponent.KNN) == "cuda"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cuda"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cuda"
    decision = policy.decision_for(RuntimeComponent.CatBoost)
    assert decision.reason == "auto_workload_at_or_above_threshold"
    assert decision.work_metric == "matrix_elements"
    assert decision.work_items == 1_000_000
    assert decision.threshold == 1_000_000

    below = runtime(catboost_cuda=True, xgboost_cuda=True).with_workload(9_999, 100)
    assert below.device_for(RuntimeComponent.CatBoost) == "cpu"
    assert below.device_for(RuntimeComponent.XGBoost) == "cuda"
    xgb_below = runtime(xgboost_cuda=True).with_workload(1_999, 100)
    assert xgb_below.device_for(RuntimeComponent.XGBoost) == "cpu"


@pytest.mark.fast
def test_explicit_cuda_overrides_auto_workload_thresholds() -> None:
    policy = runtime(
        DeviceIntent.CUDA,
        torch_cuda=True,
        catboost_cuda=True,
        xgboost_cuda=True,
    ).with_workload(10, 2)

    assert policy.device_for(RuntimeComponent.KNN) == "cuda"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cuda"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cuda"


@pytest.mark.fast
def test_auto_keeps_compute_heavy_torch_backends_accelerator_preferred() -> None:
    policy = runtime(torch_cuda=True).with_workload(10, 2)

    assert policy.device_for(RuntimeComponent.MLP) == "cuda"
    assert policy.decision_for(RuntimeComponent.MLP).reason == (
        "auto_accelerator_preferred"
    )


@pytest.mark.fast
def test_wrapper_knn_uses_largest_actual_fold_workload() -> None:
    class FakeWrapperKNN:
        shortname = "knn"
        runtime_component = RuntimeComponent.KNN
        seen = []

        def set_runtime(self, policy):
            self.runtime = policy
            return self

        def cv_score(self, X, y, groups, metric, test=False, splits=None):
            self.seen.append(
                (
                    self.runtime.decision_for(RuntimeComponent.KNN),
                    len(splits),
                    X.shape,
                )
            )
            return 0.75

    n_rows = 1_000
    X = pd.DataFrame(
        np.zeros((n_rows, 101)),
        columns=[f"f{idx}" for idx in range(101)],
    )
    y = pd.Series([0, 1] * (n_rows // 2))
    indices = np.arange(n_rows)
    splits = []
    for fold in range(5):
        query = indices[fold * 200 : (fold + 1) * 200]
        train = np.setdiff1d(indices, query)
        splits.append((train, query))

    outcome = get_dfanalyze_score(
        model_cls=FakeWrapperKNN,
        X=X,
        y=y,
        g=None,
        metric=ClassifierScorer.Accuracy,
        selected=set(),
        candidate="f0",
        is_forward=False,
        test=False,
        runtime=runtime(DeviceIntent.Auto, torch_cuda=True),
        splits=splits,
    )

    decision, n_splits, shape = FakeWrapperKNN.seen[-1]
    assert outcome.score == pytest.approx(0.75)
    assert n_splits == 5
    assert shape == (1_000, 100)
    assert decision.resolved == "cpu"
    assert decision.n_samples == 800
    assert decision.n_queries == 200
    assert decision.work_items == 16_000_000
    assert outcome.audit[-1]["stage"] == "completed"
    assert outcome.audit[-1]["n_samples"] == 800


@pytest.mark.fast
def test_cpu_knn_wrapper_avoids_nested_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_n_jobs = []

    def fake_cv_score(self, X, y, groups, metric, test=False, splits=None):
        seen_n_jobs.append(self.model_args["n_jobs"])
        return 0.5

    monkeypatch.setattr(KNNClassifier, "cv_score", fake_cv_score)
    X = pd.DataFrame({"first": np.arange(10), "second": np.arange(10)})
    y = pd.Series([0, 1] * 5)
    splits = [
        (np.arange(2, 10), np.arange(0, 2)),
        (np.r_[0:2, 4:10], np.arange(2, 4)),
    ]

    get_dfanalyze_score(
        model_cls=KNNClassifier,
        X=X,
        y=y,
        g=None,
        metric=ClassifierScorer.Accuracy,
        selected=set(),
        candidate="first",
        is_forward=True,
        test=False,
        runtime=runtime(DeviceIntent.Auto, torch_cuda=True),
        splits=splits,
    )

    assert seen_n_jobs == [1]


@pytest.mark.fast
def test_catboost_and_xgboost_tuning_fits_are_single_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEstimator:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    class FakeTrial:
        def report(self, value, step):
            return

        def should_prune(self):
            return False

        def set_user_attr(self, name, value):
            return

    X = pd.DataFrame({"first": np.arange(20), "second": np.arange(20)})
    y = pd.Series([0, 1] * 10, name="target")

    catboost = CatBoostClassifier().set_runtime(runtime(DeviceIntent.CPU))
    monkeypatch.setattr(catboost, "_assert_available", lambda: None)
    monkeypatch.setattr(catboost, "optuna_args", lambda trial: {})
    cat_args = []
    monkeypatch.setattr(
        catboost,
        "_fit_target_models",
        lambda X, y, kwargs: (cat_args.append(dict(kwargs)) or FakeEstimator()),
    )
    catboost._tuning_thread_count = 1
    catboost.optuna_objective(
        X, y, None, ClassifierScorer.Accuracy
    )(FakeTrial())

    xgboost = XGBoostClassifier().set_runtime(runtime(DeviceIntent.CPU))
    monkeypatch.setattr(xgboost, "_assert_available", lambda: None)
    monkeypatch.setattr(xgboost, "optuna_args", lambda trial: {})
    xgb_args = []
    monkeypatch.setattr(
        xgboost,
        "_fit_models",
        lambda X, y, args: (xgb_args.append(dict(args)) or FakeEstimator()),
    )
    xgboost.optuna_objective(
        X, y, None, ClassifierScorer.Accuracy
    )(FakeTrial())

    assert cat_args and all(args["thread_count"] == 1 for args in cat_args)
    assert xgb_args and all(args["n_jobs"] == 1 for args in xgb_args)
    assert xgboost.default_args["n_jobs"] == -1


@pytest.mark.fast
def test_program_options_runtime_includes_recorded_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def new_policy(_):
        calls.append(1)
        return runtime(torch_cuda=True)

    monkeypatch.setattr(cli, "get_runtime", new_policy)
    options = object.__new__(ProgramOptions)
    options.device = DeviceIntent.Auto

    options.set_runtime_workload(700, 30)
    first = options.runtime
    first.record_cpu_fallback(RuntimeComponent.KNN, "cuda_out_of_memory")

    assert options.runtime.workload is not None
    assert options.runtime.workload.n_samples == 700
    assert options.runtime.workload.n_features == 30
    assert options.runtime.device_for(RuntimeComponent.KNN) == "cpu"
    assert options.runtime.decision_for(RuntimeComponent.KNN).reason == "cuda_out_of_memory"
    assert len(calls) == 1


@pytest.mark.fast
def test_model_device_uses_actual_selected_input_and_retries_cuda_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeXGB:
        shortname = "fake-xgb"
        longname = "Fake XGBoost"
        runtime_component = RuntimeComponent.XGBoost

        def __init__(self) -> None:
            self.per_target_tuning_scores = {}
            self.tuned_model = None
            self.model = None

        def set_runtime(self, policy):
            self.runtime = policy
            return self

    n_rows = 3_000
    columns = [f"f{i}" for i in range(100)]
    train = PreparedData(
        X=pd.DataFrame(np.zeros((n_rows, len(columns))), columns=columns),
        y=pd.Series(np.arange(n_rows) % 2, name="target"),
        groups=None,
        is_classification=True,
        validate=False,
    )
    test = PreparedData(
        X=pd.DataFrame(np.zeros((10, len(columns))), columns=columns),
        y=pd.Series(np.arange(10) % 2, name="target"),
        groups=None,
        is_classification=True,
        validate=False,
    )
    calls: list[tuple[tuple[int, int], str]] = []

    def fake_run(model, X_train, **_):
        device = model.runtime.device_for(RuntimeComponent.XGBoost)
        calls.append((X_train.shape, device))
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        study = SimpleNamespace(best_params={}, best_value=1.0)
        scores = pd.DataFrame(
            {
                "metric": ["accuracy"],
                "trainset": [1.0],
                "holdout": [1.0],
                "5-fold": [1.0],
            }
        )
        evaluation = (
            scores,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            None,
            None,
            None,
        )
        model.tuned_model = SimpleNamespace(fitted=True)
        return study, evaluation

    monkeypatch.setattr(
        hypertune_module, "_tune_and_evaluate_model", fake_run
    )
    monkeypatch.setattr(
        hypertune_module, "release_accelerator_memory", lambda: None
    )
    options = SimpleNamespace(
        models=[FakeXGB],
        htune_cls_metric=ClassifierScorer.Accuracy,
        htune_reg_metric=None,
        htune_trials=1,
        runtime=runtime(DeviceIntent.Auto, xgboost_cuda=True),
        seed=42,
        _runtime_model_audit=[],
    )

    with pytest.warns(UserWarning, match="Retrying this model on CPU once"):
        evaluated = evaluate_tuned(
            prepared=train,
            prep_train=train,
            prep_test=test,
            assoc_filtered=SimpleNamespace(selected=["f0"]),
            pred_filtered=None,
            model_selected=ModelSelected(
                embed_selected=None, wrap_selected=None
            ),
            options=options,
        )

    assert len(evaluated.results) == 2
    assert calls == [
        ((3_000, 100), "cuda"),
        ((3_000, 100), "cpu"),
        ((3_000, 1), "cpu"),
    ]
    assoc_start = next(
        record
        for record in options._runtime_model_audit
        if record["selection"] == "assoc" and record["stage"] == "started"
    )
    assert assoc_start["n_features"] == 1
    assert assoc_start["reason"] == "auto_workload_below_threshold"
    assert all(
        result.model.tuned_model is not None for result in evaluated.results
    )


@pytest.mark.fast
def test_accelerator_model_snapshot_survives_release_and_reload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_xgboost = pytest.importorskip("xgboost").XGBClassifier
    n_rows = 2_000
    columns = [f"f{i}" for i in range(100)]
    train = PreparedData(
        X=pd.DataFrame(np.zeros((n_rows, len(columns))), columns=columns),
        y=pd.Series(np.arange(n_rows) % 2, name="target"),
        groups=None,
        is_classification=True,
        validate=False,
    )
    test = PreparedData(
        X=pd.DataFrame(np.zeros((10, len(columns))), columns=columns),
        y=pd.Series(np.arange(10) % 2, name="target"),
        groups=None,
        is_classification=True,
        validate=False,
    )

    def fake_run(model, X_train, prep_train, **_):
        fitted = native_xgboost(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            verbosity=0,
        )
        fitted.fit(X_train.iloc[:40], prep_train.y.iloc[:40])
        model.tuned_model = fitted
        study = SimpleNamespace(best_params={}, best_value=1.0)
        scores = pd.DataFrame(
            {
                "metric": ["accuracy"],
                "trainset": [1.0],
                "holdout": [1.0],
                "5-fold": [1.0],
            }
        )
        return (
            study,
            (
                scores,
                pd.Series(np.zeros(len(X_train)), index=X_train.index),
                pd.Series(np.zeros(len(test.X)), index=test.X.index),
                None,
                None,
                None,
            ),
        )

    monkeypatch.setattr(
        hypertune_module, "_tune_and_evaluate_model", fake_run
    )
    monkeypatch.setattr(
        hypertune_module, "release_accelerator_memory", lambda: None
    )
    options = SimpleNamespace(
        models=[XGBoostClassifier],
        htune_cls_metric=ClassifierScorer.Accuracy,
        htune_reg_metric=None,
        htune_trials=1,
        runtime=runtime(DeviceIntent.Auto, xgboost_cuda=True),
        seed=42,
        _runtime_model_audit=[],
    )

    evaluated = evaluate_tuned(
        prepared=train,
        prep_train=train,
        prep_test=test,
        assoc_filtered=None,
        pred_filtered=None,
        model_selected=ModelSelected(
            embed_selected=None, wrap_selected=None
        ),
        options=options,
    )
    result = evaluated.results[0]

    assert result.model.tuned_model is None
    assert result._model_snapshot is not None
    evaluated.save(tmp_path, fold_idx=None)
    loaded = EvaluationResults.load(tmp_path)
    restored = loaded.results[0]
    assert restored._model_snapshot is None
    assert restored.model.tuned_model is not None
    predictions = restored.model.tuned_model.predict(train.X.iloc[:3])
    assert predictions.shape == (3,)


@pytest.mark.fast
def test_accelerator_release_clears_trainer_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[bool] = []
    model = SimpleNamespace(
        shortname="fake-gandalf",
        runtime=runtime(DeviceIntent.Auto, torch_cuda=True).for_task(100, 10),
        model=object(),
        tuned_model=object(),
        trainer=object(),
        tuned_trainer=object(),
        _cleanup_after_fold=lambda: cleanup_calls.append(True),
    )
    monkeypatch.setattr(
        hypertune_module, "release_accelerator_memory", lambda: None
    )

    hypertune_module._release_fitted_model_state(
        model, RuntimeComponent.Gandalf, force=True
    )

    assert model.model is None
    assert model.tuned_model is None
    assert model.trainer is None
    assert model.tuned_trainer is None
    assert cleanup_calls == [True]


@pytest.mark.fast
def test_strict_cuda_xgboost_snapshot_restores_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame(
        np.arange(80, dtype=float).reshape(40, 2), columns=["first", "second"]
    )
    y = pd.Series(np.arange(40) % 2, name="target")
    model = XGBoostClassifier(
        model_args={"n_estimators": 2, "max_depth": 1, "n_jobs": 1}
    )
    model.set_runtime(runtime(DeviceIntent.CPU))
    model.fit(X, y)
    expected = np.asarray(model.predict(X))
    model.set_runtime(
        runtime(DeviceIntent.CUDA, xgboost_cuda=True).for_task(
            len(X), X.shape[1]
        )
    )
    payload = jsonpickle.encode(model, unpicklable=True)
    monkeypatch.setattr(hardware, "_xgboost_cuda_available", lambda: False)

    restored = jsonpickle.decode(payload)

    assert restored.runtime.intent is DeviceIntent.CPU
    np.testing.assert_array_equal(restored.predict(X), expected)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (KANEstimator, {"num_classes": 2}),
        (TabPFNClassifierV3, {}),
    ],
)
def test_strict_cuda_torch_snapshot_restores_with_cpu_runtime(
    model_cls,
    kwargs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_cls(**kwargs)
    model.set_runtime(runtime(DeviceIntent.CUDA, torch_cuda=True))
    payload = jsonpickle.encode(model, unpicklable=True)
    monkeypatch.setattr(hardware, "_torch_capabilities", lambda: (False, False))

    restored = jsonpickle.decode(payload)

    assert restored.runtime.intent is DeviceIntent.CPU
    assert restored.runtime.device_for(restored.runtime_component) == "cpu"


@pytest.mark.fast
def test_adaptive_error_oof_restarts_complete_task_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOOFModel:
        runtime_component = RuntimeComponent.MLP
        calls: list[str] = []

        def __init__(self) -> None:
            self.tuned_model = None
            self.model = None

        def set_runtime(self, policy):
            self.runtime = policy
            return self

        def refit_tuned(self, X, y, g=None, tuned_args=None) -> None:
            device = self.runtime.device_for(RuntimeComponent.MLP)
            self.calls.append(device)
            if device == "cuda" and self.calls.count("cuda") == 1:
                raise RuntimeError("CUDA out of memory")
            self.tuned_model = self

        def tuned_predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile(np.asarray([[0.75, 0.25]]), (len(X), 1))

        def _cleanup_after_fold(self) -> None:
            return

    monkeypatch.setattr(
        hypertune_module, "release_accelerator_memory", lambda: None
    )
    monkeypatch.setattr(
        "df_analyze.analysis.adaptive_error.oof.release_accelerator_memory",
        lambda: None,
    )
    X = pd.DataFrame(
        {"f0": np.arange(40, dtype=float), "f1": np.arange(40, dtype=float)}
    )
    y = pd.Series([0, 1] * 20, name="target")
    source = SimpleNamespace(
        runtime=runtime(DeviceIntent.Auto, torch_cuda=True),
        model_args=None,
    )
    result = SimpleNamespace(
        model_cls=FakeOOFModel,
        model=source,
        params={},
        selection="filter",
    )
    options = SimpleNamespace(
        _runtime_model_audit=[],
        _runtime_current_fold=2,
    )

    with pytest.warns(UserWarning, match="Restarting this complete"):
        oof, probabilities = build_oof_for_result(
            result,
            X,
            y,
            groups=None,
            n_folds=2,
            options=options,
        )

    assert FakeOOFModel.calls == ["cuda", "cpu", "cpu"]
    assert len(oof) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert [
        (record["attempt"], record["stage"], record["resolved"])
        for record in options._runtime_model_audit
    ] == [(1, "failed", "cuda"), (2, "completed", "cpu")]


@pytest.mark.fast
def test_adaptive_error_knn_uses_actual_oof_fold_workload_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKNN:
        shortname = "knn"
        runtime_component = RuntimeComponent.KNN

    seen = []

    def fit_fold(result, X_tr, y_tr, g_tr, X_val, policy):
        seen.append(policy.decision_for(RuntimeComponent.KNN))
        return {
            "y_pred": np.zeros(len(X_val), dtype=int),
            "proba": np.tile(np.asarray([[0.75, 0.25]]), (len(X_val), 1)),
            "conf": np.full(len(X_val), 0.5),
            "knn_vote": None,
            "knn_dist_weighted": None,
            "knn_min_dist": None,
            "tree_vote_agreement": None,
            "tree_leaf_support": None,
        }

    monkeypatch.setattr(adaptive_error_oof, "_fit_oof_fold", fit_fold)
    n_rows = 1_000
    X = pd.DataFrame(np.zeros((n_rows, 100)))
    y = pd.Series([0, 1] * (n_rows // 2), name="target")
    options = SimpleNamespace(_runtime_model_audit=[], _runtime_current_fold=3)
    result = SimpleNamespace(
        model_cls=FakeKNN,
        model=SimpleNamespace(
            shortname="knn",
            runtime=runtime(DeviceIntent.Auto, torch_cuda=True),
        ),
        params={},
        selection="filter",
    )

    oof, probabilities = build_oof_for_result(
        result,
        X,
        y,
        groups=None,
        n_folds=5,
        options=options,
    )

    assert len(oof) == n_rows
    assert probabilities.shape == (n_rows, 2)
    assert seen and all(decision.resolved == "cpu" for decision in seen)
    assert all(decision.n_samples == 800 for decision in seen)
    assert all(decision.n_queries == 200 for decision in seen)
    record = options._runtime_model_audit[-1]
    assert record["fold"] == 3
    assert record["stage"] == "completed"
    assert record["resolved"] == "cpu"
    assert record["n_samples"] == 800
    assert record["n_queries"] == 200
    assert record["work_items"] == 16_000_000


@pytest.mark.fast
def test_adaptive_error_holdout_restarts_complete_task_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHoldoutModel:
        shortname = "fake-holdout"
        runtime_component = RuntimeComponent.MLP

        def __init__(self) -> None:
            self.runtime = runtime(DeviceIntent.Auto, torch_cuda=True)
            self.model = object()
            self.tuned_model = object()
            self.devices: list[str] = []

        def set_runtime(self, policy):
            self.runtime = policy
            return self

        def _cleanup_after_fold(self) -> None:
            return

    model = FakeHoldoutModel()
    result = SimpleNamespace(
        model=model,
        model_cls=FakeHoldoutModel,
        params={},
        selection="filter",
    )
    sentinel = object()

    def attempt(**_kwargs):
        device = model.runtime.device_for(RuntimeComponent.MLP)
        model.devices.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory during predict_proba")
        return sentinel

    monkeypatch.setattr(
        adaptive_error_test_stage,
        "_evaluate_test_stage_attempt",
        attempt,
    )
    monkeypatch.setattr(
        adaptive_error_test_stage,
        "release_accelerator_memory",
        lambda: None,
    )
    X_train = pd.DataFrame(np.zeros((20, 3)))
    X_test = pd.DataFrame(np.zeros((5, 3)))

    with pytest.warns(UserWarning, match="Restarting this complete"):
        evaluated = adaptive_error_test_stage._evaluate_test_stage(
            result=result,
            X_train=X_train,
            X_test=X_test,
            y_train=pd.Series([0, 1] * 10),
            y_test=pd.Series([0, 1, 0, 1, 0]),
            labels_map=None,
            aer=None,
            calibrator=None,
            selected_conf_metric="proba_margin",
            selected_conf_params={},
            options=SimpleNamespace(),
            no_preds=True,
            m_preds=None,
            m_tables=None,
            m_plots=None,
            m_meta=None,
        )

    assert evaluated is sentinel
    assert model.devices == ["cuda", "cpu"]


@pytest.mark.fast
def test_device_decisions_are_auditable() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.KNN,),
        regressors=(),
        runtime=runtime(torch_cuda=True).with_workload(400, 20),
    )

    decision = _main._device_decisions(options)["knn"]

    assert decision["resolved"] == "cpu"
    assert decision["reason"] == "auto_workload_below_threshold"
    assert decision["work_items"] == 3_200_000
    assert decision["threshold"] == 20_000_000


@pytest.mark.fast
def test_runtime_audit_keeps_fold_and_ec_backends() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.KNN,),
        regressors=(),
        runtime=runtime(DeviceIntent.CUDA, torch_cuda=True).with_workload(500, 20),
    )
    options.runtime.record_cpu_fallback(
        RuntimeComponent.KNN, "cuda_out_of_memory"
    )

    snapshot = _main._runtime_snapshot(options, fold_idx=2)
    _main._record_ec_backends(
        options,
        SimpleNamespace(
            metadata={
                "ec_backends": [
                    {
                        "ec_backend_requested": "cuda",
                        "ec_backend_resolved": "numpy",
                        "ec_backend_reason": "cuda_out_of_memory",
                        "ec_backend_work_items": 1_500_000,
                    }
                ],
                "skipped_configurations": [
                    {
                        "target": "outcome",
                        "model": "mlp",
                        "reason": "fold failed",
                    }
                ],
            }
        ),
    )

    assert snapshot["fold"] == 2
    assert snapshot["resolved_devices"]["knn"] == "cpu"
    assert snapshot["device_decisions"]["knn"]["reason"] == "cuda_out_of_memory"
    assert options._ec_backends[0]["ec_backend_resolved"] == "numpy"
    assert options._partial_failures == [
        {
            "component": "error_consistency",
            "reason": "fold failed",
            "target": "outcome",
            "model": "mlp",
        }
    ]


@pytest.mark.fast
def test_main_reports_completed_with_partial_failures(monkeypatch) -> None:
    options = SimpleNamespace(to_json=lambda: None)
    statuses = []

    def run_with_partial_failure(run_options) -> None:
        run_options._model_failures = []
        run_options._partial_failures = [{"component": "risk_stability"}]

    monkeypatch.setattr(_main, "get_options", lambda: options)
    monkeypatch.setattr(_main, "_run", run_with_partial_failure)
    monkeypatch.setattr(
        _main,
        "_write_run_timing",
        lambda run_options, started_at, started_s, status, error=None: statuses.append(
            status
        ),
    )

    _main.main()

    assert statuses == ["completed_with_failures"]


@pytest.mark.fast
def test_main_fails_when_all_predictive_models_fail(monkeypatch) -> None:
    options = SimpleNamespace(to_json=lambda: None)
    timings = []

    def run_with_model_failure(run_options) -> None:
        run_options._model_failures = [
            {"model": "tabpfn-v3", "reason": "license token missing"}
        ]
        run_options._model_successes = [{"model": "dummy"}]
        run_options._partial_failures = []

    monkeypatch.setattr(_main, "get_options", lambda: options)
    monkeypatch.setattr(_main, "_run", run_with_model_failure)
    monkeypatch.setattr(
        _main,
        "_write_run_timing",
        lambda run_options, started_at, started_s, status, error=None: timings.append(
            (status, error)
        ),
    )

    with pytest.raises(RuntimeError, match="All requested predictive models failed"):
        _main.main()

    assert len(timings) == 1
    assert timings[0][0] == "failed"
    assert isinstance(timings[0][1], RuntimeError)


@pytest.mark.fast
def test_main_allows_one_predictive_model_to_succeed(monkeypatch) -> None:
    options = SimpleNamespace(to_json=lambda: None)
    statuses = []

    def run_with_partial_model_failure(run_options) -> None:
        run_options._model_failures = [
            {"model": "tabpfn-v3", "reason": "license token missing"}
        ]
        run_options._model_successes = [
            {"model": "dummy"},
            {"model": "xgb"},
        ]
        run_options._partial_failures = []

    monkeypatch.setattr(_main, "get_options", lambda: options)
    monkeypatch.setattr(_main, "_run", run_with_partial_model_failure)
    monkeypatch.setattr(
        _main,
        "_write_run_timing",
        lambda run_options, started_at, started_s, status, error=None: statuses.append(
            status
        ),
    )

    _main.main()

    assert statuses == ["completed_with_failures"]


@pytest.mark.fast
def test_knn_oom_requests_complete_model_retry() -> None:
    base = runtime(DeviceIntent.CUDA, torch_cuda=True)
    policy = base.with_workload(5_000, 100)
    model = TorchKNNClassifier(
        n_neighbors=1,
        metric="minkowski",
        p=1,
        algorithm="brute",
        device="cuda",
        runtime=policy,
    )
    model._X_cpu = np.asarray([[0.0], [1.0]], dtype=np.float32)
    model._y_cpu = np.asarray([[0], [1]])

    with pytest.raises(RuntimeError, match="complete model task"):
        model._activate_cpu("training data did not fit in GPU memory")

    decision = base.with_workload(5_000, 100).decision_for(RuntimeComponent.KNN)
    assert decision.resolved == "cuda"
    assert decision.reason == "explicit_cuda_available"
    assert model._cpu_model is None
    assert base.with_workload(5_001, 100).device_for(RuntimeComponent.KNN) == "cuda"


@pytest.mark.fast
def test_cuda_knn_rejects_unknown_arguments() -> None:
    with pytest.raises(TypeError, match="unknown_option"):
        TorchKNNClassifier(device="cpu", unknown_option=True)


@pytest.mark.fast
def test_torch_knn_classifier_on_cpu() -> None:
    X = pd.DataFrame({"x0": [0.0, 0.0, 1.0, 1.0], "x1": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    model = TorchKNNClassifier(n_neighbors=1, metric="l2", device="cpu").fit(X, y)

    pred = model.predict(pd.DataFrame({"x0": [0.1, 0.9], "x1": [0.0, 1.0]}))
    proba = model.predict_proba(pd.DataFrame({"x0": [0.1, 0.9], "x1": [0.0, 1.0]}))

    assert pred.tolist() == [0, 1]
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.fast
def test_torch_knn_regressor_distance_weights() -> None:
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0])
    model = TorchKNNRegressor(
        n_neighbors=2, weights="distance", metric="l2", device="cpu"
    ).fit(X, y)

    assert model.predict(pd.DataFrame({"x": [0.0, 1.5]})).tolist() == pytest.approx(
        [0.0, 1.5]
    )


@pytest.mark.fast
@pytest.mark.parametrize("metric", ["l1", "l2", "cosine", "correlation"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_torch_knn_classifier_matches_sklearn(metric: str, weights: str) -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(80, 7))
    X_test = rng.normal(size=(25, 7))
    y_train = rng.integers(0, 3, size=80)
    expected = KNeighborsClassifier(
        n_neighbors=5, metric=metric, weights=weights, algorithm="brute"
    ).fit(X_train, y_train)
    actual = TorchKNNClassifier(
        n_neighbors=5, metric=metric, weights=weights, device="cpu"
    ).fit(X_train, y_train)

    assert np.array_equal(actual.predict(X_test), expected.predict(X_test))
    assert np.allclose(
        actual.predict_proba(X_test), expected.predict_proba(X_test), atol=2e-5
    )


@pytest.mark.fast
@pytest.mark.parametrize("metric", ["l1", "l2", "cosine", "correlation"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_torch_knn_regressor_matches_sklearn(metric: str, weights: str) -> None:
    rng = np.random.default_rng(84)
    X_train = rng.normal(size=(80, 7))
    X_test = rng.normal(size=(25, 7))
    y_train = rng.normal(size=80)
    expected = KNeighborsRegressor(
        n_neighbors=5, metric=metric, weights=weights, algorithm="brute"
    ).fit(X_train, y_train)
    actual = TorchKNNRegressor(
        n_neighbors=5, metric=metric, weights=weights, device="cpu"
    ).fit(X_train, y_train)

    assert np.allclose(actual.predict(X_test), expected.predict(X_test), atol=2e-5)


@pytest.mark.fast
def test_torch_knn_rejects_more_neighbors_than_training_samples() -> None:
    X = np.asarray([[0.0], [1.0]])
    y = np.asarray([0, 1])
    model = TorchKNNClassifier(n_neighbors=3, device="cpu").fit(X, y)

    with pytest.raises(ValueError, match="n_neighbors <= n_samples_fit"):
        model.predict(np.asarray([[0.5]]))


@pytest.mark.fast
def test_torch_knn_reduces_batch_before_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.arange(12) % 2
    model = TorchKNNClassifier(
        n_neighbors=1, metric="l2", device="cpu", batch_size=8
    ).fit(X, y)
    original_distances = model._distances
    attempted_batch_sizes: list[int] = []

    def limited_distances(query):
        attempted_batch_sizes.append(int(query.shape[0]))
        if query.shape[0] > 2:
            raise RuntimeError("CUDA out of memory")
        return original_distances(query)

    monkeypatch.setattr(model, "_safe_batch_size", lambda: 8)
    monkeypatch.setattr(model, "_distances", limited_distances)

    prediction = model.predict(X[:5])

    assert prediction.tolist() == y[:5].tolist()
    assert attempted_batch_sizes[:3] == [5, 2, 2]
    assert model.batch_size == 2


@pytest.mark.fast
def test_torch_knn_correlation_matches_sklearn_for_constant_rows() -> None:
    X = np.asarray([[0.0], [1.0], [2.0]])
    y = np.asarray([0, 1, 1])
    query = np.asarray([[0.5]])
    expected = KNeighborsClassifier(n_neighbors=2, metric="correlation").fit(X, y)
    actual = TorchKNNClassifier(
        n_neighbors=2, metric="correlation", device="cpu"
    ).fit(X, y)

    expected_distances, expected_indices = expected.kneighbors(query)
    actual_distances, actual_indices = actual.kneighbors(query)
    assert np.array_equal(actual_indices, expected_indices)
    assert np.array_equal(np.isnan(actual_distances), np.isnan(expected_distances))
    assert np.array_equal(actual.predict(query), expected.predict(query))


@pytest.mark.fast
def test_torch_knn_validates_fit_and_predict_shapes() -> None:
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        TorchKNNRegressor(device="cpu").fit(np.zeros((3, 2)), np.zeros(2))

    model = TorchKNNRegressor(n_neighbors=1, device="cpu").fit(
        np.zeros((3, 2)), np.zeros(3)
    )
    with pytest.raises(ValueError, match="expecting 2 features"):
        model.predict(np.zeros((1, 3)))


@pytest.mark.fast
def test_torch_knn_cuda_matches_sklearn_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(126)
    X_train = rng.normal(size=(200, 12))
    X_test = rng.normal(size=(50, 12))
    y_train = rng.integers(0, 4, size=200)
    expected = KNeighborsClassifier(
        n_neighbors=7, metric="cosine", weights="distance", algorithm="brute"
    ).fit(X_train, y_train)
    actual = TorchKNNClassifier(
        n_neighbors=7, metric="cosine", weights="distance", device="cuda"
    ).fit(X_train, y_train)

    assert np.array_equal(actual.predict(X_test), expected.predict(X_test))
    assert np.allclose(
        actual.predict_proba(X_test), expected.predict_proba(X_test), atol=5e-4
    )


@pytest.mark.fast
def test_embedding_model_uses_runtime_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model:
        def __init__(self) -> None:
            self.device = None
            self.evaluating = False

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.evaluating = True
            return self

    model = Model()
    processor = object()
    monkeypatch.setattr(
        embed, "load_nlp_intfloat_ml_model_offline", lambda: (model, processor)
    )

    loaded, _ = embed.get_model(
        EmbeddingModality.NLP, runtime=runtime(DeviceIntent.CPU)
    )

    assert loaded.device == "cpu"
    assert loaded.evaluating


@pytest.mark.fast
def test_embedding_download_checks_tokenizer_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model_file = tmp_path / "model"
    tokenizer_file = tmp_path / "tokenizer"
    model_file.touch()
    called = []
    monkeypatch.setattr(embedding_download, "INTFLOAT_MODEL_FILES", [model_file])
    monkeypatch.setattr(
        embedding_download, "INTFLOAT_TOKENIZER_FILES", [tokenizer_file]
    )
    monkeypatch.setattr(
        embedding_download, "download_nlp_intfloat_ml_model", lambda: called.append(True)
    )

    embedding_download.download_models(nlp=True, vision=False)

    assert called == [True]


@pytest.mark.fast
def test_embedding_force_download_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        embedding_download,
        "download_nlp_intfloat_ml_model",
        lambda force=False: calls.append(force),
    )

    embedding_download.download_models(nlp=True, vision=False, force=True)

    assert calls == [True]


@pytest.mark.fast
def test_resolved_devices_only_lists_selected_models() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.CatBoost, DfAnalyzeClassifier.KNN),
        regressors=(),
        runtime=runtime(
            DeviceIntent.CUDA, torch_cuda=True, catboost_cuda=True
        ),
    )

    resolved = _main._resolved_devices(options)

    assert resolved["catboost"] == "cuda"
    assert resolved["knn"] == "cuda"
    assert "embedding" not in resolved


@pytest.mark.fast
def test_xgboost_uses_one_canonical_audit_key() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.XGBoost,),
        regressors=(),
        runtime=runtime(DeviceIntent.CUDA, xgboost_cuda=True),
        _runtime_model_audit=[
            {
                "model": "xgb",
                "selection": "none",
                "stage": "completed",
                "resolved": "cuda",
                "requested": "cuda",
                "reason": "explicit_cuda_available",
                "n_samples": 100,
                "n_features": 10,
                "n_queries": 25,
            }
        ],
    )

    assert _main._resolved_devices(options)["xgboost"] == "cuda"
    assert "xgb" not in _main._resolved_devices(options)
    assert _main._device_decisions(options)["xgboost"]["n_queries"] == 25


@pytest.mark.fast
def test_actual_device_audit_reports_mixed_model_tasks() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.XGBoost,),
        regressors=(),
        runtime=runtime(DeviceIntent.Auto, xgboost_cuda=True),
        _runtime_model_audit=[
            {
                "model": "xgb",
                "selection": "none",
                "stage": "completed",
                "resolved": "cuda",
                "requested": "auto",
                "reason": "auto_workload_at_or_above_threshold",
            },
            {
                "model": "xgb",
                "selection": "filter",
                "stage": "completed",
                "resolved": "cpu",
                "requested": "auto",
                "reason": "auto_workload_below_threshold",
            },
        ],
    )

    assert _main._resolved_devices(options)["xgboost"] == "mixed"
    decision = _main._device_decisions(options)["xgboost"]
    assert decision["resolved"] == "mixed"
    assert decision["devices"] == ["cpu", "cuda"]
    assert decision["tasks"] == 2


@pytest.mark.fast
def test_resolved_devices_includes_knn_wrapper() -> None:
    options = SimpleNamespace(
        is_classification=True,
        classifiers=(DfAnalyzeClassifier.CatBoost,),
        regressors=(),
        wrapper_select=WrapperSelection.StepUp,
        wrapper_model=WrapperSelectionModel.KNN,
        runtime=runtime(torch_cuda=True).with_workload(10_000, 100),
    )

    assert _main._resolved_devices(options)["knn"] == "cuda"

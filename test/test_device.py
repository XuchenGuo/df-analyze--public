from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torch.nn import Dropout, Identity

from df_analyze import _main
from df_analyze.cli import cli
from df_analyze.cli.cli import ProgramOptions, make_parser
from df_analyze.embedding import download as embedding_download
from df_analyze.embedding import embed
from df_analyze.embedding.cli import EmbeddingModality
from df_analyze.enumerables import (
    DfAnalyzeClassifier,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.models.catboost import CatBoostClassifier
from df_analyze.models.gandalf import GandalfEstimator, _dropout_layer
from df_analyze.models.kan import KANEstimator
from df_analyze.models.knn import (
    KNNClassifier,
    TorchKNNClassifier,
    TorchKNNRegressor,
)
from df_analyze.models.mlp import MLPEstimator
from df_analyze.models.xgboost import XGBoostClassifier
from df_analyze.runtime import hardware
from df_analyze.runtime.hardware import (
    DeviceIntent,
    HardwareCapabilities,
    RuntimeComponent,
    RuntimePolicy,
)


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
def test_explicit_cuda_warns_once_per_backend() -> None:
    hardware._warn_cuda_fallback.cache_clear()
    policy = runtime(DeviceIntent.CUDA)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert policy.device_for(RuntimeComponent.MLP) == "cpu"
        assert policy.device_for(RuntimeComponent.KAN) == "cpu"
        assert policy.device_for(RuntimeComponent.KNN) == "cpu"

    assert len(caught) == 1
    assert "torch backend" in str(caught[0].message)


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
    assert decision.threshold == 500_000


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
    ).with_workload(5_000, 100)

    assert policy.device_for(RuntimeComponent.KNN) == "cuda"
    assert policy.device_for(RuntimeComponent.CatBoost) == "cuda"
    assert policy.device_for(RuntimeComponent.XGBoost) == "cuda"
    decision = policy.decision_for(RuntimeComponent.CatBoost)
    assert decision.reason == "auto_workload_at_or_above_threshold"
    assert decision.work_metric == "matrix_elements"
    assert decision.work_items == 500_000
    assert decision.threshold == 500_000

    below = runtime(catboost_cuda=True, xgboost_cuda=True).with_workload(4_999, 100)
    assert below.device_for(RuntimeComponent.CatBoost) == "cpu"
    assert below.device_for(RuntimeComponent.XGBoost) == "cpu"


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
    assert decision["threshold"] == 5_000_000


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
def test_knn_oom_fallback_updates_runtime_decision() -> None:
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

    with pytest.warns(UserWarning, match="continuing with sklearn KNN on CPU"):
        model._activate_cpu("training data did not fit in GPU memory")

    decision = base.with_workload(5_000, 100).decision_for(RuntimeComponent.KNN)
    assert decision.resolved == "cpu"
    assert decision.reason == "cuda_out_of_memory"
    assert model._cpu_model.p == 1
    assert model._cpu_model.algorithm == "brute"
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

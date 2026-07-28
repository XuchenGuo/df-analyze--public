from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import Literal, Optional
from warnings import warn


class DeviceIntent(Enum):
    Auto = "auto"
    CPU = "cpu"
    CUDA = "cuda"

    @classmethod
    def parse(cls, value: str) -> str:
        return cls(str(value).lower()).value

    @classmethod
    def from_arg(cls, value: str | DeviceIntent) -> DeviceIntent:
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())

    @classmethod
    def choices(cls) -> list[str]:
        return [item.value for item in cls]


class RuntimeComponent(Enum):
    CatBoost = "catboost"
    XGBoost = "xgboost"
    KNN = "knn"
    TabPFN = "tabpfn"
    MLP = "mlp"
    KAN = "kan"
    Gandalf = "gandalf"
    Embedding = "embedding"
    LightGBM = "lightgbm"
    Sklearn = "sklearn"
    Preprocessing = "preprocessing"
    Selection = "selection"
    Univariate = "univariate"


CUDA_BACKENDS = {
    RuntimeComponent.CatBoost: "catboost",
    RuntimeComponent.XGBoost: "xgboost",
    RuntimeComponent.KNN: "torch",
    RuntimeComponent.TabPFN: "torch",
    RuntimeComponent.MLP: "torch",
    RuntimeComponent.KAN: "torch",
    RuntimeComponent.Gandalf: "torch",
    RuntimeComponent.Embedding: "torch",
}
MPS_COMPONENTS = {RuntimeComponent.Gandalf}


@dataclass(frozen=True)
class RuntimeWorkload:
    n_samples: int
    n_features: int

    def __post_init__(self) -> None:
        if self.n_samples < 0 or self.n_features < 0:
            raise ValueError("Runtime workload dimensions must be non-negative.")

    @property
    def matrix_elements(self) -> int:
        return int(self.n_samples) * int(self.n_features)

    @property
    def pairwise_elements(self) -> int:
        return int(self.n_samples) * int(self.n_samples) * int(self.n_features)


@dataclass(frozen=True)
class AutoCudaRule:
    work_metric: Literal["matrix_elements", "pairwise_elements"]
    min_work_items: int

    def work_items(self, workload: RuntimeWorkload) -> int:
        return int(getattr(workload, self.work_metric))


# These conservative crossover points prevent small traditional-learner jobs from
# paying CUDA initialization and transfer costs. Neural and embedding components
# remain accelerator-preferred because their backend is intrinsically compute-heavy.
AUTO_CUDA_RULES = {
    RuntimeComponent.KNN: AutoCudaRule("pairwise_elements", 5_000_000),
    RuntimeComponent.CatBoost: AutoCudaRule("matrix_elements", 500_000),
    RuntimeComponent.XGBoost: AutoCudaRule("matrix_elements", 500_000),
}


@dataclass(frozen=True)
class DeviceDecision:
    requested: str
    resolved: str
    reason: str
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    work_metric: Optional[str] = None
    work_items: Optional[int] = None
    threshold: Optional[int] = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


@dataclass
class _RuntimeState:
    cpu_fallbacks: dict[
        tuple[RuntimeComponent, Optional[RuntimeWorkload]], str
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareCapabilities:
    torch_cuda: bool | None
    torch_mps: bool | None
    catboost_cuda: bool | None
    xgboost_cuda: bool | None

    def torch_cuda_available(self) -> bool:
        return _torch_capabilities()[0] if self.torch_cuda is None else self.torch_cuda

    def torch_mps_available(self) -> bool:
        return _torch_capabilities()[1] if self.torch_mps is None else self.torch_mps

    def catboost_cuda_available(self) -> bool:
        return _catboost_cuda_available() if self.catboost_cuda is None else self.catboost_cuda

    def xgboost_cuda_available(self) -> bool:
        return _xgboost_cuda_available() if self.xgboost_cuda is None else self.xgboost_cuda

    def cuda_available_for(self, component: RuntimeComponent) -> bool:
        backend = CUDA_BACKENDS.get(component)
        if backend == "torch":
            return self.torch_cuda_available()
        if backend == "catboost":
            return self.catboost_cuda_available()
        if backend == "xgboost":
            return self.xgboost_cuda_available()
        return False


CPU_CAPABILITIES = HardwareCapabilities(False, False, False, False)
LAZY_CAPABILITIES = HardwareCapabilities(None, None, None, None)


@lru_cache(maxsize=1)
def _torch_capabilities() -> tuple[bool, bool]:
    try:
        import torch
    except Exception:
        return False, False

    cuda = False
    try:
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            cuda = True
    except Exception:
        pass

    mps = False
    try:
        backend = getattr(torch.backends, "mps", None)
        mps = backend is not None and bool(backend.is_available())
    except Exception:
        pass
    return cuda, mps


@lru_cache(maxsize=1)
def _catboost_cuda_available() -> bool:
    try:
        from catboost import CatBoostClassifier
        from catboost.utils import get_gpu_device_count

        if get_gpu_device_count() < 1:
            return False
        model = CatBoostClassifier(
            iterations=1,
            depth=2,
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 2,
            [0, 0, 1, 1] * 2,
        )
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _xgboost_cuda_available() -> bool:
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=1,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 2,
            [0, 0, 1, 1] * 2,
        )
        config = model.get_booster().save_config().replace(" ", "").lower()
        return '"device":"cuda' in config
    except Exception:
        return False


@lru_cache(maxsize=None)
def _warn_cuda_fallback(backend: str) -> None:
    warn(
        f"CUDA was requested, but the {backend} backend cannot use CUDA on this "
        "machine. Falling back to CPU for its components."
    )


@dataclass(frozen=True)
class RuntimePolicy:
    intent: DeviceIntent
    capabilities: HardwareCapabilities
    workload: Optional[RuntimeWorkload] = None
    _state: _RuntimeState = field(default_factory=_RuntimeState, compare=False, repr=False)

    def with_workload(self, n_samples: int, n_features: int) -> RuntimePolicy:
        return RuntimePolicy(
            intent=self.intent,
            capabilities=self.capabilities,
            workload=RuntimeWorkload(int(n_samples), int(n_features)),
            _state=self._state,
        )

    def _planned_decision_for(self, component: RuntimeComponent) -> DeviceDecision:
        base = {
            "requested": self.intent.value,
            "n_samples": None if self.workload is None else self.workload.n_samples,
            "n_features": None if self.workload is None else self.workload.n_features,
        }
        if self.intent is DeviceIntent.CPU:
            return DeviceDecision(resolved="cpu", reason="explicit_cpu", **base)

        if self.intent is DeviceIntent.CUDA:
            cuda_available = self.capabilities.cuda_available_for(component)
            if cuda_available:
                return DeviceDecision(
                    resolved="cuda", reason="explicit_cuda_available", **base
                )
            if component in CUDA_BACKENDS:
                _warn_cuda_fallback(CUDA_BACKENDS[component])
                reason = "explicit_cuda_unavailable"
            else:
                reason = "component_has_no_cuda_backend"
            return DeviceDecision(resolved="cpu", reason=reason, **base)

        rule = AUTO_CUDA_RULES.get(component)
        if rule is not None:
            if self.workload is None:
                return DeviceDecision(
                    resolved="cpu",
                    reason="auto_workload_unknown",
                    work_metric=rule.work_metric,
                    threshold=rule.min_work_items,
                    **base,
                )
            work_items = rule.work_items(self.workload)
            if work_items < rule.min_work_items:
                return DeviceDecision(
                    resolved="cpu",
                    reason="auto_workload_below_threshold",
                    work_metric=rule.work_metric,
                    work_items=work_items,
                    threshold=rule.min_work_items,
                    **base,
                )
            cuda_available = self.capabilities.cuda_available_for(component)
            if not cuda_available:
                return DeviceDecision(
                    resolved="cpu",
                    reason="auto_threshold_met_but_cuda_unavailable",
                    work_metric=rule.work_metric,
                    work_items=work_items,
                    threshold=rule.min_work_items,
                    **base,
                )
            return DeviceDecision(
                resolved="cuda",
                reason="auto_workload_at_or_above_threshold",
                work_metric=rule.work_metric,
                work_items=work_items,
                threshold=rule.min_work_items,
                **base,
            )

        cuda_available = self.capabilities.cuda_available_for(component)
        if cuda_available:
            return DeviceDecision(
                resolved="cuda", reason="auto_accelerator_preferred", **base
            )
        if component in MPS_COMPONENTS and self.capabilities.torch_mps_available():
            return DeviceDecision(resolved="mps", reason="auto_mps_available", **base)
        reason = (
            "accelerator_backend_unavailable"
            if component in CUDA_BACKENDS
            else "component_has_no_accelerator_backend"
        )
        return DeviceDecision(resolved="cpu", reason=reason, **base)

    def decision_for(self, component: RuntimeComponent) -> DeviceDecision:
        planned = self._planned_decision_for(component)
        reason = self._state.cpu_fallbacks.get((component, self.workload))
        if reason is None:
            return planned
        return replace(planned, resolved="cpu", reason=reason)

    def record_cpu_fallback(self, component: RuntimeComponent, reason: str) -> None:
        self._state.cpu_fallbacks[(component, self.workload)] = str(reason)

    def device_for(self, component: RuntimeComponent) -> str:
        return self.decision_for(component).resolved

    def uses_accelerator(self, component: RuntimeComponent) -> bool:
        return self.device_for(component) in {"cuda", "mps"}

    def tuning_jobs(self, component: RuntimeComponent, cpu_jobs: int) -> int:
        return 1 if self.uses_accelerator(component) else cpu_jobs


def configure_torch_cuda(runtime: RuntimePolicy, component: RuntimeComponent) -> None:
    if runtime.device_for(component) != "cuda":
        return
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def cleanup_torch_accelerator(
    runtime: RuntimePolicy, component: RuntimeComponent
) -> None:
    device = runtime.device_for(component)
    if device not in {"cuda", "mps"}:
        return
    import torch

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def get_runtime(intent: str | DeviceIntent = DeviceIntent.Auto) -> RuntimePolicy:
    parsed = DeviceIntent.from_arg(intent)
    capabilities = CPU_CAPABILITIES if parsed is DeviceIntent.CPU else LAZY_CAPABILITIES
    return RuntimePolicy(intent=parsed, capabilities=capabilities)

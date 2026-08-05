from __future__ import annotations

import gc
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Mapping, Optional


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
    ErrorConsistency = "error-consistency"
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
    RuntimeComponent.ErrorConsistency: "torch",
}
MPS_COMPONENTS = {RuntimeComponent.Gandalf}
COMPONENT_LABELS = {
    RuntimeComponent.CatBoost: "CatBoost",
    RuntimeComponent.XGBoost: "XGBoost",
    RuntimeComponent.KNN: "KNN",
    RuntimeComponent.TabPFN: "TabPFN",
    RuntimeComponent.MLP: "MLP",
    RuntimeComponent.KAN: "KAN",
    RuntimeComponent.Gandalf: "GANDALF",
    RuntimeComponent.Embedding: "Embedding",
    RuntimeComponent.ErrorConsistency: "error consistency",
    RuntimeComponent.LightGBM: "LightGBM",
    RuntimeComponent.Sklearn: "scikit-learn model",
    RuntimeComponent.Preprocessing: "preprocessing",
    RuntimeComponent.Selection: "feature selection",
    RuntimeComponent.Univariate: "feature analysis",
}


class CudaConfigurationError(RuntimeError):
    """Raised when strict ``--device cuda`` requirements cannot be met."""


@dataclass(frozen=True)
class RuntimeWorkload:
    n_samples: int
    n_features: int
    n_queries: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.n_samples < 0
            or self.n_features < 0
            or (self.n_queries is not None and self.n_queries < 0)
        ):
            raise ValueError("Runtime workload dimensions must be non-negative.")

    @property
    def matrix_elements(self) -> int:
        return int(self.n_samples) * int(self.n_features)

    @property
    def query_samples(self) -> int:
        return int(self.n_samples if self.n_queries is None else self.n_queries)

    @property
    def pairwise_elements(self) -> int:
        return int(self.n_samples) * self.query_samples * int(self.n_features)


@dataclass(frozen=True)
class AutoCudaRule:
    work_metric: Literal["matrix_elements", "pairwise_elements"]
    min_work_items: int

    def work_items(self, workload: RuntimeWorkload) -> int:
        return int(getattr(workload, self.work_metric))


# Keep small traditional-model jobs on CPU when CUDA startup and data transfer
# are likely to cost more than the calculation. Neural and embedding jobs still
# prefer an accelerator.
AUTO_CUDA_RULES = {
    RuntimeComponent.KNN: AutoCudaRule("pairwise_elements", 20_000_000),
    RuntimeComponent.CatBoost: AutoCudaRule("matrix_elements", 1_000_000),
    RuntimeComponent.XGBoost: AutoCudaRule("matrix_elements", 200_000),
}


@dataclass(frozen=True)
class DeviceDecision:
    requested: str
    resolved: str
    reason: str
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_queries: Optional[int] = None
    work_metric: Optional[str] = None
    work_items: Optional[int] = None
    threshold: Optional[int] = None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def device_reason_text(decision: DeviceDecision) -> str:
    if decision.reason.startswith("cuda_runtime_fallback:"):
        return "CUDA failed; retried on CPU"
    reasons = {
        "explicit_cpu": "requested by --device cpu",
        "explicit_cuda_available": "required by --device cuda",
        "explicit_cuda_unavailable": "required CUDA backend is unavailable",
        "component_has_no_cuda_backend": "no supported CUDA backend",
        "auto_workload_unknown": "workload is not known yet",
        "auto_workload_below_threshold": "workload below CUDA threshold",
        "auto_threshold_met_but_cuda_unavailable": "CUDA backend unavailable",
        "auto_workload_at_or_above_threshold": "workload meets CUDA threshold",
        "auto_accelerator_preferred": "accelerator-preferred model",
        "auto_mps_available": "MPS available",
        "accelerator_backend_unavailable": "accelerator backend unavailable",
        "component_has_no_accelerator_backend": "no accelerator backend",
    }
    return reasons.get(decision.reason, decision.reason.replace("_", " "))


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


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"
    except Exception:
        return "version unknown"


def _backend_unavailable_detail(backend: str) -> str:
    if backend == "torch":
        return (
            "PyTorch CUDA is unavailable "
            f"(installed torch: {_package_version('torch')})"
        )
    if backend == "catboost":
        return (
            "CatBoost CUDA is unavailable "
            f"(installed catboost: {_package_version('catboost')})"
        )
    if backend == "xgboost":
        return (
            "XGBoost CUDA is unavailable "
            f"(installed xgboost: {_package_version('xgboost')})"
        )
    return f"{backend} CUDA is unavailable"


@lru_cache(maxsize=1)
def cuda_device_name() -> Optional[str]:
    """Describe CUDA visibility without importing or initializing a backend."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        value = visible.strip()
        if value.lower() in {"", "-1", "none"}:
            return None
        return f"CUDA_VISIBLE_DEVICES={value} (backend device index 0)"
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) == 1:
        return names[0]
    if len(names) > 1:
        return f"{len(names)} NVIDIA GPUs detected (backend device index 0)"
    return None


@dataclass(frozen=True)
class RuntimePolicy:
    intent: DeviceIntent
    capabilities: HardwareCapabilities
    workload: Optional[RuntimeWorkload] = None
    _state: _RuntimeState = field(default_factory=_RuntimeState, compare=False, repr=False)

    def with_workload(
        self,
        n_samples: int,
        n_features: int,
        n_queries: Optional[int] = None,
    ) -> RuntimePolicy:
        return RuntimePolicy(
            intent=self.intent,
            capabilities=self.capabilities,
            workload=RuntimeWorkload(
                int(n_samples),
                int(n_features),
                None if n_queries is None else int(n_queries),
            ),
            _state=self._state,
        )

    def for_task(
        self,
        n_samples: int,
        n_features: int,
        n_queries: Optional[int] = None,
    ) -> RuntimePolicy:
        """Create an isolated policy for one model/feature-selection task."""
        return RuntimePolicy(
            intent=self.intent,
            capabilities=self.capabilities,
            workload=RuntimeWorkload(
                int(n_samples),
                int(n_features),
                None if n_queries is None else int(n_queries),
            ),
        )

    def _planned_decision_for(self, component: RuntimeComponent) -> DeviceDecision:
        base = {
            "requested": self.intent.value,
            "n_samples": None if self.workload is None else self.workload.n_samples,
            "n_features": None if self.workload is None else self.workload.n_features,
            "n_queries": None if self.workload is None else self.workload.n_queries,
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
                return DeviceDecision(
                    resolved="unavailable",
                    reason="explicit_cuda_unavailable",
                    **base,
                )
            return DeviceDecision(
                resolved="cpu",
                reason="component_has_no_cuda_backend",
                **base,
            )

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
        decision = self.decision_for(component)
        if decision.resolved == "unavailable":
            label = COMPONENT_LABELS.get(component, component.value)
            backend = CUDA_BACKENDS[component]
            raise CudaConfigurationError(
                f"Cannot satisfy --device cuda for {label}: "
                f"{_backend_unavailable_detail(backend)}. "
                "Use --device auto to allow CPU fallback, or install a "
                "CUDA-capable backend."
            )
        return decision.resolved

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


def release_accelerator_memory() -> None:
    """Release cached accelerator memory before a CPU retry."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def clear_fitted_model_state(model: object) -> None:
    """Drop estimators and trainer objects that may retain accelerator tensors."""
    for name in ("tuned_model", "model", "tuned_trainer", "trainer"):
        if hasattr(model, name):
            setattr(model, name, None)


def is_cuda_runtime_error(error: BaseException) -> bool:
    """Return whether an exception chain describes a CUDA/GPU runtime failure."""
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            import torch

            if isinstance(current, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        message = f"{type(current).__name__}: {current}".lower()
        gpu_tokens = (
            "cuda",
            "cudnn",
            "cublas",
            "nccl",
            "gpu",
            "device-side",
            "status_alloc_failed",
            "bad allocation",
        )
        if any(token in message for token in gpu_tokens):
            return True
        if "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def validate_cuda_request(
    runtime: RuntimePolicy,
    components: Mapping[str, RuntimeComponent],
) -> None:
    """Validate only selected components that actually have a CUDA backend."""
    if runtime.intent is not DeviceIntent.CUDA:
        return

    eligible = [
        (name, component)
        for name, component in components.items()
        if component in CUDA_BACKENDS
    ]
    if not eligible:
        cpu_only = ", ".join(components) or "the selected workload"
        raise CudaConfigurationError(
            "--device cuda has no effect because none of the selected models "
            "has a supported CUDA backend. "
            f"CPU-only work: {cpu_only}. Use --device cpu or --device auto."
        )

    unavailable: list[tuple[str, str]] = []
    for name, component in eligible:
        if not runtime.capabilities.cuda_available_for(component):
            unavailable.append((name, CUDA_BACKENDS[component]))
    if not unavailable:
        return

    details = "\n".join(
        f"- {name} requires CUDA, but {_backend_unavailable_detail(backend)}."
        for name, backend in unavailable
    )
    raise CudaConfigurationError(
        "Cannot satisfy --device cuda:\n\n"
        f"{details}\n\n"
        "Models without a supported CUDA backend may still run normally on CPU. "
        "Use --device auto to allow unavailable CUDA models to run on CPU, or "
        "install the required CUDA-capable backend."
    )


def format_device_plan(
    runtime: RuntimePolicy,
    components: Mapping[str, RuntimeComponent],
) -> str:
    """Create a concise startup plan without importing CUDA backends."""
    cuda_names = [
        name for name, component in components.items() if component in CUDA_BACKENDS
    ]
    cpu_names = [
        name for name, component in components.items() if component not in CUDA_BACKENDS
    ]
    lines = [f"Device mode: {runtime.intent.value}"]
    if runtime.intent is not DeviceIntent.CPU:
        device_name = cuda_device_name()
        if device_name is not None:
            lines.append(f"CUDA device: {device_name}")
    lines.extend(["", "Execution plan:"])
    if runtime.intent is DeviceIntent.CPU:
        lines.append(f"  CPU:  {', '.join(components) or 'all work'}")
        return "\n".join(lines)
    if runtime.intent is DeviceIntent.CUDA:
        lines.append(f"  CUDA: {', '.join(cuda_names)}")
        if cpu_names:
            lines.append(
                f"  CPU:  {', '.join(cpu_names)} (no supported CUDA backend)"
            )
        return "\n".join(lines)

    if cuda_names:
        lines.append(
            f"  Auto: {', '.join(cuda_names)} "
            "(device chosen per task and input size)"
        )
    if cpu_names:
        lines.append(f"  CPU:  {', '.join(cpu_names)}")
    lines.append("  Final model devices use each model's actual input size.")
    return "\n".join(lines)


def get_runtime(intent: str | DeviceIntent = DeviceIntent.Auto) -> RuntimePolicy:
    parsed = DeviceIntent.from_arg(intent)
    capabilities = CPU_CAPABILITIES if parsed is DeviceIntent.CPU else LAZY_CAPABILITIES
    return RuntimePolicy(intent=parsed, capabilities=capabilities)

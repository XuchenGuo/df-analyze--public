"""Choose the NumPy or CUDA backend for error-consistency calculations."""

from __future__ import annotations

from dataclasses import dataclass

from df_analyze.runtime.hardware import (
    DeviceIntent,
    RuntimeComponent,
    RuntimePolicy,
)


@dataclass(frozen=True)
class ECBackendDecision:
    requested: str
    resolved: str
    reason: str
    work_items: int

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "ec_backend_requested": self.requested,
            "ec_backend_resolved": self.resolved,
            "ec_backend_reason": self.reason,
            "ec_backend_work_items": self.work_items,
        }


def resolve_ec_backend(
    options,
    n_models: int,
    n_samples: int,
    runtime: RuntimePolicy | None = None,
) -> ECBackendDecision:
    runtime = runtime or getattr(options, "runtime", None)
    intent = getattr(runtime, "intent", getattr(options, "device", DeviceIntent.CPU))
    if not isinstance(intent, DeviceIntent):
        try:
            intent = DeviceIntent.from_arg(intent)
        except (TypeError, ValueError):
            intent = DeviceIntent.CPU

    n_pairs = max(0, int(n_models) * (int(n_models) - 1) // 2)
    work_items = n_pairs * max(int(n_samples), 0)
    requested = intent.value
    if intent is DeviceIntent.CPU:
        return ECBackendDecision(requested, "numpy", "device_cpu", work_items)

    if intent is DeviceIntent.CUDA:
        if runtime is None:
            raise RuntimeError("Strict CUDA error consistency requires a RuntimePolicy.")
        runtime.device_for(RuntimeComponent.ErrorConsistency)
        return ECBackendDecision(requested, "torch_cuda", "device_cuda", work_items)

    if runtime is None:
        return ECBackendDecision(requested, "numpy", "cuda_unavailable", work_items)
    decision = runtime.decision_for(RuntimeComponent.ErrorConsistency)
    if decision.resolved == "cuda":
        return ECBackendDecision(requested, "torch_cuda", "cuda_available", work_items)
    return ECBackendDecision(requested, "numpy", "cuda_unavailable", work_items)

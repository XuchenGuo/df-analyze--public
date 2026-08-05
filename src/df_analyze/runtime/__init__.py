from df_analyze.runtime.hardware import (
    CudaConfigurationError,
    DeviceIntent,
    HardwareCapabilities,
    RuntimeComponent,
    RuntimePolicy,
    cleanup_torch_accelerator,
    configure_torch_cuda,
    format_device_plan,
    get_runtime,
    is_cuda_runtime_error,
    release_accelerator_memory,
    validate_cuda_request,
)


def __getattr__(name: str):
    if name == "DeviceInstall":
        from df_analyze.runtime.install import DeviceInstall

        return DeviceInstall
    raise AttributeError(name)

__all__ = [
    "DeviceInstall",
    "CudaConfigurationError",
    "DeviceIntent",
    "HardwareCapabilities",
    "RuntimeComponent",
    "RuntimePolicy",
    "cleanup_torch_accelerator",
    "configure_torch_cuda",
    "format_device_plan",
    "get_runtime",
    "is_cuda_runtime_error",
    "release_accelerator_memory",
    "validate_cuda_request",
]

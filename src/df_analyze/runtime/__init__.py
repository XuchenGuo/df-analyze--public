from df_analyze.runtime.hardware import (
    DeviceIntent,
    HardwareCapabilities,
    RuntimeComponent,
    RuntimePolicy,
    cleanup_torch_accelerator,
    configure_torch_cuda,
    get_runtime,
)


def __getattr__(name: str):
    if name == "DeviceInstall":
        from df_analyze.runtime.install import DeviceInstall

        return DeviceInstall
    raise AttributeError(name)

__all__ = [
    "DeviceInstall",
    "DeviceIntent",
    "HardwareCapabilities",
    "RuntimeComponent",
    "RuntimePolicy",
    "cleanup_torch_accelerator",
    "configure_torch_cuda",
    "get_runtime",
]

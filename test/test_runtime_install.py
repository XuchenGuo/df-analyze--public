from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

from df_analyze._constants import VERSION
from df_analyze.runtime import bootstrap, install
from df_analyze.runtime.install import (
    DeviceInstall,
    _stale_install_lock,
    runtime_root,
    setup_environment,
)


@pytest.mark.fast
def test_project_version_sources_match() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(
        package for package in lock["package"] if package["name"] == "df-analyze"
    )

    assert VERSION == project["project"]["version"]
    assert VERSION == locked_project["version"]
    assert project["project"]["scripts"] == {
        "df-analyze": "df_analyze:main",
        "df-embed": "df_analyze.embed_entrypoint:main",
    }


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='df-analyze'\n")
    (root / "uv.lock").write_text("")


@pytest.mark.fast
def test_device_install_defaults_to_never() -> None:
    assert DeviceInstall.from_arg(None, "auto") is DeviceInstall.Never
    assert DeviceInstall.from_arg(None, "cuda") is DeviceInstall.Never
    assert DeviceInstall.from_arg("auto", "cpu") is DeviceInstall.Never


@pytest.mark.fast
def test_nvidia_probe_respects_hidden_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        install.shutil,
        "which",
        lambda name: (_ for _ in ()).throw(
            AssertionError("hidden CUDA devices must not invoke nvidia-smi")
        ),
    )

    assert not install.nvidia_gpu_available()


@pytest.mark.fast
def test_bootstrap_detects_pytorch_tasks() -> None:
    catboost = bootstrap._parse_args(["--classifiers", "catboost"], "df-analyze")
    mlp = bootstrap._parse_args(["--classifiers", "catboost", "mlp"], "df-analyze")
    wrapper = bootstrap._parse_args(
        ["--classifiers", "catboost", "--wrapper-select", "step-up", "--wrapper-model", "knn"],
        "df-analyze",
    )
    ec = bootstrap._parse_args(
        ["--classifiers", "catboost", "--error-consistency"], "df-analyze"
    )
    wrapper_cuda = bootstrap._parse_args(
        [
            "--device",
            "cuda",
            "--classifiers",
            "catboost",
            "--wrapper-select",
            "step-up",
            "--wrapper-model",
            "knn",
        ],
        "df-analyze",
    )
    ec_cuda = bootstrap._parse_args(
        [
            "--device",
            "cuda",
            "--classifiers",
            "catboost",
            "--error-consistency",
        ],
        "df-analyze",
    )

    assert not bootstrap._needs_torch(catboost, "df-analyze")
    assert bootstrap._needs_torch(mlp, "df-analyze")
    assert not bootstrap._needs_torch(wrapper, "df-analyze")
    assert not bootstrap._needs_torch(ec, "df-analyze")
    assert bootstrap._needs_torch(wrapper_cuda, "df-analyze")
    assert bootstrap._needs_torch(ec_cuda, "df-analyze")
    assert not bootstrap._can_reuse_torch(catboost, "df-analyze")
    assert bootstrap._can_reuse_torch(mlp, "df-analyze")
    assert bootstrap._can_reuse_torch(wrapper, "df-analyze")
    assert bootstrap._can_reuse_torch(ec, "df-analyze")


@pytest.mark.fast
def test_bootstrap_device_options_are_case_insensitive() -> None:
    args = bootstrap._parse_args(
        ["--device", "CUDA", "--device-install", "ASK"],
        "df-analyze",
    )

    assert args.device == "cuda"
    assert args.device_install == "ask"


@pytest.mark.fast
def test_auto_reports_cpu_only_torch_when_nvidia_gpu_is_visible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "probe_torch",
        lambda: {"usable": False, "torch": "2.9.1+cpu", "cuda": None},
    )
    monkeypatch.setattr(
        bootstrap, "cached_managed_environment", lambda root: None
    )
    monkeypatch.setattr(bootstrap, "nvidia_gpu_available", lambda: True)

    bootstrap.bootstrap(
        "df-analyze",
        tmp_path / "df-analyze.py",
        tmp_path,
        ["--device", "auto", "--classifiers", "mlp"],
    )

    error = capsys.readouterr().err
    assert "NVIDIA GPU was detected" in error
    assert "PyTorch 2.9.1+cpu" in error
    assert "--device-install auto" in error


@pytest.mark.fast
def test_cuda_task_relaunches_in_managed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(tmp_path)
    python = tmp_path / "cuda-python"
    relaunched = []
    monkeypatch.setattr(bootstrap, "probe_torch", lambda: None)
    monkeypatch.setattr(bootstrap, "nvidia_gpu_available", lambda: True)
    monkeypatch.setattr(
        bootstrap, "cached_managed_environment", lambda root: None
    )
    monkeypatch.setattr(bootstrap, "setup_environment", lambda root: python)
    monkeypatch.setattr(
        bootstrap,
        "_relaunch",
        lambda executable, script, argv, module=None: relaunched.append(
            (executable, script, argv, module)
        ),
    )

    bootstrap.bootstrap(
        "df-analyze",
        tmp_path / "df-analyze.py",
        tmp_path,
        ["--device", "cuda", "--device-install", "auto", "--classifiers", "mlp"],
    )

    assert relaunched[0][0] == python
    assert "--device" in relaunched[0][2]


@pytest.mark.fast
def test_existing_environment_is_reused_without_install_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(tmp_path)
    python = tmp_path / "cuda-python"
    relaunched = []
    monkeypatch.setattr(bootstrap, "probe_torch", lambda: None)
    monkeypatch.setattr(
        bootstrap, "cached_managed_environment", lambda root: python
    )
    monkeypatch.setattr(
        bootstrap,
        "_relaunch",
        lambda executable, script, argv, module=None: relaunched.append(executable),
    )

    bootstrap.bootstrap(
        "df-analyze",
        tmp_path / "df-analyze.py",
        tmp_path,
        ["--device", "cuda", "--classifiers", "mlp"],
    )

    assert relaunched == [python]


@pytest.mark.fast
def test_auto_knn_reuses_existing_environment_without_installing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(tmp_path)
    python = tmp_path / "cuda-python"
    relaunched = []
    monkeypatch.setattr(bootstrap, "probe_torch", lambda: None)
    monkeypatch.setattr(
        bootstrap, "cached_managed_environment", lambda root: python
    )
    monkeypatch.setattr(
        bootstrap,
        "setup_environment",
        lambda root: (_ for _ in ()).throw(
            AssertionError("auto KNN must not trigger installation")
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "_relaunch",
        lambda executable, script, argv, module=None: relaunched.append(
            executable
        ),
    )

    bootstrap.bootstrap(
        "df-analyze",
        tmp_path / "df-analyze.py",
        tmp_path,
        ["--device", "auto", "--classifiers", "knn"],
    )

    assert relaunched == [python]


@pytest.mark.fast
def test_existing_environment_requires_full_runtime_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _project(tmp_path)
    root = runtime_root(tmp_path)
    python = install.environment_python(root / ".venv")
    python.parent.mkdir(parents=True)
    python.touch()
    (root / "manifest.json").write_text(
        json.dumps({"project_fingerprint": install.project_fingerprint(tmp_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(install, "verify_environment", lambda executable: None)

    assert install.managed_environment(tmp_path) is None

    monkeypatch.setattr(
        install, "verify_environment", lambda executable: {"torch": "test"}
    )
    assert install.managed_environment(tmp_path) == python


@pytest.mark.fast
def test_noninteractive_ask_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert not bootstrap._approve_install(DeviceInstall.Ask)


@pytest.mark.fast
def test_setup_dry_run_does_not_write_environment(tmp_path: Path) -> None:
    setup_environment(tmp_path, dry_run=True)

    assert not runtime_root(tmp_path).exists()


@pytest.mark.fast
def test_setup_reports_missing_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(install.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Could not find `uv`"):
        setup_environment(tmp_path)


@pytest.mark.fast
def test_install_lock_detects_dead_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = tmp_path / "install.lock"
    lock.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(install, "_process_exists", lambda pid: False)

    assert _stale_install_lock(lock)


@pytest.mark.fast
def test_install_lock_preserves_live_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = tmp_path / "install.lock"
    lock.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(install, "_process_exists", lambda pid: True)

    assert not _stale_install_lock(lock)


@pytest.mark.fast
def test_windows_liveness_check_never_signals_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install, "_IS_WINDOWS", True)
    monkeypatch.setattr(install, "_windows_process_exists", lambda pid: True)

    def unexpected_signal(pid: int, signal: int) -> None:
        pytest.fail("Windows process liveness checks must not call os.kill")

    monkeypatch.setattr(install.os, "kill", unexpected_signal)

    assert install._process_exists(12345)


@pytest.mark.fast
@pytest.mark.skipif(not install._IS_WINDOWS, reason="Windows-specific process check")
def test_windows_process_check_detects_current_process() -> None:
    assert install._windows_process_exists(os.getpid())

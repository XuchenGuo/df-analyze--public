from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import tomllib

_IS_WINDOWS = os.name == "nt"
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class DeviceInstall(Enum):
    Auto = "auto"
    Ask = "ask"
    Never = "never"

    @classmethod
    def parse(cls, value: str) -> str:
        return cls(str(value).lower()).value

    @classmethod
    def from_arg(
        cls, value: str | DeviceInstall | None, device: str = "auto"
    ) -> DeviceInstall:
        if device == "cpu":
            return cls.Never
        if isinstance(value, cls):
            return value
        return cls.Never if value is None else cls(str(value).lower())

    @classmethod
    def choices(cls) -> list[str]:
        return [item.value for item in cls]


TORCH_PROBE = """
import json
import torch

usable = False
try:
    if torch.cuda.is_available():
        torch.empty(1, device="cuda")
        usable = True
except Exception:
    pass
print(json.dumps({"usable": usable, "torch": torch.__version__, "cuda": torch.version.cuda}))
"""


def _windows_process_exists(pid: int) -> bool:
    """Check process liveness on Windows without sending it a signal."""
    if pid <= 0:
        return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def _process_exists(pid: int) -> bool:
    if _IS_WINDOWS:
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno != errno.ESRCH
    return True


def _stale_install_lock(path: Path) -> bool:
    try:
        raw_pid = path.read_text(encoding="utf-8").strip()
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    try:
        pid = int(raw_pid)
    except ValueError:
        return age > 60
    return not _process_exists(pid)

VERIFY_RUNTIME = """
import json
import torch

torch.empty(1, device="cuda")
for name in ("skorch", "lightning", "pytorch_tabular", "transformers", "tabpfn"):
    __import__(name)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda}))
"""


def environment_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_root(project_root: Path) -> Path:
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system().lower()
    machine = platform.machine().lower().replace("x86_64", "amd64")
    return project_root / ".df-analyze-runtime" / f"cuda-{py}-{system}-{machine}"


def project_fingerprint(project_root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def nvidia_gpu_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False
    result = subprocess.run(
        [executable, "-L"], capture_output=True, text=True, check=False
    )
    return result.returncode == 0 and "GPU" in result.stdout


def _run_probe(python: Path, script: str) -> Optional[dict[str, Any]]:
    result = subprocess.run(
        [str(python), "-c", script], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None


def probe_torch(python: Path = Path(sys.executable)) -> Optional[dict[str, Any]]:
    return _run_probe(python, TORCH_PROBE)


def verify_environment(python: Path) -> Optional[dict[str, Any]]:
    return _run_probe(python, VERIFY_RUNTIME)


def managed_environment(project_root: Path) -> Optional[Path]:
    root = runtime_root(project_root)
    python = environment_python(root / ".venv")
    manifest = root / "manifest.json"
    if not python.exists() or not manifest.exists():
        return None
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if info.get("project_fingerprint") != project_fingerprint(project_root):
        return None
    return python if verify_environment(python) is not None else None


def _write_constraints(project_root: Path, path: Path) -> None:
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        name = package.get("name")
        version = package.get("version")
        source = package.get("source", {})
        if (
            isinstance(name, str)
            and isinstance(version, str)
            and name != "df-analyze"
            and "registry" in source
        ):
            versions.setdefault(name, set()).add(version)
    lines = [
        f"{name}=={next(iter(found))}"
        for name, found in sorted(versions.items())
        if len(found) == 1
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class InstallLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self) -> InstallLock:
        deadline = time.monotonic() + 30 * 60
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if _stale_install_lock(self.path):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for CUDA environment setup.")
                time.sleep(1)

    def __exit__(self, *args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def setup_environment(project_root: Path, dry_run: bool = False) -> Path:
    project_root = project_root.resolve()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Could not find `uv` on PATH.")

    root = runtime_root(project_root)
    final_env = root / ".venv"
    build_env = root / f".venv.build-{os.getpid()}"
    final_python = environment_python(final_env)
    build_python = environment_python(build_env)
    constraints = root / "constraints.txt"
    create = [uv, "venv", str(build_env), "--python", sys.executable]
    install = [
        uv,
        "pip",
        "install",
        "--python",
        str(build_python),
        "--torch-backend",
        "auto",
        "--strict",
        "--constraints",
        str(constraints),
        "--editable",
        str(project_root),
    ]
    if dry_run:
        print(subprocess.list2cmdline(create))
        print(subprocess.list2cmdline(install))
        return final_python

    root.mkdir(parents=True, exist_ok=True)
    with InstallLock(root / "install.lock"):
        existing = managed_environment(project_root)
        if existing is not None:
            return existing
        shutil.rmtree(build_env, ignore_errors=True)
        _write_constraints(project_root, constraints)
        try:
            subprocess.run(create, cwd=project_root, check=True)
            subprocess.run(install, cwd=project_root, check=True)
            verified = verify_environment(build_python)
            if verified is None:
                raise RuntimeError("The installed PyTorch environment could not use CUDA.")
            shutil.rmtree(final_env, ignore_errors=True)
            build_env.replace(final_env)
            manifest = {
                "project_fingerprint": project_fingerprint(project_root),
                "python": platform.python_version(),
                **verified,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        finally:
            shutil.rmtree(build_env, ignore_errors=True)
    return final_python


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the df-analyze CUDA runtime.")
    parser.add_argument("action", choices=["check", "setup"])
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "check":
        python = managed_environment(args.project.resolve())
        if python is None:
            print("No valid managed CUDA environment was found.")
            return 1
        print(python)
        return 0
    print(setup_environment(args.project, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

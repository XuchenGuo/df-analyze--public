from __future__ import annotations

import argparse
import csv
import os
import posixpath
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence
from zipfile import ZipFile

from df_analyze.runtime.install import (
    DeviceInstall,
    managed_environment,
    nvidia_gpu_available,
    probe_torch,
    setup_environment,
)

BOOTSTRAP_ENV = "DF_ANALYZE_BOOTSTRAPPED"
TORCH_MODELS = {"knn", "mlp", "kan", "gandalf", "tabpfn"}


def _csv_options(path: Path, separator: str) -> list[str]:
    options = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter=separator):
            values = [str(value).strip() for value in row if str(value).strip()]
            if values and values[0].startswith("--"):
                options.extend(" ".join(values).split())
    return options


def _xlsx_options(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
                for item in root
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(node for node in workbook.iter() if node.tag.endswith("}sheet"))
        rel_id = next(value for key, value in sheet.attrib.items() if key.endswith("}id"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(node.attrib["Target"] for node in rels if node.attrib["Id"] == rel_id)
        target = target.lstrip("/")
        sheet_path = target if target.startswith("xl/") else posixpath.join("xl", target)
        data = ET.fromstring(archive.read(posixpath.normpath(sheet_path)))
        options = []
        for row in (node for node in data.iter() if node.tag.endswith("}row")):
            values = []
            for cell in (node for node in row if node.tag.endswith("}c")):
                value = next((n for n in cell.iter() if n.tag.endswith("}v")), None)
                if cell.attrib.get("t") == "s" and value is not None:
                    values.append(shared[int(value.text or "0")])
                elif cell.attrib.get("t") == "inlineStr":
                    values.append(
                        "".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t"))
                    )
                elif value is not None:
                    values.append(value.text or "")
            values = [value.strip() for value in values if value.strip()]
            if values and values[0].startswith("--"):
                options.extend(" ".join(values).split())
        return options


def _spreadsheet_options(path: Optional[str], separator: str) -> list[str]:
    if path is None:
        return []
    sheet = Path(path)
    if not sheet.exists():
        return []
    if sheet.suffix.lower() == ".xlsx":
        return _xlsx_options(sheet)
    if sheet.suffix.lower() == ".csv":
        return _csv_options(sheet, separator)
    return []


def _parse_args(argv: Sequence[str], entrypoint: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--device-install", choices=DeviceInstall.choices(), default=None)
    if entrypoint == "df-analyze":
        parser.add_argument("--mode", choices=["classify", "regress"], default="classify")
        parser.add_argument("--classifiers", nargs="+", default=None)
        parser.add_argument("--regressors", nargs="+", default=None)
        parser.add_argument("--wrapper-select", default=None)
        parser.add_argument("--wrapper-model", default="linear")
        parser.add_argument(
            "--error-consistency", "--ec", action="store_true", default=False
        )
        parser.add_argument("--spreadsheet", default=None)
        parser.add_argument("--separator", default=",")
        initial = parser.parse_known_args(argv)[0]
        try:
            sheet_args = _spreadsheet_options(initial.spreadsheet, initial.separator)
        except Exception:
            sheet_args = []
        return parser.parse_known_args([*sheet_args, *argv])[0]
    if entrypoint == "df-embed":
        parser.add_argument("--download", action="store_true")
        parser.add_argument("--force-download", action="store_true")
    return parser.parse_known_args(argv)[0]


def _needs_torch(args: argparse.Namespace, entrypoint: str) -> bool:
    if entrypoint == "df-embed":
        return not (getattr(args, "download", False) or getattr(args, "force_download", False))
    if getattr(args, "error_consistency", False):
        return True
    wrapper = str(getattr(args, "wrapper_select", "") or "").lower()
    wrapper_model = str(getattr(args, "wrapper_model", "") or "").lower()
    if wrapper not in {"", "none"} and wrapper_model == "knn":
        return True
    models = args.classifiers if args.mode == "classify" else args.regressors
    return True if models is None else bool(TORCH_MODELS.intersection(models))


def _approve_install(policy: DeviceInstall) -> bool:
    if policy is DeviceInstall.Auto:
        return True
    if policy is DeviceInstall.Never:
        return False
    if not sys.stdin.isatty():
        print(
            "A CUDA PyTorch environment is required. Re-run with "
            "`--device-install auto` to install it automatically.",
            file=sys.stderr,
        )
        return False
    answer = input("Install a managed CUDA PyTorch environment? [y/N] ").strip()
    return answer.lower() in {"y", "yes"}


def _relaunch(
    python: Path,
    script: Optional[Path],
    argv: Sequence[str],
    module: Optional[str] = None,
) -> None:
    env = os.environ.copy()
    env[BOOTSTRAP_ENV] = "1"
    if script is not None and script.exists():
        command = [str(python), str(script), *argv]
    elif module is not None:
        command = [str(python), "-m", module, *argv]
    else:
        raise RuntimeError("No script or module is available for CUDA relaunch.")
    raise SystemExit(subprocess.call(command, env=env))


def bootstrap(
    entrypoint: str,
    script: Optional[Path],
    project_root: Path,
    argv: Optional[Sequence[str]] = None,
    module: Optional[str] = None,
) -> None:
    if os.environ.get(BOOTSTRAP_ENV) == "1":
        return
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in argv for arg in ("-h", "--help", "--version")):
        return
    args = _parse_args(argv, entrypoint)
    if args.device == "cpu" or not _needs_torch(args, entrypoint):
        return
    current = probe_torch()
    if current is not None and current.get("usable") is True:
        return

    project_root = project_root.resolve()
    if (project_root / "pyproject.toml").exists() and (
        project_root / "uv.lock"
    ).exists():
        existing = managed_environment(project_root)
        if existing is not None:
            _relaunch(existing, script, argv, module=module)

    policy = DeviceInstall.from_arg(args.device_install, args.device)
    if policy is DeviceInstall.Never:
        if args.device == "cuda":
            print(
                "CUDA was requested, but the current PyTorch cannot use it. "
                "Continuing on CPU. Pass `--device-install auto` to create a "
                "managed CUDA environment.",
                file=sys.stderr,
            )
        return
    if not nvidia_gpu_available():
        if args.device == "cuda":
            print("CUDA was requested, but no NVIDIA GPU was detected.", file=sys.stderr)
        return

    if not (project_root / "pyproject.toml").exists() or not (
        project_root / "uv.lock"
    ).exists():
        if args.device == "cuda":
            print(
                "Managed CUDA setup requires a source checkout with pyproject.toml "
                "and uv.lock. Continuing in the current environment.",
                file=sys.stderr,
            )
        return

    if not _approve_install(policy):
        return
    try:
        python = setup_environment(project_root)
    except Exception as exc:
        print(
            "Could not create the managed CUDA environment. Continuing in the "
            f"current environment.\n{exc}",
            file=sys.stderr,
        )
        return
    _relaunch(python, script, argv, module=module)

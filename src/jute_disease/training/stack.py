"""Verification of the measured local CUDA stack."""

import hashlib
import platform
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from jute_disease.config import ROOT_DIR

LOCK_PATH = ROOT_DIR / "uv.lock"
LIBRARIES = ("timm", "lightning", "scikit-learn", "torchvision")


class StackError(RuntimeError):
    """The installed stack cannot run the declared benchmark."""


def configure_runtime() -> dict[str, bool]:
    """Apply the owner-approved backend settings without forcing determinism."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    return {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else None


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _compare_with_lock(lock: Path) -> dict[str, Any]:
    """Require every installed locked package to match a version in the lock.

    A package the lock splits by platform or index lists several versions; the
    installed one must equal one of them. Locked packages that are not installed
    (other platform, other group) are reported, not failed.
    """
    try:
        raw = lock.read_bytes()
        document = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise StackError(f"The environment lock {lock.name} cannot be read") from error
    requires_python = document.get("requires-python")
    try:
        python_ok = requires_python is None or Version(
            platform.python_version()
        ) in SpecifierSet(requires_python)
    except (InvalidSpecifier, InvalidVersion, TypeError) as error:
        raise StackError(f"{lock.name} has an invalid requires-python") from error
    if not python_ok:
        raise StackError(
            f"Python {platform.python_version()} is outside the lock's "
            f"requires-python {requires_python}"
        )
    locked: dict[str, set[str]] = {}
    for package in document.get("package", []):
        source = package.get("source", {})
        if "editable" in source or "virtual" in source:
            continue
        name, version = package.get("name"), package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            locked.setdefault(_normalized(name), set()).add(version)
    mismatches: list[str] = []
    not_installed: list[str] = []
    checked = 0
    for name, versions in sorted(locked.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            not_installed.append(name)
            continue
        checked += 1
        try:
            matches = any(Version(installed) == Version(v) for v in versions)
        except InvalidVersion:
            matches = installed in versions
        if not matches:
            mismatches.append(
                f"{name} {installed} installed, lock has {', '.join(sorted(versions))}"
            )
    if mismatches:
        raise StackError(
            f"Installed packages differ from {lock.name}: " + "; ".join(mismatches)
        )
    return {
        "path": lock.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checked_packages": checked,
        "not_installed": not_installed,
    }


def _observed_compute_dtype(uses_cuda: bool) -> str:
    """Return the dtype a matrix product actually runs in on the selected device."""
    if not uses_cuda:
        left = torch.ones((2, 2))
        return str((left @ left).dtype)
    left = torch.ones((8, 8), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return str((left @ left).dtype)


def verify_stack(
    *,
    require_cuda: bool = True,
    precision: str = "bf16-mixed",
    accelerator: str = "auto",
    lock: Path | None = None,
) -> dict[str, Any]:
    """Describe the locked environment and fail if it differs from the lock.

    Compilation is never enabled, so ``compile_enabled`` is always false; the
    report records it so a receipt can show the stack that ran. The report also
    carries the lock comparison and the precision the device was observed to use.

    Raises:
        StackError: CUDA is required and unavailable, a library is missing, an
            installed package differs from the lock, or the requested precision
            is not the one observed.
    """
    backend = configure_runtime()
    lock_report = _compare_with_lock(LOCK_PATH if lock is None else Path(lock))
    libraries: dict[str, str] = {}
    for name in LIBRARIES:
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise StackError(f"Required library {name} is not installed") from error
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise StackError("CUDA is not available to the locked torch build")
    uses_cuda = accelerator in ("gpu", "cuda") or (accelerator == "auto" and cuda)
    if require_cuda and not uses_cuda:
        raise StackError("CUDA is required but the selected accelerator is not CUDA")
    if uses_cuda and not cuda:
        raise StackError(
            "CUDA accelerator selected but unavailable to the locked torch build"
        )
    requested = precision if uses_cuda else "32-true"
    compute_dtype = _observed_compute_dtype(uses_cuda)
    expected_dtype = "torch.bfloat16" if requested == "bf16-mixed" else "torch.float32"
    if compute_dtype != expected_dtype:
        raise StackError(
            f"Requested precision {requested} but the device computed in "
            f"{compute_dtype}"
        )
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_driver": _driver_version() if cuda else None,
        "precision": requested,
        "observed_precision": {
            "requested": requested,
            "compute_dtype": compute_dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "lock": lock_report,
        "compile_enabled": False,
        "backend": backend,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "libraries": libraries,
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        report["device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_bytes": properties.total_memory,
        }
    return report

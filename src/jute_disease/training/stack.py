"""Verification of the measured local CUDA stack."""

import platform
import subprocess
from importlib import metadata
from typing import Any

import torch

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


def verify_stack(
    *,
    require_cuda: bool = True,
    precision: str = "bf16-mixed",
    accelerator: str = "auto",
) -> dict[str, Any]:
    """Describe the locked environment and fail if CUDA is required but absent.

    Compilation is never enabled, so ``compile_enabled`` is always false; the
    report records it so a receipt can show the stack that ran.

    Raises:
        StackError: CUDA is required and unavailable, or a library is missing.
    """
    backend = configure_runtime()
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
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_driver": _driver_version() if cuda else None,
        "precision": precision if uses_cuda else "32-true",
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

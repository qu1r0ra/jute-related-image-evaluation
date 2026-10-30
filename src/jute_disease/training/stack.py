"""Verification of the measured local CUDA stack."""

from importlib import metadata
from typing import Any

import torch

LIBRARIES = ("timm", "lightning", "scikit-learn", "albumentationsx", "torchvision")


class StackError(RuntimeError):
    """The installed stack cannot run the declared benchmark."""


def verify_stack(*, require_cuda: bool = True) -> dict[str, Any]:
    """Describe the locked environment and fail if CUDA is required but absent.

    Compilation is never enabled, so ``compile_enabled`` is always false; the
    report records it so a receipt can show the stack that ran.

    Raises:
        StackError: CUDA is required and unavailable, or a library is missing.
    """
    libraries: dict[str, str] = {}
    for name in LIBRARIES:
        try:
            libraries[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise StackError(f"Required library {name} is not installed") from error
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise StackError("CUDA is not available to the locked torch build")
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "compile_enabled": False,
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

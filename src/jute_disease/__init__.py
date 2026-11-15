from importlib import import_module
from typing import Any

__all__ = ["config", "data", "evaluation", "models", "reporting", "training"]


def __getattr__(name: str) -> Any:
    """Import a subpackage on first access, so the package itself starts light."""
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Public deep-learning fit entry points."""

_FIT_EXPORTS = {"fit_cnn", "tune_cnn", "training_callbacks", "ProgressRecord"}


def __getattr__(name: str) -> object:
    if name in _FIT_EXPORTS:
        from jute_disease.training.engines.dl import fit

        return getattr(fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_FIT_EXPORTS)

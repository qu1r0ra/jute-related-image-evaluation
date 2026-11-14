"""Public classical-fit entry points.

Test predictions are produced only by ``evaluate predict-test``, never here.
"""

_FIT_EXPORTS = {"fit_classical", "prepare_features"}


def __getattr__(name: str) -> object:
    if name in _FIT_EXPORTS:
        from jute_disease.training.engines.ml import fit

        return getattr(fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_FIT_EXPORTS]

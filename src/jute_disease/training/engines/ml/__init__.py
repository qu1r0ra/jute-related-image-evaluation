"""Public classical-fit and inference entry points."""

from jute_disease.training.engines.ml.predict import predict_ml

_FIT_EXPORTS = {"fit_classical", "fit_cue", "prepare_features"}


def __getattr__(name: str) -> object:
    if name in _FIT_EXPORTS:
        from jute_disease.training.engines.ml import fit

        return getattr(fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["predict_ml", *_FIT_EXPORTS]

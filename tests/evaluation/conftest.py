from collections.abc import Callable, Mapping, Sequence

import pytest

from jute_disease.evaluation import PredictionSet

SetFactory = Callable[..., PredictionSet]


@pytest.fixture
def make_set() -> SetFactory:
    """Build a tiny prediction set from truth labels and per-seed predictions."""

    def build(
        truth: Sequence[str],
        predictions: Mapping[int, Sequence[str]],
        groups: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        classes: Sequence[str] = ("a", "b", "c"),
    ) -> PredictionSet:
        count = len(truth)
        return PredictionSet(
            classes=tuple(classes),
            image_ids=tuple(f"img{i}" for i in range(count)),
            truth=tuple(truth),
            groups=tuple(groups or [f"g{i}" for i in range(count)]),
            sources=tuple(sources or ["s"] * count),
            predictions={seed: tuple(p) for seed, p in predictions.items()},
        )

    return build

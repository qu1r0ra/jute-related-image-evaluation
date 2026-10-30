"""Deterministic, balanced sampling and loader seeds."""

import pytest
import torch

from jute_disease.data import balanced_sampler, loader_generator


def test_balanced_sampler_repeats_for_a_seed_and_differs_for_another() -> None:
    labels = [0] * 30 + [1] * 5 + [2] * 5
    first = list(balanced_sampler(labels, seed=42))
    assert first == list(balanced_sampler(labels, seed=42))
    assert first != list(balanced_sampler(labels, seed=43))
    drawn = torch.tensor([labels[i] for i in first])
    counts = torch.bincount(drawn)
    assert counts.min() > 5  # minority classes are oversampled well above 5/40
    with pytest.raises(ValueError):
        balanced_sampler([])


def test_loader_generators_are_independent_and_seeded() -> None:
    a, b = loader_generator(42), loader_generator(42)
    assert torch.equal(torch.rand(3, generator=a), torch.rand(3, generator=b))
    assert not torch.equal(
        torch.rand(3, generator=loader_generator(1)),
        torch.rand(3, generator=loader_generator(2)),
    )

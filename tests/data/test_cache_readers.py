"""Image cache, manifest readers and the deterministic sampler."""

import itertools
from pathlib import Path

import pytest
import torch
from PIL import Image

from jute_disease.data.cache import (
    DEFAULT_BUDGET_SECONDS,
    build_image_cache,
    cache_root,
    decide_cache,
    downsize,
    is_cache_complete,
)
from jute_disease.data.manifest import Manifest, load_manifest, write_manifest
from jute_disease.data.readers import ManifestDataset, open_partition
from jute_disease.data.sampling import balanced_sampler, loader_generator
from jute_disease.data.synthetic import SyntheticPlan
from jute_disease.data.transforms import PreprocessingSpec


def _cache(frozen: Manifest, plan: SyntheticPlan, tmp_path: Path, **kwargs) -> object:
    return build_image_cache(
        frozen, plan.root, tmp_path / "cache", PreprocessingSpec(), workers=2, **kwargs
    )


def test_downsize_keeps_small_images_and_scales_large_ones_down() -> None:
    small = Image.new("RGB", (40, 60))
    assert downsize(small, 64) is small
    big = downsize(Image.new("RGB", (200, 400)), 100)
    assert big.size == (100, 200)


def test_cache_holds_train_and_val_only_and_reports_completion(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    report = _cache(frozen, synthetic, tmp_path)
    root = cache_root(tmp_path / "cache", PreprocessingSpec())
    wanted = {e.image_id for n in ("train", "val") for e in frozen.partition(n)}
    assert {p.stem for p in root.glob("*.png")} == wanted
    assert report.complete
    assert report.images_done == report.images_total == len(wanted)
    assert is_cache_complete(root, frozen, ("train", "val"))
    assert not is_cache_complete(root, frozen, ("train", "val", "test"))
    assert decide_cache(report)["decision"] == "adopt-cache"


def test_the_test_partition_is_refused_unless_explicitly_allowed(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="test"):
        _cache(frozen, synthetic, tmp_path, partitions=("test",))


def test_an_exhausted_budget_yields_an_incomplete_cache_and_lazy_decode(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    ticks = itertools.count(0, 100)
    report = _cache(
        frozen, synthetic, tmp_path, budget_seconds=1, clock=lambda: float(next(ticks))
    )
    root = cache_root(tmp_path / "cache", PreprocessingSpec())
    assert not report.complete
    assert not is_cache_complete(root, frozen, ("train", "val"))
    decision = decide_cache(report)
    assert decision["decision"] == "lazy-decode"
    assert "did not finish" in decision["reasons"][0]  # type: ignore[index]


def test_a_disk_limit_forces_lazy_decode(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    report = _cache(frozen, synthetic, tmp_path)
    assert decide_cache(report, max_bytes=1)["decision"] == "lazy-decode"
    assert DEFAULT_BUDGET_SECONDS == 15 * 60


def test_a_changed_preprocessing_spec_uses_a_new_cache_directory(
    tmp_path: Path,
) -> None:
    first = cache_root(tmp_path, PreprocessingSpec())
    other = cache_root(tmp_path, PreprocessingSpec(cache_shortest_side=256))
    assert first != other


def test_dataset_serves_a_partition_in_stable_order(
    frozen: Manifest, synthetic: SyntheticPlan
) -> None:
    dataset = ManifestDataset(frozen, "val", synthetic.root)
    assert len(dataset) == len(frozen.partition("val"))
    assert dataset.image_ids == sorted(dataset.image_ids)
    image, target = dataset[0]
    assert isinstance(image, Image.Image)
    assert dataset.classes[target] == frozen.partition("val")[0].label
    assert dataset.identity["source"] == "original"
    assert dataset.identity["deterministic"] == "false"


def test_an_unknown_partition_is_rejected(
    frozen: Manifest, synthetic: SyntheticPlan
) -> None:
    with pytest.raises(ValueError, match="Unknown partition"):
        ManifestDataset(frozen, "nonexistent", synthetic.root)


def test_a_requested_cache_never_falls_back_to_the_original(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-cache"
    empty.mkdir()
    dataset = ManifestDataset(frozen, "train", synthetic.root, cache_root=empty)
    assert dataset.identity["source"].startswith("cache:")
    with pytest.raises(FileNotFoundError):
        dataset[0]
    _cache(frozen, synthetic, tmp_path)
    root = cache_root(tmp_path / "cache", PreprocessingSpec())
    cached = ManifestDataset(frozen, "train", synthetic.root, cache_root=root)
    assert cached[0][1] == dataset.targets[0]


def test_open_partition_requires_a_completed_freeze(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        open_partition("train", manifest_dir=tmp_path / "none")
    write_manifest(frozen, tmp_path / "freeze")
    dataset = open_partition(
        "train", manifest_dir=tmp_path / "freeze", image_root=synthetic.root
    )
    assert len(dataset) == len(frozen.partition("train"))


def test_version_one_freezes_remain_readable_and_keep_their_digest(
    frozen: Manifest, tmp_path: Path
) -> None:
    legacy_data = frozen.to_json()
    legacy_data["version"] = 1
    for entry in legacy_data["entries"]:
        entry.pop("pad_fraction")
    legacy = Manifest.from_json(legacy_data)
    directory = tmp_path / "legacy-freeze"
    write_manifest(legacy, directory)

    restored = load_manifest(directory)
    assert restored.version == 1
    assert restored.to_json() == legacy_data
    assert restored.digest() == legacy.digest()


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

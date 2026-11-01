"""Manifest-backed datasets and partition readers."""

from pathlib import Path

import pytest
from PIL import Image

from jute_disease.data import (
    Manifest,
    ManifestDataset,
    PreprocessingSpec,
    SyntheticPlan,
    build_image_cache,
    cache_root,
    open_partition,
)
from jute_disease.data.manifest import write_manifest


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
    assert dataset.identity["content"] == dataset.content_digest


def test_an_unknown_partition_is_rejected(
    frozen: Manifest, synthetic: SyntheticPlan
) -> None:
    with pytest.raises(ValueError, match="Unknown partition"):
        ManifestDataset(frozen, "nonexistent", synthetic.root)


def test_a_manifest_reader_rejects_changed_image_bytes(
    frozen: Manifest, synthetic: SyntheticPlan
) -> None:
    entry = frozen.partition("train")[0]
    path = synthetic.root / entry.path
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="content hash"):
            ManifestDataset(frozen, "train", synthetic.root)
    finally:
        path.write_bytes(original)


def test_a_requested_cache_never_falls_back_to_the_original(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    empty = tmp_path / "empty-cache"
    empty.mkdir()
    dataset = ManifestDataset(frozen, "train", synthetic.root, cache_root=empty)
    assert dataset.identity["source"].startswith("cache:")
    with pytest.raises(FileNotFoundError):
        dataset[0]
    build_image_cache(
        frozen,
        synthetic.root,
        tmp_path / "cache",
        PreprocessingSpec(),
        workers=2,
    )
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

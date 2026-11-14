"""Prepared image cache creation, limits and decisions."""

import itertools
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from jute_disease.data import (
    DEFAULT_BUDGET_SECONDS,
    CacheReport,
    ImageCacheError,
    Manifest,
    PreprocessingSpec,
    SyntheticPlan,
    build_image_cache,
    cache_root,
    decide_cache,
    image_cache_identity,
    is_cache_complete,
    verify_image_cache,
)
from jute_disease.data.cache import _index_lock, downsize


def _cache(
    frozen: Manifest, plan: SyntheticPlan, tmp_path: Path, **kwargs: Any
) -> CacheReport:
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


def _built(frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path) -> Path:
    _cache(frozen, synthetic, tmp_path)
    return tmp_path / "cache"


def test_the_cache_records_an_index_and_a_stable_identity(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    cache = _built(frozen, synthetic, tmp_path)
    spec = PreprocessingSpec()
    parts = ("train", "val")
    identity = image_cache_identity(cache, spec, frozen, parts)
    assert len(identity) == 64
    assert verify_image_cache(cache, spec, frozen, parts, synthetic.root) == identity
    index = json.loads((cache_root(cache, spec) / "cache_index.json").read_text())
    assert index["spec_digest"] == spec.digest()
    wanted = {e.image_id for n in parts for e in frozen.partition(n)}
    assert set(index["images"]) == wanted
    report = json.loads((cache_root(cache, spec) / "cache_report.json").read_text())
    assert report["index_sha256"]


def test_a_source_image_that_changed_after_the_build_is_refused(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    cache = _built(frozen, synthetic, tmp_path)
    images = tmp_path / "images"
    shutil.copytree(synthetic.root, images)
    entry = frozen.partition("train")[0]
    with (images / entry.path).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ImageCacheError, match="changed after the cache"):
        verify_image_cache(cache, PreprocessingSpec(), frozen, ("train", "val"), images)


def test_an_altered_cache_entry_is_refused(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    cache = _built(frozen, synthetic, tmp_path)
    spec = PreprocessingSpec()
    victim = sorted(cache_root(cache, spec).glob("*.png"))[0]
    victim.write_bytes(victim.read_bytes() + b"\0")
    with pytest.raises(ImageCacheError, match="index"):
        verify_image_cache(cache, spec, frozen, ("train", "val"), synthetic.root)


def test_a_missing_index_or_image_is_refused(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    spec = PreprocessingSpec()
    with pytest.raises(ImageCacheError, match="No image cache index"):
        image_cache_identity(tmp_path / "none", spec, frozen, ("train",))
    cache = _built(frozen, synthetic, tmp_path)
    with pytest.raises(ImageCacheError, match="not in the image cache"):
        image_cache_identity(cache, spec, frozen, ("train", "val", "test"))


@pytest.mark.serial
def test_concurrent_builders_leave_one_valid_cache(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(
            pool.map(lambda _: _cache(frozen, synthetic, tmp_path), range(4))
        )
    assert all(report.complete for report in reports)
    cache = tmp_path / "cache"
    spec = PreprocessingSpec()
    root = cache_root(cache, spec)
    assert not list(root.glob("*.tmp"))
    identity = image_cache_identity(cache, spec, frozen, ("train", "val"))
    assert (
        verify_image_cache(cache, spec, frozen, ("train", "val"), synthetic.root)
        == identity
    )
    assert {report.index_sha256 for report in reports} == {reports[0].index_sha256}


@pytest.mark.serial
def test_index_lock_excludes_a_second_holder_until_released(tmp_path: Path) -> None:
    with _index_lock(tmp_path):
        with pytest.raises(ImageCacheError, match="Timed out"):
            with _index_lock(tmp_path, timeout=0.2):
                pass
    with _index_lock(tmp_path, timeout=0.2):
        pass


def test_reports_are_published_atomically(
    frozen: Manifest, synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    cache = _built(frozen, synthetic, tmp_path)
    root = cache_root(cache, PreprocessingSpec())
    assert json.loads((root / "cache_report.json").read_text())["complete"] is True
    assert not [p for p in root.iterdir() if p.suffix == ".tmp" or ".tmp" in p.name]

"""Prepared image cache creation, limits and decisions."""

import itertools
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from jute_disease.data import (
    DEFAULT_BUDGET_SECONDS,
    CacheReport,
    Manifest,
    PreprocessingSpec,
    SyntheticPlan,
    build_image_cache,
    cache_root,
    decide_cache,
    is_cache_complete,
)
from jute_disease.data.cache import downsize


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

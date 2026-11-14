"""Where the frozen target and verified source packages live on disk."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jute_disease.data import Manifest, PreprocessingSpec
    from jute_disease.training.benchmark import FitSpec, Recipe


@dataclass(frozen=True)
class DataInputs:
    """Frozen manifests and image roots a fit may read.

    ``sources`` maps a source stage name to its manifest directory and the pinned
    package root its images live under. ``image_cache_dir`` is the optional
    content-hashed image cache; when set, every target and classical fit reads
    its pixels from it and is refused unless the cache verifies.
    """

    image_root: Path
    target_dir: Path
    sources: Mapping[str, tuple[Path, Path]]
    policy_dir: Path | None = None
    pool_roots: Mapping[str, Path] | None = None
    image_cache_dir: Path | None = None

    def target_root(self, pool: str) -> Path:
        """Resolve the image root for a policy pool, falling back to the target root."""
        if self.pool_roots is None:
            return self.image_root
        return self.pool_roots.get(pool, self.image_root)


@dataclass(frozen=True)
class ImageCache:
    """A verified image cache for one manifest, as a fit reads it."""

    root: Path
    spec: PreprocessingSpec
    identity: str


def open_image_cache(
    data: DataInputs,
    recipe: Recipe,
    manifest: Manifest,
    image_root: Path,
    partitions: tuple[str, ...] = ("train", "val"),
) -> ImageCache | None:
    """Verify the configured image cache for ``manifest``; ``None`` if unused.

    Raises:
        ImageCacheError: the cache is missing, altered or built from other images.
    """
    from jute_disease.data import PreprocessingSpec, cache_root, verify_image_cache

    if data.image_cache_dir is None:
        return None
    spec = PreprocessingSpec(size=recipe.image_size)
    identity = verify_image_cache(
        data.image_cache_dir, spec, manifest, partitions, image_root
    )
    return ImageCache(cache_root(data.image_cache_dir, spec), spec, identity)


def preprocessing_record(images: ImageCache | None) -> dict[str, object]:
    """Describe how a fit's pixels were prepared, for its receipt.

    Gated inference reads this to prepare test pixels the same way: a fit that
    read the cache saw images downsized to the cache's shortest side.
    """
    from dataclasses import asdict

    if images is None:
        return {"source": "original"}
    return {
        "source": "cache",
        "spec": asdict(images.spec),
        "spec_digest": images.spec.digest(),
    }


def target_manifest_for_fit(fit: FitSpec, recipe: Recipe, data: DataInputs) -> Manifest:
    """Return the immutable target manifest selected by one declared fit."""
    from jute_disease.data import load_manifest, load_policy_manifest

    if not getattr(fit, "pool", "") or data.policy_dir is None:
        return load_manifest(data.target_dir)
    if getattr(fit, "phase", "final") == "tune":
        return load_policy_manifest(
            data.policy_dir,
            pool=fit.pool,
            image_size=recipe.image_size,
            reserve=True,
        )
    return load_policy_manifest(
        data.policy_dir,
        pool=fit.pool,
        view=fit.view,
        policy=fit.policy,
        draw=fit.split,
        image_size=recipe.image_size,
    )

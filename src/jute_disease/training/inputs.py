"""Where the frozen target and verified source packages live on disk."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jute_disease.data import Manifest
    from jute_disease.training.benchmark import FitSpec, Recipe


@dataclass(frozen=True)
class DataInputs:
    """Frozen manifests and image roots a fit may read.

    ``sources`` maps a source stage name to its manifest directory and the pinned
    package root its images live under.
    """

    image_root: Path
    target_dir: Path
    sources: Mapping[str, tuple[Path, Path]]
    policy_dir: Path | None = None
    pool_roots: Mapping[str, Path] | None = None
    mask_checkpoint: Path | None = None
    fixture_masks: bool = False

    def target_root(self, pool: str) -> Path:
        """Resolve the image root for a policy pool, falling back to the target root."""
        if self.pool_roots is None:
            return self.image_root
        return self.pool_roots.get(pool, self.image_root)


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

"""Training data setup for optional source-domain pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jute_disease.config import NUM_WORKERS

if TYPE_CHECKING:
    from jute_disease.data import DataModule


def source_data_module(
    manifest_dir: Path | str,
    image_root: Path | str,
    target_manifest_dir: Path | str,
    batch_size: int,
    seed: int,
    *,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = False,
) -> DataModule:
    """Open source partitions after pin, owner authorization, and byte checks."""
    from jute_disease.data import DataModule

    data_module = DataModule(
        manifest_dir=manifest_dir,
        image_root=image_root,
        target_manifest_dir=target_manifest_dir,
        batch_size=batch_size,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers,
    )
    data_module.setup("fit")
    return data_module

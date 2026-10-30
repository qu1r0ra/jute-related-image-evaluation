"""Partition access for the corrected training path.

Training and the queue may read the train and validation partitions only. The
held-out test partition belongs to the final evaluator (#721), which runs from
the retained-arm freeze receipt and never feeds queue or fallback decisions.
"""

from pathlib import Path

from jute_disease.config import MANIFEST_DIR, SOURCE_INVENTORY_DIR
from jute_disease.data import ManifestDataset, load_manifest

TRAINING_PARTITIONS = ("train", "val")
# Lightning CLI subcommands that read the held-out partition.
HELD_OUT_SUBCOMMANDS = frozenset({"test", "predict"})


class HeldOutAccessError(PermissionError):
    """Training code asked for a partition only the final evaluator may open."""


def open_training_partition(
    name: str,
    transform: object | None = None,
    manifest_dir: Path = MANIFEST_DIR,
    image_root: Path = SOURCE_INVENTORY_DIR,
) -> ManifestDataset:
    """Open ``train`` or ``val`` of the completed freeze at ``manifest_dir``.

    Raises:
        HeldOutAccessError: ``name`` is not a training partition.
        FileNotFoundError: the freeze has no matching receipt (not completed).
    """
    if name not in TRAINING_PARTITIONS:
        raise HeldOutAccessError(
            f"partition {name!r} is not readable by training; "
            "test pixels belong to the final evaluator (#721)"
        )
    return ManifestDataset(load_manifest(manifest_dir), name, image_root, transform)  # type: ignore[arg-type]

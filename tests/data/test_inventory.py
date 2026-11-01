"""Inventory: source-preserving observation of an image tree."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.data import (
    InventoryRecord,
    SyntheticPlan,
    build_inventory,
    inventory_digest,
    read_inventory,
    write_inventory,
)


def test_inventory_covers_every_file_and_leaves_them_untouched(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord]
) -> None:
    assert [r.path for r in inventory] == synthetic.files
    assert all((synthetic.root / r.path).is_file() for r in inventory)
    assert {r.label for r in inventory} == {p.split("/")[0] for p in synthetic.files}
    assert all(r.source_partition is None for r in inventory)


def test_round_trip_and_digest_are_stable(inventory: list, tmp_path: Path) -> None:
    target = tmp_path / "inventory.jsonl"
    write_inventory(inventory, target)
    assert read_inventory(target) == sorted(inventory, key=lambda r: r.path)
    assert inventory_digest(list(reversed(inventory))) == inventory_digest(inventory)


def test_digest_changes_when_content_changes(inventory: list) -> None:
    changed = [
        *inventory[:-1],
        inventory[-1].__class__(**{**vars(inventory[-1]), "sha256": "0" * 64}),
    ]
    assert inventory_digest(changed) != inventory_digest(inventory)


def test_historical_partition_folder_is_recorded_not_used(tmp_path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    folder = tmp_path / "train" / "Dieback"
    folder.mkdir(parents=True)
    Image.fromarray(pixels).save(folder / "a.png")
    (record,) = build_inventory(tmp_path, workers=1)
    assert (record.label, record.source_partition) == ("Dieback", "train")


def test_unsupported_layout_and_empty_root_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No images"):
        build_inventory(tmp_path)
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(deep / "x.png")
    with pytest.raises(ValueError, match="Unsupported layout"):
        build_inventory(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_inventory(tmp_path / "missing")

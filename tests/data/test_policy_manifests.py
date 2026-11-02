import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.data import load_policy_manifest


def _entry(index: int, partition: str, label: str = "a") -> dict[str, object]:
    return {
        "path": f"{index}.png",
        "sha256": f"{index:064x}",
        "label": label,
        "family": f"family-{index}",
        "source": "synthetic",
        "width": 8,
        "height": 8,
        "partition": partition,
    }


def _write_artifact(root: Path, relative: Path, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    receipt_path = root / "receipt.json"
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else {"files": {}}
    )
    receipt["files"][relative.as_posix()] = hashlib.sha256(encoded).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_policy_manifest_exposes_only_fit_partitions(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    fit_relative = Path("A/view1/image-random/draw-0.json")
    test_relative = Path("A/view1/image-random/draw-0.protected-test.json")
    _write_artifact(
        root,
        fit_relative,
        {
            "pool": "A",
            "view": 1,
            "policy": "image-random",
            "draw": 0,
            "entries": [
                _entry(1, "train"),
                _entry(2, "val"),
            ],
        },
    )
    _write_artifact(
        root,
        test_relative,
        {
            "pool": "A",
            "view": 1,
            "policy": "image-random",
            "draw": 0,
            "entries": [_entry(3, "test")],
        },
    )

    manifest = load_policy_manifest(
        root, pool="A", view=1, policy="image-random", draw=0, image_size=8
    )

    assert len(manifest.partition("train")) == 1
    assert len(manifest.partition("val")) == 1
    assert not manifest.partition("test")
    protected = json.loads((root / test_relative).read_text(encoding="utf-8"))
    assert {row["partition"] for row in protected["entries"]} == {"test"}


def test_policy_fit_loader_refuses_test_rows(tmp_path: Path) -> None:
    root = tmp_path / "policies"
    relative = Path("A/view1/image-random/draw-0.json")
    _write_artifact(
        root,
        relative,
        {
            "pool": "A",
            "view": 1,
            "policy": "image-random",
            "draw": 0,
            "entries": [
                _entry(1, "train"),
                _entry(2, "val"),
                _entry(3, "test"),
            ],
        },
    )

    with pytest.raises(ValueError, match="non-fit partition"):
        load_policy_manifest(
            root, pool="A", view=1, policy="image-random", draw=0, image_size=8
        )


def test_reserve_manifest_is_family_disjoint_and_class_stratified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policies"
    _write_artifact(
        root,
        Path("A/reserve.json"),
        {
            "pool": "A",
            "entries": [
                _entry(1, "reserve", "a"),
                _entry(2, "reserve", "a"),
                _entry(3, "reserve", "b"),
                _entry(4, "reserve", "b"),
            ],
        },
    )

    manifest = load_policy_manifest(root, pool="A", image_size=8, reserve=True)

    assert {entry.label for entry in manifest.partition("train")} == {"a", "b"}
    assert {entry.label for entry in manifest.partition("val")} == {"a", "b"}
    train = {entry.group for entry in manifest.partition("train")}
    val = {entry.group for entry in manifest.partition("val")}
    assert train.isdisjoint(val)

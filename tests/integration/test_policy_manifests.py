"""Policy manifest generation through the command boundary."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.data import (
    InventoryRecord,
    Manifest,
    ManifestEntry,
    inventory_digest,
    write_inventory,
    write_manifest,
)


def frozen_pool(
    root: Path, gate: str = "pass", healthy_count: int = 30, copies: int = 3
) -> Path:
    """Construct a small frozen pool, with two images per family."""
    records = []
    entries = []
    families = []
    for label in ("Healthy", "Mosaic"):
        for index in range(healthy_count if label == "Healthy" else 30):
            family = []
            for copy in range(copies):
                path = f"{label}/{index:02}-{copy}.jpg"
                content_path = f"{label}/{index:02}-{0 if copy == 2 else copy}.jpg"
                sha = hashlib.sha256(content_path.encode()).hexdigest()
                records.append(
                    InventoryRecord(path, label, None, 1, 10, 10, sha, sha, (0,))
                )
                family.append(path)
                if copy != 2:
                    entries.append(
                        ManifestEntry(
                            sha,
                            path,
                            label,
                            ("train", "val", "test")[index % 3],
                            sha,
                            "fixture",
                            10,
                            10,
                            0.0,
                        )
                    )
            families.append(family)
    grouping = {
        "algorithm_version": "dhash-v1",
        "families": families,
        "excluded": [],
        "gate_outcomes": {"agreement": gate, "threshold_change": "not_requested"},
    }
    grouping["digest"] = hashlib.sha256(
        json.dumps(grouping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_inventory(records, root / "inventory.jsonl")
    manifest = Manifest(
        seed=42,
        fractions=(("train", 0.7), ("val", 0.15), ("test", 0.15)),
        entries=tuple(sorted(entries, key=lambda e: e.image_id)),
        excluded=(),
        merged=(),
        quarantined=(),
        constraints=(),
        dependencies=(
            ("grouping", grouping["digest"]),
            ("grouping_version", "dhash-v1"),
            ("inventory", inventory_digest(records)),
        ),
    )
    write_manifest(manifest, root / "target")
    (root / "target" / "grouping.json").write_text(
        json.dumps(grouping), encoding="utf-8"
    )
    return root


def test_command_refuses_pending_grouping_gate(tmp_path: Path) -> None:
    pool = frozen_pool(tmp_path / "input", "pending")
    with pytest.raises(SystemExit, match="gate"):
        main(
            [
                "freeze",
                "policies",
                "--pool",
                f"A={pool}",
                "--output",
                str(tmp_path / "output"),
                "--reserve-fraction",
                "0.2",
                "--train-fraction",
                "0.7",
                "--val-fraction",
                "0.15",
                "--test-fraction",
                "0.15",
            ]
        )
    assert not (tmp_path / "output").exists()


def test_insufficient_reserve_does_not_change_requested_fraction(
    tmp_path: Path,
) -> None:
    pool = frozen_pool(tmp_path / "input", healthy_count=5)
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="reserve family support"):
        main(
            [
                "freeze",
                "policies",
                "--pool",
                f"C={pool}",
                "--output",
                str(output),
                "--reserve-fraction",
                "0.05",
                "--train-fraction",
                "0.7",
                "--val-fraction",
                "0.15",
                "--test-fraction",
                "0.15",
            ]
        )
    assert not output.exists()


def test_command_writes_complete_paired_manifests(tmp_path: Path) -> None:
    pool = frozen_pool(tmp_path / "input")
    out = tmp_path / "output"
    main(
        [
            "freeze",
            "policies",
            "--pool",
            f"C={pool}",
            "--output",
            str(out),
            "--reserve-fraction",
            "0.2",
            "--train-fraction",
            "0.7",
            "--val-fraction",
            "0.15",
            "--test-fraction",
            "0.15",
        ]
    )
    assignments = json.loads((out / "assignments.json").read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    assert (
        receipt["files"]["assignments.json"]
        == hashlib.sha256((out / "assignments.json").read_bytes()).hexdigest()
    )
    assert [row["checkpoint"] for row in assignments[:5]] == [0, 1, 2, 0, 1]
    assert [row["checkpoint"] for row in assignments[5:10]] == [1, 2, 0, 1, 2]
    assert [sum(row["checkpoint"] == c for row in assignments) for c in range(3)] == [
        17,
        17,
        16,
    ]
    reserve = json.loads((out / "C" / "reserve.json").read_text())
    reserve_families = {row["family"] for row in reserve["entries"]}
    tests = []
    for draw in range(10):
        protected = []
        for policy in ("image-random", "exact-copy", "near-duplicate"):
            for view in (1, 2):
                manifest = json.loads(
                    (
                        out / "C" / f"view{view}" / policy / f"draw-{draw}.json"
                    ).read_text()
                )
                entries = manifest["entries"]
                assert not reserve_families.intersection(
                    row["family"] for row in entries
                )
                assert manifest["draw_seed"] == 42 + draw
                for label in ("Healthy", "Mosaic"):
                    assert all(
                        manifest["family_support"][part][label] > 0
                        for part in ("train", "val", "test")
                    )
                if policy != "image-random":
                    key = "sha256" if policy == "exact-copy" else "family"
                    assert all(
                        len({row["partition"] for row in entries if row[key] == item})
                        == 1
                        for item in {row[key] for row in entries}
                    )
                    if policy == "exact-copy":
                        assert len({row["sha256"] for row in entries}) < len(entries)
                if view == 2:
                    protected.append(
                        {row["path"] for row in entries if row["partition"] == "test"}
                    )
        assert protected[0] == protected[1] == protected[2]
        for family in {row["family"] for row in entries if row["partition"] == "test"}:
            assert all(
                row["partition"] == "test" for row in entries if row["family"] == family
            )
        tests.append(frozenset(protected[0]))
    assert len(set(tests)) > 1


def test_healthy_cap_is_deterministic_and_only_applies_to_training(
    tmp_path: Path,
) -> None:
    pool = frozen_pool(tmp_path / "input", healthy_count=1400, copies=1)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for out in outputs:
        main(
            [
                "freeze",
                "policies",
                "--pool",
                f"C={pool}",
                "--output",
                str(out),
                "--reserve-fraction",
                "0.05",
                "--train-fraction",
                "0.9",
                "--val-fraction",
                "0.05",
                "--test-fraction",
                "0.05",
            ]
        )
    receipts = [json.loads((out / "receipt.json").read_text()) for out in outputs]
    assert receipts[0] == receipts[1]
    manifest = json.loads(
        (outputs[0] / "C/view2/near-duplicate/draw-0.json").read_text()
    )
    assert manifest["family_support"]["train"]["Healthy"] == 1000
    assert manifest["healthy_cap_excluded"]
    assert all(row["partition"] == "train" for row in manifest["healthy_cap_excluded"])

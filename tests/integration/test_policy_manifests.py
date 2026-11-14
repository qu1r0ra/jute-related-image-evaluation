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
    write_isolated_manifests,
)


def frozen_pool(
    root: Path,
    gate: str = "pass",
    healthy_count: int = 30,
    copies: int = 3,
    threshold_change: str = "not_requested",
    neighbours: bool = True,
    healthy_sizes: list[int] | None = None,
) -> Path:
    """Construct a small frozen pool, with two images per family.

    ``healthy_sizes`` instead gives one Healthy family per entry with that many
    images, all byte-identical copies of one another.
    """
    records = []
    entries = []
    families = []
    for label in ("Healthy", "Mosaic"):
        sized = label == "Healthy" and healthy_sizes is not None
        for index in range(
            len(healthy_sizes or [])
            if sized
            else healthy_count
            if label == "Healthy"
            else 30
        ):
            family = []
            for copy in range(healthy_sizes[index] if sized else copies):
                path = f"{label}/{index:02}-{copy}.jpg"
                content_path = (
                    f"{label}/{index:02}-0.jpg"
                    if sized
                    else f"{label}/{index:02}-{0 if copy == 2 else copy}.jpg"
                )
                sha = hashlib.sha256(content_path.encode()).hexdigest()
                records.append(
                    InventoryRecord(path, label, None, 1, 10, 10, sha, sha, (0,))
                )
                family.append(path)
                if copy == 0 if sized else copy != 2:
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
        "gate_outcomes": {"agreement": gate, "threshold_change": threshold_change},
    }
    if neighbours:
        grouping["audit_band_neighbours"] = {
            "threshold": 4,
            "max_distance": 16,
            "pairs": [],
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
    write_isolated_manifests(manifest, root / "target")
    (root / "target" / "grouping.json").write_text(
        json.dumps(grouping), encoding="utf-8"
    )
    return root


def test_recorded_threshold_change_is_accepted_and_other_outcomes_are_not(
    tmp_path: Path,
) -> None:
    from jute_disease.data.policies import _pool

    _pool(frozen_pool(tmp_path / "applied", threshold_change="applied"))
    with pytest.raises(ValueError, match="gate outcomes"):
        _pool(frozen_pool(tmp_path / "failed", threshold_change="fail"))


def test_grouping_without_audit_band_neighbours_is_refused(tmp_path: Path) -> None:
    from jute_disease.data.policies import _pool

    with pytest.raises(ValueError, match="audit-band neighbours"):
        _pool(frozen_pool(tmp_path / "old", neighbours=False))


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
                base = out / "C" / f"view{view}" / policy / f"draw-{draw}"
                fit_manifest = json.loads(base.with_suffix(".json").read_text())
                test_manifest = json.loads(
                    base.with_suffix(".protected-test.json").read_text()
                )
                fit_entries = fit_manifest["entries"]
                test_entries = test_manifest["entries"]
                entries = fit_entries + test_entries
                assert {row["partition"] for row in fit_entries} <= {"train", "val"}
                assert {row["partition"] for row in test_entries} == {"test"}
                assert not reserve_families.intersection(
                    row["family"] for row in entries
                )
                assert (
                    fit_manifest["draw_seed"] == test_manifest["draw_seed"] == 42 + draw
                )
                for label in ("Healthy", "Mosaic"):
                    assert all(
                        fit_manifest["family_support"][part][label] > 0
                        for part in ("train", "val")
                    )
                    assert test_manifest["family_support"]["test"][label] > 0
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
                    protected.append({row["path"] for row in test_entries})
        assert protected[0] == protected[1] == protected[2]
        for family in {row["family"] for row in test_entries}:
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


POLICIES = ("image-random", "exact-copy", "near-duplicate")
CAP_ARGS = (
    "--reserve-fraction",
    "0.3",
    "--train-fraction",
    "0.7",
    "--val-fraction",
    "0.15",
    "--test-fraction",
    "0.15",
)


def _freeze_capped(pool: Path, output: Path, cap: int) -> dict:
    main(
        [
            "freeze",
            "policies",
            "--pool",
            f"C={pool}",
            "--output",
            str(output),
            *CAP_ARGS,
            "--healthy-cap",
            str(cap),
        ]
    )
    return json.loads((output / "receipt.json").read_text())


def _healthy_train(
    output: Path, view: int, policy: str, draw: int
) -> tuple[list, list]:
    """The kept and the cap-excluded Healthy training rows of one policy draw."""
    path = output / "C" / f"view{view}" / policy / f"draw-{draw}.json"
    payload = json.loads(path.read_text())
    kept = [
        row
        for row in payload["entries"]
        if row["label"] == "Healthy" and row["partition"] == "train"
    ]
    return kept, payload["healthy_cap_excluded"]


def _unit_sizes(rows: list[dict], policy: str) -> list[int]:
    """Selectable unit sizes: whole families, or single images for image-random."""
    if policy == "image-random":
        return [1] * len(rows)
    families: dict[str, int] = {}
    for row in rows:
        families[row["family"]] = families.get(row["family"], 0) + 1
    return list(families.values())


def _reachable(sizes: list[int], cap: int) -> set[int]:
    sums = {0}
    for size in sizes:
        sums |= {total + size for total in sums if total + size <= cap}
    return sums


def _assert_one_feasible_target(output: Path, cap: int, receipt: dict) -> list[tuple]:
    """Check every draw against an independent brute-force subset-sum oracle.

    Returns the (view, draw, before) pre-cap counts for the caller's scenario checks.
    """
    seen = []
    applied = {
        (a["view"], a["draw"]): a for a in receipt["healthy_cap"]["applied"]["C"]
    }
    for view in (1, 2):
        for draw in range(10):
            before, after, reach = {}, {}, None
            for policy in POLICIES:
                kept, dropped = _healthy_train(output, view, policy, draw)
                before[policy] = len(kept) + len(dropped)
                after[policy] = len(kept)
                options = _reachable(_unit_sizes(kept + dropped, policy), cap)
                reach = options if reach is None else reach & options
                if policy == "image-random":
                    assert not {row["path"] for row in kept} & {
                        row["path"] for row in dropped
                    }
                else:
                    assert not {row["family"] for row in kept} & {
                        row["family"] for row in dropped
                    }
            expected = max(reach - {0}, default=None)
            assert set(after.values()) == {expected}, (view, draw, before, after)
            assert expected <= cap
            assert applied[(view, draw)]["target"] == expected
            assert applied[(view, draw)]["healthy_train_images"] == after
            seen.append((view, draw, before))
    return seen


def test_unequal_counts_below_the_cap_are_equalised(tmp_path: Path) -> None:
    pool = frozen_pool(tmp_path / "input", healthy_sizes=[2, 3, 1, 2, 3, 2, 1, 3, 2, 2])
    output = tmp_path / "output"
    receipt = _freeze_capped(pool, output, cap=1000)
    seen = _assert_one_feasible_target(output, 1000, receipt)
    assert any(len(set(before.values())) > 1 for _, _, before in seen)
    assert all(max(before.values()) < 1000 for _, _, before in seen)


def test_policies_straddling_the_cap_share_one_count(tmp_path: Path) -> None:
    pool = frozen_pool(tmp_path / "input", healthy_sizes=[3, 2, 4, 2, 3, 2, 4, 3, 2, 4])
    output = tmp_path / "output"
    receipt = _freeze_capped(pool, output, cap=16)
    seen = _assert_one_feasible_target(output, 16, receipt)
    assert any(min(b.values()) < 16 < max(b.values()) for _, _, b in seen), seen


def test_a_feasible_target_is_found_when_a_seeded_greedy_order_would_miss_it(
    tmp_path: Path,
) -> None:
    # Near-duplicate draws whose training families are 6, 5 and 5 images: a
    # greedy pass that takes the 6 first can never reach 10, but 5 + 5 does.
    pool = frozen_pool(tmp_path / "input", healthy_sizes=[6, 5, 5, 6, 5, 5, 6])
    output = tmp_path / "output"
    receipt = _freeze_capped(pool, output, cap=10)
    seen = _assert_one_feasible_target(output, 10, receipt)
    exact = [
        (view, draw)
        for view, draw, _ in seen
        if sorted(
            _unit_sizes(
                sum(_healthy_train(output, view, "near-duplicate", draw), []),
                "near-duplicate",
            )
        )
        == [5, 5, 6]
    ]
    assert exact


def test_equal_healthy_count_that_cannot_be_reached_stops_output(
    tmp_path: Path,
) -> None:
    pool = frozen_pool(tmp_path / "input", healthy_sizes=[2] * 10)
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="no common Healthy image count"):
        _freeze_capped(pool, output, cap=1)
    assert not output.exists()

"""Manifest: deterministic group-aware split, validation and freeze receipt."""

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.data.curation import CurationRules
from jute_disease.data.inventory import InventoryRecord
from jute_disease.data.manifest import (
    MANIFEST_FILE,
    RECEIPT_FILE,
    FreezeBlockedError,
    Manifest,
    build_groups,
    build_manifest,
    load_manifest,
    validate_manifest,
    write_manifest,
)


def test_partitions_are_disjoint_complete_and_class_stratified(
    frozen: Manifest,
) -> None:
    support = frozen.support()
    assert set(support) == {"train", "val", "test"}
    for label in frozen.labels:
        counts = {p: support[p][label] for p in support}
        total = sum(counts.values())
        assert counts["val"] == pytest.approx(0.15 * total, abs=2.5)
        assert counts["test"] == pytest.approx(0.15 * total, abs=2.5)
        assert counts["train"] > counts["val"]


def test_groups_never_span_partitions_and_constraints_hold(frozen: Manifest) -> None:
    validate_manifest(frozen, min_support=2)
    partition = {e.image_id: e.partition for e in frozen.entries}
    assert frozen.constraints
    for left, right in frozen.constraints:
        assert partition[left] == partition[right]


def test_source_manifest_allows_missing_classes_in_published_held_out_partitions(
    frozen: Manifest,
) -> None:
    label = frozen.labels[0]
    entries = tuple(
        entry
        for entry in frozen.entries
        if entry.label != label or entry.partition == "train"
    )
    ids = {entry.image_id for entry in entries}
    source = replace(
        frozen,
        entries=entries,
        constraints=tuple(pair for pair in frozen.constraints if set(pair) <= ids),
        dependencies=tuple(
            sorted((*frozen.dependencies, ("source_inventory", "a" * 64)))
        ),
    )

    validate_manifest(source, min_support=1)


def test_target_manifest_still_requires_each_class_in_every_partition(
    frozen: Manifest,
) -> None:
    label = frozen.labels[0]
    entries = tuple(
        entry
        for entry in frozen.entries
        if entry.label != label or entry.partition == "train"
    )
    ids = {entry.image_id for entry in entries}
    target = replace(
        frozen,
        entries=entries,
        constraints=tuple(pair for pair in frozen.constraints if set(pair) <= ids),
    )

    with pytest.raises(FreezeBlockedError, match="in val; need 1"):
        validate_manifest(target, min_support=1)


def test_excluded_quarantined_and_merged_files_are_not_in_any_partition(
    frozen: Manifest, synthetic
) -> None:
    placed = {e.path for e in frozen.entries}
    for record in (*frozen.excluded, *frozen.quarantined, *frozen.merged):
        assert record["path"] not in placed
    assert {e["path"] for e in frozen.excluded} == set(synthetic.quarantine_files)
    assert len(frozen.entries) + len(frozen.excluded) + len(frozen.merged) + len(
        frozen.quarantined
    ) == len(synthetic.files)


def test_seed_42_manifest_is_reproducible_across_input_order(
    inventory: list[InventoryRecord], rules: CurationRules, frozen: Manifest
) -> None:
    shuffled = list(inventory)
    random.Random(7).shuffle(shuffled)
    again = build_manifest(shuffled, rules, seed=42)[0]
    assert again.digest() == frozen.digest()
    assert again.to_json() == frozen.to_json()


def test_a_different_seed_changes_the_assignment_but_not_the_inputs(
    inventory: list[InventoryRecord], rules: CurationRules, frozen: Manifest
) -> None:
    other = build_manifest(inventory, rules, seed=43)[0]
    assert other.partition_digest() != frozen.partition_digest()
    assert (
        dict(other.dependencies)["inventory"] == dict(frozen.dependencies)["inventory"]
    )


def test_dependency_hashes_track_their_inputs(
    inventory: list[InventoryRecord], rules: CurationRules, frozen: Manifest
) -> None:
    deps = dict(frozen.dependencies)
    assert set(deps) >= {"inventory", "rules", "reviews", "identity"}
    changed_rules = replace(rules, min_support=3)
    other = build_manifest(inventory, changed_rules, seed=42)[0]
    assert dict(other.dependencies)["rules"] != deps["rules"]
    extra = build_manifest(
        inventory, rules, seed=42, extra_dependencies={"preprocessing": "abc"}
    )[0]
    assert dict(extra.dependencies)["preprocessing"] == "abc"


def test_infeasible_support_blocks_the_freeze(
    inventory: list[InventoryRecord], rules: CurationRules
) -> None:
    with pytest.raises(FreezeBlockedError, match="need 50"):
        build_manifest(inventory, replace(rules, min_support=50), seed=42)


def test_a_giant_group_is_kept_whole_and_leaves_other_partitions_short(
    frozen: Manifest,
) -> None:
    from jute_disease.data.manifest import _assign_label

    placement = _assign_label({"g": 20}, 42, dict(frozen.fractions))
    assert list(placement) == ["g"]
    mixed = _assign_label({"a": 10, "b": 5, "c": 5}, 42, dict(frozen.fractions))
    assert set(mixed) == {"a", "b", "c"}


def test_build_groups_merges_components_deterministically() -> None:
    ids = ["c", "a", "b", "d"]
    groups = build_groups(ids, (("c", "b"), ("b", "a")))
    assert groups == {"a": "a", "b": "a", "c": "a", "d": "d"}


def test_validate_rejects_a_violated_constraint(frozen: Manifest) -> None:
    by_id = {e.image_id: e for e in frozen.entries}
    first = frozen.entries[0]
    other = next(e for e in frozen.entries if e.partition != first.partition)
    broken = replace(
        frozen, constraints=((first.image_id, other.image_id), *frozen.constraints)
    )
    assert first.image_id in by_id
    with pytest.raises(FreezeBlockedError, match="violated"):
        validate_manifest(broken, 1)


def test_write_then_load_round_trips_and_receipt_is_last(
    frozen: Manifest, tmp_path: Path
) -> None:
    receipt = write_manifest(frozen, tmp_path / "freeze")
    assert receipt.name == RECEIPT_FILE
    loaded = load_manifest(tmp_path / "freeze", min_support=2)
    assert loaded == frozen
    data = json.loads(receipt.read_text())
    assert data["manifest_digest"] == frozen.digest()
    assert data["images"] == len(frozen.entries)
    assert data["support"] == frozen.support()
    assert "dependencies" in data


def test_a_directory_without_a_receipt_is_not_a_completed_freeze(
    frozen: Manifest, tmp_path: Path
) -> None:
    directory = tmp_path / "freeze"
    write_manifest(frozen, directory)
    (directory / RECEIPT_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="No completed freeze"):
        load_manifest(directory)
    (directory / MANIFEST_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        load_manifest(directory)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "never-created")


def test_tampering_with_the_manifest_invalidates_the_receipt(
    frozen: Manifest, tmp_path: Path
) -> None:
    directory = tmp_path / "freeze"
    write_manifest(frozen, directory)
    path = directory / MANIFEST_FILE
    data = json.loads(path.read_text())
    data["entries"][0]["partition"] = (
        "val" if data["entries"][0]["partition"] == "train" else "train"
    )
    path.write_text(json.dumps(data))
    with pytest.raises(FreezeBlockedError, match="Receipt does not match"):
        load_manifest(directory)


def test_rewriting_a_freeze_removes_the_old_receipt_first(
    frozen: Manifest, tmp_path: Path
) -> None:
    directory = tmp_path / "freeze"
    write_manifest(frozen, directory)
    stale = (directory / RECEIPT_FILE).read_text()
    write_manifest(frozen, directory)
    assert (directory / RECEIPT_FILE).read_text() == stale


def test_bad_fractions_and_unknown_partitions_raise(
    inventory: list[InventoryRecord], rules: CurationRules, frozen: Manifest
) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        build_manifest(
            inventory, rules, fractions={"train": 0.5, "val": 0.2, "test": 0.2}
        )
    with pytest.raises(ValueError, match="Unknown partition"):
        frozen.partition("holdout")
    assert len(frozen.partition("train")) == sum(frozen.support()["train"].values())

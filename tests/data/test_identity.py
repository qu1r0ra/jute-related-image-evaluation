"""Identity: exact duplicates, reviews, canonicalization and constraints."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.data import (
    InventoryRecord,
    SyntheticPlan,
    find_candidates,
    load_reviews,
    resolve_identity,
)
from jute_disease.data.identity import (
    reviews_digest,
)


def _by_path(inventory: list[InventoryRecord]) -> dict[str, InventoryRecord]:
    return {r.path: r for r in inventory}


def _unique(inventory: list[InventoryRecord]) -> list[InventoryRecord]:
    seen: dict[str, InventoryRecord] = {}
    for record in inventory:
        seen.setdefault(record.image_id, record)
    return list(seen.values())


def test_candidates_are_order_independent(inventory: list[InventoryRecord]) -> None:
    unique = _unique(inventory)
    assert find_candidates(unique) == find_candidates(list(reversed(unique)))


def test_planted_pairs_are_candidates(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord]
) -> None:
    by_path = _by_path(inventory)
    pairs = {tuple(sorted((c.a, c.b))): c for c in find_candidates(_unique(inventory))}
    for left, right in synthetic.rotated_pairs + synthetic.lookalike_pairs:
        key = tuple(sorted((by_path[left].image_id, by_path[right].image_id)))
        assert key in pairs, (left, right)
    cross = [c for c in pairs.values() if not c.same_label]
    assert len(cross) >= len(synthetic.lookalike_pairs)


def test_unreviewed_resolution_applies_conservative_policy(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord]
) -> None:
    by_path = _by_path(inventory)
    resolution = resolve_identity(inventory)
    retained = {r.image_id for r in resolution.retained}
    quarantined = {q.path: q.reason for q in resolution.quarantined}

    merged = {m.path: m for m in resolution.merged}
    for original, copy in synthetic.byte_copies:
        assert merged[copy].reason == "byte-identical copy"
        assert merged[copy].canonical_id == by_path[original].image_id
        assert by_path[original].image_id in retained

    for original, copy in synthetic.lookalike_pairs:
        assert quarantined[original] == "unresolved cross-label candidate"
        assert quarantined[copy] == "unresolved cross-label candidate"

    constrained = {x for pair in resolution.constraints for x in pair}
    for original, copy in synthetic.rotated_pairs:
        assert {by_path[original].image_id, by_path[copy].image_id} <= retained
        assert by_path[original].image_id in constrained
    assert all(a != b for a, b in resolution.constraints)
    assert resolution.summary()["retained"] == len(resolution.retained)


def test_confirmed_same_collapses_to_the_named_original(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord], tmp_path: Path
) -> None:
    by_path = _by_path(inventory)
    original, copy = synthetic.rotated_pairs[0]
    ids = [by_path[original].image_id, by_path[copy].image_id]
    for named in (ids[0], ids[1]):
        path = tmp_path / f"reviews-{named[:6]}.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "decisions": [
                        {
                            "ids": ids,
                            "decision": "same",
                            "original": named,
                            "evidence": "viewed side by side",
                        }
                    ],
                }
            )
        )
        resolution = resolve_identity(inventory, load_reviews(path))
        retained = {r.image_id for r in resolution.retained}
        other = ids[1] if named == ids[0] else ids[0]
        assert named in retained and other not in retained
        merged = {m.image_id: m for m in resolution.merged}
        assert merged[other].canonical_id == named
        assert merged[other].reason == "reviewed as the same image"


def test_canonical_defaults_to_largest_area_then_byte_hash(
    inventory: list[InventoryRecord], synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    by_path = _by_path(inventory)
    original, copy = synthetic.rotated_pairs[0]
    small = by_path[original]
    large = replace(by_path[copy], width=small.width * 2)
    records = [large if r.path == copy else r for r in inventory]
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "decisions": [
                    {"ids": [small.image_id, large.image_id], "decision": "same"}
                ],
            }
        )
    )
    resolution = resolve_identity(records, load_reviews(path))
    retained = {r.image_id for r in resolution.retained}
    assert large.image_id in retained and small.image_id not in retained


def test_distinct_decision_removes_the_constraint(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord], tmp_path: Path
) -> None:
    by_path = _by_path(inventory)
    base = resolve_identity(inventory)
    ids = tuple(
        sorted(
            (
                by_path[synthetic.rotated_pairs[0][0]].image_id,
                by_path[synthetic.rotated_pairs[0][1]].image_id,
            )
        )
    )
    assert ids in base.constraints
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {"version": 1, "decisions": [{"ids": list(ids), "decision": "distinct"}]}
        )
    )
    reviewed = resolve_identity(inventory, load_reviews(path))
    assert ids not in reviewed.constraints
    assert {r.image_id for r in reviewed.retained} >= set(ids)


def test_cross_label_confirmed_identity_is_quarantined(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord], tmp_path: Path
) -> None:
    by_path = _by_path(inventory)
    original, copy = synthetic.lookalike_pairs[0]
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "decisions": [
                    {
                        "ids": [by_path[original].image_id, by_path[copy].image_id],
                        "decision": "same",
                    }
                ],
            }
        )
    )
    resolution = resolve_identity(inventory, load_reviews(path))
    reasons = {q.path: q.reason for q in resolution.quarantined}
    assert reasons[original] == reasons[copy] == "confirmed identity across labels"


def test_byte_identical_files_with_conflicting_labels_are_quarantined(
    inventory: list[InventoryRecord],
) -> None:
    first = inventory[0]
    twin = replace(first, path="Other/twin.png", label="Other")
    resolution = resolve_identity([first, twin])
    assert {q.path for q in resolution.quarantined} == {first.path, twin.path}
    assert resolution.retained == ()


def test_review_errors_are_loud(tmp_path: Path, inventory: list) -> None:
    def write(decisions: list, version: int = 1) -> Path:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": version, "decisions": decisions}))
        return path

    ids = [inventory[0].image_id, inventory[1].image_id]
    with pytest.raises(ValueError, match="version"):
        load_reviews(write([], version=2))
    with pytest.raises(ValueError, match="decision"):
        load_reviews(write([{"ids": ids, "decision": "maybe"}]))
    with pytest.raises(ValueError, match="original"):
        load_reviews(write([{"ids": ids, "decision": "same", "original": "x" * 64}]))
    with pytest.raises(ValueError, match="Conflicting"):
        load_reviews(
            write(
                [
                    {"ids": ids, "decision": "same"},
                    {"ids": ids[::-1], "decision": "distinct"},
                ]
            )
        )
    stale = load_reviews(write([{"ids": [ids[0], "f" * 64], "decision": "same"}]))
    with pytest.raises(ValueError, match="unknown image"):
        resolve_identity(inventory, stale)
    assert load_reviews(None) == ()


def test_reviews_digest_ignores_file_formatting_but_not_content(
    tmp_path: Path, inventory: list
) -> None:
    ids = [inventory[0].image_id, inventory[1].image_id]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"version": 1, "decisions": [{"ids": ids, "decision": "same"}]})
    )
    b.write_text(
        json.dumps(
            {"decisions": [{"decision": "same", "ids": ids[::-1]}], "version": 1},
            indent=4,
        )
    )
    assert reviews_digest(load_reviews(a)) == reviews_digest(load_reviews(b))
    b.write_text(
        json.dumps({"version": 1, "decisions": [{"ids": ids, "decision": "distinct"}]})
    )
    assert reviews_digest(load_reviews(a)) != reviews_digest(load_reviews(b))

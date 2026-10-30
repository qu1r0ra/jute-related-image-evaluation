"""Source intake: pinned counts, leaf map, grouped validation and overlap quarantine."""

import re
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.data import (
    PLANTDOC,
    PLANTVILLAGE,
    InventoryRecord,
    SourceMismatchError,
    SourceSpec,
    build_inventory,
    check_overlap,
    derive_validation,
    drop_overlap,
    find_source_root,
    inventory_digest,
    leaf_constraints,
    leaf_key,
    make_synthetic_data,
    observed_counts,
    provenance,
    readiness_blockers,
    scan_source,
    verify_counts,
)

LABELS = ("Apple scab", "Healthy")


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A source package: a wrapper folder holding train/ and test/ partitions."""
    root = tmp_path_factory.mktemp("package") / "Wrapper"
    make_synthetic_data(root / "train", per_class=12, size=64, labels=LABELS, seed=1)
    make_synthetic_data(
        root / "test", per_class=4, rotated_per_class=1, size=64, labels=LABELS, seed=2
    )
    return root.parent


@pytest.fixture(scope="module")
def records(package: Path) -> list[InventoryRecord]:
    return build_inventory(find_source_root(package), workers=2)


@pytest.fixture(scope="module")
def spec(records: list[InventoryRecord]) -> SourceSpec:
    return SourceSpec(
        name="Fixture",
        kaggle_id="fixture/package",
        version=3,
        expected=tuple(sorted(observed_counts(records).items())),
        license_evidence=("fixture",),
    )


def test_the_pinned_plantdoc_counts_are_those_of_the_ticket() -> None:
    assert dict(PLANTDOC.expected)["train"].classes == 28
    assert dict(PLANTDOC.expected)["train"].images == 2670
    assert dict(PLANTDOC.expected)["test"].classes == 27
    assert dict(PLANTDOC.expected)["test"].images == 252
    assert PLANTDOC.version == 7
    assert len(PLANTDOC.license_evidence) == 3
    assert dict(PLANTVILLAGE.expected)["train"].images == 43444
    assert PLANTVILLAGE.version == 1


def test_find_source_root_descends_wrappers_and_rejects_other_trees(
    package: Path, tmp_path: Path
) -> None:
    assert find_source_root(package).name == "Wrapper"
    (tmp_path / "loose").mkdir()
    with pytest.raises(SourceMismatchError):
        find_source_root(tmp_path)


def test_find_source_root_ignores_hidden_marker_folders(tmp_path: Path) -> None:
    (tmp_path / ".complete" / "datasets").mkdir(parents=True)
    (tmp_path / "Package" / "train").mkdir(parents=True)
    assert find_source_root(tmp_path) == tmp_path / "Package"


def test_scan_source_accepts_the_pinned_counts(package: Path, spec: SourceSpec) -> None:
    assert len(scan_source(package, spec, workers=2)) > 0


def test_a_count_mismatch_is_reported_with_expected_and_observed(
    records: list[InventoryRecord], spec: SourceSpec
) -> None:
    changed = dict(spec.expected)
    changed["train"] = replace(changed["train"], images=changed["train"].images + 1)
    wrong = replace(spec, expected=tuple(sorted(changed.items())))
    with pytest.raises(SourceMismatchError, match="expected"):
        verify_counts(records, wrong)


def test_a_missing_partition_is_a_mismatch(
    records: list[InventoryRecord], spec: SourceSpec
) -> None:
    only_train = [r for r in records if r.source_partition == "train"]
    with pytest.raises(SourceMismatchError):
        verify_counts(only_train, spec)


def test_leaf_key_is_the_lowercased_name_after_the_last_separator() -> None:
    assert leaf_key("train/Tomato/Tomato___Early_blight___ABC_123.JPG") == "abc_123"
    assert leaf_key("train/Tomato/plain.jpg") is None


def test_leaf_constraints_pair_images_of_the_same_leaf_and_label(
    records: list[InventoryRecord],
) -> None:
    train = [r for r in records if r.source_partition == "train"][:4]
    named = [
        replace(r, path=f"train/{r.label}/{r.label}___{key}.png")
        for r, key in zip(train, ("a", "b", "c", "d"), strict=True)
    ]
    leaf_map = {"a": "leaf-1", "b": "leaf-1", "c": "leaf-2"}
    pairs = leaf_constraints(named, leaf_map)
    ids = {r.image_id: r for r in named}
    assert len(pairs) <= 1
    for first, second in pairs:
        assert ids[first].label == ids[second].label
        assert named[3].image_id not in (first, second)


def test_derived_validation_is_grouped_disjoint_and_keeps_the_official_test(
    records: list[InventoryRecord],
) -> None:
    split = derive_validation(records, seed=42, fraction=0.25)
    train_ids = {r.image_id for r in split.train}
    val_ids = {r.image_id for r in split.val}
    assert not train_ids & val_ids
    assert {r.image_id for r in split.official_test} == {
        r.image_id for r in records if r.source_partition == "test"
    }
    side = dict.fromkeys(train_ids, "train") | dict.fromkeys(val_ids, "val")
    for first, second in split.constraints:
        assert side[first] == side[second]
    assert split.val
    # byte-identical copies collapse to one canonical image
    kept = len(split.train) + len(split.val) + len(split.quarantined)
    assert kept <= sum(1 for r in records if r.source_partition == "train")


def test_derived_validation_repeats_for_a_seed_and_validates_arguments(
    records: list[InventoryRecord],
) -> None:
    first = derive_validation(records, seed=42)
    again = derive_validation(list(reversed(records)), seed=42)
    assert [r.image_id for r in first.val] == [r.image_id for r in again.val]
    with pytest.raises(ValueError, match="fraction"):
        derive_validation(records, fraction=1.5)
    with pytest.raises(SourceMismatchError, match="train"):
        derive_validation([r for r in records if r.source_partition == "test"])


def test_overlap_finds_exact_and_perceptual_matches_and_drop_removes_them(
    records: list[InventoryRecord], tmp_path: Path
) -> None:
    train = [r for r in records if r.source_partition == "train"]
    target = [train[0], train[1]]
    rotated = _near_copy(train[1], tmp_path)
    report = check_overlap([*train, rotated], target)
    kinds = {m.source_id: m.kind for m in report.matches}
    assert kinds[train[0].image_id] == "byte-identical"
    assert kinds["f" * 64] == "perceptual"
    assert report.summary()["target_checked"] == 2

    pixel = replace(train[2], sha256="0" * 64, pixel_sha256=train[0].pixel_sha256)
    assert {m.kind for m in check_overlap([pixel], target).matches} == {
        "pixel-identical"
    }

    split = derive_validation(records, seed=42)
    cleaned, removed = drop_overlap(split, report)
    assert {r.image_id for r in removed} <= report.source_ids
    remaining = {r.image_id for r in (*cleaned.train, *cleaned.val)}
    assert not remaining & report.source_ids


def test_overlap_without_target_images_matches_nothing(
    records: list[InventoryRecord],
) -> None:
    assert check_overlap(records, []).matches == ()


def test_provenance_records_identity_counts_and_version_rules(
    records: list[InventoryRecord], spec: SourceSpec
) -> None:
    split = derive_validation(records, seed=42)
    result = provenance(spec, records, 3, split, check_overlap(records, []))
    assert result["version_recorded"] == 3
    assert result["observed"]["train"]["images"] == dict(spec.expected)["train"].images
    assert result["source"]["license_evidence"] == ["fixture"]
    with pytest.raises(SourceMismatchError, match="version"):
        provenance(spec, records, 4, split, None)


def test_provenance_requires_an_inventory_pin_and_owner_authorization(
    records: list[InventoryRecord], spec: SourceSpec
) -> None:
    split = derive_validation(records, seed=42)
    blocked = provenance(spec, records, 3, split, None)["readiness"]
    assert blocked["ready"] is False
    assert blocked["blockers"] == [
        "Expected source inventory SHA-256 is not pinned",
        "Owner authorization for internal research use is not recorded",
    ]
    authorized = replace(
        spec,
        expected_inventory_digest=inventory_digest(records),
        owner_internal_use_authorized=True,
        owner_authorization_record="fixture owner authorization",
    )
    assert provenance(authorized, records, 3, split, None)["readiness"] == {
        "ready": True,
        "scope": "internal_research_only",
        "owner_authorized": True,
        "publisher_identity_verified": False,
        "terms_verified": False,
        "blockers": [],
        "caveats": [
            "Publisher identity remains unverified",
            "Source terms remain unverified; owner approval does not grant redistribution rights",
        ],
    }
    claimed = replace(spec, publisher_identity_verified=True)
    assert (
        provenance(claimed, records, 3, split, None)["readiness"]["blockers"][0]
        == "Expected source inventory SHA-256 is not pinned"
    )


def test_a_source_with_an_empty_partition_is_a_mismatch(
    spec: SourceSpec, tmp_path: Path
) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(SourceMismatchError, match="No images"):
        scan_source(tmp_path, spec, workers=1)


def _near_copy(record: InventoryRecord, tmp_path: Path) -> InventoryRecord:
    """Return a record whose bytes and pixels differ but whose perceptual hash matches."""
    return replace(
        record,
        sha256="f" * 64,
        pixel_sha256="e" * 64,
        path=str(tmp_path / "near.png"),
    )


def test_downloaded_sources_are_content_pinned_and_owner_authorized_for_internal_use() -> (
    None
):
    for source in (PLANTDOC, PLANTVILLAGE):
        assert re.fullmatch(r"[0-9a-f]{64}", source.expected_inventory_digest)
        assert source.owner_internal_use_authorized is True
        assert source.owner_authorization_record
        assert source.publisher_identity_verified is False
        assert source.terms_verified is False
        assert readiness_blockers(source) == ()


def test_internal_authorization_does_not_claim_publisher_or_terms_verification(
    records: list[InventoryRecord], spec: SourceSpec
) -> None:
    authorized = replace(
        spec,
        expected_inventory_digest=inventory_digest(records),
        owner_internal_use_authorized=True,
        owner_authorization_record="Owner authorized internal research use",
    )
    split = derive_validation(records, seed=42)

    readiness = provenance(authorized, records, 3, split, None)["readiness"]

    assert readiness == {
        "ready": True,
        "scope": "internal_research_only",
        "owner_authorized": True,
        "publisher_identity_verified": False,
        "terms_verified": False,
        "blockers": [],
        "caveats": [
            "Publisher identity remains unverified",
            "Source terms remain unverified; owner approval does not grant redistribution rights",
        ],
    }

"""Freeze workflow: target freeze, replacement guard, cache probe and source intake."""

import json
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    DECISION_FILE,
    INVENTORY_FILE,
    PLANTDOC,
    REPORT_FILE,
    SOURCES,
    TARGET_DIR,
    FreezeBlockedError,
    PreprocessingSpec,
    SourceMismatchError,
    SyntheticPlan,
    build_inventory,
    cache_root,
    freeze_target,
    intake_source,
    inventory_digest,
    load_manifest,
    load_source_manifest,
    make_synthetic_data,
    observed_counts,
    probe_cache,
)


@pytest.fixture(scope="module")
def rules_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    data["min_support"] = 2
    path = tmp_path_factory.mktemp("rules") / "rules.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _freeze(synthetic: SyntheticPlan, rules_path: Path, out: Path, **kwargs) -> dict:
    return freeze_target(
        out,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
        **kwargs,
    )


def test_freeze_writes_inventory_manifest_receipt_and_report(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    report = _freeze(synthetic, rules_path, tmp_path)
    assert (tmp_path / INVENTORY_FILE).is_file()
    assert (
        json.loads((tmp_path / REPORT_FILE).read_text())["receipt"]
        == (report["receipt"])
    )
    manifest = load_manifest(tmp_path / TARGET_DIR)
    assert report["receipt"]["manifest_digest"] == manifest.digest()
    assert report["unchanged"] is False
    assert "preprocessing" in report["receipt"]["dependencies"]
    assert report["receipt"]["dependencies"]["preprocessing"] == (
        PreprocessingSpec().digest()
    )


def test_refreezing_the_same_inputs_is_idempotent(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    first = _freeze(synthetic, rules_path, tmp_path)
    again = _freeze(synthetic, rules_path, tmp_path)
    assert again["unchanged"] is True
    assert again["receipt"] == first["receipt"]


def test_a_different_completed_freeze_is_kept_unless_forced(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    first = _freeze(synthetic, rules_path, tmp_path)
    with pytest.raises(FreezeBlockedError, match="--force"):
        _freeze(synthetic, rules_path, tmp_path, seed=7)
    kept = load_manifest(tmp_path / TARGET_DIR)
    assert kept.digest() == first["receipt"]["manifest_digest"]
    replaced = _freeze(synthetic, rules_path, tmp_path, seed=7, force=True)
    assert replaced["receipt"]["manifest_digest"] != first["receipt"]["manifest_digest"]


def test_an_unsupported_dataset_blocks_the_freeze(
    synthetic: SyntheticPlan, tmp_path: Path
) -> None:
    with pytest.raises(FreezeBlockedError):
        freeze_target(
            tmp_path,
            image_root=synthetic.root,
            reviews_path=None,
            workers=2,
        )
    assert not (tmp_path / TARGET_DIR).exists()


def test_cache_probe_records_a_decision_and_never_reads_test(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    _freeze(synthetic, rules_path, tmp_path)
    decision = probe_cache(tmp_path, image_root=synthetic.root, workers=2)
    assert decision["decision"] == "adopt-cache"
    root = cache_root(tmp_path / "image_cache", PreprocessingSpec())
    assert (root / DECISION_FILE).is_file()
    manifest = load_manifest(tmp_path / TARGET_DIR)
    cached = {p.stem for p in root.glob("*.png")}
    assert cached == {
        e.image_id for name in ("train", "val") for e in manifest.partition(name)
    }


def test_cache_probe_requires_a_completed_freeze(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_cache(tmp_path)


def test_source_intake_requires_a_completed_freeze(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        intake_source(
            "plantdoc",
            tmp_path,
            tmp_path,
            package_version=7,
            package_digest=PLANTDOC.expected_inventory_digest,
        )


def test_source_intake_rejects_a_version_outside_the_source_pin(tmp_path: Path) -> None:
    with pytest.raises(SourceMismatchError, match="pinned version 7"):
        intake_source(
            "plantdoc",
            tmp_path,
            tmp_path,
            package_version=6,
            package_digest="0" * 64,
        )


@pytest.mark.parametrize("digest", ["", "not-a-digest", "0" * 63, "g" * 64])
def test_source_intake_validates_the_digest_before_scanning(
    tmp_path: Path, digest: str
) -> None:
    with pytest.raises(SourceMismatchError, match="digest"):
        intake_source(
            "plantdoc",
            tmp_path / "missing-package",
            tmp_path / "missing-freeze",
            package_version=7,
            package_digest=digest,
        )


def test_source_intake_rejects_an_unreadable_leaf_map(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "freeze"
    _freeze(synthetic, rules_path, output)
    leaf_map = tmp_path / "leaf.json"
    leaf_map.write_text("{not json", encoding="utf-8")
    with pytest.raises(FreezeBlockedError, match="Leaf map"):
        intake_source(
            "plantdoc",
            tmp_path / "missing-package",
            output,
            package_version=7,
            package_digest=PLANTDOC.expected_inventory_digest,
            leaf_map_path=leaf_map,
        )


def test_source_intake_rejects_a_package_with_other_counts(
    synthetic: SyntheticPlan, rules_path: Path, tmp_path: Path
) -> None:
    _freeze(synthetic, rules_path, tmp_path / "freeze")
    package = tmp_path / "plantdoc"
    make_synthetic_data(package / "train", per_class=6, labels=("a", "b"))
    make_synthetic_data(package / "test", per_class=4, labels=("a", "b"), seed=3)
    with pytest.raises(SourceMismatchError, match="pinned identity"):
        intake_source(
            "plantdoc",
            package,
            tmp_path / "freeze",
            package_version=7,
            package_digest=PLANTDOC.expected_inventory_digest,
        )
    assert not (tmp_path / "freeze" / "source_plantdoc.json").exists()


def test_source_names_are_pinned_case_insensitively() -> None:
    with pytest.raises(KeyError):
        intake_source(
            "nonexistent",
            Path("."),
            Path("."),
            package_version=1,
            package_digest="0" * 64,
        )
    assert PLANTDOC.name.lower() == "plantdoc"


def test_source_freeze_excludes_target_overlap_and_pins_content(
    synthetic: SyntheticPlan,
    rules_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "freeze"
    _freeze(synthetic, rules_path, output)
    target = load_manifest(output / TARGET_DIR)
    package = tmp_path / "package"
    make_synthetic_data(package / "train", per_class=20, labels=("a", "b"), seed=100)
    make_synthetic_data(package / "test", per_class=10, labels=("a", "b"), seed=200)
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    overlap_path = next((package / "train" / "a").glob("*.png"))
    copyfile(synthetic.root / target.partition("val")[0].path, overlap_path)
    records = build_inventory(package, workers=2)
    digest = inventory_digest(records)
    spec = replace(
        PLANTDOC,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=digest,
    )
    monkeypatch.setitem(SOURCES, "plantdoc", spec)
    with pytest.raises(SourceMismatchError, match="tracked content pin"):
        intake_source(
            "plantdoc", package, output, package_version=7, package_digest="0" * 64
        )
    report = intake_source(
        "plantdoc", package, output, package_version=7, package_digest=digest
    )
    assert report["package_inventory_digest"] == digest
    assert report["readiness"] == {
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
    frozen = load_manifest(output / "source_plantdoc")
    verified = load_source_manifest(
        output / "source_plantdoc", package, output / TARGET_DIR
    )
    assert verified.digest() == frozen.digest()
    assert overlap_path.relative_to(package).as_posix() in report["removed"]
    assert overlap_path.relative_to(package).as_posix() not in {
        e.path for e in frozen.entries
    }
    assert dict(frozen.dependencies)["target_manifest"] == target.digest()
    assert not {item["image_id"] for item in frozen.quarantined} & {
        entry.image_id for entry in frozen.entries
    }
    report_path = output / "source_plantdoc.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["version_recorded"] = 8
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(FreezeBlockedError, match="provenance"):
        load_source_manifest(output / "source_plantdoc", package, output / TARGET_DIR)

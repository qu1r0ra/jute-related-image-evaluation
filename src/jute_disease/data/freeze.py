"""The freeze workflow behind ``jute-disease freeze``.

Three steps share one output directory:

* :func:`freeze_target` fingerprints the target images, applies the curation
  rules and identity policy, and writes the manifest with its receipt last.
* :func:`probe_cache` builds the bounded train/validation image cache and records
  the cache decision from measured time and disk use.
* :func:`intake_source` records a pinned source package, derives its grouped
  validation split and quarantines source images that match the frozen target
  validation or test images.

None of them fits a model or reads test pixels for any purpose but fingerprinting.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from jute_disease.config import (
    CURATION_RULES_PATH,
    DEFAULT_SEED,
    IDENTITY_REVIEWS_PATH,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data.cache import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_BYTES,
    build_image_cache,
    cache_root,
    decide_cache,
    write_decision,
)
from jute_disease.data.curation import load_rules
from jute_disease.data.identity import load_reviews
from jute_disease.data.inventory import (
    build_inventory,
    inventory_digest,
    read_inventory,
    write_inventory,
)
from jute_disease.data.manifest import (
    FreezeBlockedError,
    Manifest,
    ManifestEntry,
    build_groups,
    build_manifest,
    load_manifest,
    receipt_for,
    summarize_pad_fractions,
    validate_manifest,
    write_manifest,
)
from jute_disease.data.sources import (
    PLANTDOC,
    PLANTVILLAGE,
    SourceMismatchError,
    SourceSpec,
    check_overlap,
    derive_validation,
    drop_overlap,
    find_source_root,
    leaf_constraints,
    provenance,
    readiness_blockers,
    scan_source,
)
from jute_disease.data.transforms import PreprocessingSpec, letterbox_pad_fraction

TARGET_DIR = "target"
INVENTORY_FILE = "inventory.jsonl"
REPORT_FILE = "freeze_report.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCES = {"plantdoc": PLANTDOC, "plantvillage": PLANTVILLAGE}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def freeze_target(
    output_dir: Path,
    image_root: Path = SOURCE_INVENTORY_DIR,
    rules_path: Path = CURATION_RULES_PATH,
    reviews_path: Path | None = IDENTITY_REVIEWS_PATH,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
    force: bool = False,
) -> dict:
    """Freeze the target dataset and return its report.

    The original images under ``image_root`` are only read. An existing completed
    freeze that differs from the new one is kept unless ``force`` is set.

    Raises:
        FreezeBlockedError: the dataset cannot be frozen as specified, or a different
            completed freeze already exists.
    """
    output_dir = Path(output_dir)
    rules = load_rules(rules_path)
    reviews = load_reviews(reviews_path)
    records = build_inventory(image_root, workers=workers)
    manifest, curation, resolution = build_manifest(
        records,
        rules,
        reviews,
        seed=seed,
        extra_dependencies={"preprocessing": PreprocessingSpec().digest()},
    )
    target = output_dir / TARGET_DIR
    try:
        existing = load_manifest(target)
    except FileNotFoundError, FreezeBlockedError:
        existing = None
    if existing is not None and existing.digest() != manifest.digest() and not force:
        raise FreezeBlockedError(
            f"A different completed freeze exists in {target}; "
            "inspect it, then rerun with --force to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_inventory(records, output_dir / INVENTORY_FILE)
    write_manifest(manifest, target)
    report = {
        "image_root": str(image_root),
        "receipt": receipt_for(manifest),
        "curation": curation.summary(),
        "identity": resolution.summary(),
        "pad_fraction_by_source_and_class": manifest.pad_fraction_summary(),
        "unchanged": existing is not None and existing.digest() == manifest.digest(),
    }
    _write_json(output_dir / REPORT_FILE, report)
    return report


def probe_cache(
    output_dir: Path,
    image_root: Path = SOURCE_INVENTORY_DIR,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    workers: int = 4,
) -> dict:
    """Build the bounded train/validation cache and record the cache decision."""
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir / TARGET_DIR)
    spec = PreprocessingSpec()
    report = build_image_cache(
        manifest,
        image_root,
        output_dir / "image_cache",
        spec,
        budget_seconds=budget_seconds,
        workers=workers,
    )
    decision = decide_cache(report, max_bytes=max_bytes)
    root = cache_root(output_dir / "image_cache", spec)
    write_decision(decision, root)
    return decision


def _source_report_path(manifest_dir: Path) -> Path:
    """Return the provenance sidecar path paired with a source manifest."""
    return manifest_dir.with_name(f"{manifest_dir.name}.json")


def load_source_manifest(
    manifest_dir: Path | str,
    image_root: Path | str,
    target_manifest_dir: Path | str,
    *,
    content_verifier: Any | None = None,
) -> Manifest:
    """Load a source manifest after checking identity, lineage and retained bytes.

    Only the source training and validation files are read or hashed here.

    Raises:
        FileNotFoundError: either manifest is missing.
        FreezeBlockedError: the intake receipt or dependency lineage is invalid.
        SourceMismatchError: a retained source image no longer matches its hash.
    """
    manifest_dir = Path(manifest_dir)
    manifest = load_manifest(manifest_dir)
    target = load_manifest(Path(target_manifest_dir))
    dependencies = dict(manifest.dependencies)
    if not dependencies.get("source_inventory") or not dependencies.get(
        "source_provenance"
    ):
        raise FreezeBlockedError("Source manifest lacks intake provenance")
    if dependencies.get("target_manifest") != target.digest():
        raise FreezeBlockedError("Source manifest does not match the frozen target")

    report_path = _source_report_path(manifest_dir)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeBlockedError(
            f"Source intake report is unavailable: {report_path}"
        ) from error
    if not isinstance(report, dict):
        raise FreezeBlockedError("Source intake report is not a JSON object")
    report_for_digest = dict(report)
    if report_for_digest.pop("manifest_digest", None) != manifest.digest():
        raise FreezeBlockedError("Source intake report does not match its manifest")
    report_for_digest.pop("source_root", None)
    report_digest = hashlib.sha256(
        json.dumps(report_for_digest, sort_keys=True).encode()
    ).hexdigest()
    if report_digest != dependencies["source_provenance"]:
        raise FreezeBlockedError("Source intake provenance changed")

    source = report.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("name"), str):
        raise FreezeBlockedError("Source intake report lacks a pinned source")
    spec = SOURCES.get(source["name"].lower())
    if spec is None or source != spec.to_json():
        raise FreezeBlockedError("Source intake does not match the pinned source")
    if report.get("version_recorded") != spec.version:
        raise FreezeBlockedError("Source intake version differs from the pinned source")
    if report.get("source_url") != f"https://www.kaggle.com/datasets/{spec.kaggle_id}":
        raise FreezeBlockedError("Source intake URL differs from the pinned source")
    if report.get("package_inventory_digest") != spec.expected_inventory_digest:
        raise FreezeBlockedError("Source intake differs from the tracked content pin")
    if report.get("package_inventory_digest") != dependencies["source_inventory"]:
        raise FreezeBlockedError("Source intake inventory differs from its manifest")
    support = manifest.support()
    if blockers := readiness_blockers(
        spec,
        required_labels=manifest.labels,
        validation_labels=tuple(support["val"]),
    ):
        raise FreezeBlockedError("; ".join(blockers))

    from jute_disease.data.readers import ManifestContentVerifier

    verifier = content_verifier or ManifestContentVerifier()
    try:
        verifier.verify(manifest, image_root, ("train", "val"))
    except ValueError as error:
        raise SourceMismatchError(str(error)) from error
    return manifest


def _check_package_identity(
    spec: SourceSpec, package_version: int, package_digest: str
) -> str:
    """Validate the package inputs against the tracked version and content pin."""
    if package_version != spec.version:
        raise SourceMismatchError(
            f"{spec.name} package version {package_version} differs from "
            f"the pinned version {spec.version}"
        )
    digest = package_digest.strip().lower()
    if not _SHA256.fullmatch(digest):
        raise SourceMismatchError(
            f"Package digest must be a 64-character hex SHA-256: {package_digest!r}"
        )
    if spec.expected_inventory_digest is None:
        raise FreezeBlockedError(f"{spec.name} has no tracked source inventory SHA-256")
    if digest != spec.expected_inventory_digest.lower():
        raise SourceMismatchError(
            f"{spec.name} package digest differs from its tracked content pin"
        )
    return digest


def _read_leaf_map(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        leaf_map = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeBlockedError(f"Leaf map is unreadable: {path}") from error
    if not isinstance(leaf_map, dict):
        raise FreezeBlockedError(f"Leaf map is not a JSON object: {path}")
    return leaf_map


def intake_source(
    name: str,
    source_root: Path,
    output_dir: Path,
    *,
    package_version: int,
    package_digest: str,
    leaf_map_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    workers: int = 4,
) -> dict:
    """Record a source package and check it against the frozen target.

    The package version and inventory digest are checked against the tracked
    source spec before scanning. Owner approval permits internal research use;
    unresolved publisher identity and terms remain explicit report caveats.

    Raises:
        KeyError: ``name`` is not a known source.
        FreezeBlockedError: the target or source intake prerequisites are invalid.
        SourceMismatchError: the package does not match its pinned identity.
        FileNotFoundError: the target has not been frozen yet.
    """
    spec: SourceSpec = SOURCES[name.lower()]
    expected_digest = _check_package_identity(spec, package_version, package_digest)
    leaf_map = _read_leaf_map(leaf_map_path)
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir / TARGET_DIR)
    held_out = {
        e.image_id for name_ in ("val", "test") for e in manifest.partition(name_)
    }
    target_inventory = read_inventory(output_dir / INVENTORY_FILE)
    if inventory_digest(target_inventory) != dict(manifest.dependencies).get(
        "inventory"
    ):
        raise FreezeBlockedError("Target inventory does not match its manifest")
    target = [r for r in target_inventory if r.image_id in held_out]
    if {r.image_id for r in target} != held_out:
        raise FreezeBlockedError("Target inventory is missing held-out images")

    records = scan_source(source_root, spec, workers=workers)
    digest = inventory_digest(records)
    if digest != expected_digest:
        raise SourceMismatchError(
            f"Package inventory digest mismatch: expected {expected_digest!r}, "
            f"observed {digest}"
        )
    train_records = [r for r in records if r.source_partition == "train"]
    extra = () if leaf_map is None else leaf_constraints(train_records, leaf_map)
    split = derive_validation(records, seed=seed, extra_constraints=extra)
    # Preserve the publisher's held-out partition; withhold source-training
    # candidates that could otherwise put the same photo on both sides.
    official_overlap = check_overlap(
        list(split.train + split.val), list(split.official_test)
    )
    official_bad = official_overlap.source_ids
    split = replace(
        split,
        train=tuple(r for r in split.train if r.image_id not in official_bad),
        val=tuple(r for r in split.val if r.image_id not in official_bad),
    )
    overlap = check_overlap(records, target)
    split, removed = drop_overlap(split, overlap)

    result = provenance(spec, records, package_version, split, overlap, leaf_map)
    result["removed"] = [r.path for r in removed]
    result["overlap_matches"] = [vars(match) for match in overlap.matches]
    result["official_overlap"] = official_overlap.summary()
    result["official_overlap_matches"] = [vars(m) for m in official_overlap.matches]
    result["package_inventory_digest"] = digest
    result["source_url"] = f"https://www.kaggle.com/datasets/{spec.kaggle_id}"
    result["original_class_map"] = {
        r.label: r.label for r in sorted(records, key=lambda item: item.label)
    }
    retained = (*split.train, *split.val, *split.official_test)
    ids = {r.image_id for r in retained}
    constraints = tuple(pair for pair in split.constraints if set(pair) <= ids)
    # Keep transitive components intact even when an intermediate image is
    # withheld; otherwise surviving endpoints could be separated in later folds.
    group_ids = ids | {image_id for pair in split.constraints for image_id in pair}
    groups = build_groups(sorted(group_ids), split.constraints)
    entries = tuple(
        sorted(
            (
                ManifestEntry(
                    r.image_id,
                    r.path,
                    r.label,
                    side,
                    groups[r.image_id],
                    spec.name,
                    r.width,
                    r.height,
                    letterbox_pad_fraction(r.width, r.height),
                )
                for side, members in (
                    ("train", split.train),
                    ("val", split.val),
                    ("test", split.official_test),
                )
                for r in members
            ),
            key=lambda entry: entry.image_id,
        )
    )
    expected_labels = {r.label for r in records if r.source_partition == "train"}
    if {r.label for r in retained} != expected_labels:
        raise FreezeBlockedError("Source intake lost one or more training classes")
    result["pad_fraction_by_source_and_class"] = summarize_pad_fractions(entries)
    provenance_digest = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    frozen = Manifest(
        seed=seed,
        fractions=(("train", 0.85), ("val", 0.15), ("test", 0.0)),
        entries=entries,
        excluded=tuple(
            [{"path": r.path, "reason": "target overlap"} for r in removed]
            + [
                {"path": m.source_path, "reason": "official held-out overlap"}
                for m in official_overlap.matches
            ]
        ),
        merged=(),
        quarantined=split.quarantined,
        constraints=constraints,
        dependencies=tuple(
            sorted(
                {
                    "source_inventory": digest,
                    "source_provenance": provenance_digest,
                    "target_manifest": manifest.digest(),
                }.items()
            )
        ),
    )
    validate_manifest(frozen, 1)
    result["manifest_digest"] = frozen.digest()
    result["source_root"] = str(find_source_root(source_root).resolve())
    manifest_dir = output_dir / f"source_{spec.name.lower()}"
    _write_json(_source_report_path(manifest_dir), result)
    write_manifest(frozen, manifest_dir)
    return result

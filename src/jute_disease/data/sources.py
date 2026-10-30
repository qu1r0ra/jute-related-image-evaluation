"""Optional source datasets: pinned intake, derived validation and overlap checks.

Source data (PlantVillage, PlantDoc) only pretrains; it never defines the
target split. Intake therefore records exactly which package was read, keeps the
publisher's partitions untouched, derives a grouped validation split from the
publisher's training images, and quarantines (source-side) any image that
matches a frozen target validation or test image. The target manifest is never
changed by source data.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jute_disease.config import DEFAULT_SEED
from jute_disease.data.fingerprints import PHASH_THRESHOLD
from jute_disease.data.identity import resolve_identity
from jute_disease.data.inventory import InventoryRecord, build_inventory
from jute_disease.data.manifest import PARTITIONS, _assign_label, build_groups

SOURCE_VALIDATION_FRACTION = 0.15
_BLOCK = 128
_LEAF_SEPARATOR = "___"


class SourceMismatchError(ValueError):
    """A source package does not match its pinned identity."""


@dataclass(frozen=True)
class PartitionCounts:
    """Expected class and image counts of one official partition."""

    classes: int
    images: int


@dataclass(frozen=True)
class SourceSpec:
    """Pinned source metadata and its readiness evidence for pretraining."""

    name: str
    kaggle_id: str
    version: int
    expected: tuple[tuple[str, PartitionCounts], ...]
    license_evidence: tuple[str, ...]
    expected_inventory_digest: str | None = None
    owner_internal_use_authorized: bool = False
    owner_authorization_record: str = ""
    # References that establish the Kaggle package identity independently.
    publisher_identity_evidence: tuple[str, ...] = ()
    publisher_identity_verified: bool = False
    terms_verified: bool = False

    def to_json(self) -> dict:
        """Return a JSON-safe mapping for provenance records."""
        data = asdict(self)
        data["expected"] = {name: asdict(counts) for name, counts in self.expected}
        data["license_evidence"] = list(self.license_evidence)
        data["publisher_identity_evidence"] = list(self.publisher_identity_evidence)
        return data


PLANTDOC = SourceSpec(
    name="PlantDoc",
    kaggle_id="nirmalsankalana/plantdoc-dataset",
    version=7,
    expected=(
        ("train", PartitionCounts(classes=28, images=2670)),
        ("test", PartitionCounts(classes=27, images=252)),
    ),
    expected_inventory_digest="06f9782c9be1e9d6c707845c3f07b183c75e7421dbb112a88d60cb2d512ad1fd",
    license_evidence=(
        "Kaggle metadata record: CC0",
        "Kaggle description: CC BY 4.0",
        "The project owner authorized internal research use for #717. The two "
        "license statements conflict and remain unresolved; redistribution is "
        "not authorized.",
    ),
    owner_internal_use_authorized=True,
    owner_authorization_record="Owner instruction for #717 implementation, 2026-10-30",
    publisher_identity_evidence=(),
    publisher_identity_verified=False,
    terms_verified=False,
)

PLANTVILLAGE = SourceSpec(
    name="PlantVillage",
    kaggle_id="mohitsingh1804/plantvillage",
    version=1,
    expected=(
        ("train", PartitionCounts(classes=38, images=43444)),
        ("val", PartitionCounts(classes=38, images=10861)),
    ),
    expected_inventory_digest="d6db080933a47f289fb15bc60c8c42ef1c17c7656d3700d4133e6d313fdf471a",
    license_evidence=(
        "Upstream dataset card declares cc-by-sa-3.0; the upstream repository has "
        "no LICENSE file and its README states no license.",
        "The project owner authorized internal research use for #717. The Kaggle "
        "re-package's terms remain unverified; redistribution is not authorized.",
    ),
    owner_internal_use_authorized=True,
    owner_authorization_record="Owner instruction for #717 implementation, 2026-10-30",
    publisher_identity_evidence=(),
    publisher_identity_verified=False,
    terms_verified=False,
)


def readiness_blockers(
    spec: SourceSpec,
    *,
    required_labels: tuple[str, ...] = (),
    validation_labels: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return what still prevents valid owner-authorized internal pretraining."""
    blockers = []
    digest = spec.expected_inventory_digest
    if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest.lower()) is None:
        blockers.append("Expected source inventory SHA-256 is not pinned")
    if not spec.owner_internal_use_authorized or not spec.owner_authorization_record:
        blockers.append("Owner authorization for internal research use is not recorded")
    validation = set(validation_labels)
    blockers.extend(
        f"Source validation has no support for class {label!r}"
        for label in sorted(set(required_labels) - validation)
    )
    return tuple(blockers)


def observed_counts(records: list[InventoryRecord]) -> dict[str, PartitionCounts]:
    """Return class and image counts per official partition."""
    labels: dict[str, set[str]] = {}
    images: dict[str, int] = {}
    for record in records:
        name = record.source_partition or "unpartitioned"
        labels.setdefault(name, set()).add(record.label)
        images[name] = images.get(name, 0) + 1
    return {
        name: PartitionCounts(classes=len(labels[name]), images=images[name])
        for name in sorted(images)
    }


def verify_counts(records: list[InventoryRecord], spec: SourceSpec) -> None:
    """Check observed partitions and counts against the pinned expectation.

    Raises:
        SourceMismatchError: a partition is missing, extra or has other counts.
    """
    observed = observed_counts(records)
    expected = dict(spec.expected)
    if observed != expected:
        raise SourceMismatchError(
            f"{spec.name} does not match its pinned identity: "
            f"expected {expected}, observed {observed}"
        )


def find_source_root(path: Path) -> Path:
    """Descend single-folder wrappers until the official partitions are found.

    Hidden folders (such as the downloader's ``.complete`` marker) are ignored.
    """
    current = Path(path)
    for _ in range(4):
        children = [
            p for p in current.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        if any(p.name in PARTITIONS for p in children):
            return current
        if len(children) != 1:
            break
        current = children[0]
    raise SourceMismatchError(f"No train/val/test folders found under {path}")


def scan_source(
    root: Path, spec: SourceSpec, workers: int = 4
) -> list[InventoryRecord]:
    """Fingerprint a source package and verify it against ``spec``.

    Raises:
        SourceMismatchError: the package layout, images or counts do not match.
    """
    source_root = find_source_root(root)
    try:
        records = build_inventory(source_root, workers=workers)
    except ValueError as error:
        raise SourceMismatchError(
            f"{spec.name} package is unusable: {error}"
        ) from error
    verify_counts(records, spec)
    return records


def leaf_key(path: str) -> str | None:
    """Return the leaf-map key of a PlantVillage filename, if it has one.

    The key is the basename after the last ``___``, lowercased and without its
    extension; files without the separator have no key.
    """
    name = path.rsplit("/", 1)[-1]
    if _LEAF_SEPARATOR not in name:
        return None
    return re.sub(r"\.[^.]+$", "", name.rsplit(_LEAF_SEPARATOR, 1)[1]).lower()


def leaf_constraints(
    records: list[InventoryRecord], leaf_map: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    """Pair images that the leaf map says come from the same physical leaf.

    ``leaf_map`` maps a lowercased key (see :func:`leaf_key`) to a leaf id. Files
    whose key is absent stay ungrouped here; the perceptual check still covers
    them.
    """
    members: dict[tuple[str, str], list[str]] = {}
    for record in records:
        key = leaf_key(record.path)
        if key is None or key not in leaf_map:
            continue
        members.setdefault((record.label, leaf_map[key]), []).append(record.image_id)
    pairs: list[tuple[str, str]] = []
    for ids in members.values():
        ordered = sorted(set(ids))
        pairs.extend((ordered[0], other) for other in ordered[1:])
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class SourceSplit:
    """Official test kept as published, plus a derived grouped validation split."""

    train: tuple[InventoryRecord, ...]
    val: tuple[InventoryRecord, ...]
    official_test: tuple[InventoryRecord, ...]
    quarantined: tuple[dict, ...]
    constraints: tuple[tuple[str, str], ...]
    seed: int
    fraction: float

    def summary(self) -> dict:
        """Return counts for provenance records."""
        return {
            "train": len(self.train),
            "val": len(self.val),
            "official_test": len(self.official_test),
            "quarantined": len(self.quarantined),
            "constraints": len(self.constraints),
            "seed": self.seed,
            "fraction": self.fraction,
        }


def derive_validation(
    records: list[InventoryRecord],
    seed: int = DEFAULT_SEED,
    fraction: float = SOURCE_VALIDATION_FRACTION,
    extra_constraints: tuple[tuple[str, str], ...] = (),
) -> SourceSplit:
    """Split the publisher's training images into train and grouped validation.

    The publisher's ``test`` (or ``val``) partition is kept as published. A
    publisher ``val`` partition, when present, is treated as the official
    held-out partition and is not used to derive anything. Images that are
    exact or unresolved cross-label matches are quarantined, and whole identity
    groups are assigned to one side.

    Raises:
        ValueError: ``fraction`` is not in (0, 1).
        SourceMismatchError: there are no official training images.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    official = tuple(r for r in records if r.source_partition in ("test", "val"))
    training = [r for r in records if r.source_partition == "train"]
    if not training:
        raise SourceMismatchError("The source has no official train partition")
    resolution = resolve_identity(training, ())
    retained = list(resolution.retained)
    ids = [r.image_id for r in retained]
    constraints = tuple(
        sorted(
            {
                *resolution.constraints,
                *(c for c in extra_constraints if c[0] in ids and c[1] in ids),
            }
        )
    )
    group_of = build_groups(ids, constraints)
    by_label: dict[str, dict[str, int]] = {}
    for record in retained:
        groups = by_label.setdefault(record.label, {})
        group = group_of[record.image_id]
        groups[group] = groups.get(group, 0) + 1
    fractions = {"train": 1 - fraction, "val": fraction, "test": 0.0}
    placement: dict[str, str] = {}
    for label in sorted(by_label):
        placement.update(_assign_label(by_label[label], seed, fractions))
    side = {
        name: tuple(r for r in retained if placement[group_of[r.image_id]] == name)
        for name in ("train", "val")
    }
    return SourceSplit(
        train=side["train"],
        val=side["val"],
        official_test=official,
        quarantined=tuple(vars(item) for item in resolution.quarantined),
        constraints=constraints,
        seed=seed,
        fraction=fraction,
    )


@dataclass(frozen=True)
class OverlapMatch:
    """A source image that matches a frozen target image."""

    source_id: str
    source_path: str
    target_id: str
    kind: str
    distance: int


@dataclass(frozen=True)
class OverlapReport:
    """Source images that must leave the source data because they match the target."""

    matches: tuple[OverlapMatch, ...]
    threshold: int
    source_checked: int
    target_checked: int

    @property
    def source_ids(self) -> frozenset[str]:
        """Image ids of every source image to quarantine."""
        return frozenset(m.source_id for m in self.matches)

    def summary(self) -> dict:
        """Return counts for provenance records."""
        kinds: dict[str, int] = {}
        for match in self.matches:
            kinds[match.kind] = kinds.get(match.kind, 0) + 1
        return {
            "source_checked": self.source_checked,
            "target_checked": self.target_checked,
            "threshold": self.threshold,
            "matches": len(self.matches),
            "source_images_quarantined": len(self.source_ids),
            "by_kind": dict(sorted(kinds.items())),
        }


def check_overlap(
    source: list[InventoryRecord],
    target: list[InventoryRecord],
    threshold: int = PHASH_THRESHOLD,
) -> OverlapReport:
    """Find source images that match target images exactly or perceptually.

    Each source image gets at most one match per kind, against its closest
    target image. The target is only read, never changed.
    """
    by_bytes = {r.sha256: r for r in target}
    by_pixels = {r.pixel_sha256: r for r in target}
    matches: list[OverlapMatch] = []
    for record in source:
        if record.sha256 in by_bytes:
            matches.append(
                OverlapMatch(
                    record.image_id,
                    record.path,
                    by_bytes[record.sha256].image_id,
                    "byte-identical",
                    0,
                )
            )
        elif record.pixel_sha256 in by_pixels:
            matches.append(
                OverlapMatch(
                    record.image_id,
                    record.path,
                    by_pixels[record.pixel_sha256].image_id,
                    "pixel-identical",
                    0,
                )
            )
    exact = {m.source_id for m in matches}
    remaining = [r for r in source if r.image_id not in exact]
    if remaining and target:
        target_variants = np.array([r.phash for r in target], dtype=np.uint64)
        target_base = target_variants[:, 0]
        for start in range(0, len(remaining), _BLOCK):
            chunk = remaining[start : start + _BLOCK]
            variants = np.array([r.phash for r in chunk], dtype=np.uint64)
            forward = np.bitwise_count(
                variants[:, 0, None, None] ^ target_variants[None]
            ).min(axis=2)
            backward = np.bitwise_count(
                variants[:, None, :] ^ target_base[None, :, None]
            ).min(axis=2)
            distance = np.minimum(forward, backward)
            nearest = distance.argmin(axis=1)
            for local, record in enumerate(chunk):
                closest = int(distance[local, nearest[local]])
                if closest <= threshold:
                    matches.append(
                        OverlapMatch(
                            record.image_id,
                            record.path,
                            target[int(nearest[local])].image_id,
                            "perceptual",
                            closest,
                        )
                    )
    return OverlapReport(
        matches=tuple(sorted(matches, key=lambda m: (m.source_id, m.kind))),
        threshold=threshold,
        source_checked=len(source),
        target_checked=len(target),
    )


def drop_overlap(
    split: SourceSplit, report: OverlapReport
) -> tuple[SourceSplit, tuple[InventoryRecord, ...]]:
    """Remove overlapping images from every side of a source split.

    Returns the cleaned split and the removed records.
    """
    bad = report.source_ids

    def keep(records: tuple[InventoryRecord, ...]) -> tuple[InventoryRecord, ...]:
        return tuple(r for r in records if r.image_id not in bad)

    removed = tuple(
        r for r in (*split.train, *split.val, *split.official_test) if r.image_id in bad
    )
    cleaned = SourceSplit(
        train=keep(split.train),
        val=keep(split.val),
        official_test=keep(split.official_test),
        quarantined=split.quarantined,
        constraints=split.constraints,
        seed=split.seed,
        fraction=split.fraction,
    )
    return cleaned, removed


def provenance(
    spec: SourceSpec,
    records: list[InventoryRecord],
    observed_version: int,
    split: SourceSplit,
    overlap: OverlapReport | None,
    leaf_map: Mapping[str, str] | None = None,
) -> dict:
    """Return a JSON-safe intake record: identity, caveats, counts and checks."""
    version = observed_version
    if observed_version != spec.version:
        raise SourceMismatchError(
            f"{spec.name} version {observed_version} differs from pinned {spec.version}"
        )
    keys = [leaf_key(r.path) for r in records]
    owner_authorized = bool(
        spec.owner_internal_use_authorized and spec.owner_authorization_record
    )
    required_labels = tuple(
        sorted(
            {record.label for record in records if record.source_partition == "train"}
        )
    )
    validation_labels = tuple(sorted({record.label for record in split.val}))
    blockers = readiness_blockers(
        spec,
        required_labels=required_labels,
        validation_labels=validation_labels,
    )
    publisher_identity_verified = bool(
        spec.publisher_identity_verified and spec.publisher_identity_evidence
    )
    return {
        "source": spec.to_json(),
        "version_recorded": version,
        "observed": {
            name: asdict(counts) for name, counts in observed_counts(records).items()
        },
        "split": split.summary(),
        "readiness": {
            "ready": not blockers,
            "scope": "internal_research_only",
            "owner_authorized": owner_authorized,
            "publisher_identity_verified": publisher_identity_verified,
            "terms_verified": spec.terms_verified,
            "blockers": list(blockers),
            "caveats": [
                caveat
                for unresolved, caveat in (
                    (
                        not publisher_identity_verified,
                        "Publisher identity remains unverified",
                    ),
                    (
                        not spec.terms_verified,
                        "Source terms remain unverified; owner approval does not "
                        "grant redistribution rights",
                    ),
                )
                if unresolved
            ],
        },
        "overlap": None if overlap is None else overlap.summary(),
        "leaf_map": None
        if leaf_map is None
        else {
            "key_rule": "lowercased basename after the last '___', no extension",
            "files_with_key": sum(1 for k in keys if k is not None),
            "files_matched": sum(1 for k in keys if k is not None and k in leaf_map),
        },
    }

"""Group-aware split, frozen manifest and its receipt.

The manifest is the single source of truth for which image belongs to which
partition. It is built deterministically from hashed inputs (inventory, rules,
reviews), validated against its own invariants, and written with a receipt last,
so a directory that exists without a matching receipt is never trusted.
"""

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jute_disease.config import DEFAULT_SEED, SPLITS
from jute_disease.data.curation import Curation, CurationRules, curate
from jute_disease.data.identity import (
    IdentityResolution,
    Review,
    resolve_identity,
    reviews_digest,
)
from jute_disease.data.inventory import InventoryRecord, inventory_digest
from jute_disease.data.transforms import letterbox_pad_fraction

MANIFEST_VERSION = 2
MANIFEST_FILE = "manifest.json"
RECEIPT_FILE = "receipt.json"
PARTITIONS = ("train", "val", "test")


class FreezeBlockedError(ValueError):
    """The dataset cannot be frozen as specified; nothing was written."""


@dataclass(frozen=True)
class ManifestEntry:
    """One image assigned to one partition."""

    image_id: str
    path: str
    label: str
    partition: str
    group: str
    source: str
    width: int
    height: int
    pad_fraction: float | None = None

    def to_json(self, version: int) -> dict:
        """Return this entry in the selected manifest schema."""
        data = vars(self).copy()
        if version == 1:
            data.pop("pad_fraction")
        elif self.pad_fraction is None:
            raise ValueError("Version 2 manifest entries require pad_fraction")
        return data


def summarize_pad_fractions(
    entries: tuple[ManifestEntry, ...],
) -> list[dict[str, int | float | str]]:
    """Summarize recorded pad fractions by image source and class label."""
    groups: dict[tuple[str, str], list[float]] = {}
    for entry in entries:
        if entry.pad_fraction is not None:
            groups.setdefault((entry.source, entry.label), []).append(
                entry.pad_fraction
            )
    return [
        {
            "source": source,
            "class": label,
            "images": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for (source, label), values in sorted(groups.items())
    ]


@dataclass(frozen=True)
class Manifest:
    """A frozen, validated assignment of images to partitions."""

    seed: int
    fractions: tuple[tuple[str, float], ...]
    entries: tuple[ManifestEntry, ...]
    excluded: tuple[dict, ...]
    merged: tuple[dict, ...]
    quarantined: tuple[dict, ...]
    constraints: tuple[tuple[str, str], ...]
    dependencies: tuple[tuple[str, str], ...]
    version: int = MANIFEST_VERSION

    def to_json(self) -> dict:
        """Return the canonical JSON-safe content."""
        return {
            "version": self.version,
            "seed": self.seed,
            "fractions": dict(self.fractions),
            "dependencies": dict(self.dependencies),
            "entries": [entry.to_json(self.version) for entry in self.entries],
            "excluded": list(self.excluded),
            "merged": list(self.merged),
            "quarantined": list(self.quarantined),
            "constraints": [list(pair) for pair in self.constraints],
        }

    @classmethod
    def from_json(cls, data: dict) -> Manifest:
        """Rebuild a manifest from :meth:`to_json` output."""
        version = data.get("version")
        if version not in {1, MANIFEST_VERSION}:
            raise ValueError(f"Unsupported manifest version: {data.get('version')!r}")
        entries = tuple(ManifestEntry(**item) for item in data["entries"])
        if version == MANIFEST_VERSION:
            for entry in entries:
                expected = letterbox_pad_fraction(entry.width, entry.height)
                if (
                    entry.pad_fraction is None
                    or abs(entry.pad_fraction - expected) > 1e-12
                ):
                    raise ValueError(f"Invalid pad fraction for image {entry.image_id}")
        return cls(
            seed=data["seed"],
            fractions=tuple(data["fractions"].items()),
            entries=entries,
            excluded=tuple(data["excluded"]),
            merged=tuple(data["merged"]),
            quarantined=tuple(data["quarantined"]),
            constraints=tuple(tuple(pair) for pair in data["constraints"]),
            dependencies=tuple(data["dependencies"].items()),
            version=version,
        )

    def digest(self) -> str:
        """Return the SHA-256 of the canonical content."""
        return _digest(self.to_json())

    def partition_digest(self) -> str:
        """Return a digest of the image-to-partition assignment only."""
        return _digest([[e.image_id, e.partition] for e in self.entries])

    def support(self) -> dict[str, dict[str, int]]:
        """Return image counts per partition and label."""
        table: dict[str, Counter] = {p: Counter() for p in PARTITIONS}
        for entry in self.entries:
            table[entry.partition][entry.label] += 1
        return {p: dict(sorted(counts.items())) for p, counts in table.items()}

    def pad_fraction_summary(self) -> list[dict[str, int | float | str]]:
        """Summarize recorded pad fractions by image source and class label."""
        return summarize_pad_fractions(self.entries)

    def partition(self, name: str) -> tuple[ManifestEntry, ...]:
        """Return the entries of one partition, sorted by image id."""
        if name not in PARTITIONS:
            raise ValueError(
                f"Unknown partition {name!r}; expected one of {PARTITIONS}"
            )
        return tuple(e for e in self.entries if e.partition == name)

    @property
    def labels(self) -> tuple[str, ...]:
        """Sorted labels present in the manifest."""
        return tuple(sorted({e.label for e in self.entries}))


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def identity_digest(resolution: IdentityResolution) -> str:
    """Return a digest of the identity-resolution outputs."""
    return _digest(
        {
            "retained": [r.image_id for r in resolution.retained],
            "merged": [vars(m) for m in resolution.merged],
            "quarantined": [vars(q) for q in resolution.quarantined],
            "constraints": [list(c) for c in resolution.constraints],
        }
    )


def build_groups(
    image_ids: list[str], constraints: tuple[tuple[str, str], ...]
) -> dict[str, str]:
    """Map each image id to the smallest id of its constraint component."""
    parent = {image_id: image_id for image_id in image_ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in constraints:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {image_id: find(image_id) for image_id in image_ids}


def _targets(count: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder integer targets summing to ``count``."""
    raw = {name: count * fractions[name] for name in PARTITIONS}
    floor = {name: int(value) for name, value in raw.items()}
    leftover = count - sum(floor.values())
    order = sorted(
        PARTITIONS, key=lambda n: (-(raw[n] - floor[n]), PARTITIONS.index(n))
    )
    for name in order[:leftover]:
        floor[name] += 1
    return floor


def _assign_label(
    groups: dict[str, int], seed: int, fractions: dict[str, float]
) -> dict[str, str]:
    """Assign whole groups of one label to partitions, deterministically."""
    targets = _targets(sum(groups.values()), fractions)
    filled = dict.fromkeys(PARTITIONS, 0)
    order = sorted(
        groups,
        key=lambda g: (-groups[g], hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()),
    )
    placement: dict[str, str] = {}
    for group in order:
        best = max(
            PARTITIONS,
            key=lambda p: (
                (targets[p] - filled[p]) / targets[p] if targets[p] else -1.0,
                -PARTITIONS.index(p),
            ),
        )
        placement[group] = best
        filled[best] += groups[group]
    return placement


def validate_manifest(manifest: Manifest, min_support: int) -> None:
    """Check every invariant of a manifest.

    Target manifests require every label in every partition. Source manifests
    retain publisher-held partitions as published, so only training support is
    required for every source label.

    Raises:
        FreezeBlockedError: an invariant is violated, naming the first failures.
    """
    problems: list[str] = []
    ids = [e.image_id for e in manifest.entries]
    if len(ids) != len(set(ids)):
        problems.append("an image id appears more than once")
    if ids != sorted(ids):
        problems.append("entries are not sorted by image id")
    paths = [e.path for e in manifest.entries]
    if len(paths) != len(set(paths)):
        problems.append("a path appears more than once")
    for entry in manifest.entries:
        if entry.partition not in PARTITIONS:
            problems.append(f"{entry.image_id}: unknown partition {entry.partition!r}")
    partition = {e.image_id: e.partition for e in manifest.entries}
    group_partitions: dict[str, set[str]] = {}
    group_labels: dict[str, set[str]] = {}
    for entry in manifest.entries:
        group_partitions.setdefault(entry.group, set()).add(entry.partition)
        group_labels.setdefault(entry.group, set()).add(entry.label)
    for group, parts in group_partitions.items():
        if len(parts) > 1:
            problems.append(f"group {group[:12]} spans partitions {sorted(parts)}")
        if len(group_labels[group]) > 1:
            problems.append(f"group {group[:12]} mixes labels")
    for left, right in manifest.constraints:
        if left not in partition or right not in partition:
            problems.append(
                f"constraint {left[:12]}/{right[:12]} names a missing image"
            )
        elif partition[left] != partition[right]:
            problems.append(f"constraint {left[:12]}/{right[:12]} is violated")
    support = manifest.support()
    required_partitions = (
        ("train",) if "source_inventory" in dict(manifest.dependencies) else PARTITIONS
    )
    for label in manifest.labels:
        for name in required_partitions:
            have = support[name].get(label, 0)
            if have < min_support:
                problems.append(
                    f"label {label!r} has {have} images in {name}; need {min_support}"
                )
    if problems:
        shown = "; ".join(problems[:8])
        extra = f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""
        raise FreezeBlockedError(f"Manifest is invalid: {shown}{extra}")


def build_manifest(
    records: list[InventoryRecord],
    rules: CurationRules,
    reviews: tuple[Review, ...] = (),
    seed: int = DEFAULT_SEED,
    fractions: dict[str, float] | None = None,
    extra_dependencies: dict[str, str] | None = None,
) -> tuple[Manifest, Curation, IdentityResolution]:
    """Curate, resolve identity, split by group and validate.

    Raises:
        FreezeBlockedError: the dataset cannot be split with every class supported in
            every partition and every identity constraint respected.
    """
    fractions = dict(SPLITS if fractions is None else fractions)
    if set(fractions) != set(PARTITIONS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must cover {PARTITIONS} and sum to 1: {fractions}")
    try:
        curation = curate(records, rules)
    except ValueError as error:
        raise FreezeBlockedError(str(error)) from error
    resolution = resolve_identity(list(curation.admitted), reviews)

    ids = [r.image_id for r in resolution.retained]
    group_of = build_groups(ids, resolution.constraints)
    by_label: dict[str, dict[str, int]] = {}
    for record in resolution.retained:
        groups = by_label.setdefault(record.label, {})
        group = group_of[record.image_id]
        groups[group] = groups.get(group, 0) + 1
    placement: dict[str, str] = {}
    for label in sorted(by_label):
        placement.update(_assign_label(by_label[label], seed, fractions))

    entries = tuple(
        ManifestEntry(
            image_id=r.image_id,
            path=r.path,
            label=r.label,
            partition=placement[group_of[r.image_id]],
            group=group_of[r.image_id],
            source=curation.sources[r.path],
            width=r.width,
            height=r.height,
            pad_fraction=letterbox_pad_fraction(r.width, r.height),
        )
        for r in resolution.retained
    )
    dependencies = {
        "inventory": inventory_digest(records),
        "rules": rules.digest(),
        "reviews": reviews_digest(reviews),
        "identity": identity_digest(resolution),
        **(extra_dependencies or {}),
    }
    manifest = Manifest(
        seed=seed,
        fractions=tuple(sorted(fractions.items())),
        entries=entries,
        excluded=tuple(vars(item) for item in curation.excluded),
        merged=tuple(vars(item) for item in resolution.merged),
        quarantined=tuple(vars(item) for item in resolution.quarantined),
        constraints=resolution.constraints,
        dependencies=tuple(sorted(dependencies.items())),
    )
    validate_manifest(manifest, rules.min_support)
    return manifest, curation, resolution


def receipt_for(manifest: Manifest) -> dict:
    """Return the freeze receipt: hashed dependencies, counts and support."""
    return {
        "version": manifest.version,
        "manifest_digest": manifest.digest(),
        "partition_digest": manifest.partition_digest(),
        "seed": manifest.seed,
        "fractions": dict(manifest.fractions),
        "dependencies": dict(manifest.dependencies),
        "images": len(manifest.entries),
        "excluded": len(manifest.excluded),
        "merged": len(manifest.merged),
        "quarantined": len(manifest.quarantined),
        "groups": len({e.group for e in manifest.entries}),
        "support": manifest.support(),
    }


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_manifest(manifest: Manifest, directory: Path) -> Path:
    """Write the manifest, then its receipt last, and return the receipt path.

    A reader trusts a directory only when the receipt exists and matches the
    manifest, so an interrupted write leaves nothing that looks complete.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / RECEIPT_FILE
    receipt.unlink(missing_ok=True)
    _write_atomic(
        directory / MANIFEST_FILE, json.dumps(manifest.to_json(), indent=1) + "\n"
    )
    _write_atomic(receipt, json.dumps(receipt_for(manifest), indent=2) + "\n")
    return receipt


def load_manifest(directory: Path, min_support: int | None = None) -> Manifest:
    """Load and verify a frozen manifest.

    Raises:
        FileNotFoundError: the manifest or its receipt is missing (an incomplete
            or never-completed freeze).
        FreezeBlockedError: the receipt does not match the manifest, or the manifest
            breaks an invariant.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILE
    receipt_path = directory / RECEIPT_FILE
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"No completed freeze in {directory}: manifest and receipt are required"
        )
    manifest = Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("manifest_digest") != manifest.digest():
        raise FreezeBlockedError(f"Receipt does not match the manifest in {directory}")
    validate_manifest(manifest, 1 if min_support is None else min_support)
    return manifest

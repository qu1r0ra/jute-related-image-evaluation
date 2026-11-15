"""Small synthetic frozen targets, written as manifests only (no image files)."""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from jute_disease.data.manifest import (
    Manifest,
    ManifestEntry,
    write_isolated_manifests,
)
from jute_disease.data.transforms import letterbox_pad_fraction

LABEL = "leaf"


class Member(NamedTuple):
    """One image of a synthetic pool: its name, family and partition."""

    name: str
    group: str
    partition: str


def content_id(name: str) -> str:
    """The image ID (content SHA-256) a synthetic image name stands for."""
    return hashlib.sha256(name.encode()).hexdigest()


def singles(pool: str, partition: str, count: int, start: int = 0) -> list[Member]:
    """``count`` images of one partition, each in its own family."""
    return [
        Member(f"{pool}-{partition}-{index}", f"{pool}-{partition}-g{index}", partition)
        for index in range(start, start + count)
    ]


def write_target(
    directory: Path,
    pool: str,
    members: list[Member],
    *,
    quarantined: tuple[str, ...] = (),
    grouping_version: str = "dhash-v2",
    threshold: int = 6,
) -> Path:
    """Write the fit manifest, protected-test manifest and grouping receipt."""
    grouping = {
        "algorithm_version": grouping_version,
        "parameters": {"hash": "flip-canonical-dhash64", "threshold": threshold},
        "pool": pool,
    }
    grouping["digest"] = hashlib.sha256(
        json.dumps(grouping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    entries = tuple(
        sorted(
            (
                ManifestEntry(
                    image_id=content_id(member.name),
                    path=f"{LABEL}/{member.name}.png",
                    label=LABEL,
                    partition=member.partition,
                    group=member.group,
                    source=pool,
                    width=256,
                    height=256,
                    pad_fraction=letterbox_pad_fraction(256, 256),
                )
                for member in members
            ),
            key=lambda entry: entry.image_id,
        )
    )
    manifest = Manifest(
        seed=42,
        fractions=(("train", 0.6), ("val", 0.2), ("test", 0.2)),
        entries=entries,
        excluded=(),
        merged=(),
        quarantined=tuple(
            {
                "path": f"{LABEL}/{name}.png",
                "image_id": content_id(name),
                "reason": "synthetic",
            }
            for name in quarantined
        ),
        constraints=(),
        dependencies=(
            ("grouping", grouping["digest"]),
            ("grouping_version", grouping_version),
            ("pool", pool),
        ),
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "grouping.json").write_text(json.dumps(grouping), encoding="utf-8")
    write_isolated_manifests(manifest, directory)
    return directory

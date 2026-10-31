"""Source-preserving inventory of an image tree.

The inventory only observes files: it never moves, renames or deletes them. Each
file is keyed by its path relative to the inventory root and carries its content
fingerprints, its folder label and, when present, the historical partition folder
it was found in (kept as provenance, never used for the new split).
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from jute_disease.config import IMAGE_EXTENSIONS
from jute_disease.data.fingerprints import fingerprint_image

INVENTORY_VERSION = 1
_EXTENSIONS = {ext.lower() for ext in IMAGE_EXTENSIONS}
_PARTITION_FOLDERS = {"train", "val", "test"}


@dataclass(frozen=True)
class InventoryRecord:
    """One observed image file."""

    path: str
    label: str
    source_partition: str | None
    bytes: int
    width: int
    height: int
    sha256: str
    pixel_sha256: str
    phash: tuple[int, ...]

    @property
    def image_id(self) -> str:
        """Stable identifier: the SHA-256 of the file bytes."""
        return self.sha256

    @property
    def area(self) -> int:
        """Decoded pixel area."""
        return self.width * self.height

    def to_json(self) -> dict:
        """Return a JSON-safe mapping (perceptual hashes as 16-digit hex)."""
        data = asdict(self)
        data["phash"] = [f"{value:016x}" for value in self.phash]
        return data

    @classmethod
    def from_json(cls, data: dict) -> InventoryRecord:
        """Rebuild a record from :meth:`to_json` output."""
        return cls(
            **{**data, "phash": tuple(int(value, 16) for value in data["phash"])}
        )


def _locate(root: Path, path: Path) -> tuple[str, str | None, str]:
    relative = path.relative_to(root).as_posix()
    parts = relative.split("/")
    if len(parts) == 2:
        return relative, None, parts[0]
    if len(parts) == 3 and parts[0] in _PARTITION_FOLDERS:
        return relative, parts[0], parts[1]
    raise ValueError(
        f"Unsupported layout for {relative!r}: expected <label>/<file> or "
        "<train|val|test>/<label>/<file>"
    )


def _observe(root: Path, path: Path) -> InventoryRecord:
    relative, partition, label = _locate(root, path)
    fingerprint = fingerprint_image(path)
    return InventoryRecord(
        path=relative,
        label=label,
        source_partition=partition,
        bytes=fingerprint.bytes,
        width=fingerprint.width,
        height=fingerprint.height,
        sha256=fingerprint.sha256,
        pixel_sha256=fingerprint.pixel_sha256,
        phash=fingerprint.phash,
    )


def build_inventory(root: Path, workers: int = 4) -> list[InventoryRecord]:
    """Fingerprint every image under ``root`` and return records sorted by path.

    Raises:
        FileNotFoundError: ``root`` is not a directory.
        ValueError: no images were found or a file sits outside the supported layout.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Inventory root does not exist: {root}")
    paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError(f"No images found under {root}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda p: _observe(root, p), paths))


def write_inventory(records: list[InventoryRecord], path: Path) -> None:
    """Write records as JSON lines, sorted by path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.to_json(), sort_keys=True)
        for record in sorted(records, key=lambda r: r.path)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_inventory(path: Path) -> list[InventoryRecord]:
    """Read records written by :func:`write_inventory`."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [InventoryRecord.from_json(json.loads(line)) for line in lines if line]


def inventory_digest(records: list[InventoryRecord]) -> str:
    """Return an order-independent digest of the inventory contents."""
    payload = "\n".join(
        json.dumps(record.to_json(), sort_keys=True)
        for record in sorted(records, key=lambda r: r.path)
    )
    return hashlib.sha256(payload.encode()).hexdigest()

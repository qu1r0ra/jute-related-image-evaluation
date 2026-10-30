"""Manifest-backed dataset: the only reader of frozen partitions."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from jute_disease.config import MANIFEST_DIR, SOURCE_INVENTORY_DIR
from jute_disease.data.cache import cached_path
from jute_disease.data.fingerprints import file_sha256
from jute_disease.data.manifest import Manifest, ManifestEntry, load_manifest
from jute_disease.data.transforms import PreprocessingSpec


class ManifestContentVerifier:
    """Verify manifest files, rehashing any path whose file signature changes."""

    def __init__(self) -> None:
        self._verified: dict[Path, tuple[tuple[int, int, int, int], str]] = {}

    def verify(
        self,
        manifest: Manifest,
        image_root: Path | str,
        partitions: tuple[str, ...],
    ) -> str:
        """Verify live files against frozen hashes for selected partitions.

        The caller selects partitions explicitly so training can verify train
        and validation without opening the held-out test partition.

        Raises:
            FileNotFoundError: a manifest file is missing.
            ValueError: a path escapes the image root or its content hash changed.
        """
        root = Path(image_root).resolve()
        entries = sorted(
            (entry for name in partitions for entry in manifest.partition(name)),
            key=lambda entry: entry.image_id,
        )
        content: list[tuple[str, str]] = []
        for entry in entries:
            path = (root / entry.path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Manifest path escapes image root: {entry.path!r}")
            before = path.stat()
            signature = (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_ino,
            )
            cached = self._verified.get(path)
            if cached is not None and cached[0] == signature:
                observed = cached[1]
            else:
                observed = file_sha256(path)
                after = path.stat()
                after_signature = (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_ino,
                )
                if signature != after_signature:
                    raise ValueError(
                        f"Manifest content changed while verifying {entry.path!r}"
                    )
                self._verified[path] = (signature, observed)
            if observed != entry.image_id:
                raise ValueError(
                    f"Manifest content hash mismatch for {entry.path!r}: "
                    f"expected {entry.image_id}, found {observed}"
                )
            content.append((entry.image_id, observed))
        return hashlib.sha256(
            json.dumps(content, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def verify_manifest_content(
    manifest: Manifest,
    image_root: Path | str,
    partitions: tuple[str, ...],
) -> str:
    """Verify selected manifest partitions against their original-file hashes."""
    return ManifestContentVerifier().verify(manifest, image_root, partitions)


class ManifestDataset(Dataset):
    """Images of one manifest partition, in stable image-id order.

    Pixels come from either the original files or the lossless cache, chosen
    explicitly. A requested cache that lacks a file raises instead of silently
    reading the original, so results never depend on cache presence.
    """

    def __init__(
        self,
        manifest: Manifest,
        partition: str,
        image_root: Path,
        transform: Callable | None = None,
        cache_root: Path | None = None,
        spec: PreprocessingSpec | None = None,
    ) -> None:
        self.entries: tuple[ManifestEntry, ...] = manifest.partition(partition)
        if not self.entries:
            raise ValueError(f"Partition {partition!r} has no images")
        self.classes = list(manifest.labels)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.targets = [self.class_to_idx[e.label] for e in self.entries]
        self.image_ids = [e.image_id for e in self.entries]
        self.groups = [e.group for e in self.entries]
        self.image_root = Path(image_root)
        self.content_digest = ManifestContentVerifier().verify(
            manifest, self.image_root, (partition,)
        )
        self.transform = transform
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.identity = {
            "manifest": manifest.digest(),
            "partition": partition,
            "content": self.content_digest,
            "source": "original"
            if cache_root is None
            else f"cache:{(spec or PreprocessingSpec()).digest()}",
            "transform": getattr(transform, "description", type(transform).__name__),
            "deterministic": str(getattr(transform, "deterministic", False)).lower(),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def path_of(self, index: int) -> Path:
        """Return the file read for ``index``."""
        entry = self.entries[index]
        if self.cache_root is not None:
            return cached_path(self.cache_root, entry.image_id)
        return self.image_root / entry.path

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path = self.path_of(index)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        if self.transform is not None:
            rgb = self.transform(rgb)
        return rgb, self.targets[index]


def open_partition(
    partition: str,
    transform: Callable | None = None,
    manifest_dir: Path = MANIFEST_DIR,
    image_root: Path = SOURCE_INVENTORY_DIR,
) -> ManifestDataset:
    """Open one partition of the completed freeze at ``manifest_dir``.

    Raises:
        FileNotFoundError: the freeze has no matching receipt (not completed).
    """
    return ManifestDataset(
        load_manifest(manifest_dir), partition, image_root, transform
    )

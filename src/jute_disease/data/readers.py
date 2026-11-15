"""Manifest-backed dataset: the only reader of frozen partitions."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from jute_disease.config import MANIFEST_DIR, SOURCE_INVENTORY_DIR
from jute_disease.data.cache import cached_path, downsize
from jute_disease.data.content_verifier import ManifestContentVerifier
from jute_disease.data.manifest import Manifest, ManifestEntry, load_manifest
from jute_disease.data.transforms import PreprocessingSpec


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
        downsize_spec: PreprocessingSpec | None = None,
    ) -> None:
        if cache_root is not None and downsize_spec is not None:
            raise ValueError("A dataset reads a cache or downsizes originals, not both")
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
        self.downsize_spec = downsize_spec
        if cache_root is not None:
            source = f"cache:{(spec or PreprocessingSpec()).digest()}"
        elif downsize_spec is not None:
            source = f"original-downsized:{downsize_spec.digest()}"
        else:
            source = "original"
        self.identity = {
            "manifest": manifest.digest(),
            "partition": partition,
            "content": self.content_digest,
            "source": source,
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
            if self.downsize_spec is not None:
                # The same step the image cache applies, so a fit trained through
                # the cache and its inference see identical pixels.
                rgb = downsize(rgb, self.downsize_spec.cache_shortest_side)
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

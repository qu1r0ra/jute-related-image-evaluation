"""Verify manifest files against their frozen content hashes.

Kept apart from the dataset reader so freeze and intake commands verify bytes
without importing the training stack.
"""

import hashlib
import json
import sys
from pathlib import Path

from jute_disease.data.fingerprints import file_sha256
from jute_disease.data.manifest import Manifest


class ManifestContentVerifier:
    """Verify manifest files, rehashing any path whose file signature changes.

    On Windows the inode and change time are not stable file identity (the
    inode may be unset and the change time is the creation time), so a stat
    signature is never trusted there and every file is rehashed.
    """

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
            cached = None if sys.platform == "win32" else self._verified.get(path)
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

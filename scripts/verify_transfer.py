"""Create or verify SHA-256 inventories before accepting a copied batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_WINDOWS_RESERVED = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?$", re.IGNORECASE
)
_WINDOWS_ILLEGAL = set('<>:"|?*')
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _path_key(relative_path: str) -> str:
    """Validate one portable relative path and return its collision key."""
    windows_path = PureWindowsPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
        or PurePosixPath(relative_path).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError("Transfer manifest contains an absolute or invalid path")

    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("Transfer manifest contains an unsafe path component")

    normalized: list[str] = []
    for component in components:
        canonical = unicodedata.normalize("NFC", component)
        if (
            canonical.endswith((".", " "))
            or any(
                character in _WINDOWS_ILLEGAL or ord(character) < 32
                for character in canonical
            )
            or _WINDOWS_RESERVED.fullmatch(canonical)
        ):
            raise ValueError("Transfer manifest contains a Windows-incompatible path")
        normalized.append(canonical.casefold())
    return "/".join(normalized)


def _insert_entry(
    entries: dict[str, str], keys: dict[str, str], relative_path: str, digest: str
) -> None:
    """Add one checked path and digest, rejecting duplicate portable names."""
    key = _path_key(relative_path)
    if not _SHA256.fullmatch(digest):
        raise ValueError("Transfer manifest contains an invalid SHA-256 digest")
    if relative_path in entries or key in keys:
        raise ValueError("Transfer batch contains duplicate or colliding paths")
    entries[relative_path] = digest
    keys[key] = relative_path


def inventory(root: Path) -> dict[str, str]:
    """Hash every regular file and reject unsafe paths and filesystem objects."""
    if root.is_symlink():
        raise ValueError("Transfer root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("Transfer root must be a directory")

    result: dict[str, str] = {}
    keys: dict[str, str] = {}
    for path in sorted(resolved_root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            relative_path = path.relative_to(resolved_root).as_posix()
            key = _path_key(relative_path)
            if key in keys:
                raise ValueError("Transfer batch contains duplicate or colliding paths")
            keys[key] = relative_path
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                "Transfer batches may contain only regular files and directories"
            )
        if info.st_nlink != 1:
            raise ValueError("Transfer batches may not contain hard-linked files")
        relative_path = path.relative_to(resolved_root).as_posix()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        _insert_entry(result, keys, relative_path, digest)
    if not result:
        raise ValueError("Transfer batch is empty")
    return result


def _manifest_pairs(pairs: list[tuple[str, Any]]) -> dict[str, str]:
    """Load JSON object pairs while retaining duplicate-key detection."""
    manifest: dict[str, str] = {}
    keys: dict[str, str] = {}
    for path, digest in pairs:
        if not isinstance(digest, str):
            raise ValueError("Transfer manifest contains a non-string digest")
        _insert_entry(manifest, keys, path, digest)
    return manifest


def _load_manifest(path: Path) -> dict[str, str]:
    """Read and validate a transfer manifest."""
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=_manifest_pairs)
    if not isinstance(manifest, dict):
        raise ValueError("Transfer manifest must be a JSON object")
    return manifest


def main() -> int:
    """Create or verify one batch manifest from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "verify"])
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        manifest_path = args.manifest.resolve()
        if manifest_path.is_relative_to(root):
            raise ValueError("Manifest must be outside the transfer batch")
        actual = inventory(args.root)
        if args.action == "record":
            with args.manifest.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(actual, handle, indent=2, sort_keys=True)
                handle.write("\n")
        elif actual != _load_manifest(args.manifest):
            raise ValueError("SHA-256 inventory mismatch; batch must not be accepted")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage the three pools as ``<label>/<file>`` image roots from the original releases.

Each pool is one original publication: A is the J01 Kaggle release, B the five
Harvard Dataverse J02 folders, C the JuteFolio-BD (J22) jute folders. Mirrors and
augmented releases stay out; they are used only to screen overlap. Original bytes
are copied unchanged.

    python scripts/stage_pools.py --archives data/source-archives --output data/pools
"""

import argparse
import hashlib
import json
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# J01: folder -> label.
POOL_A = {
    "Cescospora Leaf Spot": "Cercospora Leaf Spot",
    "Golden Mosaic": "Golden Mosaic",
    "Healthy Leaf": "Healthy",
}
# J02: original folders only; the -ML copy (TRAIN-2500, TEST-500) is screening-only.
POOL_B = {
    "Dieback-300": "Dieback",
    "Fresh-280": "Healthy",
    "Holed-300": "Holed",
    "Mosaic-240": "Mosaic",
    "Stem Soft Rot-270": "Stem Soft Rot",
}
# J22: flat disease folders, and the Healthy_19 varieties that are jute. KENAF,
# MESTA and HS (roselle) varieties are other crops.
POOL_C_FLAT = {"Mosaic": "Mosaic", "Soft_Rot": "Soft Rot", "Stem_Rot": "Stem Rot"}
POOL_C_JUTE_VARIETIES = {
    "BJRI_DESHI_8",
    "BJRI_DESHI_9",
    "BJRI_TOSHA_4",
    "BJRI_TOSHA_5",
    "BJRI_TOSHA_7",
    "BJRI_TOSHA_8",
    "CC_45",
    "CVL_1",
    "O_4",
    "O_9897",
}


def _target(pool: str, entry: str) -> tuple[str, str] | None:
    """Return ``(label, filename)`` for an archive entry, or None to leave it out."""
    parts = PurePosixPath(entry).parts
    if PurePosixPath(entry).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    if pool == "A" and len(parts) == 3 and parts[1] in POOL_A:
        return POOL_A[parts[1]], parts[2]
    if pool == "B" and len(parts) == 3 and parts[0] == "Jute Diesease Dataset":
        label = POOL_B.get(parts[1])
        return (label, parts[2]) if label else None
    if pool == "C":
        if len(parts) == 2 and parts[0] in POOL_C_FLAT:
            return POOL_C_FLAT[parts[0]], parts[1]
        if (
            len(parts) == 3
            and parts[0] == "Healthy_19"
            and parts[1] in POOL_C_JUTE_VARIETIES
        ):
            return "Healthy", f"{parts[1]}__{parts[2]}"
    return None


def _collision_key(label: str, name: str) -> tuple[str, str]:
    """Return the key under which two staged paths collide on any filesystem."""
    return (
        unicodedata.normalize("NFC", label).casefold(),
        unicodedata.normalize("NFC", name).casefold(),
    )


def _expected_labels(pool: str) -> set[str]:
    """Return the labels a complete pool must stage at least one image for."""
    if pool == "A":
        return set(POOL_A.values())
    if pool == "B":
        return set(POOL_B.values())
    return set(POOL_C_FLAT.values()) | {"Healthy"}


def _archive_sha256(archive: Path) -> str:
    with archive.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _plan(pool: str, bundle: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str, str]]:
    """Validate every admitted entry before anything is written."""
    planned: list[tuple[zipfile.ZipInfo, str, str]] = []
    seen: dict[tuple[str, str], str] = {}
    for info in sorted(bundle.infolist(), key=lambda item: item.filename):
        target = _target(pool, info.filename)
        if target is None:
            continue
        label, name = target
        if info.file_size == 0:
            raise ValueError(f"Pool {pool} has an empty image {info.filename!r}")
        key = _collision_key(label, name)
        if key in seen:
            raise ValueError(
                f"Pool {pool} has duplicate target paths {seen[key]!r} and "
                f"{info.filename!r} (compared case-insensitively)"
            )
        seen[key] = info.filename
        planned.append((info, label, name))
    if not planned:
        raise ValueError(f"Pool {pool} staged no images; check the archive layout")
    missing = _expected_labels(pool) - {label for _, label, _ in planned}
    if missing:
        raise ValueError(f"Pool {pool} has no images for {sorted(missing)}")
    return planned


def stage(pool: str, archive: Path, output: Path) -> dict[str, int]:
    """Copy one pool's admitted entries under ``output/<pool>/<label>/``.

    The archive is validated first and staged into a scratch directory that
    replaces ``output/<pool>`` only after every file is written, so a rejected
    archive leaves no partial pool behind. A non-empty pool directory is never
    overwritten.
    """
    destination_root = output / pool
    if destination_root.exists() and any(destination_root.iterdir()):
        raise ValueError(
            f"Pool {pool} is already staged at {destination_root}; remove it first"
        )
    archive_sha256 = _archive_sha256(archive)
    scratch = output / f".{pool}.staging"
    shutil.rmtree(scratch, ignore_errors=True)
    counts: dict[str, int] = {}
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(archive) as bundle:
            planned = _plan(pool, bundle)
            for info, label, name in planned:
                destination = scratch / label / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = bundle.read(info)
                destination.write_bytes(data)
                digest.update(
                    f"{label}/{name}\t{hashlib.sha256(data).hexdigest()}\n".encode()
                )
                counts[label] = counts.get(label, 0) + 1
        if _archive_sha256(archive) != archive_sha256:
            raise ValueError(f"Pool {pool} archive changed while staging")
        receipt = {
            "pool": pool,
            "archive": archive.name,
            "archive_sha256": archive_sha256,
            "counts": dict(sorted(counts.items())),
            "staged_digest": digest.hexdigest(),
        }
        (scratch / "stage-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if destination_root.exists():
            destination_root.rmdir()
        scratch.replace(destination_root)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return receipt["counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = {
        "A": args.archives / "J01" / "kaggle.zip",
        "B": args.archives / "J02" / "Jute Disease Dataset.zip",
        "C": args.archives / "J22" / "kaggle.zip",
    }
    for pool, archive in archives.items():
        try:
            counts = stage(pool, archive, args.output)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            parser.error(f"pool {pool}: {error}")
        sys.stdout.write(f"{pool} {json.dumps(counts)}\n")


if __name__ == "__main__":
    main()

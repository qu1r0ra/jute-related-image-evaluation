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
import sys
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


def stage(pool: str, archive: Path, output: Path) -> dict[str, int]:
    """Copy one pool's admitted entries under ``output/<pool>/<label>/``."""
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            target = _target(pool, info.filename)
            if target is None:
                continue
            if target in seen:
                raise ValueError(f"Pool {pool} has a duplicate target {target}")
            seen.add(target)
            label, name = target
            destination = output / pool / label / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = bundle.read(info)
            destination.write_bytes(data)
            digest.update(
                f"{label}/{name}\t{hashlib.sha256(data).hexdigest()}\n".encode()
            )
            counts[label] = counts.get(label, 0) + 1
    receipt = {
        "pool": pool,
        "archive": archive.name,
        "counts": dict(sorted(counts.items())),
        "staged_digest": digest.hexdigest(),
    }
    (output / pool / "stage-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
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
        sys.stdout.write(f"{pool} {json.dumps(stage(pool, archive, args.output))}\n")


if __name__ == "__main__":
    main()

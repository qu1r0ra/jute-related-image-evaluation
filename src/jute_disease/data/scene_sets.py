"""Seeded draw of the scene-tag sets from the frozen fit partitions.

One draw writes three outputs that share no image and no family: the held-out
set the scene-tag evaluation is scored on, a reserve split into a fit and a
test part, and the mask-audit inventory of every remaining eligible image.
Only manifests are read. Test images are known by identity alone, and no
target image is opened.
"""

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jute_disease.data.gates import read_grouping
from jute_disease.data.manifest import (
    FIT_PARTITIONS,
    load_manifest,
    load_protected_test_manifest,
)

SCENE_SETS_SCHEMA = "jute-scene-sets-v1"
SCENE_SET_KEYS = ("held_out", "reserve_fit", "reserve_test")
SCENE_SET_FILES = {
    "held_out": "held-out.json",
    "reserve_fit": "reserve-fit.json",
    "reserve_test": "reserve-test.json",
}
AUDIT_INVENTORY_FILE = "audit-inventory.json"
SCENE_SETS_RECEIPT = "receipt.json"
# Held-out 60/60/180; the 270-image reserve is 170 fit and 100 test, each 1:1:3.
SCENE_SET_ALLOCATION: dict[str, dict[str, int]] = {
    "A": {"held_out": 60, "reserve_fit": 34, "reserve_test": 20},
    "B": {"held_out": 60, "reserve_fit": 34, "reserve_test": 20},
    "C": {"held_out": 180, "reserve_fit": 102, "reserve_test": 60},
}
EXCLUSION_REASONS = (
    "quarantined",
    "cross_pool_repeat",
    "protected_test_family",
    "development_family",
    "excluded_family",
)
_DRAW_ORDER = ("held_out", "reserve_test", "reserve_fit")
_RECORD_KEYS = {"image_id", "path", "pool", "group", "partition"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _development_ids(development: Path) -> set[str]:
    """Image IDs and, where the file is present, content digests of the set."""
    development = Path(development)
    records = json.loads(development.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(
        isinstance(record, dict) and "image_id" in record for record in records
    ):
        raise ValueError("development manifest must be a JSON array of image records")
    ids: set[str] = set()
    for record in records:
        ids.add(str(record["image_id"]))
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = development.parent / path
        if record.get("path") and path.is_file():
            ids.add(_file_sha256(path))
    return ids


def read_excluded_ids(source: Path) -> list[str]:
    """Read a JSON array of further image IDs whose families are excluded."""
    ids = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("excluded image IDs must be a JSON array of strings")
    return ids


def read_scene_set(source: Path) -> list[dict[str, str]]:
    """Read one drawn set; relative paths resolve against the set file."""
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("scene set must be a nonempty JSON array")
    result = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise ValueError(
                f"each scene-set record needs exactly {sorted(_RECORD_KEYS)}"
            )
        path = Path(record["path"])
        if not path.is_absolute():
            path = Path(source).parent / path
        result.append({**record, "path": str(path)})
    ids = [record["image_id"] for record in result]
    if len(set(ids)) != len(ids):
        raise ValueError("scene-set image IDs must be unique")
    return result


def _draw_pool(
    pool: str,
    eligible: list[dict[str, str]],
    sizes: Mapping[str, int],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Simple random draw within a pool, keeping each family in one set."""
    order = sorted(eligible, key=lambda record: record["image_id"])
    random.Random(f"scene-sets:{seed}:{pool}").shuffle(order)
    owner: dict[str, str] = {}
    drawn: dict[str, list[dict[str, str]]] = {key: [] for key in SCENE_SET_KEYS}
    taken: set[str] = set()
    for key in _DRAW_ORDER:
        for record in order:
            if len(drawn[key]) == sizes[key]:
                break
            if record["image_id"] in taken:
                continue
            if owner.setdefault(record["group"], key) != key:
                continue
            drawn[key].append(record)
            taken.add(record["image_id"])
        if len(drawn[key]) < sizes[key]:
            raise ValueError(
                f"pool {pool} has too few eligible families: {key} needs "
                f"{sizes[key]} images and only {len(drawn[key])} could be drawn"
            )
    drawn["audit_inventory"] = [
        record for record in order if record["group"] not in owner
    ]
    return drawn


def draw_scene_sets(
    targets: Mapping[str, Path],
    image_roots: Mapping[str, Path],
    development: Path,
    output: Path,
    *,
    seed: int,
    excluded_ids: Iterable[str] = (),
    allocation: Mapping[str, Mapping[str, int]] = SCENE_SET_ALLOCATION,
) -> dict[str, Any]:
    """Draw the held-out set, the reserve and the audit inventory; return the receipt.

    Eligible images are the train and validation entries of each pool's fit
    manifest, less quarantined images, images present in more than one pool,
    and every family with a member in a protected-test partition, in the
    development set, or among ``excluded_ids``.

    Raises:
        ValueError: the pools do not match the allocation, or a pool has too
            few eligible families.
        FileExistsError: ``output`` already exists.
    """
    output = Path(output)
    pools = sorted(allocation)
    if sorted(targets) != pools or sorted(image_roots) != pools:
        raise ValueError(
            f"target directories and image roots must name exactly the pools {pools}"
        )
    for pool in pools:
        sizes = allocation[pool]
        if set(sizes) != set(SCENE_SET_KEYS) or any(
            not isinstance(size, int) or size < 1 for size in sizes.values()
        ):
            raise ValueError(f"pool {pool} needs positive {list(SCENE_SET_KEYS)} sizes")
    if output.exists():
        raise FileExistsError(f"scene-set output already exists: {output}")
    excluded = [str(image_id) for image_id in excluded_ids]
    development_ids = _development_ids(development)

    fits, tests, groupings = {}, {}, {}
    for pool in pools:
        fits[pool] = load_manifest(Path(targets[pool]))
        if dict(fits[pool].dependencies).get("manifest_scope") != "fit":
            raise ValueError(f"pool {pool} target does not hold a fit manifest")
        tests[pool] = load_protected_test_manifest(Path(targets[pool]))
        groupings[pool] = read_grouping(Path(targets[pool]))

    quarantined = {
        pool: {str(item["image_id"]) for item in fits[pool].quarantined}
        | {str(item["image_id"]) for item in tests[pool].quarantined}
        for pool in pools
    }
    pool_count: Counter[str] = Counter()
    for pool in pools:
        pool_count.update(
            {entry.image_id for entry in fits[pool].entries}
            | {entry.image_id for entry in tests[pool].entries}
            | quarantined[pool]
        )
    known = set(pool_count)

    results: dict[str, dict[str, list[dict[str, str]]]] = {}
    exclusions: dict[str, dict[str, int]] = {}
    counts: dict[str, dict[str, int]] = {}
    for pool in pools:
        entries = [e for e in fits[pool].entries if e.partition in FIT_PARTITIONS]
        test_ids = {entry.image_id for entry in tests[pool].entries}
        test_groups = {entry.group for entry in tests[pool].entries} | {
            entry.group for entry in entries if entry.image_id in test_ids
        }
        development_groups = {
            entry.group for entry in entries if entry.image_id in development_ids
        }
        excluded_groups = {
            entry.group for entry in entries if entry.image_id in set(excluded)
        }
        reasons: Counter[str] = Counter()
        eligible = []
        root = Path(image_roots[pool]).resolve()
        for entry in entries:
            if entry.image_id in quarantined[pool]:
                reasons["quarantined"] += 1
            elif pool_count[entry.image_id] > 1:
                reasons["cross_pool_repeat"] += 1
            elif entry.group in test_groups:
                reasons["protected_test_family"] += 1
            elif entry.group in development_groups:
                reasons["development_family"] += 1
            elif entry.group in excluded_groups:
                reasons["excluded_family"] += 1
            else:
                eligible.append(
                    {
                        "image_id": entry.image_id,
                        "path": str(root / entry.path),
                        "pool": pool,
                        "group": entry.group,
                        "partition": entry.partition,
                    }
                )
        results[pool] = _draw_pool(pool, eligible, allocation[pool], seed)
        counts[pool] = {
            "eligible": len(eligible),
            **{key: len(results[pool][key]) for key in SCENE_SET_KEYS},
            "audit_inventory": len(results[pool]["audit_inventory"]),
        }
        exclusions[pool] = {
            **{reason: reasons[reason] for reason in EXCLUSION_REASONS},
            "unused_family_members": len(eligible)
            - sum(len(records) for records in results[pool].values()),
        }

    payloads: dict[str, list[dict[str, str]]] = {
        SCENE_SET_FILES[key]: sorted(
            (record for pool in pools for record in results[pool][key]),
            key=lambda record: record["image_id"],
        )
        for key in SCENE_SET_KEYS
    }
    payloads[AUDIT_INVENTORY_FILE] = sorted(
        (
            {"image_id": record["image_id"], "path": record["path"]}
            for pool in pools
            for record in results[pool]["audit_inventory"]
        ),
        key=lambda record: record["image_id"],
    )
    texts = {name: json.dumps(rows, indent=2) + "\n" for name, rows in payloads.items()}
    receipt: dict[str, Any] = {
        "schema": SCENE_SETS_SCHEMA,
        "seed": seed,
        "method": "seeded simple random draw within each pool; a family is "
        "kept in one set; held-out, then reserve test, then reserve fit",
        "allocation": {pool: dict(allocation[pool]) for pool in pools},
        "grouping": {
            pool: {
                "version": groupings[pool]["algorithm_version"],
                "threshold": groupings[pool]["parameters"]["threshold"],
                "digest": groupings[pool]["digest"],
            }
            for pool in pools
        },
        "manifest_digests": {
            pool: {
                "fit": fits[pool].digest(),
                "protected_test": tests[pool].digest(),
            }
            for pool in pools
        },
        "development_manifest_digest": _file_sha256(Path(development)),
        "development": {
            "identities": len(development_ids),
            "matched": len(development_ids & known),
        },
        "excluded_ids": {
            "given": len(set(excluded)),
            "matched": len(set(excluded) & known),
        },
        "counts": counts,
        "exclusions": exclusions,
        "outputs": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in texts.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, text in texts.items():
        (output / name).write_text(text, encoding="utf-8", newline="\n")
    (output / SCENE_SETS_RECEIPT).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt

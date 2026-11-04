"""Frozen policy and view assignments for independently grouped pools."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from jute_disease.config import IMAGE_SIZE
from jute_disease.data.inventory import inventory_digest, read_inventory
from jute_disease.data.manifest import (
    Manifest,
    ManifestEntry,
    load_complete_manifest,
)
from jute_disease.data.transforms import letterbox_pad_fraction

POLICIES = ("image-random", "exact-copy", "near-duplicate")
PARTITIONS = ("train", "val", "test")


def load_policy_manifest(
    policy_dir: Path | str,
    *,
    pool: str,
    image_size: int = IMAGE_SIZE,
    view: int | None = None,
    policy: str | None = None,
    draw: int | None = None,
    reserve: bool = False,
) -> Manifest:
    """Load fit-visible partitions from one receipted policy artifact.

    The returned manifest contains only train and validation entries. The
    reserve is divided into family-disjoint tuning train and validation groups
    with a stable, class-stratified hash rank.
    """
    if pool not in {"A", "B", "C"}:
        raise ValueError("pool must be A, B or C")
    if reserve:
        if view is not None or policy is not None or draw is not None:
            raise ValueError("reserve loading does not accept a view, policy or draw")
        relative = Path(pool) / "reserve.json"
    else:
        if view not in (1, 2) or policy not in POLICIES or draw is None:
            raise ValueError("policy loading needs a view, policy and draw")
        relative = Path(pool) / f"view{view}" / str(policy) / f"draw-{draw}.json"
    root = Path(policy_dir).resolve()
    path = root / relative
    encoded = path.read_bytes()
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_digest = receipt.get("files", {}).get(relative.as_posix())
    observed_digest = hashlib.sha256(encoded).hexdigest()
    if expected_digest != observed_digest:
        raise ValueError(f"Policy artifact is not covered by its receipt: {relative}")
    payload = json.loads(encoded)
    if payload.get("pool") != pool:
        raise ValueError("Policy artifact pool does not match its path")
    if not reserve and any(
        payload.get(key) != value
        for key, value in (("view", view), ("policy", policy), ("draw", draw))
    ):
        raise ValueError("Policy artifact dimensions do not match its path")

    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Policy artifact must contain entries")
    if reserve:
        families: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            family = row.get("family")
            label = row.get("label")
            if not isinstance(family, str) or not family or not isinstance(label, str):
                raise ValueError("Reserve entries need a family and class label")
            families[label].append(row)
        family_partitions: dict[str, str] = {}
        for label, family_rows in families.items():
            family_ids = sorted({row["family"] for row in family_rows})
            if len(family_ids) < 2:
                raise ValueError(f"Reserve needs at least two families for {label}")
            ranked = sorted(
                family_ids,
                key=lambda family: hashlib.sha256(
                    f"reserve:{pool}:{label}:{family}".encode()
                ).hexdigest(),
            )
            val_count = min(len(ranked) - 1, max(1, round(len(ranked) * 0.2)))
            validation = set(ranked[:val_count])
            for family in ranked:
                family_partitions[family] = "val" if family in validation else "train"
        entries = [
            _manifest_entry(row, family_partitions[row["family"]], image_size)
            for row in rows
        ]
    else:
        if any(row.get("partition") not in ("train", "val") for row in rows):
            raise ValueError("Fit policy manifest contains a non-fit partition")
        entries = [_manifest_entry(row, row["partition"], image_size) for row in rows]
    if (
        not entries
        or not any(entry.partition == "train" for entry in entries)
        or not any(entry.partition == "val" for entry in entries)
    ):
        raise ValueError("Policy artifact needs non-empty train and validation data")
    dependencies = {
        "policy_file": observed_digest,
        "config": str(payload.get("config_digest", "")),
    }
    return Manifest(
        seed=int(payload.get("draw_seed", payload.get("seed", 0))),
        fractions=(("train", 0.0), ("val", 0.0), ("test", 0.0)),
        entries=tuple(
            sorted(entries, key=lambda entry: (entry.partition, entry.image_id))
        ),
        excluded=(),
        merged=(),
        quarantined=(),
        constraints=(),
        dependencies=tuple(sorted(dependencies.items())),
        version=2,
    )


def _manifest_entry(row: dict, partition: str, image_size: int) -> ManifestEntry:
    try:
        width, height = int(row["width"]), int(row["height"])
        entry = ManifestEntry(
            image_id=str(row["sha256"]),
            path=str(row["path"]),
            label=str(row["label"]),
            partition=partition,
            group=str(row["family"]),
            source=str(row["source"]),
            width=width,
            height=height,
            pad_fraction=letterbox_pad_fraction(width, height, image_size),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Policy entry is missing valid manifest fields") from error
    if partition not in ("train", "val") or width < 1 or height < 1:
        raise ValueError("Policy entry has an invalid fit partition or dimensions")
    if len(entry.image_id) != 64:
        raise ValueError("Policy entry needs a SHA-256 image identity")
    return entry


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _select(keys: set[str], fraction: float, seed: str) -> set[str]:
    count = round(len(keys) * fraction)
    return set(sorted(keys, key=lambda key: _rank(seed, key))[:count])


def _pool(root: Path) -> tuple[list[dict], dict[str, str]]:
    target = root / "target"
    manifest = load_complete_manifest(target)
    dependencies = dict(manifest.dependencies)
    if "grouping" not in dependencies:
        raise ValueError("A frozen grouping receipt is required")
    grouping = json.loads((target / "grouping.json").read_text(encoding="utf-8"))
    outcomes = grouping.get("gate_outcomes", {})
    if (
        not isinstance(outcomes, dict)
        or outcomes.get("agreement") != "pass"
        or any(
            outcome not in ("pass", "not_requested", "applied")
            for outcome in outcomes.values()
        )
    ):
        raise ValueError("grouping gate outcomes have not passed")
    inventory = read_inventory(root / "inventory.jsonl")
    if inventory_digest(inventory) != dependencies.get("inventory"):
        raise ValueError("inventory does not match the frozen manifest")
    by_path = {record.path: record for record in inventory}
    source_by_sha = {entry.image_id: entry.source for entry in manifest.entries}
    rows = []
    seen = set()
    for family in grouping["families"]:
        if (
            not family
            or len(family) != len(set(family))
            or any(path in seen or path not in by_path for path in family)
        ):
            raise ValueError("invalid grouping family membership")
        seen.update(family)
        members = [by_path[path] for path in family]
        if len({record.label for record in members}) != 1:
            raise ValueError("grouping family has conflicting labels")
        family_id = hashlib.sha256(json.dumps(sorted(family)).encode()).hexdigest()
        for record in members:
            if record.sha256 not in source_by_sha:
                raise ValueError("grouping family is absent from frozen manifest")
            rows.append(
                {
                    "path": record.path,
                    "sha256": record.sha256,
                    "label": record.label,
                    "family": family_id,
                    "source": source_by_sha[record.sha256],
                    "width": record.width,
                    "height": record.height,
                }
            )
    if not rows or {row["sha256"] for row in rows} != set(source_by_sha):
        raise ValueError("grouping families do not cover the frozen manifest")
    sha_families = {}
    for row in rows:
        sha_families.setdefault(row["sha256"], set()).add(row["family"])
    if any(len(families) != 1 for families in sha_families.values()):
        raise ValueError("exact copies cross grouping families")
    lineage = {
        "grouping_digest": grouping["digest"],
        "inventory_digest": dependencies["inventory"],
        "manifest_digest": manifest.digest(),
    }
    return sorted(rows, key=lambda row: row["path"]), lineage


def _split(
    rows: list[dict],
    policy: str,
    draw_seed: int,
    val: float,
    test: float,
    protected: set[str] | None,
) -> list[dict]:
    unit = {"image-random": "path", "exact-copy": "sha256", "near-duplicate": "family"}[
        policy
    ]
    result = []
    for label in sorted({row["label"] for row in rows}):
        subset = [row for row in rows if row["label"] == label]
        fixed = [
            row
            for row in subset
            if protected is not None and row["family"] in protected
        ]
        candidates = [
            row for row in subset if protected is None or row["family"] not in protected
        ]
        keys = {row[unit] for row in candidates}
        if protected is None:
            test_keys = _select(keys, test, f"{draw_seed}:{policy}:{label}:test")
            remaining = keys - test_keys
        else:
            test_keys = set()
            remaining = keys
        val_keys = _select(
            remaining, val / (1 - test), f"{draw_seed}:{policy}:{label}:val"
        )
        for row in candidates:
            partition = (
                "test"
                if row[unit] in test_keys
                else "val"
                if row[unit] in val_keys
                else "train"
            )
            result.append({**row, "partition": partition})
        result.extend({**row, "partition": "test"} for row in fixed)
    return sorted(result, key=lambda row: row["path"])


def _support(rows: list[dict], labels: list[str]) -> dict:
    return {
        part: {
            label: len(
                {
                    row["family"]
                    for row in rows
                    if row["partition"] == part and row["label"] == label
                }
            )
            for label in labels
        }
        for part in PARTITIONS
    }


def write_policy_manifests(
    pools: dict[str, str],
    output: Path,
    reserve_fraction: float,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict:
    """Write a complete, recorded split matrix from verified grouping freezes."""
    fractions = (reserve_fraction, train_fraction, val_fraction, test_fraction)
    if not all(
        math.isfinite(value) and 0 < value < 1 for value in fractions
    ) or not math.isclose(
        train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-9
    ):
        raise ValueError(
            "fractions must be finite and positive; "
            "final split fractions must sum to one"
        )
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    if not pools or any(name not in {"A", "B", "C"} for name in pools):
        raise ValueError("pool names must be A, B or C")
    config = {
        "seed": seed,
        "reserve_fraction": reserve_fraction,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "reserve_denominator": "admitted families per class",
        "split_denominator": "non-reserve policy units per class",
        "view2_test": "whole near-duplicate families per class, shared across policies",
    }
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = {"config.json": {**config, "digest": config_digest}}
    artifacts["assignments.json"] = [
        {"split": split, "seed": target_seed, "checkpoint": (target_seed + split) % 3}
        for split in range(10)
        for target_seed in range(5)
    ]
    for name, path in sorted(pools.items()):
        rows, lineage = _pool(Path(path))
        labels = sorted({row["label"] for row in rows})
        reserve = set()
        for label in labels:
            reserve |= _select(
                {row["family"] for row in rows if row["label"] == label},
                reserve_fraction,
                f"{seed}:{name}:{label}:reserve",
            )
        reserved = [row for row in rows if row["family"] in reserve]
        available = [row for row in rows if row["family"] not in reserve]
        if any(
            len({row["family"] for row in reserved if row["label"] == label}) < 2
            for label in labels
        ):
            raise ValueError(f"insufficient reserve family support for pool {name}")
        artifacts[f"{name}/reserve.json"] = {
            "pool": name,
            "seed": seed,
            "fraction": reserve_fraction,
            "config_digest": config_digest,
            **lineage,
            "family_support": {
                label: len({row["family"] for row in reserved if row["label"] == label})
                for label in labels
            },
            "entries": reserved,
        }
        for draw in range(10):
            draw_seed = seed + draw
            protected = set()
            for label in labels:
                protected |= _select(
                    {row["family"] for row in available if row["label"] == label},
                    test_fraction,
                    f"{draw_seed}:{name}:{label}:protected",
                )
            for view in (1, 2):
                for policy in POLICIES:
                    entries = _split(
                        available,
                        policy,
                        draw_seed,
                        val_fraction,
                        test_fraction,
                        protected if view == 2 else None,
                    )
                    excluded = []
                    if name == "C":
                        healthy = [
                            row
                            for row in entries
                            if row["label"].lower() == "healthy"
                            and row["partition"] == "train"
                        ]
                        key = "path" if policy == "image-random" else "family"
                        units = {row[key] for row in healthy}
                        if len(units) > 1000:
                            keep = set(
                                sorted(
                                    units,
                                    key=lambda unit: _rank(f"{seed}:healthy-cap", unit),
                                )[:1000]
                            )
                            excluded = [
                                row
                                for row in entries
                                if row["label"].lower() == "healthy"
                                and row["partition"] == "train"
                                and row[key] not in keep
                            ]
                            excluded_paths = {row["path"] for row in excluded}
                            entries = [
                                row
                                for row in entries
                                if row["path"] not in excluded_paths
                            ]
                    support = _support(entries, labels)
                    if any(
                        count == 0
                        for counts in support.values()
                        for count in counts.values()
                    ):
                        raise ValueError(
                            f"insufficient per-class family support for pool {name}, "
                            f"view {view}, {policy}, draw {draw}: {support}"
                        )
                    common = {
                        "pool": name,
                        "view": view,
                        "policy": policy,
                        "draw": draw,
                        "draw_seed": draw_seed,
                        **lineage,
                        "config_digest": config_digest,
                        "reserve_fraction": reserve_fraction,
                        "val_fraction": val_fraction,
                        "test_fraction": test_fraction,
                        "healthy_cap_seed": seed,
                        "healthy_cap_excluded": excluded,
                    }
                    fit_entries = [
                        row for row in entries if row["partition"] in ("train", "val")
                    ]
                    protected_entries = [
                        row for row in entries if row["partition"] == "test"
                    ]
                    fit_support = _support(fit_entries, labels)
                    protected_support = _support(protected_entries, labels)
                    artifacts[f"{name}/view{view}/{policy}/draw-{draw}.json"] = {
                        **common,
                        "family_support": {
                            part: fit_support[part] for part in ("train", "val")
                        },
                        "entries": fit_entries,
                        "image_support": {
                            part: {
                                label: sum(
                                    row["partition"] == part and row["label"] == label
                                    for row in fit_entries
                                )
                                for label in labels
                            }
                            for part in ("train", "val")
                        },
                    }
                    artifacts[
                        f"{name}/view{view}/{policy}/draw-{draw}.protected-test.json"
                    ] = {
                        **common,
                        "family_support": {"test": protected_support["test"]},
                        "entries": protected_entries,
                        "image_support": {
                            "test": {
                                label: sum(
                                    row["label"] == label for row in protected_entries
                                )
                                for label in labels
                            }
                        },
                    }
    output.mkdir(parents=True, exist_ok=False)
    digests = {}
    for relative, payload in artifacts.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        destination.write_bytes(encoded)
        digests[relative] = hashlib.sha256(encoded).hexdigest()
    receipt = {"version": 1, "config_digest": config_digest, "files": digests}
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output),
        "pools": sorted(pools),
        "artifacts": len(artifacts),
        "receipt": "receipt.json",
    }

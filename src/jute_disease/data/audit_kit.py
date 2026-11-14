"""Blind owner kit that joins grouping pairs with mask overlays.

The kit holds only what the owner needs to rate: pair images and mask overlays,
with no agent ratings, algorithm decisions, strata or predicted scenes. Mask
items come first, as the mask audit asks for them before the grouping pairs.
Each sitting writes a sheet of items not yet rated, so the owner can stop and
resume without seeing a rated item again.
"""

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from jute_disease.data.audit import cohort_digest
from jute_disease.data.audit_sampling import grouping_threshold
from jute_disease.data.masks import MASK_QUALITY_RATINGS, SCENE_TAGS
from jute_disease.data.masks_audit import AUDIT_SCHEMA as MASK_AUDIT_SCHEMA

KIT_SCHEMA = "jute-audit-kit-v1"
KIT_FILE = "kit.json"
RATING_VALUES = ("exact", "near-duplicate", "very similar", "different")
SHEET_COLUMNS = (
    "item_id",
    "kind",
    "image_a",
    "image_b",
    "overlay_path",
    "rating",
    "mask_quality",
    "scene_tag",
)
PAIR_COLUMNS = ("pair_id", "image_a", "image_b", "rating")
MASK_COLUMNS = ("audit_id", "overlay_path", "mask_quality", "scene_tag")
OWNER_PAIR_SHEET = "owner.csv"
OWNER_MASK_SHEET = "owner-masks.csv"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_audit_kit(
    cohort_dir: Path, mask_audit_dir: Path, output: Path, seed: int = 42
) -> dict[str, Any]:
    """Write the blind kit and return a summary.

    The grouping cohort must name its grouping version and agent rater. The mask
    items are the owner subset of the mask audit, and their overlays are copied
    into the kit under neutral names.
    """
    cohort_dir, mask_audit_dir, output = (
        Path(cohort_dir),
        Path(mask_audit_dir),
        Path(output),
    )
    cohort = _read_json(cohort_dir / "agent.json")
    version = cohort.get("grouping_version")
    threshold = grouping_threshold(version)
    agent = cohort.get("agent") or {}
    if not all(str(agent.get(key, "")).strip() for key in ("name", "version")):
        raise ValueError("the cohort does not name the agent rater")
    masks = _read_json(mask_audit_dir / "audit.json")
    if masks.get("schema") != MASK_AUDIT_SCHEMA:
        raise ValueError(f"mask audit schema must be {MASK_AUDIT_SCHEMA}")
    records = {record["audit_id"]: record for record in masks["records"]}
    owner_ids = list(masks["owner_audit_ids"])
    if unknown := sorted(set(owner_ids) - set(records)):
        raise ValueError(f"mask audit owner subset names unknown ids: {unknown}")
    rng = random.Random(seed)
    mask_items = [
        {
            "item_id": f"mask:{audit_id}",
            "kind": "mask",
            "image_a": "",
            "image_b": "",
            "overlay_path": f"overlays/{audit_id}.png",
            "source_overlay_path": records[audit_id]["overlay_path"],
        }
        for audit_id in owner_ids
    ]
    pair_items = [
        {
            "item_id": f"pair:{row['pair_id']}",
            "kind": "pair",
            "image_a": row["image_a"],
            "image_b": row["image_b"],
            "overlay_path": "",
        }
        for row in cohort["pairs"]
    ]
    rng.shuffle(mask_items)
    rng.shuffle(pair_items)
    items = mask_items + pair_items
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("kit items must be unique")
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    for item in mask_items:
        source = mask_audit_dir / item["source_overlay_path"]
        if not source.is_file():
            raise ValueError(f"mask overlay is missing: {source}")
        shutil.copyfile(source, output / item["overlay_path"])
    kit = {
        "schema": KIT_SCHEMA,
        "grouping_version": version,
        "hamming_threshold": threshold,
        "agent": {"name": agent["name"], "version": agent["version"]},
        "cohort_digest": cohort_digest(cohort),
        "seed": seed,
        "items": items,
    }
    (output / KIT_FILE).write_text(json.dumps(kit, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "mask_items": len(mask_items),
        "pair_items": len(pair_items),
        "grouping_version": version,
        "agent": kit["agent"],
        "next_action": "Run data audit-sitting to write the first owner sheet.",
    }


def _is_rated(row: dict[str, str]) -> bool:
    if row["kind"] == "pair":
        return row["rating"] in RATING_VALUES
    return (
        row["mask_quality"] in MASK_QUALITY_RATINGS and row["scene_tag"] in SCENE_TAGS
    )


def _check_row(row: dict[str, str]) -> None:
    """Refuse a value outside the rubric; blank cells mean not yet rated."""
    if row["kind"] == "pair":
        values = {"rating": RATING_VALUES}
    else:
        values = {"mask_quality": MASK_QUALITY_RATINGS, "scene_tag": SCENE_TAGS}
    for column, allowed in values.items():
        if row[column] and row[column] not in allowed:
            raise ValueError(f"{row['item_id']}: invalid {column} {row[column]!r}")


def _load_kit(kit_dir: Path) -> dict[str, Any]:
    kit = _read_json(Path(kit_dir) / KIT_FILE)
    if kit.get("schema") != KIT_SCHEMA:
        raise ValueError(f"kit schema must be {KIT_SCHEMA}")
    return kit


def _ratings(
    kit_dir: Path, kit: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], int]:
    """Merge the rated rows of all sitting sheets; return them and the sheet count."""
    items = {item["item_id"] for item in kit["items"]}
    sheets = sorted(Path(kit_dir).glob("sitting-*.csv"))
    rated: dict[str, dict[str, str]] = {}
    for sheet in sheets:
        with sheet.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["item_id"] not in items:
                    raise ValueError(f"{sheet.name}: unknown item {row['item_id']}")
                _check_row(row)
                if not _is_rated(row):
                    continue
                kept = {
                    "rating": row["rating"],
                    "mask_quality": row["mask_quality"],
                    "scene_tag": row["scene_tag"],
                }
                if rated.setdefault(row["item_id"], kept) != kept:
                    raise ValueError(
                        f"{row['item_id']} is rated differently in two sittings"
                    )
    return rated, len(sheets)


def next_audit_sitting(kit_dir: Path, limit: int | None = None) -> dict[str, Any]:
    """Write the next sitting sheet of unrated items and report progress."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    kit_dir = Path(kit_dir)
    kit = _load_kit(kit_dir)
    rated, sittings = _ratings(kit_dir, kit)
    pending = [item for item in kit["items"] if item["item_id"] not in rated]
    chosen = pending[:limit] if limit else pending
    summary: dict[str, Any] = {
        "rated": len(rated),
        "remaining": len(pending),
        "sheet": None,
        "items": len(chosen),
    }
    if not chosen:
        summary["next_action"] = "Every item is rated; run data audit-collect."
        return summary
    path = kit_dir / f"sitting-{sittings + 1:03d}.csv"
    _write_sheet(
        path,
        [
            {
                **{column: item.get(column, "") for column in SHEET_COLUMNS},
                "rating": "",
                "mask_quality": "",
                "scene_tag": "",
            }
            for item in chosen
        ],
    )
    summary["sheet"] = str(path)
    summary["remaining"] = len(pending) - len(chosen)
    summary["next_action"] = "Fill the sheet; rerun data audit-sitting to resume."
    return summary


def collect_audit_kit(kit_dir: Path) -> dict[str, Any]:
    """Write the owner rating sheets for evaluation once every item is rated."""
    kit_dir = Path(kit_dir)
    kit = _load_kit(kit_dir)
    rated, _ = _ratings(kit_dir, kit)
    if missing := [i["item_id"] for i in kit["items"] if i["item_id"] not in rated]:
        raise ValueError(
            f"{len(missing)} kit items are still unrated; resume a sitting"
        )
    pairs, masks = [], []
    for item in kit["items"]:
        rating = rated[item["item_id"]]
        if item["kind"] == "pair":
            pairs.append(
                {
                    "pair_id": item["item_id"].removeprefix("pair:"),
                    "image_a": item["image_a"],
                    "image_b": item["image_b"],
                    "rating": rating["rating"],
                }
            )
        else:
            masks.append(
                {
                    "audit_id": item["item_id"].removeprefix("mask:"),
                    "overlay_path": item["source_overlay_path"],
                    "mask_quality": rating["mask_quality"],
                    "scene_tag": rating["scene_tag"],
                }
            )
    for name, columns, rows in (
        (OWNER_PAIR_SHEET, PAIR_COLUMNS, pairs),
        (OWNER_MASK_SHEET, MASK_COLUMNS, masks),
    ):
        with (kit_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "grouping_version": kit["grouping_version"],
        "agent": kit["agent"],
        "pair_ratings": len(pairs),
        "mask_ratings": len(masks),
        "pair_sheet": str(kit_dir / OWNER_PAIR_SHEET),
        "mask_sheet": str(kit_dir / OWNER_MASK_SHEET),
    }

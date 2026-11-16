"""Blind owner kit that joins grouping pairs with mask overlays.

The kit holds only what the owner needs to rate: pair images and mask overlays,
with no agent ratings, algorithm decisions, strata or predicted scenes. Mask
items come first, as the mask audit asks for them before the grouping pairs.
A held-out scene set adds one scene item per image between the two: the image
alone, under a neutral name, to be given a scene tag from the rubric.
Each sitting writes a sheet of items not yet rated, so the owner can stop and
resume without seeing a rated item again.
"""

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from jute_disease.data.audit import cohort_digest, validate_review_record
from jute_disease.data.audit_sampling import grouping_threshold
from jute_disease.data.masks import (
    MASK_QUALITY_RATINGS,
    SCENE_TAGS,
)
from jute_disease.data.masks_audit import AUDIT_SCHEMA as MASK_AUDIT_SCHEMA
from jute_disease.data.rating_rubric import (
    MASK_RUBRIC,
    MASK_RUBRIC_VERSION,
    PAIR_RUBRIC,
    PAIR_RUBRIC_VERSION,
    RATING_RUBRIC_VERSION,
    SCENE_RUBRIC,
    SCENE_RUBRIC_VERSION,
)
from jute_disease.data.scene_sets import read_scene_set

LEGACY_KIT_SCHEMA = "jute-audit-kit-v1"
LEGACY_SCENE_KIT_SCHEMA = "jute-audit-kit-v2"
KIT_SCHEMA = "jute-audit-kit-v3"
SCENE_KIT_SCHEMA = "jute-audit-kit-v4"
KIT_SCHEMAS = (
    LEGACY_KIT_SCHEMA,
    LEGACY_SCENE_KIT_SCHEMA,
    KIT_SCHEMA,
    SCENE_KIT_SCHEMA,
)
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
    "rubric_version",
    "scene_rubric_version",
    "instrument_version",
    "review_status",
    "review_history",
)
PAIR_COLUMNS = (
    "pair_id",
    "image_a",
    "image_b",
    "rating",
    "rubric_version",
    "instrument_version",
    "review_status",
    "review_history",
)
MASK_COLUMNS = (
    "audit_id",
    "overlay_path",
    "mask_quality",
    "scene_tag",
    "rubric_version",
    "scene_rubric_version",
    "instrument_version",
    "review_status",
    "review_history",
)
OWNER_PAIR_SHEET = "owner.csv"
SCENE_COLUMNS = (
    "image_id",
    "scene_tag",
    "rubric_version",
    "instrument_version",
    "review_history",
)
OWNER_MASK_SHEET = "owner-masks.csv"
OWNER_SCENE_SHEET = "owner-scenes.csv"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _scene_sources(held_out: Path) -> dict[str, Path]:
    """Map each held-out image ID to its file, checking content against the ID."""
    sources: dict[str, Path] = {}
    for record in read_scene_set(held_out):
        path = Path(record["path"])
        if not path.is_file():
            raise ValueError(f"held-out image is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_id"]:
            raise ValueError(f"held-out image does not match its image ID: {path}")
        sources[record["image_id"]] = path
    return sources


def build_audit_kit(
    cohort_dir: Path,
    mask_audit_dir: Path,
    output: Path,
    seed: int = 42,
    held_out: Path | None = None,
) -> dict[str, Any]:
    """Write the blind kit and return a summary.

    The grouping cohort must name its grouping version and agent rater. The mask
    items are the owner subset of the mask audit, and their overlays are copied
    into the kit under neutral names. With ``held_out``, each image of that
    scene set becomes one scene item: a copy named by its image ID, with no
    overlay, pool, label, source path or automatic tag.
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
            "pool": row["pool"],
            "overlay_path": "",
        }
        for row in cohort["pairs"]
    ]
    scene_sources = _scene_sources(Path(held_out)) if held_out is not None else {}
    scene_items = [
        {
            "item_id": f"scene:{image_id}",
            "kind": "scene",
            "image_a": f"scenes/{image_id}{scene_sources[image_id].suffix.lower()}",
            "image_b": "",
            "overlay_path": "",
        }
        for image_id in sorted(scene_sources)
    ]
    rng.shuffle(mask_items)
    rng.shuffle(pair_items)
    # Shuffled last so a kit without scene items keeps the order it always had.
    rng.shuffle(scene_items)
    items = mask_items + scene_items + pair_items
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("kit items must be unique")
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    for item in mask_items:
        source = mask_audit_dir / item["source_overlay_path"]
        if not source.is_file():
            raise ValueError(f"mask overlay is missing: {source}")
        shutil.copyfile(source, output / item["overlay_path"])
    if scene_items:
        (output / "scenes").mkdir()
    for item in scene_items:
        image_id = item["item_id"].removeprefix("scene:")
        shutil.copyfile(scene_sources[image_id], output / item["image_a"])
    kit: dict[str, Any] = {
        "schema": SCENE_KIT_SCHEMA if scene_items else KIT_SCHEMA,
        "grouping_version": version,
        "hamming_threshold": threshold,
        "agent": {"name": agent["name"], "version": agent["version"]},
        "cohort_digest": cohort_digest(cohort),
        "rubric_version": RATING_RUBRIC_VERSION,
        "pair_rubric_version": PAIR_RUBRIC_VERSION,
        "pair_rubric": PAIR_RUBRIC,
        "mask_rubric_version": MASK_RUBRIC_VERSION,
        "mask_rubric": MASK_RUBRIC,
        "scene_rubric_version": SCENE_RUBRIC_VERSION,
        "scene_rubric": SCENE_RUBRIC,
        "seed": seed,
        "items": items,
    }
    (output / KIT_FILE).write_text(json.dumps(kit, indent=2) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "mask_items": len(mask_items),
        "scene_items": len(scene_items),
        "pair_items": len(pair_items),
        "grouping_version": version,
        "agent": kit["agent"],
        "next_action": "Run data audit-sitting to write the first owner sheet.",
    }


def audit_row_complete(row: dict[str, str]) -> bool:
    """Return whether a sitting row has all ratings required by the collector."""
    if row["kind"] == "pair":
        return row["rating"] in RATING_VALUES
    if row["kind"] == "scene":
        return row["scene_tag"] in SCENE_TAGS
    return (
        row["mask_quality"] in MASK_QUALITY_RATINGS and row["scene_tag"] in SCENE_TAGS
    )


def validate_audit_row(row: dict[str, str]) -> None:
    """Refuse a value outside the rubric; blank cells mean not yet rated."""
    if row["kind"] == "pair":
        values = {"rating": RATING_VALUES}
    elif row["kind"] == "scene":
        if row["rating"] or row["mask_quality"]:
            raise ValueError(f"{row['item_id']}: a scene item takes only a scene tag")
        values = {"scene_tag": SCENE_TAGS}
    else:
        values = {"mask_quality": MASK_QUALITY_RATINGS, "scene_tag": SCENE_TAGS}
    for column, allowed in values.items():
        if row[column] and row[column] not in allowed:
            raise ValueError(f"{row['item_id']}: invalid {column} {row[column]!r}")
    validate_review_record(
        row["kind"], row.get("review_status", ""), row.get("review_history", "")
    )


def _load_kit(kit_dir: Path) -> dict[str, Any]:
    kit = _read_json(Path(kit_dir) / KIT_FILE)
    if kit.get("schema") not in KIT_SCHEMAS:
        raise ValueError(f"kit schema must be one of {list(KIT_SCHEMAS)}")
    return kit


def _ratings(
    kit_dir: Path, kit: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], int]:
    """Merge the rated rows of all sitting sheets; return them and the sheet count."""
    items = {item["item_id"] for item in kit["items"]}
    sheets = sorted(Path(kit_dir).glob("sitting-*.csv"))
    rated: dict[str, dict[str, str]] = {}
    review_history: dict[str, list[dict[str, Any]]] = {}
    required_rubrics = {
        "pair": kit.get("pair_rubric_version"),
        "mask": kit.get("mask_rubric_version"),
        "scene": kit.get("scene_rubric_version"),
    }
    for sheet in sheets:
        with sheet.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["item_id"] not in items:
                    raise ValueError(f"{sheet.name}: unknown item {row['item_id']}")
                validate_audit_row(row)
                events = json.loads(row.get("review_history") or "[]")
                review_history.setdefault(row["item_id"], []).extend(events)
                if row.get("review_status") == "open":
                    raise ValueError(
                        f"{row['item_id']} has an unresolved review flag; "
                        "resolve it before collection"
                    )
                if not audit_row_complete(row):
                    continue
                required_rubric = required_rubrics.get(row["kind"])
                if required_rubric and row.get("rubric_version") != required_rubric:
                    legacy_unchanged_scene = (
                        row["kind"] == "scene"
                        and not row.get("rubric_version")
                        and not row.get("scene_rubric_version")
                    )
                    if not legacy_unchanged_scene:
                        raise ValueError(
                            f"{row['item_id']} uses rubric "
                            f"{row.get('rubric_version') or 'legacy-unversioned'}; "
                            f"explicitly rate it under {required_rubric} "
                            "before collection"
                        )
                if (
                    required_rubric
                    and row["kind"] in {"pair", "mask"}
                    and row.get("instrument_version") != kit.get("rubric_version")
                ):
                    raise ValueError(
                        f"{row['item_id']} uses instrument "
                        f"{row.get('instrument_version') or 'legacy-unversioned'}; "
                        f"explicitly adopt {kit.get('rubric_version')} "
                        "before collection"
                    )
                kept = {
                    "rating": row["rating"],
                    "mask_quality": row["mask_quality"],
                    "scene_tag": row["scene_tag"],
                    "rubric_version": row.get("rubric_version", ""),
                    "scene_rubric_version": row.get("scene_rubric_version", ""),
                    "instrument_version": row.get("instrument_version", ""),
                    "review_status": row.get("review_status", ""),
                }
                existing = rated.setdefault(row["item_id"], kept)
                if {key: existing[key] for key in kept} != kept:
                    raise ValueError(
                        f"{row['item_id']} is rated differently in two sittings"
                    )
                existing["review_history"] = json.dumps(
                    review_history[row["item_id"]], ensure_ascii=True
                )
    for item_id, row in rated.items():
        row["review_history"] = json.dumps(review_history[item_id], ensure_ascii=True)
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
    if any(item["kind"] == "scene" for item in chosen):
        summary["scene_rubric"] = kit.get("scene_rubric", SCENE_RUBRIC)
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
                "rubric_version": {
                    "pair": kit.get("pair_rubric_version", ""),
                    "mask": kit.get("mask_rubric_version", ""),
                    "scene": kit.get("scene_rubric_version", ""),
                }[item["kind"]],
                "scene_rubric_version": kit.get("scene_rubric_version", ""),
                "instrument_version": kit.get("rubric_version", ""),
                "review_status": "",
                "review_history": "[]",
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
    pairs, masks, scenes = [], [], []
    for item in kit["items"]:
        rating = rated[item["item_id"]]
        if item["kind"] == "pair":
            pairs.append(
                {
                    "pair_id": item["item_id"].removeprefix("pair:"),
                    "image_a": item["image_a"],
                    "image_b": item["image_b"],
                    "rating": rating["rating"],
                    "rubric_version": rating["rubric_version"],
                    "instrument_version": rating["instrument_version"],
                    "review_status": rating["review_status"],
                    "review_history": rating["review_history"],
                }
            )
        elif item["kind"] == "scene":
            scenes.append(
                {
                    "image_id": item["item_id"].removeprefix("scene:"),
                    "scene_tag": rating["scene_tag"],
                    "rubric_version": rating["scene_rubric_version"]
                    or (rating["rubric_version"] if item["kind"] == "scene" else ""),
                    "instrument_version": rating["instrument_version"],
                    "review_history": rating["review_history"],
                }
            )
        else:
            masks.append(
                {
                    "audit_id": item["item_id"].removeprefix("mask:"),
                    "overlay_path": item["source_overlay_path"],
                    "mask_quality": rating["mask_quality"],
                    "scene_tag": rating["scene_tag"],
                    "rubric_version": rating["rubric_version"],
                    "scene_rubric_version": rating["scene_rubric_version"],
                    "instrument_version": rating["instrument_version"],
                    "review_status": rating["review_status"],
                    "review_history": rating["review_history"],
                }
            )
    sheets = [
        (OWNER_PAIR_SHEET, PAIR_COLUMNS, pairs),
        (OWNER_MASK_SHEET, MASK_COLUMNS, masks),
    ]
    if scenes:
        sheets.append((OWNER_SCENE_SHEET, SCENE_COLUMNS, scenes))
    for name, columns, rows in sheets:
        with (kit_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    summary: dict[str, Any] = {
        "grouping_version": kit["grouping_version"],
        "agent": kit["agent"],
        "pair_ratings": len(pairs),
        "mask_ratings": len(masks),
        "scene_ratings": len(scenes),
        "pair_sheet": str(kit_dir / OWNER_PAIR_SHEET),
        "mask_sheet": str(kit_dir / OWNER_MASK_SHEET),
    }
    if scenes:
        summary["scene_sheet"] = str(kit_dir / OWNER_SCENE_SHEET)
        summary["scene_rubric_version"] = kit.get("scene_rubric_version")
    return summary

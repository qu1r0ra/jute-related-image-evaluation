import csv
import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.data import build_audit_kit, collect_audit_kit, next_audit_sitting
from jute_disease.data.masks import SCENE_RUBRIC

HIDDEN = ("poolA", "disease-label", "held-out-source", "reserve", "predicted")


def _sources(tmp_path: Path, scenes: int = 5) -> tuple[Path, Path, Path, list[str]]:
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    pairs = [
        {
            "pair_id": f"p{index}",
            "image_a": f"a{index}.png",
            "image_b": f"b{index}.png",
            "stratum": "near",
            "algorithm_merge": True,
            "inclusion_probability": 1.0,
        }
        for index in range(3)
    ]
    (cohort / "agent.json").write_text(
        json.dumps(
            {
                "grouping_version": "dhash-v2",
                "threshold": 6,
                "denominators": {"near": 3},
                "agent": {"name": "agent", "version": "1"},
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )
    masks = tmp_path / "masks"
    (masks / "overlays").mkdir(parents=True)
    records = []
    for index in range(4):
        (masks / "overlays" / f"m{index}.png").write_bytes(f"overlay {index}".encode())
        records.append(
            {"audit_id": f"m{index}", "overlay_path": f"overlays/m{index}.png"}
        )
    (masks / "audit.json").write_text(
        json.dumps(
            {
                "schema": "jute-mask-audit-v2",
                "records": records,
                "owner_audit_ids": ["m0", "m1", "m2"],
            }
        ),
        encoding="utf-8",
    )
    images = tmp_path / "held-out-source" / "poolA" / "disease-label"
    images.mkdir(parents=True)
    held_out, ids = [], []
    for index in range(scenes):
        image = images / f"image-{index}.jpg"
        image.write_bytes(f"scene image {index}".encode())
        image_id = hashlib.sha256(image.read_bytes()).hexdigest()
        ids.append(image_id)
        held_out.append(
            {
                "image_id": image_id,
                "path": str(image),
                "pool": "A",
                "group": f"g{index}",
                "partition": "train",
            }
        )
    held_out_file = tmp_path / "held-out.json"
    held_out_file.write_text(json.dumps(held_out), encoding="utf-8")
    return cohort, masks, held_out_file, ids


def _rows(sheet: str) -> list[dict[str, str]]:
    with Path(sheet).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fill(sheet: str, scene_tag: str = "field") -> None:
    rows = _rows(sheet)
    for row in rows:
        if row["kind"] == "pair":
            row["rating"] = "different"
        elif row["kind"] == "mask":
            row["mask_quality"], row["scene_tag"] = "pass", "unknown"
        else:
            row["scene_tag"] = scene_tag
    with Path(sheet).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_kit_without_a_held_out_set_is_unchanged(tmp_path: Path) -> None:
    cohort, masks, _, _ = _sources(tmp_path)

    summary = build_audit_kit(cohort, masks, tmp_path / "kit")

    kit = json.loads((tmp_path / "kit" / "kit.json").read_text("utf-8"))
    assert kit["schema"] == "jute-audit-kit-v1"
    assert [item["kind"] for item in kit["items"]] == ["mask"] * 3 + ["pair"] * 3
    assert summary["scene_items"] == 0
    assert not (tmp_path / "kit" / "scenes").exists()
    sitting = next_audit_sitting(tmp_path / "kit")
    assert "scene_rubric" not in sitting
    _fill(sitting["sheet"])
    collected = collect_audit_kit(tmp_path / "kit")
    assert collected["scene_ratings"] == 0
    assert not (tmp_path / "kit" / "owner-scenes.csv").exists()


def test_kit_carries_one_blind_scene_item_per_held_out_image(tmp_path: Path) -> None:
    cohort, masks, held_out, ids = _sources(tmp_path)
    plain = tmp_path / "plain"
    build_audit_kit(cohort, masks, plain, seed=3)

    summary = build_audit_kit(
        cohort, masks, tmp_path / "kit", seed=3, held_out=held_out
    )

    text = (tmp_path / "kit" / "kit.json").read_text("utf-8")
    kit = json.loads(text)
    assert kit["schema"] == "jute-audit-kit-v2"
    assert kit["scene_rubric"] == SCENE_RUBRIC
    assert summary["scene_items"] == 5
    kinds = [item["kind"] for item in kit["items"]]
    assert kinds == ["mask"] * 3 + ["scene"] * 5 + ["pair"] * 3
    scenes = [item for item in kit["items"] if item["kind"] == "scene"]
    assert sorted(item["item_id"] for item in scenes) == sorted(
        f"scene:{image_id}" for image_id in ids
    )
    assert [item["item_id"] for item in scenes] != [f"scene:{i}" for i in ids]
    for item in scenes:
        image_id = item["item_id"].removeprefix("scene:")
        assert item["image_a"] == f"scenes/{image_id}.jpg"
        assert item["image_b"] == item["overlay_path"] == ""
        assert set(item) == {"item_id", "kind", "image_a", "image_b", "overlay_path"}
        copied = tmp_path / "kit" / item["image_a"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == image_id
    assert not any(token in text for token in HIDDEN)
    plain_items = json.loads((plain / "kit.json").read_text("utf-8"))["items"]
    assert [i for i in kit["items"] if i["kind"] != "scene"] == plain_items


def test_scene_items_round_trip_to_their_own_owner_sheet(tmp_path: Path) -> None:
    cohort, masks, held_out, ids = _sources(tmp_path)
    kit_dir = tmp_path / "kit"
    build_audit_kit(cohort, masks, kit_dir, held_out=held_out)

    first = next_audit_sitting(kit_dir, limit=6)
    assert first["scene_rubric"] == SCENE_RUBRIC
    rows = _rows(first["sheet"])
    scene_rows = [row for row in rows if row["kind"] == "scene"]
    assert len(scene_rows) == 3
    assert all(
        row["image_a"].startswith("scenes/") and not row["overlay_path"]
        for row in scene_rows
    )
    assert not any(token in Path(first["sheet"]).read_text("utf-8") for token in HIDDEN)
    _fill(first["sheet"], "clean-background")
    with pytest.raises(ValueError, match="still unrated"):
        collect_audit_kit(kit_dir)
    second = next_audit_sitting(kit_dir)
    assert second["items"] == 5
    _fill(second["sheet"], "field")
    assert next_audit_sitting(kit_dir)["sheet"] is None

    collected = collect_audit_kit(kit_dir)

    assert collected["scene_ratings"] == 5
    assert collected["mask_ratings"] == 3 and collected["pair_ratings"] == 3
    sheet = _rows(collected["scene_sheet"])
    assert Path(collected["scene_sheet"]).name == "owner-scenes.csv"
    assert sorted(row["image_id"] for row in sheet) == sorted(ids)
    assert {row["rubric_version"] for row in sheet} == {"scene-tag-v2"}
    assert (
        sorted(row["scene_tag"] for row in sheet)
        == ["clean-background"] * 3 + ["field"] * 2
    )
    assert len(_rows(collected["mask_sheet"])) == 3


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("scene_tag", "forest", "invalid scene_tag"),
        ("mask_quality", "pass", "takes only a scene tag"),
        ("rating", "exact", "takes only a scene tag"),
    ],
)
def test_scene_rows_take_only_a_rubric_scene_tag(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    cohort, masks, held_out, _ = _sources(tmp_path)
    kit_dir = tmp_path / "kit"
    build_audit_kit(cohort, masks, kit_dir, held_out=held_out)
    sheet = next_audit_sitting(kit_dir)["sheet"]
    rows = _rows(sheet)
    next(row for row in rows if row["kind"] == "scene")[column] = value
    with Path(sheet).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        next_audit_sitting(kit_dir)


def test_kit_refuses_a_held_out_image_that_does_not_match_its_id(
    tmp_path: Path,
) -> None:
    cohort, masks, held_out, _ = _sources(tmp_path)
    records = json.loads(held_out.read_text("utf-8"))
    Path(records[0]["path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="does not match its image ID"):
        build_audit_kit(cohort, masks, tmp_path / "kit", held_out=held_out)

    assert not (tmp_path / "kit").exists()


def test_audit_kit_command_accepts_the_held_out_set(tmp_path: Path, capsys) -> None:
    cohort, masks, held_out, _ = _sources(tmp_path)

    main(
        [
            "data",
            "audit-kit",
            str(cohort),
            str(masks),
            str(tmp_path / "kit"),
            "--held-out",
            str(held_out),
        ]
    )

    assert json.loads(capsys.readouterr().out)["scene_items"] == 5
    main(["data", "audit-sitting", str(tmp_path / "kit")])
    assert json.loads(capsys.readouterr().out)["scene_rubric"]["version"] == (
        "scene-tag-v2"
    )

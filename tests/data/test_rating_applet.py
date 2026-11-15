import csv
import hashlib
import json
from pathlib import Path

from jute_disease.rating_applet import create_app


def test_load_resumes_first_incomplete_item_without_exposing_identity(
    tmp_path: Path,
) -> None:
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "kit.json").write_text(json.dumps({"scene_rubric": {}}), encoding="utf-8")
    sheet = kit / "sitting-001.csv"
    columns = [
        "item_id",
        "kind",
        "image_a",
        "image_b",
        "overlay_path",
        "rating",
        "mask_quality",
        "scene_tag",
    ]
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [
                dict(
                    zip(
                        columns,
                        [
                            "secret-1",
                            "scene",
                            "scenes/secret.png",
                            "",
                            "",
                            "",
                            "",
                            "field",
                        ],
                        strict=True,
                    )
                ),
                dict(
                    zip(
                        columns,
                        [
                            "secret-2",
                            "mask",
                            "",
                            "",
                            "overlays/secret.png",
                            "",
                            "pass",
                            "",
                        ],
                        strict=True,
                    )
                ),
            ]
        )
    original = sheet.read_bytes()

    response = create_app(sheet).test_client().get("/api/sitting")

    assert response.status_code == 200
    assert response.json["position"] == 2
    assert response.json["completed"] == 1
    assert b"secret" not in response.data
    assert sheet.read_bytes() == original


def test_save_survives_restart_and_preserves_other_values(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag,other\n"
        "private,mask,,,overlays/a.png,,pass,,keep\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]

    saved = client.post(
        "/api/items/1",
        json={"revision": revision, "field": "scene_tag", "value": "field"},
    )

    assert saved.status_code == 200
    reopened = create_app(sheet).test_client().get("/api/sitting").json
    assert reopened["completed"] == 1
    assert "private,mask,,,overlays/a.png,,pass,field,keep" in sheet.read_text()


def test_unchanged_choice_preserves_sheet_bytes(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    original = b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,field\n"
    sheet.write_bytes(original)
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    response = client.post(
        "/api/items/1",
        json={"revision": revision, "field": "scene_tag", "value": "field"},
    )
    assert response.status_code == 200
    assert sheet.read_bytes() == original


def test_collected_sitting_refuses_changes(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    original = b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,field\n"
    sheet.write_bytes(original)
    (tmp_path / "owner-scenes.csv").write_text("collected\n", encoding="utf-8")
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]

    response = client.post(
        "/api/items/1",
        json={"revision": revision, "field": "scene_tag", "value": "unknown"},
    )

    assert response.status_code == 403
    assert sheet.read_bytes() == original


def test_opening_partial_sitting_records_private_adoption(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    original = b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,field\n"
    sheet.write_bytes(original)

    client = create_app(sheet).test_client()
    assert client.get("/api/sitting").status_code == 200

    runtime = tmp_path / ".rating-applet"
    backups = list(runtime.rglob("*.backup"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    records = list(runtime.rglob("*.jsonl"))
    assert len(records) == 1
    event = json.loads(records[0].read_text(encoding="utf-8").splitlines()[0])
    assert event["type"] == "adopt"
    assert event["completed_items"] == ["private"]
    assert event["digest"] == hashlib.sha256(original).hexdigest()
    assert event["instrument_version"]
    assert sheet.read_bytes() == original


def test_null_choice_is_refused_without_changing_sheet(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    original = b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,field\n"
    sheet.write_bytes(original)
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]

    response = client.post(
        "/api/items/1",
        json={"revision": revision, "field": "scene_tag", "value": None},
    )

    assert response.status_code == 400
    assert sheet.read_bytes() == original


def test_finish_requires_completion_and_protects_sitting(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,scenes/a.png,,,,,\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    assert client.post("/api/finish", json={"revision": revision}).status_code == 409
    saved = client.post(
        "/api/items/1",
        json={"revision": revision, "field": "scene_tag", "value": "field"},
    )
    assert (
        client.post(
            "/api/finish", json={"revision": saved.json["revision"]}
        ).status_code
        == 200
    )
    reopened = create_app(sheet).test_client()
    assert reopened.get("/api/sitting").json["read_only"] is True
    assert (
        reopened.post(
            "/api/items/1",
            json={
                "revision": saved.json["revision"],
                "field": "scene_tag",
                "value": "unknown",
            },
        ).status_code
        == 403
    )

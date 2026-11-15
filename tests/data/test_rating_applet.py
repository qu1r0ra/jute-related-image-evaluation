import csv
import hashlib
import json
from pathlib import Path

import pytest

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


def test_interrupted_replacement_recovers_confirmed_revision(
    tmp_path: Path, monkeypatch
) -> None:
    import os

    sheet = tmp_path / "sitting-001.csv"
    original = b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,\n"
    sheet.write_bytes(original)
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    with monkeypatch.context() as patch:
        patch.setattr(
            os, "replace", lambda *_: (_ for _ in ()).throw(OSError("failed"))
        )
        response = client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": "field"},
        )
    assert response.status_code != 200
    assert sheet.read_bytes() == original
    reopened = create_app(sheet).test_client().get("/api/sitting")
    assert reopened.status_code == 200
    assert reopened.json["revision"] == revision
    assert reopened.json["completed"] == 0


def test_external_edit_requires_recovery(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,\n"
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    sheet.write_bytes(sheet.read_bytes().replace(b"a.png", b"b.png"))
    assert (
        client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": "field"},
        ).status_code
        == 409
    )
    assert create_app(sheet).test_client().get("/api/sitting").status_code == 409


def test_save_interrupted_after_replacement_recovers_as_committed(
    tmp_path: Path, monkeypatch
) -> None:
    import os

    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,\n"
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    replace = os.replace

    def interrupted(source, destination) -> None:
        replace(source, destination)
        raise OSError("interrupted after replacement")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        response = client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": "field"},
        )
    assert response.status_code != 200
    reopened = create_app(sheet).test_client().get("/api/sitting")
    assert reopened.status_code == 200
    assert reopened.json["completed"] == 1
    assert reopened.json["revision"] != revision


def test_reopening_requires_explicit_reason_and_is_recorded(tmp_path: Path) -> None:
    from jute_disease.rating_applet import reopen_sitting

    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,field\n"
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    assert client.post("/api/finish", json={"revision": revision}).status_code == 200
    with pytest.raises(ValueError):
        reopen_sitting(sheet, " ")
    reopen_sitting(sheet, "Correct an entry error")
    revision = client.get("/api/sitting").json["revision"]
    assert (
        client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": "unknown"},
        ).status_code
        == 200
    )
    events = [
        json.loads(line)
        for line in next((tmp_path / ".rating-applet").rglob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert any(
        event["type"] == "reopen" and event["reason"] == "Correct an entry error"
        for event in events
    )
    (tmp_path / "owner-scenes.csv").write_text("collected", encoding="utf-8")
    with pytest.raises(ValueError):
        reopen_sitting(sheet, "Another correction")


def _concurrent_save(
    sheet: str, revision: str, ready, start, results, value: str
) -> None:
    client = create_app(Path(sheet)).test_client()
    ready.put(True)
    start.wait(30)
    results.put(
        client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": value},
        ).status_code
    )


@pytest.mark.serial
def test_two_processes_cannot_both_write_same_revision(tmp_path: Path) -> None:
    import multiprocessing

    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,\n"
    )
    revision = create_app(sheet).test_client().get("/api/sitting").json["revision"]
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    processes = [
        context.Process(
            target=_concurrent_save,
            args=(str(sheet), revision, ready, start, results, value),
        )
        for value in ("field", "unknown")
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            assert ready.get(timeout=60)
        start.set()
        assert sorted(results.get(timeout=60) for _ in processes) == [200, 409]
    finally:
        start.set()
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert create_app(sheet).test_client().get("/api/sitting").json["completed"] == 1


def test_synthetic_app_ratings_are_collected(tmp_path: Path) -> None:
    from jute_disease.data import collect_audit_kit, next_audit_sitting

    kit = {
        "schema": "jute-audit-kit-v2",
        "grouping_version": "test",
        "agent": {},
        "items": [
            {
                "item_id": "mask:m",
                "kind": "mask",
                "overlay_path": "overlays/m.png",
                "source_overlay_path": "original/m.png",
            },
            {"item_id": "scene:s", "kind": "scene", "image_a": "scenes/s.png"},
        ],
    }
    (tmp_path / "kit.json").write_text(json.dumps(kit), encoding="utf-8")
    sheet = Path(next_audit_sitting(tmp_path)["sheet"])
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    for position, field, value in [
        (1, "mask_quality", "pass"),
        (1, "scene_tag", "unknown"),
        (2, "scene_tag", "field"),
    ]:
        response = client.post(
            f"/api/items/{position}",
            json={"revision": revision, "field": field, "value": value},
        )
        assert response.status_code == 200
        revision = response.json["revision"]
    result = collect_audit_kit(tmp_path)
    assert result["mask_ratings"] == result["scene_ratings"] == 1
    with (tmp_path / "owner-masks.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["mask_quality"] == "pass"
    with (tmp_path / "owner-scenes.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["scene_tag"] == "field"
    assert (
        client.post(
            "/api/items/2",
            json={"revision": revision, "field": "scene_tag", "value": "unknown"},
        ).status_code
        == 403
    )


def test_ambiguous_interrupted_save_blocks_reopening(
    tmp_path: Path, monkeypatch
) -> None:
    import os

    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,scene,scenes/a.png,,,,,\n"
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]

    def external_change(source, destination) -> None:
        sheet.write_bytes(sheet.read_bytes().replace(b"a.png", b"b.png"))
        raise OSError("external change during replacement")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", external_change)
        response = client.post(
            "/api/items/1",
            json={"revision": revision, "field": "scene_tag", "value": "field"},
        )
    assert response.status_code != 200
    assert create_app(sheet).test_client().get("/api/sitting").status_code == 409


def test_old_revision_is_stale_even_after_values_return_to_original(
    tmp_path: Path,
) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\r\nprivate,scene,scenes/a.png,,,,,field\r\n"
    )
    client = create_app(sheet).test_client()
    old = client.get("/api/sitting").json["revision"]
    changed = client.post(
        "/api/items/1", json={"revision": old, "field": "scene_tag", "value": "unknown"}
    )
    restored = client.post(
        "/api/items/1",
        json={
            "revision": changed.json["revision"],
            "field": "scene_tag",
            "value": "field",
        },
    )
    assert restored.status_code == 200
    assert (
        client.post(
            "/api/items/1",
            json={"revision": old, "field": "scene_tag", "value": "unknown"},
        ).status_code
        == 409
    )


def test_invalid_external_edit_opens_in_recovery_state(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_bytes(
        b"item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nsecret-item,scene,scenes/a.png,,,,,field\n"
    )
    create_app(sheet)
    sheet.write_bytes(sheet.read_bytes().replace(b"field", b"invalid"))
    response = create_app(sheet).test_client().get("/api/sitting")
    assert response.status_code == 409
    assert b"secret-item" not in response.data


def test_image_routes_only_serve_contained_sitting_images(tmp_path: Path) -> None:
    from PIL import Image

    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "scenes").mkdir()
    (kit / "kit.json").write_text('{"scene_rubric": {"anchors": {}}}', encoding="utf-8")
    Image.new("RGB", (1800, 900), "gray").save(kit / "scenes" / "private.png")
    sheet = kit / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,scenes/private.png,,,,,\n"
        "escape,scene,../outside.png,,,,,\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    item = client.get("/api/items/1")
    assert item.status_code == 200
    assert b"private" not in item.data
    preview = client.get("/images/1/preview")
    assert preview.status_code == 200
    from io import BytesIO

    assert Image.open(BytesIO(preview.data)).size == (1400, 700)
    assert client.get("/images/1/original").status_code == 200
    assert client.get("/images/2/original").status_code == 404
    assert client.get("/images/3/original").status_code == 404

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
    assert reopened.get("/api/sitting").json["finished"] is True
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


def test_synthetic_pair_mask_scene_roundtrip_through_collector(
    tmp_path: Path,
) -> None:
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
            {
                "item_id": "pair:p",
                "kind": "pair",
                "pool": "A",
                "image_a": "a.png",
                "image_b": "b.png",
            },
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
        (3, "rating", "near-duplicate"),
    ]:
        response = client.post(
            f"/api/items/{position}",
            json={"revision": revision, "field": field, "value": value},
        )
        assert response.status_code == 200
        revision = response.json["revision"]
    result = collect_audit_kit(tmp_path)
    assert result["mask_ratings"] == result["scene_ratings"] == 1
    assert result["pair_ratings"] == 1
    with (tmp_path / "owner-masks.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["mask_quality"] == "pass"
    with (tmp_path / "owner-scenes.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["scene_tag"] == "field"
    with (tmp_path / "owner.csv").open(newline="", encoding="utf-8") as handle:
        pair = list(csv.DictReader(handle))[0]
    assert pair == {
        "pair_id": "p",
        "image_a": "a.png",
        "image_b": "b.png",
        "rating": "near-duplicate",
    }
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


@pytest.mark.serial
def test_browser_navigation_disables_old_choices_until_new_item_is_loaded(
    tmp_path: Path,
) -> None:
    from threading import Event, Thread

    from flask import request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 60), "gray").save(tmp_path / "scene-a.png")
    Image.new("RGB", (80, 60), "blue").save(tmp_path / "scene-b.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private-a,scene,scene-a.png,,,,,field\n"
        "private-b,scene,scene-b.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    delay_second_item_request = Event()
    item_request_seen = Event()
    release_item_request = Event()

    @app.before_request
    def hold_second_item_request() -> object:
        if (
            delay_second_item_request.is_set()
            and request.method == "GET"
            and request.path == "/api/items/2"
        ):
            item_request_seen.set()
            assert release_item_request.wait(10)
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            requests: list[tuple[str, str]] = []
            page.on(
                "request",
                lambda browser_request: requests.append(
                    (browser_request.method, browser_request.url)
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Unknown").wait_for()
            page.wait_for_function(
                "document.querySelector('#image-frame img')?.complete "
                "&& document.querySelector('#image-frame img')?.naturalWidth > 0"
            )

            page.get_by_role("button", name="Previous").click()
            page.get_by_text("Item 1 of 2", exact=True).wait_for()
            page.wait_for_function(
                "document.querySelector('#image-frame img')?.complete "
                "&& document.querySelector('#image-frame img')?.naturalWidth > 0"
            )
            delay_second_item_request.set()
            page.get_by_role("button", name="Next").click()
            assert item_request_seen.wait(5)
            assert page.get_by_role("button", name="Unknown").is_disabled()
            page.keyboard.press("E")
            assert not any(
                method == "POST" and url.endswith("/api/items/2")
                for method, url in requests
            )
            release_item_request.set()
            page.get_by_text("Item 2 of 2", exact=True).wait_for()
            page.get_by_role("button", name="Unknown").wait_for(state="visible")
            assert not any(
                method == "POST" and url.endswith("/api/items/2")
                for method, url in requests
            )
            with sheet.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert rows[1]["scene_tag"] == ""
            browser.close()
    finally:
        release_item_request.set()
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_save_refuses_a_revision_changed_before_refresh(
    tmp_path: Path,
) -> None:
    from threading import Thread

    from flask import request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 60), "gray").save(tmp_path / "scene.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,scene.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    inject_external_change = True

    @app.after_request
    def simulate_another_tab(response) -> object:
        nonlocal inject_external_change
        if (
            inject_external_change
            and request.method == "POST"
            and request.path == "/api/items/1"
            and response.status_code == 200
        ):
            inject_external_change = False
            external = app.test_client()
            revision = external.get("/api/sitting").json["revision"]
            changed = external.post(
                "/api/items/1",
                json={
                    "revision": revision,
                    "field": "scene_tag",
                    "value": "unknown",
                },
            )
            assert changed.status_code == 200
        return response

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Field").wait_for()
            page.wait_for_function(
                "document.querySelector('#image-frame img')?.complete "
                "&& document.querySelector('#image-frame img')?.naturalWidth > 0"
            )

            page.keyboard.press("W")
            page.get_by_text(
            "Couldn't save this choice: The sitting changed. Reload it. Reload to check the value before making another choice.",
                exact=True,
            ).wait_for()
            assert page.get_by_role("button", name="Field").is_disabled()
            with sheet.open(newline="", encoding="utf-8") as handle:
                assert next(csv.DictReader(handle))["scene_tag"] == "unknown"
            browser.close()
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_scene_shortcuts_do_not_submit_mask_fields(tmp_path: Path) -> None:
    from threading import Thread

    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 60), "gray").save(tmp_path / "scene.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,scene.png,,,,,\n",
        encoding="utf-8",
    )
    server = make_server("127.0.0.1", 0, create_app(sheet), threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            requests: list[tuple[str, str]] = []
            page.on(
                "request",
                lambda browser_request: requests.append(
                    (browser_request.method, browser_request.url)
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Field").wait_for()
            page.wait_for_function(
                "document.querySelector('#image-frame img')?.complete "
                "&& document.querySelector('#image-frame img')?.naturalWidth > 0"
            )

            page.keyboard.press("1")
            assert not any(
                method == "POST" and url.endswith("/api/items/1")
                for method, url in requests
            )
            page.keyboard.press("W")
            page.get_by_text("Saved", exact=True).wait_for()
            with sheet.open(newline="", encoding="utf-8") as handle:
                saved = next(csv.DictReader(handle))
            assert saved["mask_quality"] == ""
            assert saved["scene_tag"] == "field"
            browser.close()
    finally:
        server.shutdown()
        thread.join()


def test_reopen_cli_translates_invalid_state_to_argument_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import sys

    from jute_disease.rating_applet.__main__ import main

    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,scene.png,,,,,field\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rating-applet", str(sheet), "--reopen-reason", "Correct an entry error"],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 2
    output = capsys.readouterr()
    assert "error: Unable to reopen sitting: This sitting cannot be reopened." in (
        output.err
    )
    assert "Traceback" not in output.err


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


def test_preview_cache_is_private_and_refreshes_when_source_changes(
    tmp_path: Path,
) -> None:
    from io import BytesIO

    from PIL import Image

    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "kit.json").write_text('{"scene_rubric": {"anchors": {}}}', encoding="utf-8")
    source = kit / "image.png"
    Image.new("RGB", (1800, 900), "red").save(source)
    sheet = kit / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    client = create_app(sheet).test_client()

    first = client.get("/images/1/preview")
    first_original = client.get("/images/1/original")

    assert first.status_code == 200
    assert first_original.status_code == 200
    cache = kit / ".rating-applet" / sheet.name / "previews"
    original_cache = kit / ".rating-applet" / sheet.name / "originals"
    cached_files = list(cache.glob("*.png"))
    assert len(cached_files) == 1
    assert len(list(original_cache.glob("*.png"))) == 1
    assert source.read_bytes() == original
    first_color = Image.open(BytesIO(first.data)).getpixel((20, 20))
    assert first_color == (255, 0, 0)
    first_original_color = Image.open(BytesIO(first_original.data)).getpixel((20, 20))
    assert first_original_color == (255, 0, 0)
    assert Image.open(BytesIO(first_original.data)).size == (1800, 900)
    assert client.get("/images/1/original").data == first_original.data
    Image.new("RGB", (1800, 900), "blue").save(source)

    second = client.get("/images/1/preview")
    second_original = client.get("/images/1/original")

    assert second.status_code == 200
    assert second_original.status_code == 200
    assert len(list(cache.glob("*.png"))) == 2
    assert len(list(original_cache.glob("*.png"))) == 2
    assert Image.open(BytesIO(second.data)).getpixel((20, 20)) == (0, 0, 255)
    assert Image.open(BytesIO(second_original.data)).getpixel((20, 20)) == (0, 0, 255)
    assert b"private" not in second.data


@pytest.mark.serial
def test_browser_rates_and_finishes_synthetic_scene(tmp_path: Path) -> None:
    from threading import Thread

    from flask import jsonify, request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {"field": "Outdoor structure"}}}),
        encoding="utf-8",
    )
    Image.new("RGB", (80, 40), "gray").save(tmp_path / "image.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nsecret,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    fail_finish_once = True

    @app.before_request
    def fail_first_finish() -> object:
        nonlocal fail_finish_once
        if request.method == "POST" and request.path == "/api/finish" and fail_finish_once:
            fail_finish_once = False
            return jsonify(error="Synthetic finish failure."), 409
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Field").click()
            page.get_by_text("Saved", exact=True).wait_for()
            page.get_by_role("button", name="Finish sitting").click()
            page.get_by_role("button", name="Confirm finish sitting").wait_for()
            assert page.get_by_text("Read-only", exact=True).count() == 0
            page.get_by_role("button", name="Confirm finish sitting").click()
            page.get_by_text(
                "Couldn't finish the sitting: Synthetic finish failure. Reload to check its state. If it remains editable, you can try finishing again.",
                exact=True,
            ).wait_for()
            assert page.get_by_text("Read-only", exact=True).count() == 0
            page.reload()
            page.get_by_role("button", name="Finish sitting").wait_for()
            page.get_by_role("button", name="Finish sitting").click()
            page.get_by_role("button", name="Confirm finish sitting").wait_for()
            page.get_by_role("button", name="Confirm finish sitting").click()
            page.get_by_text("Sitting finished", exact=False).wait_for()
            page.reload()
            page.get_by_text("This sitting is finished and read-only.", exact=True).wait_for()
            page.get_by_text("Read-only", exact=True).wait_for()
            browser.close()
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_delayed_save_blocks_more_input_until_confirmed(tmp_path: Path) -> None:
    from threading import Event, Thread

    from flask import request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 40), "gray").save(tmp_path / "image.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    save_started, release_save = Event(), Event()

    @app.before_request
    def delay_rating_save() -> None:
        if request.method == "POST" and request.path == "/api/items/1":
            save_started.set()
            release_save.wait(10)
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Field").click()
            assert save_started.wait(10)
            page.get_by_text("Saving…", exact=True).wait_for()
            assert page.get_by_role("button", name="Field").is_disabled()
            assert page.get_by_role("button", name="Next").is_disabled()
            assert page.get_by_text("Item 1 of 1", exact=True).is_visible()
            page.locator("#position").evaluate("heading => heading.focus()")
            release_save.set()
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.get_by_text("Item 1 of 1", exact=True).is_visible()
            assert page.evaluate("document.activeElement.id") == "position"
            browser.close()
    finally:
        release_save.set()
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_failed_save_is_not_claimed_saved_and_can_resume(
    tmp_path: Path,
) -> None:
    from threading import Thread

    from flask import jsonify, request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 40), "gray").save(tmp_path / "image.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,image.png,,,,,field\n"
        "private-incomplete,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    fail_next_save = True

    @app.before_request
    def fail_first_rating_save() -> object:
        nonlocal fail_next_save
        if (
            request.method == "POST"
            and request.path == "/api/items/1"
            and fail_next_save
        ):
            fail_next_save = False
            return jsonify(error="Synthetic save failure."), 409
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_text("Item 2 of 2", exact=True).wait_for()
            page.get_by_role("button", name="Previous").click()
            page.get_by_text("Item 1 of 2", exact=True).wait_for()
            page.get_by_role("button", name="Unknown").click()
            page.get_by_text(
                "Couldn't save this choice: Synthetic save failure. Reload to check the value before making another choice.",
                exact=True,
            ).wait_for()
            assert page.get_by_role("button", name="Unknown").is_disabled()
            assert page.get_by_role("button", name="Next").is_disabled()
            assert page.get_by_text(
                "1 completed · 1 remaining", exact=True
            ).is_visible()

            page.reload()
            page.get_by_text("Item 1 of 2", exact=True).wait_for()
            page.get_by_text(
                "A previous save did not confirm. The current value was reloaded from the sitting; review it before continuing.",
                exact=True,
            ).wait_for()
            assert page.evaluate(
                "sessionStorage.getItem('rating-applet-recovery-position')"
            ) is None
            page.get_by_role("button", name="Unknown").wait_for(state="visible")
            assert page.get_by_role("button", name="Unknown").is_enabled()
            page.get_by_role("button", name="Unknown").click()
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.get_by_text(
                "1 completed · 1 remaining", exact=True
            ).is_visible()
            assert page.evaluate(
                "sessionStorage.getItem('rating-applet-recovery-position')"
            ) is None
            page.get_by_role("button", name="Next").click()
            page.get_by_text("Item 2 of 2", exact=True).wait_for()
            page.reload()
            page.get_by_text("Item 2 of 2", exact=True).wait_for()
            assert page.get_by_text("A previous save did not confirm.").count() == 0
            browser.close()
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_opens_app_from_a_cross_site_top_level_link(tmp_path: Path) -> None:
    from threading import Thread

    from flask import request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (80, 40), "gray").save(tmp_path / "image.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet)
    navigation_sites = []

    @app.before_request
    def observe_navigation_site() -> None:
        if request.path == "/":
            navigation_sites.append(request.headers.get("Sec-Fetch-Site"))
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            source_port = server.server_port + 1
            source_url = f"http://localhost:{source_port}/"
            target_url = f"http://127.0.0.1:{server.server_port}/"
            page.route(
                source_url,
                lambda route: route.fulfill(
                    content_type="text/html",
                    body=f'<a href="{target_url}">Open rating app</a>',
                ),
            )
            page.goto(source_url)
            page.get_by_role("link", name="Open rating app").click()
            page.get_by_role("heading", name="Rating sitting").wait_for()
            page.get_by_role("img", name="Image for this item").wait_for()
            assert navigation_sites == ["cross-site"]
            assert page.get_by_text(
                "0 completed · 1 remaining", exact=True
            ).is_visible()
            browser.close()
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_mask_keyboard_undo_and_unreadable_image(tmp_path: Path) -> None:
    from threading import Thread

    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    (tmp_path / "kit.json").write_text(
        json.dumps({"scene_rubric": {"anchors": {}}}), encoding="utf-8"
    )
    Image.new("RGB", (120, 90), "gray").save(tmp_path / "mask.png")
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nsecret,mask,,,mask.png,,,\nsecret2,scene,missing.png,,,,,\n",
        encoding="utf-8",
    )
    server = make_server("127.0.0.1", 0, create_app(sheet), threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Pass").wait_for()
            page.get_by_role("img", name="Image for this item").wait_for()
            page.keyboard.press("1")
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.evaluate(
                "document.activeElement.dataset.value"
            ) == "pass"
            page.evaluate("document.activeElement.blur()")
            page.keyboard.press("w")
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.evaluate(
                "document.activeElement.dataset.value"
            ) == "field"
            page.get_by_role("button", name="Next").wait_for(state="visible")
            page.wait_for_function("!document.querySelector('#next').disabled")
            page.get_by_role("button", name="Undo last choice").click()
            page.wait_for_function(
                "document.querySelector('#next').disabled && document.querySelector('#save').textContent === 'Saved'"
            )
            page.keyboard.press("w")
            page.wait_for_function("!document.querySelector('#next').disabled")
            page.get_by_role("button", name="Inspect full resolution").click()
            page.get_by_role("button", name="Fit preview").wait_for()
            page.wait_for_function("!document.querySelector('#next').disabled")
            page.get_by_role("button", name="Next").click()
            page.get_by_text(
                "This image could not be loaded. Rating and navigation are blocked. Check that its file is available, then reload the sitting.",
                exact=True,
            ).wait_for()
            assert page.get_by_role("button", name="Next").is_disabled()
            assert page.get_by_role("button", name="Field").is_disabled()
            assert page.get_by_role("button", name="Clean background").is_disabled()
            page.keyboard.press("w")
            with sheet.open(newline="", encoding="utf-8") as handle:
                unreadable_row = list(csv.DictReader(handle))[1]
            assert unreadable_row["scene_tag"] == ""
            browser.close()
    finally:
        server.shutdown()
        thread.join()


def test_cross_site_requests_cannot_read_local_sitting(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nsecret,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    assert (
        client.get("/api/sitting", headers={"Host": "attacker.example"}).status_code
        == 403
    )
    assert (
        client.get("/api/sitting", headers={"Sec-Fetch-Site": "cross-site"}).status_code
        == 403
    )


def test_cross_site_top_level_navigation_opens_only_the_static_app_shell(
    tmp_path: Path,
) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "secret,scene,image.png,,,,,\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    navigation = {
        "Host": "127.0.0.1:8765",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    }

    page = client.get("/", headers=navigation)

    assert page.status_code == 200
    assert page.headers["Cache-Control"] == "no-store"
    assert b"secret" not in page.data
    assert client.get("/static/app.js").headers["Cache-Control"] == "no-store"
    assert client.get("/api/sitting", headers=navigation).status_code == 403
    assert client.post("/api/items/1", headers=navigation).status_code == 403


def test_pair_sitting_can_be_finished_with_existing_rating(tmp_path: Path) -> None:
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\nprivate,pair,a.png,b.png,,exact,,\n",
        encoding="utf-8",
    )
    client = create_app(sheet).test_client()
    revision = client.get("/api/sitting").json["revision"]
    assert client.get("/api/items/1").json["kind"] == "pair"
    response = client.post("/api/finish", json={"revision": revision})
    assert response.status_code == 200
    assert response.json["read_only"] is True


def test_pair_http_api_exposes_blind_choices_and_opaque_allowlisted_images(
    tmp_path: Path,
) -> None:
    from PIL import Image

    kit = tmp_path / "kit"
    pool = tmp_path / "pool"
    kit.mkdir()
    pool.mkdir()
    (pool / "a.png").parent.mkdir(exist_ok=True)
    Image.new("RGB", (2400, 1200), "red").save(pool / "a.png")
    Image.new("RGB", (800, 1600), "blue").save(pool / "b.png")
    Image.new("RGB", (20, 20), "green").save(tmp_path / "outside.png")
    (kit / "kit.json").write_text(
        json.dumps(
            {
                "items": [
                    {"item_id": "private-pair", "kind": "pair", "pool": "A"},
                    {"item_id": "escape-pair", "kind": "pair", "pool": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    sheet = kit / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private-pair,pair,A/a.png,A/b.png,,,,\n"
        "escape-pair,pair,A/../outside.png,A/b.png,,,,\n",
        encoding="utf-8",
    )

    client = create_app(sheet, pool_roots={"A": pool}).test_client()

    sitting = client.get("/api/sitting").json
    item = client.get("/api/items/1")
    assert item.status_code == 200
    assert item.json["kind"] == "pair"
    assert item.json["rating"] == ""
    assert item.json["ratings"] == [
        "exact",
        "near-duplicate",
        "very similar",
        "different",
    ]
    assert item.json["pair_boundary"]["near_duplicate"] == (
        "the same physical leaf or scene captured again, such as a burst frame, "
        "a small shift or zoom, a crop or a re-save"
    )
    assert "private-pair" not in item.get_data(as_text=True)
    assert "pool" not in item.get_data(as_text=True)

    for side in ("a", "b"):
        response = client.get(f"/images/1/{side}/preview")
        assert response.status_code == 200
        assert response.content_type == "image/png"
        assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/images/1/c/original").status_code == 404
    assert client.get("/images/2/a/preview").status_code == 404

    invalid = client.post(
        "/api/items/1",
        json={"revision": sitting["revision"], "field": "scene_tag", "value": "field"},
    )
    assert invalid.status_code == 400
    invalid_value = client.post(
        "/api/items/1",
        json={
            "revision": sitting["revision"],
            "field": "rating",
            "value": "not a rubric value",
        },
    )
    assert invalid_value.status_code == 400
    revision = sitting["revision"]
    for rating in ("exact", "near-duplicate", "very similar", "different"):
        saved = client.post(
            "/api/items/1",
            json={"revision": revision, "field": "rating", "value": rating},
        )
        assert saved.status_code == 200
        revision = saved.json["revision"]
    with sheet.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["rating"] == "different"


@pytest.mark.serial
def test_browser_rates_pair_with_keyboard_zoom_swap_and_finish(tmp_path: Path) -> None:
    from threading import Thread

    from flask import jsonify, request
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    pool = tmp_path / "pool"
    pool.mkdir()
    Image.new("RGB", (1600, 800), "red").save(pool / "a.png")
    Image.new("RGB", (800, 1600), "blue").save(pool / "b.png")
    Image.new("RGB", (80, 120), "gray").save(tmp_path / "scene.png")
    Image.new("RGB", (100, 60), "green").save(tmp_path / "mask.png")
    (tmp_path / "kit.json").write_text(
        json.dumps(
            {
                "scene_rubric": {"anchors": {"field": "Outdoor structure"}},
                "items": [
                    {"item_id": "private-pair-a", "kind": "pair", "pool": "A"},
                    {"item_id": "private-pair-b", "kind": "pair", "pool": "A"},
                    {
                        "item_id": "private-scene",
                        "kind": "scene",
                        "image_a": "scene.png",
                    },
                    {"item_id": "private-mask", "kind": "mask"},
                ],
            }
        ),
        encoding="utf-8",
    )
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "private-pair-a,pair,A/a.png,A/b.png,,,,\n"
        "private-pair-b,pair,A/b.png,A/a.png,,,,\n"
        "private-scene,scene,scene.png,,,,,\n"
        "private-mask,mask,,,mask.png,,,\n",
        encoding="utf-8",
    )
    app = create_app(sheet, pool_roots={"A": pool})
    fail_next_save = True

    @app.before_request
    def fail_first_pair_save() -> object:
        nonlocal fail_next_save
        if (
            request.method == "POST"
            and request.path == "/api/items/1"
            and fail_next_save
        ):
            fail_next_save = False
            return jsonify(error="Synthetic pair save failure."), 409
        return None

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            requests: list[str] = []
            page.on("request", lambda request: requests.append(request.url))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.get_by_role("button", name="Near-duplicate").wait_for()
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.id") == "zoom"
            assert (
                page.locator("#zoom").evaluate(
                    "button => getComputedStyle(button).outlineStyle"
                )
                == "solid"
            )
            assert (
                page.get_by_role("button", name="Previous").get_attribute(
                    "aria-keyshortcuts"
                )
                == "["
            )
            assert (
                page.get_by_role("button", name="Next").get_attribute(
                    "aria-keyshortcuts"
                )
                == "]"
            )
            nav_bottom = page.locator("nav").evaluate(
                "navigation => navigation.getBoundingClientRect().bottom"
            )
            assert nav_bottom <= 900
            page.wait_for_function(
                "[...document.querySelectorAll('#pair-frames img')].length === 2 "
                "&& [...document.querySelectorAll('#pair-frames img')].every(i => i.complete && i.naturalWidth > 0)"
            )
            assert (
                page.locator("#image-frame").evaluate(
                    "frame => getComputedStyle(frame).display"
                )
                == "none"
            )
            panel_sizes = page.locator(".pair-image-frame").evaluate_all(
                "frames => frames.map(frame => [Math.round(frame.getBoundingClientRect().width), "
                "Math.round(frame.getBoundingClientRect().height)])"
            )
            assert all(
                abs(left - right) <= 1
                for left, right in zip(panel_sizes[0], panel_sizes[1], strict=True)
            )
            preview_sizes = page.locator(".pair-image-frame").evaluate_all(
                "frames => frames.map(frame => [Math.round(frame.querySelector('img').getBoundingClientRect().width), "
                "Math.round(frame.querySelector('img').getBoundingClientRect().height)])"
            )
            assert all(
                image_width <= frame_width and image_height <= frame_height
                for (image_width, image_height), (frame_width, frame_height)
                in zip(preview_sizes, panel_sizes, strict=True)
            )
            preview_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => Number.parseFloat(image.style.width))"
            )
            page.locator("#scale").evaluate(
                "input => { input.value = '2'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
            )
            preview_zoom_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => Number.parseFloat(image.style.width))"
            )
            assert preview_zoom_widths == pytest.approx(
                [width * 2 for width in preview_widths]
            )
            assert page.locator("#pair-frames").evaluate(
                "container => [...container.querySelectorAll('.pair-image-frame')].every(frame => "
                "frame.getBoundingClientRect().right <= container.getBoundingClientRect().right + 1)"
            )
            assert any("/images/1/a/preview" in request for request in requests)
            assert any("/images/1/b/preview" in request for request in requests)

            page.get_by_role("button", name="Swap image order").click()
            assert (
                page.locator(".pair-panel[data-side='a']").evaluate(
                    "panel => panel.style.order"
                )
                == "2"
            )
            assert (
                page.locator(".pair-panel[data-side='b']").evaluate(
                    "panel => panel.style.order"
                )
                == "1"
            )
            assert (
                page.get_by_role("button", name="Near-duplicate").get_attribute(
                    "aria-pressed"
                )
                == "false"
            )

            page.get_by_role("button", name="1:1 image pixels").click()
            page.get_by_role("button", name="Fit preview").wait_for()
            page.wait_for_function(
                "[...document.querySelectorAll('#pair-frames img')].every(i => i.complete && i.naturalWidth > 0)"
            )
            assert any("/images/1/a/original" in request for request in requests)
            assert any("/images/1/b/original" in request for request in requests)
            page.get_by_role("button", name="Use zoom scale").wait_for()
            actual_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => Number.parseFloat(image.style.width))"
            )
            rendered_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => image.getBoundingClientRect().width)"
            )
            assert actual_widths == [1600, 800]
            assert rendered_widths == actual_widths
            assert (
                page.get_by_role("button", name="Use zoom scale").get_attribute(
                    "aria-pressed"
                )
                == "true"
            )
            page.locator("#scale").evaluate(
                "input => { input.value = '2'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
            )
            zoom_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => Number.parseFloat(image.style.width))"
            )
            rendered_zoom_widths = page.locator("#pair-frames img").evaluate_all(
                "images => images.map(image => image.getBoundingClientRect().width)"
            )
            assert zoom_widths == [3200, 1600]
            assert rendered_zoom_widths == zoom_widths
            assert (
                page.get_by_role("button", name="1:1 image pixels").get_attribute(
                    "aria-pressed"
                )
                == "false"
            )
            pan = page.locator(".pair-image-frame").evaluate_all(
                "async frames => { const [a, b] = frames; "
                "const maxLeft = a.scrollWidth - a.clientWidth; "
                "const maxTop = a.scrollHeight - a.clientHeight; "
                "a.scrollLeft = maxLeft * 0.4; a.scrollTop = maxTop * 0.3; "
                "await new Promise(requestAnimationFrame); "
                "return [a.scrollLeft / maxLeft, b.scrollLeft / (b.scrollWidth - b.clientWidth), "
                "a.scrollTop / maxTop, b.scrollTop / (b.scrollHeight - b.clientHeight)]; }"
            )
            assert pan == pytest.approx([0.4, 0.4, 0.3, 0.3], abs=0.02)
            page.locator("#scale").evaluate(
                "input => { input.value = '3'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
            )
            retained_pan = page.locator(".pair-image-frame").evaluate_all(
                "frames => { const [a, b] = frames; return ["
                "a.scrollLeft / (a.scrollWidth - a.clientWidth), "
                "b.scrollLeft / (b.scrollWidth - b.clientWidth), "
                "a.scrollTop / (a.scrollHeight - a.clientHeight), "
                "b.scrollTop / (b.scrollHeight - b.clientHeight)]; }"
            )
            assert retained_pan == pytest.approx([0.4, 0.4, 0.3, 0.3], abs=0.02)

            page.locator("#zoom").focus()
            page.keyboard.press("2")
            page.get_by_text(
                "Couldn't save this choice: Synthetic pair save failure. Reload to check the value before making another choice.",
                exact=True,
            ).wait_for()
            assert page.get_by_role("button", name="Next").is_disabled()
            page.reload()
            page.get_by_role("button", name="Near-duplicate").wait_for()
            page.wait_for_function(
                "[...document.querySelectorAll('#pair-frames img')].every(i => i.complete && i.naturalWidth > 0)"
            )
            page.keyboard.press("2")
            page.get_by_text("Saved", exact=True).wait_for()
            assert (
                page.get_by_role("button", name="Near-duplicate").get_attribute(
                    "aria-pressed"
                )
                == "true"
            )
            assert (
                page.get_by_role("button", name="Near-duplicate")
                .locator(".shortcut")
                .evaluate("shortcut => getComputedStyle(shortcut).color")
                == "rgb(255, 255, 255)"
            )
            page.get_by_role("button", name="Undo last choice").click()
            page.get_by_text("Saved", exact=True).wait_for()
            assert (
                page.get_by_role("button", name="Near-duplicate").get_attribute(
                    "aria-pressed"
                )
                == "false"
            )
            page.evaluate("document.activeElement.blur()")
            page.keyboard.press("2")
            page.get_by_text("Saved", exact=True).wait_for()
            page.locator("#zoom").focus()
            page.keyboard.press("]")
            page.get_by_text("Item 2 of 4", exact=True).wait_for()
            assert page.evaluate("document.activeElement.id") == "zoom"
            page.wait_for_function(
                "[...document.querySelectorAll('#pair-frames img')].every(i => i.complete && i.naturalWidth > 0)"
            )
            assert (
                page.get_by_role("button", name="Swap image order").get_attribute(
                    "aria-pressed"
                )
                == "false"
            )
            assert page.get_by_role(
                "button", name="Inspect full resolution"
            ).is_visible()
            assert page.locator("#scale").input_value() == "1"
            page.get_by_role("button", name="Exact").click()
            page.get_by_text("Saved", exact=True).wait_for()
            page.get_by_role("button", name="Next").click()
            page.get_by_text("Item 3 of 4", exact=True).wait_for()
            page.get_by_role("button", name="Field").wait_for(state="visible")
            assert (
                page.locator("#pair-frames").evaluate(
                    "frames => getComputedStyle(frames).display"
                )
                == "none"
            )
            assert page.locator("#pair-frames").is_hidden()
            assert page.get_by_role("button", name="Swap image order").is_hidden()
            page.get_by_role("button", name="Field").click()
            page.get_by_text("Saved", exact=True).wait_for()
            page.get_by_role("button", name="Next").click()
            page.get_by_text("Item 4 of 4", exact=True).wait_for()
            page.get_by_role("group", name="Mask quality").wait_for()
            assert (
                page.locator("#pair-frames").evaluate(
                    "frames => getComputedStyle(frames).display"
                )
                == "none"
            )
            page.wait_for_function(
                "document.querySelector('#image-frame img')?.complete "
                "&& document.querySelector('#image-frame img')?.naturalWidth > 0"
            )
            page.get_by_role("button", name="Pass").click()
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.get_by_role("button", name="Next").is_disabled()
            page.get_by_role("button", name="Clean background").click()
            page.get_by_text("Saved", exact=True).wait_for()
            page.get_by_role("button", name="Finish sitting").click()
            page.get_by_role("button", name="Confirm finish sitting").wait_for()
            assert page.get_by_text(
                "Finishing makes this sitting read-only. Choose “Confirm finish sitting” to continue.",
                exact=True,
            ).is_visible()
            assert page.get_by_text("Read-only", exact=True).count() == 0
            page.get_by_role("button", name="Field").click()
            page.get_by_text("Saved", exact=True).wait_for()
            assert page.get_by_role("button", name="Confirm finish sitting").is_hidden()

            finish = page.get_by_role("button", name="Finish sitting")
            finish.dblclick()
            page.get_by_role("button", name="Confirm finish sitting").wait_for()
            assert page.evaluate("document.activeElement.id") == "finish"
            assert page.request.get(
                f"http://127.0.0.1:{server.server_port}/api/sitting"
            ).json()["read_only"] is False
            page.keyboard.press("Enter")
            assert page.request.get(
                f"http://127.0.0.1:{server.server_port}/api/sitting"
            ).json()["read_only"] is False
            page.keyboard.press("Escape")
            assert page.get_by_role("button", name="Confirm finish sitting").is_hidden()
            assert page.request.get(
                f"http://127.0.0.1:{server.server_port}/api/sitting"
            ).json()["read_only"] is False

            finish.click()
            page.get_by_role("button", name="Confirm finish sitting").wait_for()
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.id") == "confirm-finish"
            page.keyboard.press("Enter")
            page.get_by_text("Sitting finished", exact=False).wait_for()
            page.reload()
            page.get_by_text("Read-only", exact=True).wait_for()
            assert page.get_by_role("button", name="Near-duplicate").is_disabled()
            assert "private-pair" not in page.content()
            browser.close()
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.serial
def test_browser_pair_layout_uses_desktop_space_and_holds_at_1000x700(
    tmp_path: Path,
) -> None:
    from threading import Thread

    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    pool = tmp_path / "pool"
    pool.mkdir()
    Image.new("RGB", (1200, 800), "#7799aa").save(pool / "a.png")
    Image.new("RGB", (800, 1200), "#aa9977").save(pool / "b.png")
    (tmp_path / "kit.json").write_text(
        json.dumps(
            {"items": [{"item_id": "synthetic-pair", "kind": "pair", "pool": "A"}]}
        ),
        encoding="utf-8",
    )
    sheet = tmp_path / "sitting-001.csv"
    sheet.write_text(
        "item_id,kind,image_a,image_b,overlay_path,rating,mask_quality,scene_tag\n"
        "synthetic-pair,pair,A/a.png,A/b.png,,,,\n",
        encoding="utf-8",
    )
    server = make_server(
        "127.0.0.1", 0, create_app(sheet, pool_roots={"A": pool}), threaded=True
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function(
                "[...document.querySelectorAll('.pair-image-frame img')].length === 2 "
                "&& [...document.querySelectorAll('.pair-image-frame img')].every(i => i.complete && i.naturalWidth > 0)"
            )

            def layout() -> list[list[int]]:
                return page.locator(".pair-image-frame").evaluate_all(
                    "frames => frames.map(frame => [Math.round(frame.clientWidth), Math.round(frame.clientHeight)])"
                )

            sizes_1440 = layout()
            assert all(width >= 400 and height >= 500 for width, height in sizes_1440)
            assert abs(sizes_1440[0][0] - sizes_1440[1][0]) <= 1
            assert abs(sizes_1440[0][1] - sizes_1440[1][1]) <= 1
            assert page.locator(".workspace").evaluate(
                "workspace => getComputedStyle(workspace).gridTemplateColumns.split(' ').length"
            ) == 2

            page.set_viewport_size({"width": 1920, "height": 1080})
            sizes_1920 = layout()
            assert all(width >= 540 and height >= 600 for width, height in sizes_1920)
            assert page.locator("nav").evaluate(
                "navigation => getComputedStyle(navigation).position"
            ) == "sticky"

            page.set_viewport_size({"width": 1000, "height": 700})
            sizes_1000 = layout()
            assert all(width >= 280 and height >= 300 for width, height in sizes_1000)
            assert page.locator(".workspace").evaluate(
                "workspace => getComputedStyle(workspace).gridTemplateColumns.split(' ').length"
            ) == 2
            buttons = page.locator("[aria-label='Rating choices'] .rating-choice")
            assert buttons.count() == 4
            navigation = page.locator("nav").bounding_box()
            assert navigation is not None
            for index in range(buttons.count()):
                button = buttons.nth(index)
                box = button.bounding_box()
                assert box is not None and box["y"] + box["height"] <= navigation["y"]
            assert page.locator("#finish").is_visible()
            assert page.locator("nav").evaluate(
                "navigation => navigation.getBoundingClientRect().bottom <= 700"
            )
            browser.close()
    finally:
        server.shutdown()
        thread.join()

"""The rebuild kit reproduces the full manifest identity, not only the split map."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jute_disease.data.manifest import load_complete_manifest

SCRIPT = Path(__file__).parents[2] / "scripts" / "rebuild_splits.py"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _rebuild(kit: Path, images: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--kit-manifest",
            str(kit),
            "--image-root",
            str(images),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("kit-identity") / "fixture"
    result = _cli("manuscript", "fixture", "--output", str(output))
    assert result.returncode == 0, result.stderr
    return output


def _tampered_kit(fixture_run: Path, tmp_path: Path, change) -> Path:
    kit = json.loads(
        (fixture_run / "export" / "rebuild-kit.json").read_text(encoding="utf-8")
    )
    change(kit)
    path = tmp_path / "tampered-kit.json"
    path.write_text(json.dumps(kit), encoding="utf-8")
    return path


def test_rebuild_reproduces_the_complete_manifest_bytes(
    fixture_run: Path, tmp_path: Path
) -> None:
    complete = load_complete_manifest(fixture_run / "freeze" / "target")
    expected_bytes = (json.dumps(complete.to_json(), indent=1) + "\n").encode("utf-8")

    result = _rebuild(
        fixture_run / "export" / "rebuild-kit.json",
        fixture_run / "images",
        tmp_path / "rebuilt.json",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["manifest_digest"] == complete.digest()
    assert report["manifest_sha256"] == hashlib.sha256(expected_bytes).hexdigest()


def test_kit_records_the_manifest_it_rebuilds(fixture_run: Path) -> None:
    kit = json.loads(
        (fixture_run / "export" / "rebuild-kit.json").read_text(encoding="utf-8")
    )
    complete = load_complete_manifest(fixture_run / "freeze" / "target")

    assert kit["version"] == 2
    assert kit["manifest_digest"] == complete.digest()
    assert kit["manifest"] == complete.to_json()


def test_rebuild_rejects_a_changed_manifest_field(
    fixture_run: Path, tmp_path: Path
) -> None:
    def change(kit: dict) -> None:
        kit["manifest"]["dependencies"]["tampered"] = "x"

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "does not match its recorded digest" in result.stderr
    assert "Traceback" not in result.stderr


def test_rebuild_rejects_an_inconsistent_manifest_entry(
    fixture_run: Path, tmp_path: Path
) -> None:
    def change(kit: dict) -> None:
        kit["manifest"]["entries"][0]["width"] += 1

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "manifest is malformed" in result.stderr


def test_rebuild_rejects_a_changed_recorded_digest(
    fixture_run: Path, tmp_path: Path
) -> None:
    def change(kit: dict) -> None:
        kit["manifest_digest"] = "0" * 64

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "recorded digest" in result.stderr


def test_rebuild_rejects_an_excluded_record_dropped_from_the_manifest(
    fixture_run: Path, tmp_path: Path
) -> None:
    def change(kit: dict) -> None:
        kit["manifest"]["excluded"] = [{"image_id": "x", "reason": "added"}]

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "recorded digest" in result.stderr


def test_rebuild_rejects_a_kit_without_the_manifest(
    fixture_run: Path, tmp_path: Path
) -> None:
    def change(kit: dict) -> None:
        for key in ("manifest", "manifest_digest", "manifest_sha256"):
            kit.pop(key)

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "no complete manifest identity" in result.stderr


def test_rebuild_rejects_a_version_1_kit(fixture_run: Path, tmp_path: Path) -> None:
    def change(kit: dict) -> None:
        kit["version"] = 1

    kit = _tampered_kit(fixture_run, tmp_path, change)

    result = _rebuild(kit, fixture_run / "images", tmp_path / "out.json")

    assert result.returncode != 0
    assert "version 1 records only the split map" in result.stderr


def test_check_rejects_a_kit_that_differs_from_the_manifest(
    fixture_run: Path, tmp_path: Path
) -> None:
    export = fixture_run / "export-tampered"
    shutil.copytree(fixture_run / "export", export)
    kit_path = export / "rebuild-kit.json"
    kit = json.loads(kit_path.read_text(encoding="utf-8"))
    kit["manifest_sha256"] = "1" * 64
    kit_path.write_text(json.dumps(kit), encoding="utf-8")

    try:
        result = _cli("manuscript", "check", str(export))
    finally:
        shutil.rmtree(export)

    assert result.returncode != 0
    assert "rebuild kit" in result.stderr

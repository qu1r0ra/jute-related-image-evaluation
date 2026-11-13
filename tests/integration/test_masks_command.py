"""The mask-audit commands, run as a user would, against a synthetic freeze."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import make_synthetic_data

pytestmark = pytest.mark.slow

SCENES = ("clean-background", "field", "unknown")


def _image(path: Path, scene: str, seed: int) -> None:
    pixels = np.full((32, 32, 3), 245, dtype=np.uint8)
    if scene == "field":
        for y in range(32):
            for x in range(32):
                value = 30 + ((x * 31 + y * 47 + seed * 13) % 190)
                pixels[y, x] = (value, value // 2, value // 3)
    if scene == "unknown":
        pixels[3:29, 3:29] = (0, 180, 0)
    else:
        pixels[6:22, 6:22] = (0, 180, 0)
    pixels[0, 0] = (245, 245, seed % 256)
    pixels[0, 1] = (245, 245, (seed // 256) % 256)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _write_records(
    directory: Path, name: str, scenes: list[str], *, tagged: bool, start: int
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for offset, scene in enumerate(scenes):
        index = start + offset
        file_name = f"{name}-{index:04d}.png"
        _image(directory / file_name, scene, index)
        record = {"image_id": f"{name}-{index:04d}", "path": file_name}
        if tagged:
            record["scene_tag"] = scene
        records.append(record)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _run(argv: list[str]) -> SystemExit:
    with pytest.raises(SystemExit) as raised:
        main(argv)
    return raised.value


@pytest.fixture(scope="module")
def freeze(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("masks-command")
    images = base / "images"
    make_synthetic_data(images, per_class=20, quarantine_count=4)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (base / "reviews.json").write_text(
        json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "freeze",
            "target",
            "--image-root",
            str(images),
            "--rules",
            str(base / "rules.json"),
            "--reviews",
            str(base / "reviews.json"),
            "--output",
            str(base / "freeze"),
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    target = base / "freeze" / "target"
    entries = json.loads(
        (target / "protected-test" / "manifest.json").read_text(encoding="utf-8")
    )["entries"]
    return {"base": base, "images": images, "target": target, "test": entries}


@pytest.fixture(scope="module")
def calibrated(freeze: dict[str, Path], tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("masks-calibrated")
    scenes = [SCENES[index % 3] for index in range(100)]
    development = _write_records(work, "dev", scenes, tagged=True, start=0)
    thresholds = work / "thresholds.json"
    main(
        [
            "data",
            "masks-calibrate",
            str(development),
            "--output",
            str(thresholds),
            "--target-dir",
            str(freeze["target"]),
        ]
    )
    return {"work": work, "thresholds": thresholds}


def _audit(
    freeze: dict[str, Path],
    calibrated: dict[str, Path],
    inventory: Path,
    output: Path,
    *extra: str,
) -> None:
    main(
        [
            "data",
            "masks-audit",
            str(inventory),
            str(calibrated["thresholds"]),
            "--output",
            str(output),
            "--target-dir",
            str(freeze["target"]),
            "--sample-count",
            "12",
            "--owner-count",
            "6",
            *extra,
        ]
    )


def _inventory(work: Path, name: str) -> Path:
    scenes = ["clean-background"] * 10 + ["field"] * 6 + ["unknown"] * 4
    return _write_records(work, name, scenes, tagged=False, start=1000)


def _rate_all_pass(audit: Path) -> tuple[Path, Path, Path]:
    manifest = json.loads((audit / "audit.json").read_text(encoding="utf-8"))
    with (audit / "agent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        for row in manifest["records"]:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "overlay_path": row["overlay_path"],
                    "mask_quality": "pass",
                }
            )
    by_id = {row["audit_id"]: row for row in manifest["records"]}
    with (audit / "owner.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audit_id", "overlay_path", "mask_quality", "scene_tag"),
        )
        writer.writeheader()
        for audit_id in manifest["owner_audit_ids"]:
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "overlay_path": by_id[audit_id]["overlay_path"],
                    "mask_quality": "pass",
                    "scene_tag": by_id[audit_id]["predicted_scene"],
                }
            )
    return audit / "audit.json", audit / "agent.csv", audit / "owner.csv"


def test_checkpoint_option_is_removed_from_both_mask_commands(
    freeze: dict[str, Path], calibrated: dict[str, Path], tmp_path: Path
) -> None:
    inventory = _inventory(tmp_path, "inventory")
    audit = _run(
        [
            "data",
            "masks-audit",
            str(inventory),
            str(calibrated["thresholds"]),
            "--output",
            str(tmp_path / "audit"),
            "--target-dir",
            str(freeze["target"]),
            "--checkpoint",
            str(tmp_path),
        ]
    )
    calibrate = _run(
        [
            "data",
            "masks-calibrate",
            str(calibrated["work"] / "dev.json"),
            "--output",
            str(tmp_path / "t.json"),
            "--target-dir",
            str(freeze["target"]),
            "--checkpoint",
            str(tmp_path),
        ]
    )
    assert audit.code == 2
    assert calibrate.code == 2
    assert not (tmp_path / "audit").exists()


def test_calibration_refuses_a_development_set_that_overlaps_the_test_partition(
    freeze: dict[str, Path], tmp_path: Path
) -> None:
    scenes = [SCENES[index % 3] for index in range(100)]
    development = _write_records(tmp_path, "dev", scenes, tagged=True, start=0)
    records = json.loads(development.read_text(encoding="utf-8"))
    leaked = freeze["test"][0]
    shutil.copy(freeze["images"] / leaked["path"], tmp_path / "leaked.png")
    records[0]["path"] = "leaked.png"
    development.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "thresholds.json"

    refused = _run(
        [
            "data",
            "masks-calibrate",
            str(development),
            "--output",
            str(output),
            "--target-dir",
            str(freeze["target"]),
        ]
    )

    assert "test partition" in str(refused.code)
    assert not output.exists()


def test_audit_refuses_an_inventory_that_overlaps_the_test_partition(
    freeze: dict[str, Path], calibrated: dict[str, Path], tmp_path: Path
) -> None:
    inventory = _inventory(tmp_path, "inventory")
    records = json.loads(inventory.read_text(encoding="utf-8"))
    leaked = freeze["test"][1]
    shutil.copy(freeze["images"] / leaked["path"], tmp_path / "leaked.png")
    records[0]["path"] = "leaked.png"
    inventory.write_text(json.dumps(records), encoding="utf-8")

    refused = _run(
        [
            "data",
            "masks-audit",
            str(inventory),
            str(calibrated["thresholds"]),
            "--output",
            str(tmp_path / "audit"),
            "--target-dir",
            str(freeze["target"]),
            "--sample-count",
            "12",
            "--owner-count",
            "6",
        ]
    )

    assert "test partition" in str(refused.code)
    assert not (tmp_path / "audit").exists()


def test_audit_refuses_thresholds_calibrated_on_the_test_partition(
    freeze: dict[str, Path], calibrated: dict[str, Path], tmp_path: Path
) -> None:
    thresholds = json.loads(calibrated["thresholds"].read_text(encoding="utf-8"))
    thresholds["development_sources"][0]["sha256"] = freeze["test"][0]["image_id"]
    forged = tmp_path / "thresholds.json"
    forged.write_text(json.dumps(thresholds), encoding="utf-8")
    inventory = _inventory(tmp_path, "inventory")

    refused = _run(
        [
            "data",
            "masks-audit",
            str(inventory),
            str(forged),
            "--output",
            str(tmp_path / "audit"),
            "--target-dir",
            str(freeze["target"]),
            "--sample-count",
            "12",
            "--owner-count",
            "6",
        ]
    )

    assert "test partition" in str(refused.code)
    assert not (tmp_path / "audit").exists()


def test_audit_refuses_thresholds_from_a_non_colour_threshold_backend(
    freeze: dict[str, Path], calibrated: dict[str, Path], tmp_path: Path
) -> None:
    thresholds = json.loads(calibrated["thresholds"].read_text(encoding="utf-8"))
    thresholds["mask_model"] = "excess-green-otsu-largest-component+Sam2Model"
    thresholds["mask_version"] = "5.22.1+abcdef123456"
    refined = tmp_path / "thresholds.json"
    refined.write_text(json.dumps(thresholds), encoding="utf-8")
    inventory = _inventory(tmp_path, "inventory")

    refused = _run(
        [
            "data",
            "masks-audit",
            str(inventory),
            str(refined),
            "--output",
            str(tmp_path / "audit"),
            "--target-dir",
            str(freeze["target"]),
        ]
    )

    assert "colour-threshold" in str(refused.code)
    assert not (tmp_path / "audit").exists()


def test_evaluation_refuses_an_audit_recorded_with_another_backend(
    freeze: dict[str, Path], calibrated: dict[str, Path], tmp_path: Path
) -> None:
    inventory = _inventory(tmp_path, "inventory")
    _audit(freeze, calibrated, inventory, tmp_path / "audit")
    manifest_path, agent, owner = _rate_all_pass(tmp_path / "audit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mask_backend"] = {
        "model": "excess-green-otsu-largest-component+Sam2Model",
        "version": "5.22.1+abcdef123456",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    refused = _run(
        [
            "data",
            "masks-evaluate",
            str(manifest_path),
            str(agent),
            str(owner),
            "--agent-name",
            "rater",
            "--agent-version",
            "1",
        ]
    )

    assert "colour-threshold" in str(refused.code)


def test_a_zero_point_estimate_does_not_pass_when_the_exact_bound_fails(
    freeze: dict[str, Path],
    calibrated: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory = _inventory(tmp_path, "inventory")
    _audit(freeze, calibrated, inventory, tmp_path / "audit")
    capsys.readouterr()
    manifest, agent, owner = _rate_all_pass(tmp_path / "audit")

    main(
        [
            "data",
            "masks-evaluate",
            str(manifest),
            str(agent),
            str(owner),
            "--agent-name",
            "rater",
            "--agent-version",
            "1",
        ]
    )

    fail_rate = json.loads(capsys.readouterr().out)["mask_quality"]["fail_rate"]
    assert fail_rate["estimate"] == 0
    assert fail_rate["upper_95"] > 0.10
    assert fail_rate["gate_pass"] is False
    assert "Clopper-Pearson" in fail_rate["method"]


def test_exact_bound_within_the_threshold_passes_the_fail_rate_gate(
    freeze: dict[str, Path],
    calibrated: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenes = ["clean-background"] * 300 + ["field"] * 3 + ["unknown"] * 3
    inventory = _write_records(tmp_path, "large", scenes, tagged=False, start=2000)
    _audit(
        freeze,
        calibrated,
        inventory,
        tmp_path / "audit",
        "--sample-count",
        "100",
        "--owner-count",
        "20",
    )
    capsys.readouterr()
    manifest, agent, owner = _rate_all_pass(tmp_path / "audit")

    main(
        [
            "data",
            "masks-evaluate",
            str(manifest),
            str(agent),
            str(owner),
            "--agent-name",
            "rater",
            "--agent-version",
            "1",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    fail_rate = report["mask_quality"]["fail_rate"]
    assert 0 <= fail_rate["upper_95"] <= report["mask_quality"]["fail_rate_limit"]
    assert fail_rate["gate_pass"] is True


def test_evaluation_records_the_backend_and_a_digest_for_the_freeze_to_bind(
    freeze: dict[str, Path],
    calibrated: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory = _inventory(tmp_path, "inventory")
    _audit(freeze, calibrated, inventory, tmp_path / "audit")
    manifest, agent, owner = _rate_all_pass(tmp_path / "audit")
    capsys.readouterr()
    argv = [
        "data",
        "masks-evaluate",
        str(manifest),
        str(agent),
        str(owner),
        "--agent-name",
        "rater",
        "--agent-version",
        "1",
    ]

    main(argv)
    first = json.loads(capsys.readouterr().out)
    main(argv)
    second = json.loads(capsys.readouterr().out)

    assert first["mask_backend"] == {
        "model": "excess-green-otsu-largest-component",
        "version": "1",
    }
    assert len(first["digest"]) == 64
    assert first["digest"] == second["digest"]
    audit_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert audit_manifest["partition_check"]["target_manifest_digest"]
    assert first["audit_manifest_digest"] != first["digest"]

    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["seed"] = 7
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    main(argv)
    assert json.loads(capsys.readouterr().out)["digest"] != first["digest"]


def test_fixture_still_runs_end_to_end_on_the_colour_threshold_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "masks-fixture", "--output", str(tmp_path / "fixture")])

    report = json.loads(capsys.readouterr().out)
    assert report["fixture_mode"] is True
    assert report["mask_backend"]["model"] == "excess-green-otsu-largest-component"
    assert report["mask_quality"]["fail_rate"]["gate_pass"] is True

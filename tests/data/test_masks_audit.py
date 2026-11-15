import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.cli import main
from jute_disease.data.masks import SCENE_TAGS, ColorThresholdBackend, SceneThresholds
from jute_disease.data.masks_audit import (
    CandidateImage,
    ProtectedTestImages,
    evaluate_masks_audit,
    fit_thresholds_from_manifest,
    read_development_observations,
    run_masks_audit,
)

PROTECTED = ProtectedTestImages(
    ("b" * 64,),
    frozenset(
        hashlib.sha256(f"test-{index}".encode()).hexdigest() for index in range(3)
    ),
)


def _thresholds(count: int = 100) -> SceneThresholds:
    ids = tuple(f"dev-{index}" for index in range(count))
    sources = tuple(
        (image_id, hashlib.sha256(f"source-{index}".encode()).hexdigest())
        for index, image_id in enumerate(ids)
    )
    return SceneThresholds(
        background_fraction_min=0.5,
        color_uniformity_min=0.9,
        development_ids=ids,
        development_sources=sources,
        development_digest="a" * 64,
        score=1.0,
        mask_model=ColorThresholdBackend.model,
        mask_version=ColorThresholdBackend.version,
    )


def _image(path: Path, scene: str, index: int) -> None:
    pixels = np.full((8, 8, 3), 245, dtype=np.uint8)
    if scene == "field":
        for y in range(8):
            pixels[y, :, :] = 30 + y * 25, 60 + y * 17, 10 + y * 9
    if scene == "unknown":
        pixels[1:8, 1:8] = (0, 180, 0)
    else:
        pixels[2:6, 2:6] = (0, 180, 0)
    pixels[0, 0] = (245, 245, index % 256)
    pixels[0, 1] = (245, 245, index // 256)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _candidates(tmp_path: Path) -> list[CandidateImage]:
    result = []
    counts = {"clean-background": 10, "field": 6, "unknown": 4}
    index = 0
    for tag, count in counts.items():
        for _ in range(count):
            path = tmp_path / f"candidate-{index:02d}.png"
            _image(path, tag, index)
            result.append(CandidateImage(f"candidate-{index:02d}", path))
            index += 1
    return result


def _fill_ratings(manifest: dict, out: Path) -> None:
    with (out / "agent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        for index, row in enumerate(manifest["records"]):
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "overlay_path": row["overlay_path"],
                    "mask_quality": ("pass", "minor", "fail")[index % 3],
                }
            )
    with (out / "owner.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audit_id", "overlay_path", "mask_quality", "scene_tag"),
        )
        writer.writeheader()
        for index, audit_id in enumerate(manifest["owner_audit_ids"]):
            row = next(
                row for row in manifest["records"] if row["audit_id"] == audit_id
            )
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "overlay_path": row["overlay_path"],
                    "mask_quality": ("pass", "minor", "fail")[index % 3],
                    "scene_tag": row["predicted_scene"],
                }
            )


def test_stratified_audit_oversamples_rare_scenes_and_blinds_sheets(
    tmp_path: Path,
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(),
        ColorThresholdBackend(),
        tmp_path / "audit",
        PROTECTED,
        sample_count=12,
        owner_count=6,
        seed=10,
    )

    assert manifest["partition_check"]["overlaps"] == 0
    assert manifest["partition_check"]["target_manifest_digests"] == ["b" * 64]
    assert manifest["mask_backend"] == {
        "model": ColorThresholdBackend.model,
        "version": ColorThresholdBackend.version,
    }
    assert manifest["sampling"]["population_by_scene"] == {
        "clean-background": 10,
        "field": 6,
        "unknown": 4,
    }
    assert manifest["sampling"]["sampled_by_scene"] == {
        "clean-background": 4,
        "field": 4,
        "unknown": 4,
    }
    id_scene_order = [row["predicted_scene"] for row in manifest["records"]]
    assert (
        sum(
            id_scene_order[index] != id_scene_order[index - 1]
            for index in range(1, len(id_scene_order))
        )
        > len(SCENE_TAGS) - 1
    )
    owner_columns = next(csv.reader((tmp_path / "audit" / "owner.csv").open()))
    agent_columns = next(csv.reader((tmp_path / "audit" / "agent.csv").open()))
    assert owner_columns == ["audit_id", "overlay_path", "mask_quality", "scene_tag"]
    assert agent_columns == ["audit_id", "overlay_path", "mask_quality"]
    metadata = json.loads(
        (tmp_path / "audit" / manifest["records"][0]["metadata_path"]).read_text()
    )
    assert metadata["model"] == ColorThresholdBackend.model
    assert metadata["model_version"] == ColorThresholdBackend.version
    assert metadata["rubric"]["version"] == "leaf-mask-v1"

    _fill_ratings(manifest, tmp_path / "audit")
    report = evaluate_masks_audit(
        manifest,
        tmp_path / "audit" / "agent.csv",
        tmp_path / "audit" / "owner.csv",
        resamples=200,
        agent_rater={"name": "fixture-agent", "version": "test"},
    )
    confusion = report["scene_tag"]["confusion_matrix"]
    assert confusion["labels"] == list(SCENE_TAGS)
    assert len(confusion["counts"]) == 3
    assert all(len(row) == 3 for row in confusion["counts"])
    assert sum(map(sum, confusion["counts"])) == 6
    assert report["mask_quality"]["fail_rate"]["population"] == 20
    assert len(report["digest"]) == 64
    assert report["mask_backend"] == manifest["mask_backend"]


@pytest.mark.parametrize("rating", ["", "invalid"])
def test_evaluation_rejects_invalid_unpaired_agent_rating(
    tmp_path: Path, rating: str
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(),
        ColorThresholdBackend(),
        tmp_path / "audit",
        PROTECTED,
        sample_count=12,
        owner_count=6,
        seed=10,
    )
    _fill_ratings(manifest, tmp_path / "audit")
    unpaired_id = next(
        row["audit_id"]
        for row in manifest["records"]
        if row["audit_id"] not in manifest["owner_audit_ids"]
    )
    agent_path = tmp_path / "audit" / "agent.csv"
    with agent_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["audit_id"] == unpaired_id:
            row["mask_quality"] = rating
    with agent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        ValueError, match=f"invalid mask_quality rating for {unpaired_id}"
    ):
        evaluate_masks_audit(
            manifest,
            agent_path,
            tmp_path / "audit" / "owner.csv",
            resamples=200,
            agent_rater={"name": "fixture-agent", "version": "test"},
        )


def test_audit_sample_includes_each_nonempty_stratum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every nonempty scene stratum"):
        run_masks_audit(
            _candidates(tmp_path),
            _thresholds(),
            ColorThresholdBackend(),
            tmp_path / "audit",
            PROTECTED,
            sample_count=2,
            owner_count=1,
        )


def test_audit_rejects_development_image_content_overlap(tmp_path: Path) -> None:
    path = tmp_path / "same.png"
    _image(path, "clean-background", 1)
    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    thresholds = _thresholds()
    sources = list(thresholds.development_sources)
    sources[0] = (sources[0][0], image_hash)
    thresholds = SceneThresholds(
        background_fraction_min=thresholds.background_fraction_min,
        color_uniformity_min=thresholds.color_uniformity_min,
        development_ids=thresholds.development_ids,
        development_sources=tuple(sources),
        development_digest=thresholds.development_digest,
        score=thresholds.score,
        mask_model=thresholds.mask_model,
        mask_version=thresholds.mask_version,
    )
    candidates = [CandidateImage(f"candidate-{index}", path) for index in range(3)]

    with pytest.raises(ValueError, match="development image content"):
        run_masks_audit(
            candidates,
            thresholds,
            ColorThresholdBackend(),
            tmp_path / "audit",
            PROTECTED,
            sample_count=3,
            owner_count=1,
        )


def test_audit_refuses_inventory_overlapping_the_test_partition(
    tmp_path: Path,
) -> None:
    candidates = _candidates(tmp_path)
    leaked = hashlib.sha256(candidates[0].path.read_bytes()).hexdigest()
    protected = ProtectedTestImages(("b" * 64,), frozenset({leaked}))

    with pytest.raises(ValueError, match="test partition"):
        run_masks_audit(
            candidates,
            _thresholds(),
            ColorThresholdBackend(),
            tmp_path / "audit",
            protected,
            sample_count=12,
            owner_count=6,
        )


def test_audit_refuses_development_set_overlapping_the_test_partition(
    tmp_path: Path,
) -> None:
    thresholds = _thresholds()
    protected = ProtectedTestImages(
        ("b" * 64,), frozenset({thresholds.development_sources[7][1]})
    )

    with pytest.raises(ValueError, match="calibration development set.*test partition"):
        run_masks_audit(
            _candidates(tmp_path),
            thresholds,
            ColorThresholdBackend(),
            tmp_path / "audit",
            protected,
            sample_count=12,
            owner_count=6,
        )


@pytest.mark.parametrize("model", ["fixture-green-pixel-mask-stub", "sam2-refiner"])
def test_audit_refuses_a_backend_other_than_colour_threshold(
    tmp_path: Path, model: str
) -> None:
    class Other:
        version = "1"

        def predict(self, image: Image.Image) -> None:
            raise AssertionError("an unaudited backend must never run")

    Other.model = model
    with pytest.raises(ValueError, match="colour-threshold"):
        run_masks_audit(
            _candidates(tmp_path),
            _thresholds(),
            Other(),
            tmp_path / "audit",
            PROTECTED,
            sample_count=12,
            owner_count=6,
        )


def _rate(manifest: dict, tmp_path: Path, fails: int) -> dict:
    """Evaluate an audit in which exactly ``fails`` sampled masks are failures."""
    out = tmp_path / "audit"
    _fill_ratings(manifest, out)
    with (out / "agent.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        row["mask_quality"] = "fail" if index < fails else "pass"
    with (out / "agent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("audit_id", "overlay_path", "mask_quality")
        )
        writer.writeheader()
        writer.writerows(rows)
    return evaluate_masks_audit(
        manifest,
        out / "agent.csv",
        out / "owner.csv",
        resamples=200,
        agent_rater={"name": "fixture-agent", "version": "test"},
    )["mask_quality"]["fail_rate"]


def test_fail_rate_gate_uses_the_exact_upper_bound_not_the_point_estimate(
    tmp_path: Path,
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(),
        ColorThresholdBackend(),
        tmp_path / "audit",
        PROTECTED,
        sample_count=12,
        owner_count=6,
        seed=10,
    )

    clean = _rate(manifest, tmp_path, fails=0)
    assert clean["estimate"] == 0.0
    assert clean["upper_95"] > clean["estimate"]
    # A clean small sample still cannot certify a 10% limit.
    assert clean["gate_pass"] is False

    failing = _rate(manifest, tmp_path, fails=2)
    assert failing["estimate"] < failing["upper_95"] <= 1.0
    assert failing["gate_pass"] is False
    assert "Clopper-Pearson" in failing["method"]


def test_fixture_command_reports_digest_and_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "masks-fixture", "--output", str(tmp_path / "fixture")])
    report = json.loads(capsys.readouterr().out)
    assert report["fixture_mode"] is True
    assert report["evidence_status"] == "synthetic fixture only; not an audit result"
    assert report["mask_backend"]["model"] == ColorThresholdBackend.model
    assert len(report["digest"]) == 64
    assert report["rating_counts"]["owner_scene"]


def _two_targets(tmp_path: Path) -> tuple[Path, Path]:
    from .frozen_targets import singles, write_target

    targets = []
    for pool in ("A", "B"):
        members = [
            *singles(pool, "train", 2),
            *singles(pool, "val", 1),
            *singles(pool, "test", 2),
        ]
        targets.append(write_target(tmp_path / pool, pool, members))
    return targets[0], targets[1]


def test_protected_test_images_union_several_targets_and_name_each_digest(
    tmp_path: Path,
) -> None:
    from jute_disease.data import load_protected_test_images

    from .frozen_targets import content_id

    first, second = _two_targets(tmp_path)

    one = load_protected_test_images(first)
    both = load_protected_test_images(first, second)

    assert len(one.manifest_digests) == 1 and len(one.image_ids) == 2
    assert len(set(both.manifest_digests)) == 2
    assert both.manifest_digests[0] == one.manifest_digests[0]
    assert both.image_ids == {
        content_id(f"{pool}-test-{index}") for pool in "AB" for index in range(2)
    }
    with pytest.raises(ValueError, match="at least one"):
        load_protected_test_images()


def test_audit_refuses_an_overlap_with_any_named_target_and_records_every_digest(
    tmp_path: Path,
) -> None:
    candidates = _candidates(tmp_path)
    leaked = hashlib.sha256(candidates[0].path.read_bytes()).hexdigest()
    clean = ProtectedTestImages(("b" * 64, "c" * 64), PROTECTED.image_ids)
    second_pool_leak = ProtectedTestImages(
        ("b" * 64, "c" * 64), PROTECTED.image_ids | {leaked}
    )

    with pytest.raises(ValueError, match="test partition"):
        run_masks_audit(
            candidates,
            _thresholds(),
            ColorThresholdBackend(),
            tmp_path / "refused",
            second_pool_leak,
            sample_count=12,
            owner_count=6,
        )
    manifest = run_masks_audit(
        candidates,
        _thresholds(),
        ColorThresholdBackend(),
        tmp_path / "audit",
        clean,
        sample_count=12,
        owner_count=6,
    )

    assert manifest["partition_check"]["target_manifest_digests"] == [
        "b" * 64,
        "c" * 64,
    ]
    assert manifest["scene_rubric"]["version"] == "scene-tag-v2"


def test_mask_commands_accept_several_target_dirs(tmp_path: Path) -> None:
    from .frozen_targets import content_id

    first, second = _two_targets(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    records = []
    for index in range(100):
        scene = SCENE_TAGS[index % 3]
        _image(work / f"dev-{index}.png", scene, index)
        records.append(
            {"image_id": f"dev-{index}", "path": f"dev-{index}.png", "scene_tag": scene}
        )
    development = work / "development.json"
    development.write_text(json.dumps(records), encoding="utf-8")
    targets = ["--target-dir", str(first), "--target-dir", str(second)]

    main(
        [
            "data",
            "masks-calibrate",
            str(development),
            "--output",
            str(work / "thresholds.json"),
            *targets,
        ]
    )
    inventory = work / "inventory.json"
    scenes = ["clean-background"] * 10 + ["field"] * 6 + ["unknown"] * 4
    rows = []
    for index, scene in enumerate(scenes, start=1000):
        _image(work / f"inv-{index}.png", scene, index)
        rows.append({"image_id": f"inv-{index}", "path": f"inv-{index}.png"})
    inventory.write_text(json.dumps(rows), encoding="utf-8")
    audit = [
        "data",
        "masks-audit",
        str(inventory),
        str(work / "thresholds.json"),
        "--sample-count",
        "12",
        "--owner-count",
        "6",
    ]
    main([*audit, "--output", str(work / "audit"), *targets])
    main([*audit, "--output", str(work / "single"), "--target-dir", str(first)])

    check = json.loads((work / "audit" / "audit.json").read_text(encoding="utf-8"))
    assert len(check["partition_check"]["target_manifest_digests"]) == 2
    assert check["partition_check"]["test_image_count"] == 4

    # An inventory image whose content is a test image of the second target only.
    leak = work / "leak.png"
    _image(leak, "field", 4242)
    leaked_id = hashlib.sha256(leak.read_bytes()).hexdigest()
    manifest_path = second / "protected-test" / "manifest.json"
    assert content_id("B-test-0") in manifest_path.read_text(encoding="utf-8")
    from .frozen_targets import singles, write_target

    leaking = write_target(
        tmp_path / "leaking",
        "B",
        [*singles("B", "train", 2), *singles("B", "val", 1), *singles("B", "test", 2)],
    )
    path = leaking / "protected-test"
    data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    # The leaking target is rewritten by hand below.
    data["entries"][0]["image_id"] = leaked_id
    data["entries"].sort(key=lambda entry: entry["image_id"])
    from jute_disease.data.manifest import Manifest, write_manifest

    write_manifest(Manifest.from_json(data), path)
    rows[0] = {"image_id": "inv-leak", "path": "leak.png"}
    inventory.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="test partition"):
        main(
            [
                *audit,
                "--output",
                str(work / "refused"),
                "--target-dir",
                str(first),
                "--target-dir",
                str(leaking),
            ]
        )
    assert not (work / "refused").exists()

    # A development image whose content is a test image of the second target only.
    records[0] = {"image_id": "dev-leak", "path": "leak.png", "scene_tag": "field"}
    development.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(SystemExit, match="test partition"):
        main(
            [
                "data",
                "masks-calibrate",
                str(development),
                "--output",
                str(work / "refused.json"),
                "--target-dir",
                str(first),
                "--target-dir",
                str(leaking),
            ]
        )
    assert not (work / "refused.json").exists()


def _development(work: Path, count: int) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(count):
        scene = SCENE_TAGS[index % 3]
        _image(work / f"dev-{index}.png", scene, index)
        records.append(
            {"image_id": f"dev-{index}", "path": f"dev-{index}.png", "scene_tag": scene}
        )
    development = work / "development.json"
    development.write_text(json.dumps(records), encoding="utf-8")
    return development


def test_90_image_development_manifest_fits_and_the_audit_accepts_it(
    tmp_path: Path,
) -> None:
    development = _development(tmp_path / "dev", 90)

    thresholds = fit_thresholds_from_manifest(
        development, ColorThresholdBackend(), PROTECTED
    )
    manifest = run_masks_audit(
        _candidates(tmp_path),
        SceneThresholds.from_dict(thresholds.to_dict()),
        ColorThresholdBackend(),
        tmp_path / "audit",
        PROTECTED,
        sample_count=12,
        owner_count=6,
    )

    assert thresholds.to_dict()["development_count"] == 90
    assert len(thresholds.development_sources) == 90
    assert manifest["records"]


def test_audit_accepts_thresholds_recording_90_development_images(
    tmp_path: Path,
) -> None:
    manifest = run_masks_audit(
        _candidates(tmp_path),
        _thresholds(90),
        ColorThresholdBackend(),
        tmp_path / "audit",
        PROTECTED,
        sample_count=12,
        owner_count=6,
    )

    assert manifest["records"]


@pytest.mark.parametrize("count", [0, 101])
def test_development_manifest_outside_1_to_100_images_is_refused(
    tmp_path: Path, count: int
) -> None:
    development = _development(tmp_path / "dev", count)

    with pytest.raises(ValueError, match="between 1 and 100 images"):
        read_development_observations(development, ColorThresholdBackend(), PROTECTED)


def test_development_manifest_with_a_repeated_id_is_refused(tmp_path: Path) -> None:
    development = _development(tmp_path / "dev", 90)
    records = json.loads(development.read_text(encoding="utf-8"))
    records[1]["image_id"] = records[0]["image_id"]
    development.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="IDs must be unique"):
        fit_thresholds_from_manifest(development, ColorThresholdBackend(), PROTECTED)


def test_development_manifest_with_a_repeated_image_is_refused(tmp_path: Path) -> None:
    development = _development(tmp_path / "dev", 90)
    records = json.loads(development.read_text(encoding="utf-8"))
    records[3]["path"] = records[0]["path"]
    development.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="unique source content"):
        fit_thresholds_from_manifest(development, ColorThresholdBackend(), PROTECTED)

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.cli import main
from jute_disease.data import (
    POOL_POPULATIONS,
    compare_scene_tags,
    run_scene_test,
    score_scene_test,
)
from jute_disease.data.masks import ColorThresholdBackend, SceneThresholds

RUBRIC = "scene-tag-v2"
CLEAN, FIELD, UNKNOWN = "clean-background", "field", "unknown"
COLUMNS = ("image_id", "scene_tag", "rubric_version")
# (pool, scene the image shows and the thresholds predict, final owner tag).
# A: 1 of 2 right; B: 2 of 2; C: 3 of 4. Clean images: A 2 (1 found),
# B 1 (1 found), C 2 (1 found).
WORLD = (
    ("A", CLEAN, CLEAN),
    ("A", FIELD, CLEAN),
    ("B", CLEAN, CLEAN),
    ("B", FIELD, FIELD),
    ("C", CLEAN, CLEAN),
    ("C", FIELD, CLEAN),
    ("C", FIELD, FIELD),
    ("C", FIELD, FIELD),
)
WORLD_ROWS = [(pool, owner, predicted) for pool, predicted, owner in WORLD]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tags(path: Path, tags: dict[str, str], rubric: str = RUBRIC) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows((image_id, tag, rubric) for image_id, tag in tags.items())
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _held_out(tmp_path: Path, count: int) -> tuple[Path, list[str]]:
    """A held-out set of ``count`` image records; no image file is written."""
    ids = [
        hashlib.sha256(f"image-{index}".encode()).hexdigest() for index in range(count)
    ]
    path = tmp_path / "held-out.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": image_id,
                    "path": f"{image_id}.png",
                    "pool": "ABC"[index % 3],
                    "group": f"g{index}",
                    "partition": "train",
                }
                for index, image_id in enumerate(ids)
            ]
        ),
        encoding="utf-8",
    )
    return path, ids


def _rows(
    pool: str, found: int, missed: int, field: int = 0
) -> list[tuple[str, str, str]]:
    """Clean images found and missed, and correctly tagged field images."""
    return [
        *[(pool, CLEAN, CLEAN)] * found,
        *[(pool, CLEAN, FIELD)] * missed,
        *[(pool, FIELD, FIELD)] * field,
    ]


# --- Compare step -----------------------------------------------------------


def test_compare_lists_disagreements_and_reports_each_judges_kappa(
    tmp_path: Path,
) -> None:
    held_out, ids = _held_out(tmp_path, 12)
    owner = dict.fromkeys(ids[:6], CLEAN) | dict.fromkeys(ids[6:], FIELD)
    # Haiku differs on one clean and one field image: observed agreement
    # 10/12, chance agreement 0.5, so kappa is (10/12 - 1/2) / (1/2) = 2/3.
    haiku = owner | {ids[0]: FIELD, ids[6]: CLEAN}
    luna = dict(owner)
    output = tmp_path / "compare"

    report = compare_scene_tags(
        held_out,
        _write_tags(tmp_path / "owner-scenes.csv", owner),
        {
            "haiku": _write_tags(tmp_path / "haiku.csv", haiku),
            "luna": _write_tags(tmp_path / "luna.csv", luna),
        },
        output,
        seed=7,
        resamples=500,
    )

    assert report == json.loads((output / "compare.json").read_text("utf-8"))
    assert report["judges"]["haiku"]["kappa"] == pytest.approx(2 / 3)
    assert report["judges"]["haiku"]["agreed"] == 10
    assert (
        report["judges"]["haiku"]["lower_95"]
        <= 2 / 3
        <= report["judges"]["haiku"]["upper_95"]
    )
    assert report["judges"]["luna"]["kappa"] == pytest.approx(1.0)
    assert report["judges"]["luna"]["sample_size"] == 12
    assert report["disagreements"] == sorted([ids[0], ids[6]])
    assert report["seed"] == 7
    assert report["inputs"] == {
        "held_out": _sha(held_out),
        "owner_first_pass": _sha(tmp_path / "owner-scenes.csv"),
        "judges": {
            "haiku": _sha(tmp_path / "haiku.csv"),
            "luna": _sha(tmp_path / "luna.csv"),
        },
    }
    listed = {row["image_id"]: row for row in _read_csv(output / "disagreements.csv")}
    assert listed == {
        ids[0]: {"image_id": ids[0], "owner": CLEAN, "haiku": FIELD, "luna": CLEAN},
        ids[6]: {"image_id": ids[6], "owner": FIELD, "haiku": CLEAN, "luna": FIELD},
    }
    first_pass = _read_csv(output / "first-pass.csv")
    assert {row["image_id"]: row["owner"] for row in first_pass} == owner


def _compare_fifty(tmp_path: Path, name: str, seed: int, count: int = 30) -> dict:
    held_out, ids = _held_out(tmp_path, 50)
    owner = {image_id: (CLEAN, FIELD)[index % 2] for index, image_id in enumerate(ids)}
    haiku = owner | dict.fromkeys(ids[:6], UNKNOWN)
    luna = owner | dict.fromkeys(ids[4:10], UNKNOWN)
    report = compare_scene_tags(
        held_out,
        _write_tags(tmp_path / "owner-scenes.csv", owner),
        {
            "haiku": _write_tags(tmp_path / "haiku.csv", haiku),
            "luna": _write_tags(tmp_path / "luna.csv", luna),
        },
        tmp_path / name,
        seed=seed,
        recheck_count=count,
        resamples=200,
    )
    return {**report, "ids": ids}


def test_recheck_set_is_drawn_only_from_unanimous_images(tmp_path: Path) -> None:
    report = _compare_fifty(tmp_path, "compare", seed=7)
    ids = report["ids"]

    recheck = report["recheck"]["image_ids"]
    assert len(recheck) == len(set(recheck)) == 30
    assert set(recheck) <= set(ids[10:])
    assert report["disagreements"] == sorted(ids[:10])
    assert report["recheck"]["unanimous_images"] == 40
    sheet = _read_csv(tmp_path / "compare" / "recheck.csv")
    assert [row["image_id"] for row in sheet] == recheck
    assert {row["scene_tag"] for row in sheet} == {""}
    assert {row["rubric_version"] for row in sheet} == {RUBRIC}
    assert set(sheet[0]) == {"image_id", "path", "scene_tag", "rubric_version"}


def test_recheck_draw_repeats_for_a_seed_and_changes_with_it(tmp_path: Path) -> None:
    first = _compare_fifty(tmp_path, "first", seed=7)
    again = _compare_fifty(tmp_path, "again", seed=7)
    other = _compare_fifty(tmp_path, "other", seed=8)

    assert first["recheck"]["image_ids"] == again["recheck"]["image_ids"]
    assert first["recheck"]["image_ids"] != other["recheck"]["image_ids"]
    assert (tmp_path / "first" / "recheck.csv").read_bytes() == (
        tmp_path / "again" / "recheck.csv"
    ).read_bytes()


def test_recheck_takes_every_unanimous_image_when_there_are_few(
    tmp_path: Path,
) -> None:
    report = _compare_fifty(tmp_path, "compare", seed=7, count=45)

    assert sorted(report["recheck"]["image_ids"]) == sorted(report["ids"][10:])


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "missing"),
        ("duplicated", "more than once"),
        ("unknown", "not in the held-out set"),
        ("tag", "outside the rubric"),
        ("blank", "outside the rubric"),
        ("rubric", "rubric version"),
    ],
)
@pytest.mark.parametrize("sheet", ["owner", "judge"])
def test_compare_refuses_a_damaged_tag_sheet(
    tmp_path: Path, sheet: str, damage: str, message: str
) -> None:
    held_out, ids = _held_out(tmp_path, 6)
    good = dict.fromkeys(ids, FIELD)
    rows = [[image_id, FIELD, RUBRIC] for image_id in ids]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicated":
        rows.append(rows[0])
    elif damage == "unknown":
        rows.append(["f" * 64, FIELD, RUBRIC])
    elif damage == "tag":
        rows[2][1] = "greenhouse"
    elif damage == "blank":
        rows[2][1] = ""
    else:
        rows[2][2] = "scene-tag-v1"
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([COLUMNS, *rows])
    fine = _write_tags(tmp_path / "good.csv", good)

    with pytest.raises(ValueError, match=message):
        compare_scene_tags(
            held_out,
            bad if sheet == "owner" else fine,
            {"haiku": bad if sheet == "judge" else fine},
            tmp_path / "compare",
            seed=7,
        )
    assert not (tmp_path / "compare").exists()


def test_compare_refuses_an_existing_output_and_no_judge(tmp_path: Path) -> None:
    held_out, ids = _held_out(tmp_path, 6)
    sheet = _write_tags(tmp_path / "tags.csv", dict.fromkeys(ids, FIELD))

    with pytest.raises(ValueError, match="at least one judge"):
        compare_scene_tags(held_out, sheet, {}, tmp_path / "compare", seed=7)
    (tmp_path / "compare").mkdir()
    with pytest.raises(FileExistsError):
        compare_scene_tags(
            held_out, sheet, {"haiku": sheet}, tmp_path / "compare", seed=7
        )


# --- Scoring ----------------------------------------------------------------


def test_pool_populations_are_the_protocol_ones() -> None:
    assert POOL_POPULATIONS == {"A": 920, "B": 1390, "C": 9574}
    assert sum(POOL_POPULATIONS.values()) == 11884


def test_pooled_accuracy_weights_each_pool_by_its_population_share() -> None:
    result = score_scene_test(WORLD_ROWS)

    assert result["pools"]["A"]["accuracy"]["estimate"] == pytest.approx(1 / 2)
    assert result["pools"]["B"]["accuracy"]["estimate"] == pytest.approx(1.0)
    assert result["pools"]["C"]["accuracy"]["estimate"] == pytest.approx(3 / 4)
    assert result["pools"]["C"]["share"] == pytest.approx(9574 / 11884)
    # (920 * 1/2 + 1390 * 1 + 9574 * 3/4) / 11884, not the unweighted 6/8.
    assert result["pooled"]["accuracy"]["estimate"] == pytest.approx(9030.5 / 11884)


def test_pooled_clean_recall_uses_pool_weighted_confusion_counts() -> None:
    result = score_scene_test(WORLD_ROWS)

    # Each image stands for population / sample images of its pool:
    # A 920/2 = 460, B 1390/2 = 695, C 9574/4 = 2393.5.
    # Clean images found: 460 + 695 + 2393.5 = 3548.5.
    # Clean images: 2 * 460 + 695 + 2 * 2393.5 = 6402.
    recall = result["pooled"]["clean_recall"]
    assert recall["weighted_correct"] == pytest.approx(3548.5)
    assert recall["weighted_support"] == pytest.approx(6402)
    assert recall["estimate"] == pytest.approx(3548.5 / 6402)
    # The mean of the pool recalls (1/2, 1, 1/2) would be 2/3.
    assert recall["estimate"] != pytest.approx(2 / 3)
    assert result["pooled"]["confusion"][CLEAN][CLEAN] == pytest.approx(3548.5)
    assert result["pooled"]["confusion"][CLEAN][FIELD] == pytest.approx(2853.5)


def test_pool_intervals_are_wilson_intervals() -> None:
    rows = [
        *_rows("A", found=9, missed=1),
        *_rows("B", found=10, missed=0),
        *_rows("C", found=10, missed=0),
    ]

    result = score_scene_test(rows)

    # 9 of 10 at z = 1.959964: centre (0.9 + z²/20) / (1 + z²/10) = 0.78899,
    # half-width z * sqrt(0.9 * 0.1 / 10 + z²/400) / (1 + z²/10) = 0.19314.
    for key in ("accuracy", "clean_recall"):
        measure = result["pools"]["A"][key]
        assert (measure["correct"], measure["total"]) == (9, 10)
        assert measure["lower_95"] == pytest.approx(0.59585, abs=1e-4)
        assert measure["upper_95"] == pytest.approx(0.98212, abs=1e-4)
    # 10 of 10: upper bound 1, lower bound 10 / (10 + z²) = 0.72247.
    assert result["pools"]["B"]["accuracy"]["lower_95"] == pytest.approx(
        0.72247, abs=1e-4
    )
    assert result["pools"]["B"]["accuracy"]["upper_95"] == pytest.approx(1.0)


def test_pooled_interval_is_wilson_at_the_effective_sample_size() -> None:
    result = score_scene_test(WORLD_ROWS)

    # Effective size 11884² / (920²/2 + 1390²/2 + 9574²/4) = 5.81080, with
    # p = 0.759887: centre 0.656456, half-width 0.288642.
    accuracy = result["pooled"]["accuracy"]
    assert accuracy["effective_sample_size"] == pytest.approx(141229456 / 24304619)
    assert accuracy["lower_95"] == pytest.approx(0.36781, abs=1e-4)
    assert accuracy["upper_95"] == pytest.approx(0.94510, abs=1e-4)
    # Clean images: 6402² / (2 * 460² + 695² + 2 * 2393.5²) = 3.32054.
    recall = result["pooled"]["clean_recall"]
    assert recall["effective_sample_size"] == pytest.approx(
        6402**2 / (2 * 460**2 + 695**2 + 2 * 2393.5**2)
    )
    assert recall["lower_95"] < recall["estimate"] < recall["upper_95"]


def test_a_pool_with_under_ten_clean_images_has_insufficient_evidence() -> None:
    rows = [
        *_rows("A", found=8, missed=1, field=3),
        *_rows("B", found=9, missed=1, field=3),
        *_rows("C", found=12, missed=0, field=3),
    ]

    result = score_scene_test(rows)

    assert result["pools"]["A"]["clean_recall"] == {
        "status": "insufficient evidence",
        "correct": 8,
        "total": 9,
    }
    assert result["pools"]["B"]["clean_recall"]["status"] == "reported"
    assert result["pools"]["B"]["clean_recall"]["estimate"] == pytest.approx(0.9)
    # The pooled recall still counts every pool's clean images.
    assert result["pooled"]["clean_recall"]["weighted_support"] == pytest.approx(
        9 * 920 / 12 + 10 * 1390 / 13 + 12 * 9574 / 15
    )


def test_a_pool_with_no_clean_image_reports_that_it_has_none() -> None:
    rows = [
        *_rows("A", found=0, missed=0, field=5),
        *_rows("B", found=10, missed=0),
        *_rows("C", found=10, missed=0),
    ]

    result = score_scene_test(rows)

    assert result["pools"]["A"]["clean_recall"] == {
        "status": "none",
        "correct": 0,
        "total": 0,
    }
    assert result["pooled"]["clean_recall"]["estimate"] == pytest.approx(1.0)


def _with_unknown(count: int) -> list[tuple[str, str, str]]:
    return [
        *_rows("A", found=10, missed=0),
        *_rows("B", found=10, missed=0),
        *_rows("C", found=10, missed=0),
        *[("C", UNKNOWN, UNKNOWN)] * (count - 3),
        *[("C", UNKNOWN, FIELD)] * 2,
        ("A", UNKNOWN, UNKNOWN),
    ]


def test_under_ten_unknown_images_is_insufficient_evidence() -> None:
    assert score_scene_test(_with_unknown(9))["unknown"] == {
        "status": "insufficient evidence",
        "support": 9,
    }


def test_ten_unknown_images_are_reported_with_their_support() -> None:
    unknown = score_scene_test(_with_unknown(10))["unknown"]

    assert unknown["status"] == "reported"
    assert unknown["support"] == 10
    assert unknown["support_by_pool"] == {"A": 1, "B": 0, "C": 9}
    # A's one image weighs 920/11; C's nine weigh 9574/19 each, seven found.
    weight_a, weight_c = 920 / 11, 9574 / 19
    assert unknown["estimate"] == pytest.approx(
        (weight_a + 7 * weight_c) / (weight_a + 9 * weight_c)
    )


def test_images_never_tagged_unknown_by_the_thresholds_still_score() -> None:
    # A development set with no unknown image gives thresholds that class
    # nothing as unknown; owner-unknown images then count as errors.
    rows = [
        *_rows("A", found=10, missed=0),
        *_rows("B", found=10, missed=0),
        *_rows("C", found=8, missed=0),
        ("C", UNKNOWN, FIELD),
        ("C", UNKNOWN, CLEAN),
    ]

    result = score_scene_test(rows)

    assert result["pools"]["C"]["accuracy"]["estimate"] == pytest.approx(0.8)
    assert result["unknown"] == {"status": "insufficient evidence", "support": 2}


def _even(correct: int, found: int) -> list[tuple[str, str, str]]:
    """Every pool: 20 images, 10 of them clean; ``correct`` and ``found`` right."""
    pool_rows = [
        *[(CLEAN, CLEAN)] * found,
        *[(CLEAN, FIELD)] * (10 - found),
        *[(FIELD, FIELD)] * (correct - found),
        *[(FIELD, CLEAN)] * (10 - (correct - found)),
    ]
    return [
        (pool, owner, predicted) for pool in "ABC" for owner, predicted in pool_rows
    ]


@pytest.mark.parametrize(
    ("correct", "found", "passed"),
    [
        (18, 8, True),  # exactly 90% accuracy and 80% recall: both bars are met
        (17, 8, False),  # 85% accuracy
        (17, 7, False),  # 70% recall
        (20, 10, True),
    ],
)
def test_pass_needs_both_pooled_point_estimates(
    correct: int, found: int, passed: bool
) -> None:
    result = score_scene_test(_even(correct, found))

    assert result["pooled"]["accuracy"]["estimate"] == pytest.approx(correct / 20)
    assert result["pooled"]["clean_recall"]["estimate"] == pytest.approx(found / 10)
    assert result["passed"] is passed
    assert result["bars"] == {"accuracy": 0.9, "clean_recall": 0.8}


def test_recall_decides_on_the_pooled_estimate_not_each_pool() -> None:
    # Pool A finds 5 of 10 clean images, yet it is under 8% of the population.
    rows = [
        *_rows("A", found=5, missed=5, field=90),
        *_rows("B", found=10, missed=0, field=90),
        *_rows("C", found=10, missed=0, field=90),
    ]

    result = score_scene_test(rows)

    assert result["pools"]["A"]["clean_recall"]["estimate"] == pytest.approx(0.5)
    assert result["pooled"]["clean_recall"]["estimate"] == pytest.approx(
        (920 * 5 + 1390 * 10 + 9574 * 10) / (11884 * 10)
    )
    assert result["passed"] is True


def test_no_clean_image_anywhere_cannot_pass() -> None:
    result = score_scene_test([(pool, FIELD, FIELD) for pool in "ABC"] * 10)

    assert result["pooled"]["clean_recall"] == {"status": "none"}
    assert result["passed"] is False


def test_scoring_refuses_a_missing_pool_and_a_foreign_tag() -> None:
    with pytest.raises(ValueError, match="pools"):
        score_scene_test([("A", CLEAN, CLEAN), ("B", CLEAN, CLEAN)])
    with pytest.raises(ValueError, match="outside the rubric"):
        score_scene_test([*WORLD_ROWS, ("C", "greenhouse", CLEAN)])


# --- Test step --------------------------------------------------------------


def _thresholds(tmp_path: Path, ids: tuple[str, ...] | None = None) -> Path:
    ids = ids or tuple(f"dev-{index}" for index in range(100))
    thresholds = SceneThresholds(
        background_fraction_min=0.5,
        color_uniformity_min=0.9,
        development_ids=ids,
        development_sources=tuple(
            (image_id, hashlib.sha256(f"source-{index}".encode()).hexdigest())
            for index, image_id in enumerate(ids)
        ),
        development_digest="a" * 64,
        score=1.0,
        mask_model=ColorThresholdBackend.model,
        mask_version=ColorThresholdBackend.version,
    )
    path = tmp_path / "scene-thresholds.json"
    path.write_text(json.dumps(thresholds.to_dict()), encoding="utf-8")
    return path


def _image(path: Path, scene: str, index: int) -> None:
    """An image the 0.5 / 0.9 thresholds class as ``scene``."""
    pixels = np.full((8, 8, 3), 245, dtype=np.uint8)
    if scene == FIELD:
        for y in range(8):
            pixels[y, :, :] = 30 + y * 25, 60 + y * 17, 10 + y * 9
    if scene == UNKNOWN:
        pixels[1:8, 1:8] = (0, 180, 0)
    else:
        pixels[2:6, 2:6] = (0, 180, 0)
    pixels[0, 0] = (245, 245, index)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def _test_world(tmp_path: Path) -> dict:
    """The WORLD images, a compare output, and the owner's later sheets.

    The owner first tagged the seventh image unknown, Haiku called it field,
    and the owner's final tag is field. Every other first tag is unanimous.
    """
    images = tmp_path / "images"
    images.mkdir()
    records, final = [], {}
    for index, (pool, scene, owner) in enumerate(WORLD):
        path = images / f"{index}.png"
        _image(path, scene, index)
        image_id = _sha(path)
        records.append(
            {
                "image_id": image_id,
                "path": f"images/{index}.png",
                "pool": pool,
                "group": f"g{index}",
                "partition": "train",
            }
        )
        final[image_id] = owner
    ids = list(final)
    held_out = tmp_path / "held-out.json"
    held_out.write_text(json.dumps(records), encoding="utf-8")
    first = final | {ids[6]: UNKNOWN}
    owner = _write_tags(tmp_path / "owner-scenes.csv", first)
    haiku = _write_tags(tmp_path / "haiku.csv", final)
    compare = tmp_path / "compare"
    report = compare_scene_tags(
        held_out,
        owner,
        {"haiku": haiku},
        compare,
        seed=7,
        recheck_count=3,
        resamples=200,
    )
    recheck_ids = report["recheck"]["image_ids"]
    recheck = {image_id: first[image_id] for image_id in recheck_ids}
    recheck[recheck_ids[0]] = UNKNOWN
    return {
        "ids": ids,
        "first": first,
        "final_tags": final,
        "recheck_tags": recheck,
        "held_out": held_out,
        "thresholds": _thresholds(tmp_path),
        "compare": compare,
        "owner": owner,
        "haiku": haiku,
        "final": _write_tags(tmp_path / "final.csv", final),
        "recheck": _write_tags(tmp_path / "recheck-tags.csv", recheck),
        "output": tmp_path / "evaluation.json",
    }


def _run(world: dict, **changes) -> dict:
    paths = {**world, **changes}
    return run_scene_test(
        paths["held_out"],
        paths["thresholds"],
        paths["compare"],
        paths["owner"],
        paths["final"],
        paths["output"],
        ColorThresholdBackend(),
        recheck=paths["recheck"],
    )


def test_test_step_scores_the_frozen_thresholds_against_final_tags(
    tmp_path: Path,
) -> None:
    world = _test_world(tmp_path)

    record = _run(world)

    assert record == json.loads(world["output"].read_text("utf-8"))
    assert record["pooled"]["accuracy"]["estimate"] == pytest.approx(9030.5 / 11884)
    assert record["pooled"]["clean_recall"]["estimate"] == pytest.approx(3548.5 / 6402)
    assert record["pools"]["A"]["images"] == 2
    assert record["pools"]["A"]["clean_recall"]["status"] == "insufficient evidence"
    assert record["unknown"] == {"status": "insufficient evidence", "support": 0}
    assert record["passed"] is False
    assert record["rubric_version"] == RUBRIC
    assert [
        (row["image_id"], row["pool"], row["owner_final"], row["predicted"])
        for row in record["images"]
    ] == sorted(
        (image_id, pool, owner, scene)
        for image_id, (pool, scene, owner) in zip(world["ids"], WORLD, strict=True)
    )
    assert record["final_tags"] == {
        "changed_from_first_pass": [world["ids"][6]],
        "disagreements_reviewed": 1,
    }


def test_test_record_binds_the_digests_of_its_inputs(tmp_path: Path) -> None:
    world = _test_world(tmp_path)

    record = _run(world)

    assert record["inputs"] == {
        "held_out": _sha(world["held_out"]),
        "thresholds": _sha(world["thresholds"]),
        "compare": _sha(world["compare"] / "compare.json"),
        "owner_first_pass": _sha(world["owner"]),
        "judges": {"haiku": _sha(world["haiku"])},
        "final_tags": _sha(world["final"]),
        "recheck_tags": _sha(world["recheck"]),
    }
    payload = {key: value for key, value in record.items() if key != "digest"}
    assert (
        record["digest"]
        == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_test_step_reports_recheck_agreement(tmp_path: Path) -> None:
    world = _test_world(tmp_path)

    record = _run(world)

    changed = next(iter(world["recheck_tags"]))
    assert record["recheck"] == {
        "status": "reported",
        "images": 3,
        "agreed": 2,
        "agreement": pytest.approx(2 / 3),
        "changed": [
            {"image_id": changed, "first": world["first"][changed], "recheck": UNKNOWN}
        ],
    }


def test_test_step_runs_without_recheck_tags(tmp_path: Path) -> None:
    world = _test_world(tmp_path)

    record = _run(world, recheck=None)

    assert record["recheck"] == {"status": "not supplied"}
    assert record["inputs"]["recheck_tags"] is None


def test_a_second_test_run_is_refused(tmp_path: Path) -> None:
    world = _test_world(tmp_path)
    _run(world)
    written = world["output"].read_bytes()

    with pytest.raises(FileExistsError, match="run once"):
        _run(world)
    assert world["output"].read_bytes() == written


def test_test_step_refuses_a_final_tag_changed_outside_the_disagreements(
    tmp_path: Path,
) -> None:
    world = _test_world(tmp_path)
    moved = world["final_tags"] | {world["ids"][0]: FIELD}

    with pytest.raises(ValueError, match="not a listed disagreement"):
        _run(world, final=_write_tags(tmp_path / "moved.csv", moved))
    assert not world["output"].exists()


@pytest.mark.parametrize(
    ("sheet", "damage", "message"),
    [
        ("final", "missing", "missing"),
        ("final", "duplicated", "more than once"),
        ("final", "unknown", "not in the held-out set"),
        ("final", "tag", "outside the rubric"),
        ("recheck", "missing", "missing"),
        ("recheck", "unknown", "not in the recheck set"),
        ("recheck", "tag", "outside the rubric"),
    ],
)
def test_test_step_refuses_a_damaged_tag_sheet(
    tmp_path: Path, sheet: str, damage: str, message: str
) -> None:
    world = _test_world(tmp_path)
    tags = world["final_tags"] if sheet == "final" else world["recheck_tags"]
    rows = [[image_id, tag, RUBRIC] for image_id, tag in tags.items()]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicated":
        rows.append(rows[0])
    elif damage == "unknown":
        rows.append(["f" * 64, FIELD, RUBRIC])
    else:
        rows[0][1] = "greenhouse"
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([COLUMNS, *rows])

    with pytest.raises(ValueError, match=message):
        _run(world, **{sheet: bad})
    assert not world["output"].exists()


def test_test_step_refuses_inputs_the_compare_step_did_not_see(
    tmp_path: Path,
) -> None:
    world = _test_world(tmp_path)
    other = _write_tags(tmp_path / "other.csv", world["final_tags"])

    with pytest.raises(ValueError, match="first-pass tags differ"):
        _run(world, owner=other)
    records = json.loads(world["held_out"].read_text("utf-8"))
    world["held_out"].write_text(json.dumps(records[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out set differs"):
        _run(world)


def test_test_step_refuses_an_image_that_does_not_match_its_id(
    tmp_path: Path,
) -> None:
    world = _test_world(tmp_path)
    _image(tmp_path / "images" / "0.png", FIELD, 200)

    with pytest.raises(ValueError, match="does not match its image ID"):
        _run(world)


def test_test_step_refuses_a_held_out_image_used_for_development(
    tmp_path: Path,
) -> None:
    world = _test_world(tmp_path)
    ids = (world["ids"][0], *(f"dev-{index}" for index in range(99)))

    with pytest.raises(ValueError, match="development"):
        _run(world, thresholds=_thresholds(tmp_path, ids))


# --- Command ----------------------------------------------------------------


def test_scene_evaluate_command_compares_then_tests_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = _test_world(tmp_path)
    compare = tmp_path / "cli-compare"

    main(
        [
            "data",
            "scene-evaluate",
            "compare",
            "--held-out",
            str(world["held_out"]),
            "--owner",
            str(world["owner"]),
            "--judge",
            f"haiku={world['haiku']}",
            "--output",
            str(compare),
            "--seed",
            "7",
            "--recheck-count",
            "3",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["disagreements"] == 1
    assert summary["recheck_images"] == 3
    assert (compare / "recheck.csv").read_bytes() == (
        world["compare"] / "recheck.csv"
    ).read_bytes()

    arguments = [
        "data",
        "scene-evaluate",
        "test",
        "--held-out",
        str(world["held_out"]),
        "--thresholds",
        str(world["thresholds"]),
        "--compare",
        str(compare),
        "--owner",
        str(world["owner"]),
        "--final",
        str(world["final"]),
        "--recheck",
        str(world["recheck"]),
        "--output",
        str(world["output"]),
    ]
    main(arguments)
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is False
    assert summary["pooled_accuracy"] == pytest.approx(9030.5 / 11884)
    assert json.loads(world["output"].read_text("utf-8"))["passed"] is False

    with pytest.raises(SystemExit, match="scene evaluation blocked.*run once"):
        main(arguments)


def test_scene_evaluate_command_reports_a_bad_judge_option(tmp_path: Path) -> None:
    world = _test_world(tmp_path)

    with pytest.raises(SystemExit, match="scene evaluation blocked.*NAME=PATH"):
        main(
            [
                "data",
                "scene-evaluate",
                "compare",
                "--held-out",
                str(world["held_out"]),
                "--owner",
                str(world["owner"]),
                "--judge",
                str(world["haiku"]),
                "--output",
                str(tmp_path / "out"),
            ]
        )

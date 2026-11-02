"""The public evaluator gate is exercised only against the synthetic fixture."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import SyntheticPlan, freeze_target


@dataclass(frozen=True)
class Inputs:
    image_root: Path
    target_dir: Path


@pytest.fixture
def inputs(synthetic: SyntheticPlan, tmp_path: Path) -> Inputs:
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    freeze_dir = tmp_path / "freeze"
    freeze_target(
        freeze_dir,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )
    target_dir = freeze_dir / "target"
    assert (target_dir / "evaluator" / "test-manifest.json").is_file()
    return Inputs(synthetic.root, target_dir)


def _mini_benchmark(tmp_path: Path) -> Path:
    source = Path(__file__).parents[3] / "configs" / "benchmark.json"
    benchmark = json.loads(source.read_text(encoding="utf-8"))
    benchmark["recipe"]["seeds"] = [42, 1337, 7, 21, 84]
    benchmark["recipe"]["source_seeds"] = [42, 1337, 7]
    benchmark["cnn_systems"] = {
        "mobilenet_v2_direct": {
            "model": "mobilenetv2_100.ra_in1k",
            "micro_batch": 8,
            "chain": [],
        }
    }
    benchmark["classical"] = {
        "estimators": ["lr"],
        "features": ["crafted"],
        "seed": 42,
    }
    benchmark["matrix"] = {
        "pools": ["A"],
        "views": [1],
        "splits": [0],
        "primary_split": 0,
        "policies": ["image-random"],
    }
    benchmark["assignments"] = [
        {"split": 0, "seed": index, "checkpoint": index % 3} for index in range(5)
    ]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")
    return path


def _outcomes(path: Path, *, mask_audit: str = "passed") -> None:
    path.write_text(
        json.dumps(
            {
                "issue": 114,
                "letterbox_comparison": "letterbox",
                "mask_audit": {"status": mask_audit},
                "post_pilot_budget_hours": 100.0,
                "scene_thresholds": {
                    "background_fraction_min": 0.1,
                    "color_uniformity_min": 0.9,
                },
                "border_frame_width_px": 32,
                "border_leaf_share_cutoff": 0.05,
                "selected_hyperparameters": {"fixture": {"value": 1}},
            }
        ),
        encoding="utf-8",
    )


def test_cli_refuses_test_read_without_both_records_then_evaluates_fixture(
    inputs: Inputs, tmp_path: Path
) -> None:
    benchmark = _mini_benchmark(tmp_path)
    selection = tmp_path / "selection.json"
    run_record = tmp_path / "run.json"
    output = tmp_path / "evaluation"
    args = [
        "evaluate",
        "gated",
        "--selection",
        str(selection),
        "--run-record",
        str(run_record),
        "--target-dir",
        str(inputs.target_dir),
        "--image-root",
        str(inputs.image_root),
        "--output-dir",
        str(output),
        "--fixture",
    ]
    with pytest.raises(
        SystemExit, match="evaluation blocked: Cannot read selection record"
    ):
        main(args)

    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes)
    main(
        [
            "evaluate",
            "matrix-freeze",
            "--validation-outcomes",
            str(outcomes),
            "--benchmark",
            str(benchmark),
            "--target-dir",
            str(inputs.target_dir),
            "--output",
            str(selection),
        ]
    )
    with pytest.raises(SystemExit, match="evaluation blocked: Cannot read run record"):
        main(args)

    main(
        [
            "evaluate",
            "fixture-run-record",
            "--selection",
            str(selection),
            "--output",
            str(run_record),
        ]
    )
    main(args)

    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert report["leaf_only"]["sample_size"] <= 2_000
    assert report["leaf_only"]["fit_scores"]
    assert len(report["background_only"]["fit_scores"]) == 10
    assert report["border"]
    cell = next(iter(next(iter(report["class_by_scene"].values())).values()))
    assert "class_cells" in cell
    assert all("accuracy_wilson_95" in row for row in cell["class_cells"].values())
    for model_scenes in report["class_by_scene"].values():
        for scene_row in model_scenes.values():
            eligible = all(count >= 30 for count in scene_row["class_counts"].values())
            assert ("macro_f1" in scene_row) is eligible

    later_selection = tmp_path / "another-folder" / "selection.json"
    main(
        [
            "evaluate",
            "matrix-freeze",
            "--validation-outcomes",
            str(outcomes),
            "--benchmark",
            str(benchmark),
            "--target-dir",
            str(inputs.target_dir),
            "--output",
            str(later_selection),
        ]
    )
    later_record = json.loads(later_selection.read_text(encoding="utf-8"))["record"]
    assert later_record["created_after_test_reads"] is True
    assert "test_read_history_sha256" in later_record
    with pytest.raises(SystemExit, match="selection was superseded"):
        main(args)


def test_default_matrix_freeze_records_all_probe_routes(
    inputs: Inputs, tmp_path: Path
) -> None:
    root = Path(__file__).parents[3]
    selection_path = tmp_path / "selection.json"
    outcomes_path = tmp_path / "validation-114.json"
    _outcomes(outcomes_path)
    main(
        [
            "evaluate",
            "matrix-freeze",
            "--validation-outcomes",
            str(outcomes_path),
            "--benchmark",
            str(root / "configs" / "benchmark.json"),
            "--target-dir",
            str(inputs.target_dir),
            "--output",
            str(selection_path),
        ]
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["record"][
        "selection"
    ]
    background = selection["background_only"]
    assert background["seeds"] == [42, 1337, 7, 21, 84]
    assert len(background["fit_ids"]) == 495
    assert background["cnn_route"] == "mobilenetv2_100.ra_in1k (ImageNet)"
    assert len(selection["pools"]) == 3
    assert len(selection["policies"]) == 3
    assert selection["primary_split"] == 0
    leaf = selection["leaf_only"]
    assert leaf["sample_size"] == 2_000
    assert leaf["fit_ids"] == selection["final_fit_ids"]
    assert selection["border"]["frame_width_px_start"] == 32
    assert selection["border"]["leaf_share_cutoff_start"] == 0.05
    assert selection["border"]["null_permutations"] == 1_000
    assert len(selection["border"]["fit_ids"]) == 351


def test_matrix_freeze_requires_recorded_issue_114_outcomes(
    inputs: Inputs, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="#114 validation outcomes are required"):
        main(
            [
                "evaluate",
                "matrix-freeze",
                "--validation-outcomes",
                str(tmp_path / "missing.json"),
                "--benchmark",
                str(_mini_benchmark(tmp_path)),
                "--target-dir",
                str(inputs.target_dir),
                "--output",
                str(tmp_path / "selection.json"),
            ]
        )
    assert not (tmp_path / "selection.json").exists()

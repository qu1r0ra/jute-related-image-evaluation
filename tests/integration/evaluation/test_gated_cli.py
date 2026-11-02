"""Exercise the public evaluator through the synthetic fixture and artifacts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.dummy import DummyClassifier

from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG, CURATION_RULES_PATH
from jute_disease.data import (
    PreprocessingSpec,
    SyntheticPlan,
    freeze_target,
    load_manifest,
)
from jute_disease.evaluation import (
    GatedEvaluationError,
    evaluate_gated_test,
    write_matrix_freeze,
)
from jute_disease.evaluation import (
    write_gated_run_record as write_run_record,
)
from jute_disease.training import RunStore, fit_identity_digest, load_benchmark


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
    public_manifest = load_manifest(target_dir)
    assert not public_manifest.partition("test")
    assert all(
        entry["partition"] != "test"
        for entry in json.loads(
            (target_dir / "manifest.json").read_text(encoding="utf-8")
        )["entries"]
    )
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


def _outcomes(
    path: Path,
    *,
    mask_audit: str = "passed",
    letterbox: str = "letterbox",
    border_leaf_share_cutoff: float = 0.05,
) -> None:
    from jute_disease.models.ml import classical_configurations

    benchmark = load_benchmark(BENCHMARK_CONFIG)
    selected_hyperparameters = {
        f"classical/tune/{pool}/{estimator}-{features}": (
            classical_configurations(estimator)[0]
        )
        for estimator in benchmark.estimators
        for features in benchmark.features
        for pool in benchmark.pools
    }
    selected_hyperparameters.update(
        {
            f"cnn/tune/{system}/pool-{pool}": benchmark.recipe.learning_rate
            for system in benchmark.systems
            for pool in benchmark.pools
        }
    )
    path.write_text(
        json.dumps(
            {
                "issue": 114,
                "letterbox_comparison": letterbox,
                "mask_audit": {"status": mask_audit},
                "post_pilot_budget_hours": 100.0,
                "scene_thresholds": {
                    "background_fraction_min": 0.1,
                    "color_uniformity_min": 0.9,
                },
                "border_frame_width_px": 32,
                "border_leaf_share_cutoff": border_leaf_share_cutoff,
                "selected_hyperparameters": selected_hyperparameters,
            }
        ),
        encoding="utf-8",
    )


def _complete_fit_receipt(
    store: RunStore,
    fit_id: str,
    artifact_path: Path,
    model_spec: dict[str, object],
    selection_record: dict[str, object],
) -> None:
    training_recipe = selection_record["training_recipe"]
    assert isinstance(training_recipe, dict)
    assignment_rows = selection_record["assignment_table"]
    assignment_raw_sha256 = hashlib.sha256(b"fixture assignments").hexdigest()
    assignment_content_sha256 = hashlib.sha256(
        json.dumps(assignment_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    preprocessing = PreprocessingSpec(
        size=selection_record["preprocessing"]["size"],
        resize=selection_record["preprocessing"]["resize"],
    )
    commit = "a" * 40
    run_options = {
        "pretrained": True,
        "accelerator": "cuda",
        "require_cuda": True,
    }
    fit_inputs = {
        "fit_id": fit_id,
        "kind": model_spec["kind"],
        "seed": model_spec["seed"],
        **{
            key: model_spec[key]
            for key in (
                "pool",
                "policy",
                "split",
                "estimator",
                "features",
                "system",
                "model",
                "view",
            )
            if key in model_spec
        },
    }
    identity_inputs = {
        "fit": fit_inputs,
        "recipe": training_recipe,
        "manifests": {
            "target": selection_record["manifest_digest"],
            "target_content": "b" * 64,
        },
        "upstream_sha256": None,
        "environment": {**run_options, "tuning": None},
        "config_sha256": selection_record["benchmark_sha256"],
        "code_commit": commit,
        "environment_lock_sha256": selection_record["environment_lock_sha256"],
        "preprocessing_sha256": preprocessing.digest(),
        "assignment_table_sha256": assignment_raw_sha256,
        "source_checkpoint_ancestry": [],
        "evaluation_fit_spec": model_spec,
        "evaluation_selection_sha256": hashlib.sha256(
            (
                json.dumps(selection_record, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
        ).hexdigest(),
    }
    metadata = {
        "identity_inputs": identity_inputs,
        "official": True,
        "provenance": {
            "commit": commit,
            "working_tree_clean": True,
            "official": True,
        },
        "configuration": {"sha256": selection_record["benchmark_sha256"]},
        "environment_lock": {"sha256": selection_record["environment_lock_sha256"]},
        "preprocessing": {
            "spec": asdict(preprocessing),
            "sha256": preprocessing.digest(),
        },
        "assignment_table": {
            "present": True,
            "path": "assignments.json",
            "validated": True,
            "sha256": assignment_raw_sha256,
            "content_sha256": assignment_content_sha256,
            "assignments": assignment_rows,
        },
        "source_checkpoint_ancestry": [],
        "seed": model_spec["seed"],
        "runtime_contract": {
            "precision": training_recipe["precision"],
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
            "torch_compile": False,
            "num_workers": training_recipe["num_workers"],
        },
        "run_options": run_options,
        "runtime": {
            "python": "3.12.0",
            "torch": "2.5.1",
            "torch_cuda": "12.4",
            "cuda_driver": "555.1",
            "cuda_available": True,
            "precision": training_recipe["precision"],
            "compile_enabled": False,
            "backend": {
                "tf32_matmul": True,
                "cudnn_tf32": True,
                "cudnn_benchmark": False,
            },
            "deterministic_algorithms": False,
            "cudnn_deterministic": False,
            "float32_matmul_precision": "highest",
            "libraries": {
                "timm": "1.0.0",
                "lightning": "2.5.0",
                "scikit-learn": "1.6.0",
                "torchvision": "0.20.0",
            },
            "device": {
                "name": "fixture GPU",
                "capability": [8, 0],
                "memory_bytes": 16_000_000_000,
            },
        },
    }
    identity = fit_identity_digest(identity_inputs)
    run = store.start(fit_id, identity, metadata=metadata)
    results = {
        "fit": {"fit_id": fit_id},
        "evaluation_fit_spec": model_spec,
    }
    if model_spec["kind"] == "classical":
        results["selected_parameters"] = model_spec["selected_parameters"]
    elif model_spec["kind"] == "cnn":
        results["recipe"] = {
            **training_recipe,
            "learning_rate": model_spec["selected_learning_rate"],
        }
    run.complete(
        results=results,
        artifacts={"model": artifact_path},
        metadata=metadata,
    )


def test_cli_refuses_test_read_without_both_records_then_evaluates_fixture(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _mini_benchmark(tmp_path)
    selection = tmp_path / "selection.json"
    run_record = tmp_path / "run.json"
    output = tmp_path / "evaluation"
    args = [
        "evaluate",
        "score-test",
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
            "freeze-matrix",
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
            "write-fixture-run-record",
            "--selection",
            str(selection),
            "--output",
            str(run_record),
        ]
    )
    main(args)

    test_rows = json.loads(
        (inputs.target_dir / "evaluator" / "test-manifest.json").read_text(
            encoding="utf-8"
        )
    )["record"]["entries"]
    test_image = inputs.image_root / test_rows[0]["path"]
    with pytest.raises(SystemExit) as missing_target_dir:
        main(["evaluate", "predict", str(test_image)])
    assert missing_target_dir.value.code == 2
    with pytest.raises(SystemExit, match="target test images can only be read"):
        main(
            [
                "evaluate",
                "predict",
                str(test_image),
                "--target-dir",
                str(inputs.target_dir),
            ]
        )
    with pytest.raises(SystemExit, match="image root differs from the freeze report"):
        main(
            [
                "evaluate",
                "predict",
                str(test_image),
                "--target-dir",
                str(inputs.target_dir),
                "--image-root",
                str(tmp_path / "other-image-root"),
            ]
        )

    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert report["border_leaf_share_cutoff"] == 0.05
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

    history_path = inputs.target_dir / "evaluator" / "test-read-history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [row["status"] for row in history] == ["started", "completed"]
    import jute_disease.evaluation.gated as gated

    with monkeypatch.context() as patch:
        patch.setattr(gated, "_pipeline_code_tree_sha256", lambda root=None: "0" * 64)
        with pytest.raises(
            SystemExit, match="evaluator code differs from the frozen pipeline"
        ):
            main(args)
    with monkeypatch.context() as patch:
        patch.setattr(gated, "_environment_lock_sha256", lambda root=None: "0" * 64)
        with pytest.raises(
            SystemExit, match="environment lock differs from the frozen pipeline"
        ):
            main(args)
    assert [
        json.loads(line) for line in history_path.read_text().splitlines()
    ] == history

    original_run_record = run_record.read_bytes()
    malformed_envelope = json.loads(original_run_record)
    malformed_envelope["record"]["completed_analysis_ids"].append([])
    canonical = (
        json.dumps(malformed_envelope["record"], sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    malformed_envelope["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    run_record.write_text(
        json.dumps(malformed_envelope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        SystemExit,
        match="evaluation blocked: run record does not cover every frozen analysis",
    ):
        main(args)
    assert [
        json.loads(line)["status"] for line in history_path.read_text().splitlines()
    ] == ["started", "completed"]
    run_record.write_bytes(original_run_record)

    test_rows = json.loads(
        (inputs.target_dir / "evaluator" / "test-manifest.json").read_text()
    )["record"]["entries"]
    missing_image = inputs.image_root / test_rows[0]["path"]
    image_bytes = missing_image.read_bytes()
    missing_image.unlink()
    try:
        with pytest.raises(SystemExit, match="cannot read frozen test image"):
            main(args)
    finally:
        missing_image.write_bytes(image_bytes)
    failed_read = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert failed_read[-1]["status"] == "started"

    later_selection = tmp_path / "another-folder" / "selection.json"
    main(
        [
            "evaluate",
            "freeze-matrix",
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


def test_matrix_freeze_translates_file_errors_to_cli_exit(
    inputs: Inputs, tmp_path: Path
) -> None:
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes)
    with pytest.raises(SystemExit, match="matrix freeze blocked:.*benchmark.json"):
        main(
            [
                "evaluate",
                "freeze-matrix",
                "--validation-outcomes",
                str(outcomes),
                "--benchmark",
                str(tmp_path / "missing-benchmark.json"),
                "--target-dir",
                str(inputs.target_dir),
                "--output",
                str(tmp_path / "selection.json"),
            ]
        )


def test_fixture_run_record_translates_malformed_hashed_selection_to_cli_exit(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    run_record_path = tmp_path / "run.json"
    record = {"issue": 139}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    selection_path.write_text(
        json.dumps(
            {
                "format": "jute-evaluation-selection-v1",
                "record": record,
                "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
        match="fixture run record blocked: selection record has an invalid matrix schema",
    ):
        main(
            [
                "evaluate",
                "write-fixture-run-record",
                "--selection",
                str(selection_path),
                "--output",
                str(run_record_path),
            ]
        )
    assert not run_record_path.exists()


def test_score_test_rejects_hash_valid_selection_with_unknown_view(
    inputs: Inputs, tmp_path: Path
) -> None:
    outcomes = tmp_path / "validation-114.json"
    selection_path = tmp_path / "selection.json"
    _outcomes(outcomes)
    main(
        [
            "evaluate",
            "freeze-matrix",
            "--validation-outcomes",
            str(outcomes),
            "--benchmark",
            str(_mini_benchmark(tmp_path)),
            "--target-dir",
            str(inputs.target_dir),
            "--output",
            str(selection_path),
        ]
    )

    envelope = json.loads(selection_path.read_text(encoding="utf-8"))
    record = envelope["record"]
    fit_id = record["selection"]["planned_fit_ids"][0]
    record["selection"]["model_specs"][fit_id]["input_view"] = "unknown-view"
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    envelope["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    selection_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        SystemExit,
        match="evaluation blocked: selection record has an invalid matrix schema",
    ):
        main(
            [
                "evaluate",
                "score-test",
                "--selection",
                str(selection_path),
                "--run-record",
                str(tmp_path / "missing-run-record.json"),
                "--target-dir",
                str(inputs.target_dir),
                "--image-root",
                str(inputs.image_root),
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--fixture",
            ]
        )
    assert not (inputs.target_dir / "evaluator" / "test-read-history.jsonl").exists()


def test_gated_api_runs_hash_bound_artifacts_on_masked_views(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jute_disease.data as data
    import jute_disease.evaluation.gated as gated
    import jute_disease.models.dl as dl

    class TinyBackbone(torch.nn.Module):
        out_features = 3

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.mean(dim=(2, 3))

    monkeypatch.setattr(gated, "_assert_committed_selection", lambda _: None)
    monkeypatch.setattr(data, "Sam2Refiner", lambda _: data.ColorThresholdBackend())
    monkeypatch.setattr(
        dl, "TimmBackbone", lambda model_name, pretrained: TinyBackbone()
    )
    benchmark = _mini_benchmark(tmp_path)
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes, letterbox="resize-center-crop")
    selection = tmp_path / "selection.json"
    write_matrix_freeze(
        validation_outcomes=outcomes,
        benchmark_path=benchmark,
        target_dir=inputs.target_dir,
        output_path=selection,
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    classical_path = models_dir / "classical.joblib"
    cue_path = models_dir / "cue.joblib"
    cnn_path = models_dir / "cnn.ckpt"
    class_count = len(
        json.loads(selection.read_text(encoding="utf-8"))["record"]["classes"]
    )
    classical = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((class_count, 548), dtype=np.float32), np.arange(class_count)
    )
    cue = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((class_count, 1), dtype=np.float32), np.arange(class_count)
    )
    joblib.dump(classical, classical_path)
    joblib.dump(cue, cue_path)
    classifier = dl.Classifier(TinyBackbone(), num_classes=class_count)
    torch.save({"state_dict": classifier.state_dict()}, cnn_path)
    paths = {"classical": classical_path, "cue": cue_path, "cnn": cnn_path}
    selection_record = json.loads(selection.read_text(encoding="utf-8"))["record"]
    run_store = RunStore(tmp_path / "training-runs")
    model_index = {}
    for fit_id, spec in selection_record["selection"]["model_specs"].items():
        artifact_path = paths[spec["kind"]]
        model_index[fit_id] = {
            "path": str(artifact_path),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "fit_spec": spec,
        }
        _complete_fit_receipt(run_store, fit_id, artifact_path, spec, selection_record)
    gradcam_receipt_path = tmp_path / "gradcam-test-pass.json"
    gradcam_receipt = {"analysis_id": "gradcam/test-pass", "status": "passed"}
    gradcam_receipt_path.write_text(json.dumps(gradcam_receipt), encoding="utf-8")
    index_path = tmp_path / "model-index.json"
    index_path.write_text(
        json.dumps(
            {
                "run_root": str(run_store.root),
                "models": model_index,
                "analyses": {
                    "gradcam/test-pass": {
                        "path": str(gradcam_receipt_path),
                        "sha256": hashlib.sha256(
                            gradcam_receipt_path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    run_record = tmp_path / "run.json"
    invalid_index = json.loads(index_path.read_text(encoding="utf-8"))
    first_fit_id = next(iter(invalid_index["models"]))
    invalid_index["models"][first_fit_id]["fit_spec"]["training_view"] = (
        "__mismatched-view__"
    )
    invalid_index_path = tmp_path / "invalid-model-index.json"
    invalid_index_path.write_text(json.dumps(invalid_index), encoding="utf-8")
    with pytest.raises(GatedEvaluationError, match="model fit spec differs"):
        write_run_record(selection, invalid_index_path, tmp_path / "invalid-run.json")

    first_receipt_path = run_store.run_dir(first_fit_id) / "receipt.json"
    original_receipt = first_receipt_path.read_bytes()
    invalid_receipt = json.loads(original_receipt)
    invalid_receipt["metadata"]["official"] = False
    invalid_receipt["attempts"][-1]["metadata"]["official"] = False
    first_receipt_path.write_text(json.dumps(invalid_receipt), encoding="utf-8")
    try:
        with pytest.raises(GatedEvaluationError, match="does not satisfy #107"):
            write_run_record(selection, index_path, tmp_path / "unofficial-run.json")
    finally:
        first_receipt_path.write_bytes(original_receipt)

    invalid_receipt = json.loads(original_receipt)
    invalid_receipt["metadata"]["identity_inputs"]["evaluation_fit_spec"][
        "validation_selected_hyperparameters"
    ] = {"tampered": True}
    invalid_receipt["attempts"][-1]["metadata"] = invalid_receipt["metadata"]
    first_receipt_path.write_text(json.dumps(invalid_receipt), encoding="utf-8")
    try:
        with pytest.raises(GatedEvaluationError, match="does not satisfy #107"):
            write_run_record(
                selection, index_path, tmp_path / "unbound-hyperparameters.json"
            )
    finally:
        first_receipt_path.write_bytes(original_receipt)

    invalid_receipt = json.loads(original_receipt)
    first_spec = selection_record["selection"]["model_specs"][first_fit_id]
    if first_spec["kind"] == "classical":
        invalid_receipt["results"]["selected_parameters"] = {"tampered": True}
    elif first_spec["kind"] == "cnn":
        invalid_receipt["results"]["recipe"]["learning_rate"] *= 2
    invalid_receipt["attempts"][-1]["metadata"] = invalid_receipt["metadata"]
    first_receipt_path.write_text(json.dumps(invalid_receipt), encoding="utf-8")
    try:
        with pytest.raises(GatedEvaluationError, match="does not satisfy #107"):
            write_run_record(
                selection, index_path, tmp_path / "unbound-fit-choice.json"
            )
    finally:
        first_receipt_path.write_bytes(original_receipt)

    missing_analysis = json.loads(index_path.read_text(encoding="utf-8"))
    missing_analysis["analyses"] = {}
    missing_analysis_path = tmp_path / "missing-analysis-index.json"
    missing_analysis_path.write_text(json.dumps(missing_analysis), encoding="utf-8")
    with pytest.raises(GatedEvaluationError, match="analysis index must cover"):
        write_run_record(
            selection, missing_analysis_path, tmp_path / "invalid-run.json"
        )

    write_run_record(selection, index_path, run_record)

    with pytest.raises(
        GatedEvaluationError, match="production evaluation requires a local SAM 2"
    ):
        evaluate_gated_test(
            selection_path=selection,
            run_record_path=run_record,
            target_dir=inputs.target_dir,
            image_root=inputs.image_root,
            output_dir=tmp_path / "without-sam",
        )

    output = evaluate_gated_test(
        selection_path=selection,
        run_record_path=run_record,
        target_dir=inputs.target_dir,
        image_root=inputs.image_root,
        output_dir=tmp_path / "evaluation",
        mask_checkpoint=tmp_path / "local-sam-checkpoint",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    specs = selection_record["selection"]["model_specs"]
    cue_id = next(fit_id for fit_id, spec in specs.items() if spec["kind"] == "cue")
    final_id = selection_record["selection"]["final_fit_ids"][0]
    assert report["completed_analysis_ids"] == selection_record["analysis_ids"]
    assert report["prediction_views"][cue_id] in {"pad-fraction", "aspect-ratio"}
    assert report["prediction_views"][final_id] == "raw-test"
    assert report["preprocessing"]["resize"] == "resize-center-crop"
    assert report["mask_pipeline"]["model"] == "excess-green-otsu-largest-component"
    assert report["model_provenance"][final_id]["training_view"] == "raw-training"
    assert report["leaf_prediction_view"][final_id] == "leaf-only"
    assert (
        report["leaf_only"]["fit_scores"][final_id]["n"]
        == report["leaf_only"]["sample_size"]
    )
    assert set(report["evaluated_view_sha256"]) == {
        "raw-test",
        "background-only",
        "leaf-only",
        "border-frame",
    }
    border_result = next(iter(report["border"].values()))
    assert len(border_result["fit_ids"]) == 3
    assert "seeds within primary split" in border_result["bootstrap_method"]
    if border_result["headline"] is None:
        assert border_result["symptom_survival"] is None
        assert border_result["symptom_survival_accuracy_ci_95"] is None
        assert border_result["symptom_survival_macro_f1_ci_95"] is None
        assert border_result["headline_omission_reason"] in {
            "mask audit failed",
            "a class has fewer than 50 retained test images",
        }
    else:
        assert border_result["symptom_survival"] == border_result["headline"]
    assert len(report["frame_leaf_shares"]) == report["test_images"]
    assert all(0 <= row["leaf_pixel_share"] <= 1 for row in report["frame_leaf_shares"])
    assert border_result["rule"] in {
        "border information detected",
        "no detectable border information at this frame width",
    }


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
            "freeze-matrix",
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
    record = json.loads(selection_path.read_text(encoding="utf-8"))["record"]
    selection = record["selection"]
    choices = record["validation_selected_hyperparameters"]
    background = selection["background_only"]
    assert background["seeds"] == [42, 1337, 7, 21, 84]
    assert len(background["fit_ids"]) == 495
    assert background["cnn_route"] == "mobilenetv2_100.ra_in1k (ImageNet)"
    background_spec = selection["model_specs"][background["fit_ids"][0]]
    assert background_spec["training_view"] == background_spec["input_view"]
    assert background_spec["selected_parameters"] == choices["classical/tune/A/gnb-raw"]
    background_cnn_id = next(
        fit_id
        for fit_id in background["fit_ids"]
        if selection["model_specs"][fit_id]["kind"] == "cnn"
    )
    background_cnn_spec = selection["model_specs"][background_cnn_id]
    assert (
        background_cnn_spec["selected_learning_rate"]
        == choices["cnn/tune/mobilenet_v2_direct/pool-A"]
    )
    assert background["leaf_mask"].endswith("+sam2")
    assert len(selection["pools"]) == 3
    assert len(selection["policies"]) == 3
    assert selection["primary_split"] == 0
    leaf = selection["leaf_only"]
    assert leaf["sample_size"] == 2_000
    assert leaf["fit_ids"] == selection["final_fit_ids"]
    assert selection["border"]["frame_width_px_start"] == 32
    assert selection["border"]["leaf_share_cutoff_start"] == 0.05
    assert selection["border"]["null_permutations"] == 1_000
    assert selection["border"]["classical_configurations_per_estimator"] == 9
    border_spec = selection["model_specs"][selection["border"]["fit_ids"][0]]
    assert border_spec["training_view"] == "border-frame"
    assert border_spec["input_view"] == "border-frame"
    assert border_spec["selected_parameters"] == border_spec["parameters"]
    assert len(selection["border"]["fit_ids"]) == 2_511


def test_matrix_freeze_requires_recorded_issue_114_outcomes(
    inputs: Inputs, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="#114 validation outcomes are required"):
        main(
            [
                "evaluate",
                "freeze-matrix",
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


def test_matrix_freeze_requires_a_validation_choice_for_each_tuning_route(
    inputs: Inputs, tmp_path: Path
) -> None:
    benchmark = _mini_benchmark(tmp_path)
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes)
    recorded = json.loads(outcomes.read_text(encoding="utf-8"))
    del recorded["selected_hyperparameters"]["classical/tune/A/lr-crafted"]
    outcomes.write_text(json.dumps(recorded), encoding="utf-8")
    selection = tmp_path / "selection.json"
    with pytest.raises(
        SystemExit, match="validation outcomes omit selected hyperparameters"
    ):
        main(
            [
                "evaluate",
                "freeze-matrix",
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
    assert not selection.exists()


def test_matrix_freeze_rejects_border_leaf_share_cutoff_above_five_percent(
    inputs: Inputs, tmp_path: Path
) -> None:
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes, border_leaf_share_cutoff=0.06)
    selection = tmp_path / "selection.json"
    with pytest.raises(SystemExit, match="at most 5%"):
        main(
            [
                "evaluate",
                "freeze-matrix",
                "--validation-outcomes",
                str(outcomes),
                "--benchmark",
                str(_mini_benchmark(tmp_path)),
                "--target-dir",
                str(inputs.target_dir),
                "--output",
                str(selection),
            ]
        )
    assert not selection.exists()


def test_failed_mask_audit_blocks_mask_dependent_report_scores(
    inputs: Inputs, tmp_path: Path
) -> None:
    benchmark = _mini_benchmark(tmp_path)
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes, mask_audit="failed")
    selection = tmp_path / "selection.json"
    run_record = tmp_path / "run.json"
    output = tmp_path / "evaluation"
    main(
        [
            "evaluate",
            "freeze-matrix",
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
    main(
        [
            "evaluate",
            "write-fixture-run-record",
            "--selection",
            str(selection),
            "--output",
            str(run_record),
        ]
    )
    main(
        [
            "evaluate",
            "score-test",
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
    )
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert report["background_only"] == {
        "fit_scores": None,
        "status": "blocked by mask audit",
    }
    assert report["leaf_only"]["sample_size"] == 0
    assert report["leaf_only"]["fit_scores"] is None
    assert not report["class_by_scene"]
    for row in report["border"].values():
        assert row["headline"] is None
        assert row["symptom_survival"] is None
        assert row["symptom_survival_accuracy_ci_95"] is None
        assert row["symptom_survival_macro_f1_ci_95"] is None
        assert row["headline_omission_reason"] == "mask audit failed"
    assert len(report["frame_leaf_shares"]) == report["test_images"]


def test_symptom_survival_is_withheld_below_class_support_gate(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jute_disease.data as data

    class AllLeafMaskBackend:
        model = "synthetic-all-leaf"
        version = "fixture"

        def predict(self, image: Image.Image) -> SimpleNamespace:
            return SimpleNamespace(
                mask=np.ones((image.height, image.width), dtype=np.bool_)
            )

    monkeypatch.setattr(data, "ColorThresholdBackend", AllLeafMaskBackend)
    benchmark = _mini_benchmark(tmp_path)
    outcomes = tmp_path / "validation-114.json"
    _outcomes(outcomes)
    selection = tmp_path / "selection.json"
    run_record = tmp_path / "run.json"
    output = tmp_path / "evaluation"
    main(
        [
            "evaluate",
            "freeze-matrix",
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
    main(
        [
            "evaluate",
            "write-fixture-run-record",
            "--selection",
            str(selection),
            "--output",
            str(run_record),
        ]
    )
    main(
        [
            "evaluate",
            "score-test",
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
    )

    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert all(row["leaf_pixel_share"] == 1 for row in report["frame_leaf_shares"])
    for row in report["border"].values():
        assert any(count < 50 for count in row["retained_class_counts"].values())
        assert all(count == 0 for count in row["retained_class_counts"].values())
        assert all(count == 0 for count in row["retained_scene_counts"].values())
        assert row["unrestricted"]
        assert row["headline"] is None
        assert row["symptom_survival"] is None
        assert row["symptom_survival_accuracy_ci_95"] is None
        assert row["symptom_survival_macro_f1_ci_95"] is None
        assert row["headline_omission_reason"] == (
            "a class has fewer than 50 retained test images"
        )

"""Synthetic coverage for the public embedding and Grad-CAM diagnostics APIs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from jute_disease.evaluation import (
    Authorization,
    BoundOutput,
    FreezeRecordError,
    RunEntry,
    aggregate_leaf_scores,
    class_covering_sample,
    diagnostic_configuration,
    embedding_coordinates,
    evaluate_image_diagnostics,
    gradcam_maps,
    leaf_attention_score,
    leaf_score_gate,
    letterbox_leaf_mask,
    parameter_randomization_correlations,
    require_analysis,
    sanity_check_results,
    select_gradcam_sample,
    select_median_validation_run,
    validate_primary_policy_runs,
    write_diagnostics_export,
)


def _run(
    index: int,
    *,
    score: float = 0.5,
    split: int = 0,
    assignment_digest: str | None = None,
) -> RunEntry:
    return RunEntry(
        fit_id=f"cnn/mobilenet_v2_direct/pool-A/view-1/policy-X/split-{split}/seed-{index}",
        system="mobilenet_v2_direct",
        seed=(42, 1337, 7, 21, 84)[index],
        variant="full",
        predictions=f"predictions-{index}.json",
        predictions_sha256="0" * 64,
        checkpoint=f"checkpoints/{index}.ckpt",
        checkpoint_sha256="1" * 64,
        training_receipt_sha256="2" * 64,
        validation_macro_f1=score,
        pool="A",
        view=1,
        policy="near-duplicate",
        split=split,
        seed_index=index,
        assignment_table_sha256=assignment_digest,
    )


def test_analysis_gate_checks_both_analysis_and_run_membership() -> None:
    run = _run(0)
    authorization = Authorization(
        "record",
        False,
        (run,),
        {"analysis_ids": ["gradcam-test-pass"]},
    )
    with pytest.raises(FreezeRecordError, match="embeddings-test-pass"):
        require_analysis(authorization, "embeddings-test-pass", [run.fit_id])

    authorization = Authorization(
        "record",
        False,
        (run,),
        {"analysis_ids": ["embeddings-test-pass"]},
    )
    with pytest.raises(FreezeRecordError, match="not authorized"):
        require_analysis(authorization, "embeddings-test-pass", ["unknown-fit"])


def _absent_output(kind: str, tmp_path: Path) -> BoundOutput:
    """A bound output nobody will read: the entry point must refuse before it does."""
    return BoundOutput(kind, tmp_path / f"absent-{kind}.json", "0" * 64, {})


def test_test_image_entry_point_refuses_before_manifest_or_image_access(
    tmp_path: Path,
) -> None:
    authorization = Authorization("record", False, (), {"analysis_ids": []})
    with pytest.raises(FreezeRecordError, match="embeddings-test-pass"):
        evaluate_image_diagnostics(
            tmp_path / "absent-target",
            authorization,
            tmp_path / "absent-runs",
            tmp_path / "absent-images",
            tmp_path / "output",
            frozen_backbone_checkpoint=tmp_path / "absent-imagenet.ckpt",
            assignment_table_path=tmp_path / "absent-assignments.json",
            scene_tags=_absent_output("scene_tags", tmp_path),
            sanity_checks=_absent_output("sanity_checks", tmp_path),
        )
    assert not (tmp_path / "absent-images").exists()


def test_embeddings_are_seeded_aligned_and_record_all_frozen_settings() -> None:
    rng = np.random.default_rng(18)
    ids = [f"synthetic-{index:03d}" for index in range(52)]
    features = rng.normal(size=(len(ids), 50))
    first = embedding_coordinates(features, ids)
    second = embedding_coordinates(features, ids)

    assert first["image_ids"] == ids
    assert first["pca"]["components"] == 50
    assert set(first["coordinates"]) == {
        "tsne_perplexity_30",
        "tsne_perplexity_50",
        "umap_n_neighbors_15",
        "umap_n_neighbors_50",
    }
    for name, coordinates in first["coordinates"].items():
        assert np.asarray(coordinates).shape == (len(ids), 2)
        np.testing.assert_allclose(coordinates, second["coordinates"][name])
    assert first["settings"]["seed"] == 0
    assert first["settings"]["pca"]["n_components"] == 50
    assert first["settings"]["tsne"]["perplexities"] == [30, 50]
    assert first["settings"]["umap"]["n_neighbors"] == [15, 50]
    assert first["settings"]["software_versions"]["scikit-learn"]
    assert first["settings"]["software_versions"]["umap-learn"]
    assert (
        first["claim_boundary"]
        == "Do not interpret cluster size or inter-point distance."
    )
    assert diagnostic_configuration()["gradcam"]["sample_seed"] == 0


def test_embeddings_reject_missing_support_and_unaligned_ids() -> None:
    with pytest.raises(ValueError, match="at least 51"):
        embedding_coordinates(np.zeros((50, 50)), [str(i) for i in range(50)])
    with pytest.raises(ValueError, match="aligned"):
        embedding_coordinates(np.zeros((51, 50)), [str(i) for i in range(50)])


def test_policy_sample_is_reproducible_and_keeps_disagreement_ids() -> None:
    ids = [f"id-{index}" for index in range(6)]
    truth = ["a", "a", "a", "b", "b", "b"]
    predictions = ["a", "a", "b", "b", "a", "b"]
    agreed = dict(zip(ids, truth, strict=True))
    policies = {
        "image-random": {**agreed, "id-0": "a", "id-2": "b"},
        "exact-copy": {**agreed, "id-0": "b", "id-2": "a"},
        "near-duplicate": {**agreed, "id-0": "a", "id-2": "b"},
    }
    first = select_gradcam_sample(
        ids, truth, predictions, ("a", "b"), per_class=1, policy_predictions=policies
    )
    second = select_gradcam_sample(
        ids, truth, predictions, ("a", "b"), per_class=1, policy_predictions=policies
    )
    assert first == second
    assert first["selected"]["disagreement"] == ["id-0", "id-2"]
    assert len(first["selected"]["correct"]) == 2
    assert len(first["selected"]["error"]) == 2
    assert first["policy_disagreement_count"] == 2
    assert first["all_policy_disagreements_included"] is True
    incomplete = {**policies, "exact-copy": {"id-0": "b"}}
    with pytest.raises(ValueError, match="miss test images"):
        select_gradcam_sample(
            ids,
            truth,
            predictions,
            ("a", "b"),
            per_class=1,
            policy_predictions=incomplete,
        )


def test_primary_policy_assignment_table_is_receipt_bound(tmp_path: Path) -> None:
    assignments = [
        {"split": 0, "seed": index, "checkpoint": index % 3} for index in range(5)
    ]
    path = tmp_path / "assignments.json"
    path.write_text(json.dumps(assignments), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    runs = [_run(index, assignment_digest=digest) for index in range(5)]

    result = validate_primary_policy_runs(path, runs)
    assert result["primary_split"] == 0
    assert result["policy"] == "near-duplicate"
    assert result["sha256"] == digest
    assert result["source_checkpoint_by_seed"] == {
        str(run.seed): run.seed_index % 3 for run in runs
    }

    with pytest.raises(FreezeRecordError, match="digest"):
        validate_primary_policy_runs(path, [_run(index) for index in range(5)])
    with pytest.raises(FreezeRecordError, match="primary split"):
        validate_primary_policy_runs(path, [_run(index, split=1) for index in range(5)])


def test_median_validation_run_uses_receipt_scores_and_seed_tie_break() -> None:
    runs = [
        _run(index, score=score)
        for index, score in enumerate((0.4, 0.6, 0.6, 0.6, 0.8))
    ]
    selected = select_median_validation_run(runs)
    assert selected.validation_macro_f1 == 0.6
    assert selected.seed == min(runs[1].seed, runs[2].seed, runs[3].seed)
    with pytest.raises(ValueError, match="receipt-bound"):
        select_median_validation_run(
            [_run(0, score=0.5), replace(_run(1), validation_macro_f1=None)]
        )


def test_cam_score_ignores_padding_and_counts_zero_cam_as_undefined() -> None:
    content = np.zeros((4, 4), dtype=bool)
    content[1:3, :] = True
    leaf = np.zeros((4, 4), dtype=bool)
    leaf[1, :2] = True
    cam = np.zeros((4, 4), dtype=float)
    cam[0, :] = 100.0
    cam[1, 0] = 1.0
    assert leaf_attention_score(cam, leaf, content) == pytest.approx(0.75)
    assert leaf_attention_score(np.zeros_like(cam), leaf, content) is None
    assert letterbox_leaf_mask(np.ones((2, 4), dtype=bool), content).sum() == 8

    summary = aggregate_leaf_scores(
        [
            {"class": "a", "scene": "field", "score": 0.2},
            {"class": "a", "scene": "field", "score": None},
            {"class": "b", "scene": "clean-background", "score": -0.1},
        ]
    )
    assert summary["class_scene"][0]["undefined_all_zero_cam"] == 1
    assert summary["by_scene"]["field"]["median"] == pytest.approx(0.2)


def test_cam_and_sanity_check_public_apis_are_deterministic() -> None:
    class TinyCamModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 1, bias=False)
            self.classifier = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.classifier.weight.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.classifier(torch.relu(self.conv(images)).mean(dim=(2, 3)))

    model = TinyCamModel()
    images = torch.rand((2, 3, 4, 4))
    first = gradcam_maps(model, images, (0, 1), model.conv)
    second = gradcam_maps(model, images, (0, 1), model.conv)
    assert first.shape == (2, 4, 4)
    assert np.all(first >= 0)
    np.testing.assert_allclose(first, second)

    correlations = parameter_randomization_correlations(
        first, {"randomized": np.flip(first, axis=-1).copy()}
    )
    assert len(correlations["randomized"]) == 2


def test_sanity_check_results_refuse_unaligned_inputs_and_unhashed_checkpoints() -> (
    None
):
    model = nn.Linear(2, 2)
    images = torch.zeros((2, 3, 4, 4))
    with pytest.raises(ValueError, match="align"):
        sanity_check_results(
            model,
            images,
            [0],
            [0, 1],
            checkpoint_sha256="a" * 64,
            image_ids=["a", "b"],
        )
    with pytest.raises(ValueError, match="SHA-256"):
        sanity_check_results(
            model,
            images,
            [0, 1],
            [1, 0],
            checkpoint_sha256="not-a-digest",
            image_ids=["a", "b"],
        )


class _TinyBackbone(nn.Module):
    """A timm-shaped backbone with BatchNorm, nested blocks and a conv head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4)),
            nn.Sequential(nn.Conv2d(4, 4, 3, padding=1)),
        )
        self.conv_head = nn.Conv2d(4, 6, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(torch.relu(self.bn1(self.conv_stem(images))))
        return torch.relu(self.conv_head(hidden)).mean(dim=(2, 3))


class _TinyExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _TinyExtractor()
        self.classifier = nn.Linear(6, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(images))


def _sanity_inputs() -> tuple[nn.Module, torch.Tensor, list[int], list[int]]:
    torch.manual_seed(5)
    model = _TinyClassifier().eval()
    images = torch.rand((6, 3, 8, 8), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        targets = [int(row.argmax()) for row in model(images)]
    return model, images, targets, [0, 1, 2, 0, 1, 2]


def _run_sanity(
    inputs: tuple[nn.Module, torch.Tensor, list[int], list[int]],
) -> dict[str, object]:
    model, images, targets, labels = inputs
    return sanity_check_results(
        model,
        images,
        targets,
        labels,
        checkpoint_sha256="a" * 64,
        image_ids=[f"img-{index}" for index in range(len(targets))],
    )


def test_sanity_checks_give_identical_results_for_identical_inputs() -> None:
    inputs = _sanity_inputs()
    assert _run_sanity(inputs) == _run_sanity(inputs)


def test_sanity_checks_leave_the_global_random_state_and_model_untouched() -> None:
    inputs = _sanity_inputs()
    before = {key: value.clone() for key, value in inputs[0].state_dict().items()}
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _run_sanity(inputs)
    assert torch.equal(torch.rand(3), expected)
    for key, value in inputs[0].state_dict().items():
        assert torch.equal(value, before[key])


def test_a_constant_randomized_map_counts_as_changed_not_undefined() -> None:
    reference = np.random.default_rng(0).random((2, 4, 4))
    changed = parameter_randomization_correlations(
        reference, {"stage": np.zeros_like(reference)}
    )
    assert changed["stage"] == [0.0, 0.0]
    undefined = parameter_randomization_correlations(
        np.zeros_like(reference), {"stage": reference}
    )
    assert undefined["stage"] == [None, None]


def test_an_inverted_map_is_as_correlated_as_an_identical_one() -> None:
    reference = np.random.default_rng(0).random((2, 4, 4))
    correlations = parameter_randomization_correlations(
        reference, {"inverted": -reference, "same": reference.copy()}
    )
    assert correlations["inverted"] == pytest.approx([1.0, 1.0])
    assert correlations["same"] == pytest.approx([1.0, 1.0])


def test_parameter_randomization_reinitialises_each_layer_by_default() -> None:
    result = _run_sanity(_sanity_inputs())
    rule = result["pass_rule"]
    assert rule["parameter_randomization_initialisation"] == (
        "each layer's default reset_parameters, seeded"
    )
    medians = result["parameter_randomization"]["evidence"]["stage_median_correlation"]
    assert len(medians) == 6
    assert len(set(medians.values())) > 2


def test_label_randomization_is_reported_against_a_true_label_control() -> None:
    result = _run_sanity(_sanity_inputs())
    label = result["label_randomization"]
    assert label["status"] == "descriptive"
    evidence = label["evidence"]
    assert evidence["gates_leaf_score"] is False
    assert evidence["median_correlation"] is not None
    assert evidence["control_median_correlation"] is not None
    assert evidence["refit_steps"] == 100
    assert evidence["control"] == "same seed, steps and sample, refit on true labels"
    assert result["pass_rule"]["label_randomization_gates_leaf_score"] is False


def test_pass_rule_records_the_prespecified_heuristic_thresholds() -> None:
    rule = _run_sanity(_sanity_inputs())["pass_rule"]
    assert rule["fixed_in_code"] is True
    assert rule["statistic"] == "median absolute Spearman correlation"
    assert rule["max_median_correlation"] == 0.5
    assert rule["min_defined_share"] == 0.5
    assert rule["sample_size"] == 60
    assert rule["prespecified_heuristic"] is True
    assert rule["significance_test"] is False
    assert rule["applies_to"] == "the sampled images only"


def test_sanity_checks_refuse_a_single_class_sample() -> None:
    model, images, targets, _ = _sanity_inputs()
    with pytest.raises(ValueError, match="two classes"):
        sanity_check_results(
            model,
            images,
            targets,
            [1] * len(targets),
            checkpoint_sha256="a" * 64,
            image_ids=[f"img-{index}" for index in range(len(targets))],
        )


def test_class_covering_sample_keeps_every_class_and_is_deterministic() -> None:
    ids = [f"img-{index:03d}" for index in range(100)]
    labels = ["a"] * 90 + ["b"] * 8 + ["c"] * 2
    sample = class_covering_sample(ids, labels, 12, seed=0)
    assert len(sample) == 12
    assert sample == class_covering_sample(ids, labels, 12, seed=0)
    assert {labels[ids.index(image_id)] for image_id in sample} == {"a", "b", "c"}
    assert class_covering_sample(ids, labels, 500, seed=0) == tuple(sorted(ids))
    capped = class_covering_sample(ids, labels, 2, seed=0)
    assert len(capped) == 2


def test_class_covering_sample_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="invalid"):
        class_covering_sample(["a"], ["x", "y"], 1, seed=0)
    with pytest.raises(ValueError, match="invalid"):
        class_covering_sample(["a"], ["x"], 0, seed=0)


def test_leaf_attention_gate_withholds_unless_both_gating_checks_pass() -> None:
    assert not leaf_score_gate(parameter_randomization=False, mask_audit_passed=True)[
        "reported"
    ]
    assert not leaf_score_gate(parameter_randomization=True, mask_audit_passed=False)[
        "reported"
    ]
    gate = leaf_score_gate(parameter_randomization=True, mask_audit_passed=True)
    assert gate["reported"]
    assert "label_randomization" not in gate["checks"]


def test_export_records_command_configuration_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-receipt.json"
    source.write_text('{"synthetic":true}', encoding="utf-8")
    payload = {
        "embeddings": {
            "synthetic": {
                "image_ids": ["a", "b"],
                "coordinates": {"tsne_perplexity_30": [[0.1, 0.2], [0.3, 0.4]]},
            }
        }
    }
    output = write_diagnostics_export(
        tmp_path / "export",
        payload,
        generating_command=("jute-disease", "evaluate", "diagnostics"),
        configuration={"seed": 0},
        source_artifacts=(source,),
    )
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["generating_command"][-1] == "diagnostics"
    assert exported["configuration"] == {"seed": 0}
    assert (
        exported["source_artifacts"][0]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (output.parent / "synthetic-tsne_perplexity_30.csv").is_file()

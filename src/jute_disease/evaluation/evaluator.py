"""The gated evaluator: the only code that opens the test partition.

Every entry point takes an :class:`Authorization` from the freeze-record gate, so
no test label or prediction is read without a verified selection and run record.
One call evaluates one frozen target manifest; looping over pools and policies
belongs to the caller.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from jute_disease.data import (
    POLICIES,
    SCENE_TAGS,
    ColorThresholdBackend,
    ManifestDataset,
    MaskBackend,
    create_pipeline,
)
from jute_disease.data.manifest import (
    load_complete_manifest,
    load_protected_test_manifest,
)
from jute_disease.evaluation.breakdowns import (
    LEAF_SAMPLE_SIZE,
    class_scene_cells,
    scene_summaries,
    stratified_sample,
)
from jute_disease.evaluation.diagnostics import (
    aggregate_leaf_scores,
    diagnostic_configuration,
    embedding_coordinates,
    gradcam_maps,
    leaf_attention_score,
    leaf_score_gate,
    letterbox_leaf_mask,
    penultimate_features,
    require_analysis,
    sanity_check_record,
    select_gradcam_sample,
    select_median_validation_run,
    validate_primary_policy_runs,
    write_diagnostics_export,
)
from jute_disease.evaluation.freeze_record import (
    Authorization,
    FreezeRecordError,
    RunEntry,
)
from jute_disease.evaluation.probes import (
    background_probe,
    border_probe,
    subset,
)
from jute_disease.evaluation.scores import PredictionSet, evaluate_predictions


def leaf_sample_ids(
    target_dir: Path | str,
    freeze_record: Mapping[str, Any],
    size: int = LEAF_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[str, ...]:
    """The stratified test sample on which leaf-only checkpoints are evaluated.

    ``freeze_record`` is a record returned by ``verify_freeze_record``; the sample
    must be known before leaf-only checkpoints are scored.
    """
    del freeze_record  # presence proves the gate was passed
    entries = load_protected_test_manifest(target_dir).partition("test")
    return stratified_sample(
        [e.image_id for e in entries], [e.label for e in entries], size, seed
    )


def _load(entry: RunEntry, base_dir: Path, ids: tuple[str, ...]) -> tuple[str, ...]:
    raw = (base_dir / entry.predictions).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.predictions_sha256:
        raise FreezeRecordError(
            f"predictions changed since the run record: {entry.fit_id}"
        )
    payload = json.loads(raw)
    if tuple(payload.get("image_ids", ())) != ids:
        raise FreezeRecordError(
            f"predictions for {entry.fit_id} do not match test images"
        )
    labels = tuple(payload.get("predictions", ()))
    if len(labels) != len(ids):
        raise FreezeRecordError(f"predictions for {entry.fit_id} are incomplete")
    return labels


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_test(
    target_dir: Path | str,
    authorization: Authorization,
    base_dir: Path | str,
    *,
    scenes: Mapping[str, str] | None = None,
    frame_leaf_share: Mapping[str, float] | None = None,
    sample_size: int = LEAF_SAMPLE_SIZE,
    resamples: int = 10_000,
    permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score the authorized fits on the test partition and run every analysis."""
    manifest = load_complete_manifest(target_dir)
    entries = tuple(sorted(manifest.partition("test"), key=lambda e: e.image_id))
    if not entries:
        raise FreezeRecordError("target manifest has no test images")
    ids = tuple(e.image_id for e in entries)
    classes = tuple(manifest.labels)
    sample = set(stratified_sample(ids, [e.label for e in entries], sample_size, seed))
    sample_ids = tuple(i for i in ids if i in sample)
    base = Path(base_dir)

    runs: dict[tuple[str, str], dict[int, tuple[str, ...]]] = defaultdict(dict)
    for run in authorization.runs:
        expected = sample_ids if run.variant == "leaf_only" else ids
        group = runs[(run.system, run.variant)]
        if run.seed in group:
            raise FreezeRecordError(
                f"more than one run for {run.system} {run.variant} seed {run.seed}; "
                "restrict the call to one pool, view, policy and split with --fit-id"
            )
        group[run.seed] = _load(run, base, expected)

    def build(system: str, variant: str) -> PredictionSet:
        chosen = sample_ids if variant == "leaf_only" else ids
        by_id = {e.image_id: e for e in entries}
        return PredictionSet(
            classes,
            chosen,
            tuple(by_id[i].label for i in chosen),
            tuple(by_id[i].group for i in chosen),
            tuple(by_id[i].source for i in chosen),
            runs[(system, variant)],
        )

    report: dict[str, Any] = {
        "record_hash": authorization.record_hash,
        "post_hoc": authorization.post_hoc,
        "test_images": len(ids),
        "systems": {},
    }
    if authorization.post_hoc:
        report["note"] = "freeze record created after a test read; results are post-hoc"
    scenes = dict(scenes or {})
    full_systems = sorted(s for s, v in runs if v == "full")
    for system in full_systems:
        full = build(system, "full")
        result: dict[str, Any] = {"evaluation": _plain(evaluate_predictions(full))}
        if scenes:
            result["class_scene_cells"] = _plain(class_scene_cells(full, scenes))
            result["scene_summaries"] = _plain(scene_summaries(full, scenes))
        report["systems"][system] = result

    leaf_required = {run.fit_id for run in authorization.runs if run.variant == "full"}
    leaf_covered = {run.of for run in authorization.runs if run.variant == "leaf_only"}
    uncovered = sorted(leaf_required - leaf_covered)
    if uncovered:
        raise FreezeRecordError(
            f"leaf-only evaluation misses final checkpoints: {uncovered}"
        )
    report["leaf_sample"] = {"size": len(sample_ids), "image_ids": list(sample_ids)}
    report["leaf_only_coverage"] = {
        "complete": bool(leaf_required) and not uncovered,
        "missing": uncovered,
    }
    for system in sorted(s for s, v in runs if v == "leaf_only"):
        leaf = build(system, "leaf_only")
        entry = report["systems"].setdefault(system, {})
        entry["leaf_only"] = _plain(evaluate_predictions(leaf))
        if (system, "full") in runs:
            entry["full_on_leaf_sample"] = _plain(
                evaluate_predictions(subset(build(system, "full"), sample_ids))
            )

    options = {"resamples": resamples, "permutations": permutations, "seed": seed}
    for system in sorted(s for s, v in runs if v == "background_only"):
        result = background_probe(build(system, "background_only"), **options)
        report["systems"].setdefault(system, {})["background_probe"] = _plain(result)

    pad_systems = sorted(s for s, v in runs if v == "pad_cue")
    pad = build(pad_systems[0], "pad_cue") if pad_systems else None
    for system in sorted(s for s, v in runs if v == "border"):
        if frame_leaf_share is None:
            raise FreezeRecordError("border probe needs per-image frame leaf shares")
        body = authorization.body
        border = border_probe(
            build(system, "border"),
            frame_leaf_share,
            scenes,
            mask_audit_passed=bool(body["mask_audit"]["passed"]),
            leaf_cutoff=float(body["border_probe"]["final"]["leaf_cutoff"]),
            width_px=int(body["border_probe"]["final"]["width_px"]),
            pad_cue=pad,
            **options,
        )
        report["systems"].setdefault(system, {})["border_probe"] = _plain(border)
    return report


def evaluate_image_diagnostics(
    target_dir: Path | str,
    authorization: Authorization,
    base_dir: Path | str,
    image_root: Path | str,
    output_dir: Path | str,
    *,
    frozen_backbone_checkpoint: Path | str,
    assignment_table_path: Path | str,
    model_name: str = "mobilenetv2_100.ra_in1k",
    scenes: Mapping[str, str] | None = None,
    policy_predictions: Mapping[str, Mapping[str, str]] | None = None,
    mask_backend: MaskBackend | None = None,
    sanity_checks: Mapping[str, Any] | None = None,
    generating_command: Sequence[str] = (),
    source_artifacts: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Read authorized test images and export embeddings and Grad-CAM results.

    This is the diagnostics image-reading entry point. It checks both analysis
    IDs, run membership and checkpoint receipts before opening the test partition.
    """
    fit_ids = [run.fit_id for run in authorization.runs]
    require_analysis(authorization, "embeddings-test-pass", fit_ids)
    require_analysis(authorization, "gradcam-test-pass", fit_ids)
    assignment = validate_primary_policy_runs(assignment_table_path, authorization.runs)

    manifest = load_complete_manifest(target_dir)
    entries = tuple(sorted(manifest.partition("test"), key=lambda item: item.image_id))
    image_ids = tuple(entry.image_id for entry in entries)
    if not image_ids:
        raise FreezeRecordError("target manifest has no test images")
    scene_map = dict(scenes or {})
    missing_scenes = sorted(set(image_ids) - set(scene_map))
    invalid_scenes = sorted(
        (image_id, scene_map.get(image_id))
        for image_id in image_ids
        if scene_map.get(image_id) not in SCENE_TAGS
    )
    if missing_scenes or invalid_scenes:
        raise FreezeRecordError(
            "diagnostics need a frozen scene tag for every test image; "
            f"missing={missing_scenes[:5]}, invalid={invalid_scenes[:5]}"
        )
    if policy_predictions is None or set(policy_predictions) != set(POLICIES):
        raise FreezeRecordError(
            "diagnostics need predictions for image-random, exact-copy and "
            "near-duplicate policies"
        )
    dataset = ManifestDataset(manifest, "test", Path(image_root))
    if any(
        run.validation_macro_f1 is None
        or run.checkpoint is None
        or run.checkpoint_sha256 is None
        or run.training_receipt_sha256 is None
        for run in authorization.runs
    ):
        raise FreezeRecordError(
            "diagnostics require receipt-bound validation macro-F1 and "
            "checkpoint artifacts"
        )

    selected = select_median_validation_run(authorization.runs)
    base = Path(base_dir).resolve()
    checkpoint_path = (base / str(selected.checkpoint)).resolve()
    if not checkpoint_path.is_relative_to(base) or not checkpoint_path.is_file():
        raise FreezeRecordError(
            "authorized checkpoint path is missing or escapes the run store"
        )
    if (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        != selected.checkpoint_sha256
    ):
        raise FreezeRecordError(
            "authorized checkpoint bytes differ from the run record"
        )
    receipt_path = base.joinpath(*selected.fit_id.split("/")) / "receipt.json"
    if (
        not receipt_path.is_file()
        or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        != selected.training_receipt_sha256
    ):
        raise FreezeRecordError("training receipt changed since the run record")

    from jute_disease.models.dl import (
        Classifier,
        TimmBackbone,
        load_backbone_checkpoint,
        read_checkpoint_state,
    )

    model = Classifier(TimmBackbone(model_name, pretrained=False), len(manifest.labels))
    model.load_state_dict(read_checkpoint_state(checkpoint_path), strict=True)
    model.eval()
    mask_model = mask_backend or ColorThresholdBackend()
    class_to_idx = {label: index for index, label in enumerate(manifest.labels)}
    image_pipeline = create_pipeline(is_dl=False, include_mask=True)
    tensor_pipeline = create_pipeline(is_dl=True)
    prediction_labels = _load(selected, base, image_ids)
    sampled = select_gradcam_sample(
        image_ids,
        [entry.label for entry in entries],
        prediction_labels,
        manifest.labels,
        policy_predictions=policy_predictions,
    )
    figure_ids = {
        image_id
        for selected_ids in sampled["selected"].values()
        for image_id in selected_ids
    }
    score_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    frozen_rows: list[np.ndarray] = []
    figure_rows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    labels_by_id = {entry.image_id: entry for entry in entries}
    recorded_predictions = dict(zip(image_ids, prediction_labels, strict=True))

    # ManifestDataset verifies the frozen bytes before its first image opens.
    for image_id in image_ids:
        entry = labels_by_id[image_id]
        with Image.open(dataset.image_root / entry.path) as source:
            original = source.convert("RGB")
        transformed = image_pipeline(original)
        content_mask = transformed.valid_mask
        letterboxed = transformed.pixels
        tensor = tensor_pipeline(original).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            predicted_index = int(logits.argmax(dim=1).item())
            features = penultimate_features(model, tensor)
        predicted_label = manifest.labels[predicted_index]
        feature_rows.append(features.detach().cpu().numpy().reshape(-1))
        mask_prediction = mask_model.predict(original)
        leaf_mask = letterbox_leaf_mask(mask_prediction.mask, content_mask)
        true_index = class_to_idx[entry.label]
        targets = (
            (true_index,)
            if true_index == predicted_index
            else (true_index, predicted_index)
        )
        cams = gradcam_maps(model, tensor.expand(len(targets), -1, -1, -1), targets)
        for target_index, cam in zip(targets, cams, strict=True):
            score_rows.append(
                {
                    "image_id": image_id,
                    "class": manifest.labels[target_index],
                    "scene": scene_map.get(image_id, "unknown"),
                    "true_class": entry.label,
                    "target": "true" if target_index == true_index else "predicted",
                    "target_class": manifest.labels[target_index],
                    "score": leaf_attention_score(cam, leaf_mask, content_mask),
                }
            )
            if image_id in figure_ids:
                figure_rows[(image_id, target_index)] = (letterboxed, cam, leaf_mask)
        if predicted_label != recorded_predictions[image_id]:
            raise FreezeRecordError(
                "selected checkpoint predictions do not match its authorized "
                f"run for {image_id}"
            )

    frozen_path = Path(frozen_backbone_checkpoint).resolve()
    if not frozen_path.is_file():
        raise FreezeRecordError("local frozen ImageNet backbone checkpoint is missing")
    frozen_backbone = TimmBackbone(model_name, pretrained=False)
    load_backbone_checkpoint(frozen_backbone.backbone, frozen_path)
    frozen_backbone.eval()
    for image_id in image_ids:
        entry = labels_by_id[image_id]
        with Image.open(dataset.image_root / entry.path) as source:
            tensor = tensor_pipeline(source.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            frozen_rows.append(
                frozen_backbone(tensor).detach().cpu().numpy().reshape(-1)
            )

    checks = sanity_check_record(
        None if sanity_checks is None else sanity_checks.get("parameter_randomization"),
        None if sanity_checks is None else sanity_checks.get("label_randomization"),
        checkpoint_sha256=selected.checkpoint_sha256,
    )
    mask_audit_passed = bool(authorization.body.get("mask_audit", {}).get("passed"))
    gate = leaf_score_gate(
        parameter_randomization=checks["parameter_randomization"]["status"] == "pass",
        label_randomization=checks["label_randomization"]["status"] == "pass",
        mask_audit_passed=mask_audit_passed,
    )
    from jute_disease.reporting import compute_training_costs

    training_costs = compute_training_costs(base)
    report: dict[str, Any] = {
        "record_hash": authorization.record_hash,
        "post_hoc": authorization.post_hoc,
        "assignment": assignment,
        "training_costs": training_costs,
        "scope": {
            "fit_id": selected.fit_id,
            "system": selected.system,
            "seed": selected.seed,
            "validation_macro_f1": selected.validation_macro_f1,
            "test_images": len(image_ids),
            "route": "ImageNet",
            "pool": assignment["pool"],
            "view": assignment["view"],
            "policy": assignment["policy"],
            "primary_split": 0,
            "claim_boundary": (
                "descriptive diagnostics only; no model ranking or "
                "pathology localization"
            ),
        },
        "embeddings": {
            "fine_tuned_penultimate": embedding_coordinates(
                np.stack(feature_rows), image_ids
            ),
            "frozen_imagenet_backbone": embedding_coordinates(
                np.stack(frozen_rows), image_ids
            ),
        },
        "gradcam_sample": sampled,
        "sanity_checks": checks,
        "leaf_attention_gate": gate,
    }
    if gate["reported"]:
        report["leaf_attention"] = aggregate_leaf_scores(score_rows)
        report["leaf_attention"]["target_rows"] = score_rows
    else:
        report["leaf_attention"] = {"reported": False, "reason": gate["reason"]}

    figure_root = Path(output_dir) / "gradcam"
    figure_root.mkdir(parents=True)
    figure_paths: list[str] = []
    for (image_id, target_index), (rgb, cam, leaf_mask) in figure_rows.items():
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(rgb)
        axes[0].set_title("Letterboxed image")
        axes[1].imshow(rgb)
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(rgb)
        axes[2].imshow(leaf_mask, cmap="Greens", alpha=0.35)
        axes[2].set_title("Automatic leaf mask")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{labels_by_id[image_id].label}; CAM target: "
            f"{manifest.labels[target_index]}"
        )
        figure.tight_layout()
        file_id = hashlib.sha256(image_id.encode()).hexdigest()[:16]
        figure_path = figure_root / f"{file_id}-target-{target_index}.png"
        figure.savefig(figure_path, dpi=150)
        figure_paths.append(figure_path.relative_to(Path(output_dir)).as_posix())
        plt.close(figure)
    report["gradcam_figures"] = sorted(figure_paths)
    cost_receipts = (
        training_costs["receipts"] + training_costs["classical_search"]["receipts"]
    )
    source_paths = [
        Path(target_dir) / "manifest.json",
        Path(target_dir) / "receipt.json",
        Path(assignment_table_path),
        Path(frozen_backbone_checkpoint),
        checkpoint_path,
        receipt_path,
        *source_artifacts,
        *(base / run.predictions for run in authorization.runs),
        *(
            base.joinpath(*item["fit_id"].split("/")) / "receipt.json"
            for item in cost_receipts
        ),
        *(dataset.image_root / entry.path for entry in entries),
    ]
    for optional_artifact in (
        Path(target_dir) / "grouping.json",
        Path(target_dir) / "assignments.json",
    ):
        if optional_artifact.is_file():
            source_paths.append(optional_artifact)
    source_paths = list(dict.fromkeys(Path(path).resolve() for path in source_paths))
    export_path = write_diagnostics_export(
        output_dir,
        report,
        generating_command=generating_command,
        configuration={
            **diagnostic_configuration(),
            "model_name": model_name,
            "analysis_ids": ["embeddings-test-pass", "gradcam-test-pass"],
            "assignment_table_sha256": assignment["sha256"],
            "mask_backend": {
                "model": getattr(mask_model, "model", type(mask_model).__name__),
                "version": getattr(mask_model, "version", None),
            },
        },
        source_artifacts=source_paths,
    )
    return {"export": str(export_path), "report": report}

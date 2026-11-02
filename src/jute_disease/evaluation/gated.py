"""Immutable evaluation selections and the only authorized test-image reader."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from jute_disease.training import load_benchmark

SELECTION_FORMAT = "jute-evaluation-selection-v1"
RUN_FORMAT = "jute-evaluation-run-v1"
CURRENT_SELECTION_FORMAT = "jute-evaluation-current-selection-v1"
CURRENT_SELECTION_FILE = "evaluator/current-selection.json"
TEST_READ_HISTORY_FILE = "evaluator/test-read-history.jsonl"


class GatedEvaluationError(ValueError):
    """A freeze, completed run or test evaluation failed its gate."""


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _read_envelope(
    path: Path, description: str, expected_format: str
) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatedEvaluationError(f"Cannot read {description}: {path}") from error
    payload = envelope.get("record") if isinstance(envelope, dict) else None
    if (
        not isinstance(payload, dict)
        or envelope.get("format") != expected_format
        or envelope.get("sha256") != _digest(payload)
    ):
        raise GatedEvaluationError(f"{description} is missing or has an invalid hash")
    return payload


def _read_selection(path: Path) -> dict[str, Any]:
    selection = _read_envelope(path, "selection record", SELECTION_FORMAT)
    try:
        matrix = selection["selection"]
        planned_fit_ids = matrix["planned_fit_ids"]
        final_fit_ids = matrix["final_fit_ids"]
        background_fit_ids = matrix["background_only"]["fit_ids"]
        leaf_only = matrix["leaf_only"]
        leaf_fit_ids = leaf_only["fit_ids"]
        border = matrix["border"]
        border_fit_ids = border["fit_ids"]
        model_specs = matrix["model_specs"]
        letterbox = selection["letterbox_decision"]
        preprocessing = selection["preprocessing"]
        scene_thresholds = selection["validation_outcomes"]["scene_thresholds"]
        frame_width = selection["border_frame_width_px"]["final"]
        leaf_share_cutoff = selection["border_leaf_share_cutoff"]["final"]
        mask_audit_outcome = selection["mask_audit_outcome"]
        created_after_test_reads = selection["created_after_test_reads"]
        leaf_sample_size = leaf_only["sample_size"]
        border_seeds = border["seeds"]
        null_permutations = border["null_permutations"]
    except (KeyError, TypeError) as error:
        raise GatedEvaluationError(
            "selection record has an invalid matrix schema"
        ) from error

    def valid_ids(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(item, str) and item for item in value)
            and len(value) == len(set(value))
        )

    def valid_seeds(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(type(item) is int for item in value)
            and len(value) == len(set(value))
        )

    if (
        not isinstance(matrix, dict)
        or not valid_ids(planned_fit_ids)
        or not valid_ids(final_fit_ids)
        or not valid_ids(background_fit_ids)
        or not valid_ids(leaf_fit_ids)
        or not valid_ids(border_fit_ids)
        or set(final_fit_ids + background_fit_ids + border_fit_ids)
        != set(planned_fit_ids)
        or len(final_fit_ids + background_fit_ids + border_fit_ids)
        != len(planned_fit_ids)
        or leaf_fit_ids != final_fit_ids
        or not isinstance(model_specs, dict)
        or set(model_specs) != set(planned_fit_ids)
        or any(not isinstance(spec, dict) for spec in model_specs.values())
        or type(leaf_sample_size) is not int
        or leaf_sample_size < 1
        or not valid_seeds(border_seeds)
        or type(null_permutations) is not int
        or null_permutations < 1
        or not isinstance(letterbox, str)
        or letterbox not in {"letterbox", "resize-center-crop"}
        or not isinstance(preprocessing, dict)
        or preprocessing.get("resize") != letterbox
        or type(preprocessing.get("size")) is not int
        or preprocessing["size"] < 1
        or not isinstance(scene_thresholds, dict)
        or any(
            not isinstance(scene_thresholds.get(name), (int, float))
            or isinstance(scene_thresholds.get(name), bool)
            or not 0 <= scene_thresholds[name] <= 1
            for name in ("background_fraction_min", "color_uniformity_min")
        )
        or type(frame_width) is not int
        or frame_width < 1
        or not isinstance(leaf_share_cutoff, (int, float))
        or isinstance(leaf_share_cutoff, bool)
        or not 0 <= leaf_share_cutoff <= 1
        or not isinstance(mask_audit_outcome, str)
        or mask_audit_outcome not in {"passed", "failed"}
        or not isinstance(created_after_test_reads, bool)
        or any(
            not isinstance(spec.get("kind"), str)
            or spec.get("kind") not in {"classical", "cnn", "cue"}
            or not isinstance(spec.get("seed"), int)
            or isinstance(spec.get("seed"), bool)
            or not isinstance(spec.get("training_view"), str)
            or not isinstance(spec.get("input_view"), str)
            or spec.get("preprocessing") != preprocessing
            for spec in model_specs.values()
        )
    ):
        raise GatedEvaluationError("selection record has an invalid matrix schema")
    return selection


@contextmanager
def _durable_temporary_file(path: Path, content: str) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_immutable(path: Path, format_name: str, payload: dict[str, Any]) -> str:
    path = Path(path)
    digest = _digest(payload)
    envelope = {"format": format_name, "record": payload, "sha256": digest}
    content = json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with _durable_temporary_file(path, content) as temporary:
        try:
            # Hard-link publication is exclusive: an existing record is never replaced.
            os.link(temporary, path)
        except FileExistsError as error:
            raise GatedEvaluationError(
                f"{record_description(path)} already exists; create a new freeze"
            ) from error
        return digest


def _write_json_atomic(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with _durable_temporary_file(path, content) as temporary:
        os.replace(temporary, path)


def _append_test_read_event(path: Path, event: dict[str, Any]) -> None:
    """Durably append the pre-read journal before any test image is opened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, _canonical(event))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_description(path: Path) -> str:
    """Describe a protected record in a concise error message."""
    return "selection record" if path.name == "selection.json" else "record"


def probe_plan(benchmark_path: Path | str) -> dict[str, Any]:
    """Expand the accepted background, leaf-only and border probe matrices."""
    project_root = Path(__file__).resolve().parents[3]
    probe_config = json.loads(
        (project_root / "configs" / "probes.json").read_text(encoding="utf-8")
    )
    background_config = probe_config["background_only"]
    leaf_config = probe_config["leaf_only"]
    border_config = probe_config["border"]
    benchmark = load_benchmark(benchmark_path)
    from jute_disease.models.ml import classical_configurations

    pools, policies = list(benchmark.pools), list(benchmark.policies)
    seeds = list(benchmark.recipe.seeds)
    primary = benchmark.primary_split
    final_ids = [
        fit.fit_id
        for fit in benchmark.fits()
        if fit.phase == "final" and fit.kind.value in {"target", "classical"}
    ]
    background: list[str] = []
    model_specs: dict[str, dict[str, Any]] = {}
    for estimator in benchmark.estimators:
        for features in benchmark.features:
            for pool in pools:
                for policy in policies:
                    for seed in seeds[: background_config["seed_count"]]:
                        fit_id = (
                            f"probe/background/classical/{estimator}-{features}/pool-{pool}"
                            f"/{policy}/split-{primary}/seed-{seed}"
                        )
                        background.append(fit_id)
                        model_specs[fit_id] = {
                            "kind": "classical",
                            "estimator": estimator,
                            "features": features,
                            "pool": pool,
                            "policy": policy,
                            "split": primary,
                            "seed": seed,
                            "training_view": background_config["training_view"],
                            "input_view": "background-only",
                        }
    for pool in pools:
        for policy in policies:
            for seed in seeds[: background_config["seed_count"]]:
                fit_id = (
                    f"probe/background/mobilenetv2_100.ra_in1k/pool-{pool}"
                    f"/{policy}/split-{primary}/seed-{seed}"
                )
                background.append(fit_id)
                model_specs[fit_id] = {
                    "kind": "cnn",
                    "model": "mobilenetv2_100.ra_in1k",
                    "pool": pool,
                    "policy": policy,
                    "split": primary,
                    "seed": seed,
                    "training_view": background_config["training_view"],
                    "input_view": "background-only",
                }
    border: list[str] = []
    for estimator in benchmark.estimators:
        for features in benchmark.features:
            for pool in pools:
                for policy in policies:
                    for seed in seeds[: border_config["seed_count"]]:
                        for configuration_index, parameters in enumerate(
                            classical_configurations(estimator)
                        ):
                            fit_id = (
                                f"probe/border/classical/{estimator}-{features}/pool-{pool}"
                                f"/{policy}/split-{primary}/seed-{seed}"
                                f"/config-{configuration_index:02d}"
                            )
                            border.append(fit_id)
                            model_specs[fit_id] = {
                                "kind": "classical",
                                "estimator": estimator,
                                "features": features,
                                "pool": pool,
                                "policy": policy,
                                "split": primary,
                                "seed": seed,
                                "configuration": configuration_index,
                                "parameters": parameters,
                                "training_view": border_config["training_view"],
                                "input_view": "border-frame",
                            }
    for pool in pools:
        for policy in policies:
            for seed in seeds[: border_config["seed_count"]]:
                fit_id = (
                    f"probe/border/mobilenetv2_100.ra_in1k/pool-{pool}"
                    f"/{policy}/split-{primary}/seed-{seed}"
                )
                border.append(fit_id)
                model_specs[fit_id] = {
                    "kind": "cnn",
                    "model": "mobilenetv2_100.ra_in1k",
                    "pool": pool,
                    "policy": policy,
                    "split": primary,
                    "seed": seed,
                    "training_view": border_config["training_view"],
                    "input_view": "border-frame",
                }
                for cue in ("pad-fraction", "aspect-ratio"):
                    cue_id = (
                        f"probe/border/{cue}/pool-{pool}/{policy}"
                        f"/split-{primary}/seed-{seed}"
                    )
                    border.append(cue_id)
                    model_specs[cue_id] = {
                        "kind": "cue",
                        "cue": cue,
                        "pool": pool,
                        "policy": policy,
                        "split": primary,
                        "seed": seed,
                        "training_view": cue,
                        "input_view": cue,
                    }
    fits = {
        fit.fit_id: fit
        for fit in benchmark.fits()
        if fit.phase == "final" and fit.kind.value in {"target", "classical"}
    }
    for fit_id in final_ids:
        fit = fits[fit_id]
        if fit.kind.value == "classical":
            model_specs[fit_id] = {
                "kind": "classical",
                "estimator": fit.estimator,
                "features": fit.features,
                "pool": fit.pool,
                "policy": fit.policy,
                "split": fit.split,
                "seed": fit.seed,
                "training_view": "raw-test",
                "input_view": "raw-test",
            }
        else:
            model_specs[fit_id] = {
                "kind": "cnn",
                "model": fit.model,
                "pool": fit.pool,
                "policy": fit.policy,
                "split": fit.split,
                "seed": fit.seed,
                "training_view": "raw-test",
                "input_view": "raw-test",
            }
    return {
        "primary_split": primary,
        "pools": pools,
        "policies": policies,
        "background_only": {
            "fit_ids": background,
            "classical_estimators": list(benchmark.estimators),
            "classical_features": list(benchmark.features),
            "cnn_route": "mobilenetv2_100.ra_in1k (ImageNet)",
            "seeds": seeds[: background_config["seed_count"]],
            "leaf_mask": background_config["leaf_mask"],
        },
        "leaf_only": {
            "fit_ids": final_ids,
            "sample_size": leaf_config["sample_size"],
            "sampling": leaf_config["sampling"],
        },
        "border": {
            "fit_ids": border,
            "classical_configurations_per_estimator": 9,
            "classical_estimators": list(benchmark.estimators),
            "classical_features": list(benchmark.features),
            "cnn_route": "mobilenetv2_100.ra_in1k (ImageNet)",
            "seeds": seeds[: border_config["seed_count"]],
            "frame_width_px_start": border_config["frame_width_px_start"],
            "leaf_share_cutoff_start": border_config["leaf_share_cutoff_start"],
            "pad_fraction_baseline": True,
            "null_permutations": border_config["null_permutations"],
            "family_level_permutation": border_config["family_level_permutation"],
        },
        "gradcam_test_pass": {"planned": True, "leaf_score_implemented": False},
        "final_fit_ids": final_ids,
        "model_specs": model_specs,
        "planned_fit_ids": sorted(set(final_ids + background + border)),
        "analysis_ids": sorted(
            set(final_ids + background + border + ["gradcam/test-pass"])
        ),
    }


def write_matrix_freeze(
    *,
    validation_outcomes: Path | str,
    benchmark_path: Path | str,
    target_dir: Path | str,
    output_path: Path | str,
) -> tuple[Path, str]:
    """Write a hash-bound selection only after recorded #114 outcomes."""
    from jute_disease.data import load_manifest

    outcomes_path = Path(validation_outcomes)
    try:
        outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatedEvaluationError("#114 validation outcomes are required") from error
    if not isinstance(outcomes, dict) or outcomes.get("issue") != 114:
        raise GatedEvaluationError("validation outcomes must identify issue #114")
    letterbox = outcomes.get("letterbox_comparison")
    audit = outcomes.get("mask_audit")
    budget = outcomes.get("post_pilot_budget_hours")
    scene = outcomes.get("scene_thresholds")
    frame = outcomes.get("border_frame_width_px")
    leaf_cutoff = outcomes.get("border_leaf_share_cutoff")
    hyperparameters = outcomes.get("selected_hyperparameters")
    if letterbox not in {"letterbox", "resize-center-crop"}:
        raise GatedEvaluationError("#114 letterbox comparison outcome is required")
    if not isinstance(audit, dict) or audit.get("status") not in {"passed", "failed"}:
        raise GatedEvaluationError("#114 mask-audit outcome must be recorded")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise GatedEvaluationError("#114 post-pilot budget must be recorded")
    if not isinstance(scene, dict) or any(
        not isinstance(scene.get(key), (int, float)) or not 0 <= scene[key] <= 1
        for key in ("background_fraction_min", "color_uniformity_min")
    ):
        raise GatedEvaluationError("#114 scene thresholds must be recorded")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1:
        raise GatedEvaluationError("border frame width must be recorded")
    if not isinstance(leaf_cutoff, (int, float)) or not 0 <= leaf_cutoff <= 1:
        raise GatedEvaluationError("border leaf-share cutoff must be recorded")
    if not isinstance(hyperparameters, dict) or not hyperparameters:
        raise GatedEvaluationError(
            "validation-selected hyperparameters must be recorded"
        )

    target = Path(target_dir)
    read_history = target / TEST_READ_HISTORY_FILE
    manifest_path = target / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = load_manifest(target)
    except (OSError, ValueError) as error:
        raise GatedEvaluationError("a completed target freeze is required") from error
    benchmark_bytes = Path(benchmark_path).read_bytes()
    root = Path(__file__).resolve().parents[3]
    tree_hash = hashlib.sha256()
    for relative in ("src", "configs"):
        for file in sorted((root / relative).rglob("*")):
            if file.is_file() and (
                (relative == "src" and file.suffix == ".py")
                or (relative == "configs" and file.suffix in {".json", ".yaml", ".yml"})
            ):
                tree_hash.update(file.relative_to(root).as_posix().encode())
                tree_hash.update(hashlib.sha256(file.read_bytes()).digest())
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    plan = probe_plan(benchmark_path)
    image_size = load_benchmark(benchmark_path).recipe.image_size
    preprocessing = {"resize": letterbox, "size": image_size}
    validation_hyperparameters_sha256 = _digest(hyperparameters)
    for fit_id, model_spec in plan["model_specs"].items():
        model_spec["preprocessing"] = preprocessing
        if model_spec["kind"] != "cue":
            model_spec["validation_hyperparameters_sha256"] = (
                validation_hyperparameters_sha256
            )
        if fit_id in plan["final_fit_ids"]:
            model_spec["training_view"] = "raw-training"
    payload = {
        "issue": 139,
        "manifest_digest": manifest.digest(),
        "classes": list(manifest.labels),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "validation_outcomes_sha256": hashlib.sha256(
            outcomes_path.read_bytes()
        ).hexdigest(),
        "validation_outcomes": outcomes,
        "selection": plan,
        "planned_fit_ids": plan["planned_fit_ids"],
        "analysis_ids": plan["analysis_ids"],
        "validation_selected_hyperparameters": hyperparameters,
        "letterbox_decision": letterbox,
        "preprocessing": preprocessing,
        "mask_audit_outcome": audit["status"],
        "border_frame_width_px": {
            "starting": plan["border"]["frame_width_px_start"],
            "final": frame,
        },
        "border_leaf_share_cutoff": {
            "starting": plan["border"]["leaf_share_cutoff_start"],
            "final": float(leaf_cutoff),
        },
        "post_pilot_budget_hours": float(budget),
        "pipeline_code_tree_sha256": tree_hash.hexdigest(),
        "environment_lock_sha256": lock_hash,
        "created_after_test_reads": False,
    }
    if read_history.is_file():
        payload["created_after_test_reads"] = True
        payload["test_read_history_sha256"] = hashlib.sha256(
            read_history.read_bytes()
        ).hexdigest()
    output = Path(output_path)
    digest = _write_immutable(output, SELECTION_FORMAT, payload)
    _write_json_atomic(
        target / CURRENT_SELECTION_FILE,
        {
            "format": CURRENT_SELECTION_FORMAT,
            "selection_sha256": digest,
        },
    )
    return output, digest


def write_fixture_run_record(
    selection_path: Path | str, output_path: Path | str
) -> tuple[Path, str]:
    """Write a completion record for the synthetic CLI gate fixture."""
    selection = _read_selection(Path(selection_path))
    fit_ids = selection["selection"]["planned_fit_ids"]
    record = {
        "issue": 139,
        "selection_sha256": _digest(selection),
        "status": "completed",
        "completed_analysis_ids": fit_ids,
        "fixture_predictions": True,
    }
    output = Path(output_path)
    return output, _write_immutable(output, RUN_FORMAT, record)


def write_run_record(
    selection_path: Path | str,
    model_index_path: Path | str,
    output_path: Path | str,
) -> tuple[Path, str]:
    """Bind every frozen analysis ID to an already completed model artifact."""
    selection_file = Path(selection_path)
    selection = _read_selection(selection_file)
    index_file = Path(model_index_path)
    try:
        index = json.loads(index_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatedEvaluationError(
            "a completed model-artifact index is required"
        ) from error
    models = index.get("models") if isinstance(index, dict) else None
    expected = set(selection["selection"]["planned_fit_ids"])
    model_specs = selection["selection"].get("model_specs")
    if not isinstance(model_specs, dict) or set(model_specs) != expected:
        raise GatedEvaluationError("selection lacks a model spec for every analysis")
    if not isinstance(models, dict) or set(models) != expected:
        raise GatedEvaluationError(
            "model-artifact index must cover every frozen analysis exactly once"
        )
    artifacts: dict[str, dict[str, str]] = {}
    for fit_id in sorted(expected):
        row = models[fit_id]
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise GatedEvaluationError(f"model artifact path is missing for {fit_id}")
        if row.get("fit_spec") != model_specs[fit_id]:
            raise GatedEvaluationError(f"model fit spec differs for {fit_id}")
        path = Path(row["path"])
        if not path.is_absolute():
            path = index_file.parent / path
        path = path.resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise GatedEvaluationError(
                f"model artifact is missing for {fit_id}"
            ) from error
        if row.get("sha256") != digest:
            raise GatedEvaluationError(f"model artifact hash differs for {fit_id}")
        artifacts[fit_id] = {
            "path": str(path),
            "sha256": digest,
            "fit_spec_sha256": _digest(model_specs[fit_id]),
        }
    record = {
        "issue": 139,
        "selection_sha256": _digest(selection),
        "status": "completed",
        "completed_analysis_ids": sorted(expected),
        "model_artifacts": artifacts,
        "model_index_sha256": hashlib.sha256(index_file.read_bytes()).hexdigest(),
        "fixture_predictions": False,
    }
    output = Path(output_path)
    return output, _write_immutable(output, RUN_FORMAT, record)


def _assert_committed_selection(selection_path: Path) -> None:
    """Require the exact frozen selection bytes to exist in the current commit."""
    root = Path(__file__).resolve().parents[3]
    try:
        relative = selection_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise GatedEvaluationError(
            "selection record must be committed in this repository"
        ) from error
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True,
        check=False,
    )
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        capture_output=True,
        check=False,
    )
    if (
        tracked.returncode
        or committed.returncode
        or committed.stdout != selection_path.read_bytes()
    ):
        raise GatedEvaluationError(
            "selection record must be tracked and committed before test reads"
        )


def _test_access_records(
    selection_path: Path, run_path: Path, *, fixture: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both gates before the caller loads the manifest or test pixels."""
    selection = _read_selection(selection_path)
    run_record = _read_envelope(run_path, "run record", RUN_FORMAT)
    if not fixture:
        _assert_committed_selection(selection_path)
    selection_digest = _digest(selection)
    if run_record.get("selection_sha256") != selection_digest:
        raise GatedEvaluationError("run record does not match the frozen selection")
    if run_record.get("status") != "completed":
        raise GatedEvaluationError("run record is not complete")
    expected = set(selection["selection"]["planned_fit_ids"])
    completed = run_record.get("completed_analysis_ids")
    if (
        not isinstance(completed, list)
        or len(completed) != len(set(completed))
        or set(completed) != expected
    ):
        raise GatedEvaluationError("run record does not cover every frozen analysis")
    model_specs = selection["selection"].get("model_specs")
    if not isinstance(model_specs, dict) or set(model_specs) != expected:
        raise GatedEvaluationError("selection lacks a model spec for every analysis")
    fixture_record = run_record.get("fixture_predictions") is True
    if fixture != fixture_record:
        raise GatedEvaluationError("fixture mode and run record type do not match")
    if not fixture:
        artifacts = run_record.get("model_artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != expected:
            raise GatedEvaluationError(
                "run record must bind every analysis to a model artifact"
            )
        for fit_id, artifact in artifacts.items():
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("path"), str)
                or not isinstance(artifact.get("sha256"), str)
                or artifact.get("fit_spec_sha256") != _digest(model_specs[fit_id])
            ):
                raise GatedEvaluationError(
                    f"model artifact record is invalid for {fit_id}"
                )
            path = Path(artifact["path"])
            if not path.is_file() or hashlib.sha256(
                path.read_bytes()
            ).hexdigest() != artifact.get("sha256"):
                raise GatedEvaluationError(
                    f"model artifact is missing or changed for {fit_id}"
                )
    return selection, run_record


def _fixture_prediction(fit_id: str, image_id: str, classes: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{fit_id}\0{image_id}".encode()).digest()
    return classes[int.from_bytes(digest[:8], "big") % len(classes)]


def _sample_indices(
    entries: list[Any], scene_tags: list[str], sample_size: int
) -> list[int]:
    if len(entries) <= sample_size:
        return list(range(len(entries)))
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (entry, scene) in enumerate(zip(entries, scene_tags, strict=True)):
        strata[(entry.label, scene)].append(index)
    quotas = {
        key: sample_size * len(indices) // len(entries)
        for key, indices in strata.items()
    }
    for key in sorted(strata)[: sample_size - sum(quotas.values())]:
        quotas[key] += 1
    selected: list[int] = []
    for key, indices in sorted(strata.items()):
        ordered = sorted(indices, key=lambda i: entries[i].image_id)
        selected.extend(ordered[: quotas[key]])
    return sorted(selected)


def _metrics(
    entries: list[Any], predictions: list[str], classes: tuple[str, ...]
) -> dict[str, Any]:
    correct = sum(
        entry.label == predicted
        for entry, predicted in zip(entries, predictions, strict=True)
    )
    counts = Counter(entry.label for entry in entries)
    f1s = []
    for label in classes:
        tp = sum(
            entry.label == predicted == label
            for entry, predicted in zip(entries, predictions, strict=True)
        )
        predicted_count = predictions.count(label)
        support = counts[label]
        f1s.append(2 * tp / (support + predicted_count) if support else None)
    supported = [value for value in f1s if value is not None]
    return {
        "n": len(entries),
        "accuracy": correct / len(entries) if entries else None,
        "macro_f1": float(np.mean(supported)) if supported else None,
    }


def _family_metric_counts(
    entries: list[Any],
    predictions: list[str],
    classes: tuple[str, ...],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Summarize support, predictions and correct labels within each family."""
    class_index = {label: index for index, label in enumerate(classes)}
    family_indices: dict[str, list[int]] = defaultdict(list)
    family_labels: dict[str, int] = {}
    for index, (entry, _) in enumerate(zip(entries, predictions, strict=True)):
        label_index = class_index[entry.label]
        if entry.group in family_labels and family_labels[entry.group] != label_index:
            raise GatedEvaluationError("a family spans multiple true classes")
        family_labels[entry.group] = label_index
        family_indices[entry.group].append(index)
    family_ids = sorted(family_indices)
    family_support = np.zeros((len(family_ids), len(classes)), dtype=np.int64)
    family_predictions = np.zeros_like(family_support)
    family_true_positives = np.zeros_like(family_support)
    for family_index, family in enumerate(family_ids):
        indices = family_indices[family]
        true_label = family_labels[family]
        predicted = np.asarray(
            [class_index[predictions[index]] for index in indices], dtype=np.int64
        )
        family_support[family_index, true_label] = len(indices)
        family_predictions[family_index] = np.bincount(
            predicted, minlength=len(classes)
        )
        family_true_positives[family_index, true_label] = int(
            np.count_nonzero(predicted == true_label)
        )
    family_stats = np.concatenate(
        (family_support, family_predictions, family_true_positives), axis=1
    )
    family_label_indices = np.asarray(
        [family_labels[family] for family in family_ids], dtype=np.int64
    )
    return family_ids, family_stats, family_predictions, family_label_indices


def _hierarchical_bootstrap_plan(
    seed_count: int,
    family_count: int,
    *,
    resamples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw paired seed and family multiplicities for every comparison."""
    if seed_count < 1 or family_count < 1:
        return (
            np.empty((0, seed_count), dtype=np.int64),
            np.empty((0, seed_count, family_count), dtype=np.int16),
        )
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0, seed_count, size=(resamples, seed_count))
    family_weights = np.zeros((resamples, seed_count, family_count), dtype=np.int16)
    for resample in range(resamples):
        for draw in range(seed_count):
            selected = rng.integers(0, family_count, size=family_count)
            family_weights[resample, draw] = np.bincount(
                selected, minlength=family_count
            )
    return seed_draws, family_weights


def _hierarchical_intervals(
    family_stats_by_seed: np.ndarray,
    seed_draws: np.ndarray,
    family_weights: np.ndarray,
    class_count: int,
) -> dict[str, dict[str, float] | None]:
    """Compute intervals after resampling seeds, then paired test families."""
    if seed_draws.size == 0:
        return {"accuracy": None, "macro_f1": None}
    accuracy = np.empty(len(seed_draws), dtype=np.float64)
    macro_f1 = np.empty(len(seed_draws), dtype=np.float64)
    for start in range(0, len(seed_draws), 64):
        stop = min(start + 64, len(seed_draws))
        selected_stats = family_stats_by_seed[seed_draws[start:stop]]
        weighted = np.einsum(
            "rsf,rsfd->rsd",
            family_weights[start:stop],
            selected_stats,
            optimize=True,
        )
        support = weighted[..., :class_count]
        predicted = weighted[..., class_count : 2 * class_count]
        true_positive = weighted[..., 2 * class_count :]
        denominator = support + predicted
        class_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=denominator > 0,
        )
        supported_classes = support > 0
        macro_by_seed = np.divide(
            (class_f1 * supported_classes).sum(axis=-1),
            supported_classes.sum(axis=-1),
            out=np.zeros(supported_classes.shape[:-1], dtype=np.float64),
            where=supported_classes.sum(axis=-1) > 0,
        )
        accuracy_by_seed = np.divide(
            true_positive.sum(axis=-1),
            support.sum(axis=-1),
            out=np.zeros(support.shape[:-1], dtype=np.float64),
            where=support.sum(axis=-1) > 0,
        )
        macro_f1[start:stop] = macro_by_seed.mean(axis=1)
        accuracy[start:stop] = accuracy_by_seed.mean(axis=1)
    return {
        "accuracy": {
            "lower": float(np.percentile(accuracy, 2.5)),
            "upper": float(np.percentile(accuracy, 97.5)),
        },
        "macro_f1": {
            "lower": float(np.percentile(macro_f1, 2.5)),
            "upper": float(np.percentile(macro_f1, 97.5)),
        },
    }


def _family_permutation_macro_f1(
    predictions_by_seed: np.ndarray,
    permuted_family_labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Return paired family-label permutation scores averaged across seeds."""
    permutation_count = len(permuted_family_labels)
    seed_scores = np.zeros((predictions_by_seed.shape[0], permutation_count))
    for seed_index, predictions in enumerate(predictions_by_seed):
        confusion = np.zeros(
            (permutation_count, class_count, class_count), dtype=np.int64
        )
        for true_label in range(class_count):
            family_mask = (permuted_family_labels == true_label).astype(np.int64)
            confusion[:, true_label, :] = family_mask @ predictions
        support = confusion.sum(axis=2)
        true_positive = np.diagonal(confusion, axis1=1, axis2=2)
        predicted = confusion.sum(axis=1)
        denominator = support + predicted
        class_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        supported_classes = support > 0
        seed_scores[seed_index] = np.divide(
            (class_f1 * supported_classes).sum(axis=1),
            supported_classes.sum(axis=1),
            out=np.zeros(permutation_count, dtype=np.float64),
            where=supported_classes.sum(axis=1) > 0,
        )
    return seed_scores.mean(axis=0)


def _mean_seed_metrics(seed_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Report seed means while preserving the observed per-seed metrics."""
    metrics = list(seed_metrics.values())
    accuracy = [row["accuracy"] for row in metrics if row["accuracy"] is not None]
    macro_f1 = [row["macro_f1"] for row in metrics if row["macro_f1"] is not None]
    return {
        "n": metrics[0]["n"] if metrics else 0,
        "accuracy": float(np.mean(accuracy)) if accuracy else None,
        "macro_f1": float(np.mean(macro_f1)) if macro_f1 else None,
        "per_seed": seed_metrics,
    }


def _scene_report(
    entries: list[Any],
    scenes: list[str],
    predictions: list[str],
    classes: tuple[str, ...],
) -> dict[str, Any]:
    from jute_disease.evaluation import wilson_interval

    rows: dict[str, Any] = {}
    for scene in sorted(set(scenes)):
        indices = [index for index, value in enumerate(scenes) if value == scene]
        subset = [entries[index] for index in indices]
        predicted = [predictions[index] for index in indices]
        metrics = _metrics(subset, predicted, classes)
        by_class: dict[str, int] = {}
        class_cells: dict[str, Any] = {}
        for label in classes:
            indices_for_class = [
                i for i, entry in enumerate(subset) if entry.label == label
            ]
            by_class[label] = len(indices_for_class)
            correct = sum(
                subset[index].label == predicted[index] for index in indices_for_class
            )
            class_interval = (
                wilson_interval(correct, len(indices_for_class))
                if indices_for_class
                else None
            )
            class_cells[label] = {
                "n": len(indices_for_class),
                "accuracy": correct / len(indices_for_class)
                if indices_for_class
                else None,
                "accuracy_wilson_95": None
                if class_interval is None
                else {
                    "lower": class_interval.lower,
                    "upper": class_interval.upper,
                },
            }
        interval = (
            wilson_interval(round(metrics["accuracy"] * metrics["n"]), metrics["n"])
            if metrics["n"]
            else None
        )
        row = {
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "accuracy_wilson_95": None
            if interval is None
            else {
                "lower": interval.lower,
                "upper": interval.upper,
            },
            "class_counts": by_class,
            "class_cells": class_cells,
        }
        if all(value >= 30 for value in by_class.values()):
            row["macro_f1"] = metrics["macro_f1"]
        rows[scene] = row
    return rows


def _decode_model_labels(values: np.ndarray, classes: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if isinstance(value, str) and value in classes:
            labels.append(value)
            continue
        try:
            index = int(value)
        except (TypeError, ValueError) as error:
            raise GatedEvaluationError(
                "model emitted an unknown class label"
            ) from error
        if index < 0 or index >= len(classes):
            raise GatedEvaluationError("model emitted an unknown class index")
        labels.append(classes[index])
    return labels


def _resize_center_crop(
    image: Image.Image, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen shortest-side resize and center crop to RGB and mask."""
    width, height = image.size
    scale = size / min(width, height)
    resized_width = max(size, round(width * scale))
    resized_height = max(size, round(height * scale))
    left = (resized_width - size) // 2
    top = (resized_height - size) // 2
    rgb = np.asarray(
        image.resize((resized_width, resized_height), Image.Resampling.BILINEAR).crop(
            (left, top, left + size, top + size)
        )
    )
    resized_mask = (
        np.asarray(
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
            .resize((resized_width, resized_height), Image.Resampling.NEAREST)
            .crop((left, top, left + size, top + size))
        )
        > 0
    )
    return rgb, resized_mask


def _predict_model_artifact(
    spec: dict[str, Any],
    artifact_path: Path,
    entries: list[Any],
    classes: tuple[str, ...],
    view_name: str,
    read_views: Callable[
        [Any], tuple[dict[str, np.ndarray], np.ndarray, float, float, str]
    ],
) -> list[str]:
    """Apply one hash-checked trained artifact to a selected evaluator view."""
    import joblib
    import torch

    from jute_disease.data import ImageWithMask
    from jute_disease.models.dl import Classifier, TimmBackbone, read_checkpoint_state
    from jute_disease.models.ml import CraftedFeatureExtractor, RawPixelFeatureExtractor

    kind = spec["kind"]
    model: Any
    extractor: Any = None
    if kind in {"classical", "cue"}:
        try:
            model = joblib.load(artifact_path)
        except Exception as error:
            raise GatedEvaluationError(
                f"cannot load model artifact for {artifact_path.name}"
            ) from error
        if kind == "classical":
            extractor = (
                CraftedFeatureExtractor()
                if spec["features"] == "crafted"
                else RawPixelFeatureExtractor()
            )
    elif kind == "cnn":
        try:
            model = Classifier(
                TimmBackbone(spec["model"], pretrained=False),
                num_classes=len(classes),
            )
            state = read_checkpoint_state(artifact_path)
            missing, unexpected = model.load_state_dict(state, strict=False)
            invalid_missing = [
                key
                for key in missing
                if not key.startswith(("train_metrics.", "val_metrics."))
            ]
            invalid_unexpected = [
                key
                for key in unexpected
                if not key.startswith(("train_metrics.", "val_metrics."))
            ]
            if invalid_missing or invalid_unexpected:
                raise ValueError("checkpoint weights do not match the CNN route")
            model.eval()
        except Exception as error:
            raise GatedEvaluationError(
                f"cannot load CNN artifact for {artifact_path.name}"
            ) from error
    else:
        raise GatedEvaluationError(f"unknown model kind {kind!r}")

    predictions: list[str] = []
    batch_size = 16
    if kind == "cnn":
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    for offset in range(0, len(entries), batch_size):
        batch_entries = entries[offset : offset + batch_size]
        batch_inputs: list[Any] = []
        for entry in batch_entries:
            views, valid, _, pad_fraction, _ = read_views(entry)
            if kind == "cue":
                value = (
                    pad_fraction
                    if spec["cue"] == "pad-fraction"
                    else entry.width / entry.height
                )
                batch_inputs.append([value])
            elif kind == "classical":
                batch_inputs.append(
                    extractor.extract(ImageWithMask(views[view_name], valid))
                )
            else:
                tensor = (
                    torch.from_numpy(
                        np.asarray(views[view_name], dtype=np.float32).copy()
                    )
                    .permute(2, 0, 1)
                    .div_(255.0)
                )
                batch_inputs.append((tensor - mean) / std)
        if kind in {"classical", "cue"}:
            values = model.predict(np.asarray(batch_inputs, dtype=np.float32))
        else:
            with torch.inference_mode():
                logits = model(torch.stack(batch_inputs))
            values = logits.argmax(dim=1).cpu().numpy()
        predictions.extend(_decode_model_labels(np.asarray(values), classes))
    return predictions


def evaluate_gated_test(
    *,
    selection_path: Path | str,
    run_record_path: Path | str,
    target_dir: Path | str,
    image_root: Path | str,
    output_dir: Path | str,
    fixture: bool = False,
    mask_checkpoint: Path | str | None = None,
) -> Path:
    """Score frozen model artifacts; this is the only test-image reader."""
    from jute_disease.data import (
        EVALUATOR_TEST_MANIFEST,
        ColorThresholdBackend,
        ManifestEntry,
        Sam2Refiner,
        classical_pipeline,
        classify_scene,
        extract_scene_features,
        load_manifest,
    )

    selection_file, run_file = Path(selection_path), Path(run_record_path)
    selection, run_record = _test_access_records(
        selection_file, run_file, fixture=fixture
    )
    try:
        current = json.loads(
            (Path(target_dir) / CURRENT_SELECTION_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise GatedEvaluationError("a current matrix freeze is required") from error
    if (
        not isinstance(current, dict)
        or current.get("format") != CURRENT_SELECTION_FORMAT
        or current.get("selection_sha256") != _digest(selection)
    ):
        raise GatedEvaluationError("selection was superseded by a later matrix freeze")
    if not fixture and mask_checkpoint is None:
        raise GatedEvaluationError(
            "production evaluation requires a local SAM 2 mask checkpoint"
        )
    try:
        backend = (
            ColorThresholdBackend() if fixture else Sam2Refiner(Path(mask_checkpoint))
        )
    except (ImportError, OSError, ValueError) as error:
        raise GatedEvaluationError(
            f"cannot load local SAM 2 mask checkpoint: {error}"
        ) from error
    evaluation_id = uuid4().hex
    history_path = Path(target_dir) / TEST_READ_HISTORY_FILE
    _append_test_read_event(
        history_path,
        {
            "evaluation_id": evaluation_id,
            "status": "started",
            "selection_sha256": _digest(selection),
            "run_record_sha256": _digest(run_record),
            "read_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    # No manifest partition or image bytes are read until both immutable records pass.
    manifest = load_manifest(Path(target_dir))
    if manifest.digest() != selection.get("manifest_digest"):
        raise GatedEvaluationError("selection does not match the target freeze")
    if list(manifest.labels) != selection.get("classes"):
        raise GatedEvaluationError(
            "selection class order differs from the target freeze"
        )
    test_manifest_path = Path(target_dir) / EVALUATOR_TEST_MANIFEST
    try:
        envelope = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatedEvaluationError(
            "separate evaluator test manifest is required"
        ) from error
    record = envelope.get("record") if isinstance(envelope, dict) else None
    if (
        not isinstance(record, dict)
        or envelope.get("format") != "jute-evaluator-test-manifest-v1"
        or envelope.get("sha256") != _digest(record)
        or record.get("partition") != "test"
        or record.get("source_manifest_digest") != manifest.digest()
        or record.get("version") != manifest.version
    ):
        raise GatedEvaluationError(
            "separate evaluator test manifest is invalid or stale"
        )
    try:
        entries = [ManifestEntry(**row) for row in record["entries"]]
    except (KeyError, TypeError, ValueError) as error:
        raise GatedEvaluationError(
            "separate evaluator test manifest has invalid entries"
        ) from error
    expected_entries = [
        entry.to_json(manifest.version) for entry in manifest.partition("test")
    ]
    if [
        entry.to_json(manifest.version) for entry in entries
    ] != expected_entries or any(entry.partition != "test" for entry in entries):
        raise GatedEvaluationError(
            "separate evaluator test manifest differs from the freeze"
        )
    entries.sort(key=lambda entry: entry.image_id)
    if not entries:
        raise GatedEvaluationError("frozen test partition is empty")
    classes = tuple(manifest.labels)
    resize_decision = selection["letterbox_decision"]
    preprocessing = selection.get("preprocessing")
    if (
        not isinstance(preprocessing, dict)
        or preprocessing.get("resize") != resize_decision
        or type(preprocessing.get("size")) is not int
        or preprocessing["size"] < 1
        or any(
            spec.get("preprocessing") != preprocessing
            for spec in selection["selection"]["model_specs"].values()
        )
    ):
        raise GatedEvaluationError("frozen preprocessing specification is invalid")
    image_size = preprocessing["size"]
    transform = classical_pipeline(size=image_size, include_mask=True)
    scene_thresholds = SimpleNamespace(
        **selection["validation_outcomes"]["scene_thresholds"]
    )
    frame_width = int(selection["border_frame_width_px"]["final"])
    pad_rgb = np.asarray((124, 116, 104), dtype=np.uint8)
    temporary = tempfile.TemporaryDirectory(prefix="jute-gated-")
    temporary_dir = Path(temporary.name)
    pixels_cache = np.memmap(
        temporary_dir / "pixels.dat",
        mode="w+",
        dtype=np.uint8,
        shape=(len(entries), image_size, image_size, 3),
    )
    packed_width = (image_size * image_size + 7) // 8
    mask_cache = np.memmap(
        temporary_dir / "leaf-mask.dat",
        mode="w+",
        dtype=np.uint8,
        shape=(len(entries), packed_width),
    )
    entry_indices = {entry.image_id: index for index, entry in enumerate(entries)}
    prepared: dict[str, tuple[tuple[int, int, int, int], float, float, str]] = {}

    def read_views(
        entry: ManifestEntry,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, float, str]:
        cache_index = entry_indices[entry.image_id]
        if entry.image_id not in prepared:
            image_path = Path(image_root) / entry.path
            try:
                with Image.open(image_path) as source:
                    original = source.convert("RGB")
            except OSError as error:
                raise GatedEvaluationError(
                    f"cannot read frozen test image {entry.image_id}"
                ) from error
            mask = backend.predict(original).mask
            scene = classify_scene(
                extract_scene_features(original, mask), scene_thresholds
            )
            if resize_decision == "letterbox":
                boxed = transform(original)
                pixels, valid = boxed.pixels, boxed.valid_mask
                y, x = np.where(valid)
                top, bottom, left, right = (
                    int(y.min()),
                    int(y.max()) + 1,
                    int(x.min()),
                    int(x.max()) + 1,
                )
                resized = (
                    np.asarray(
                        Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (right - left, bottom - top), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
                pad_fraction = 1 - float(valid.mean())
            else:
                pixels, resized = _resize_center_crop(original, mask, image_size)
                top, bottom, left, right = 0, image_size, 0, image_size
                pad_fraction = 0.0
            resized_mask = np.zeros((image_size, image_size), dtype=bool)
            resized_mask[top:bottom, left:right] = resized
            frame = np.zeros((image_size, image_size), dtype=bool)
            frame[top : min(bottom, top + frame_width), left:right] = True
            frame[max(top, bottom - frame_width) : bottom, left:right] = True
            frame[top:bottom, left : min(right, left + frame_width)] = True
            frame[top:bottom, max(left, right - frame_width) : right] = True
            leaf_share = float(np.logical_and(frame, resized_mask).sum() / frame.sum())
            pixels_cache[cache_index] = pixels
            mask_cache[cache_index] = np.packbits(resized_mask.ravel())
            prepared[entry.image_id] = (
                (top, bottom, left, right),
                leaf_share,
                pad_fraction,
                scene,
            )
        bounds, leaf_share, pad_fraction, scene = prepared[entry.image_id]
        pixels = pixels_cache[cache_index]
        top, bottom, left, right = bounds
        valid = np.zeros((image_size, image_size), dtype=bool)
        valid[top:bottom, left:right] = True
        resized_mask = (
            np.unpackbits(mask_cache[cache_index], count=image_size * image_size)
            .reshape((image_size, image_size))
            .astype(bool)
        )
        frame = np.zeros_like(valid)
        frame[top : min(bottom, top + frame_width), left:right] = True
        frame[max(top, bottom - frame_width) : bottom, left:right] = True
        frame[top:bottom, left : min(right, left + frame_width)] = True
        frame[top:bottom, max(left, right - frame_width) : right] = True
        views = {
            "raw-test": pixels,
            "background-only": np.where(resized_mask[..., None], pad_rgb, pixels),
            "leaf-only": np.where(resized_mask[..., None], pixels, pad_rgb),
            "border-frame": np.where(frame[..., None], pixels, pad_rgb),
        }
        return views, valid, leaf_share, pad_fraction, scene

    scenes: list[str] = []
    border_leaf_shares: list[float] = []
    pad_fractions: list[float] = []
    image_view_hashes = {
        name: hashlib.sha256()
        for name in ("raw-test", "background-only", "leaf-only", "border-frame")
    }
    for entry in entries:
        views, _, leaf_share, pad_fraction, scene = read_views(entry)
        scenes.append(scene)
        border_leaf_shares.append(leaf_share)
        pad_fractions.append(pad_fraction)
        for name, view_pixels in views.items():
            image_view_hashes[name].update(entry.image_id.encode("ascii"))
            image_view_hashes[name].update(
                np.asarray(view_pixels, dtype=np.uint8).tobytes()
            )

    expected_ids = run_record["completed_analysis_ids"]

    def predictions_for(
        fit_id: str, selected: list[Any], *, leaf: bool = False
    ) -> list[str]:
        if fixture:
            suffix = ":leaf-only" if leaf else ""
            return [
                _fixture_prediction(f"{fit_id}{suffix}", entry.image_id, classes)
                for entry in selected
            ]
        spec = selection["selection"]["model_specs"][fit_id]
        artifact_path = Path(run_record["model_artifacts"][fit_id]["path"])
        view_name = "leaf-only" if leaf else spec["input_view"]
        return _predict_model_artifact(
            spec, artifact_path, selected, classes, view_name, read_views
        )

    final_ids = selection["selection"]["final_fit_ids"]
    background_ids = selection["selection"]["background_only"]["fit_ids"]
    border_ids = selection["selection"]["border"]["fit_ids"]
    # The record covers every planned fit, preventing partial or selective test reads.
    if set(final_ids + background_ids + border_ids) != set(expected_ids):
        raise GatedEvaluationError("selection and run record analysis IDs differ")
    truths = [entry.label for entry in entries]
    mask_audit_passed = selection["mask_audit_outcome"] == "passed"
    sample_target = selection["selection"]["leaf_only"]["sample_size"]
    sample_indices = (
        _sample_indices(entries, scenes, min(sample_target, len(entries)))
        if mask_audit_passed
        else []
    )
    sample_entries = [entries[index] for index in sample_indices]
    background_scores = (
        {
            fit_id: _metrics(entries, predictions_for(fit_id, entries), classes)
            for fit_id in background_ids
        }
        if mask_audit_passed
        else None
    )
    leaf_scores = (
        {
            fit_id: _metrics(
                sample_entries,
                predictions_for(fit_id, sample_entries, leaf=True),
                classes,
            )
            for fit_id in final_ids
        }
        if mask_audit_passed
        else None
    )
    scene_tables: dict[str, Any] = {}
    if mask_audit_passed:
        for fit_id in final_ids:
            scene_tables[fit_id] = _scene_report(
                entries, scenes, predictions_for(fit_id, entries), classes
            )

    border_groups: dict[str, list[str]] = defaultdict(list)
    for fit_id in border_ids:
        group_id = "/".join(
            component
            for component in fit_id.split("/")
            if not component.startswith("seed-")
        )
        border_groups[group_id].append(fit_id)
    family_ids = sorted({entry.group for entry in entries})
    family_labels: dict[str, int] = {}
    class_indices = {label: index for index, label in enumerate(classes)}
    for entry in entries:
        label_index = class_indices[entry.label]
        if entry.group in family_labels and family_labels[entry.group] != label_index:
            raise GatedEvaluationError("a family spans multiple true classes")
        family_labels[entry.group] = label_index
    family_label_vector = np.asarray(
        [family_labels[family] for family in family_ids], dtype=np.int64
    )
    null_count = selection["selection"]["border"]["null_permutations"]
    null_rng = np.random.default_rng(42)
    permuted_family_labels = np.vstack(
        [null_rng.permutation(family_label_vector) for _ in range(null_count)]
    )
    border_seeds = selection["selection"]["border"]["seeds"]
    seed_draws, family_weights = _hierarchical_bootstrap_plan(
        len(border_seeds),
        len(family_ids),
        resamples=null_count,
        seed=42,
    )
    retained = [
        index
        for index, share in enumerate(border_leaf_shares)
        if share <= selection["border_leaf_share_cutoff"]["final"]
    ]
    retained_entries = [entries[index] for index in retained]
    retained_family_ids = sorted({entry.group for entry in retained_entries})
    retained_class_counts = dict.fromkeys(classes, 0)
    retained_class_counts.update(Counter(entry.label for entry in retained_entries))
    retained_scene_counts = dict.fromkeys(sorted(set(scenes)), 0)
    retained_scene_counts.update(Counter(scenes[index] for index in retained))
    headline_allowed = mask_audit_passed and all(
        retained_class_counts.get(label, 0) >= 50 for label in classes
    )
    survival_seed_draws: np.ndarray | None = None
    survival_family_weights: np.ndarray | None = None
    if headline_allowed:
        survival_seed_draws, survival_family_weights = _hierarchical_bootstrap_plan(
            len(border_seeds),
            len(retained_family_ids),
            resamples=null_count,
            seed=43,
        )
    border_results: dict[str, Any] = {}
    cue_baselines: dict[str, Any] = {}
    expected_seed_count = len(border_seeds)
    for group_id, unsorted_fit_ids in sorted(border_groups.items()):
        fit_ids = sorted(
            unsorted_fit_ids,
            key=lambda fit_id: border_seeds.index(
                selection["selection"]["model_specs"][fit_id]["seed"]
            ),
        )
        if len(fit_ids) != expected_seed_count:
            raise GatedEvaluationError(
                f"border model group does not cover every frozen seed: {group_id}"
            )
        group_spec = {
            key: value
            for key, value in selection["selection"]["model_specs"][fit_ids[0]].items()
            if key != "seed"
        }
        seed_metrics: dict[str, dict[str, Any]] = {}
        family_stats_by_seed = []
        predicted_counts_by_seed = []
        retained_stats_by_seed = []
        retained_seed_metrics: dict[str, dict[str, Any]] = {}
        retained_family_order: list[str] | None = None
        is_cue = group_spec["kind"] == "cue"
        for fit_id in fit_ids:
            seed = selection["selection"]["model_specs"][fit_id]["seed"]
            predicted = predictions_for(fit_id, entries)
            seed_metrics[str(seed)] = _metrics(entries, predicted, classes)
            current_families, family_stats, family_predictions, _ = (
                _family_metric_counts(entries, predicted, classes)
            )
            if current_families != family_ids:
                raise GatedEvaluationError("border family order changed between seeds")
            family_stats_by_seed.append(family_stats)
            predicted_counts_by_seed.append(family_predictions)
            if not is_cue and headline_allowed:
                retained_predictions = [predicted[index] for index in retained]
                retained_seed_metrics[str(seed)] = _metrics(
                    retained_entries, retained_predictions, classes
                )
                current_retained_families, retained_stats, _, _ = _family_metric_counts(
                    retained_entries, retained_predictions, classes
                )
                if retained_family_order is None:
                    retained_family_order = current_retained_families
                elif current_retained_families != retained_family_order:
                    raise GatedEvaluationError(
                        "retained family order changed between border seeds"
                    )
                retained_stats_by_seed.append(retained_stats)

        if is_cue:
            intervals = _hierarchical_intervals(
                np.stack(family_stats_by_seed),
                seed_draws,
                family_weights,
                len(classes),
            )
            null = _family_permutation_macro_f1(
                np.stack(predicted_counts_by_seed),
                permuted_family_labels,
                len(classes),
            )
            lower = intervals["macro_f1"]["lower"] if intervals["macro_f1"] else None
            upper_null = float(np.percentile(null, 95))
            cue_baselines[group_id] = {
                "fit_ids": fit_ids,
                "fit_spec": group_spec,
                "metrics": _mean_seed_metrics(seed_metrics),
                "accuracy_ci_95": intervals["accuracy"],
                "macro_f1_ci_95": intervals["macro_f1"],
                "majority_class_accuracy": max(Counter(truths).values()) / len(truths),
                "permutation_macro_f1_95th": upper_null,
                "rule": "cue baseline exceeds the family-label null"
                if lower is not None and lower > upper_null
                else "cue baseline does not exceed the family-label null",
            }
            continue

        full_stats_array = np.stack(family_stats_by_seed)
        full_intervals = _hierarchical_intervals(
            full_stats_array,
            seed_draws,
            family_weights,
            len(classes),
        )
        survival_intervals: dict[str, Any] | None = None
        symptom_survival: dict[str, Any] | None = None
        if headline_allowed:
            if retained_family_order != retained_family_ids:
                raise GatedEvaluationError(
                    "retained family order differs from the probe"
                )
            if survival_seed_draws is None or survival_family_weights is None:
                raise GatedEvaluationError("symptom-survival bootstrap plan is missing")
            retained_stats_array = np.stack(retained_stats_by_seed)
            survival_intervals = _hierarchical_intervals(
                retained_stats_array,
                survival_seed_draws,
                survival_family_weights,
                len(classes),
            )
            symptom_survival = _mean_seed_metrics(retained_seed_metrics)
        null = _family_permutation_macro_f1(
            np.stack(predicted_counts_by_seed),
            permuted_family_labels,
            len(classes),
        )
        lower = (
            full_intervals["macro_f1"]["lower"] if full_intervals["macro_f1"] else None
        )
        upper_null = float(np.percentile(null, 95))
        unrestricted = _mean_seed_metrics(seed_metrics)
        border_results[group_id] = {
            "fit_ids": fit_ids,
            "fit_spec": group_spec,
            "unrestricted": unrestricted,
            "symptom_survival": symptom_survival,
            "retained_class_counts": retained_class_counts,
            "retained_scene_counts": retained_scene_counts,
            "frame_leaf_share_mean": float(np.mean(border_leaf_shares)),
            "pad_fraction_baseline_mean": float(np.mean(pad_fractions)),
            "majority_class_accuracy": max(Counter(truths).values()) / len(truths),
            "accuracy_ci_95": full_intervals["accuracy"],
            "macro_f1_ci_95": full_intervals["macro_f1"],
            "symptom_survival_accuracy_ci_95": None
            if survival_intervals is None
            else survival_intervals["accuracy"],
            "symptom_survival_macro_f1_ci_95": None
            if survival_intervals is None
            else survival_intervals["macro_f1"],
            "macro_f1_lower_95": lower,
            "permutation_macro_f1_95th": upper_null,
            "permutations": null_count,
            "bootstrap_method": (
                "hierarchical: seeds within primary split, then test families; "
                "family multiplicities paired across systems"
            ),
            "rule": "border information detected"
            if lower is not None and lower > upper_null
            else "no detectable border information at this frame width",
            "headline": symptom_survival if headline_allowed else None,
            "headline_omission_reason": None
            if headline_allowed
            else (
                "mask audit failed"
                if not mask_audit_passed
                else "a class has fewer than 50 retained test images"
            ),
            "limitation": (
                "Leaf share in a 32 px strip is more sensitive to mask-edge errors "
                "than whole-leaf mask quality."
            ),
        }
    model_provenance = {}
    if not fixture:
        model_provenance = {
            fit_id: {
                "training_view": selection["selection"]["model_specs"][fit_id][
                    "training_view"
                ],
                "input_view": selection["selection"]["model_specs"][fit_id][
                    "input_view"
                ],
                "fit_spec_sha256": run_record["model_artifacts"][fit_id][
                    "fit_spec_sha256"
                ],
                "artifact_sha256": run_record["model_artifacts"][fit_id]["sha256"],
            }
            for fit_id in expected_ids
        }
    report = {
        "issue": 139,
        "selection_sha256": _digest(selection),
        "run_record_sha256": _digest(run_record),
        "manifest_digest": manifest.digest(),
        "post_hoc": selection["created_after_test_reads"],
        "test_images": len(entries),
        "evaluated_view_sha256": {
            name: digest.hexdigest() for name, digest in image_view_hashes.items()
        },
        "preprocessing": {
            "resize": resize_decision,
            "size": image_size,
        },
        "mask_pipeline": {
            "model": backend.model,
            "version": backend.version,
            "checkpoint_sha256": getattr(backend, "checkpoint_sha256", None),
            "checkpoint_config_sha256": getattr(
                backend, "checkpoint_config_sha256", None
            ),
        },
        "leaf_only": {
            "status": "completed" if mask_audit_passed else "blocked by mask audit",
            "sample_size": len(sample_indices),
            "fit_scores": leaf_scores,
        },
        "prediction_views": {
            fit_id: selection["selection"]["model_specs"][fit_id]["input_view"]
            for fit_id in expected_ids
        },
        "model_provenance": model_provenance,
        "leaf_prediction_view": dict.fromkeys(final_ids, "leaf-only"),
        "background_only": {
            "status": "completed" if mask_audit_passed else "blocked by mask audit",
            "fit_scores": background_scores,
        },
        "border": border_results,
        "border_cue_baselines": cue_baselines,
        "frame_leaf_shares": [
            {"image_id": entry.image_id, "leaf_pixel_share": share}
            for entry, share in zip(entries, border_leaf_shares, strict=True)
        ],
        "class_by_scene": scene_tables,
        "class_by_scene_status": (
            "completed" if mask_audit_passed else "blocked by mask audit"
        ),
    }
    output = Path(output_dir) / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _append_test_read_event(
        history_path,
        {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "selection_sha256": _digest(selection),
            "run_record_sha256": _digest(run_record),
            "read_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "evaluation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
    )
    return output

"""Immutable evaluation selections and the only authorized test-image reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from jute_disease.training import (
    RunState,
    RunStore,
    fit_identity_digest,
    load_benchmark,
)

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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _hyperparameter_key(model_spec: dict[str, Any]) -> str | None:
    """Return the #106 tuning receipt that supplies a model's frozen choice."""
    if model_spec.get("kind") == "classical":
        if "configuration" in model_spec:
            return None
        return (
            f"classical/tune/{model_spec['pool']}/"
            f"{model_spec['estimator']}-{model_spec['features']}"
        )
    if model_spec.get("kind") == "cnn":
        return f"cnn/tune/{model_spec['system']}/pool-{model_spec['pool']}"
    return None


def _selection_choice_matches(
    model_spec: dict[str, Any], selected_hyperparameters: dict[str, Any]
) -> bool:
    """Check that the fit's bound choice is the one recorded from validation."""
    kind = model_spec.get("kind")
    if kind == "cue":
        return (
            "selected_parameters" not in model_spec
            and "selected_learning_rate" not in model_spec
        )
    if kind == "classical" and "configuration" in model_spec:
        return isinstance(model_spec.get("parameters"), dict) and model_spec.get(
            "selected_parameters"
        ) == model_spec.get("parameters")
    try:
        key = _hyperparameter_key(model_spec)
    except KeyError:
        return False
    if key is None or key not in selected_hyperparameters:
        return False
    choice = selected_hyperparameters[key]
    if kind == "classical":
        return (
            isinstance(choice, dict)
            and model_spec.get("validation_hyperparameters_key") == key
            and model_spec.get("selected_parameters") == choice
        )
    return (
        kind == "cnn"
        and isinstance(choice, int | float)
        and not isinstance(choice, bool)
        and math.isfinite(choice)
        and choice > 0
        and model_spec.get("validation_hyperparameters_key") == key
        and model_spec.get("selected_learning_rate") == choice
    )


def _receipt_choice_matches(
    model_spec: dict[str, Any], results: dict[str, Any]
) -> bool:
    """Verify the completed fit used the choice bound into its model spec."""
    if model_spec.get("kind") == "classical":
        return results.get("selected_parameters") == model_spec.get(
            "selected_parameters"
        )
    if model_spec.get("kind") == "cnn":
        recipe = results.get("recipe")
        return isinstance(recipe, dict) and recipe.get(
            "learning_rate"
        ) == model_spec.get("selected_learning_rate")
    return model_spec.get("kind") == "cue" and results.get("cue") == model_spec.get(
        "cue"
    )


def _pipeline_code_tree_sha256(root: Path | None = None) -> str:
    """Hash source and configuration files that define the evaluator pipeline."""
    project_root = root or Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in ("src", "configs"):
        for file in sorted((project_root / relative).rglob("*")):
            if file.is_file() and (
                (relative == "src" and file.suffix == ".py")
                or (relative == "configs" and file.suffix in {".json", ".yaml", ".yml"})
            ):
                digest.update(file.relative_to(project_root).as_posix().encode())
                digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def _environment_lock_sha256(root: Path | None = None) -> str:
    """Hash the dependency lock file committed with the evaluator code."""
    project_root = root or Path(__file__).resolve().parents[3]
    return hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest()


def _valid_assignment_rows(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(row, dict)
            and all(
                type(row.get(key)) is int for key in ("split", "seed", "checkpoint")
            )
            and row["split"] >= 0
            and row["seed"] >= 0
            and row["checkpoint"] == (row["seed"] + row["split"]) % 3
            for row in value
        )
        and len({(row["split"], row["seed"]) for row in value}) == len(value)
    )


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
        analysis_ids = matrix["analysis_ids"]
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
        leaf_sampling_seed = leaf_only["sampling_seed"]
        border_seeds = border["seeds"]
        null_permutations = border["null_permutations"]
        training_recipe = selection["training_recipe"]
        selected_hyperparameters = selection["validation_selected_hyperparameters"]
        selected_hyperparameters_sha256 = selection[
            "validation_selected_hyperparameters_sha256"
        ]
        assignment_table = selection["assignment_table"]
        pipeline_code_tree_sha256 = selection["pipeline_code_tree_sha256"]
        environment_lock_sha256 = selection["environment_lock_sha256"]
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
        or not valid_ids(analysis_ids)
        or set(analysis_ids) != set(planned_fit_ids) | {"gradcam/test-pass"}
        or selection.get("planned_fit_ids") != planned_fit_ids
        or selection.get("analysis_ids") != analysis_ids
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
        or any(
            not _is_sha256(spec.get("fit_manifest_digest"))
            or (
                "policy_artifact_sha256" in spec
                and not _is_sha256(spec.get("policy_artifact_sha256"))
            )
            for spec in model_specs.values()
        )
        or not isinstance(selected_hyperparameters, dict)
        or not selected_hyperparameters
        or selected_hyperparameters_sha256 != _digest(selected_hyperparameters)
        or not _valid_assignment_rows(assignment_table)
        or not _is_sha256(pipeline_code_tree_sha256)
        or not _is_sha256(environment_lock_sha256)
        or type(leaf_sample_size) is not int
        or leaf_sample_size < 1
        or type(leaf_sampling_seed) is not int
        or not valid_seeds(border_seeds)
        or type(null_permutations) is not int
        or null_permutations < 1
        or not isinstance(letterbox, str)
        or letterbox not in {"letterbox", "resize-center-crop"}
        or not isinstance(preprocessing, dict)
        or preprocessing.get("resize") != letterbox
        or type(preprocessing.get("size")) is not int
        or preprocessing["size"] < 1
        or not isinstance(training_recipe, dict)
        or not isinstance(training_recipe.get("precision"), str)
        or type(training_recipe.get("num_workers")) is not int
        or training_recipe["num_workers"] < 0
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
        or not 0 <= leaf_share_cutoff <= 0.05
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
            or (
                spec["input_view"]
                not in (
                    {"pad-fraction", "aspect-ratio"}
                    if spec["kind"] == "cue"
                    else {"raw-test", "background-only", "border-frame"}
                )
            )
            or spec.get("preprocessing") != preprocessing
            or spec.get("validation_selected_hyperparameters")
            != selected_hyperparameters
            or spec.get("validation_hyperparameters_sha256")
            != selected_hyperparameters_sha256
            or not _selection_choice_matches(spec, selected_hyperparameters)
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

    pools, policies, views = (
        list(benchmark.pools),
        list(benchmark.policies),
        list(benchmark.views),
    )
    seeds = list(benchmark.recipe.seeds)
    primary = benchmark.primary_split
    cnn_systems = [
        name
        for name, definition in benchmark.systems.items()
        if definition["model"] == "mobilenetv2_100.ra_in1k" and not definition["chain"]
    ]
    if len(cnn_systems) != 1:
        raise GatedEvaluationError(
            "the frozen probe requires one MobileNetV2 ImageNet system"
        )
    cnn_system = cnn_systems[0]
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
                for view in views[:1]:
                    for policy in policies:
                        for seed in seeds[: background_config["seed_count"]]:
                            fit_id = (
                                f"probe/background/classical/{estimator}-{features}/pool-{pool}"
                                f"/view-{view}/{policy}/split-{primary}/seed-{seed}"
                            )
                            background.append(fit_id)
                            model_specs[fit_id] = {
                                "kind": "classical",
                                "estimator": estimator,
                                "features": features,
                                "pool": pool,
                                "view": view,
                                "policy": policy,
                                "split": primary,
                                "seed": seed,
                                "training_view": background_config["training_view"],
                                "input_view": "background-only",
                            }
    for pool in pools:
        for view in views[:1]:
            for policy in policies:
                for seed in seeds[: background_config["seed_count"]]:
                    fit_id = (
                        f"probe/background/mobilenetv2_100.ra_in1k/pool-{pool}"
                        f"/view-{view}/{policy}/split-{primary}/seed-{seed}"
                    )
                    background.append(fit_id)
                    model_specs[fit_id] = {
                        "kind": "cnn",
                        "system": cnn_system,
                        "model": "mobilenetv2_100.ra_in1k",
                        "pool": pool,
                        "view": view,
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
                for view in views[:1]:
                    for policy in policies:
                        for seed in seeds[: border_config["seed_count"]]:
                            for configuration_index, parameters in enumerate(
                                classical_configurations(estimator)
                            ):
                                fit_id = (
                                    f"probe/border/classical/{estimator}-{features}/pool-{pool}"
                                    f"/view-{view}/{policy}/split-{primary}/seed-{seed}"
                                    f"/config-{configuration_index:02d}"
                                )
                                border.append(fit_id)
                                model_specs[fit_id] = {
                                    "kind": "classical",
                                    "estimator": estimator,
                                    "features": features,
                                    "pool": pool,
                                    "view": view,
                                    "policy": policy,
                                    "split": primary,
                                    "seed": seed,
                                    "configuration": configuration_index,
                                    "parameters": parameters,
                                    "training_view": border_config["training_view"],
                                    "input_view": "border-frame",
                                }
    for pool in pools:
        for view in views[:1]:
            for policy in policies:
                for seed in seeds[: border_config["seed_count"]]:
                    fit_id = (
                        f"probe/border/mobilenetv2_100.ra_in1k/pool-{pool}"
                        f"/view-{view}/{policy}/split-{primary}/seed-{seed}"
                    )
                    border.append(fit_id)
                    model_specs[fit_id] = {
                        "kind": "cnn",
                        "system": cnn_system,
                        "model": "mobilenetv2_100.ra_in1k",
                        "pool": pool,
                        "view": view,
                        "policy": policy,
                        "split": primary,
                        "seed": seed,
                        "training_view": border_config["training_view"],
                        "input_view": "border-frame",
                    }
                    for cue in ("pad-fraction", "aspect-ratio"):
                        cue_id = (
                            f"probe/border/{cue}/pool-{pool}/view-{view}/{policy}"
                            f"/split-{primary}/seed-{seed}"
                        )
                        border.append(cue_id)
                        model_specs[cue_id] = {
                            "kind": "cue",
                            "cue": cue,
                            "pool": pool,
                            "view": view,
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
                "view": fit.view,
                "training_view": "raw-test",
                "input_view": "raw-test",
            }
        else:
            model_specs[fit_id] = {
                "kind": "cnn",
                "system": fit.system,
                "model": fit.model,
                "pool": fit.pool,
                "policy": fit.policy,
                "split": fit.split,
                "seed": fit.seed,
                "view": fit.view,
                "training_view": "raw-test",
                "input_view": "raw-test",
            }
    return {
        "primary_split": primary,
        "pools": pools,
        "policies": policies,
        "training_recipe": asdict(benchmark.recipe),
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
            "sampling_seed": leaf_config["sampling_seed"],
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


def _bind_validation_hyperparameters(
    plan: dict[str, Any],
    selected_hyperparameters: dict[str, Any],
) -> None:
    """Copy each validation choice onto every fit that must use that choice."""
    from jute_disease.models.ml import classical_configurations

    model_specs = plan["model_specs"]
    required_keys = {
        key
        for model_spec in model_specs.values()
        if (key := _hyperparameter_key(model_spec)) is not None
    }
    missing = required_keys - set(selected_hyperparameters)
    if missing:
        raise GatedEvaluationError(
            "#114 validation outcomes omit selected hyperparameters for "
            + ", ".join(sorted(missing))
        )
    for model_spec in model_specs.values():
        if model_spec["kind"] == "cue":
            continue
        if model_spec["kind"] == "classical" and "configuration" in model_spec:
            model_spec["selected_parameters"] = model_spec["parameters"]
            continue
        key = _hyperparameter_key(model_spec)
        if key is None:
            raise GatedEvaluationError("model has no validation hyperparameter key")
        choice = selected_hyperparameters[key]
        if model_spec["kind"] == "classical":
            if not isinstance(choice, dict) or choice not in classical_configurations(
                model_spec["estimator"]
            ):
                raise GatedEvaluationError(
                    f"#114 selected parameters are invalid for {key}"
                )
            model_spec["selected_parameters"] = choice
        else:
            if (
                isinstance(choice, bool)
                or not isinstance(choice, int | float)
                or not math.isfinite(choice)
                or choice <= 0
            ):
                raise GatedEvaluationError(
                    f"#114 selected learning rate is invalid for {key}"
                )
            model_spec["selected_learning_rate"] = float(choice)
        model_spec["validation_hyperparameters_key"] = key


def write_matrix_freeze(
    *,
    validation_outcomes: Path | str,
    benchmark_path: Path | str,
    target_dir: Path | str,
    output_path: Path | str,
    policy_dir: Path | str | None = None,
) -> tuple[Path, str]:
    """Write a hash-bound selection only after recorded #114 outcomes."""
    from jute_disease.data import (
        EVALUATOR_TEST_SOURCE_DIGEST,
        load_manifest,
        load_policy_manifest,
    )

    outcomes_path = Path(validation_outcomes)
    try:
        outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatedEvaluationError("#114 validation outcomes are required") from error
    if not isinstance(outcomes, dict) or outcomes.get("issue") != 114:
        raise GatedEvaluationError("validation outcomes must identify issue #114")
    if outcomes.get("partition") != "val":
        raise GatedEvaluationError(
            "#114 outcomes must come from the validation partition"
        )
    letterbox = outcomes.get("letterbox_comparison")
    audit = outcomes.get("mask_audit")
    budget = outcomes.get("post_pilot_budget_hours")
    scene = outcomes.get("scene_thresholds")
    frame = outcomes.get("border_frame_width_px")
    leaf_cutoff = outcomes.get("border_leaf_share_cutoff")
    hyperparameters = outcomes.get("selected_hyperparameters")
    if (
        not isinstance(letterbox, dict)
        or len(letterbox) < 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in letterbox.values()
        )
    ):
        raise GatedEvaluationError(
            "#114 outcomes need a numeric, two-candidate letterbox comparison"
        )
    choice = outcomes.get("letterbox_decision")
    if (
        not isinstance(choice, str)
        or choice not in {"letterbox", "resize-center-crop"}
        or choice not in letterbox
    ):
        raise GatedEvaluationError(
            "#114 letterbox decision must name one of the compared candidates"
        )
    letterbox = choice
    if not isinstance(audit, dict) or type(audit.get("passed")) is not bool:
        raise GatedEvaluationError("#114 mask-audit pass/fail outcome must be recorded")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise GatedEvaluationError("#114 post-pilot budget must be recorded")
    if not isinstance(scene, dict) or any(
        not isinstance(scene.get(key), (int, float)) or not 0 <= scene[key] <= 1
        for key in ("background_fraction_min", "color_uniformity_min")
    ):
        raise GatedEvaluationError("#114 scene thresholds must be recorded")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1:
        raise GatedEvaluationError("border frame width must be recorded")
    if (
        not isinstance(leaf_cutoff, (int, float))
        or isinstance(leaf_cutoff, bool)
        or not 0 <= leaf_cutoff <= 0.05
    ):
        raise GatedEvaluationError("border leaf-share cutoff must be at most 5%")
    if not isinstance(hyperparameters, dict) or not hyperparameters:
        raise GatedEvaluationError(
            "validation-selected hyperparameters must be recorded"
        )
    recorded_selections = outcomes.get("selections")
    if not isinstance(recorded_selections, dict) or any(
        not isinstance(recorded_selections.get(key), dict)
        or recorded_selections[key].get("selected") != value
        for key, value in hyperparameters.items()
    ):
        raise GatedEvaluationError(
            "validation-selected hyperparameters must match recorded #114 outcomes"
        )

    target = Path(target_dir)
    project_root = Path(__file__).resolve().parents[3]
    read_histories = {
        "evaluator": target / TEST_READ_HISTORY_FILE,
        "project": project_root / "artifacts" / "evaluation" / "test-reads.jsonl",
    }
    manifest_path = target / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = load_manifest(target)
    except (OSError, ValueError) as error:
        raise GatedEvaluationError("a completed target freeze is required") from error
    full_manifest_digest = dict(manifest.dependencies).get(EVALUATOR_TEST_SOURCE_DIGEST)
    if not _is_sha256(full_manifest_digest):
        raise GatedEvaluationError("public target manifest has no held-out commitment")
    benchmark_bytes = Path(benchmark_path).read_bytes()
    pipeline_code_tree_sha256 = _pipeline_code_tree_sha256()
    lock_hash = _environment_lock_sha256()
    plan = probe_plan(benchmark_path)
    benchmark = load_benchmark(benchmark_path)
    _bind_validation_hyperparameters(plan, hyperparameters)
    image_size = benchmark.recipe.image_size
    preprocessing = {"resize": letterbox, "size": image_size}
    validation_hyperparameters_sha256 = _digest(hyperparameters)
    for fit_id, model_spec in plan["model_specs"].items():
        model_spec["preprocessing"] = preprocessing
        model_spec["fit_manifest_digest"] = manifest.digest()
        if policy_dir is not None and all(
            key in model_spec for key in ("pool", "view", "policy", "split")
        ):
            try:
                fit_manifest = load_policy_manifest(
                    policy_dir,
                    pool=model_spec["pool"],
                    view=model_spec["view"],
                    policy=model_spec["policy"],
                    draw=model_spec["split"],
                    image_size=image_size,
                )
            except (OSError, ValueError, KeyError) as error:
                raise GatedEvaluationError(
                    f"receipted policy manifest is required for {fit_id}"
                ) from error
            model_spec["fit_manifest_digest"] = fit_manifest.digest()
            model_spec["policy_artifact_sha256"] = dict(fit_manifest.dependencies)[
                "policy_file"
            ]
        model_spec["validation_hyperparameters_sha256"] = (
            validation_hyperparameters_sha256
        )
        model_spec["validation_selected_hyperparameters"] = hyperparameters
        if fit_id in plan["final_fit_ids"]:
            model_spec["training_view"] = "raw-training"
    payload = {
        "issue": 139,
        "manifest_digest": full_manifest_digest,
        "public_manifest_digest": manifest.digest(),
        "classes": list(manifest.labels),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "assignment_table": [
            {"split": split, "seed": seed, "checkpoint": checkpoint}
            for split, seed, checkpoint in benchmark.assignments
        ],
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "validation_outcomes_sha256": hashlib.sha256(
            outcomes_path.read_bytes()
        ).hexdigest(),
        "validation_outcomes": outcomes,
        "selection": plan,
        "planned_fit_ids": plan["planned_fit_ids"],
        "analysis_ids": plan["analysis_ids"],
        "training_recipe": plan["training_recipe"],
        "validation_selected_hyperparameters": hyperparameters,
        "validation_selected_hyperparameters_sha256": (
            validation_hyperparameters_sha256
        ),
        "letterbox_decision": letterbox,
        "preprocessing": preprocessing,
        "mask_audit_outcome": "passed" if audit["passed"] else "failed",
        "border_frame_width_px": {
            "starting": plan["border"]["frame_width_px_start"],
            "final": frame,
        },
        "border_leaf_share_cutoff": {
            "starting": plan["border"]["leaf_share_cutoff_start"],
            "final": float(leaf_cutoff),
        },
        "post_pilot_budget_hours": float(budget),
        "pipeline_code_tree_sha256": pipeline_code_tree_sha256,
        "environment_lock_sha256": lock_hash,
        "created_after_test_reads": False,
    }
    existing_read_histories = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in read_histories.items()
        if path.is_file()
    }
    if existing_read_histories:
        payload["created_after_test_reads"] = True
        payload["test_read_histories"] = existing_read_histories
        payload["test_read_history_sha256"] = _digest(existing_read_histories)
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
    analysis_ids = selection["selection"]["analysis_ids"]
    record = {
        "issue": 139,
        "selection_sha256": _digest(selection),
        "status": "completed",
        "completed_analysis_ids": analysis_ids,
        "analysis_receipts": {
            "gradcam/test-pass": {
                "status": "not-implemented-by-decision",
                "leaf_score_implemented": False,
            }
        },
        "fixture_predictions": True,
    }
    output = Path(output_path)
    return output, _write_immutable(output, RUN_FORMAT, record)


def write_training_model_index(
    selection_path: Path | str,
    training_run_root: Path | str,
    output_path: Path | str,
) -> tuple[Path, str]:
    """Build the evaluator artifact index from completed queue receipts."""
    selection = _read_selection(Path(selection_path))
    selection_body = selection["selection"]
    expected = set(selection_body["planned_fit_ids"])
    extra = set(selection_body["analysis_ids"]) - expected
    if extra != {"gradcam/test-pass"}:
        raise GatedEvaluationError("selection has unsupported non-fit analyses")
    run_root = Path(training_run_root).expanduser().resolve()
    store = RunStore(run_root)
    models: dict[str, dict[str, Any]] = {}
    for fit_id in sorted(expected):
        receipt = store.receipt(fit_id, missing_ok=True)
        if receipt is None or store.state(fit_id) is not RunState.COMPLETED:
            raise GatedEvaluationError(
                f"completed run receipt is required for {fit_id}"
            )
        spec = selection_body["model_specs"][fit_id]
        artifact_name = "checkpoint" if spec["kind"] == "cnn" else "model"
        artifact = receipt.get("artifacts", {}).get(artifact_name)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise GatedEvaluationError(f"model artifact is missing for {fit_id}")
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = store.run_dir(fit_id) / artifact_path
        artifact_path = artifact_path.resolve()
        try:
            observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as error:
            raise GatedEvaluationError(
                f"model artifact is missing for {fit_id}"
            ) from error
        if observed != artifact.get("sha256"):
            raise GatedEvaluationError(f"model artifact changed for {fit_id}")
        models[fit_id] = {
            "fit_spec": spec,
            "path": str(artifact_path),
            "sha256": observed,
        }
    payload = {
        "run_root": str(run_root),
        "models": models,
        "analyses": {},
    }
    output = Path(output_path)
    _write_json_atomic(output, payload)
    return output, hashlib.sha256(output.read_bytes()).hexdigest()


def _runtime_observations_valid(runtime: Any, precision: str) -> bool:
    if not isinstance(runtime, dict):
        return False
    device = runtime.get("device")
    libraries = runtime.get("libraries")
    return (
        all(
            isinstance(runtime.get(key), str) and runtime[key]
            for key in ("python", "torch", "torch_cuda", "cuda_driver")
        )
        and runtime.get("cuda_available") is True
        and runtime.get("precision") == precision
        and runtime.get("compile_enabled") is False
        and isinstance(libraries, dict)
        and {"timm", "lightning", "scikit-learn", "torchvision"}.issubset(libraries)
        and all(isinstance(version, str) and version for version in libraries.values())
        and runtime.get("backend")
        == {
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
        }
        and isinstance(runtime.get("deterministic_algorithms"), bool)
        and isinstance(runtime.get("cudnn_deterministic"), bool)
        and isinstance(runtime.get("float32_matmul_precision"), str)
        and bool(runtime["float32_matmul_precision"])
        and isinstance(device, dict)
        and isinstance(device.get("name"), str)
        and bool(device["name"])
        and isinstance(device.get("capability"), list)
        and len(device["capability"]) == 2
        and all(type(value) is int for value in device["capability"])
        and type(device.get("memory_bytes")) is int
        and device["memory_bytes"] > 0
    )


def _verified_training_receipt(
    fit_id: str,
    model_spec: dict[str, Any],
    selection: dict[str, Any],
    store: RunStore,
) -> dict[str, Any]:
    """Require a clean, official, complete #107 receipt for one frozen fit."""
    receipt = store.receipt(fit_id, missing_ok=True)
    if receipt is None or store.state(fit_id) is not RunState.COMPLETED:
        raise GatedEvaluationError(f"completed run receipt is required for {fit_id}")
    metadata = receipt.get("metadata")
    if not isinstance(metadata, dict):
        raise GatedEvaluationError(f"run receipt metadata is missing for {fit_id}")
    attempts = receipt.get("attempts")
    results = receipt.get("results")
    preprocessing = metadata.get("preprocessing")
    provenance = metadata.get("provenance")
    configuration = metadata.get("configuration")
    environment_lock = metadata.get("environment_lock")
    assignment_table = metadata.get("assignment_table")
    ancestry = metadata.get("source_checkpoint_ancestry")
    runtime_contract = metadata.get("runtime_contract")
    run_options = metadata.get("run_options")
    identity_inputs = metadata.get("identity_inputs")
    fit_inputs = (
        identity_inputs.get("fit") if isinstance(identity_inputs, dict) else None
    )
    input_manifests = (
        identity_inputs.get("manifests") if isinstance(identity_inputs, dict) else None
    )
    input_environment = (
        identity_inputs.get("environment")
        if isinstance(identity_inputs, dict)
        else None
    )
    input_ancestry = (
        identity_inputs.get("source_checkpoint_ancestry")
        if isinstance(identity_inputs, dict)
        else None
    )
    identity = receipt.get("identity")
    assignment_rows = (
        assignment_table.get("assignments")
        if isinstance(assignment_table, dict)
        else None
    )
    preprocessing_spec = (
        preprocessing.get("spec") if isinstance(preprocessing, dict) else None
    )
    preprocessing_digest = (
        hashlib.sha256(
            json.dumps(preprocessing_spec, sort_keys=True).encode()
        ).hexdigest()
        if isinstance(preprocessing_spec, dict)
        else None
    )
    recipe = selection["training_recipe"]
    expected_runtime_contract = {
        "precision": recipe["precision"],
        "tf32_matmul": True,
        "cudnn_tf32": True,
        "cudnn_benchmark": False,
        "torch_compile": False,
        "num_workers": recipe["num_workers"],
    }
    try:
        identity_digest = fit_identity_digest(identity_inputs)
    except TypeError, ValueError:
        identity_digest = None
    assignment_content_sha256 = (
        hashlib.sha256(
            json.dumps(assignment_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if isinstance(assignment_rows, list)
        else None
    )
    code_commit = provenance.get("commit") if isinstance(provenance, dict) else None
    upstream_sha256 = (
        identity_inputs.get("upstream_sha256")
        if isinstance(identity_inputs, dict)
        else None
    )
    ancestry_valid = isinstance(ancestry, list) and all(
        isinstance(row, dict)
        and isinstance(row.get("fit_id"), str)
        and bool(row["fit_id"])
        and _is_sha256(row.get("identity"))
        and _is_sha256(row.get("checkpoint_sha256"))
        for row in ancestry
    )
    fit_matches_spec = isinstance(fit_inputs, dict) and all(
        fit_inputs.get(key) == model_spec.get(key)
        for key in (
            "seed",
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
    )
    result_fit = results.get("fit") if isinstance(results, dict) else None
    result_fit_id = (
        result_fit.get("fit_id") if isinstance(result_fit, dict) else result_fit
    )
    if (
        metadata.get("official") is not True
        or not _is_sha256(identity)
        or identity_digest != identity
        or not isinstance(identity_inputs, dict)
        or identity_inputs.get("evaluation_fit_spec") != model_spec
        or identity_inputs.get("evaluation_selection_sha256") != _digest(selection)
        or not isinstance(fit_inputs, dict)
        or fit_inputs.get("fit_id") != fit_id
        or not fit_matches_spec
        or not isinstance(results, dict)
        or result_fit_id != fit_id
        or results.get("evaluation_fit_spec") != model_spec
        or not _receipt_choice_matches(model_spec, results)
        or not isinstance(identity_inputs.get("recipe"), dict)
        or identity_inputs.get("recipe") != recipe
        or not isinstance(input_manifests, dict)
        or input_manifests.get("target") != selection.get("manifest_digest")
        or input_manifests.get("fit") != model_spec.get("fit_manifest_digest")
        or not input_manifests
        or any(not _is_sha256(value) for value in input_manifests.values())
        or identity_inputs.get("config_sha256") != selection["benchmark_sha256"]
        or identity_inputs.get("environment_lock_sha256")
        != selection["environment_lock_sha256"]
        or identity_inputs.get("preprocessing_sha256")
        != (preprocessing.get("sha256") if isinstance(preprocessing, dict) else None)
        or not _is_sha256(preprocessing_digest)
        or preprocessing_digest != preprocessing.get("sha256")
        or not isinstance(preprocessing_spec, dict)
        or preprocessing_spec.get("size") != selection["preprocessing"]["size"]
        or preprocessing_spec.get("resize") != selection["preprocessing"]["resize"]
        or not isinstance(provenance, dict)
        or provenance.get("official") is not True
        or provenance.get("working_tree_clean") is not True
        or not isinstance(code_commit, str)
        or len(code_commit) not in {40, 64}
        or any(char not in "0123456789abcdef" for char in code_commit)
        or identity_inputs.get("code_commit") != code_commit
        or not isinstance(configuration, dict)
        or configuration.get("sha256") != selection["benchmark_sha256"]
        or not isinstance(environment_lock, dict)
        or environment_lock.get("sha256") != selection["environment_lock_sha256"]
        or metadata.get("seed") != model_spec.get("seed")
        or not isinstance(assignment_table, dict)
        or assignment_table.get("present") is not True
        or assignment_table.get("validated") is not True
        or not _is_sha256(assignment_table.get("sha256"))
        or not _valid_assignment_rows(assignment_rows)
        or assignment_rows != selection.get("assignment_table")
        or assignment_table.get("content_sha256") != assignment_content_sha256
        or identity_inputs.get("assignment_table_sha256")
        != assignment_table.get("sha256")
        or not ancestry_valid
        or input_ancestry != ancestry
        or (upstream_sha256 is not None and not _is_sha256(upstream_sha256))
        or (upstream_sha256 is None and ancestry)
        or (
            upstream_sha256 is not None
            and (not ancestry or ancestry[-1]["checkpoint_sha256"] != upstream_sha256)
        )
        or runtime_contract != expected_runtime_contract
        or not isinstance(run_options, dict)
        or run_options.get("pretrained") is not True
        or run_options.get("require_cuda") is not True
        or run_options.get("accelerator") != "cuda"
        or run_options.get("fixture_masks") is not False
        or (
            model_spec.get("training_view") == "background-only"
            and not _is_sha256(run_options.get("mask_checkpoint_sha256"))
        )
        or not isinstance(input_environment, dict)
        or any(
            input_environment.get(key) != run_options.get(key) for key in run_options
        )
        or not isinstance(attempts, list)
        or not attempts
        or not all(isinstance(attempt, dict) for attempt in attempts)
        or attempts[-1].get("metadata") != metadata
        or any(
            not isinstance(attempt.get("metadata"), dict)
            or attempt["metadata"].get("identity_inputs") != identity_inputs
            or attempt["metadata"].get("official") is not True
            or attempt["metadata"].get("provenance") != provenance
            or not _runtime_observations_valid(
                attempt["metadata"].get("runtime"), recipe["precision"]
            )
            or (
                attempt.get("state") != RunState.COMPLETED.value
                and not isinstance(attempt.get("reason"), str)
            )
            for attempt in attempts
        )
        or not _runtime_observations_valid(metadata.get("runtime"), recipe["precision"])
    ):
        raise GatedEvaluationError(f"run receipt does not satisfy #107 for {fit_id}")
    return receipt


def _verify_analysis_receipt(
    analysis_id: str, row: Any, index_path: Path
) -> dict[str, str]:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
        raise GatedEvaluationError(
            f"analysis receipt path is missing for {analysis_id}"
        )
    path = Path(row["path"])
    if not path.is_absolute():
        path = index_path.parent / path
    path = path.resolve()
    try:
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, ValueError) as error:
        raise GatedEvaluationError(
            f"completed analysis receipt is required for {analysis_id}"
        ) from error
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    if (
        not isinstance(receipt, dict)
        or receipt.get("analysis_id") != analysis_id
        or receipt.get("status") != "passed"
        or row.get("sha256") != digest
    ):
        raise GatedEvaluationError(f"analysis receipt is invalid for {analysis_id}")
    return {"path": str(path), "sha256": digest}


def write_run_record(
    selection_path: Path | str,
    model_index_path: Path | str,
    output_path: Path | str,
) -> tuple[Path, str]:
    """Bind every frozen analysis ID to completed receipts and model artifacts."""
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
    analysis_ids = selection["selection"]["analysis_ids"]
    extra_analysis_ids = set(analysis_ids) - expected
    model_specs = selection["selection"].get("model_specs")
    if not isinstance(model_specs, dict) or set(model_specs) != expected:
        raise GatedEvaluationError("selection lacks a model spec for every analysis")
    if not isinstance(models, dict) or set(models) != expected:
        raise GatedEvaluationError(
            "model-artifact index must cover every frozen fit exactly once"
        )
    run_root_value = index.get("run_root") if isinstance(index, dict) else None
    if not isinstance(run_root_value, str):
        raise GatedEvaluationError("a completed training run root is required")
    run_root = Path(run_root_value)
    if not run_root.is_absolute():
        run_root = index_file.parent / run_root
    run_root = run_root.resolve()
    store = RunStore(run_root)
    artifacts: dict[str, dict[str, str]] = {}
    training_receipts: dict[str, dict[str, str]] = {}
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
        receipt = _verified_training_receipt(
            fit_id, model_specs[fit_id], selection, store
        )
        run_artifacts = receipt["artifacts"]
        receipt_contains_model = any(
            isinstance(receipt_artifact, dict)
            and isinstance(receipt_artifact.get("path"), str)
            and receipt_artifact.get("sha256") == digest
            and (
                Path(receipt_artifact["path"])
                if Path(receipt_artifact["path"]).is_absolute()
                else store.run_dir(fit_id) / receipt_artifact["path"]
            ).resolve()
            == path
            for receipt_artifact in run_artifacts.values()
        )
        if not receipt_contains_model:
            raise GatedEvaluationError(
                f"model artifact is not bound by the training receipt for {fit_id}"
            )
        receipt_path = store.run_dir(fit_id) / "receipt.json"
        training_receipts[fit_id] = {
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "identity": receipt["identity"],
        }
        artifacts[fit_id] = {
            "path": str(path),
            "sha256": digest,
            "fit_spec_sha256": _digest(model_specs[fit_id]),
        }
    gradcam_id = "gradcam/test-pass"
    selection_gradcam = selection["selection"].get("gradcam_test_pass")
    if (
        gradcam_id not in extra_analysis_ids
        or not isinstance(selection_gradcam, dict)
        or selection_gradcam.get("planned") is not True
        or selection_gradcam.get("leaf_score_implemented") is not False
    ):
        raise GatedEvaluationError(
            "selection must freeze the no-Grad-CAM-score test-pass decision"
        )
    receipt_analysis_ids = extra_analysis_ids - {gradcam_id}
    analysis_rows = index.get("analyses") if isinstance(index, dict) else None
    if (
        not isinstance(analysis_rows, dict)
        or set(analysis_rows) != receipt_analysis_ids
    ):
        raise GatedEvaluationError(
            "analysis index must cover every non-fit analysis exactly once"
        )
    analysis_receipts = {
        analysis_id: _verify_analysis_receipt(
            analysis_id, analysis_rows[analysis_id], index_file
        )
        for analysis_id in sorted(extra_analysis_ids)
        if analysis_id != gradcam_id
    }
    analysis_receipts[gradcam_id] = {
        "status": "not-implemented-by-decision",
        "leaf_score_implemented": False,
    }
    record = {
        "issue": 139,
        "selection_sha256": _digest(selection),
        "status": "completed",
        "completed_analysis_ids": analysis_ids,
        "model_artifacts": artifacts,
        "training_run_root": str(run_root),
        "training_receipts": training_receipts,
        "analysis_receipts": analysis_receipts,
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
    expected_analysis_ids = set(selection["selection"]["analysis_ids"])
    completed = run_record.get("completed_analysis_ids")
    if (
        not isinstance(completed, list)
        or not all(isinstance(item, str) and item for item in completed)
        or len(completed) != len(set(completed))
        or set(completed) != expected_analysis_ids
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
                "run record must bind every frozen fit to a model artifact"
            )
        run_root_value = run_record.get("training_run_root")
        training_receipts = run_record.get("training_receipts")
        if (
            not isinstance(run_root_value, str)
            or not isinstance(training_receipts, dict)
            or set(training_receipts) != expected
        ):
            raise GatedEvaluationError(
                "run record must bind every frozen fit to its training receipt"
            )
        store = RunStore(run_root_value)
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
            receipt_path = store.run_dir(fit_id) / "receipt.json"
            receipt_record = training_receipts[fit_id]
            if (
                not isinstance(receipt_record, dict)
                or not receipt_path.is_file()
                or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                != receipt_record.get("sha256")
            ):
                raise GatedEvaluationError(
                    f"training receipt is missing or changed for {fit_id}"
                )
            receipt = _verified_training_receipt(
                fit_id, model_specs[fit_id], selection, store
            )
            if receipt.get("identity") != receipt_record.get("identity"):
                raise GatedEvaluationError(
                    f"training receipt identity changed for {fit_id}"
                )
        expected_extra_analyses = expected_analysis_ids - expected
        analysis_receipts = run_record.get("analysis_receipts")
        if (
            not isinstance(analysis_receipts, dict)
            or set(analysis_receipts) != expected_extra_analyses
        ):
            raise GatedEvaluationError(
                "run record must bind every non-fit analysis to its result receipt"
            )
        for analysis_id, receipt_record in analysis_receipts.items():
            if analysis_id == "gradcam/test-pass":
                if receipt_record != {
                    "status": "not-implemented-by-decision",
                    "leaf_score_implemented": False,
                }:
                    raise GatedEvaluationError(
                        "Grad-CAM test-pass must preserve the no-score decision"
                    )
                continue
            if (
                not isinstance(receipt_record, dict)
                or not isinstance(receipt_record.get("path"), str)
                or not isinstance(receipt_record.get("sha256"), str)
            ):
                raise GatedEvaluationError(
                    f"analysis receipt is invalid for {analysis_id}"
                )
            receipt_path = Path(receipt_record["path"])
            if not receipt_path.is_file():
                raise GatedEvaluationError(
                    f"analysis receipt is missing for {analysis_id}"
                )
            receipt_bytes = receipt_path.read_bytes()
            try:
                analysis_receipt = json.loads(receipt_bytes)
            except ValueError as error:
                raise GatedEvaluationError(
                    f"analysis receipt is invalid for {analysis_id}"
                ) from error
            if (
                hashlib.sha256(receipt_bytes).hexdigest() != receipt_record["sha256"]
                or not isinstance(analysis_receipt, dict)
                or analysis_receipt.get("analysis_id") != analysis_id
                or analysis_receipt.get("status") != "passed"
            ):
                raise GatedEvaluationError(
                    f"analysis receipt is invalid for {analysis_id}"
                )
    return selection, run_record


def _restore_gated_test_manifest(
    directory: Path,
    public_manifest: Any,
    selection_path: Path,
    run_path: Path,
    *,
    fixture: bool,
) -> Any:
    """Restore held-out assignments only after rechecking both access records."""
    from jute_disease.data import (
        EVALUATOR_TEST_MANIFEST,
        EVALUATOR_TEST_MANIFEST_FORMAT,
        EVALUATOR_TEST_SOURCE_DIGEST,
        Manifest,
        ManifestEntry,
        validate_manifest,
    )

    selection, run_record = _test_access_records(
        selection_path, run_path, fixture=fixture
    )
    try:
        envelope = json.loads(
            (Path(directory) / EVALUATOR_TEST_MANIFEST).read_text(encoding="utf-8")
        )
        if not isinstance(envelope, dict) or not isinstance(
            envelope.get("record"), dict
        ):
            raise ValueError("invalid envelope")
        record = envelope["record"]
        entries = tuple(ManifestEntry(**row) for row in record["entries"])
        constraints = tuple(tuple(pair) for pair in record["constraints"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GatedEvaluationError(
            "separate evaluator test manifest is invalid"
        ) from error
    actual_record_digest = hashlib.sha256(_canonical(record)).hexdigest()
    dependencies = dict(public_manifest.dependencies)
    source_digest = dependencies.pop(EVALUATOR_TEST_SOURCE_DIGEST, None)
    visible_ids = {entry.image_id for entry in public_manifest.entries}
    if (
        not isinstance(public_manifest, Manifest)
        or _digest(selection) != run_record.get("selection_sha256")
        or run_record.get("status") != "completed"
        or envelope.get("format") != EVALUATOR_TEST_MANIFEST_FORMAT
        or envelope.get("sha256") != actual_record_digest
        or record.get("partition") != "test"
        or record.get("version") != public_manifest.version
        or record.get("public_manifest_digest") != public_manifest.digest()
        or record.get("source_manifest_digest") != source_digest
        or source_digest != selection.get("manifest_digest")
        or any(entry.partition != "test" for entry in entries)
        or len({entry.image_id for entry in entries}) != len(entries)
        or {entry.image_id for entry in entries} & visible_ids
        or any(
            len(pair) != 2 or not all(isinstance(item, str) for item in pair)
            for pair in constraints
        )
    ):
        raise GatedEvaluationError("separate evaluator test manifest is stale")
    full_manifest = replace(
        public_manifest,
        entries=tuple(
            sorted(
                (*public_manifest.entries, *entries), key=lambda entry: entry.image_id
            )
        ),
        constraints=tuple(sorted((*public_manifest.constraints, *constraints))),
        dependencies=tuple(sorted(dependencies.items())),
    )
    if full_manifest.digest() != selection.get("manifest_digest"):
        raise GatedEvaluationError(
            "separate evaluator test manifest does not match the frozen selection"
        )
    validate_manifest(full_manifest, 1)
    return full_manifest


def _policy_test_entries(
    policy_dir: Path,
    model_spec: dict[str, Any],
    selection: dict[str, Any],
) -> list[Any]:
    """Read one receipt-bound policy test support after both evaluator gates."""
    from jute_disease.data import ManifestEntry, letterbox_pad_fraction

    relative = (
        Path(model_spec["pool"])
        / f"view{model_spec['view']}"
        / str(model_spec["policy"])
        / f"draw-{model_spec['split']}.json"
    )
    artifact_path = policy_dir / relative
    try:
        artifact_bytes = artifact_path.read_bytes()
        receipt = json.loads((policy_dir / "receipt.json").read_text(encoding="utf-8"))
        payload = json.loads(artifact_bytes)
    except (OSError, ValueError) as error:
        raise GatedEvaluationError(
            f"receipted test support is missing for {relative.as_posix()}"
        ) from error
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    files = receipt.get("files") if isinstance(receipt, dict) else None
    lineage = payload.get("lineage") if isinstance(payload, dict) else None
    if (
        digest != model_spec.get("policy_artifact_sha256")
        or not isinstance(receipt, dict)
        or not isinstance(files, dict)
        or files.get(relative.as_posix()) != digest
        or not isinstance(payload, dict)
        or payload.get("pool") != model_spec["pool"]
        or payload.get("view") != model_spec["view"]
        or payload.get("policy") != model_spec["policy"]
        or payload.get("draw") != model_spec["split"]
        or not isinstance(lineage, dict)
        or lineage.get("manifest_digest") != selection.get("public_manifest_digest")
    ):
        raise GatedEvaluationError(
            f"policy artifact is not bound to the frozen support: {relative.as_posix()}"
        )
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise GatedEvaluationError("policy artifact entries are invalid")
    entries = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("partition") not in {
            "train",
            "val",
            "test",
        }:
            raise GatedEvaluationError(
                "policy artifact contains an invalid partition row"
            )
        if row["partition"] != "test":
            continue
        try:
            image_id = row["sha256"]
            width, height = row["width"], row["height"]
            if (
                not isinstance(image_id, str)
                or not _is_sha256(image_id)
                or image_id in seen
                or type(width) is not int
                or type(height) is not int
                or width < 1
                or height < 1
                or row["label"] not in selection["classes"]
                or not isinstance(row["family"], str)
                or not row["family"]
                or not isinstance(row["source"], str)
                or not row["source"]
                or not isinstance(row["path"], str)
                or not row["path"]
            ):
                raise ValueError("invalid policy test row")
            entries.append(
                ManifestEntry(
                    image_id=image_id,
                    path=row["path"],
                    label=row["label"],
                    partition="test",
                    group=row["family"],
                    source=row["source"],
                    width=width,
                    height=height,
                    pad_fraction=letterbox_pad_fraction(
                        width, height, selection["preprocessing"]["size"]
                    ),
                )
            )
            seen.add(image_id)
        except (KeyError, TypeError, ValueError) as error:
            raise GatedEvaluationError(
                "policy test support contains an invalid row"
            ) from error
    if not entries:
        raise GatedEvaluationError(
            f"policy test support is empty: {relative.as_posix()}"
        )
    return sorted(entries, key=lambda entry: entry.image_id)


def _fixture_prediction(fit_id: str, image_id: str, classes: tuple[str, ...]) -> str:
    digest = hashlib.sha256(f"{fit_id}\0{image_id}".encode()).digest()
    return classes[int.from_bytes(digest[:8], "big") % len(classes)]


def _sample_indices(
    entries: list[Any], scene_tags: list[str], sample_size: int, seed: int
) -> list[int]:
    if len(entries) <= sample_size:
        return list(range(len(entries)))
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (entry, scene) in enumerate(zip(entries, scene_tags, strict=True)):
        strata[(entry.label, scene)].append(index)
    exact_quotas = {
        key: sample_size * len(indices) / len(entries)
        for key, indices in strata.items()
    }
    quotas = {key: int(quota) for key, quota in exact_quotas.items()}
    remaining = sample_size - sum(quotas.values())
    remainder_order = sorted(
        strata,
        key=lambda key: (-(exact_quotas[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1
    selected: list[int] = []
    for key, indices in sorted(strata.items()):
        ordered = sorted(
            indices,
            key=lambda i: (
                hashlib.sha256(f"{seed}\0{entries[i].image_id}".encode()).digest(),
                entries[i].image_id,
            ),
        )
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


def _support_seed(family_ids: list[str], purpose: str) -> int:
    """Give identical family supports the same deterministic resampling seed."""
    digest = hashlib.sha256(_canonical({"purpose": purpose, "families": family_ids}))
    return int.from_bytes(digest.digest()[:8], "big")


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
        represented_counts = [value for value in by_class.values() if value > 0]
        if represented_counts and all(value >= 30 for value in represented_counts):
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


def _predict_model_artifact(
    spec: dict[str, Any],
    artifact_path: Path,
    entries: list[Any],
    classes: tuple[str, ...],
    view_name: str,
    read_views: Callable[
        [Any, str], tuple[dict[str, np.ndarray], np.ndarray, float, float, str]
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
            views, valid, _, pad_fraction, _ = read_views(entry, spec["pool"])
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
    policy_dir: Path | str | None = None,
    pool_roots: dict[str, Path | str] | None = None,
) -> Path:
    """Score frozen model artifacts; this is the only test-image reader."""
    from jute_disease.data import (
        EVALUATOR_TEST_SOURCE_DIGEST,
        ColorThresholdBackend,
        ManifestEntry,
        Sam2Refiner,
        classical_pipeline,
        classify_scene,
        extract_scene_features,
        file_sha256,
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
    try:
        pipeline_code_tree_sha256 = _pipeline_code_tree_sha256()
        environment_lock_sha256 = _environment_lock_sha256()
    except OSError as error:
        raise GatedEvaluationError(
            "frozen evaluator code or environment is unavailable"
        ) from error
    if pipeline_code_tree_sha256 != selection["pipeline_code_tree_sha256"]:
        raise GatedEvaluationError("evaluator code differs from the frozen pipeline")
    if environment_lock_sha256 != selection["environment_lock_sha256"]:
        raise GatedEvaluationError("environment lock differs from the frozen pipeline")
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
    if dict(manifest.dependencies).get(EVALUATOR_TEST_SOURCE_DIGEST) != selection.get(
        "manifest_digest"
    ):
        raise GatedEvaluationError("selection does not match the target freeze")
    if list(manifest.labels) != selection.get("classes"):
        raise GatedEvaluationError(
            "selection class order differs from the target freeze"
        )
    classes = tuple(manifest.labels)
    if fixture:
        full_manifest = _restore_gated_test_manifest(
            Path(target_dir),
            manifest,
            selection_file,
            run_file,
            fixture=True,
        )
        if full_manifest.digest() != selection.get("manifest_digest"):
            raise GatedEvaluationError(
                "selection does not match the complete target freeze"
            )
        fixture_entries = sorted(
            full_manifest.partition("test"), key=lambda entry: entry.image_id
        )
        if not fixture_entries:
            raise GatedEvaluationError("frozen test partition is empty")
        entries_by_fit = dict.fromkeys(
            selection["selection"]["planned_fit_ids"], fixture_entries
        )
    else:
        if manifest.digest() != selection.get("public_manifest_digest"):
            raise GatedEvaluationError(
                "selection does not match the public target manifest"
            )
        if policy_dir is None:
            raise GatedEvaluationError(
                "production evaluation requires receipted policy test supports"
            )
        entries_by_fit = {
            fit_id: _policy_test_entries(Path(policy_dir), spec, selection)
            for fit_id, spec in selection["selection"]["model_specs"].items()
            if fit_id in selection["selection"]["planned_fit_ids"]
        }
        if set(entries_by_fit) != set(selection["selection"]["planned_fit_ids"]):
            raise GatedEvaluationError(
                "the frozen selection has no policy support for every planned fit"
            )
    pool_for_fit = {
        fit_id: ""
        if fixture
        else str(selection["selection"]["model_specs"][fit_id].get("pool", ""))
        for fit_id in entries_by_fit
    }
    entry_by_key: dict[tuple[str, str], ManifestEntry] = {}
    for fit_id, fit_entries in entries_by_fit.items():
        pool = pool_for_fit[fit_id]
        for entry in fit_entries:
            entry_by_key[(pool, entry.image_id)] = entry
    support_keys = sorted(entry_by_key)
    entries = [entry_by_key[key] for key in support_keys]
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
    transform = classical_pipeline(
        size=image_size, include_mask=True, resize=resize_decision
    )
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
    frame_cache = np.memmap(
        temporary_dir / "border-frame.dat",
        mode="w+",
        dtype=np.uint8,
        shape=(len(entries), packed_width),
    )
    entry_indices = {key: index for index, key in enumerate(support_keys)}
    prepared: dict[
        tuple[str, str], tuple[tuple[int, int, int, int], float, float, str]
    ] = {}

    def read_views(
        entry: ManifestEntry,
        pool: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, float, str]:
        key = (pool, entry.image_id)
        cache_index = entry_indices[key]
        if key not in prepared:
            roots = pool_roots or {}
            image_root_path = Path(roots.get(pool, image_root)).resolve()
            image_path = (image_root_path / entry.path).resolve()
            if not image_path.is_relative_to(image_root_path):
                raise GatedEvaluationError(
                    f"frozen test path escapes its image root: {entry.image_id}"
                )
            try:
                if file_sha256(image_path) != entry.image_id:
                    raise GatedEvaluationError(
                        f"frozen test image hash changed: {entry.image_id}"
                    )
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
            boxed = transform(original)
            pixels, valid = boxed.pixels, boxed.valid_mask
            y, x = np.where(valid)
            top, bottom, left, right = (
                int(y.min()),
                int(y.max()) + 1,
                int(x.min()),
                int(x.max()) + 1,
            )
            resized_mask = transform.resize_mask(mask)
            pad_fraction = 1 - float(valid.mean())
            frame = np.zeros((image_size, image_size), dtype=bool)
            frame[top : min(bottom, top + frame_width), left:right] = True
            frame[max(top, bottom - frame_width) : bottom, left:right] = True
            frame[top:bottom, left : min(right, left + frame_width)] = True
            frame[top:bottom, max(left, right - frame_width) : right] = True
            leaf_share = float(np.logical_and(frame, resized_mask).sum() / frame.sum())
            pixels_cache[cache_index] = pixels
            mask_cache[cache_index] = np.packbits(resized_mask.ravel())
            frame_cache[cache_index] = np.packbits(frame.ravel())
            prepared[key] = (
                (top, bottom, left, right),
                leaf_share,
                pad_fraction,
                scene,
            )
        bounds, leaf_share, pad_fraction, scene = prepared[key]
        pixels = pixels_cache[cache_index]
        top, bottom, left, right = bounds
        valid = np.zeros((image_size, image_size), dtype=bool)
        valid[top:bottom, left:right] = True
        resized_mask = (
            np.unpackbits(mask_cache[cache_index], count=image_size * image_size)
            .reshape((image_size, image_size))
            .astype(bool)
        )
        frame = (
            np.unpackbits(frame_cache[cache_index], count=image_size * image_size)
            .reshape((image_size, image_size))
            .astype(bool)
        )
        views = {
            "raw-test": pixels,
            "background-only": np.where(resized_mask[..., None], pad_rgb, pixels),
            "leaf-only": np.where(resized_mask[..., None], pixels, pad_rgb),
            "border-frame": np.where(frame[..., None], pixels, pad_rgb),
        }
        return views, valid, leaf_share, pad_fraction, scene

    image_view_hashes = {
        name: hashlib.sha256()
        for name in ("raw-test", "background-only", "leaf-only", "border-frame")
    }
    for pool, image_id in support_keys:
        entry = entry_by_key[(pool, image_id)]
        views, _, _, _, _ = read_views(entry, pool)
        for name, view_pixels in views.items():
            image_view_hashes[name].update(f"{pool}:{entry.image_id}".encode("ascii"))
            image_view_hashes[name].update(
                np.asarray(view_pixels, dtype=np.uint8).tobytes()
            )

    expected_ids = selection["selection"]["planned_fit_ids"]

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
    mask_audit_passed = selection["mask_audit_outcome"] == "passed"
    sample_target = selection["selection"]["leaf_only"]["sample_size"]
    sample_seed = selection["selection"]["leaf_only"]["sampling_seed"]

    def support_context(
        fit_id: str,
    ) -> tuple[list[Any], list[str], list[float], list[float]]:
        fit_entries = entries_by_fit[fit_id]
        pool = pool_for_fit[fit_id]
        metadata = [prepared[(pool, entry.image_id)] for entry in fit_entries]
        return (
            fit_entries,
            [item[3] for item in metadata],
            [item[1] for item in metadata],
            [item[2] for item in metadata],
        )

    def support_digest(fit_id: str) -> str:
        return hashlib.sha256(
            _canonical([entry.image_id for entry in entries_by_fit[fit_id]])
        ).hexdigest()

    sample_sizes: dict[str, int] = {}
    sample_image_ids: dict[str, list[str]] = {}
    full_sample_scores: dict[str, Any] | None = {} if mask_audit_passed else None
    leaf_scores: dict[str, Any] | None = {} if mask_audit_passed else None
    background_scores = {} if mask_audit_passed else None
    scene_tables: dict[str, Any] = {}
    for fit_id in final_ids:
        fit_entries, fit_scenes, _, _ = support_context(fit_id)
        scene_tables[fit_id] = _scene_report(
            fit_entries, fit_scenes, predictions_for(fit_id, fit_entries), classes
        )
    if mask_audit_passed:
        for fit_id in final_ids:
            fit_entries, fit_scenes, _, _ = support_context(fit_id)
            indices = _sample_indices(
                fit_entries,
                fit_scenes,
                min(sample_target, len(fit_entries)),
                sample_seed,
            )
            sampled = [fit_entries[index] for index in indices]
            sample_sizes[fit_id] = len(sampled)
            sample_image_ids[fit_id] = [entry.image_id for entry in sampled]
            full_sample_scores[fit_id] = _metrics(
                sampled, predictions_for(fit_id, sampled), classes
            )
            leaf_scores[fit_id] = _metrics(
                sampled,
                predictions_for(fit_id, sampled, leaf=True),
                classes,
            )
        for fit_id in background_ids:
            fit_entries = entries_by_fit[fit_id]
            background_scores[fit_id] = _metrics(
                fit_entries, predictions_for(fit_id, fit_entries), classes
            )

    border_groups: dict[str, list[str]] = defaultdict(list)
    for fit_id in border_ids:
        group_id = "/".join(
            component
            for component in fit_id.split("/")
            if not component.startswith("seed-")
        )
        border_groups[group_id].append(fit_id)
    null_count = selection["selection"]["border"]["null_permutations"]
    border_seeds = selection["selection"]["border"]["seeds"]
    cutoff = selection["border_leaf_share_cutoff"]["final"]
    border_results: dict[str, Any] = {}
    cue_baselines: dict[str, Any] = {}
    for group_id, unsorted_fit_ids in sorted(border_groups.items()):
        fit_ids = sorted(
            unsorted_fit_ids,
            key=lambda fit_id: border_seeds.index(
                selection["selection"]["model_specs"][fit_id]["seed"]
            ),
        )
        if len(fit_ids) != len(border_seeds):
            raise GatedEvaluationError(
                f"border model group does not cover every frozen seed: {group_id}"
            )
        group_spec = {
            key: value
            for key, value in selection["selection"]["model_specs"][fit_ids[0]].items()
            if key != "seed"
        }
        group_entries, group_scenes, leaf_shares, pad_fractions = support_context(
            fit_ids[0]
        )
        expected_support = [entry.image_id for entry in group_entries]
        if any(
            [entry.image_id for entry in entries_by_fit[fit_id]] != expected_support
            for fit_id in fit_ids[1:]
        ):
            raise GatedEvaluationError(
                f"border seeds do not share one frozen test support: {group_id}"
            )
        truths = [entry.label for entry in group_entries]
        family_ids = sorted({entry.group for entry in group_entries})
        class_indices = {label: index for index, label in enumerate(classes)}
        family_labels: dict[str, int] = {}
        for entry in group_entries:
            label_index = class_indices[entry.label]
            if (
                entry.group in family_labels
                and family_labels[entry.group] != label_index
            ):
                raise GatedEvaluationError("a family spans multiple true classes")
            family_labels[entry.group] = label_index
        family_label_vector = np.asarray(
            [family_labels[family] for family in family_ids], dtype=np.int64
        )
        full_support_seed = _support_seed(family_ids, "border-full-support")
        null_rng = np.random.default_rng(
            _support_seed(family_ids, "border-full-label-permutation")
        )
        permuted_family_labels = np.vstack(
            [null_rng.permutation(family_label_vector) for _ in range(null_count)]
        )
        seed_draws, family_weights = _hierarchical_bootstrap_plan(
            len(border_seeds),
            len(family_ids),
            resamples=null_count,
            seed=full_support_seed,
        )
        retained = [index for index, share in enumerate(leaf_shares) if share <= cutoff]
        retained_entries = [group_entries[index] for index in retained]
        retained_scenes = [group_scenes[index] for index in retained]
        retained_class_counts = dict.fromkeys(classes, 0)
        retained_class_counts.update(Counter(entry.label for entry in retained_entries))
        retained_scene_counts = dict.fromkeys(sorted(set(group_scenes)), 0)
        retained_scene_counts.update(Counter(retained_scenes))
        headline_allowed = mask_audit_passed and all(
            retained_class_counts.get(label, 0) >= 50 for label in classes
        )
        retained_family_ids = sorted({entry.group for entry in retained_entries})
        seed_metrics: dict[str, dict[str, Any]] = {}
        family_stats_by_seed = []
        predicted_counts_by_seed = []
        retained_stats_by_seed = []
        retained_predicted_counts_by_seed = []
        retained_seed_metrics: dict[str, dict[str, Any]] = {}
        retained_family_order: list[str] | None = None
        is_cue = group_spec["kind"] == "cue"
        for fit_id in fit_ids:
            seed = selection["selection"]["model_specs"][fit_id]["seed"]
            predicted = predictions_for(fit_id, group_entries)
            seed_metrics[str(seed)] = _metrics(group_entries, predicted, classes)
            current_families, family_stats, family_predictions, _ = (
                _family_metric_counts(group_entries, predicted, classes)
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
                (
                    current_retained_families,
                    retained_stats,
                    retained_predictions_by_family,
                    _,
                ) = _family_metric_counts(
                    retained_entries, retained_predictions, classes
                )
                if retained_family_order is None:
                    retained_family_order = current_retained_families
                elif current_retained_families != retained_family_order:
                    raise GatedEvaluationError(
                        "retained family order changed between border seeds"
                    )
                retained_stats_by_seed.append(retained_stats)
                retained_predicted_counts_by_seed.append(retained_predictions_by_family)

        full_intervals = _hierarchical_intervals(
            np.stack(family_stats_by_seed),
            seed_draws,
            family_weights,
            len(classes),
        )
        full_null = _family_permutation_macro_f1(
            np.stack(predicted_counts_by_seed),
            permuted_family_labels,
            len(classes),
        )
        full_lower = (
            full_intervals["macro_f1"]["lower"] if full_intervals["macro_f1"] else None
        )
        full_upper_null = float(np.percentile(full_null, 95))
        unrestricted = _mean_seed_metrics(seed_metrics)
        if is_cue:
            cue_baselines[group_id] = {
                "fit_ids": fit_ids,
                "fit_spec": group_spec,
                "metrics": unrestricted,
                "accuracy_ci_95": full_intervals["accuracy"],
                "macro_f1_ci_95": full_intervals["macro_f1"],
                "majority_class_accuracy": max(Counter(truths).values()) / len(truths),
                "permutation_macro_f1_95th": full_upper_null,
                "rule": "cue baseline exceeds the family-label null"
                if full_lower is not None and full_lower > full_upper_null
                else "cue baseline does not exceed the family-label null",
            }
            continue

        symptom_survival: dict[str, Any] | None = None
        survival_intervals: dict[str, Any] | None = None
        survival_upper_null: float | None = None
        survival_lower: float | None = None
        survival_rule: str | None = None
        if headline_allowed:
            if retained_family_order != retained_family_ids:
                raise GatedEvaluationError(
                    "retained family order differs from the symptom-survival support"
                )
            retained_seed_draws, retained_family_weights = _hierarchical_bootstrap_plan(
                len(border_seeds),
                len(retained_family_ids),
                resamples=null_count,
                seed=_support_seed(retained_family_ids, "border-retained-support"),
            )
            survival_intervals = _hierarchical_intervals(
                np.stack(retained_stats_by_seed),
                retained_seed_draws,
                retained_family_weights,
                len(classes),
            )
            retained_family_labels = np.asarray(
                [
                    class_indices[
                        next(
                            entry.label
                            for entry in retained_entries
                            if entry.group == family
                        )
                    ]
                    for family in retained_family_ids
                ],
                dtype=np.int64,
            )
            retained_rng = np.random.default_rng(
                _support_seed(retained_family_ids, "border-retained-label-permutation")
            )
            retained_permuted_labels = np.vstack(
                [
                    retained_rng.permutation(retained_family_labels)
                    for _ in range(null_count)
                ]
            )
            retained_null = _family_permutation_macro_f1(
                np.stack(retained_predicted_counts_by_seed),
                retained_permuted_labels,
                len(classes),
            )
            survival_lower = (
                survival_intervals["macro_f1"]["lower"]
                if survival_intervals["macro_f1"]
                else None
            )
            survival_upper_null = float(np.percentile(retained_null, 95))
            survival_rule = (
                "border information detected among symptom-survival images"
                if survival_lower is not None and survival_lower > survival_upper_null
                else "no detectable border information among symptom-survival images"
            )
            symptom_survival = _mean_seed_metrics(retained_seed_metrics)
        omission_reason = (
            None
            if headline_allowed
            else (
                "mask audit failed"
                if not mask_audit_passed
                else "a class has fewer than 50 retained test images"
            )
        )
        border_results[group_id] = {
            "fit_ids": fit_ids,
            "fit_spec": group_spec,
            "test_support_n": len(group_entries),
            "test_support_sha256": support_digest(fit_ids[0]),
            "unrestricted": unrestricted,
            "symptom_survival": symptom_survival,
            "retained_class_counts": retained_class_counts,
            "retained_scene_counts": retained_scene_counts,
            "frame_leaf_share_mean": float(np.mean(leaf_shares)),
            "frame_leaf_shares": [
                {"image_id": entry.image_id, "leaf_pixel_share": share}
                for entry, share in zip(group_entries, leaf_shares, strict=True)
            ],
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
            "symptom_survival_permutation_macro_f1_95th": survival_upper_null,
            "symptom_survival_macro_f1_lower_95": survival_lower,
            "symptom_survival_rule": survival_rule,
            "macro_f1_lower_95": full_lower,
            "permutation_macro_f1_95th": full_upper_null,
            "permutations": null_count,
            "bootstrap_method": (
                "hierarchical: seeds within primary split, then test families; "
                "family multiplicities paired across systems"
            ),
            "rule": "border information detected"
            if full_lower is not None and full_lower > full_upper_null
            else "no detectable border information at this frame width",
            "headline": symptom_survival,
            "headline_omission_reason": omission_reason,
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
        "border_leaf_share_cutoff": selection["border_leaf_share_cutoff"]["final"],
        "completed_analysis_ids": run_record["completed_analysis_ids"],
        "manifest_digest": selection["manifest_digest"],
        "post_hoc": selection["created_after_test_reads"],
        "test_images": len(entries),
        "test_supports": {
            fit_id: {
                "n": len(entries_by_fit[fit_id]),
                "sha256": support_digest(fit_id),
            }
            for fit_id in expected_ids
        },
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
            "sample_size": max(sample_sizes.values(), default=0),
            "sample_sizes": sample_sizes,
            "sampling_seed": sample_seed,
            "sample_image_ids": sample_image_ids,
            "fit_scores": leaf_scores,
            "full_image_sample_scores": full_sample_scores,
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
            {
                "pool": pool,
                "image_id": entry_by_key[(pool, image_id)].image_id,
                "leaf_pixel_share": prepared[(pool, image_id)][1],
            }
            for pool, image_id in support_keys
        ],
        "class_by_scene": scene_tables,
        "class_by_scene_status": "completed",
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

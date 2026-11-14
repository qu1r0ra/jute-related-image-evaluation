"""Computed, logged inputs to the image diagnostics.

The diagnostics read three things that an operator could otherwise type in: the
Grad-CAM sanity checks, the scene tags with their frame leaf shares, and the
predictions of the three split policies. The first two are produced by logged
commands and written as ledger-bound outputs; the third is read from the run
record. A bound output is a self-hashed file whose SHA-256 is also recorded in
the test-read ledger under the freeze record, so a hand-written file, or one
edited after the command wrote it, has no matching ledger row and is refused.
"""

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from jute_disease.data import (
    POLICIES,
    ColorThresholdBackend,
    ManifestDataset,
    SceneThresholds,
    classify_scene,
    create_pipeline,
    extract_scene_features,
    require_colour_threshold,
)
from jute_disease.data.manifest import load_complete_manifest
from jute_disease.evaluation.breakdowns import stratified_sample
from jute_disease.evaluation.diagnostics import (
    GRADCAM_ANALYSIS_ID,
    SANITY_SAMPLE_SIZE,
    SANITY_SEED,
    letterbox_leaf_mask,
    require_analysis,
    sanity_check_results,
    select_median_validation_run,
    validate_primary_policy_runs,
)
from jute_disease.evaluation.freeze_record import (
    Authorization,
    FreezeRecordError,
    RunEntry,
    bound_path,
    canonical_hash,
    default_reads_log_path,
    log_test_read,
    read_reads_log,
)
from jute_disease.evaluation.probes import frame_leaf_share

OUTPUT_SCHEMA = 1
SANITY_CHECKS_KIND = "sanity_checks"
SCENE_TAGS_KIND = "scene_tags"
OUTPUT_SUFFIX = "_output"


@dataclass(frozen=True)
class BoundOutput:
    """A diagnostics input whose file digest the ledger recorded under the record."""

    kind: str
    path: Path
    sha256: str
    body: dict[str, Any]


def write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a reader sees either nothing or all of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_bound_output(
    path: Path,
    kind: str,
    body: Mapping[str, Any],
    *,
    record_hash: str,
    reads_log: Path,
    additional_logs: Sequence[Path] = (),
) -> str:
    """Write a self-hashed output atomically, then log its digest under the record.

    The ledger row comes last, so a crash between the two leaves a file with no
    row, which :func:`read_bound_output` refuses.
    """
    if path.exists():
        raise FreezeRecordError(f"refusing to replace {path}")
    data = (
        json.dumps(
            {"hash": canonical_hash(body), "body": body},
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_atomically(path, data)
    digest = hashlib.sha256(data).hexdigest()
    log_test_read(
        reads_log,
        record_hash,
        [],
        operation=kind + OUTPUT_SUFFIX,
        additional_logs=additional_logs,
        details={"output_sha256": digest},
    )
    return digest


def read_bound_output(
    path: Path, kind: str, *, record_hash: str, reads_log: Path
) -> BoundOutput:
    """Return an output only if a logged command wrote exactly these bytes.

    Raises:
        FreezeRecordError: the file is unreadable, not self-hashed, bound to another
            freeze record, or has no ledger row recording its digest.
    """
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        body = payload["body"]
        stored = payload["hash"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FreezeRecordError(
            f"{kind.replace('_', ' ')} file is unreadable or malformed: {error}"
        ) from error
    label = kind.replace("_", " ")
    if (
        not isinstance(body, dict)
        or stored != canonical_hash(body)
        or body.get("schema") != OUTPUT_SCHEMA
        or body.get("kind") != kind
    ):
        raise FreezeRecordError(f"{label} file is not a computed {label} output")
    if body.get("freeze_record_hash") != record_hash:
        raise FreezeRecordError(
            f"{label} were computed under a different freeze record"
        )
    digest = hashlib.sha256(raw).hexdigest()
    logged = any(
        row.get("operation") == kind + OUTPUT_SUFFIX
        and row.get("record_hash") == record_hash
        and (row.get("details") or {}).get("output_sha256") == digest
        for row in read_reads_log(reads_log)
    )
    if not logged:
        raise FreezeRecordError(
            f"{label} were not computed by a logged command; "
            f"run evaluate {kind.replace('_', '-')} and use its output"
        )
    return BoundOutput(kind, Path(path), digest, body)


def require_diagnostic_cell(
    authorization: Authorization, assignment: Mapping[str, Any]
) -> None:
    """The runs must come from the pool, view and policy the freeze record fixed."""
    frozen = authorization.body.get("diagnostic")
    cell = {key: assignment[key] for key in ("pool", "view", "policy")}
    if not isinstance(frozen, dict) or {key: frozen.get(key) for key in cell} != cell:
        raise FreezeRecordError(
            f"diagnostic runs {cell} are not the pool, view and policy "
            f"fixed in the freeze record: {frozen}"
        )


def primary_policy_runs(
    authorization: Authorization,
    assignment_table_path: Path | str,
) -> tuple[tuple[RunEntry, ...], dict[str, Any], dict[str, tuple[RunEntry, ...]]]:
    """Split the authorized runs by policy and return the primary cell's runs.

    Every policy needs its five seeds bound to the assignment table, all at one
    pool and view, and the primary policy must be the one the freeze record fixed.
    """
    groups: dict[str, list[RunEntry]] = {}
    for run in authorization.runs:
        groups.setdefault(str(run.policy), []).append(run)
    if set(groups) != set(POLICIES):
        raise FreezeRecordError(
            "diagnostics need the authorized runs of image-random, exact-copy "
            f"and near-duplicate; got {sorted(groups)}"
        )
    assignments = {
        policy: validate_primary_policy_runs(assignment_table_path, runs)
        for policy, runs in groups.items()
    }
    cells = {(item["pool"], item["view"]) for item in assignments.values()}
    if len(cells) != 1:
        raise FreezeRecordError("policy runs must share one pool and view")
    frozen = authorization.body.get("diagnostic")
    policy = frozen.get("policy") if isinstance(frozen, dict) else None
    if policy not in groups:
        raise FreezeRecordError("freeze record fixes no diagnostic policy")
    assignment = assignments[policy]
    require_diagnostic_cell(authorization, assignment)
    return (
        tuple(groups[policy]),
        assignment,
        {name: tuple(runs) for name, runs in groups.items()},
    )


def load_run_predictions(
    entry: RunEntry, base_dir: Path, ids: Sequence[str]
) -> tuple[str, ...]:
    """Read a run's predictions after checking them against the run record."""
    raw = (base_dir / str(entry.predictions)).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.predictions_sha256:
        raise FreezeRecordError(
            f"predictions changed since the run record: {entry.fit_id}"
        )
    payload = json.loads(raw)
    if tuple(payload.get("image_ids", ())) != tuple(ids):
        raise FreezeRecordError(
            f"predictions for {entry.fit_id} do not match test images"
        )
    labels = tuple(payload.get("predictions", ()))
    if len(labels) != len(ids):
        raise FreezeRecordError(f"predictions for {entry.fit_id} are incomplete")
    return labels


def policy_predictions(
    policy_runs: Mapping[str, Sequence[RunEntry]],
    base_dir: Path,
    ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Each policy's predictions, from its validation-median run in the record."""
    result: dict[str, dict[str, str]] = {}
    for policy, runs in policy_runs.items():
        median = select_median_validation_run(runs)
        labels = load_run_predictions(median, base_dir, ids)
        result[policy] = dict(zip(ids, labels, strict=True))
    return result


def verified_checkpoint(run: RunEntry, base_dir: Path) -> bytes:
    """The checkpoint bytes of ``run``, checked against the record and its receipt."""
    if (
        run.validation_macro_f1 is None
        or run.checkpoint is None
        or run.checkpoint_sha256 is None
        or run.training_receipt_sha256 is None
    ):
        raise FreezeRecordError(
            "diagnostics require receipt-bound validation macro-F1 and "
            "checkpoint artifacts"
        )
    base = base_dir.resolve()
    path = (base / run.checkpoint).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise FreezeRecordError(
            "authorized checkpoint path is missing or escapes the run store"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != run.checkpoint_sha256:
        raise FreezeRecordError(
            "authorized checkpoint bytes differ from the run record"
        )
    receipt = base.joinpath(*run.fit_id.split("/")) / "receipt.json"
    if (
        not receipt.is_file()
        or hashlib.sha256(receipt.read_bytes()).hexdigest()
        != run.training_receipt_sha256
    ):
        raise FreezeRecordError("training receipt changed since the run record")
    return raw


def _ids_digest(image_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


def compute_sanity_checks(
    authorization: Authorization,
    *,
    target_dir: Path,
    image_root: Path,
    base_dir: Path,
    assignment_table_path: Path,
    output: Path,
    root: Path,
    additional_logs: Sequence[Path] = (),
    model_name: str = "mobilenetv2_100.ra_in1k",
) -> BoundOutput:
    """Compute both Grad-CAM sanity checks on the selected run and log them.

    The intent row is appended before any test image is opened. The pass rule is
    fixed in :mod:`jute_disease.evaluation.diagnostics`; nothing here takes a
    verdict from the caller.
    """
    fit_ids = [run.fit_id for run in authorization.runs]
    require_analysis(authorization, GRADCAM_ANALYSIS_ID, fit_ids)
    runs, assignment, _ = primary_policy_runs_for_one_cell(
        authorization, assignment_table_path
    )
    selected = select_median_validation_run(runs)
    checkpoint = verified_checkpoint(selected, base_dir)
    ledger = default_reads_log_path(root)
    log_test_read(
        ledger,
        authorization.record_hash,
        fit_ids,
        operation=SANITY_CHECKS_KIND,
        additional_logs=additional_logs,
        details={
            "selected_fit_id": selected.fit_id,
            "checkpoint_sha256": selected.checkpoint_sha256,
            "sample_size": SANITY_SAMPLE_SIZE,
            "sample_seed": SANITY_SEED,
        },
    )
    manifest = load_complete_manifest(target_dir)
    entries = tuple(sorted(manifest.partition("test"), key=lambda e: e.image_id))
    sample_ids = stratified_sample(
        [e.image_id for e in entries],
        [e.label for e in entries],
        SANITY_SAMPLE_SIZE,
        SANITY_SEED,
    )
    by_id = {entry.image_id: entry for entry in entries}
    dataset = ManifestDataset(manifest, "test", image_root)
    from jute_disease.models.dl import Classifier, TimmBackbone, read_checkpoint_state

    model = Classifier(TimmBackbone(model_name, pretrained=False), len(manifest.labels))
    model.load_state_dict(read_checkpoint_state(io.BytesIO(checkpoint)), strict=True)
    model.eval()
    pipeline = create_pipeline(is_dl=True)
    tensors = []
    for image_id in sample_ids:
        with Image.open(dataset.image_root / by_id[image_id].path) as source:
            tensors.append(pipeline(source.convert("RGB")))
    images = torch.stack(tensors)
    with torch.no_grad():
        targets = [int(row.argmax().item()) for row in model(images)]
    class_index = {label: index for index, label in enumerate(manifest.labels)}
    truth = np.asarray([class_index[by_id[i].label] for i in sample_ids])
    permuted = np.random.default_rng(SANITY_SEED).permutation(truth)
    results = sanity_check_results(
        model,
        images,
        targets,
        [int(value) for value in permuted],
        checkpoint_sha256=str(selected.checkpoint_sha256),
        image_ids=sample_ids,
    )
    body = {
        "schema": OUTPUT_SCHEMA,
        "kind": SANITY_CHECKS_KIND,
        "freeze_record_hash": authorization.record_hash,
        "selected_fit_id": selected.fit_id,
        "checkpoint_sha256": selected.checkpoint_sha256,
        "assignment_table_sha256": assignment["sha256"],
        "sample_image_ids_sha256": _ids_digest(sample_ids),
        **results,
    }
    digest = write_bound_output(
        output,
        SANITY_CHECKS_KIND,
        body,
        record_hash=authorization.record_hash,
        reads_log=ledger,
        additional_logs=additional_logs,
    )
    return BoundOutput(SANITY_CHECKS_KIND, output, digest, body)


def primary_policy_runs_for_one_cell(
    authorization: Authorization, assignment_table_path: Path | str
) -> tuple[tuple[RunEntry, ...], dict[str, Any], dict[str, tuple[RunEntry, ...]]]:
    """The primary cell's runs when the authorization lists one policy or all three."""
    policies = {str(run.policy) for run in authorization.runs}
    if len(policies) == 1:
        assignment = validate_primary_policy_runs(
            assignment_table_path, authorization.runs
        )
        require_diagnostic_cell(authorization, assignment)
        return (
            tuple(authorization.runs),
            assignment,
            {assignment["policy"]: tuple(authorization.runs)},
        )
    return primary_policy_runs(authorization, assignment_table_path)


def audited_thresholds(
    freeze_body: Mapping[str, Any], root: Path | str
) -> tuple[SceneThresholds, dict[str, str], str]:
    """The scene thresholds and backend of the mask audit the freeze record binds."""
    audit = freeze_body.get("mask_audit")
    if not isinstance(audit, dict):
        raise FreezeRecordError("freeze record binds no mask audit")
    raw = bound_path(audit.get("audit_manifest"), root).read_bytes()
    if hashlib.sha256(raw).hexdigest() != audit.get("audit_manifest_sha256"):
        raise FreezeRecordError("bound audit manifest changed since the freeze record")
    manifest = json.loads(raw)
    backend = manifest.get("mask_backend")
    if not isinstance(backend, dict):
        raise FreezeRecordError("audit manifest records no mask backend")
    require_colour_threshold(str(backend.get("model")), str(backend.get("version")))
    thresholds = SceneThresholds.from_dict(manifest["scene_thresholds"])
    if (thresholds.mask_model, thresholds.mask_version) != (
        backend["model"],
        backend["version"],
    ):
        raise FreezeRecordError("audited thresholds were fit on another mask backend")
    return (
        thresholds,
        {"model": backend["model"], "version": backend["version"]},
        str(audit["audit_manifest_sha256"]),
    )


def compute_scene_tags(
    record: Mapping[str, Any],
    *,
    root: Path,
    target_dir: Path,
    image_root: Path,
    output: Path,
    additional_logs: Sequence[Path] = (),
) -> BoundOutput:
    """Tag every test image and measure its frame leaf share, bound to the audit.

    The cutoffs and the backend come from the audit manifest the freeze record
    binds by digest, and are recorded in the output, so the tags cannot come from
    other thresholds. The intent row precedes the first test image read.
    """
    freeze_body = record["body"]
    thresholds, backend_record, manifest_sha256 = audited_thresholds(freeze_body, root)
    backend = ColorThresholdBackend()
    if (backend.model, backend.version) != (
        backend_record["model"],
        backend_record["version"],
    ):
        raise FreezeRecordError(
            "installed mask backend is not the audited backend: "
            f"{backend.model} {backend.version}"
        )
    width = int(freeze_body["border_probe"]["final"]["width_px"])
    ledger = default_reads_log_path(root)
    thresholds_sha256 = canonical_hash(thresholds.to_dict())
    log_test_read(
        ledger,
        record["hash"],
        [],
        operation=SCENE_TAGS_KIND,
        additional_logs=additional_logs,
        details={
            "thresholds_sha256": thresholds_sha256,
            "audit_manifest_sha256": manifest_sha256,
            "mask_backend": backend_record,
            "border_width_px": width,
        },
    )
    manifest = load_complete_manifest(target_dir)
    dataset = ManifestDataset(manifest, "test", image_root)
    entries = sorted(dataset.entries, key=lambda e: e.image_id)
    pipeline = create_pipeline(is_dl=False, include_mask=True)
    scenes: dict[str, str] = {}
    shares: dict[str, float] = {}
    for entry in entries:
        with Image.open(dataset.image_root / entry.path) as source:
            original = source.convert("RGB")
        prediction = backend.predict(original)
        scenes[entry.image_id] = classify_scene(
            extract_scene_features(original, prediction.mask), thresholds
        )
        content_mask = pipeline(original).valid_mask
        leaf = letterbox_leaf_mask(prediction.mask, content_mask)
        shares[entry.image_id] = frame_leaf_share(leaf, content_mask, width)
    body = {
        "schema": OUTPUT_SCHEMA,
        "kind": SCENE_TAGS_KIND,
        "freeze_record_hash": record["hash"],
        "thresholds_sha256": thresholds_sha256,
        "audit_manifest_sha256": manifest_sha256,
        "mask_backend": backend_record,
        "border_width_px": width,
        "image_ids_sha256": _ids_digest([e.image_id for e in entries]),
        "scenes": scenes,
        "frame_leaf_share": shares,
    }
    digest = write_bound_output(
        output,
        SCENE_TAGS_KIND,
        body,
        record_hash=record["hash"],
        reads_log=ledger,
        additional_logs=additional_logs,
    )
    return BoundOutput(SCENE_TAGS_KIND, output, digest, body)


def require_scene_tags_bound(
    scene_tags: BoundOutput, freeze_body: Mapping[str, Any]
) -> None:
    """The tags must come from the audit the freeze record binds, at its width."""
    audit = freeze_body.get("mask_audit") or {}
    border = (freeze_body.get("border_probe") or {}).get("final") or {}
    if scene_tags.body.get("audit_manifest_sha256") != audit.get(
        "audit_manifest_sha256"
    ) or scene_tags.body.get("border_width_px") != border.get("width_px"):
        raise FreezeRecordError(
            "scene tags are not bound to the audited thresholds and border width "
            "of the freeze record"
        )


__all__ = [
    "BoundOutput",
    "audited_thresholds",
    "compute_scene_tags",
    "compute_sanity_checks",
    "load_run_predictions",
    "policy_predictions",
    "primary_policy_runs",
    "read_bound_output",
    "require_scene_tags_bound",
    "verified_checkpoint",
    "write_atomically",
    "write_bound_output",
]

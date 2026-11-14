"""Matrix-freeze record, run record and the gate in front of every test read.

The freeze record is one hashed JSON file listing every planned fit and analysis,
the validation-selected choices, the letterbox decision, the digest of the mask
audit report, the digests of every policy-matrix file (so every pool, view, policy
and draw's protected-test manifest), the diagnostic pool, view and policy, the
border-probe values, the post-pilot budget and the hashes of the pipeline code and
environment lock. The mask-audit outcome is derived from the report, never typed
in. The evaluator refuses a test read for any fit that is not listed in a record
whose hash matches, or that has no run record bound to that record. A new record
invalidates earlier reads; a record created after any test read is flagged
post-hoc. Reads are appended under a lock and flushed to disk.
"""

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jute_disease.data import (
    POLICIES,
    audit_report_outcome,
    policy_matrix_binding,
    require_colour_threshold,
)
from jute_disease.data.manifest import (
    load_complete_manifest,
    load_manifest,
    load_protected_test_manifest,
)
from jute_disease.training import FileLock, FitLockedError, RunStore

RECORD_SCHEMA = 4
RUN_RECORD_SCHEMA = 2
LOG_LOCK_TIMEOUT_SECONDS = 30.0
RESIDUAL_AUDIT_OPERATION = "residual_audit"
READS_LOG_RELATIVE_PATH = Path("artifacts/evaluation/test-reads.jsonl")
PIPELINE_DIRS = ("src", "configs")
LOCKFILE = "uv.lock"
REQUIRED_ANALYSES = (
    "background-probe",
    "border-probe",
    "embeddings-test-pass",
    "gradcam-test-pass",
    "leaf-only",
    "scene-cells",
)
REQUIRED_KEYS = (
    "fit_ids",
    "analysis_ids",
    "selections",
    "letterbox_decision",
    "diagnostic",
    "border_probe",
    "budget",
)
MATRIX_KEYS = ("pools", "views", "policies", "draws")


class FreezeRecordError(ValueError):
    """A freeze or run record cannot be written or does not verify."""


class TestReadRefusedError(PermissionError):
    """The evaluator refuses a test read."""

    __test__ = False


def default_reads_log_path(root: Path | str) -> Path:
    """Return the project-wide log path used to detect prior test reads."""
    return Path(root) / READS_LOG_RELATIVE_PATH


def canonical_hash(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_tree_hash(root: Path | str) -> str:
    """Hash of src, configs and the lockfile, so an unrelated commit leaves it alone."""
    base = Path(root)
    entries: list[tuple[str, str]] = []
    for name in PIPELINE_DIRS:
        for path in sorted((base / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append((path.relative_to(base).as_posix(), _file_hash(path)))
    lock = base / LOCKFILE
    if lock.is_file():
        entries.append((LOCKFILE, _file_hash(lock)))
    if not entries:
        raise FreezeRecordError(f"no pipeline files under {base}")
    return canonical_hash(entries)


def environment_lock_hash(root: Path | str) -> str:
    lock = Path(root) / LOCKFILE
    if not lock.is_file():
        raise FreezeRecordError(f"environment lock not found: {lock}")
    return _file_hash(lock)


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise FreezeRecordError(f"refusing to replace existing {path}") from error


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TestReadRefusedError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise TestReadRefusedError(f"{description} must be a JSON object")
    return payload


def read_reads_log(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_audit_read(entry: Mapping[str, Any]) -> bool:
    """A residual-audit row that names its pool, sample IDs and pinned weights.

    The operation string alone never exempts a read: the row must read no fit and
    carry the digests that let a reviewer re-derive what was opened.
    """
    details = entry.get("details")
    return (
        entry.get("operation") == RESIDUAL_AUDIT_OPERATION
        and entry.get("fit_ids") == []
        and isinstance(details, dict)
        and isinstance(details.get("pool"), str)
        and bool(details["pool"])
        and _is_sha256(details.get("sample_ids_sha256"))
        and _is_sha256(details.get("weights_sha256"))
    )


def _split_reads(
    entries: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate reads that can steer a fit from the listed residual-audit reads.

    The residual audit opens pixels only and reads no label, score or prediction,
    so a well-formed audit row is listed in the record but leaves it ground-truth
    clean. Anything else, including a row that only claims the audit operation, is
    a label read.
    """
    labels: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for entry in entries:
        (audits if _is_audit_read(entry) else labels).append(entry)
    return labels, audits


def _merged_reads(
    root: Path | str, reads_log: Path | str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every distinct logged read from the project ledger and any custom log."""
    default_log = default_reads_log_path(root)
    logs = [default_log]
    if reads_log is not None and Path(reads_log).resolve() != default_log.resolve():
        logs.append(Path(reads_log))
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for log in logs:
        for entry in read_reads_log(log):
            key = canonical_hash(entry)
            if key not in seen:
                seen.add(key)
                merged.append(entry)
    return _split_reads(merged)


def _portable(path: Path | str, root: Path | str) -> str:
    """Store ``path`` relative to the project root when it lies inside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def bound_path(stored: object, root: Path | str) -> Path:
    """Resolve a path a record stored, relative to ``root`` unless absolute."""
    if not isinstance(stored, str) or not stored:
        raise TestReadRefusedError("freeze record stores an invalid path")
    path = Path(stored)
    return path if path.is_absolute() else (Path(root) / path).resolve()


def _unique_strings(value: object, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise FreezeRecordError(f"{name} must be a non-empty list of unique strings")
    return list(value)


def _check_outcomes(
    spec: Mapping[str, Any],
    outcomes: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
) -> None:
    """Require recorded #114 validation outcomes behind every decision."""
    if not outcomes or outcomes.get("partition") != "val":
        raise FreezeRecordError(
            "recorded #114 validation outcomes (partition 'val') are required"
        )
    comparison = outcomes.get("letterbox_comparison")
    if not isinstance(comparison, dict) or len(comparison) < 2:
        raise FreezeRecordError(
            "validation outcomes need the two-fit letterbox comparison"
        )
    if not all(isinstance(v, int | float) for v in comparison.values()):
        raise FreezeRecordError("letterbox comparison values must be numbers")
    decision = spec.get("letterbox_decision")
    choice = decision.get("choice") if isinstance(decision, dict) else None
    if choice not in comparison:
        raise FreezeRecordError("letterbox decision is not one of the compared fits")
    recorded_audit = outcomes.get("mask_audit")
    if recorded_audit is not None and (
        not isinstance(recorded_audit, dict)
        or recorded_audit.get("digest") != audit["digest"]
    ):
        raise FreezeRecordError(
            "recorded validation outcome names a different mask audit report"
        )
    recorded = outcomes.get("selections")
    selections = spec.get("selections")
    if not isinstance(recorded, dict) or not isinstance(selections, dict):
        raise FreezeRecordError("validation outcomes need the selection outcomes")
    missing = sorted(set(selections) - set(recorded))
    if missing:
        raise FreezeRecordError(
            f"selections without a recorded validation outcome: {missing}"
        )
    mismatched = sorted(
        key
        for key, selected in selections.items()
        if not isinstance(recorded[key], dict)
        or recorded[key].get("selected") != selected
    )
    if mismatched:
        raise FreezeRecordError(
            f"freeze selections must match recorded validation choices: {mismatched}"
        )


def _check_border(border: object) -> None:
    ok = isinstance(border, dict) and all(
        isinstance(border.get(stage), dict)
        and isinstance(border[stage].get("width_px"), int)
        and isinstance(border[stage].get("leaf_cutoff"), int | float)
        for stage in ("start", "final")
    )
    if not ok:
        raise FreezeRecordError(
            "border_probe needs start and final width_px and leaf_cutoff"
        )


def _read_audit_files(report: Path, manifest: Path) -> tuple[bytes, bytes, dict]:
    """Read the report and its audit manifest and derive the outcome."""
    raw = report.read_bytes()
    raw_manifest = manifest.read_bytes()
    outcome = audit_report_outcome(json.loads(raw), json.loads(raw_manifest))
    return raw, raw_manifest, outcome


def _bind_mask_audit(
    report_path: Path | str, manifest_path: Path | str, root: Path | str
) -> dict[str, Any]:
    """Bind a genuine evaluation report, its audit manifest and the derived outcome."""
    report = Path(report_path).resolve()
    manifest = Path(manifest_path).resolve()
    try:
        raw, raw_manifest, outcome = _read_audit_files(report, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FreezeRecordError(
            f"cannot bind mask audit report {report}: {error}"
        ) from error
    return {
        **outcome,
        "report": _portable(report, root),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_manifest": _portable(manifest, root),
        "audit_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
    }


def _bind_policy_matrix(
    policy_dir: Path | str,
    matrix: Mapping[str, Sequence[Any]] | None,
    root: Path | str,
) -> dict[str, Any]:
    """Verify every policy file against its receipt and return the binding."""
    requested = dict(matrix or {})
    unknown = sorted(set(requested) - set(MATRIX_KEYS))
    if unknown:
        raise FreezeRecordError(f"unknown matrix dimensions {unknown}")
    pools = requested.get("pools")
    views = tuple(requested.get("views") or (1, 2))
    policies = tuple(requested.get("policies") or POLICIES)
    draws = tuple(requested.get("draws") or range(10))
    try:
        binding = policy_matrix_binding(
            policy_dir,
            pools=tuple(pools) if pools else None,
            views=views,
            policies=policies,
            draws=draws,
        )
    except (OSError, ValueError) as error:
        raise FreezeRecordError(f"cannot bind the policy matrix: {error}") from error
    found = sorted({name.split("/", 1)[0] for name in binding["files"]})
    return {
        "schema": 1,
        "path": _portable(policy_dir, root),
        **binding,
        "matrix": {
            "pools": sorted(pools) if pools else found,
            "views": sorted(views),
            "policies": sorted(policies),
            "draws": sorted(draws),
        },
    }


def _check_diagnostic(diagnostic: object, matrix: Mapping[str, Any]) -> None:
    ok = (
        isinstance(diagnostic, dict)
        and set(diagnostic) == {"pool", "view", "policy"}
        and diagnostic["pool"] in matrix["pools"]
        and diagnostic["view"] in matrix["views"]
        and diagnostic["policy"] in matrix["policies"]
    )
    if not ok:
        raise FreezeRecordError(
            "diagnostic must name one pool, view and policy of the frozen matrix"
        )


def write_freeze_record(
    spec: Mapping[str, Any],
    validation_outcomes: Mapping[str, Any] | None,
    output: Path | str,
    root: Path | str,
    *,
    target_dir: Path | str,
    policy_dir: Path | str,
    mask_audit_report: Path | str,
    mask_audit_manifest: Path | str,
    matrix: Mapping[str, Sequence[Any]] | None = None,
    required_fit_ids: Iterable[str] = (),
    reads_log: Path | str | None = None,
) -> dict[str, Any]:
    """Write the hashed matrix-freeze record; never replaces an existing file.

    The one record binds the target manifests, every policy-matrix file (and so
    every draw's protected-test manifest), the mask audit report and its audit
    manifest by digest, and the diagnostic pool, view and policy. The mask-audit
    outcome is derived from the report, so a spec that carries its own is refused,
    and a fixture report is never accepted. Earlier label reads make the record
    post hoc; well-formed residual-audit reads are listed as ``audit_reads``.
    """
    if "mask_audit" in spec:
        raise FreezeRecordError(
            "the mask audit outcome is derived from the audit report; "
            "remove mask_audit from the spec and pass the report"
        )
    missing_keys = [key for key in REQUIRED_KEYS if key not in spec]
    if missing_keys:
        raise FreezeRecordError(f"freeze spec is missing {missing_keys}")
    fit_ids = _unique_strings(spec["fit_ids"], "fit_ids")
    analysis_ids = _unique_strings(spec["analysis_ids"], "analysis_ids")
    absent = sorted(set(REQUIRED_ANALYSES) - set(analysis_ids))
    if absent:
        raise FreezeRecordError(f"analysis_ids must list {absent}")
    unlisted = sorted(set(required_fit_ids) - set(fit_ids))
    if unlisted:
        raise FreezeRecordError(
            f"planned fits missing from the record ({len(unlisted)}): {unlisted[:5]}"
        )
    if not isinstance(spec["selections"], dict) or not spec["selections"]:
        raise FreezeRecordError("selections must list the validation-selected choices")
    mask_audit = _bind_mask_audit(mask_audit_report, mask_audit_manifest, root)
    policy_binding = _bind_policy_matrix(policy_dir, matrix, root)
    _check_diagnostic(spec["diagnostic"], policy_binding["matrix"])
    _check_border(spec["border_probe"])
    _check_outcomes(spec, validation_outcomes, mask_audit)
    manifests = _manifest_binding(target_dir)
    prior_reads, audit_reads = _merged_reads(root, reads_log)
    body = {
        **{key: spec[key] for key in REQUIRED_KEYS},
        "fit_ids": sorted(fit_ids),
        "analysis_ids": sorted(analysis_ids),
        "invalidated_reads": prior_reads,
        "audit_reads": audit_reads,
        "mask_audit": mask_audit,
        "policy_binding": policy_binding,
        "manifest_binding": manifests,
        "validation_outcomes_hash": canonical_hash(validation_outcomes),
        "pipeline_tree_hash": pipeline_tree_hash(root),
        "environment_lock_hash": environment_lock_hash(root),
        "post_hoc": bool(prior_reads),
    }
    record = {
        "schema": RECORD_SCHEMA,
        "hash": canonical_hash(body),
        "created": datetime.now(UTC).isoformat(),
        "body": body,
    }
    _write_new(Path(output), record)
    return record


def load_freeze_record(path: Path | str) -> dict[str, Any]:
    """Read a freeze record and verify that its hash matches its body."""
    record = _read_json(Path(path), "freeze record")
    body = record.get("body")
    if (
        record.get("schema") != RECORD_SCHEMA
        or not isinstance(body, dict)
        or record.get("hash") != canonical_hash(body)
    ):
        raise TestReadRefusedError("freeze record hash does not match its content")
    return record


def _manifest_binding(target_dir: Path | str) -> dict[str, Any]:
    """Validate and bind fit and protected-test manifest files to a record."""
    directory = Path(target_dir).resolve()
    fit_path = directory / "manifest.json"
    test_path = directory / "protected-test" / "manifest.json"
    try:
        fit = load_manifest(directory)
        test = load_protected_test_manifest(directory)
        load_complete_manifest(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, FreezeRecordError):
            raise
        raise FreezeRecordError(
            f"cannot bind invalid target manifests: {error}"
        ) from error
    if any(entry.partition == "test" for entry in fit.entries):
        raise FreezeRecordError("fit manifest contains protected test rows")
    fit_dependencies = dict(fit.dependencies)
    test_dependencies = dict(test.dependencies)
    if fit_dependencies.pop("manifest_scope", None) != "fit":
        raise FreezeRecordError("target manifest does not have fit scope")
    if test_dependencies.pop("manifest_scope", None) != "protected_test":
        raise FreezeRecordError("protected-test manifest does not have test scope")
    if fit_dependencies != test_dependencies:
        raise FreezeRecordError(
            "fit and protected-test manifests have mismatched lineage"
        )
    return {
        "schema": 1,
        "fit": {
            "schema": fit.version,
            "path": str(fit_path),
            "sha256": _file_hash(fit_path),
        },
        "protected_test": {
            "schema": test.version,
            "path": str(test_path),
            "sha256": _file_hash(test_path),
        },
    }


@dataclass(frozen=True)
class RunEntry:
    """One completed fit whose test predictions the evaluator may read."""

    fit_id: str
    system: str
    seed: int
    variant: str
    predictions: str
    predictions_sha256: str
    of: str | None = None
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    model: str | None = None
    model_sha256: str | None = None
    training_receipt_sha256: str | None = None
    lineage_fit_id: str | None = None
    validation_macro_f1: float | None = None
    pool: str | None = None
    view: int | None = None
    policy: str | None = None
    split: int | None = None
    seed_index: int | None = None
    assignment_table_sha256: str | None = None
    prediction_receipt_sha256: str | None = None
    leaf_receipt_sha256: str | None = None


VARIANTS = ("full", "leaf_only", "background_only", "border", "pad_cue")
PREDICTION_RECEIPT_SCHEMA = 1
LEAF_RECEIPT_SCHEMA = 1


def training_description(
    receipt: Mapping[str, Any], fit_id: str
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return a fit's description, artifact kind and artifact entry from its receipt.

    Deep fits carry a ``checkpoint`` artifact and a fit-description mapping;
    classical fits carry a ``model`` artifact and a fit ID string.

    Raises:
        FreezeRecordError: the receipt is malformed or binds no checkpoint or model.
    """
    results = receipt.get("results")
    artifacts = receipt.get("artifacts")
    if not isinstance(results, dict) or not isinstance(artifacts, dict):
        raise FreezeRecordError(f"training receipt of {fit_id} is malformed")
    fit = results.get("fit")
    if isinstance(fit, dict):
        description: dict[str, Any] = {**results, **fit}
    elif isinstance(fit, str):
        description = {"fit_id": fit, **results}
    else:
        raise FreezeRecordError(f"training receipt has no fit metadata for {fit_id}")
    kind = "checkpoint" if "checkpoint" in artifacts else "model"
    artifact = artifacts.get(kind)
    if (
        not isinstance(artifact, dict)
        or "path" not in artifact
        or not _is_sha256(artifact.get("sha256"))
    ):
        raise FreezeRecordError(
            f"training receipt of {fit_id} binds no checkpoint or model"
        )
    return description, kind, artifact


def training_receipt_digest(run_dir: Path) -> str:
    """SHA-256 of the receipt file the run store wrote for one fit."""
    try:
        return _file_hash(run_dir / "receipt.json")
    except OSError as error:
        raise FreezeRecordError(
            f"cannot read the training receipt in {run_dir}"
        ) from error


def receipt_manifest_digest(description: Mapping[str, Any]) -> str:
    """The digest of the manifest a fit trained on, as its receipt recorded it.

    A classical receipt records ``manifest``; a deep receipt records the train
    manifest under ``lineage.manifests``.
    """
    direct = description.get("manifest")
    lineage = description.get("lineage")
    manifests = lineage.get("manifests") if isinstance(lineage, dict) else None
    train = manifests.get("train") if isinstance(manifests, dict) else None
    digest = direct if isinstance(direct, str) else train
    if not _is_sha256(digest):
        raise FreezeRecordError("training receipt records no train manifest digest")
    return str(digest)


def prediction_receipt_path(run_dir: Path, record_hash: str) -> Path:
    """Where the gate writes the receipt of predictions made under one record."""
    return run_dir / f"predictions.{record_hash}.receipt.json"


def write_prediction_receipt(
    run_dir: Path,
    *,
    record_hash: str,
    fit_id: str,
    training_receipt_sha256: str,
    artifact_kind: str,
    artifact_sha256: str,
    fit_manifest_sha256: str,
    predictions_sha256: str,
) -> str:
    """Write the digest-bound receipt of one gated prediction and return its SHA-256.

    It binds the freeze record, the training receipt, the checkpoint or model, the
    manifest the fit trained on and the predictions, so a run record can tell the
    predictions of this fit from another fit's or a hand-written file.
    """
    body = {
        "schema": PREDICTION_RECEIPT_SCHEMA,
        "freeze_record_hash": record_hash,
        "fit_id": fit_id,
        "training_receipt_sha256": training_receipt_sha256,
        "artifact": {"kind": artifact_kind, "sha256": artifact_sha256},
        "fit_manifest_sha256": fit_manifest_sha256,
        "predictions_sha256": predictions_sha256,
    }
    return _write_receipt(prediction_receipt_path(run_dir, record_hash), body)


def _write_receipt(path: Path, body: Mapping[str, Any]) -> str:
    """Write a self-hashed receipt once and return its file SHA-256."""
    data = (
        json.dumps({"hash": canonical_hash(body), "body": body}, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != data:
        raise FreezeRecordError(f"receipt already exists and differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _read_receipt(
    path: Path,
    *,
    schema: int,
    record_hash: str,
    fit_id: str,
    absent: str,
    description: str,
) -> tuple[dict[str, Any], str]:
    """Return a verified receipt body and its file SHA-256.

    Raises:
        FreezeRecordError: ``absent`` when the file cannot be read, or when it is
            not a self-consistent receipt of ``fit_id`` under ``record_hash``.
    """
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as error:
        raise FreezeRecordError(absent) from error
    body = payload.get("body") if isinstance(payload, dict) else None
    if (
        not isinstance(body, dict)
        or payload.get("hash") != canonical_hash(body)
        or body.get("schema") != schema
        or body.get("freeze_record_hash") != record_hash
        or body.get("fit_id") != fit_id
    ):
        raise FreezeRecordError(f"{description} of {fit_id} does not verify")
    return body, hashlib.sha256(raw).hexdigest()


def read_prediction_receipt(
    run_dir: Path, record_hash: str, fit_id: str
) -> tuple[dict[str, Any], str]:
    """Return a verified prediction receipt body and its file SHA-256."""
    return _read_receipt(
        prediction_receipt_path(run_dir, record_hash),
        schema=PREDICTION_RECEIPT_SCHEMA,
        record_hash=record_hash,
        fit_id=fit_id,
        absent=(
            f"no prediction receipt for {fit_id} under this freeze record; "
            "run evaluate predict-test"
        ),
        description="prediction receipt",
    )


def leaf_receipt_path(run_dir: Path, record_hash: str) -> Path:
    """Where the gate writes the receipt of leaf-only predictions under one record."""
    return run_dir / f"predictions.{record_hash}.leaf.receipt.json"


def sample_ids_digest(image_ids: Sequence[str]) -> str:
    """Digest of the image IDs a leaf-only run predicted, in order."""
    return canonical_hash(list(image_ids))


def write_leaf_prediction_receipt(
    run_dir: Path,
    *,
    record_hash: str,
    fit_id: str,
    of: str,
    source_training_receipt_sha256: str,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    preprocessing: Mapping[str, Any],
    sample_ids: Sequence[str],
    predictions_sha256: str,
) -> str:
    """Write the digest-bound receipt of one leaf-only prediction; return its SHA-256.

    It binds the freeze record, the full fit whose weights predicted (its training
    receipt and checkpoint or model), the preprocessing the leaf-only transform
    applied, the sample of test images and the predictions, so a run record can
    tell this producer's predictions from a hand-written file.
    """
    body = {
        "schema": LEAF_RECEIPT_SCHEMA,
        "freeze_record_hash": record_hash,
        "fit_id": fit_id,
        "of": of,
        "source_training_receipt_sha256": source_training_receipt_sha256,
        "source_artifact": {
            "kind": source_artifact_kind,
            "sha256": source_artifact_sha256,
        },
        "preprocessing": dict(preprocessing),
        "sample_ids_sha256": sample_ids_digest(sample_ids),
        "sample_size": len(sample_ids),
        "predictions_sha256": predictions_sha256,
    }
    return _write_receipt(leaf_receipt_path(run_dir, record_hash), body)


def read_leaf_prediction_receipt(
    run_dir: Path, record_hash: str, fit_id: str
) -> tuple[dict[str, Any], str]:
    """Return a verified leaf-only receipt body and its file SHA-256."""
    return _read_receipt(
        leaf_receipt_path(run_dir, record_hash),
        schema=LEAF_RECEIPT_SCHEMA,
        record_hash=record_hash,
        fit_id=fit_id,
        absent=(
            f"no leaf-only receipt for {fit_id} under this freeze record; "
            "run evaluate predict-leaf-only"
        ),
        description="leaf-only receipt",
    )


def _inside(base: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` under ``base``, refusing anything that escapes it."""
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise FreezeRecordError(f"{relative} lies outside the run store")
    return path


def _fit_lineage(
    store: RunStore,
    base: Path,
    fit_id: str,
    run: Mapping[str, Any],
    record_hash: str,
) -> tuple[dict[str, Any], str]:
    """Bind one fit's receipt, artifact and gated prediction receipt by digest.

    Returns the lineage entry and the predictions digest the prediction receipt
    pins. A fit without a verified training receipt, or whose predictions were not
    produced by ``evaluate predict-test`` under this record from this very receipt
    and checkpoint or model, is refused.
    """
    receipt = store.receipt(fit_id, missing_ok=True)
    if receipt is None:
        raise FreezeRecordError(f"no verified training receipt for {fit_id}")
    run_dir = store.run_dir(fit_id)
    receipt_sha256 = training_receipt_digest(run_dir)
    results = receipt["results"] if isinstance(receipt.get("results"), dict) else {}
    description, kind, artifact = training_description(receipt, fit_id)
    if description.get("fit_id", fit_id) != fit_id:
        raise FreezeRecordError(f"training receipt fit ID does not match {fit_id}")
    if description.get("seed") != int(run["seed"]):
        raise FreezeRecordError(
            f"training receipt seed does not match run record for {fit_id}"
        )
    system = description.get("system")
    if system is not None and system != run["system"]:
        raise FreezeRecordError(
            f"training receipt system does not match run record for {fit_id}"
        )
    try:
        path = _inside(base, run_dir.relative_to(base) / str(artifact["path"]))
        observed = _file_hash(path)
    except (OSError, ValueError) as error:
        raise FreezeRecordError(
            f"cannot read the {kind} of {fit_id} under the run store"
        ) from error
    if observed != artifact["sha256"]:
        raise FreezeRecordError(
            f"{kind} artifact does not match training receipt for {fit_id}"
        )
    prediction, prediction_receipt_sha256 = read_prediction_receipt(
        run_dir, record_hash, fit_id
    )
    if (
        prediction.get("training_receipt_sha256") != receipt_sha256
        or prediction.get("artifact") != {"kind": kind, "sha256": observed}
        or prediction.get("fit_manifest_sha256") != receipt_manifest_digest(description)
    ):
        raise FreezeRecordError(
            f"prediction receipt of {fit_id} was made from a different fit; "
            "run evaluate predict-test again"
        )
    metrics = results.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    score = metrics.get("val_f1", metrics.get("val_macro_f1"))
    entry: dict[str, Any] = {
        kind: path.relative_to(base).as_posix(),
        f"{kind}_sha256": observed,
        "training_receipt_sha256": receipt_sha256,
        "prediction_receipt_sha256": prediction_receipt_sha256,
        "lineage_fit_id": fit_id,
    }
    if isinstance(score, int | float) and not isinstance(score, bool):
        entry["validation_macro_f1"] = float(score)
    for field in ("pool", "view", "policy", "split", "seed_index"):
        if field in description:
            entry[field] = description[field]
    metadata = receipt.get("metadata")
    assignment = (
        metadata.get("assignment_table") if isinstance(metadata, dict) else None
    )
    digest = assignment.get("sha256") if isinstance(assignment, dict) else None
    if isinstance(digest, str):
        entry["assignment_table_sha256"] = digest
    return entry, str(prediction["predictions_sha256"])


def _predicted_ids(raw: bytes, fit_id: str) -> list[str]:
    try:
        payload = json.loads(raw)
        ids = payload["image_ids"]
    except (ValueError, KeyError, TypeError) as error:
        raise FreezeRecordError(f"predictions of {fit_id} are unreadable") from error
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise FreezeRecordError(f"predictions of {fit_id} are unreadable")
    return ids


def _leaf_lineage(
    store: RunStore,
    fit_id: str,
    run: Mapping[str, Any],
    record_hash: str,
    source: Mapping[str, Any],
    raw: bytes,
) -> str:
    """Bind a leaf-only run to the receipt of ``evaluate predict-leaf-only``.

    ``source`` is the verified lineage of the full fit the run names in ``of`` and
    ``raw`` the predictions file as read. The receipt must name that fit's training
    receipt and weights, the exact predictions bytes and the exact image IDs, and
    the preprocessing it records must be the full fit's own with the colour-threshold
    mask applied at the fit's letterbox size. Returns the receipt's SHA-256.
    """
    of = str(run["of"])
    receipt, receipt_sha256 = read_leaf_prediction_receipt(
        store.run_dir(fit_id), record_hash, fit_id
    )
    kind = "checkpoint" if "checkpoint_sha256" in source else "model"
    if (
        receipt.get("of") != of
        or receipt.get("source_training_receipt_sha256")
        != source["training_receipt_sha256"]
        or receipt.get("source_artifact")
        != {"kind": kind, "sha256": source[f"{kind}_sha256"]}
    ):
        raise FreezeRecordError(
            f"leaf-only receipt of {fit_id} was made from a different fit than {of}; "
            "run evaluate predict-leaf-only again"
        )
    if receipt.get("predictions_sha256") != hashlib.sha256(raw).hexdigest():
        raise FreezeRecordError(
            f"predictions of {fit_id} are not the ones predict-leaf-only wrote "
            "for that fit under this freeze record"
        )
    ids = _predicted_ids(raw, fit_id)
    if receipt.get("sample_ids_sha256") != sample_ids_digest(ids) or receipt.get(
        "sample_size"
    ) != len(ids):
        raise FreezeRecordError(
            f"the sample of {fit_id} is not the one its leaf-only receipt records"
        )
    source_receipt = store.receipt(of, missing_ok=True)
    if source_receipt is None:
        raise FreezeRecordError(f"no verified training receipt for {of}")
    description, _, _ = training_description(source_receipt, of)
    preprocessing = receipt.get("preprocessing")
    leaf = preprocessing.get("leaf_only") if isinstance(preprocessing, dict) else None
    if (
        not isinstance(preprocessing, dict)
        or not isinstance(leaf, dict)
        or preprocessing.get("fit") != description.get("preprocessing")
    ):
        raise FreezeRecordError(
            f"leaf-only receipt of {fit_id} records preprocessing that is not {of}'s"
        )
    try:
        require_colour_threshold(
            str(leaf.get("mask_model")), str(leaf.get("mask_version"))
        )
    except ValueError as error:
        raise FreezeRecordError(f"leaf-only receipt of {fit_id}: {error}") from error
    size = (description.get("recipe") or {}).get("image_size")
    if size is not None and leaf.get("letterbox") != size:
        raise FreezeRecordError(
            f"leaf-only receipt of {fit_id} masked at another letterbox size than {of}"
        )
    return receipt_sha256


def write_run_record(
    freeze_record: Path | str,
    runs: Sequence[Mapping[str, Any]],
    run_store: Path | str,
    output: Path | str,
) -> dict[str, Any]:
    """Bind completed fits, their receipts and their predictions to one freeze record.

    ``run_store`` is the run-store root, wherever ``output`` is written. Every
    run, including a probe variant, binds the receipt of the fit whose weights
    produced its predictions (the ``of`` fit for a leaf-only run) together with
    that fit's checkpoint or model digest, the gated prediction receipt and the
    predictions digest. A fit's predictions must be the ones ``predict-test`` wrote
    for that fit under this record; another fit's file or a hand-written one is
    refused. A leaf-only run must in addition carry the receipt of
    ``predict-leaf-only``, which binds the full fit's weights, the preprocessing,
    the sampled image IDs and the predictions digest.
    """
    record = load_freeze_record(freeze_record)
    listed = set(record["body"]["fit_ids"])
    base = Path(run_store).resolve()
    store = RunStore(base)
    entries: dict[str, dict[str, Any]] = {}
    for run in runs:
        fit_id = run.get("fit_id")
        if fit_id not in listed:
            raise FreezeRecordError(
                f"fit {fit_id!r} is not listed in the freeze record"
            )
        if fit_id in entries:
            raise FreezeRecordError(f"duplicate run for {fit_id}")
        variant = run.get("variant", "full")
        if variant not in VARIANTS:
            raise FreezeRecordError(f"unknown variant {variant!r} for {fit_id}")
        if variant == "leaf_only" and not run.get("of"):
            raise FreezeRecordError(f"leaf-only run {fit_id} must name its checkpoint")
        source = str(run["of"]) if variant == "leaf_only" else str(fit_id)
        if source not in listed:
            raise FreezeRecordError(
                f"fit {source!r} is not listed in the freeze record"
            )
        path = _inside(base, str(run["predictions"]))
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise FreezeRecordError(f"cannot read predictions: {path}") from error
        digest = hashlib.sha256(raw).hexdigest()
        lineage, gated_digest = _fit_lineage(store, base, source, run, record["hash"])
        if variant == "leaf_only":
            lineage["leaf_receipt_sha256"] = _leaf_lineage(
                store, str(fit_id), run, record["hash"], lineage, raw
            )
        elif digest != gated_digest:
            raise FreezeRecordError(
                f"predictions of {fit_id} are not the ones predict-test wrote "
                "for that fit under this freeze record"
            )
        entries[fit_id] = {
            "system": run["system"],
            "seed": int(run["seed"]),
            "variant": variant,
            "predictions": path.relative_to(base).as_posix(),
            "predictions_sha256": digest,
            "of": run.get("of"),
            **lineage,
        }
    if not entries:
        raise FreezeRecordError("a run record needs at least one run")
    body = {"freeze_record_hash": record["hash"], "fits": entries}
    payload = {"schema": RUN_RECORD_SCHEMA, "hash": canonical_hash(body), "body": body}
    _write_new(Path(output), payload)
    return payload


@dataclass(frozen=True)
class Authorization:
    """What the gate verified before a test read."""

    record_hash: str
    post_hoc: bool
    runs: tuple[RunEntry, ...]
    body: Mapping[str, Any]

    @property
    def missing_fit_ids(self) -> tuple[str, ...]:
        """Frozen fits with no authorized run; a read without them is not the matrix."""
        covered = {run.fit_id for run in self.runs}
        return tuple(sorted(set(self.body.get("fit_ids", ())) - covered))


def _verify_mask_audit(bound: object, root: Path | str) -> None:
    """The bound report and audit manifest must be unchanged and still genuine."""
    if (
        not isinstance(bound, dict)
        or not isinstance(bound.get("report"), str)
        or not isinstance(bound.get("audit_manifest"), str)
    ):
        raise TestReadRefusedError("freeze record binds no mask audit report")
    try:
        raw, raw_manifest, outcome = _read_audit_files(
            bound_path(bound["report"], root),
            bound_path(bound["audit_manifest"], root),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TestReadRefusedError(
            f"bound mask audit report is missing or invalid: {error}"
        ) from error
    if (
        hashlib.sha256(raw).hexdigest() != bound.get("report_sha256")
        or hashlib.sha256(raw_manifest).hexdigest()
        != bound.get("audit_manifest_sha256")
        or any(bound.get(key) != value for key, value in outcome.items())
    ):
        raise TestReadRefusedError(
            "bound mask audit report changed since the freeze record"
        )


def _verify_policy_binding(bound: object, root: Path | str) -> None:
    """Every bound policy file must still match the record, byte for byte."""
    matrix = bound.get("matrix") if isinstance(bound, dict) else None
    if (
        not isinstance(bound, dict)
        or bound.get("schema") != 1
        or not isinstance(bound.get("path"), str)
        or not isinstance(matrix, dict)
    ):
        raise TestReadRefusedError("freeze record has no supported policy binding")
    try:
        current = policy_matrix_binding(bound_path(bound["path"], root), **matrix)
    except (OSError, ValueError, TypeError) as error:
        raise TestReadRefusedError(
            f"bound policy matrix no longer verifies: {error}"
        ) from error
    if any(bound.get(key) != value for key, value in current.items()):
        raise TestReadRefusedError(
            "bound policy matrix changed since the freeze record"
        )


def verify_freeze_record(
    freeze_record: Path | str | None,
    root: Path | str,
    target_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load the record and check that the code and environment are unchanged."""
    if freeze_record is None or not Path(freeze_record).is_file():
        raise TestReadRefusedError("test read refused: no frozen selection record")
    record = load_freeze_record(freeze_record)
    body = record["body"]
    if body["pipeline_tree_hash"] != pipeline_tree_hash(root):
        raise TestReadRefusedError(
            "pipeline code changed since the freeze record; a new record is required"
        )
    if body["environment_lock_hash"] != environment_lock_hash(root):
        raise TestReadRefusedError("environment lock changed since the freeze record")
    _verify_mask_audit(body.get("mask_audit"), root)
    _verify_policy_binding(body.get("policy_binding"), root)
    binding = body.get("manifest_binding")
    if not isinstance(binding, dict) or binding.get("schema") != 1:
        raise TestReadRefusedError("freeze record has no supported manifest binding")
    fit = binding.get("fit")
    protected = binding.get("protected_test")
    if not isinstance(fit, dict) or not isinstance(protected, dict):
        raise TestReadRefusedError("freeze record manifest binding is incomplete")
    try:
        fit_path = Path(fit["path"])
        protected_path = Path(protected["path"])
    except (KeyError, TypeError, ValueError) as error:
        raise TestReadRefusedError(
            "freeze record has invalid manifest paths"
        ) from error
    directory = (
        Path(target_dir).resolve()
        if target_dir is not None
        else fit_path.parent.resolve()
    )
    for item, path, expected in (
        (fit, fit_path, directory / "manifest.json"),
        (protected, protected_path, directory / "protected-test" / "manifest.json"),
    ):
        if path.resolve() != expected.resolve() or not path.is_file():
            raise TestReadRefusedError(
                "bound manifest is missing or has a changed path"
            )
        if not isinstance(item.get("sha256"), str) or item["sha256"] != _file_hash(
            path
        ):
            raise TestReadRefusedError(
                "bound manifest bytes changed since the freeze record"
            )
    try:
        fit_manifest = load_manifest(directory)
        test_manifest = load_protected_test_manifest(directory)
        load_complete_manifest(directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TestReadRefusedError("bound manifest or receipt is invalid") from error
    if (
        fit.get("schema") != fit_manifest.version
        or protected.get("schema") != test_manifest.version
    ):
        raise TestReadRefusedError(
            "bound manifest schema changed since the freeze record"
        )
    if any(entry.partition == "test" for entry in fit_manifest.entries):
        raise TestReadRefusedError("fit manifest contains protected test rows")
    if any(entry.partition != "test" for entry in test_manifest.entries):
        raise TestReadRefusedError("protected-test manifest contains non-test rows")
    if dict(fit_manifest.dependencies).get("manifest_scope") != "fit":
        raise TestReadRefusedError("fit manifest has an invalid scope")
    return record


def authorize_test_read(
    freeze_record: Path | str | None,
    run_record: Path | str | None,
    root: Path | str,
    fit_ids: Sequence[str] | None = None,
    target_dir: Path | str | None = None,
) -> Authorization:
    """Return the authorized runs, or refuse the test read."""
    record = verify_freeze_record(freeze_record, root, target_dir)
    body = record["body"]
    if run_record is None or not Path(run_record).is_file():
        raise TestReadRefusedError("test read refused: no run record")
    runs = _read_json(Path(run_record), "run record")
    run_body = runs.get("body")
    if not isinstance(run_body, dict) or runs.get("hash") != canonical_hash(run_body):
        raise TestReadRefusedError("run record hash does not match its content")
    if run_body.get("freeze_record_hash") != record["hash"]:
        raise TestReadRefusedError("run record belongs to a different freeze record")
    fits = run_body["fits"]
    wanted = list(fits) if fit_ids is None else list(fit_ids)
    listed = set(body["fit_ids"])
    for fit_id in wanted:
        if fit_id not in listed or fit_id not in fits:
            raise TestReadRefusedError(
                f"test read refused: {fit_id} is not in the freeze and run records"
            )
    entries = tuple(
        RunEntry(fit_id=fit_id, **fits[fit_id]) for fit_id in sorted(wanted)
    )
    return Authorization(record["hash"], bool(body["post_hoc"]), entries, body)


def log_test_read(
    reads_log: Path | str,
    record_hash: str,
    fit_ids: Sequence[str],
    *,
    operation: str = "evaluate_test",
    additional_logs: Sequence[Path | str] = (),
    details: Mapping[str, Any] | None = None,
    refuse_conflict: bool = False,
) -> None:
    """Append a test read so a later record can be flagged post-hoc.

    ``details`` names what the read used (training receipt and artifact digests for
    a prediction; pool, sample IDs and weights for a residual audit), so the
    ledger row can be checked later rather than trusted by its operation string.

    With ``refuse_conflict`` the check against earlier rows of the same fit and
    operation runs after every log's lock is held and before any line is
    written, so two readers cannot both pass the check and then both append.

    Raises:
        TestReadRefusedError: ``refuse_conflict`` and an earlier row for this
            record and fit carries different details; nothing is written.
        FreezeRecordError: a log is locked by another writer.
    """
    line: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(),
        "record_hash": record_hash,
        "fit_ids": list(fit_ids),
        "operation": operation,
    }
    if details is not None:
        line["details"] = dict(details)
    encoded = json.dumps(line, sort_keys=True) + "\n"
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in (Path(reads_log), *(Path(path) for path in additional_logs)):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)
    with contextlib.ExitStack() as held:
        for path in paths:
            held.enter_context(_locked(path))
        if refuse_conflict:
            for fit_id in fit_ids:
                refuse_conflicting_repeat(
                    paths, record_hash, fit_id, operation, details or {}
                )
        for path in paths:
            _append_durably(path, encoded)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold the exclusive lock of one log, waiting for another writer to finish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path.with_name(f"{path.name}.lock"))
    deadline = time.monotonic() + LOG_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock.acquire()
            break
        except FitLockedError as error:
            if time.monotonic() >= deadline:
                raise FreezeRecordError(
                    f"test-read log is locked by another writer: {path}"
                ) from error
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.release()


def _append_durably(path: Path, text: str) -> None:
    """Append one line and flush it to disk; the caller holds the log's lock.

    The line is on disk before the caller opens any test image.
    """
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def refuse_conflicting_repeat(
    logs: Iterable[Path | str],
    record_hash: str,
    fit_id: str,
    operation: str,
    details: Mapping[str, Any],
) -> None:
    """Refuse a second read of one fit under one record from a different fit state.

    A repeat read may reproduce the same predictions, but the ledger must not
    show one record reading two different checkpoints or training receipts for
    the same fit ID.

    Raises:
        TestReadRefusedError: an earlier row for this record and fit carries
            different details.
    """
    for log in logs:
        for entry in read_reads_log(Path(log)):
            earlier = entry.get("details")
            if (
                entry.get("record_hash") == record_hash
                and entry.get("fit_ids") == [fit_id]
                and entry.get("operation") == operation
                and isinstance(earlier, dict)
                and any(earlier.get(key) != value for key, value in details.items())
            ):
                raise TestReadRefusedError(
                    f"test read refused: {fit_id} was already read under this "
                    "freeze record with a different training receipt or artifact"
                )

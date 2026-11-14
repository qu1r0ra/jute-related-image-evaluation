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

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jute_disease.data import (
    POLICIES,
    audit_report_outcome,
    policy_matrix_binding,
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


def _label_reads(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the reads that can steer a fit; the residual audit reads pixels only."""
    return [
        entry for entry in entries if entry.get("operation") != RESIDUAL_AUDIT_OPERATION
    ]


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


def _bind_mask_audit(report_path: Path | str) -> dict[str, Any]:
    """Read an evaluation report and bind its digest and derived outcome."""
    path = Path(report_path).resolve()
    try:
        raw = path.read_bytes()
        outcome = audit_report_outcome(json.loads(raw))
    except (OSError, ValueError) as error:
        raise FreezeRecordError(f"cannot bind mask audit report {path}: {error}") from (
            error
        )
    return {
        **outcome,
        "report": str(path),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _bind_policy_matrix(
    policy_dir: Path | str, matrix: Mapping[str, Sequence[Any]] | None
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
        "path": str(Path(policy_dir).resolve()),
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
    matrix: Mapping[str, Sequence[Any]] | None = None,
    required_fit_ids: Iterable[str] = (),
    reads_log: Path | str | None = None,
) -> dict[str, Any]:
    """Write the hashed matrix-freeze record; never replaces an existing file.

    The one record binds the target manifests, every policy-matrix file (and so
    every draw's protected-test manifest), the mask audit report digest and the
    diagnostic pool, view and policy. The mask-audit outcome is derived from the
    report, so a spec that carries its own is refused.
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
    mask_audit = _bind_mask_audit(mask_audit_report)
    policy_binding = _bind_policy_matrix(policy_dir, matrix)
    _check_diagnostic(spec["diagnostic"], policy_binding["matrix"])
    _check_border(spec["border_probe"])
    _check_outcomes(spec, validation_outcomes, mask_audit)
    manifests = _manifest_binding(target_dir)
    default_log = default_reads_log_path(root)
    prior_reads = _label_reads(read_reads_log(default_log))
    if reads_log is not None and Path(reads_log).resolve() != default_log.resolve():
        known_reads = {canonical_hash(entry) for entry in prior_reads}
        for entry in _label_reads(read_reads_log(Path(reads_log))):
            entry_hash = canonical_hash(entry)
            if entry_hash not in known_reads:
                prior_reads.append(entry)
                known_reads.add(entry_hash)
    body = {
        **{key: spec[key] for key in REQUIRED_KEYS},
        "fit_ids": sorted(fit_ids),
        "analysis_ids": sorted(analysis_ids),
        "invalidated_reads": prior_reads,
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


VARIANTS = ("full", "leaf_only", "background_only", "border", "pad_cue")


def _inside(base: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` under ``base``, refusing anything that escapes it."""
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise FreezeRecordError(f"{relative} lies outside the run store")
    return path


def _fit_lineage(
    store: RunStore, base: Path, fit_id: str, run: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one fit's receipt and its checkpoint or model by digest.

    Deep fits carry a ``checkpoint`` artifact and a fit-description mapping;
    classical fits carry a ``model`` artifact and a fit ID string. Both are
    accepted, and a fit without a verified receipt is refused.
    """
    receipt = store.receipt(fit_id, missing_ok=True)
    if receipt is None:
        raise FreezeRecordError(f"no verified training receipt for {fit_id}")
    run_dir = store.run_dir(fit_id)
    try:
        raw_receipt = (run_dir / "receipt.json").read_bytes()
    except OSError as error:
        raise FreezeRecordError(f"cannot read the training receipt of {fit_id}") from (
            error
        )
    results = receipt.get("results")
    artifacts = receipt.get("artifacts")
    if not isinstance(results, dict) or not isinstance(artifacts, dict):
        raise FreezeRecordError(f"training receipt of {fit_id} is malformed")
    fit = results.get("fit")
    if isinstance(fit, str):
        description: dict[str, Any] = {"fit_id": fit, **results}
    elif isinstance(fit, dict):
        description = fit
    else:
        raise FreezeRecordError(f"training receipt has no fit metadata for {fit_id}")
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
    kind = "checkpoint" if "checkpoint" in artifacts else "model"
    artifact = artifacts.get(kind)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
        raise FreezeRecordError(
            f"training receipt of {fit_id} binds no checkpoint or model"
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
    metrics = results.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    score = metrics.get("val_f1", metrics.get("val_macro_f1"))
    entry: dict[str, Any] = {
        kind: path.relative_to(base).as_posix(),
        f"{kind}_sha256": observed,
        "training_receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
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
    return entry


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
    that fit's checkpoint or model digest and the predictions digest.
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
            digest = _file_hash(path)
        except OSError as error:
            raise FreezeRecordError(f"cannot read predictions: {path}") from error
        entries[fit_id] = {
            "system": run["system"],
            "seed": int(run["seed"]),
            "variant": variant,
            "predictions": path.relative_to(base).as_posix(),
            "predictions_sha256": digest,
            "of": run.get("of"),
            **_fit_lineage(store, base, source, run),
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


def _verify_mask_audit(bound: object) -> None:
    """The bound report must still exist, unchanged, with the same outcome."""
    if not isinstance(bound, dict) or not isinstance(bound.get("report"), str):
        raise TestReadRefusedError("freeze record binds no mask audit report")
    try:
        raw = Path(bound["report"]).read_bytes()
        outcome = audit_report_outcome(json.loads(raw))
    except (OSError, ValueError) as error:
        raise TestReadRefusedError(
            "bound mask audit report is missing or invalid"
        ) from error
    if hashlib.sha256(raw).hexdigest() != bound.get("report_sha256") or any(
        bound.get(key) != value for key, value in outcome.items()
    ):
        raise TestReadRefusedError(
            "bound mask audit report changed since the freeze record"
        )


def _verify_policy_binding(bound: object) -> None:
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
        current = policy_matrix_binding(bound["path"], **matrix)
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
    _verify_mask_audit(body.get("mask_audit"))
    _verify_policy_binding(body.get("policy_binding"))
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
) -> None:
    """Append a test read so a later record can be flagged post-hoc."""
    line = {
        "time": datetime.now(UTC).isoformat(),
        "record_hash": record_hash,
        "fit_ids": list(fit_ids),
        "operation": operation,
    }
    encoded = json.dumps(line, sort_keys=True) + "\n"
    paths = [Path(reads_log), *(Path(path) for path in additional_logs)]
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _append_durably(path, encoded)


def _append_durably(path: Path, text: str) -> None:
    """Append one line under an exclusive lock and flush it to disk.

    Concurrent readers of the project wait for the lock instead of interleaving,
    and the line is on disk before the caller opens any test image.
    """
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
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        lock.release()


def invalidated_reads(
    reads_log: Path | str | None, record_hash: str
) -> list[dict[str, Any]]:
    """Earlier reads made under a different record; they no longer stand.

    The residual-contamination audit is logged but never invalidates: it opens
    protected-test pixels to look for near-duplicates of training images and
    reads no label, score or prediction, so nothing it sees can steer a fit.
    """
    return [
        entry
        for entry in read_reads_log(None if reads_log is None else Path(reads_log))
        if entry["record_hash"] != record_hash
        and entry.get("operation") != RESIDUAL_AUDIT_OPERATION
    ]

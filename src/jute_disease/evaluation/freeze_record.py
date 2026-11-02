"""Matrix-freeze record, run record and the gate in front of every test read.

The freeze record is one hashed JSON file listing every planned fit and analysis,
the validation-selected choices, the letterbox decision, the mask-audit outcome,
the border-probe values, the post-pilot budget and the hashes of the pipeline
code and environment lock. The evaluator refuses a test read for any fit that is
not listed in a record whose hash matches, or that has no run record bound to
that record. A new record invalidates earlier reads; a record created after any
test read is flagged post-hoc.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECORD_SCHEMA = 2
RUN_RECORD_SCHEMA = 1
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
    "mask_audit",
    "border_probe",
    "budget",
)


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
    spec: Mapping[str, Any], outcomes: Mapping[str, Any] | None
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
    audit = outcomes.get("mask_audit")
    freeze_audit = spec.get("mask_audit")
    if (
        not isinstance(audit, dict)
        or not isinstance(audit.get("passed"), bool)
        or not isinstance(freeze_audit, dict)
        or audit["passed"] != freeze_audit.get("passed")
    ):
        raise FreezeRecordError(
            "freeze mask-audit outcome must match the recorded validation outcome"
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


def write_freeze_record(
    spec: Mapping[str, Any],
    validation_outcomes: Mapping[str, Any] | None,
    output: Path | str,
    root: Path | str,
    *,
    required_fit_ids: Iterable[str] = (),
    reads_log: Path | str | None = None,
) -> dict[str, Any]:
    """Write the hashed matrix-freeze record; never replaces an existing file."""
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
    if not isinstance(spec["mask_audit"], dict) or not isinstance(
        spec["mask_audit"].get("passed"), bool
    ):
        raise FreezeRecordError("mask_audit needs a boolean 'passed' outcome")
    _check_border(spec["border_probe"])
    _check_outcomes(spec, validation_outcomes)
    default_log = default_reads_log_path(root)
    prior_reads = read_reads_log(default_log)
    if reads_log is not None and Path(reads_log).resolve() != default_log.resolve():
        known_reads = {canonical_hash(entry) for entry in prior_reads}
        for entry in read_reads_log(Path(reads_log)):
            entry_hash = canonical_hash(entry)
            if entry_hash not in known_reads:
                prior_reads.append(entry)
                known_reads.add(entry_hash)
    body = {
        **{key: spec[key] for key in REQUIRED_KEYS},
        "fit_ids": sorted(fit_ids),
        "analysis_ids": sorted(analysis_ids),
        "invalidated_reads": prior_reads,
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
    training_receipt_sha256: str | None = None
    validation_macro_f1: float | None = None
    pool: str | None = None
    view: int | None = None
    policy: str | None = None
    split: int | None = None
    seed_index: int | None = None
    assignment_table_sha256: str | None = None


VARIANTS = ("full", "leaf_only", "background_only", "border", "pad_cue")


def write_run_record(
    freeze_record: Path | str,
    runs: Sequence[Mapping[str, Any]],
    base_dir: Path | str,
    output: Path | str,
) -> dict[str, Any]:
    """Bind completed fits and their prediction digests to one freeze record."""
    record = load_freeze_record(freeze_record)
    listed = set(record["body"]["fit_ids"])
    entries: dict[str, dict[str, Any]] = {}
    from jute_disease.training import RunStore

    store = RunStore(base_dir)
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
        path = Path(base_dir) / str(run["predictions"])
        try:
            digest = _file_hash(path)
        except OSError as error:
            raise FreezeRecordError(f"cannot read predictions: {path}") from error
        entry = {
            "system": run["system"],
            "seed": int(run["seed"]),
            "variant": variant,
            "predictions": str(run["predictions"]),
            "predictions_sha256": digest,
            "of": run.get("of"),
        }
        if variant == "full":
            receipt = store.receipt(str(fit_id), missing_ok=True)
            if receipt is not None:
                try:
                    raw_receipt = (
                        store.run_dir(str(fit_id)) / "receipt.json"
                    ).read_bytes()
                    checkpoint = receipt.get("artifacts", {}).get("checkpoint")
                    results = receipt.get("results", {})
                    fit = results.get("fit", {})
                    metrics = results.get("metrics", {})
                    score = metrics.get("val_f1", metrics.get("val_macro_f1"))
                    if not isinstance(fit, dict):
                        raise FreezeRecordError(
                            f"training receipt has no fit metadata for {fit_id}"
                        )
                    if fit.get("fit_id", fit_id) != fit_id:
                        raise FreezeRecordError(
                            f"training receipt fit ID does not match {fit_id}"
                        )
                    if fit.get("seed") != int(run["seed"]):
                        raise FreezeRecordError(
                            "training receipt seed does not match run record "
                            f"for {fit_id}"
                        )
                    receipt_system = fit.get("system")
                    if receipt_system is not None and receipt_system != run["system"]:
                        raise FreezeRecordError(
                            "training receipt system does not match run record "
                            f"for {fit_id}"
                        )
                    if isinstance(checkpoint, dict):
                        relative_checkpoint = str(checkpoint["path"])
                        fit_dir = store.run_dir(str(fit_id))
                        checkpoint_path = (fit_dir / relative_checkpoint).resolve()
                        checkpoint_path.relative_to(Path(base_dir).resolve())
                        if not checkpoint_path.is_file() or _file_hash(
                            checkpoint_path
                        ) != checkpoint.get("sha256"):
                            raise FreezeRecordError(
                                "checkpoint artifact does not match training "
                                f"receipt for {fit_id}"
                            )
                        entry["checkpoint"] = checkpoint_path.relative_to(
                            Path(base_dir).resolve()
                        ).as_posix()
                        entry["checkpoint_sha256"] = str(checkpoint["sha256"])
                        entry["training_receipt_sha256"] = hashlib.sha256(
                            raw_receipt
                        ).hexdigest()
                        if isinstance(score, int | float) and not isinstance(
                            score, bool
                        ):
                            entry["validation_macro_f1"] = float(score)
                    for field in ("pool", "view", "policy", "split", "seed_index"):
                        if field in fit:
                            entry[field] = fit[field]
                    metadata = receipt.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    assignment = metadata.get("assignment_table")
                    if not isinstance(assignment, dict):
                        assignment = {}
                    assignment_digest = assignment.get("sha256")
                    if isinstance(assignment_digest, str):
                        entry["assignment_table_sha256"] = assignment_digest
                except (KeyError, TypeError, ValueError, OSError) as error:
                    if isinstance(error, FreezeRecordError):
                        raise
                    raise FreezeRecordError(
                        f"cannot read checkpoint lineage for {fit_id}"
                    ) from error
        entries[fit_id] = entry
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


def verify_freeze_record(
    freeze_record: Path | str | None, root: Path | str
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
    return record


def authorize_test_read(
    freeze_record: Path | str | None,
    run_record: Path | str | None,
    root: Path | str,
    fit_ids: Sequence[str] | None = None,
) -> Authorization:
    """Return the authorized runs, or refuse the test read."""
    record = verify_freeze_record(freeze_record, root)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)


def invalidated_reads(
    reads_log: Path | str | None, record_hash: str
) -> list[dict[str, Any]]:
    """Earlier reads made under a different record; they no longer stand."""
    return [
        entry
        for entry in read_reads_log(None if reads_log is None else Path(reads_log))
        if entry["record_hash"] != record_hash
    ]

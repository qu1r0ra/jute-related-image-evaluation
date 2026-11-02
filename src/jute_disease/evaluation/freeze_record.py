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

RECORD_SCHEMA = 1
RUN_RECORD_SCHEMA = 1
PIPELINE_DIRS = ("src", "configs")
LOCKFILE = "uv.lock"
REQUIRED_ANALYSES = (
    "background-probe",
    "border-probe",
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
    choice = spec["letterbox_decision"].get("choice")
    if choice not in comparison:
        raise FreezeRecordError("letterbox decision is not one of the compared fits")
    recorded = outcomes.get("selections")
    if not isinstance(recorded, dict):
        raise FreezeRecordError("validation outcomes need the selection outcomes")
    missing = sorted(set(spec["selections"]) - set(recorded))
    if missing:
        raise FreezeRecordError(
            f"selections without a recorded validation outcome: {missing}"
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
    if not isinstance(spec["mask_audit"].get("passed"), bool):
        raise FreezeRecordError("mask_audit needs a boolean 'passed' outcome")
    _check_border(spec["border_probe"])
    _check_outcomes(spec, validation_outcomes)
    prior_reads = read_reads_log(None if reads_log is None else Path(reads_log))
    body = {
        **{key: spec[key] for key in REQUIRED_KEYS},
        "fit_ids": sorted(fit_ids),
        "analysis_ids": sorted(analysis_ids),
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
        entries[fit_id] = {
            "system": run["system"],
            "seed": int(run["seed"]),
            "variant": variant,
            "predictions": str(run["predictions"]),
            "predictions_sha256": digest,
            "of": run.get("of"),
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
    reads_log: Path | str, record_hash: str, fit_ids: Sequence[str]
) -> None:
    """Append a test read so a later record can be flagged post-hoc."""
    path = Path(reads_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(UTC).isoformat(),
        "record_hash": record_hash,
        "fit_ids": list(fit_ids),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def invalidated_reads(
    reads_log: Path | str | None, record_hash: str
) -> list[dict[str, Any]]:
    """Earlier reads made under a different record; they no longer stand."""
    return [
        entry
        for entry in read_reads_log(None if reads_log is None else Path(reads_log))
        if entry["record_hash"] != record_hash
    ]

"""Resumable training queue: deadline admission, receipts and stage execution."""

import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jute_disease.config import ARTIFACTS_DIR, DEFAULT_SEED, MANIFEST_DIR

if TYPE_CHECKING:
    from jute_disease.data import SourceSpec

CLASSICAL_FIT_SECONDS = 1800
CLASSICAL_AGGREGATE_SECONDS = 5 * 3600
SHARED_PREP_SECONDS = 1800
ARMS = ("plantvillage", "plantdoc")
RECIPES_DIR = Path("configs/corrected")
QUEUE_DIR = ARTIFACTS_DIR / "queue"
SOURCES_DIR = ARTIFACTS_DIR / "sources"

_CLI = (sys.executable, "-m", "jute_disease.cli")


def check_deadline(config: dict[str, Any], now: str) -> dict[str, Any]:
    """Check a forecast using a default and minimum two-hour reserve.

    The cutoff, its timezone and verification evidence must be explicit.
    A passing forecast does not start or admit a run.
    """
    if (
        not config.get("cutoff")
        or config.get("verified") is not True
        or not isinstance(config.get("verification"), str)
        or not config["verification"].strip()
        or not config.get("timezone")
    ):
        raise ValueError(
            "an explicit verified cutoff, timezone and evidence are required"
        )
    try:
        zone = ZoneInfo(config["timezone"])
        cutoff = datetime.fromisoformat(config["cutoff"])
        current = datetime.fromisoformat(now)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("invalid cutoff, timezone or clock") from error
    if cutoff.utcoffset() is None or current.utcoffset() is None:
        raise ValueError("cutoff and clock must include a UTC offset")
    if cutoff.utcoffset() != cutoff.astimezone(zone).utcoffset():
        raise ValueError("cutoff offset does not match timezone")
    seconds = []
    for key in ("mandatory_seconds", "optional_seconds", "finalization_seconds"):
        value = config.get(key, 2 * 3600 if key == "finalization_seconds" else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a nonnegative finite number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative finite number")
        seconds.append(value)
    mandatory, optional, reserve = seconds
    if reserve < 2 * 3600:
        raise ValueError("finalization reserve must be at least two hours")
    projected = current.astimezone(UTC) + timedelta(
        seconds=mandatory + optional + reserve
    )
    if projected > cutoff.astimezone(UTC):
        raise ValueError("forecast would consume the finalization reserve")
    return {"forecast_fits": True, "projected_end": projected.isoformat()}


class QueueNotAdmittedError(RuntimeError):
    """The queue was asked to run without an explicit admission."""


class QueueEscalationError(RuntimeError):
    """A mandatory stage cannot be admitted or completed; the owner decides."""


@dataclass(frozen=True)
class Stage:
    """One queue stage; ``command`` runs in an isolated subprocess."""

    name: str
    group: str
    command: tuple[str, ...]
    timeout_seconds: float | None
    estimate_seconds: float
    mandatory: bool
    outputs: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Stage":
        fields = dict(data)
        for key in ("command", "outputs", "inputs", "upstream"):
            fields[key] = tuple(fields.get(key, ()))
        return cls(**fields)


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    timed_out: bool = False
    seconds: float = 0.0


Executor = Callable[[Stage, float | None], ExecResult]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _train_dl_fit(seed: int, recipe: Path, root: Path) -> tuple[str, ...]:
    return (
        *_CLI,
        "train",
        "dl",
        "fit",
        "--seed_everything",
        str(seed),
        "--config",
        str(recipe),
        "--trainer.default_root_dir",
        str(root),
    )


def build_plan(
    estimates: Mapping[str, float],
    seed: int = DEFAULT_SEED,
    *,
    recipes_dir: Path = RECIPES_DIR,
    state_dir: Path = QUEUE_DIR,
    target_manifest_dir: Path = MANIFEST_DIR,
    source_root: Path = SOURCES_DIR,
) -> list[Stage]:
    """Return the ordered stages for one seed.

    Order follows the benchmark: core CNN fits, the ten classical fits, then
    the optional PlantVillage and PlantDoc arms (source, then target). Core and
    arm stages are configured by recipes under ``recipes_dir``; ``estimates``
    holds a refreshed ``core_seconds``, ``plantvillage_seconds`` and
    ``plantdoc_seconds`` forecast.

    Raises:
        ValueError: an estimate or a recipe is missing.
    """
    # Imported here: the data and model packages import this package's logger.
    from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS

    for key in ("core_seconds", *(f"{arm}_seconds" for arm in ARMS)):
        value = estimates.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative number of seconds")
    core = sorted((recipes_dir / "core").glob("*.yaml"))
    if not core:
        raise ValueError(f"no core recipes under {recipes_dir / 'core'}")
    target_receipt = str(target_manifest_dir / "receipt.json")
    outputs = state_dir / "outputs"
    stages: list[Stage] = []
    for recipe in core:
        root = outputs / f"core-{recipe.stem}"
        stages.append(
            Stage(
                name=f"core-{recipe.stem}",
                group="core",
                command=_train_dl_fit(seed, recipe, root),
                timeout_seconds=None,
                estimate_seconds=estimates["core_seconds"] / len(core),
                mandatory=True,
                outputs=(str(root),),
                inputs=(str(recipe), target_receipt),
            )
        )
    for feature_type in FEATURE_EXTRACTORS:
        for classifier in ML_CLASSIFIERS:
            name = f"classical-{classifier}-{feature_type}"
            directory = outputs / name
            stages.append(
                Stage(
                    name=name,
                    group="classical",
                    command=(
                        *_CLI,
                        "train",
                        "ml",
                        "--classifier",
                        classifier,
                        "--feature_type",
                        feature_type,
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(directory),
                    ),
                    timeout_seconds=CLASSICAL_FIT_SECONDS,
                    estimate_seconds=CLASSICAL_FIT_SECONDS,
                    mandatory=True,
                    outputs=(str(directory / f"{classifier}_{feature_type}.joblib"),),
                    inputs=(target_receipt,),
                )
            )
    for arm in ARMS:
        recipe = recipes_dir / arm / "target.yaml"
        if not recipe.is_file():
            raise ValueError(f"missing {arm} target recipe: {recipe}")
        manifest = source_root / arm / "manifest"
        checkpoint = outputs / arm / "source.ckpt"
        half = estimates[f"{arm}_seconds"] / 2
        stages.append(
            Stage(
                name=f"{arm}-source",
                group=arm,
                command=(
                    *_CLI,
                    "train",
                    "pretrain",
                    "--manifest_dir",
                    str(manifest),
                    "--image_root",
                    str(source_root / arm / "package"),
                    "--target_manifest_dir",
                    str(target_manifest_dir),
                    "--output_path",
                    str(checkpoint),
                    "--seed",
                    str(seed),
                ),
                timeout_seconds=None,
                estimate_seconds=half,
                mandatory=False,
                outputs=(str(checkpoint),),
                inputs=(str(manifest / "receipt.json"), target_receipt),
                source=arm,
            )
        )
        root = outputs / f"{arm}-target"
        stages.append(
            Stage(
                name=f"{arm}-target",
                group=arm,
                command=_train_dl_fit(seed, recipe, root),
                timeout_seconds=None,
                estimate_seconds=half,
                mandatory=False,
                outputs=(str(root),),
                inputs=(str(recipe), target_receipt),
                upstream=(f"{arm}-source",),
                source=arm,
            )
        )
    return stages


def plan_to_json(plan: Sequence[Stage], seed: int = DEFAULT_SEED) -> dict[str, Any]:
    return {"seed": seed, "stages": [stage.to_json() for stage in plan]}


def plan_from_json(data: dict[str, Any]) -> list[Stage]:
    return [Stage.from_json(item) for item in data["stages"]]


def _hash_path(path: str) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
    elif target.is_dir():
        for child in sorted(p for p in target.rglob("*") if p.is_file()):
            digest.update(child.relative_to(target).as_posix().encode())
            digest.update(child.read_bytes())
    else:
        return "absent"
    return digest.hexdigest()


def _identities(stages: Sequence[Stage]) -> dict[str, str]:
    """Bind each stage to its command, input contents and upstream identity."""
    identities: dict[str, str] = {}
    for stage in stages:
        payload = {
            "command": list(stage.command),
            "inputs": {path: _hash_path(path) for path in stage.inputs},
            "upstream": {name: identities[name] for name in stage.upstream},
        }
        identities[stage.name] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
    return identities


def _receipt_path(state_dir: Path, name: str) -> Path:
    return state_dir / "receipts" / f"{name}.json"


def _write_receipt(state_dir: Path, name: str, payload: dict[str, Any]) -> None:
    path = _receipt_path(state_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(temporary, path)


def _read_receipt(state_dir: Path, name: str) -> dict[str, Any] | None:
    try:
        return json.loads(_receipt_path(state_dir, name).read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _completed_receipt(
    state_dir: Path, stage: Stage, identity: str
) -> dict[str, Any] | None:
    """Return the completed receipt when it is compatible and its outputs remain."""
    found = _read_receipt(state_dir, stage.name)
    if (
        found is not None
        and found.get("status") == "completed"
        and found.get("identity") == identity
        and all(Path(path).exists() for path in found.get("outputs_present", ()))
    ):
        return found
    return None


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _kill_tree(process: "subprocess.Popen[bytes]") -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process.kill()


def subprocess_executor(stage: Stage, timeout: float | None) -> ExecResult:
    """Run a stage in its own process tree; kill the tree at the timeout."""
    options: dict[str, Any]
    if os.name == "nt":
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    else:
        options = {"start_new_session": True}
    started = time.monotonic()
    process = subprocess.Popen(list(stage.command), **options)
    try:
        code = process.wait(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        code = process.wait() or -9
        timed_out = True
    except BaseException:
        _kill_tree(process)
        process.wait()
        raise
    return ExecResult(code, timed_out, time.monotonic() - started)


def _shortfall(
    config: dict[str, Any], now: datetime, mandatory: float, optional: float
) -> str | None:
    """Return why the forecast fails, or None; nothing is started either way."""
    try:
        check_deadline(
            {**config, "mandatory_seconds": mandatory, "optional_seconds": optional},
            now.isoformat(),
        )
    except ValueError as error:
        return str(error)
    return None


def run_queue(
    plan: Sequence[Stage],
    state_dir: Path,
    config: dict[str, Any],
    *,
    admit: bool,
    executor: Executor = subprocess_executor,
    clock: Clock = _utc_now,
    sources: Mapping[str, "SourceSpec"] | None = None,
    aggregate_seconds: float = CLASSICAL_AGGREGATE_SECONDS,
) -> dict[str, list[str]]:
    """Run the plan with receipts, resume and deadline admission.

    Mandatory stages (core and classical) run first and a shortfall escalates.
    Optional arms run PlantVillage then PlantDoc; an arm that
    ``readiness_blockers`` blocks, or that does not fit the deadline, is omitted
    with its cause, PlantDoc first. Nothing opens the test partition.

    Raises:
        QueueNotAdmittedError: ``admit`` is false.
        QueueEscalationError: a mandatory stage cannot be admitted or completed.
    """
    if not admit:
        raise QueueNotAdmittedError(
            "the queue never starts without an explicit admission"
        )
    from jute_disease.data import SOURCES, readiness_blockers

    known = SOURCES if sources is None else sources
    stages = list(plan)
    identities = _identities(stages)
    summary: dict[str, list[str]] = {
        "completed": [],
        "skipped": [],
        "omitted": [],
        "failed": [],
    }

    def done(stage: Stage) -> dict[str, Any] | None:
        return _completed_receipt(state_dir, stage, identities[stage.name])

    def settle(stage: Stage, status: str, **extra: Any) -> None:
        _write_receipt(
            state_dir,
            stage.name,
            {
                "stage": stage.name,
                "status": status,
                "identity": identities[stage.name],
                **extra,
            },
        )

    def omit(stage: Stage, cause: str) -> None:
        settle(stage, "omitted", cause=cause)
        summary["omitted"].append(stage.name)

    def execute(stage: Stage, timeout: float | None) -> tuple[str, ExecResult]:
        extra: dict[str, Any] = {"command": list(stage.command)}
        # Outputs without a compatible completed receipt are partial: reject them.
        rejected = [path for path in stage.outputs if Path(path).exists()]
        for path in rejected:
            _remove(Path(path))
        if rejected:
            extra["rejected_partial"] = rejected
        started = clock().isoformat()
        settle(stage, "running", started_at=started, **extra)
        try:
            result = executor(stage, timeout)
        except KeyboardInterrupt:
            settle(stage, "interrupted", started_at=started, **extra)
            raise
        if result.timed_out:
            status = "timed_out"
        else:
            status = "completed" if result.returncode == 0 else "failed"
        settle(
            stage,
            status,
            started_at=started,
            finished_at=clock().isoformat(),
            seconds=result.seconds,
            returncode=result.returncode,
            outputs_present=[p for p in stage.outputs if Path(p).exists()],
            **extra,
        )
        return status, result

    # Mandatory stages: forecast first, then run in plan order.
    mandatory = [stage for stage in stages if stage.mandatory]
    used = sum(
        found.get("seconds", 0.0)
        for stage in mandatory
        if stage.group == "classical" and (found := done(stage)) is not None
    )
    pending = [stage for stage in mandatory if done(stage) is None]
    classical = [stage for stage in pending if stage.group == "classical"]
    forecast = sum(s.estimate_seconds for s in pending if s.group != "classical")
    if classical:
        fitting = sum(s.estimate_seconds for s in classical)
        forecast += min(fitting, max(aggregate_seconds - used, 0)) + SHARED_PREP_SECONDS
    if (error := _shortfall(config, clock(), forecast, 0)) is not None:
        raise QueueEscalationError(f"mandatory stages cannot be admitted: {error}")

    for stage in mandatory:
        if done(stage) is not None:
            summary["skipped"].append(stage.name)
            continue
        timeout = stage.timeout_seconds
        if stage.group == "classical":
            remaining = aggregate_seconds - used
            if remaining <= 0:
                settle(stage, "failed", cause="aggregate fitting allowance exhausted")
                summary["failed"].append(stage.name)
                raise QueueEscalationError(
                    f"aggregate fitting allowance of {aggregate_seconds:.0f} s is "
                    f"exhausted before {stage.name}"
                )
            timeout = remaining if timeout is None else min(timeout, remaining)
        status, result = execute(stage, timeout)
        if stage.group == "classical":
            used += result.seconds
        if status != "completed":
            summary["failed"].append(stage.name)
            raise QueueEscalationError(f"mandatory stage {stage.name} {status}")
        summary["completed"].append(stage.name)

    # Optional arms in plan order, each admitted against the refreshed clock.
    candidates: list[list[Stage]] = []
    for group in ARMS:
        arm = [stage for stage in stages if stage.group == group]
        if not arm:
            continue
        spec = known.get(arm[0].source or "")
        blockers = (
            readiness_blockers(spec) if spec is not None else ("Source is not pinned",)
        )
        if blockers:
            for stage in arm:
                omit(stage, "source not ready: " + "; ".join(blockers))
        else:
            candidates.append(arm)

    def arm_seconds(arm: list[Stage]) -> float:
        return sum(s.estimate_seconds for s in arm if done(s) is None)

    while candidates:
        kept = list(candidates)
        reason = ""
        while kept:
            reason = _shortfall(config, clock(), 0, sum(map(arm_seconds, kept))) or ""
            if not reason:
                break
            kept.pop()
        for arm in candidates[len(kept) :]:
            for stage in arm:
                omit(stage, f"deadline: {reason}")
        if not kept:
            break
        arm, candidates = kept[0], kept[1:]
        broken = ""
        for stage in arm:
            if broken:
                omit(stage, f"upstream stage {broken}")
            elif done(stage) is not None:
                summary["skipped"].append(stage.name)
            else:
                status, _ = execute(stage, stage.timeout_seconds)
                if status == "completed":
                    summary["completed"].append(stage.name)
                else:
                    summary["failed"].append(stage.name)
                    broken = f"{stage.name} {status}"
    return summary

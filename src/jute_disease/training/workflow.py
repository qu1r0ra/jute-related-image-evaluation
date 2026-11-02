"""Connect the queue to the fit engines through one child process per fit.

The queue decides what runs and when. This module decides how: it opens a run in
the store, launches ``train fit`` as a child so the time limit is enforced from
outside, and turns the child's result into a receipt. The same ``run_job`` body
runs in the child and, in tests, in process, so there is one code path.

Imports of ``jute_disease.data`` and the fit engines are deferred: both import
``jute_disease.training``, which exports this module.
"""

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from jute_disease.training.benchmark import (
    Benchmark,
    FitKind,
    FitSpec,
    fit_identity_digest,
    fit_identity_payload,
    load_benchmark,
)
from jute_disease.training.inputs import DataInputs, target_manifest_for_fit
from jute_disease.training.process import ChildResult, run_child
from jute_disease.training.queue import (
    Executor,
    FailureKind,
    IdentityFor,
    Outcome,
)
from jute_disease.training.receipts import (
    Resume,
    RunState,
    RunStore,
    read_json,
    write_json_atomic,
)
from jute_disease.training.resume import (
    CHECKPOINT_DIR,
    RESUME_FILE,
    ResumePlan,
    mark_finished,
    prepare_checkpoints,
)
from jute_disease.training.stack import verify_stack

RESULT_FILE = "result.json"
LOG_FILE = "child.log"
WORKSPACE_FILE = "workspace.json"
EVALUATION_SELECTION_FORMAT = "jute-evaluation-selection-v1"
CNN_KINDS = (FitKind.SOURCE, FitKind.TARGET)


class UpstreamError(RuntimeError):
    """The checkpoint a chained fit starts from is missing or no longer verifies."""


def _read_evaluation_selection(
    path: Path | str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Read fit specifications from a hash-valid #139 selection record."""
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        record = envelope["record"]
        model_specs = record["selection"]["model_specs"]
        canonical = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        selection_sha256 = hashlib.sha256(canonical).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("evaluation selection is missing or malformed") from error
    if (
        not isinstance(envelope, dict)
        or envelope.get("format") != EVALUATION_SELECTION_FORMAT
        or record.get("issue") != 139
        or envelope.get("sha256") != selection_sha256
        or not isinstance(model_specs, dict)
        or not model_specs
        or any(
            not isinstance(fit_id, str)
            or not fit_id
            or not isinstance(model_spec, dict)
            for fit_id, model_spec in model_specs.items()
        )
    ):
        raise ValueError("evaluation selection is missing or has an invalid hash")
    return model_specs, selection_sha256


@dataclass(frozen=True)
class Workspace:
    """Everything a child process needs to run any fit, saved as one JSON file."""

    benchmark_path: Path
    data: DataInputs
    runs_dir: Path
    cache_dir: Path
    pretrained: bool
    require_cuda: bool
    accelerator: str
    evaluation_fit_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    evaluation_selection_sha256: str | None = None
    benchmark: Benchmark = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        benchmark = load_benchmark(self.benchmark_path)
        assignments = load_assignment_table(self)
        if assignments["present"]:
            found = {(row["split"], row["seed"]) for row in assignments["assignments"]}
            expected = {
                (split, seed_index)
                for split in benchmark.splits
                for seed_index in range(len(benchmark.recipe.seeds))
            }
            missing = expected - found
            if missing:
                raise ValueError(
                    f"assignment table is missing split/seed rows: {sorted(missing)}"
                )
            sources = len(benchmark.recipe.source_seeds)
            if any(row["checkpoint"] >= sources for row in assignments["assignments"]):
                raise ValueError(
                    f"assignment table names a checkpoint beyond the {sources} "
                    "source seeds"
                )
            benchmark = replace(
                benchmark,
                assignments=tuple(
                    (row["split"], row["seed"], row["checkpoint"])
                    for row in assignments["assignments"]
                ),
            )
        if not self.require_cuda:
            benchmark = replace(
                benchmark,
                recipe=replace(benchmark.recipe, num_workers=0),
            )
        if self.evaluation_fit_specs and (
            self.evaluation_selection_sha256 is None
            or len(self.evaluation_selection_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.evaluation_selection_sha256
            )
        ):
            raise ValueError("evaluation fit specs need a hash-bound selection")
        object.__setattr__(self, "benchmark", benchmark)

    def save(self, path: Path | str) -> None:
        sources = {
            name: [str(manifest), str(root)]
            for name, (manifest, root) in self.data.sources.items()
        }
        write_json_atomic(
            path,
            {
                "benchmark_path": str(self.benchmark_path),
                "image_root": str(self.data.image_root),
                "target_dir": str(self.data.target_dir),
                "policy_dir": (
                    None if self.data.policy_dir is None else str(self.data.policy_dir)
                ),
                "pool_roots": {
                    key: str(value)
                    for key, value in (self.data.pool_roots or {}).items()
                },
                "mask_checkpoint": (
                    None
                    if self.data.mask_checkpoint is None
                    else str(self.data.mask_checkpoint)
                ),
                "fixture_masks": self.data.fixture_masks,
                "sources": sources,
                "runs_dir": str(self.runs_dir),
                "cache_dir": str(self.cache_dir),
                "pretrained": self.pretrained,
                "require_cuda": self.require_cuda,
                "accelerator": self.accelerator,
                "evaluation_fit_specs": self.evaluation_fit_specs,
                "evaluation_selection_sha256": self.evaluation_selection_sha256,
            },
        )

    @property
    def evaluation_fits(self) -> tuple[FitSpec, ...]:
        """Executable #139 probe fits not already declared by the benchmark."""
        declared = {fit.fit_id for fit in self.benchmark.fits()}
        fits: list[FitSpec] = []
        for fit_id, spec in self.evaluation_fit_specs.items():
            if fit_id in declared:
                continue
            kind_name = spec.get("kind")
            if kind_name == "cnn":
                kind = FitKind.TARGET
                system = spec.get("system")
                definition = self.benchmark.systems.get(system)
                if not isinstance(definition, dict):
                    raise ValueError(f"{fit_id} names an unknown CNN system")
                model = spec.get("model")
                if model != definition.get("model"):
                    raise ValueError(f"{fit_id} model differs from its CNN system")
                micro_batch = definition["micro_batch"]
                accumulation = self.benchmark.recipe.effective_batch // micro_batch
            elif kind_name in {"classical", "cue"}:
                kind = FitKind.CLASSICAL
                system = ""
                model = ""
                micro_batch = 0
                accumulation = 1
            else:
                raise ValueError(f"{fit_id} has an unsupported evaluation fit kind")
            fits.append(
                FitSpec(
                    fit_id=fit_id,
                    kind=kind,
                    seed=spec["seed"],
                    system=system,
                    model=model,
                    micro_batch=micro_batch,
                    accumulation=accumulation,
                    estimator=spec.get(
                        "estimator", "cue" if kind_name == "cue" else ""
                    ),
                    features=spec.get(
                        "features", "metadata" if kind_name == "cue" else ""
                    ),
                    pool=spec["pool"],
                    view=spec.get("view", 0),
                    policy=spec["policy"],
                    split=spec["split"],
                    phase="probe",
                )
            )
        return tuple(fits)

    @classmethod
    def load(cls, path: Path | str) -> Workspace:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            benchmark_path=Path(raw["benchmark_path"]),
            data=DataInputs(
                image_root=Path(raw["image_root"]),
                target_dir=Path(raw["target_dir"]),
                sources={
                    name: (Path(manifest), Path(root))
                    for name, (manifest, root) in raw["sources"].items()
                },
                policy_dir=(
                    None if raw.get("policy_dir") is None else Path(raw["policy_dir"])
                ),
                pool_roots={
                    key: Path(value) for key, value in raw.get("pool_roots", {}).items()
                },
                mask_checkpoint=(
                    None
                    if raw.get("mask_checkpoint") is None
                    else Path(raw["mask_checkpoint"])
                ),
                fixture_masks=raw.get("fixture_masks", False),
            ),
            runs_dir=Path(raw["runs_dir"]),
            cache_dir=Path(raw["cache_dir"]),
            pretrained=raw["pretrained"],
            require_cuda=raw["require_cuda"],
            accelerator=raw["accelerator"],
            evaluation_fit_specs=raw.get("evaluation_fit_specs", {}),
            evaluation_selection_sha256=raw.get("evaluation_selection_sha256"),
        )


def open_workspace(
    *,
    benchmark_path: Path | str,
    data: DataInputs,
    runs_dir: Path | str,
    cache_dir: Path | str,
    pretrained: bool = True,
    require_cuda: bool = True,
    accelerator: str = "auto",
    evaluation_selection_path: Path | str | None = None,
) -> Workspace:
    """Resolve paths to absolute and create the output roots.

    A source package root may be the download folder; the manifest paths are
    relative to the folder that holds the partitions, which is found when
    available. An unavailable optional source is validated at its own stage.
    """
    from jute_disease.data import SourceMismatchError, find_source_root

    def source_root(root: Path | str) -> Path:
        path = Path(root).resolve()
        try:
            return find_source_root(path).resolve()
        except OSError, SourceMismatchError:
            # Optional inputs are checked when their own stage is attempted.
            return path

    runs = Path(runs_dir).resolve()
    cache = Path(cache_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    evaluation_fit_specs: dict[str, dict[str, Any]] = {}
    evaluation_selection_sha256 = None
    if evaluation_selection_path is not None:
        evaluation_fit_specs, evaluation_selection_sha256 = _read_evaluation_selection(
            evaluation_selection_path
        )
    return Workspace(
        benchmark_path=Path(benchmark_path).resolve(),
        data=DataInputs(
            image_root=Path(data.image_root).resolve(),
            target_dir=Path(data.target_dir).resolve(),
            sources={
                name: (Path(manifest).resolve(), source_root(root))
                for name, (manifest, root) in data.sources.items()
            },
            policy_dir=(
                None if data.policy_dir is None else Path(data.policy_dir).resolve()
            ),
            pool_roots={
                key: Path(root).resolve()
                for key, root in (data.pool_roots or {}).items()
            },
            mask_checkpoint=(
                None
                if data.mask_checkpoint is None
                else Path(data.mask_checkpoint).expanduser().resolve()
            ),
            fixture_masks=data.fixture_masks,
        ),
        runs_dir=runs,
        cache_dir=cache,
        pretrained=pretrained,
        require_cuda=require_cuda,
        accelerator=accelerator,
        evaluation_fit_specs=evaluation_fit_specs,
        evaluation_selection_sha256=evaluation_selection_sha256,
    )


@dataclass(frozen=True)
class Job:
    """One child invocation: a fit, where to work, and what it may start from.

    ``bounds`` are train and validation batch limits; a bounded job is a timing
    or smoke run and is never a fit. A CNN fit given ``checkpoint_dir`` keeps
    its full training state there, and ``resume_from`` continues from it.
    """

    fit_id: str
    attempt_dir: Path
    result_path: Path
    upstream_checkpoint: Path | None = None
    bounds: tuple[int, int] | None = None
    checkpoint_dir: Path | None = None
    resume_from: Path | None = None

    def to_args(self) -> list[str]:
        args = [
            "--fit-id",
            self.fit_id,
            "--attempt-dir",
            str(self.attempt_dir),
            "--result",
            str(self.result_path),
        ]
        if self.upstream_checkpoint is not None:
            args += ["--upstream-checkpoint", str(self.upstream_checkpoint)]
        if self.bounds is not None:
            args += ["--bounds", f"{self.bounds[0]},{self.bounds[1]}"]
        if self.checkpoint_dir is not None:
            args += ["--checkpoint-dir", str(self.checkpoint_dir)]
        if self.resume_from is not None:
            args += ["--resume-from", str(self.resume_from)]
        return args


Launcher = Callable[[Job, float | None, Path], ChildResult]


def subprocess_launcher(workspace_file: Path | str) -> Launcher:
    """Launch ``train fit`` in a child interpreter of the locked environment."""
    workspace_path = str(Path(workspace_file).resolve())

    def launch(job: Job, timeout: float | None, log: Path) -> ChildResult:
        argv = [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "train",
            "fit",
            "--workspace",
            workspace_path,
            *job.to_args(),
        ]
        return run_child(argv, timeout, log)

    return launch


def find_fit(workspace: Workspace, fit_id: str) -> FitSpec:
    """The declared fit named ``fit_id``.

    Raises:
        KeyError: no declared fit has that id.
    """
    for fit in workspace.benchmark.fits():
        if fit.fit_id == fit_id:
            return fit
    for fit in workspace.evaluation_fits:
        if fit.fit_id == fit_id:
            return fit
    raise KeyError(f"No declared fit {fit_id!r}")


def _training_fit_spec(
    evaluation_fit_spec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Translate the final model's raw test view to its raw training view."""
    if (
        evaluation_fit_spec is not None
        and evaluation_fit_spec.get("training_view") == "raw-test"
    ):
        return evaluation_fit_spec | {"training_view": "raw-training"}
    return evaluation_fit_spec


def run_job(workspace: Workspace, job: Job) -> None:
    """The child body: verify the stack, run one engine, write the result file.

    Raises:
        StackError: CUDA is required and unavailable, or a library is missing.
        ValueError: the fit is chained but the job has no upstream checkpoint.
    """
    from jute_disease.training.engines.dl import fit_cnn, tune_cnn
    from jute_disease.training.engines.ml import (
        fit_classical,
        fit_cue,
        prepare_features,
    )

    fit = find_fit(workspace, job.fit_id)
    recipe = workspace.benchmark.recipe
    data = workspace.data
    evaluation_fit_spec = workspace.evaluation_fit_specs.get(job.fit_id)
    training_fit_spec = _training_fit_spec(evaluation_fit_spec)
    stack = verify_stack(
        require_cuda=workspace.require_cuda,
        precision=workspace.benchmark.recipe.precision,
        accelerator=workspace.accelerator,
    )
    write_json_atomic(job.attempt_dir / "runtime.json", stack)
    if fit.kind is FitKind.PREPARE:
        result = prepare_features(fit, recipe, data, workspace.cache_dir)
    elif fit.kind is FitKind.CLASSICAL:
        if evaluation_fit_spec is not None and evaluation_fit_spec.get("kind") == "cue":
            result = fit_cue(
                fit,
                recipe,
                data,
                job.attempt_dir,
                cue=evaluation_fit_spec["cue"],
            )
        else:
            store = RunStore(workspace.runs_dir)
            prepare = find_fit(workspace, "classical/prepare")
            if (
                fit.phase != "probe"
                and store.resolve(prepare.fit_id, make_identity(workspace, store))
                is not Resume.SKIP
            ):
                raise ValueError(
                    "Prepared feature cache has no valid compatible receipt"
                )
            selected_parameters = tuning_selection(workspace, store, fit)
            preprocessing = (
                evaluation_fit_spec.get("preprocessing", {})
                if evaluation_fit_spec is not None
                else {}
            )
            result = fit_classical(
                fit,
                recipe,
                data,
                job.attempt_dir,
                workspace.cache_dir,
                require_cache=(
                    data.policy_dir is None and fit.phase not in {"tune", "probe"}
                ),
                selected_parameters=selected_parameters,
                training_view=(
                    training_fit_spec.get("training_view", "raw-training")
                    if training_fit_spec is not None
                    else "raw-training"
                ),
                resize=preprocessing.get("resize", "letterbox"),
                border_frame_width_px=(
                    evaluation_fit_spec.get("border_frame_width_px", 0)
                    if evaluation_fit_spec is not None
                    else 0
                ),
                fixture_masks=data.fixture_masks,
            )
    else:
        bounded: dict[str, Any] = {}
        if job.bounds is not None:
            # A bounded run needs no chain; one given a checkpoint keeps it.
            if job.upstream_checkpoint is None:
                fit = replace(fit, upstream=None)
            recipe = replace(recipe, max_epochs=1)
            bounded = {
                "limit_train_batches": job.bounds[0],
                "limit_val_batches": job.bounds[1],
            }
        if fit.phase == "tune":
            result = tune_cnn(
                fit,
                recipe,
                data,
                job.attempt_dir,
                upstream_checkpoint=job.upstream_checkpoint,
                pretrained=workspace.pretrained,
                accelerator=workspace.accelerator,
                **bounded,
            )
        else:
            store = RunStore(workspace.runs_dir)
            selection = tuning_selection(workspace, store, fit)
            if selection is not None:
                recipe = replace(recipe, learning_rate=selection)
            result = fit_cnn(
                fit,
                recipe,
                data,
                job.attempt_dir,
                upstream_checkpoint=job.upstream_checkpoint,
                pretrained=workspace.pretrained,
                accelerator=workspace.accelerator,
                checkpoint_dir=job.checkpoint_dir,
                resume_from=job.resume_from,
                evaluation_fit_spec=training_fit_spec,
                border_frame_width_px=(
                    evaluation_fit_spec.get("border_frame_width_px", 0)
                    if evaluation_fit_spec is not None
                    else 0
                ),
                **bounded,
            )
    result_values = dict(result.results)
    if evaluation_fit_spec is not None:
        result_values["evaluation_fit_spec"] = evaluation_fit_spec
    write_json_atomic(
        job.result_path,
        {
            "results": {**result_values, "stack": stack},
            "artifacts": {name: str(path) for name, path in result.artifacts.items()},
        },
    )


def tuning_selection(workspace: Workspace, store: RunStore, fit: FitSpec) -> Any:
    """The frozen selection for ``fit``: parameters, a learning rate, or ``None``.

    Raises:
        ValueError: the benchmark has receipted policies and the tuning receipt is
            missing, or a receipt holds an invalid selection.
    """
    evaluation_fit_spec = workspace.evaluation_fit_specs.get(fit.fit_id)
    if evaluation_fit_spec is not None:
        if fit.kind is FitKind.CLASSICAL:
            selected = evaluation_fit_spec.get("selected_parameters")
            if not isinstance(selected, dict):
                raise ValueError(f"{fit.fit_id} has no frozen validation parameters")
            return selected
        if fit.kind in CNN_KINDS:
            rate = evaluation_fit_spec.get("selected_learning_rate")
            if (
                isinstance(rate, bool)
                or not isinstance(rate, int | float)
                or not 0 < rate < float("inf")
            ):
                raise ValueError(f"{fit.fit_id} has no frozen learning rate")
            return float(rate)

    tune_id = workspace.benchmark.tuning_dependency(fit)
    if tune_id is None:
        return None
    receipt = store.receipt(tune_id, missing_ok=True)
    if receipt is None:
        if workspace.data.policy_dir is not None:
            raise ValueError(f"{fit.fit_id} has no frozen tuning receipt: {tune_id}")
        return None
    results = receipt.get("results", {})
    if fit.kind is FitKind.CLASSICAL:
        selected = results.get("selected_parameters")
        if not isinstance(selected, dict):
            raise ValueError(f"Classical tuning receipt is invalid: {tune_id}")
        return selected
    rate = results.get("selected_learning_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not 0 < rate < float("inf")
    ):
        raise ValueError(f"CNN tuning receipt is invalid: {tune_id}")
    return float(rate)


def _tuning_marker(workspace: Workspace, store: RunStore, fit: FitSpec) -> Any:
    """The selection as identity input; ``unavailable`` while it is not frozen."""
    try:
        return tuning_selection(workspace, store, fit)
    except ValueError:
        return "unavailable"


def _manifest_digests(
    workspace: Workspace, fit: FitSpec, verifier: Any
) -> dict[str, str]:
    from jute_disease.data import (
        EVALUATOR_TEST_SOURCE_DIGEST,
        load_manifest,
        load_source_manifest,
    )

    target = (
        target_manifest_for_fit(fit, workspace.benchmark.recipe, workspace.data)
        if fit.kind in (FitKind.TARGET, FitKind.CLASSICAL)
        else load_manifest(workspace.data.target_dir)
    )
    digests = {
        "target": dict(target.dependencies).get(
            EVALUATOR_TEST_SOURCE_DIGEST, target.digest()
        ),
        "fit": target.digest(),
        "target_content": verifier.verify(
            target,
            workspace.data.target_root(fit.pool)
            if fit.pool
            else workspace.data.image_root,
            ("train", "val"),
        ),
    }
    if fit.kind is FitKind.SOURCE:
        manifest_dir, image_root = workspace.data.sources[fit.stage]
        source = load_source_manifest(
            manifest_dir,
            image_root,
            workspace.data.target_dir,
            content_verifier=verifier,
        )
        digests[fit.stage] = source.digest()
    return digests


def _upstream_checkpoint(store: RunStore, fit: FitSpec) -> tuple[Path, str] | None:
    if fit.upstream is None:
        return None
    receipt = store.receipt(fit.upstream, missing_ok=True)
    if receipt is None or "checkpoint" not in receipt["artifacts"]:
        raise UpstreamError(f"upstream {fit.upstream} has no verified checkpoint")
    entry = receipt["artifacts"]["checkpoint"]
    return store.run_dir(fit.upstream) / entry["path"], entry["sha256"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    """Capture the checkout identity without exposing dirty path names."""
    from jute_disease.config import ROOT_DIR

    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("training needs a readable Git commit") from error
    try:
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT_DIR),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError, subprocess.CalledProcessError:
        dirty = True
    return {
        "commit": commit,
        "working_tree_clean": not dirty,
        "official": not dirty,
    }


def load_assignment_table(workspace: Workspace) -> dict[str, Any]:
    """Read and validate the frozen seed/split to source-checkpoint table."""
    directories = (
        *((workspace.data.policy_dir.resolve(),) if workspace.data.policy_dir else ()),
        workspace.data.target_dir.resolve(),
        *workspace.data.target_dir.resolve().parents,
    )
    for directory in directories:
        path = directory / "assignments.json"
        if not path.is_file():
            continue
        raw_bytes = path.read_bytes()
        rows = json.loads(raw_bytes)
        if not isinstance(rows, list) or not rows:
            raise ValueError("assignments.json must contain a non-empty row list")
        seen: set[tuple[int, int]] = set()
        for row in rows:
            if not isinstance(row, dict) or not all(
                type(row.get(key)) is int for key in ("split", "seed", "checkpoint")
            ):
                raise ValueError(
                    "assignment rows need integer split, seed and checkpoint"
                )
            split, seed, checkpoint = row["split"], row["seed"], row["checkpoint"]
            if (split, seed) in seen:
                raise ValueError("assignment table has duplicate split/seed rows")
            if checkpoint != (seed + split) % 3:
                raise ValueError("assignment table violates rotating checkpoint rule")
            seen.add((split, seed))
        return {
            "present": True,
            "path": "assignments.json",
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "content_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "assignments": rows,
            "validated": True,
        }
    return {
        "present": False,
        "path": None,
        "sha256": None,
        "content_sha256": None,
        "assignments": [],
        "validated": False,
    }


def _source_checkpoint_ancestry(store: RunStore, fit: FitSpec) -> list[dict[str, str]]:
    """List verified source checkpoint bytes from oldest ancestor to parent."""
    chain: list[dict[str, str]] = []
    upstream = fit.upstream
    seen: set[str] = set()
    while upstream is not None:
        if upstream in seen:
            raise UpstreamError(f"cycle in source ancestry at {upstream}")
        seen.add(upstream)
        receipt = store.receipt(upstream, missing_ok=True)
        if receipt is None or "checkpoint" not in receipt.get("artifacts", {}):
            chain.append(
                {
                    "fit_id": upstream,
                    "identity": "unavailable",
                    "checkpoint_sha256": "unavailable",
                }
            )
            break
        chain.append(
            {
                "fit_id": upstream,
                "identity": str(receipt["identity"]),
                "checkpoint_sha256": str(receipt["artifacts"]["checkpoint"]["sha256"]),
            }
        )
        source_fit = receipt.get("results", {}).get("fit", {})
        upstream = source_fit.get("upstream") if isinstance(source_fit, dict) else None
    return list(reversed(chain))


def _receipt_context(
    workspace: Workspace, store: RunStore, fit: FitSpec
) -> dict[str, Any]:
    from jute_disease.config import ROOT_DIR
    from jute_disease.data.transforms import PreprocessingSpec

    config_path = workspace.benchmark_path.resolve()
    lock_path = ROOT_DIR / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("the locked environment file uv.lock is missing")
    provenance = _git_provenance()
    assignments = load_assignment_table(workspace)
    evaluation_fit_spec = workspace.evaluation_fit_specs.get(fit.fit_id)
    selected_preprocessing = (
        evaluation_fit_spec.get("preprocessing", {})
        if evaluation_fit_spec is not None
        else {}
    )
    preprocessing = PreprocessingSpec(
        size=workspace.benchmark.recipe.image_size,
        resize=selected_preprocessing.get("resize", "letterbox"),
    )
    preprocessing_payload = asdict(preprocessing)
    ancestry = _source_checkpoint_ancestry(store, fit)
    context = {
        "provenance": provenance,
        "official": provenance["official"],
        "configuration": {
            "path": config_path.name,
            "sha256": _sha256_file(config_path),
        },
        "environment_lock": {"path": "uv.lock", "sha256": _sha256_file(lock_path)},
        "preprocessing": {
            "spec": preprocessing_payload,
            "sha256": preprocessing.digest(),
        },
        "assignment_table": assignments,
        "source_checkpoint_ancestry": ancestry,
        "seed": fit.seed,
        "runtime_contract": {
            "precision": workspace.benchmark.recipe.precision,
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
            "torch_compile": False,
            "num_workers": workspace.benchmark.recipe.num_workers,
        },
        "run_options": {
            "pretrained": workspace.pretrained,
            "accelerator": workspace.accelerator,
            "require_cuda": workspace.require_cuda,
            "mask_checkpoint_sha256": _mask_checkpoint_digest(
                workspace.data.mask_checkpoint
            ),
            "fixture_masks": workspace.data.fixture_masks,
        },
    }
    if fit.kind is FitKind.PREPARE:
        context["run_options"]["cache_dir"] = str(workspace.cache_dir.resolve())
    return context


def _mask_checkpoint_digest(checkpoint: Path | None) -> str | None:
    """Hash local mask model files without loading the optional model stack."""
    if checkpoint is None:
        return None
    root = checkpoint.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"mask checkpoint directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("mask checkpoint directory contains no files")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _identity_inputs(
    workspace: Workspace,
    store: RunStore,
    fit: FitSpec,
    context: dict[str, Any],
    verifier: Any,
) -> dict[str, Any]:
    """Build the exact identity payload stored with the attempt provenance."""
    manifests = _manifest_digests(workspace, fit, verifier)
    try:
        upstream = _upstream_checkpoint(store, fit)
        upstream_sha256 = None if upstream is None else upstream[1]
    except UpstreamError:
        upstream_sha256 = "unavailable"
    return fit_identity_payload(
        fit,
        workspace.benchmark.recipe,
        manifests,
        upstream_sha256,
        {
            **context["run_options"],
            "tuning": _tuning_marker(workspace, store, fit),
        },
        config_sha256=context["configuration"]["sha256"],
        code_commit=context["provenance"]["commit"],
        environment_lock_sha256=context["environment_lock"]["sha256"],
        preprocessing_sha256=context["preprocessing"]["sha256"],
        assignment_table_sha256=context["assignment_table"]["sha256"],
        source_checkpoint_ancestry=context["source_checkpoint_ancestry"],
        evaluation_fit_spec=workspace.evaluation_fit_specs.get(fit.fit_id),
        evaluation_selection_sha256=(
            workspace.evaluation_selection_sha256
            if fit.fit_id in workspace.evaluation_fit_specs
            else None
        ),
    )


def _classify_child_failure(returncode: int, tail: str) -> FailureKind:
    lowered = tail.lower()
    if any(
        marker in lowered
        for marker in (
            "nan loss",
            "inf loss",
            "infinite loss",
            "non-finite loss",
            "nonfinite loss",
            "divergence detected",
            "training diverged",
        )
    ):
        return FailureKind.DIVERGENCE
    if returncode < 0 or any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "no space left on device",
            "input/output error",
            "i/o error",
            "memoryerror",
            "killed process",
            "process was killed",
        )
    ):
        return FailureKind.INFRASTRUCTURE
    return FailureKind.OTHER


def make_identity(workspace: Workspace, store: RunStore) -> IdentityFor:
    """The compatibility identity of a fit, covering its current upstream bytes.

    An upstream with no verified checkpoint yields the marker ``unavailable``, so
    a fit never matches a receipt made from a checkpoint that cannot be checked.
    """
    from jute_disease.data import ManifestContentVerifier

    verifier = ManifestContentVerifier()

    def identity(fit: FitSpec) -> str:
        context = _receipt_context(workspace, store, fit)
        inputs = _identity_inputs(workspace, store, fit, context, verifier)
        context["identity_inputs"] = inputs
        return fit_identity_digest(inputs)

    return identity


def make_executor(
    workspace: Workspace, store: RunStore, identity: IdentityFor, launcher: Launcher
) -> Executor:
    """Run one fit as a child, bounded by ``timeout``, and receipt the result."""

    def execute(fit: FitSpec, timeout: float) -> Outcome:
        from jute_disease.data import ManifestContentVerifier

        evaluation_fit_spec = workspace.evaluation_fit_specs.get(fit.fit_id)
        requires_feature_cache = (
            fit.kind is FitKind.CLASSICAL
            and fit.phase != "probe"
            and not (
                evaluation_fit_spec is not None
                and evaluation_fit_spec.get("kind") == "cue"
            )
        )
        if requires_feature_cache:
            prepare = find_fit(workspace, "classical/prepare")
            if store.resolve(prepare.fit_id, identity(prepare)) is not Resume.SKIP:
                return Outcome(
                    RunState.FAILED,
                    0.0,
                    "Prepared feature cache has no valid compatible receipt",
                )
        try:
            upstream = _upstream_checkpoint(store, fit)
        except UpstreamError as error:
            return Outcome(RunState.FAILED, 0.0, str(error))
        context = _receipt_context(workspace, store, fit)
        identity_inputs = _identity_inputs(
            workspace, store, fit, context, ManifestContentVerifier()
        )
        fit_identity_value = fit_identity_digest(identity_inputs)
        if fit_identity_value != identity(fit):
            return Outcome(
                RunState.FAILED,
                0.0,
                "fit identity inputs changed while preparing the run receipt",
            )
        context["identity_inputs"] = identity_inputs
        run = store.start(fit.fit_id, fit_identity_value, metadata=context)

        def finish_failure(
            state: RunState, reason: str, failure_kind: FailureKind | None = None
        ) -> None:
            metadata: dict[str, Any] = dict(context)
            runtime = read_json(run.attempt_dir / "runtime.json")
            if runtime is not None:
                metadata["runtime"] = runtime
            if failure_kind is not None:
                metadata["failure_kind"] = failure_kind.value
            run.fail(state, reason, metadata=metadata)

        checkpoint_dir = None
        plan = ResumePlan()
        if fit.kind in CNN_KINDS:
            checkpoint_dir = store.run_dir(fit.fit_id) / CHECKPOINT_DIR
            plan = prepare_checkpoints(
                checkpoint_dir, run.identity, run.attempt_dir.name
            )
            write_json_atomic(
                run.attempt_dir / RESUME_FILE,
                {
                    "resumed_from": plan.resumed_from,
                    "refusal": plan.refusal,
                },
            )
        job = Job(
            fit.fit_id,
            run.attempt_dir,
            run.attempt_dir / RESULT_FILE,
            None if upstream is None else upstream[0],
            checkpoint_dir=checkpoint_dir,
            resume_from=plan.checkpoint,
        )
        try:
            child = launcher(job, timeout, run.attempt_dir / LOG_FILE)
        except BaseException as error:
            state = (
                RunState.FAILED
                if isinstance(error, Exception)
                else (RunState.INTERRUPTED)
            )
            finish_failure(state, f"{type(error).__name__}: {error}")
            raise
        if child.timed_out:
            reason = f"exceeded the {timeout:.0f} s limit"
            finish_failure(RunState.TIMED_OUT, reason, FailureKind.INFRASTRUCTURE)
            return Outcome(
                RunState.TIMED_OUT,
                child.seconds,
                reason,
                FailureKind.INFRASTRUCTURE,
            )
        if child.returncode != 0:
            reason = f"child exited {child.returncode}: {child.tail}"
            failure_kind = _classify_child_failure(child.returncode, child.tail)
            finish_failure(RunState.FAILED, reason, failure_kind)
            return Outcome(RunState.FAILED, child.seconds, reason, failure_kind)
        try:
            payload = json.loads(job.result_path.read_text(encoding="utf-8"))
            artifacts = {
                name: Path(path) for name, path in payload["artifacts"].items()
            }
            results = payload["results"]
            if plan.refusal is not None:
                results["training_state"]["refused"] = plan.refusal
            runtime = results.get("stack")
            receipt_metadata = {**context, "runtime": runtime}
            run.complete(results, artifacts, metadata=receipt_metadata)
            if checkpoint_dir is not None:
                mark_finished(checkpoint_dir, run.identity, run.attempt_dir.name)
        except (OSError, ValueError, KeyError) as error:
            reason = f"child exited 0 without a usable result: {error}"
            failure_kind = (
                FailureKind.INFRASTRUCTURE
                if isinstance(error, OSError)
                else FailureKind.OTHER
            )
            finish_failure(RunState.FAILED, reason, failure_kind)
            return Outcome(RunState.FAILED, child.seconds, reason, failure_kind)
        return Outcome(RunState.COMPLETED, child.seconds)

    return execute

"""The declared benchmark: recipe, fits and run identities, read from one config."""

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any


class FitKind(StrEnum):
    PREPARE = "prepare"
    CLASSICAL = "classical"
    SOURCE = "source"
    TARGET = "target"


@dataclass(frozen=True)
class Recipe:
    """The frozen training recipe shared by every CNN stage."""

    image_size: int
    seeds: tuple[int, ...]
    source_seeds: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    scheduler_patience: int
    scheduler_factor: float
    precision: str
    compile_model: bool
    num_workers: int
    weighted_sampler: bool
    effective_batch: int


@dataclass(frozen=True)
class QueueLimits:
    """The run timeout, in seconds, of one attempt of each kind of step."""

    prepare_seconds: float
    classical_seconds: float
    source_seconds: float
    target_seconds: float

    def timeout(self, kind: FitKind) -> float:
        return {
            FitKind.PREPARE: self.prepare_seconds,
            FitKind.CLASSICAL: self.classical_seconds,
            FitKind.SOURCE: self.source_seconds,
            FitKind.TARGET: self.target_seconds,
        }[kind]


# Learning-rate candidates of one tuning fit, as multiples of the base rate.
LEARNING_RATE_MULTIPLIERS = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class FitSpec:
    """One executable fit. Every variant is a value here, not a separate script."""

    fit_id: str
    kind: FitKind
    seed: int
    system: str = ""
    stage: str = "jute"
    model: str = ""
    micro_batch: int = 0
    accumulation: int = 1
    estimator: str = ""
    features: str = ""
    upstream: str | None = None
    pool: str = ""
    view: int = 0
    policy: str = ""
    split: int = 0
    seed_index: int = 0
    phase: str = "final"


@dataclass(frozen=True)
class Benchmark:
    recipe: Recipe
    systems: dict[str, dict[str, Any]]
    estimators: tuple[str, ...]
    features: tuple[str, ...]
    classical_seed: int
    limits: QueueLimits
    pools: tuple[str, ...]
    policies: tuple[str, ...]
    views: tuple[int, ...]
    splits: tuple[int, ...]
    primary_split: int
    assignments: tuple[tuple[int, int, int], ...]

    def tuning_dependency(self, fit: FitSpec) -> str | None:
        """The tuning fit whose frozen selection a final ``fit`` reuses."""
        if fit.phase in {"tune", "probe"} or fit.kind is FitKind.PREPARE:
            return None
        if fit.kind is FitKind.CLASSICAL:
            return f"classical/tune/{fit.pool}/{fit.estimator}-{fit.features}"
        if fit.kind is FitKind.SOURCE:
            tuning = self.source_tuning_fit()
            return None if tuning is None else tuning.fit_id
        return f"cnn/tune/{fit.system}/pool-{fit.pool}"

    def timeout(self, fit: FitSpec) -> float:
        """The limit for one attempt; a CNN tuning fit trains every candidate."""
        seconds = self.limits.timeout(fit.kind)
        if fit.phase == "tune" and fit.kind in (FitKind.SOURCE, FitKind.TARGET):
            return seconds * len(LEARNING_RATE_MULTIPLIERS)
        return seconds

    def _checkpoint_for(self, split: int, seed_index: int) -> int:
        matches = [
            checkpoint
            for row_split, row_seed, checkpoint in self.assignments
            if (row_split, row_seed) == (split, seed_index)
        ]
        if len(matches) != 1:
            raise ValueError(
                "assignment table has no unique row for split "
                f"{split}, seed {seed_index}"
            )
        return matches[0]

    def fits(self) -> tuple[FitSpec, ...]:
        """Every declared fit, in declaration order."""
        simple_matrix = (
            len(self.pools) * len(self.policies) * len(self.views) * len(self.splits)
            == 1
        )
        out: list[FitSpec] = [FitSpec("classical/prepare", FitKind.PREPARE, 0)]
        for estimator in self.estimators:
            for features in self.features:
                for pool in self.pools:
                    out.append(
                        FitSpec(
                            f"classical/tune/{pool}/{estimator}-{features}",
                            FitKind.CLASSICAL,
                            self.classical_seed,
                            estimator=estimator,
                            features=features,
                            pool=pool,
                            phase="tune",
                        )
                    )
        for estimator in self.estimators:
            for features in self.features:
                for pool in self.pools:
                    for view in self.views:
                        for policy in self.policies:
                            for split in self.splits:
                                repetitions = (
                                    self.recipe.seeds
                                    if estimator == "rf"
                                    else (self.classical_seed,)
                                )
                                for seed_index, seed in enumerate(repetitions):
                                    fit_id = (
                                        f"classical/{estimator}-{features}/pool-{pool}"
                                        f"/view-{view}/{policy}/split-{split}/seed-{seed}"
                                    )
                                    if simple_matrix and len(repetitions) == 1:
                                        fit_id = f"classical/{estimator}-{features}"
                                    out.append(
                                        FitSpec(
                                            fit_id,
                                            FitKind.CLASSICAL,
                                            seed,
                                            estimator=estimator,
                                            features=features,
                                            pool=pool,
                                            view=view,
                                            policy=policy,
                                            split=split,
                                            seed_index=seed_index,
                                        )
                                    )
        source_tuning = self.source_tuning_fit()
        if source_tuning is not None:
            out.append(source_tuning)
        sources = self.source_fits()
        out.extend(sources.values())
        for system, definition in self.systems.items():
            chain = definition["chain"]
            if chain:
                checkpoint = self._checkpoint_for(self.primary_split, 0)
                source_seed = self.recipe.source_seeds[checkpoint]
                upstream = (
                    f"source/{chain[-1]}"
                    if len(self.recipe.source_seeds) == 1
                    else f"source/{chain[-1]}/seed-{source_seed}"
                )
            else:
                upstream = None
            for pool in self.pools:
                out.append(
                    FitSpec(
                        f"cnn/tune/{system}/pool-{pool}",
                        FitKind.TARGET,
                        self.recipe.seeds[0],
                        system=system,
                        model=definition["model"],
                        micro_batch=definition["micro_batch"],
                        accumulation=self.recipe.effective_batch
                        // definition["micro_batch"],
                        upstream=upstream,
                        pool=pool,
                        split=self.primary_split,
                        seed_index=0,
                        phase="tune",
                    )
                )
            split_values = (
                (self.primary_split,)
                if definition.get("analysis", "primary") == "secondary"
                else self.splits
            )
            for pool in self.pools:
                for view in self.views:
                    for policy in self.policies:
                        for split in split_values:
                            for seed_index, seed in enumerate(self.recipe.seeds):
                                checkpoint = self._checkpoint_for(split, seed_index)
                                source_seed = self.recipe.source_seeds[checkpoint]
                                upstream = (
                                    (
                                        f"source/{chain[-1]}"
                                        if len(self.recipe.source_seeds) == 1
                                        else f"source/{chain[-1]}/seed-{source_seed}"
                                    )
                                    if chain
                                    else None
                                )
                                fit_id = (
                                    f"cnn/{system}/pool-{pool}/view-{view}/{policy}"
                                    f"/split-{split}/seed-{seed}"
                                )
                                if simple_matrix:
                                    fit_id = f"cnn/{system}/seed-{seed}"
                                out.append(
                                    FitSpec(
                                        fit_id,
                                        FitKind.TARGET,
                                        seed,
                                        system=system,
                                        model=definition["model"],
                                        micro_batch=definition["micro_batch"],
                                        accumulation=self.recipe.effective_batch
                                        // definition["micro_batch"],
                                        upstream=upstream,
                                        pool=pool,
                                        view=view,
                                        policy=policy,
                                        split=split,
                                        seed_index=seed_index,
                                    )
                                )
        return tuple(out)

    def source_tuning_fit(self) -> FitSpec | None:
        """One source-validation learning-rate sweep shared by every source fit."""
        for definition in self.systems.values():
            chain = definition["chain"]
            if chain:
                stage = chain[0]
                return FitSpec(
                    f"source/tune/{stage}",
                    FitKind.SOURCE,
                    self.recipe.source_seeds[0],
                    stage=stage,
                    model=definition["model"],
                    micro_batch=definition["micro_batch"],
                    accumulation=self.recipe.effective_batch
                    // definition["micro_batch"],
                    seed_index=0,
                    phase="tune",
                )
        return None

    def source_fits(self) -> dict[str, FitSpec]:
        """The shared source stages, each trained for every declared source seed."""
        out: dict[str, FitSpec] = {}
        for definition in self.systems.values():
            for stage in definition["chain"]:
                for index, seed in enumerate(self.recipe.source_seeds):
                    fit_id = (
                        f"source/{stage}"
                        if len(self.recipe.source_seeds) == 1
                        else f"source/{stage}/seed-{seed}"
                    )
                    previous = None
                    if stage != definition["chain"][0]:
                        previous_stage = definition["chain"][
                            definition["chain"].index(stage) - 1
                        ]
                        previous = (
                            f"source/{previous_stage}"
                            if len(self.recipe.source_seeds) == 1
                            else f"source/{previous_stage}/seed-{seed}"
                        )
                    spec = FitSpec(
                        fit_id,
                        FitKind.SOURCE,
                        seed,
                        stage=stage,
                        model=definition["model"],
                        micro_batch=definition["micro_batch"],
                        accumulation=self.recipe.effective_batch
                        // definition["micro_batch"],
                        upstream=previous,
                        seed_index=index,
                    )
                    if out.setdefault(fit_id, spec) != spec:
                        raise ValueError(f"Conflicting definitions of {fit_id}")
        return out


def _checked(section: str, raw: dict[str, Any], cls: type) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - names)
    missing = sorted(names - set(raw))
    if unknown or missing:
        raise ValueError(
            f"Benchmark {section} has unknown keys {unknown} and missing keys {missing}"
        )
    return dict(raw)


def load_benchmark(path: Path | str) -> Benchmark:
    """Read and validate the benchmark configuration.

    Raises:
        ValueError: a section has unknown or missing keys, or a system is invalid.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    recipe_raw = _checked("recipe", raw["recipe"], Recipe)
    recipe_raw["seeds"] = tuple(recipe_raw["seeds"])
    recipe_raw["source_seeds"] = tuple(recipe_raw["source_seeds"])
    recipe = Recipe(**recipe_raw)
    if not recipe.source_seeds:
        raise ValueError("Benchmark must declare at least one source seed per stage")
    if recipe.precision != "bf16-mixed":
        raise ValueError("Benchmark precision must be bf16-mixed on CUDA")
    if recipe.compile_model:
        raise ValueError("Benchmark recipe must keep compile_model off on this stack")
    limits = QueueLimits(**_checked("queue", raw["queue"], QueueLimits))
    for name, seconds in asdict(limits).items():
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise ValueError(f"Benchmark queue {name} must be a number")
        if not 0 < seconds < float("inf"):
            raise ValueError(f"Benchmark queue {name} must be a finite positive number")
    systems = raw["cnn_systems"]
    for name, definition in systems.items():
        if set(definition) not in (
            {"model", "micro_batch", "chain"},
            {"model", "micro_batch", "chain", "analysis"},
        ):
            raise ValueError(
                f"System {name} must define model, micro_batch, chain, "
                "and optional analysis"
            )
        if recipe.effective_batch % definition["micro_batch"]:
            raise ValueError(f"System {name}: micro_batch must divide effective_batch")
    classical = raw["classical"]
    if set(classical) != {"estimators", "features", "seed"}:
        raise ValueError("Benchmark classical must define estimators, features, seed")
    matrix = raw["matrix"]
    if set(matrix) != {"pools", "policies", "views", "splits", "primary_split"}:
        raise ValueError("Benchmark matrix has unknown or missing dimensions")
    matrix_size = (
        len(matrix["pools"])
        * len(matrix["policies"])
        * len(matrix["views"])
        * len(matrix["splits"])
    )
    if matrix_size > 1 and len(recipe.source_seeds) != 3:
        raise ValueError("The full benchmark must declare three source seeds per stage")
    assignments = tuple(
        (row["split"], row["seed"], row["checkpoint"]) for row in raw["assignments"]
    )
    if len(assignments) != len({(split, seed) for split, seed, _ in assignments}):
        raise ValueError("Benchmark assignment table has duplicate split/seed rows")
    if any(
        checkpoint != (seed + split) % 3 or checkpoint >= len(recipe.source_seeds)
        for split, seed, checkpoint in assignments
    ):
        raise ValueError("Benchmark assignment table violates its rotation rule")
    return Benchmark(
        recipe=recipe,
        systems=systems,
        estimators=tuple(classical["estimators"]),
        features=tuple(classical["features"]),
        classical_seed=classical["seed"],
        limits=limits,
        pools=tuple(matrix["pools"]),
        policies=tuple(matrix["policies"]),
        views=tuple(matrix["views"]),
        splits=tuple(matrix["splits"]),
        primary_split=matrix["primary_split"],
        assignments=assignments,
    )


def fit_identity(
    fit: FitSpec,
    recipe: Recipe,
    manifests: dict[str, str],
    upstream_sha256: str | None,
    environment: dict[str, Any] | None = None,
    *,
    config_sha256: str | None = None,
    code_commit: str | None = None,
    environment_lock_sha256: str | None = None,
    preprocessing_sha256: str | None = None,
    assignment_table_sha256: str | None = None,
    source_checkpoint_ancestry: list[dict[str, str]] | None = None,
    evaluation_fit_spec: dict[str, Any] | None = None,
    evaluation_selection_sha256: str | None = None,
) -> str:
    """Digest of everything that makes two runs of ``fit`` interchangeable.

    Covers the fit, recipe, data manifests, source checkpoint ancestry,
    preprocessing, assignment table, config, code commit and locked environment.
    Dynamic backend and package-version observations belong in receipts, not in
    this compatibility identity.
    """
    payload = fit_identity_payload(
        fit,
        recipe,
        manifests,
        upstream_sha256,
        environment,
        config_sha256=config_sha256,
        code_commit=code_commit,
        environment_lock_sha256=environment_lock_sha256,
        preprocessing_sha256=preprocessing_sha256,
        assignment_table_sha256=assignment_table_sha256,
        source_checkpoint_ancestry=source_checkpoint_ancestry,
        evaluation_fit_spec=evaluation_fit_spec,
        evaluation_selection_sha256=evaluation_selection_sha256,
    )
    return fit_identity_digest(payload)


def fit_identity_payload(
    fit: FitSpec,
    recipe: Recipe,
    manifests: dict[str, str],
    upstream_sha256: str | None,
    environment: dict[str, Any] | None = None,
    *,
    config_sha256: str | None = None,
    code_commit: str | None = None,
    environment_lock_sha256: str | None = None,
    preprocessing_sha256: str | None = None,
    assignment_table_sha256: str | None = None,
    source_checkpoint_ancestry: list[dict[str, str]] | None = None,
    evaluation_fit_spec: dict[str, Any] | None = None,
    evaluation_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the complete identity inputs for one fit receipt.

    ``evaluation_fit_spec`` binds a fit to an externally frozen evaluator route,
    including its validation-selected hyperparameters and input view.
    """
    payload = {
        "fit": asdict(fit),
        "recipe": asdict(recipe),
        "manifests": manifests,
        "upstream_sha256": upstream_sha256,
        "environment": environment or {},
        "config_sha256": config_sha256,
        "code_commit": code_commit,
        "environment_lock_sha256": environment_lock_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "assignment_table_sha256": assignment_table_sha256,
        "source_checkpoint_ancestry": source_checkpoint_ancestry or [],
    }
    if evaluation_fit_spec is not None:
        payload["evaluation_fit_spec"] = evaluation_fit_spec
        payload["evaluation_selection_sha256"] = evaluation_selection_sha256
    return payload


def fit_identity_digest(identity_inputs: dict[str, Any]) -> str:
    """Hash the exact structured inputs recorded by a training receipt."""
    text = json.dumps(identity_inputs, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()

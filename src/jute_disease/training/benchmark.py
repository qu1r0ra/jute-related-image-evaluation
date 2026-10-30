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
    source_seed: int
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
    classical_fit_seconds: float
    classical_aggregate_seconds: float
    shared_preparation_seconds: float
    reserve_hours: float
    forecast_margin: float


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


@dataclass(frozen=True)
class Benchmark:
    recipe: Recipe
    systems: dict[str, dict[str, Any]]
    estimators: tuple[str, ...]
    features: tuple[str, ...]
    classical_seed: int
    limits: QueueLimits

    def fits(self) -> tuple[FitSpec, ...]:
        """Every declared fit, in declaration order."""
        out: list[FitSpec] = [FitSpec("classical/prepare", FitKind.PREPARE, 0)]
        for estimator in self.estimators:
            for features in self.features:
                out.append(
                    FitSpec(
                        f"classical/{estimator}-{features}",
                        FitKind.CLASSICAL,
                        self.classical_seed,
                        estimator=estimator,
                        features=features,
                    )
                )
        sources = self.source_fits()
        out.extend(sources.values())
        for system, definition in self.systems.items():
            chain = definition["chain"]
            upstream = f"source/{chain[-1]}" if chain else None
            for seed in self.recipe.seeds:
                out.append(
                    FitSpec(
                        f"cnn/{system}/seed-{seed}",
                        FitKind.TARGET,
                        seed,
                        system=system,
                        model=definition["model"],
                        micro_batch=definition["micro_batch"],
                        accumulation=self.recipe.effective_batch
                        // definition["micro_batch"],
                        upstream=upstream,
                    )
                )
        return tuple(out)

    def source_fits(self) -> dict[str, FitSpec]:
        """The shared source stages, each trained once at the source seed."""
        out: dict[str, FitSpec] = {}
        for definition in self.systems.values():
            previous: str | None = None
            for stage in definition["chain"]:
                fit_id = f"source/{stage}"
                spec = FitSpec(
                    fit_id,
                    FitKind.SOURCE,
                    self.recipe.source_seed,
                    stage=stage,
                    model=definition["model"],
                    micro_batch=definition["micro_batch"],
                    accumulation=self.recipe.effective_batch
                    // definition["micro_batch"],
                    upstream=previous,
                )
                if out.setdefault(fit_id, spec) != spec:
                    raise ValueError(f"Conflicting definitions of {fit_id}")
                previous = fit_id
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
    recipe = Recipe(**recipe_raw)
    limits = QueueLimits(**_checked("queue", raw["queue"], QueueLimits))
    systems = raw["cnn_systems"]
    for name, definition in systems.items():
        if set(definition) != {"model", "micro_batch", "chain"}:
            raise ValueError(f"System {name} must define model, micro_batch, chain")
        if recipe.effective_batch % definition["micro_batch"]:
            raise ValueError(f"System {name}: micro_batch must divide effective_batch")
    classical = raw["classical"]
    if set(classical) != {"estimators", "features", "seed"}:
        raise ValueError("Benchmark classical must define estimators, features, seed")
    return Benchmark(
        recipe=recipe,
        systems=systems,
        estimators=tuple(classical["estimators"]),
        features=tuple(classical["features"]),
        classical_seed=classical["seed"],
        limits=limits,
    )


def fit_identity(
    fit: FitSpec,
    recipe: Recipe,
    manifests: dict[str, str],
    upstream_sha256: str | None,
    environment: dict[str, Any] | None = None,
) -> str:
    """Digest of everything that makes two runs of ``fit`` interchangeable.

    Covers the fit, the recipe, the manifests it reads, the exact bytes of the
    source checkpoint it starts from, and run choices outside the recipe such as
    the initialization, so a changed manifest, config or upstream checkpoint
    never matches an earlier receipt.
    """
    payload = {
        "fit": asdict(fit),
        "recipe": asdict(recipe),
        "manifests": manifests,
        "upstream_sha256": upstream_sha256,
        "environment": environment or {},
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()

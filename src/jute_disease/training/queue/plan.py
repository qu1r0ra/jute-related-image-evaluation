"""The serial stage order of the benchmark."""

from dataclasses import dataclass

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec


@dataclass(frozen=True)
class Stage:
    """A group of fits admitted together.

    Mandatory stages always run; optional stages are admitted against the
    deadline in order. A prerequisite stage can feed multiple optional arms.
    """

    name: str
    mandatory: bool
    fits: tuple[FitSpec, ...]


def build_plan(benchmark: Benchmark) -> tuple[Stage, ...]:
    """Run shared source prerequisites, then PlantDoc before PlantVillage fallback."""
    fits = benchmark.fits()
    targets = [fit for fit in fits if fit.kind is FitKind.TARGET]
    direct = [fit for fit in targets if not benchmark.systems[fit.system]["chain"]]
    classical = [
        fit for fit in fits if fit.kind in (FitKind.PREPARE, FitKind.CLASSICAL)
    ]
    stages = [
        Stage("core", True, tuple(direct)),
        Stage("classical", True, tuple(classical)),
    ]
    sources = tuple(benchmark.source_fits().values())
    shared_ids = {source.upstream for source in sources if source.upstream}
    shared_sources = [source for source in sources if source.fit_id in shared_ids]

    for source in shared_sources:
        stages.append(Stage(f"{source.stage}_source", False, (source,)))

    for source in sources:
        if source in shared_sources:
            continue
        arm = [fit for fit in targets if fit.upstream == source.fit_id]
        stages.append(Stage(source.stage, False, (source, *arm)))

    for source in shared_sources:
        arm = tuple(fit for fit in targets if fit.upstream == source.fit_id)
        if arm:
            stages.append(Stage(source.stage, False, arm))
    return tuple(stages)

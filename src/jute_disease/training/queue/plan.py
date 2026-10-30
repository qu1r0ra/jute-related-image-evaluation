"""The serial stage order of the benchmark."""

from dataclasses import dataclass

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec


@dataclass(frozen=True)
class Stage:
    """A group of fits admitted together.

    Mandatory stages always run; optional stages are admitted against the
    deadline, in order, and an optional arm only counts when all its seeds finish.
    """

    name: str
    mandatory: bool
    fits: tuple[FitSpec, ...]


def build_plan(benchmark: Benchmark) -> tuple[Stage, ...]:
    """Order: core CNN fits, classical fits, then each source chain and its arm."""
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
    for source_id, source in benchmark.source_fits().items():
        arm = [fit for fit in targets if fit.upstream == source_id]
        stages.append(Stage(source.stage, False, (source, *arm)))
    return tuple(stages)

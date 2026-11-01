"""The serial stage order of the benchmark."""

from dataclasses import dataclass

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec


@dataclass(frozen=True)
class Stage:
    """A group of fits run together, in order.

    A stage that starts with a source fit is an arm: its targets build on that
    source checkpoint and are omitted if it does not complete.
    """

    name: str
    fits: tuple[FitSpec, ...]


def build_plan(benchmark: Benchmark) -> tuple[Stage, ...]:
    """Order: core CNN fits, classical fits, then each source chain and its arm."""
    fits = benchmark.fits()
    targets = [fit for fit in fits if fit.kind is FitKind.TARGET]
    direct = [fit for fit in targets if not benchmark.systems[fit.system]["chain"]]
    classical = [
        fit for fit in fits if fit.kind in (FitKind.PREPARE, FitKind.CLASSICAL)
    ]
    stages = [Stage("core", tuple(direct)), Stage("classical", tuple(classical))]
    for source_id, source in benchmark.source_fits().items():
        arm = [fit for fit in targets if fit.upstream == source_id]
        stages.append(Stage(source.stage, (source, *arm)))
    return tuple(stages)

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


def build_plan(
    benchmark: Benchmark, evaluation_fits: tuple[FitSpec, ...] = ()
) -> tuple[Stage, ...]:
    """Order: core CNN fits, classical fits, then each source chain and its arm."""
    fits = benchmark.fits()
    targets = [fit for fit in fits if fit.kind is FitKind.TARGET]
    direct = [fit for fit in targets if not benchmark.systems[fit.system]["chain"]]
    classical = [
        fit for fit in fits if fit.kind in (FitKind.PREPARE, FitKind.CLASSICAL)
    ]
    stages = [Stage("core", tuple(direct)), Stage("classical", tuple(classical))]
    sources = benchmark.source_fits()
    for stage_name in dict.fromkeys(fit.stage for fit in sources.values()):
        source_tuning = benchmark.source_tuning_fit()
        tune = (
            (source_tuning,)
            if source_tuning is not None and source_tuning.stage == stage_name
            else ()
        )
        source_fits = [fit for fit in sources.values() if fit.stage == stage_name]
        source_ids = {fit.fit_id for fit in source_fits}
        arm = [fit for fit in targets if fit.upstream in source_ids]
        stages.append(Stage(stage_name, (*tune, *source_fits, *arm)))
    if evaluation_fits:
        known = {fit.fit_id for stage in stages for fit in stage.fits}
        additional = tuple(fit for fit in evaluation_fits if fit.fit_id not in known)
        if additional:
            stages.append(Stage("evaluation-probes", additional))
    return tuple(stages)

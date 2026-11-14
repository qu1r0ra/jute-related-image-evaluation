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
    return tuple(stages)


def build_probe_plan(benchmark: Benchmark) -> tuple[Stage, ...]:
    """The border and pad-cue probe fits, after the fits a border probe reuses.

    A border probe takes its tuning selection from its base fit, and the queue
    omits a fit whose tuning did not complete in the same run. The stage so
    starts with each base fit's tuning fit and the prepare fit that tuning
    needs; a completed one is skipped, not retrained.
    """
    declared = {fit.fit_id: fit for fit in benchmark.fits()}
    borders = benchmark.border_fits()
    closure: dict[str, FitSpec] = {}
    for border in borders:
        tuning_id = benchmark.tuning_dependency(border)
        tuning = None if tuning_id is None else declared[tuning_id]
        if tuning is None:
            continue
        prepare_id = benchmark.prepare_dependency(tuning)
        if prepare_id is not None:
            closure[prepare_id] = declared[prepare_id]
        closure[tuning.fit_id] = tuning
    return (Stage("probes", (*closure.values(), *benchmark.pad_cue_fits(), *borders)),)

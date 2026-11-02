from importlib import import_module
from typing import Any

from jute_disease.evaluation.metrics import EVAL_METRICS, TRAIN_METRICS, format_metrics
from jute_disease.evaluation.resampling import (
    CONFIDENCE_LEVEL,
    EVAL_SEED,
    RESAMPLES,
    Interval,
    Resampling,
    resample_metrics,
)
from jute_disease.evaluation.scores import (
    ClassScore,
    Evaluation,
    PredictionSet,
    RunScores,
    SeedSummary,
    evaluate_predictions,
    one_image_per_family,
    score_run,
    summarize_seeds,
)
from jute_disease.evaluation.statistics import (
    Agreement,
    agreement_gate,
    kendall_tau,
    wilson_interval,
)

_LAZY_EXPORTS = {
    "class_scene_cells": (".breakdowns", "class_scene_cells"),
    "scene_summaries": (".breakdowns", "scene_summaries"),
    "stratified_sample": (".breakdowns", "stratified_sample"),
    "evaluate_test": (".evaluator", "evaluate_test"),
    "leaf_sample_ids": (".evaluator", "leaf_sample_ids"),
    "FreezeRecordError": (".freeze_record", "FreezeRecordError"),
    "TestReadRefusedError": (".freeze_record", "TestReadRefusedError"),
    "authorize_test_read": (".freeze_record", "authorize_test_read"),
    "default_reads_log_path": (".freeze_record", "default_reads_log_path"),
    "log_test_read": (".freeze_record", "log_test_read"),
    "verify_freeze_record": (".freeze_record", "verify_freeze_record"),
    "write_freeze_record": (".freeze_record", "write_freeze_record"),
    "write_run_record": (".freeze_record", "write_run_record"),
    "background_only_image": (".probes", "background_only_image"),
    "border_probe": (".probes", "border_probe"),
    "frame_leaf_share": (".probes", "frame_leaf_share"),
    "frame_mask": (".probes", "frame_mask"),
    "frame_only_image": (".probes", "frame_only_image"),
    "pad_cue_fits": (".probes", "pad_cue_fits"),
    "planned_probe_fit_ids": (".probes", "planned_probe_fit_ids"),
    "probe_fits": (".probes", "probe_fits"),
}


def __getattr__(name: str) -> Any:
    """Load heavier evaluation exports only when a caller requests them."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CONFIDENCE_LEVEL",
    "EVAL_METRICS",
    "EVAL_SEED",
    "RESAMPLES",
    "TRAIN_METRICS",
    "Agreement",
    "ClassScore",
    "Evaluation",
    "Interval",
    "PredictionSet",
    "Resampling",
    "RunScores",
    "SeedSummary",
    "agreement_gate",
    "evaluate_predictions",
    "format_metrics",
    "kendall_tau",
    "one_image_per_family",
    "resample_metrics",
    "score_run",
    "summarize_seeds",
    "wilson_interval",
    "FreezeRecordError",
    "TestReadRefusedError",
    "authorize_test_read",
    "background_only_image",
    "border_probe",
    "class_scene_cells",
    "default_reads_log_path",
    "evaluate_test",
    "frame_leaf_share",
    "frame_mask",
    "frame_only_image",
    "leaf_sample_ids",
    "log_test_read",
    "pad_cue_fits",
    "planned_probe_fit_ids",
    "probe_fits",
    "scene_summaries",
    "stratified_sample",
    "verify_freeze_record",
    "write_freeze_record",
    "write_run_record",
]

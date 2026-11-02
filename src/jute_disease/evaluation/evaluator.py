"""The gated evaluator: the only code that opens the test partition.

Every entry point takes an :class:`Authorization` from the freeze-record gate, so
no test label or prediction is read without a verified selection and run record.
One call evaluates one frozen target manifest; looping over pools and policies
belongs to the caller.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jute_disease.data import load_manifest
from jute_disease.evaluation.breakdowns import (
    LEAF_SAMPLE_SIZE,
    class_scene_cells,
    scene_summaries,
    stratified_sample,
)
from jute_disease.evaluation.freeze_record import (
    Authorization,
    FreezeRecordError,
    RunEntry,
)
from jute_disease.evaluation.probes import (
    background_probe,
    border_probe,
    subset,
)
from jute_disease.evaluation.scores import PredictionSet, evaluate_predictions


def leaf_sample_ids(
    target_dir: Path | str,
    freeze_record: Mapping[str, Any],
    size: int = LEAF_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[str, ...]:
    """The stratified test sample on which leaf-only checkpoints are evaluated.

    ``freeze_record`` is a record returned by ``verify_freeze_record``; the sample
    must be known before leaf-only checkpoints are scored.
    """
    del freeze_record  # presence proves the gate was passed
    entries = load_manifest(Path(target_dir)).partition("test")
    return stratified_sample(
        [e.image_id for e in entries], [e.label for e in entries], size, seed
    )


def _load(entry: RunEntry, base_dir: Path, ids: tuple[str, ...]) -> tuple[str, ...]:
    raw = (base_dir / entry.predictions).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.predictions_sha256:
        raise FreezeRecordError(
            f"predictions changed since the run record: {entry.fit_id}"
        )
    payload = json.loads(raw)
    if tuple(payload.get("image_ids", ())) != ids:
        raise FreezeRecordError(
            f"predictions for {entry.fit_id} do not match test images"
        )
    labels = tuple(payload.get("predictions", ()))
    if len(labels) != len(ids):
        raise FreezeRecordError(f"predictions for {entry.fit_id} are incomplete")
    return labels


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_test(
    target_dir: Path | str,
    authorization: Authorization,
    base_dir: Path | str,
    *,
    scenes: Mapping[str, str] | None = None,
    frame_leaf_share: Mapping[str, float] | None = None,
    sample_size: int = LEAF_SAMPLE_SIZE,
    resamples: int = 10_000,
    permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score the authorized fits on the test partition and run every analysis."""
    manifest = load_manifest(Path(target_dir))
    entries = tuple(sorted(manifest.partition("test"), key=lambda e: e.image_id))
    if not entries:
        raise FreezeRecordError("target manifest has no test images")
    ids = tuple(e.image_id for e in entries)
    classes = tuple(manifest.labels)
    sample = set(stratified_sample(ids, [e.label for e in entries], sample_size, seed))
    sample_ids = tuple(i for i in ids if i in sample)
    base = Path(base_dir)

    runs: dict[tuple[str, str], dict[int, tuple[str, ...]]] = defaultdict(dict)
    for run in authorization.runs:
        expected = sample_ids if run.variant == "leaf_only" else ids
        group = runs[(run.system, run.variant)]
        if run.seed in group:
            raise FreezeRecordError(
                f"more than one run for {run.system} {run.variant} seed {run.seed}; "
                "restrict the call to one pool, view, policy and split with --fit-id"
            )
        group[run.seed] = _load(run, base, expected)

    def build(system: str, variant: str) -> PredictionSet:
        chosen = sample_ids if variant == "leaf_only" else ids
        by_id = {e.image_id: e for e in entries}
        return PredictionSet(
            classes,
            chosen,
            tuple(by_id[i].label for i in chosen),
            tuple(by_id[i].group for i in chosen),
            tuple(by_id[i].source for i in chosen),
            runs[(system, variant)],
        )

    report: dict[str, Any] = {
        "record_hash": authorization.record_hash,
        "post_hoc": authorization.post_hoc,
        "test_images": len(ids),
        "systems": {},
    }
    if authorization.post_hoc:
        report["note"] = "freeze record created after a test read; results are post-hoc"
    scenes = dict(scenes or {})
    full_systems = sorted(s for s, v in runs if v == "full")
    for system in full_systems:
        full = build(system, "full")
        result: dict[str, Any] = {"evaluation": _plain(evaluate_predictions(full))}
        if scenes:
            result["class_scene_cells"] = _plain(class_scene_cells(full, scenes))
            result["scene_summaries"] = _plain(scene_summaries(full, scenes))
        report["systems"][system] = result

    leaf_required = {run.fit_id for run in authorization.runs if run.variant == "full"}
    leaf_covered = {run.of for run in authorization.runs if run.variant == "leaf_only"}
    uncovered = sorted(leaf_required - leaf_covered)
    if leaf_covered and uncovered:
        raise FreezeRecordError(
            f"leaf-only evaluation misses final checkpoints: {uncovered}"
        )
    report["leaf_sample"] = {"size": len(sample_ids), "image_ids": list(sample_ids)}
    report["leaf_only_coverage"] = {
        "complete": bool(leaf_required) and not uncovered,
        "missing": uncovered,
    }
    for system in sorted(s for s, v in runs if v == "leaf_only"):
        leaf = build(system, "leaf_only")
        entry = report["systems"].setdefault(system, {})
        entry["leaf_only"] = _plain(evaluate_predictions(leaf))
        if (system, "full") in runs:
            entry["full_on_leaf_sample"] = _plain(
                evaluate_predictions(subset(build(system, "full"), sample_ids))
            )

    options = {"resamples": resamples, "permutations": permutations, "seed": seed}
    for system in sorted(s for s, v in runs if v == "background_only"):
        result = background_probe(build(system, "background_only"), **options)
        report["systems"].setdefault(system, {})["background_probe"] = _plain(result)

    pad_systems = sorted(s for s, v in runs if v == "pad_cue")
    pad = build(pad_systems[0], "pad_cue") if pad_systems else None
    for system in sorted(s for s, v in runs if v == "border"):
        if frame_leaf_share is None:
            raise FreezeRecordError("border probe needs per-image frame leaf shares")
        body = authorization.body
        border = border_probe(
            build(system, "border"),
            frame_leaf_share,
            scenes,
            mask_audit_passed=bool(body["mask_audit"]["passed"]),
            leaf_cutoff=float(body["border_probe"]["final"]["leaf_cutoff"]),
            width_px=int(body["border_probe"]["final"]["width_px"]),
            pad_cue=pad,
            **options,
        )
        report["systems"].setdefault(system, {})["border_probe"] = _plain(border)
    return report

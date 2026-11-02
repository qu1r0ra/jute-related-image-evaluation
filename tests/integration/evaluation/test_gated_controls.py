"""Gate controls and the full analysis run through the ``jute-disease`` CLI."""

import json
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import load_manifest
from jute_disease.training import load_benchmark

from .test_gated_evaluator import (
    FIT,
    _cli,
    _freeze,
    _outcomes,
    _predictions,
    _probe_ids,
    _spec,
    _write,
)


def test_freeze_record_needs_recorded_validation_outcomes(
    tmp_path: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _spec(tmp_path, [], _probe_ids(capsys))
    base = ["evaluate", "freeze-record", "--spec", spec, "--root", root]
    with pytest.raises(SystemExit, match="validation outcomes"):
        _cli(*base, "--output", tmp_path / "none.json")
    wrong = _write(tmp_path / "wrong.json", {"partition": "test"})
    with pytest.raises(SystemExit, match="validation outcomes"):
        _cli(*base, "--outcomes", wrong, "--output", tmp_path / "wrong-out.json")
    assert not (tmp_path / "none.json").exists()
    assert not (tmp_path / "wrong-out.json").exists()


def test_freeze_record_must_list_every_probe_fit(
    tmp_path: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probes = _probe_ids(capsys)
    spec = json.loads(_spec(tmp_path, [], probes).read_text(encoding="utf-8"))
    spec["fit_ids"].remove(probes[0])
    _write(tmp_path / "short.json", spec)
    with pytest.raises(SystemExit, match="missing from the record"):
        _cli(
            "evaluate",
            "freeze-record",
            "--spec",
            tmp_path / "short.json",
            "--outcomes",
            _outcomes(tmp_path),
            "--output",
            tmp_path / "short-out.json",
            "--root",
            root,
        )


def test_probe_plan_is_primary_split_classical_and_imagenet_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = load_benchmark(BENCHMARK_CONFIG)
    probes = _probe_ids(capsys)
    background = [p for p in probes if p.startswith("probe/background/")]
    border = [p for p in probes if p.startswith("probe/border/")]
    assert border and any(p.startswith("probe/pad-cue/") for p in probes)
    assert all("split-0" in p or "/tune/" in p for p in background + border)
    assert all(
        "classical" in p or "mobilenet_v2_direct" in p for p in background + border
    )

    def final_cnn_seeds(group: list[str]) -> set[str]:
        return {
            p.rsplit("seed-", 1)[1] for p in group if "/cnn/" in p and "/tune/" not in p
        }

    assert len(final_cnn_seeds(background)) == len(benchmark.recipe.seeds)
    assert len(final_cnn_seeds(border)) == 3
    for pool in benchmark.pools:
        for policy in benchmark.policies:
            assert any(f"pool-{pool}" in p and policy in p for p in background)


def _setup_runs(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, Path, Path]:
    record = _freeze(tmp_path, root, capsys, [FIT.format(0), "fixture/other"])
    _predictions(target, tmp_path / "runs" / "0.json", 7, None)
    _write(
        tmp_path / "runs" / "runs.json",
        [
            {
                "fit_id": FIT.format(0),
                "system": "mobilenet_v2_direct",
                "seed": 0,
                "predictions": "0.json",
            }
        ],
    )
    run_record = tmp_path / "runs" / "run-record.json"
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        record,
        "--runs",
        tmp_path / "runs" / "runs.json",
        "--output",
        run_record,
    )
    return record, run_record, tmp_path / "runs" / "0.json"


def _test_args(
    tmp_path: Path, target: Path, root: Path, record: Path, run_record: Path
) -> list[object]:
    return [
        "evaluate",
        "test",
        "--target-dir",
        target,
        "--root",
        root,
        "--freeze-record",
        record,
        "--run-record",
        run_record,
        "--output",
        tmp_path / "out.json",
        "--resamples",
        10,
        "--permutations",
        10,
    ]


def test_changed_pipeline_code_refuses_the_read(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, run_record, _ = _setup_runs(tmp_path, target, root, capsys)
    (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="pipeline code changed"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_edited_freeze_record_refuses_the_read(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, run_record, _ = _setup_runs(tmp_path, target, root, capsys)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["body"]["budget"] = {"a100_gpu_hours": 99}
    _write(record, payload)
    with pytest.raises(SystemExit, match="hash does not match"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_unlisted_fit_cannot_enter_a_run_record(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, _, _ = _setup_runs(tmp_path, target, root, capsys)
    _write(
        tmp_path / "runs" / "bad.json",
        [{"fit_id": "not/planned", "system": "s", "seed": 0, "predictions": "0.json"}],
    )
    with pytest.raises(SystemExit, match="not listed in the freeze record"):
        _cli(
            "evaluate",
            "run-record",
            "--freeze-record",
            record,
            "--runs",
            tmp_path / "runs" / "bad.json",
            "--output",
            tmp_path / "runs" / "bad-record.json",
        )


def test_changed_predictions_refuse_the_read(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, run_record, predictions = _setup_runs(tmp_path, target, root, capsys)
    predictions.write_text(
        predictions.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="predictions changed"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_record_made_after_a_read_is_flagged_post_hoc(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record, run_record, _ = _setup_runs(tmp_path, target, root, capsys)
    log = tmp_path / "reads.jsonl"
    _cli(*_test_args(tmp_path, target, root, record, run_record), "--reads-log", log)
    assert log.read_text(encoding="utf-8").strip()
    second = _freeze(tmp_path, root, capsys, [FIT.format(0)], "second.json", log)
    assert json.loads(second.read_text(encoding="utf-8"))["body"]["post_hoc"] is True
    assert json.loads(record.read_text(encoding="utf-8"))["body"]["post_hoc"] is False


def test_all_analyses_run_through_the_evaluator(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = load_manifest(target).partition("test")
    fits = {
        "full": FIT.format(0),
        "leaf_only": "fixture/leaf/seed-0",
        "background_only": "fixture/background/seed-0",
        "border": "fixture/border/seed-0",
        "pad_cue": "fixture/pad/seed-0",
    }
    record = _freeze(tmp_path, root, capsys, list(fits.values()))
    sample_args = ["--freeze-record", record, "--target-dir", target, "--root", root]
    capsys.readouterr()
    _cli("evaluate", "leaf-sample", *sample_args, "--sample-size", 12)
    sample = json.loads(capsys.readouterr().out)
    assert len(sample) == 12
    runs = []
    for variant, fit_id in fits.items():
        file = f"{variant}.json"
        _predictions(
            target,
            tmp_path / "runs" / file,
            5,
            sample if variant == "leaf_only" else None,
        )
        runs.append(
            {
                "fit_id": fit_id,
                "system": "pad" if variant == "pad_cue" else "mobilenet_v2_direct",
                "seed": 0,
                "variant": variant,
                "predictions": file,
                "of": fits["full"] if variant == "leaf_only" else None,
            }
        )
    _write(tmp_path / "runs" / "runs.json", runs)
    run_record = tmp_path / "runs" / "run-record.json"
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        record,
        "--runs",
        tmp_path / "runs" / "runs.json",
        "--output",
        run_record,
    )
    features = _write(
        tmp_path / "features.json",
        {
            "scenes": {e.image_id: f"scene-{i % 2}" for i, e in enumerate(entries)},
            "frame_leaf_share": {e.image_id: 0.0 for e in entries},
        },
    )
    _cli(
        *_test_args(tmp_path, target, root, record, run_record),
        "--features",
        features,
        "--sample-size",
        12,
    )
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    system = report["systems"]["mobilenet_v2_direct"]
    assert system["class_scene_cells"] and system["scene_summaries"]
    assert system["leaf_only"] and system["full_on_leaf_sample"]
    assert system["background_probe"]["statement"]
    border = system["border_probe"]
    assert border["headline"] is None and "fewer than 50" in border["headline_withheld"]
    assert border["pad_cue"] is not None
    assert report["leaf_sample"]["image_ids"] == sample


def test_two_runs_for_one_system_and_seed_are_refused(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _freeze(tmp_path, root, capsys, [FIT.format(0), "fixture/dup"])
    _predictions(target, tmp_path / "runs" / "0.json", 7, None)
    entry = {"system": "mobilenet_v2_direct", "seed": 0, "predictions": "0.json"}
    _write(
        tmp_path / "runs" / "runs.json",
        [{**entry, "fit_id": FIT.format(0)}, {**entry, "fit_id": "fixture/dup"}],
    )
    run_record = tmp_path / "runs" / "run-record.json"
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        record,
        "--runs",
        tmp_path / "runs" / "runs.json",
        "--output",
        run_record,
    )
    with pytest.raises(SystemExit, match="more than one run"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))

"""Gate controls and the full analysis run through the ``jute-disease`` CLI."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from shutil import copytree

import pytest

from jute_disease import cli as cli_module
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data.manifest import (
    load_manifest,
    load_protected_test_manifest,
    write_manifest,
)
from jute_disease.training import RunStore, load_benchmark

from .conftest import audit_manifest_of
from .test_gated_evaluator import (
    FIT,
    LEAF_FIT,
    SYSTEM,
    _cli,
    _complete_fit,
    _freeze,
    _issue_receipts,
    _leaf_run,
    _outcomes,
    _probe_ids,
    _spec,
    _write,
)


def test_freeze_record_needs_recorded_validation_outcomes(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _spec(tmp_path, [], _probe_ids(capsys))
    base = [
        "evaluate",
        "freeze-record",
        "--spec",
        spec,
        "--target-dir",
        target,
        "--root",
        root,
        "--policy-dir",
        policy_dir,
        "--mask-audit-report",
        mask_audit_report,
        "--mask-audit-manifest",
        audit_manifest_of(mask_audit_report),
    ]
    with pytest.raises(SystemExit, match="validation outcomes"):
        _cli(*base, "--output", tmp_path / "none.json")
    wrong = _write(tmp_path / "wrong.json", {"partition": "test"})
    with pytest.raises(SystemExit, match="validation outcomes"):
        _cli(*base, "--outcomes", wrong, "--output", tmp_path / "wrong-out.json")
    assert not (tmp_path / "none.json").exists()
    assert not (tmp_path / "wrong-out.json").exists()


def test_freeze_record_must_list_every_probe_fit(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
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
            "--target-dir",
            target,
            "--root",
            root,
            "--policy-dir",
            policy_dir,
            "--mask-audit-report",
            mask_audit_report,
            "--mask-audit-manifest",
            audit_manifest_of(mask_audit_report),
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
    assert len(final_cnn_seeds(border)) == len(benchmark.recipe.seeds)
    pad_cue_seeds = {
        p.rsplit("seed-", 1)[1] for p in probes if p.startswith("probe/pad-cue/")
    }
    assert len(pad_cue_seeds) == len(benchmark.recipe.seeds)
    for pool in benchmark.pools:
        for policy in benchmark.policies:
            assert any(f"pool-{pool}" in p and policy in p for p in background)


def _setup_runs(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path, Path, Path]:
    leaf = LEAF_FIT.format(0)
    record = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), leaf],
    )
    store = RunStore(tmp_path / "runs")
    full = _complete_fit(store, target, FIT.format(0), 0, 3)
    _write(
        tmp_path / "runs" / "runs.json",
        [full, _leaf_run(store, target, leaf, FIT.format(0), 0, 2)],
    )
    _issue_receipts(store, record)
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
    return record, run_record, store.run_dir(FIT.format(0)) / "predictions.json"


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


def test_freeze_record_binds_both_manifest_paths_schemas_and_bytes(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _freeze(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys, [FIT.format(0)]
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    binding = payload["body"]["manifest_binding"]
    assert binding["schema"] == 1
    for name, relative in (
        ("fit", "manifest.json"),
        ("protected_test", "protected-test/manifest.json"),
    ):
        path = target / relative
        assert binding[name]["schema"] == 2
        assert Path(binding[name]["path"]) == path.resolve()
        assert binding[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_record_refuses_a_missing_protected_manifest(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    incomplete = tmp_path / "incomplete-target"
    copytree(target, incomplete)
    (incomplete / "protected-test" / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="freeze record blocked"):
        _freeze(
            tmp_path,
            incomplete,
            root,
            policy_dir,
            mask_audit_report,
            capsys,
            [FIT.format(0)],
        )
    assert not (tmp_path / "freeze-record.json").exists()


def test_freeze_record_refuses_mismatched_split_rules(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mismatched = tmp_path / "mismatched-target"
    copytree(target, mismatched)
    protected = load_manifest(mismatched / "protected-test")
    write_manifest(
        replace(protected, seed=protected.seed + 1), mismatched / "protected-test"
    )

    with pytest.raises(SystemExit, match="mismatched split rules"):
        _freeze(
            tmp_path,
            mismatched,
            root,
            policy_dir,
            mask_audit_report,
            capsys,
            [FIT.format(0)],
        )
    assert not (tmp_path / "freeze-record.json").exists()


def test_changed_pipeline_code_refuses_the_read(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, run_record, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="pipeline code changed"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_edited_freeze_record_refuses_the_read(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, run_record, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["body"]["budget"] = {"a100_gpu_hours": 99}
    _write(record, payload)
    with pytest.raises(SystemExit, match="hash does not match"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_unlisted_fit_cannot_enter_a_run_record(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, _, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
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
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, run_record, predictions = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    predictions.write_text(
        predictions.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="predictions changed"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))


def test_changed_protected_manifest_refuses_test_read_before_logging(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, run_record, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    manifest = target / "protected-test" / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="manifest bytes changed"):
        _cli(*_test_args(tmp_path, target, root, record, run_record))
    assert not (root / "artifacts/evaluation/test-reads.jsonl").exists()


def test_diagnostics_cli_gates_and_logs_authorized_manifest_read(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, run_record, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    protected_path = target / "protected-test" / "manifest.json"
    original = protected_path.read_bytes()
    protected_path.write_bytes(original + b" ")
    arguments = [
        "evaluate",
        "diagnostics",
        "--freeze-record",
        record,
        "--run-record",
        run_record,
        "--target-dir",
        target,
        "--image-root",
        tmp_path / "images",
        "--assignment-table",
        tmp_path / "assignments.json",
        "--frozen-backbone-checkpoint",
        tmp_path / "backbone.ckpt",
        "--context",
        _write(tmp_path / "context.json", {}),
        "--output",
        tmp_path / "diagnostics",
        "--fit-id",
        FIT.format(0),
        "--root",
        root,
    ]
    with pytest.raises(SystemExit, match="manifest bytes changed"):
        _cli(*arguments)
    read_log = root / "artifacts/evaluation/test-reads.jsonl"
    assert not read_log.exists()

    protected_path.write_bytes(original)
    monkeypatch.setattr(
        cli_module,
        "evaluate_image_diagnostics",
        lambda *args, **kwargs: {"export": str(tmp_path / "diagnostics")},
    )
    _cli(*arguments)
    rows = [json.loads(line) for line in read_log.read_text().splitlines()]
    assert rows[-1]["operation"] == "image_diagnostics"
    assert rows[-1]["record_hash"] == json.loads(record.read_text())["hash"]
    assert rows[-1]["fit_ids"] == [FIT.format(0)]


def test_leaf_sample_cli_gates_and_logs_authorized_read(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _freeze(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys, [FIT.format(0)]
    )
    capsys.readouterr()
    _cli(
        "evaluate",
        "leaf-sample",
        "--freeze-record",
        record,
        "--target-dir",
        target,
        "--root",
        root,
        "--sample-size",
        12,
    )
    sample = json.loads(capsys.readouterr().out)
    assert len(sample) == 12
    rows = [
        json.loads(line)
        for line in (root / "artifacts/evaluation/test-reads.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        row["operation"] == "leaf_sample"
        and row["record_hash"] == json.loads(record.read_text())["hash"]
        for row in rows
    )


def test_record_made_after_a_read_is_flagged_post_hoc(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record, run_record, _ = _setup_runs(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys
    )
    log = tmp_path / "reads.jsonl"
    _cli(*_test_args(tmp_path, target, root, record, run_record), "--reads-log", log)
    assert log.read_text(encoding="utf-8").strip()
    second = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0)],
        "second.json",
        log,
    )
    assert json.loads(second.read_text(encoding="utf-8"))["body"]["post_hoc"] is True
    assert json.loads(record.read_text(encoding="utf-8"))["body"]["post_hoc"] is False


def test_all_analyses_run_through_the_evaluator(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    passing_mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entries = load_protected_test_manifest(target).entries
    fits = {
        "full": FIT.format(0),
        "leaf_only": "fixture/leaf/seed-0",
        "background_only": "fixture/background/seed-0",
        "border": "fixture/border/seed-0",
        "pad_cue": "fixture/pad/seed-0",
    }
    record = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        passing_mask_audit_report,
        capsys,
        list(fits.values()),
    )
    sample_args = ["--freeze-record", record, "--target-dir", target, "--root", root]
    capsys.readouterr()
    _cli("evaluate", "leaf-sample", *sample_args, "--sample-size", 12)
    sample = json.loads(capsys.readouterr().out)
    assert len(sample) == 12
    reads = [
        json.loads(line)
        for line in (root / "artifacts/evaluation/test-reads.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        row["operation"] == "leaf_sample"
        and row["record_hash"] == json.loads(record.read_text(encoding="utf-8"))["hash"]
        for row in reads
    )
    store = RunStore(tmp_path / "runs")
    runs = []
    for variant, fit_id in fits.items():
        if variant == "leaf_only":
            runs.append(_leaf_run(store, target, fit_id, fits["full"], 0, 5, sample))
            continue
        system = "pad" if variant == "pad_cue" else SYSTEM
        trained = variant if variant in ("border", "pad_cue") else ""
        run = _complete_fit(store, target, fit_id, 0, 5, system=system, variant=trained)
        runs.append({**run, "variant": variant, "of": None})
    _write(tmp_path / "runs" / "runs.json", runs)
    _issue_receipts(store, record)
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
    for result in (system["background_probe"], border["unrestricted"]):
        assert 0 <= result["exchangeable_image_share"] <= 1
        assert result["exchangeable_images"] <= result["n"]
        assert result["exchangeable_families"] <= result["families"]
        assert result["p_value_standard_error"] >= 0
    assert border["headline"] is None and "fewer than 50" in border["headline_withheld"]
    assert border["pad_cue"] is not None
    assert report["leaf_sample"]["image_ids"] == sample


def test_two_runs_for_one_system_and_seed_are_refused(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), "fixture/dup"],
    )
    store = RunStore(tmp_path / "runs")
    runs = [
        _complete_fit(store, target, FIT.format(0), 0, 3),
        _complete_fit(store, target, "fixture/dup", 0, 3),
    ]
    _write(tmp_path / "runs" / "runs.json", runs)
    _issue_receipts(store, record)
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

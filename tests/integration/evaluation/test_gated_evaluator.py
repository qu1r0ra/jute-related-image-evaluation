"""The gated evaluator through the ``jute-disease`` CLI on the synthetic fixture."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease import cli as cli_module
from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import load_manifest
from jute_disease.data.manifest import load_protected_test_manifest
from jute_disease.evaluation import (
    receipt_manifest_digest,
    training_description,
    training_receipt_digest,
    write_prediction_receipt,
)
from jute_disease.training import RunStore, load_benchmark

from .conftest import audit_manifest_of

LABEL_COUNT = 2
SYSTEM = "mobilenet_v2_direct"
FIT = "fixture/mobilenet_v2_direct/seed-{}"
LEAF_FIT = "fixture/mobilenet_v2_direct/leaf-seed-{}"
ANALYSES = [
    "background-probe",
    "border-probe",
    "embeddings-test-pass",
    "gradcam-test-pass",
    "leaf-only",
    "scene-cells",
]


def _cli(*args: object) -> None:
    main([str(a) for a in args])


def _declared_fits() -> list[str]:
    benchmark = load_benchmark(BENCHMARK_CONFIG)
    return [fit.fit_id for fit in benchmark.fits()]


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _spec(tmp_path: Path, extra_fits: list[str], probes: list[str]) -> Path:
    return _write(
        tmp_path / "spec.json",
        {
            "fit_ids": _declared_fits() + probes + extra_fits,
            "analysis_ids": ANALYSES,
            "selections": {"cnn/tune/pool-A": 1.0},
            "letterbox_decision": {"choice": "letterbox"},
            "diagnostic": {"pool": "A", "view": 1, "policy": "image-random"},
            "border_probe": {
                "start": {"width_px": 32, "leaf_cutoff": 0.05},
                "final": {"width_px": 32, "leaf_cutoff": 0.05},
            },
            "budget": {"a100_gpu_hours": 10},
        },
    )


def _outcomes(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "outcomes.json",
        {
            "partition": "val",
            "letterbox_comparison": {"letterbox": 0.9, "resize": 0.8},
            "selections": {"cnn/tune/pool-A": {"selected": 1.0, "val_macro_f1": 0.9}},
        },
    )


def _probe_ids(capsys: pytest.CaptureFixture[str]) -> list[str]:
    capsys.readouterr()
    _cli("evaluate", "probe-plan")
    return json.loads(capsys.readouterr().out)


def _freeze(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
    extra_fits: list[str],
    name: str = "freeze-record.json",
    reads_log: Path | None = None,
) -> Path:
    arguments: list[object] = [
        "evaluate",
        "freeze-record",
        "--spec",
        _spec(tmp_path, extra_fits, _probe_ids(capsys)),
        "--outcomes",
        _outcomes(tmp_path),
        "--output",
        tmp_path / name,
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
    if reads_log is not None:
        arguments += ["--reads-log", reads_log]
    _cli(*arguments)
    return tmp_path / name


def _predictions(target: Path, path: Path, flip: int, ids: list[str] | None) -> None:
    entries = sorted(
        load_protected_test_manifest(target).entries, key=lambda e: e.image_id
    )
    labels = sorted({e.label for e in load_manifest(target).entries})
    chosen = [e for e in entries if ids is None or e.image_id in ids]
    predicted = [
        e.label
        if (i % 4 != flip)
        else labels[(labels.index(e.label) + 1) % len(labels)]
        for i, e in enumerate(chosen)
    ]
    _write(path, {"image_ids": [e.image_id for e in chosen], "predictions": predicted})


def _receipt(
    store: RunStore,
    fit_id: str,
    seed: int,
    manifest: str,
    *,
    classical: bool = False,
    system: str = SYSTEM,
) -> None:
    """Record a completed fit the way the trainer does, with a real artifact.

    ``manifest`` is the digest of the train manifest the fit trained on, recorded
    where the trainers record it: ``manifest`` for a classical fit and
    ``lineage.manifests.train`` for a deep fit.
    """
    run = store.start(fit_id, "identity")
    if classical:
        artifact = run.attempt_dir / "model.joblib"
        artifact.write_bytes(f"model-{fit_id}".encode())
        run.complete(
            {
                "fit": fit_id,
                "seed": seed,
                "manifest": manifest,
                "metrics": {"val_macro_f1": 0.5},
            },
            artifacts={"model": artifact},
        )
        return
    artifact = run.attempt_dir / "best.ckpt"
    artifact.write_bytes(f"weights-{fit_id}".encode())
    run.complete(
        {
            "fit": {"fit_id": fit_id, "system": system, "seed": seed},
            "lineage": {"manifests": {"train": manifest}},
            "metrics": {"val_f1": 0.5},
        },
        artifacts={"checkpoint": artifact},
    )


def _issue_receipts(store: RunStore, freeze: Path) -> None:
    """Write the prediction receipt ``predict-test`` writes, for each fit with a file.

    The fixtures write predictions straight into the store, so this stands in for
    the gate: one receipt per full fit, under the given freeze record. A
    leaf-only run has no training receipt of its own and is skipped.
    """
    record_hash = json.loads(freeze.read_text(encoding="utf-8"))["hash"]
    for receipt_file in sorted(store.root.rglob("receipt.json")):
        run_dir = receipt_file.parent
        predictions = run_dir / "predictions.json"
        fit_id = run_dir.relative_to(store.root).as_posix()
        receipt = store.receipt(fit_id, missing_ok=True)
        if receipt is None or not predictions.exists():
            continue
        description, kind, artifact = training_description(receipt, fit_id)
        write_prediction_receipt(
            run_dir,
            record_hash=record_hash,
            fit_id=fit_id,
            training_receipt_sha256=training_receipt_digest(run_dir),
            artifact_kind=kind,
            artifact_sha256=artifact["sha256"],
            fit_manifest_sha256=receipt_manifest_digest(description),
            predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        )


def _complete_fit(
    store: RunStore,
    target: Path,
    fit_id: str,
    seed: int,
    flip: int,
    *,
    system: str = SYSTEM,
    classical: bool = False,
) -> dict[str, object]:
    """Complete one fit in the store and return its run-record entry."""
    _receipt(
        store,
        fit_id,
        seed,
        load_manifest(target).digest(),
        classical=classical,
        system=system,
    )
    _predictions(target, store.run_dir(fit_id) / "predictions.json", flip, None)
    return {
        "fit_id": fit_id,
        "system": system,
        "seed": seed,
        "predictions": f"{fit_id}/predictions.json",
    }


def _leaf_run(
    store: RunStore,
    target: Path,
    fit_id: str,
    of: str,
    seed: int,
    flip: int,
    ids: list[str] | None = None,
) -> dict[str, object]:
    """Write a leaf-only run's predictions and return its run-record entry."""
    _predictions(target, store.run_dir(fit_id) / "predictions.json", flip, ids)
    return {
        "fit_id": fit_id,
        "system": SYSTEM,
        "seed": seed,
        "variant": "leaf_only",
        "predictions": f"{fit_id}/predictions.json",
        "of": of,
    }


def _gate_arguments(tmp_path: Path, target: Path, root: Path) -> tuple[object, ...]:
    return (
        "evaluate",
        "test",
        "--target-dir",
        target,
        "--root",
        root,
        "--output",
        tmp_path / "out" / "test.json",
        "--resamples",
        20,
        "--permutations",
        20,
    )


def _record_runs(
    target: Path, store: RunStore, *, classical: bool = False
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for seed in (0, 1):
        full = FIT.format(seed)
        runs.append(_complete_fit(store, target, full, seed, seed, classical=classical))
        runs.append(
            _leaf_run(store, target, LEAF_FIT.format(seed), full, seed, seed + 2)
        )
    return runs


def test_test_read_refused_without_records_and_permitted_after(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _gate_arguments(tmp_path, target, root)
    with pytest.raises(SystemExit, match="no frozen selection record"):
        _cli(*arguments, "--freeze-record", tmp_path / "missing.json")
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match="no frozen selection record"):
        _cli(
            "evaluate",
            "leaf-sample",
            "--target-dir",
            target,
            "--root",
            root,
            "--freeze-record",
            tmp_path / "missing.json",
            "--sample-size",
            12,
        )
    assert not (root / "artifacts/evaluation/test-reads.jsonl").exists()

    freeze = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), FIT.format(1), LEAF_FIT.format(0), LEAF_FIT.format(1)],
    )
    store = RunStore(tmp_path / "runs")
    runs = _record_runs(target, store)
    _issue_receipts(store, freeze)
    runs_file = _write(tmp_path / "runs.json", runs)
    with pytest.raises(SystemExit, match="no run record"):
        _cli(*arguments, "--freeze-record", freeze)
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        runs_file,
        "--output",
        tmp_path / "runs" / "run-record.json",
    )

    authorized: dict[str, object] = {}

    def _authorized_stub(
        target_dir: Path, auth: object, run_dir: Path, **kwargs: object
    ) -> dict[str, object]:
        del run_dir, kwargs
        authorized["target_dir"] = target_dir
        authorized["auth"] = auth
        return {
            "post_hoc": False,
            "record_hash": auth.record_hash,
        }

    monkeypatch.setattr(cli_module, "evaluate_test", _authorized_stub)
    _cli(
        *arguments,
        "--freeze-record",
        freeze,
        "--run-record",
        tmp_path / "runs" / "run-record.json",
    )
    report = json.loads((tmp_path / "out" / "test.json").read_text(encoding="utf-8"))
    assert report["post_hoc"] is False
    assert authorized["target_dir"] == target
    assert authorized["auth"].record_hash == report["record_hash"]
    assert authorized["auth"].runs
    read_log = root / "artifacts/evaluation/test-reads.jsonl"
    rows = [json.loads(line) for line in read_log.read_text().splitlines()]
    assert any(
        row["operation"] == "evaluate_test"
        and row["record_hash"] == json.loads(freeze.read_text())["hash"]
        for row in rows
    )


def test_run_record_binds_each_fit_by_digest_wherever_it_is_written(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), FIT.format(1), LEAF_FIT.format(0), LEAF_FIT.format(1)],
    )
    store = RunStore(tmp_path / "store")
    runs_file = _write(tmp_path / "runs.json", _record_runs(target, store))
    _issue_receipts(store, freeze)
    elsewhere = tmp_path / "elsewhere" / "run-record.json"
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        runs_file,
        "--run-store",
        tmp_path / "store",
        "--output",
        elsewhere,
    )
    fits = json.loads(elsewhere.read_text(encoding="utf-8"))["body"]["fits"]
    full, leaf = fits[FIT.format(0)], fits[LEAF_FIT.format(0)]
    receipt = store.receipt(FIT.format(0))
    assert full["checkpoint_sha256"] == receipt["artifacts"]["checkpoint"]["sha256"]
    assert full["training_receipt_sha256"]
    assert full["lineage_fit_id"] == FIT.format(0)
    assert leaf["lineage_fit_id"] == FIT.format(0)
    assert leaf["checkpoint_sha256"] == full["checkpoint_sha256"]
    assert leaf["training_receipt_sha256"] == full["training_receipt_sha256"]
    assert leaf["predictions_sha256"] != full["predictions_sha256"]


def test_run_record_accepts_classical_receipts(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), FIT.format(1), LEAF_FIT.format(0), LEAF_FIT.format(1)],
    )
    store = RunStore(tmp_path / "runs")
    runs_file = _write(
        tmp_path / "runs.json", _record_runs(target, store, classical=True)
    )
    _issue_receipts(store, freeze)
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        runs_file,
        "--output",
        tmp_path / "runs" / "run-record.json",
    )
    fits = json.loads((tmp_path / "runs" / "run-record.json").read_text())["body"][
        "fits"
    ]
    entry = fits[FIT.format(1)]
    assert (
        entry["model_sha256"]
        == store.receipt(FIT.format(1))["artifacts"]["model"]["sha256"]
    )
    assert "checkpoint_sha256" not in entry or entry["checkpoint_sha256"] is None
    assert entry["lineage_fit_id"] == FIT.format(1)


def test_run_record_refuses_a_fit_without_a_receipt(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FIT.format(0), LEAF_FIT.format(0)],
    )
    store = RunStore(tmp_path / "runs")
    runs = [r for r in _record_runs(target, store) if "-0" in r["fit_id"]]
    _issue_receipts(store, freeze)
    (store.run_dir(FIT.format(0)) / "receipt.json").unlink()
    runs_file = _write(tmp_path / "runs.json", runs)
    with pytest.raises(SystemExit, match="no verified training receipt"):
        _cli(
            "evaluate",
            "run-record",
            "--freeze-record",
            freeze,
            "--runs",
            runs_file,
            "--output",
            tmp_path / "runs" / "run-record.json",
        )
    assert not (tmp_path / "runs" / "run-record.json").exists()

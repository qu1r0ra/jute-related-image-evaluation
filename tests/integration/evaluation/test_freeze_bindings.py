"""What the freeze record binds, what the gate re-verifies, and the read log."""

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from jute_disease.data import load_policy_complete_manifest
from jute_disease.evaluation import (
    FreezeRecordError,
    TestReadRefusedError,
    log_test_read,
    verify_freeze_record,
    write_freeze_record,
)
from jute_disease.evaluation import freeze_record as freeze_module
from jute_disease.training import FileLock

from .conftest import audit_manifest_of, redigest

FIT = "cnn/mobilenet_v2_direct/pool-A/view-1/near-duplicate/split-0/seed-42"
ANALYSES = [
    "background-probe",
    "border-probe",
    "embeddings-test-pass",
    "gradcam-test-pass",
    "leaf-only",
    "scene-cells",
]


def _spec(**overrides: Any) -> dict[str, Any]:
    return {
        "fit_ids": [FIT],
        "analysis_ids": ANALYSES,
        "selections": {"letterbox": "letterbox"},
        "letterbox_decision": {"choice": "letterbox"},
        "diagnostic": {"pool": "A", "view": 1, "policy": "near-duplicate"},
        "border_probe": {
            "start": {"width_px": 32, "leaf_cutoff": 0.05},
            "final": {"width_px": 32, "leaf_cutoff": 0.05},
        },
        "budget": {},
        **overrides,
    }


def _outcomes(**overrides: Any) -> dict[str, Any]:
    return {
        "partition": "val",
        "letterbox_comparison": {"letterbox": 0.9, "resize": 0.8},
        "selections": {"letterbox": {"selected": "letterbox"}},
        **overrides,
    }


def _write(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    report: Path,
    *,
    spec: dict[str, Any] | None = None,
    outcomes: dict[str, Any] | None = None,
    name: str = "freeze.json",
    **extra: Any,
) -> Path:
    extra.setdefault("mask_audit_manifest", audit_manifest_of(report))
    write_freeze_record(
        spec or _spec(),
        outcomes or _outcomes(),
        tmp_path / name,
        root,
        target_dir=target,
        policy_dir=policy_dir,
        mask_audit_report=report,
        **extra,
    )
    return tmp_path / name


def _body(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["body"]


def _copy_audit(source: Path, directory: Path) -> Path:
    """Copy a report and its audit manifest so a test can edit either."""
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "report.json"
    shutil.copyfile(source, report)
    shutil.copyfile(audit_manifest_of(source), audit_manifest_of(report))
    return report


def test_the_record_binds_the_mask_audit_report_by_digest(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    passing_mask_audit_report: Path,
) -> None:
    failed = _body(_write(tmp_path, target, root, policy_dir, mask_audit_report))
    report = json.loads(mask_audit_report.read_text(encoding="utf-8"))
    audit = failed["mask_audit"]
    assert audit["digest"] == report["digest"]
    assert audit["audit_manifest_digest"] == report["audit_manifest_digest"]
    assert (
        audit["report_sha256"]
        == hashlib.sha256(mask_audit_report.read_bytes()).hexdigest()
    )
    assert audit["passed"] is False

    passed = _body(
        _write(
            tmp_path, target, root, policy_dir, passing_mask_audit_report, name="p.json"
        )
    )
    assert passed["mask_audit"]["passed"] is True


def test_an_operator_cannot_write_the_audit_outcome(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    with pytest.raises(FreezeRecordError, match="derived from the audit report"):
        _write(
            tmp_path,
            target,
            root,
            policy_dir,
            mask_audit_report,
            spec=_spec(mask_audit={"passed": True}),
        )
    with pytest.raises(FreezeRecordError, match="different mask audit report"):
        _write(
            tmp_path,
            target,
            root,
            policy_dir,
            mask_audit_report,
            outcomes=_outcomes(mask_audit={"digest": "0" * 64}),
        )
    assert not (tmp_path / "freeze.json").exists()


def test_a_report_edited_to_pass_is_refused_at_write(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    edited = _copy_audit(mask_audit_report, tmp_path / "audit")
    report = json.loads(edited.read_text(encoding="utf-8"))
    for gate in ("mask_quality", "scene_tag"):
        report[gate]["gate_pass"] = True
    edited.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(FreezeRecordError, match="digest does not match"):
        _write(tmp_path, target, root, policy_dir, edited)


def test_the_gate_refuses_a_report_that_changed_after_the_freeze(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    passing_mask_audit_report: Path,
) -> None:
    report = _copy_audit(passing_mask_audit_report, tmp_path / "audit")
    freeze = _write(tmp_path, target, root, policy_dir, report)
    verify_freeze_record(freeze, root, target)

    edited = json.loads(report.read_text(encoding="utf-8"))
    edited["scene_tag"]["lower_95"] = 0.95
    report.write_text(json.dumps(redigest(edited)), encoding="utf-8")
    with pytest.raises(TestReadRefusedError, match="changed since"):
        verify_freeze_record(freeze, root, target)

    report.unlink()
    with pytest.raises(TestReadRefusedError, match="missing or invalid"):
        verify_freeze_record(freeze, root, target)


def test_the_gate_refuses_an_audit_manifest_that_changed_after_the_freeze(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    report = _copy_audit(mask_audit_report, tmp_path / "audit")
    freeze = _write(tmp_path, target, root, policy_dir, report)
    manifest = audit_manifest_of(report)
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(TestReadRefusedError, match="changed since"):
        verify_freeze_record(freeze, root, target)


def test_a_fixture_mask_audit_is_not_evidence(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    report = _copy_audit(mask_audit_report, tmp_path / "audit")
    edited = json.loads(report.read_text(encoding="utf-8"))
    edited["fixture_mode"] = True
    report.write_text(json.dumps(redigest(edited)), encoding="utf-8")
    with pytest.raises(FreezeRecordError, match="fixture mask audit"):
        _write(tmp_path, target, root, policy_dir, report)
    assert not (tmp_path / "freeze.json").exists()


def test_a_gate_flag_edited_to_pass_is_refused_even_with_a_redone_digest(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    report = _copy_audit(mask_audit_report, tmp_path / "audit")
    edited = json.loads(report.read_text(encoding="utf-8"))
    for gate in ("mask_quality", "scene_tag"):
        edited[gate]["gate_pass"] = True
    edited["mask_quality"]["fail_rate"]["gate_pass"] = True
    report.write_text(json.dumps(redigest(edited)), encoding="utf-8")
    with pytest.raises(FreezeRecordError, match="disagree with its numbers"):
        _write(tmp_path, target, root, policy_dir, report)


def test_the_outcome_is_recomputed_from_the_numbers(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    passing_mask_audit_report: Path,
) -> None:
    report = _copy_audit(passing_mask_audit_report, tmp_path / "audit")
    edited = json.loads(report.read_text(encoding="utf-8"))
    edited["mask_quality"]["fail_rate"]["upper_95"] = 0.5
    edited["mask_quality"]["fail_rate"]["gate_pass"] = False
    edited["mask_quality"]["gate_pass"] = False
    report.write_text(json.dumps(redigest(edited)), encoding="utf-8")
    body = _body(_write(tmp_path, target, root, policy_dir, report))
    assert body["mask_audit"]["passed"] is False


def test_a_report_for_another_audit_manifest_is_refused(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    report = _copy_audit(mask_audit_report, tmp_path / "audit")
    other = json.loads(audit_manifest_of(report).read_text(encoding="utf-8"))
    other["owner_count"] = other["owner_count"] + 1
    audit_manifest_of(report).write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(
        FreezeRecordError, match="does not belong to the audit manifest"
    ):
        _write(tmp_path, target, root, policy_dir, report)


def test_paths_inside_the_root_are_stored_relative_to_it(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    inside = _copy_audit(mask_audit_report, root / "evidence")
    matrix = root / "matrix"
    shutil.copytree(policy_dir, matrix)
    freeze = _write(tmp_path, target, root, matrix, inside)
    body = _body(freeze)
    assert body["mask_audit"]["report"] == "evidence/report.json"
    assert body["mask_audit"]["audit_manifest"] == "evidence/audit-manifest.json"
    assert body["policy_binding"]["path"] == "matrix"
    verify_freeze_record(freeze, root, target)


def test_the_record_binds_every_policy_draw_and_the_gate_rechecks_them(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    matrix = tmp_path / "matrix"
    shutil.copytree(policy_dir, matrix)
    freeze = _write(tmp_path, target, root, matrix, mask_audit_report)
    binding = _body(freeze)["policy_binding"]
    assert Path(binding["path"]) == matrix.resolve()
    assert {name.split("/", 1)[0] for name in binding["files"]} == {"A", "B", "C"}
    tests = [name for name in binding["files"] if name.endswith(".protected-test.json")]
    assert tests, "protected-test files of every draw are bound"
    verify_freeze_record(freeze, root, target)

    name = sorted(binding["files"])[0]
    victim = matrix / name
    original = victim.read_bytes()
    victim.write_bytes(original + b" ")
    with pytest.raises(TestReadRefusedError, match="bound policy matrix"):
        verify_freeze_record(freeze, root, target)
    victim.write_bytes(original)
    verify_freeze_record(freeze, root, target)

    victim.unlink()
    with pytest.raises(TestReadRefusedError, match="bound policy matrix"):
        verify_freeze_record(freeze, root, target)


def test_an_incomplete_matrix_is_not_bound(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    matrix = tmp_path / "matrix"
    shutil.copytree(policy_dir, matrix)
    receipt = next(
        path for path in matrix.iterdir() if path.is_file() and "receipt" in path.name
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["files"].pop(sorted(payload["files"])[0])
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FreezeRecordError, match="cannot bind the policy matrix"):
        _write(tmp_path, target, root, matrix, mask_audit_report)


@pytest.mark.parametrize(
    "diagnostic",
    [
        {"pool": "Z", "view": 1, "policy": "near-duplicate"},
        {"pool": "A", "view": 9, "policy": "near-duplicate"},
        {"pool": "A", "view": 1, "policy": "bogus"},
        {"pool": "A", "view": 1},
    ],
)
def test_the_diagnostic_must_name_one_cell_of_the_frozen_matrix(
    diagnostic: dict[str, Any],
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    with pytest.raises(FreezeRecordError, match="diagnostic must name"):
        _write(
            tmp_path,
            target,
            root,
            policy_dir,
            mask_audit_report,
            spec=_spec(diagnostic=diagnostic),
        )


def test_the_diagnostic_must_lie_inside_a_narrowed_matrix(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    narrowed = {"pools": ["A"]}
    _write(tmp_path, target, root, policy_dir, mask_audit_report, matrix=narrowed)
    with pytest.raises(FreezeRecordError, match="diagnostic must name"):
        _write(
            tmp_path,
            target,
            root,
            policy_dir,
            mask_audit_report,
            spec=_spec(diagnostic={"pool": "B", "view": 1, "policy": "near-duplicate"}),
            matrix=narrowed,
            name="other.json",
        )


def test_a_pool_fit_reads_the_complete_manifest_of_its_draw(
    policy_dir: Path, tmp_path: Path
) -> None:
    cell = {"pool": "A", "view": 1, "policy": "near-duplicate", "draw": 0}
    manifest = load_policy_complete_manifest(policy_dir, **cell)
    assert {"train", "val", "test"} <= {entry.partition for entry in manifest.entries}

    matrix = tmp_path / "matrix"
    shutil.copytree(policy_dir, matrix)
    held_out = matrix / "A/view1/near-duplicate/draw-0.protected-test.json"
    held_out.write_bytes(held_out.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt"):
        load_policy_complete_manifest(matrix, **cell)


def test_a_residual_audit_read_does_not_make_a_record_post_hoc(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    log = freeze_module.default_reads_log_path(root)
    details = {
        "pool": "A",
        "sample_ids_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
    }
    log_test_read(
        log,
        "none",
        [],
        operation=freeze_module.RESIDUAL_AUDIT_OPERATION,
        details=details,
    )
    clean = _body(_write(tmp_path, target, root, policy_dir, mask_audit_report))
    assert clean["post_hoc"] is False
    assert clean["invalidated_reads"] == []
    assert [row["details"] for row in clean["audit_reads"]] == [details]

    log_test_read(log, "earlier-record", [FIT], operation="predict_test")
    flagged = _body(
        _write(tmp_path, target, root, policy_dir, mask_audit_report, name="b.json")
    )
    assert flagged["post_hoc"] is True
    assert [row["operation"] for row in flagged["invalidated_reads"]] == [
        "predict_test"
    ]


@pytest.mark.parametrize(
    "row",
    [
        {"fit_ids": ["pool-A"], "details": {"pool": "A"}},
        {"fit_ids": []},
        {"fit_ids": [], "details": {"pool": "A", "sample_ids_sha256": "a" * 64}},
        {
            "fit_ids": [],
            "details": {
                "pool": "",
                "sample_ids_sha256": "a" * 64,
                "weights_sha256": "b" * 64,
            },
        },
        {
            "fit_ids": [FIT],
            "details": {
                "pool": "A",
                "sample_ids_sha256": "a" * 64,
                "weights_sha256": "b" * 64,
            },
        },
    ],
)
def test_a_row_that_only_claims_the_audit_operation_makes_the_record_post_hoc(
    row: dict[str, Any],
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
) -> None:
    log_test_read(
        freeze_module.default_reads_log_path(root),
        "forged",
        row["fit_ids"],
        operation=freeze_module.RESIDUAL_AUDIT_OPERATION,
        details=row.get("details"),
    )
    body = _body(_write(tmp_path, target, root, policy_dir, mask_audit_report))
    assert body["post_hoc"] is True
    assert body["audit_reads"] == []
    assert [entry["record_hash"] for entry in body["invalidated_reads"]] == ["forged"]


def test_the_read_log_is_appended_under_a_lock_and_flushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_lock = freeze_module.FileLock
    real_fsync = os.fsync

    class Spy(real_lock):  # type: ignore[misc, valid-type]
        def acquire(self) -> None:
            super().acquire()
            events.append("acquire")

        def release(self) -> None:
            events.append("release")
            super().release()

    def fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(freeze_module, "FileLock", Spy)
    monkeypatch.setattr(freeze_module.os, "fsync", fsync)
    log = tmp_path / "logs" / "reads.jsonl"
    log_test_read(log, "hash", ["a"], operation="predict_test")
    assert events == ["acquire", "fsync", "release"]
    row = json.loads(log.read_text(encoding="utf-8"))
    assert (row["record_hash"], row["fit_ids"], row["operation"]) == (
        "hash",
        ["a"],
        "predict_test",
    )


def test_concurrent_readers_never_interleave_log_lines(tmp_path: Path) -> None:
    log = tmp_path / "reads.jsonl"
    failures: list[BaseException] = []

    def work(worker: int) -> None:
        try:
            for index in range(20):
                log_test_read(
                    log, f"hash-{worker}", [f"fit-{index}"], operation="predict_test"
                )
        except BaseException as error:  # noqa: BLE001 - surfaced below
            failures.append(error)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 120


def test_a_log_held_by_another_writer_is_refused_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze_module, "LOG_LOCK_TIMEOUT_SECONDS", 0.2)
    log = tmp_path / "reads.jsonl"
    holder = FileLock(log.with_name(f"{log.name}.lock"))
    holder.acquire()
    try:
        with pytest.raises(FreezeRecordError, match="locked by another writer"):
            log_test_read(log, "hash", ["a"], operation="predict_test")
    finally:
        holder.release()
    assert not log.exists()

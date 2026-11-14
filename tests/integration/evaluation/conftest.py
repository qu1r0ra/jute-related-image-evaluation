"""Frozen synthetic target and a stand-in repository tree for the evaluator tests."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    evaluate_masks_audit,
    freeze_target,
    make_synthetic_data,
    run_masks_fixture,
)

from ..test_policy_manifests import frozen_pool


@pytest.fixture(scope="module")
def target(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("gated")
    plan = make_synthetic_data(base / "images", per_class=20, size=64)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    freeze_target(
        base / "freeze",
        image_root=plan.root,
        rules_path=base / "rules.json",
        reviews_path=None,
        workers=2,
    )
    return base / "freeze" / "target"


@pytest.fixture(scope="module")
def policy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The frozen policy matrix for pools A, B and C, written by the real command."""
    base = tmp_path_factory.mktemp("policies")
    pools = [f"{name}={frozen_pool(base / f'pool-{name}')}" for name in "ABC"]
    arguments = ["freeze", "policies"]
    for pool in pools:
        arguments += ["--pool", pool]
    main(
        [
            *arguments,
            "--output",
            str(base / "matrix"),
            "--reserve-fraction",
            "0.2",
            "--train-fraction",
            "0.7",
            "--val-fraction",
            "0.15",
            "--test-fraction",
            "0.15",
            "--seed",
            "42",
        ]
    )
    return base / "matrix"


AUDIT_MANIFEST_FILE = "audit-manifest.json"


def audit_manifest_of(report: Path) -> Path:
    """The audit manifest written beside a mask-audit report fixture."""
    return report.with_name(AUDIT_MANIFEST_FILE)


def redigest(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute a report's digest so an edit is not caught by the digest alone."""
    body = {key: value for key, value in report.items() if key != "digest"}
    report["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


@pytest.fixture(scope="module")
def mask_audit_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A genuine (non-fixture) mask-audit evaluation report and its audit manifest.

    The fixture run supplies the sample and the ratings; the manifest is then
    re-evaluated as a real audit so the freeze gate accepts it as evidence. Its
    mask-quality gate does not pass, which is the point: the freeze record
    derives the outcome from the report, never from the operator.
    """
    output = tmp_path_factory.mktemp("masks") / "audit"
    run_masks_fixture(output)
    manifest = json.loads((output / "audit" / "audit.json").read_text("utf-8"))
    manifest["fixture_mode"] = False
    manifest.pop("fixture_note", None)
    report = evaluate_masks_audit(
        manifest,
        output / "audit" / "agent.csv",
        output / "audit" / "owner.csv",
        agent_rater={"name": "audit-agent", "version": "1"},
    )
    path = tmp_path_factory.mktemp("genuine-masks") / "evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    audit_manifest_of(path).write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def passing_mask_audit_report(
    mask_audit_report: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The genuine report with both gates passing by its numbers and its digest redone.

    The probes only report a headline when the audit passed, so the tests that
    exercise them need a report whose recomputed outcome is a pass. The bounds
    are edited rather than the flags, because the gate recomputes the flags.
    """
    report = json.loads(mask_audit_report.read_text(encoding="utf-8"))
    report["mask_quality"]["lower_95"] = 0.9
    report["mask_quality"]["fail_rate"]["upper_95"] = 0.01
    report["mask_quality"]["fail_rate"]["gate_pass"] = True
    report["scene_tag"]["lower_95"] = 0.9
    for gate in ("mask_quality", "scene_tag"):
        report[gate]["gate_pass"] = True
    path = tmp_path_factory.mktemp("passing-masks") / "evaluation.json"
    path.write_text(json.dumps(redigest(report)), encoding="utf-8")
    shutil.copyfile(audit_manifest_of(mask_audit_report), audit_manifest_of(path))
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "configs").mkdir()
    (tmp_path / "repo" / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "repo" / "configs" / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo" / "uv.lock").write_text("lock", encoding="utf-8")
    return tmp_path / "repo"

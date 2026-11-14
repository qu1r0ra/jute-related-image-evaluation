"""Frozen synthetic target and a stand-in repository tree for the evaluator tests."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import freeze_target, make_synthetic_data, run_masks_fixture

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


@pytest.fixture(scope="module")
def mask_audit_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real mask-audit evaluation report from the fixture run.

    The fixture's mask-quality gate does not pass, which is the point: the
    freeze record derives the outcome from the report, never from the operator.
    """
    output = tmp_path_factory.mktemp("masks") / "audit"
    run_masks_fixture(output)
    return output / "evaluation.json"


@pytest.fixture(scope="module")
def passing_mask_audit_report(
    mask_audit_report: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The fixture report with both gates passing and its digest recomputed.

    The probes only report a headline when the audit passed, so the tests that
    exercise them need a report whose derived outcome is a pass. The digest is
    recomputed so the report is accepted as untampered.
    """
    report = json.loads(mask_audit_report.read_text(encoding="utf-8"))
    for gate in ("mask_quality", "scene_tag"):
        report[gate]["gate_pass"] = True
    body = {key: value for key, value in report.items() if key != "digest"}
    report["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = tmp_path_factory.mktemp("passing-masks") / "evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "configs").mkdir()
    (tmp_path / "repo" / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "repo" / "configs" / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo" / "uv.lock").write_text("lock", encoding="utf-8")
    return tmp_path / "repo"

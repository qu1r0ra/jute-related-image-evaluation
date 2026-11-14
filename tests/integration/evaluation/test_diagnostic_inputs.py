"""Diagnostics inputs through the ``jute-disease`` CLI on the synthetic dataset.

The Grad-CAM sanity checks, the scene tags and the policy predictions are each
computed by a logged command and bound to the run record, so a hand-written
value is refused. The export is written atomically with a digest per output.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from PIL import Image

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    POLICIES,
    create_pipeline,
    freeze_target,
    load_manifest,
    make_synthetic_data,
)
from jute_disease.data.manifest import load_protected_test_manifest
from jute_disease.evaluation import default_reads_log_path
from jute_disease.models.dl import Classifier, TimmBackbone
from jute_disease.training import RunStore

from .conftest import audit_manifest_of
from .test_gated_evaluator import (
    SYSTEM,
    _cli,
    _issue_receipts,
    _outcomes,
    _spec,
    _write,
)

MODEL = "mobilenetv2_100.ra_in1k"
PRIMARY_POLICY = "image-random"
ASSIGNMENT_ROWS = [{"split": 0, "seed": k, "checkpoint": k % 3} for k in range(5)]


class Env(NamedTuple):
    """Everything one test needs: a frozen target, a store of 15 fits and records."""

    target: Path
    image_root: Path
    base: Path
    store: RunStore
    freeze: Path
    run_record: Path
    assignment: Path
    frozen_backbone: Path
    fit_ids: list[str]
    primary_ids: list[str]


def _fit_id(policy: str, k: int) -> str:
    return f"diag/{policy}/seed-{k}"


def _sorted_test_ids(target: Path) -> list[Any]:
    return sorted(
        load_protected_test_manifest(target).entries, key=lambda e: e.image_id
    )


def _model_predictions(
    target: Path, image_root: Path, checkpoint: Path, labels: list[str]
) -> list[str]:
    """What the checkpoint predicts for each sorted test image, the way it is read."""
    model = Classifier(TimmBackbone(MODEL, pretrained=False), len(labels))
    model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=True)
    model.eval()
    pipeline = create_pipeline(is_dl=True)
    predicted = []
    for entry in _sorted_test_ids(target):
        with Image.open(image_root / entry.path) as source:
            tensor = pipeline(source.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            predicted.append(labels[int(model(tensor).argmax(dim=1).item())])
    return predicted


def _complete(
    store: RunStore,
    fit_id: str,
    *,
    seed: int,
    policy: str,
    k: int,
    manifest: str,
    shared: Path,
    assignment_sha: str,
) -> None:
    run = store.start(fit_id, "identity")
    artifact = run.attempt_dir / "best.ckpt"
    try:
        os.link(shared, artifact)
    except OSError:
        shutil.copyfile(shared, artifact)
    run.complete(
        {
            "fit": {
                "fit_id": fit_id,
                "kind": "target",
                "upstream": None,
                "system": SYSTEM,
                "seed": seed,
                "pool": "A",
                "view": 1,
                "policy": policy,
                "split": 0,
                "seed_index": k,
            },
            "lineage": {"manifests": {"train": manifest}},
            "metrics": {"val_f1": 0.5 + 0.01 * k},
        },
        artifacts={"checkpoint": artifact},
        metadata={
            "assignment_table": {"sha256": assignment_sha},
            "runtime": {
                "cuda_available": True,
                "device": {"name": "NVIDIA A100 synthetic fixture"},
            },
        },
    )


def _build(tmp: Path, policy_dir: Path, audit_report: Path) -> Env:
    plan = make_synthetic_data(tmp / "images", per_class=60, size=64)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (tmp / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    freeze_target(
        tmp / "freeze",
        image_root=plan.root,
        rules_path=tmp / "rules.json",
        reviews_path=None,
        workers=2,
    )
    target = tmp / "freeze" / "target"
    labels = list(load_manifest(target).labels)

    torch.manual_seed(0)
    classifier = Classifier(TimmBackbone(MODEL, pretrained=False), len(labels))
    shared = tmp / "shared.ckpt"
    torch.save({"state_dict": classifier.state_dict()}, shared)
    frozen = tmp / "frozen-backbone.pt"
    torch.save(TimmBackbone(MODEL, pretrained=False).state_dict(), frozen)
    assignment = _write(tmp / "assignments.json", ASSIGNMENT_ROWS)
    assignment_sha = hashlib.sha256(assignment.read_bytes()).hexdigest()

    store = RunStore(tmp / "store")
    manifest = load_manifest(target).digest()
    entries = _sorted_test_ids(target)
    ids = [e.image_id for e in entries]
    truth = [e.label for e in entries]
    primary = _model_predictions(target, plan.root, shared, labels)
    runs: list[dict[str, object]] = []
    fit_ids: list[str] = []
    for policy in POLICIES:
        if policy == PRIMARY_POLICY:
            predicted = primary
        else:
            shift = 1 if policy == "exact-copy" else 2
            predicted = [
                labels[(labels.index(p) + shift) % len(labels)] if i % 5 == 0 else p
                for i, p in enumerate(primary)
            ]
        for k in range(5):
            fit_id = _fit_id(policy, k)
            _complete(
                store,
                fit_id,
                seed=10 + k,
                policy=policy,
                k=k,
                manifest=manifest,
                shared=shared,
                assignment_sha=assignment_sha,
            )
            _write(
                store.run_dir(fit_id) / "predictions.json",
                {"image_ids": ids, "predictions": predicted},
            )
            runs.append(
                {
                    "fit_id": fit_id,
                    "system": SYSTEM,
                    "seed": 10 + k,
                    "predictions": f"{fit_id}/predictions.json",
                }
            )
            fit_ids.append(fit_id)
    del truth

    stand_in = _stand_in_root(tmp / "freeze-root")
    probes = io.StringIO()
    with contextlib.redirect_stdout(probes):
        _cli("evaluate", "probe-plan")
    spec = _spec(tmp, fit_ids, json.loads(probes.getvalue()))
    freeze = tmp / "freeze-record.json"
    _cli(
        "evaluate",
        "freeze-record",
        "--spec",
        spec,
        "--outcomes",
        _outcomes(tmp),
        "--output",
        freeze,
        "--target-dir",
        target,
        "--root",
        stand_in,
        "--policy-dir",
        policy_dir,
        "--mask-audit-report",
        audit_report,
        "--mask-audit-manifest",
        audit_manifest_of(audit_report),
    )
    _issue_receipts(store, freeze)
    run_record = tmp / "run-record.json"
    _write(tmp / "runs.json", runs)
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        tmp / "runs.json",
        "--run-store",
        store.root,
        "--output",
        run_record,
    )
    return Env(
        target=target,
        image_root=plan.root,
        base=store.root,
        store=store,
        freeze=freeze,
        run_record=run_record,
        assignment=assignment,
        frozen_backbone=frozen,
        fit_ids=fit_ids,
        primary_ids=[_fit_id(PRIMARY_POLICY, k) for k in range(5)],
    )


def _stand_in_root(path: Path) -> Path:
    (path / "src").mkdir(parents=True)
    (path / "configs").mkdir()
    (path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (path / "configs" / "c.json").write_text("{}", encoding="utf-8")
    (path / "uv.lock").write_text("lock", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def env(
    tmp_path_factory: pytest.TempPathFactory,
    policy_dir: Path,
    passing_mask_audit_report: Path,
) -> Env:
    return _build(
        tmp_path_factory.mktemp("diag"), policy_dir, passing_mask_audit_report
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh stand-in repository, so each test starts with an empty ledger.

    The pipeline hash covers only file content, so a record frozen against the
    module's stand-in still verifies here.
    """
    return _stand_in_root(tmp_path / "repo")


def _ledger(repo: Path) -> list[dict[str, Any]]:
    path = default_reads_log_path(repo)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _operations(repo: Path) -> list[str]:
    return [row["operation"] for row in _ledger(repo)]


def _sanity_arguments(env: Env, repo: Path, output: Path) -> list[object]:
    arguments: list[object] = [
        "evaluate",
        "sanity-checks",
        "--freeze-record",
        env.freeze,
        "--run-record",
        env.run_record,
        "--target-dir",
        env.target,
        "--image-root",
        env.image_root,
        "--assignment-table",
        env.assignment,
        "--run-store",
        env.base,
        "--root",
        repo,
        "--output",
        output,
    ]
    for fit_id in env.primary_ids:
        arguments += ["--fit-id", fit_id]
    return arguments


@contextlib.contextmanager
def _captured() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def test_sanity_checks_are_computed_and_logged(
    env: Env, repo: Path, tmp_path: Path
) -> None:
    output = tmp_path / "sanity-checks.json"
    _cli(*_sanity_arguments(env, repo, output))

    receipt = json.loads(output.read_text(encoding="utf-8"))
    body = receipt["body"]
    selected = body["selected_fit_id"]
    assert selected in env.primary_ids
    assert body["parameter_randomization"]["status"] in ("pass", "fail")
    assert body["label_randomization"]["status"] in ("pass", "fail")
    assert body["pass_rule"]["fixed_in_code"] is True
    # The status follows from the computed correlations, not from an input.
    stages = body["parameter_randomization"]["evidence"]["stage_median_correlation"]
    assert stages
    ledger = _ledger(repo)
    operations = [row["operation"] for row in ledger]
    assert operations == ["sanity_checks", "sanity_checks_output"]
    assert (
        ledger[1]["details"]["output_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert ledger[0]["details"]["checkpoint_sha256"]
    assert ledger[0]["fit_ids"] == env.primary_ids


def _scene_arguments(env: Env, repo: Path, output: Path) -> list[object]:
    return [
        "evaluate",
        "scene-tags",
        "--freeze-record",
        env.freeze,
        "--target-dir",
        env.target,
        "--image-root",
        env.image_root,
        "--root",
        repo,
        "--output",
        output,
    ]


def test_scene_tags_cover_every_test_image_and_bind_the_audit(
    env: Env, repo: Path, tmp_path: Path, passing_mask_audit_report: Path
) -> None:
    output = tmp_path / "scene-tags.json"
    _cli(*_scene_arguments(env, repo, output))

    body = json.loads(output.read_text(encoding="utf-8"))["body"]
    test_ids = [e.image_id for e in _sorted_test_ids(env.target)]
    assert sorted(body["scenes"]) == test_ids
    assert sorted(body["frame_leaf_share"]) == test_ids
    assert all(0.0 <= share <= 1.0 for share in body["frame_leaf_share"].values())
    audit = audit_manifest_of(passing_mask_audit_report)
    manifest = json.loads(audit.read_text(encoding="utf-8"))
    assert (
        body["audit_manifest_sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()
    )
    assert (
        body["thresholds_sha256"]
        == hashlib.sha256(
            json.dumps(
                manifest["scene_thresholds"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    )
    assert body["mask_backend"] == manifest["mask_backend"]
    ledger = _ledger(repo)
    assert [row["operation"] for row in ledger] == ["scene_tags", "scene_tags_output"]
    assert (
        ledger[0]["details"]["audit_manifest_sha256"] == body["audit_manifest_sha256"]
    )
    assert (
        ledger[1]["details"]["output_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )


class Staged(NamedTuple):
    """A repository whose ledger holds one genuine run of both logged commands."""

    root: Path
    scene_tags: Path
    sanity_checks: Path


@pytest.fixture(scope="module")
def staged(env: Env, tmp_path_factory: pytest.TempPathFactory) -> Staged:
    base = tmp_path_factory.mktemp("staged")
    root = _stand_in_root(base / "repo")
    scene_tags = base / "scene-tags.json"
    sanity_checks = base / "sanity-checks.json"
    with _captured():
        _cli(*_scene_arguments(env, root, scene_tags))
        _cli(*_sanity_arguments(env, root, sanity_checks))
    return Staged(root, scene_tags, sanity_checks)


@pytest.fixture
def staged_repo(staged: Staged, tmp_path: Path) -> Path:
    """A private copy of the staged repository, so each test owns its ledger."""
    root = tmp_path / "staged-repo"
    shutil.copytree(staged.root, root)
    return root


def _diagnostics_arguments(
    env: Env,
    root: Path,
    scene_tags: Path,
    sanity_checks: Path,
    output: Path,
    fit_ids: list[str] | None = None,
) -> list[object]:
    arguments: list[object] = [
        "evaluate",
        "diagnostics",
        "--freeze-record",
        env.freeze,
        "--run-record",
        env.run_record,
        "--target-dir",
        env.target,
        "--image-root",
        env.image_root,
        "--assignment-table",
        env.assignment,
        "--frozen-backbone-checkpoint",
        env.frozen_backbone,
        "--scene-tags",
        scene_tags,
        "--sanity-checks",
        sanity_checks,
        "--run-store",
        env.base,
        "--root",
        root,
        "--output",
        output,
    ]
    for fit_id in fit_ids or env.fit_ids:
        arguments += ["--fit-id", fit_id]
    return arguments


def _hand_written(path: Path, genuine: Path, edit: Any = None) -> Path:
    """A self-consistent file nobody computed: valid hash, no ledger row."""
    body = json.loads(genuine.read_text(encoding="utf-8"))["body"]
    if edit is not None:
        edit(body)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _write(path, {"hash": hashlib.sha256(canonical).hexdigest(), "body": body})


def test_diagnostics_refuse_hand_written_sanity_checks(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    def always_pass(body: dict[str, Any]) -> None:
        body["parameter_randomization"]["status"] = "pass"
        body["label_randomization"]["status"] = "pass"

    forged = _hand_written(tmp_path / "forged.json", staged.sanity_checks, always_pass)
    output = tmp_path / "diagnostics"
    with pytest.raises(SystemExit, match="sanity checks were not computed"):
        _cli(
            *_diagnostics_arguments(env, staged_repo, staged.scene_tags, forged, output)
        )
    assert not output.exists()
    assert "image_diagnostics" not in _operations(staged_repo)

    # A genuine file edited after the command wrote it is refused the same way.
    edited = tmp_path / "edited.json"
    payload = json.loads(staged.sanity_checks.read_text(encoding="utf-8"))
    payload["body"]["pass_rule"]["max_median_correlation"] = 1.0
    _write(edited, payload)
    with pytest.raises(SystemExit, match="sanity checks"):
        _cli(
            *_diagnostics_arguments(env, staged_repo, staged.scene_tags, edited, output)
        )
    assert "image_diagnostics" not in _operations(staged_repo)


def test_diagnostics_and_test_refuse_hand_written_scene_tags(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    def all_leaf(body: dict[str, Any]) -> None:
        body["scenes"] = dict.fromkeys(body["scenes"], "leaf-on-plain-background")

    forged = _hand_written(tmp_path / "forged-scenes.json", staged.scene_tags, all_leaf)
    output = tmp_path / "diagnostics"
    with pytest.raises(SystemExit, match="scene tags were not computed"):
        _cli(
            *_diagnostics_arguments(
                env, staged_repo, forged, staged.sanity_checks, output
            )
        )
    assert not output.exists()
    assert "image_diagnostics" not in _operations(staged_repo)

    arguments: list[object] = [
        "evaluate",
        "test",
        "--freeze-record",
        env.freeze,
        "--run-record",
        env.run_record,
        "--target-dir",
        env.target,
        "--run-store",
        env.base,
        "--root",
        staged_repo,
        "--output",
        tmp_path / "report.json",
        "--scene-tags",
        forged,
    ]
    for fit_id in env.primary_ids:
        arguments += ["--fit-id", fit_id]
    with pytest.raises(SystemExit, match="scene tags were not computed"):
        _cli(*arguments)
    assert "evaluate_test" not in _operations(staged_repo)


def _run_diagnostics(
    env: Env, staged: Staged, root: Path, output: Path, **overrides: Any
) -> dict[str, Any]:
    arguments = _diagnostics_arguments(
        env,
        root,
        overrides.get("scene_tags", staged.scene_tags),
        overrides.get("sanity_checks", staged.sanity_checks),
        output,
        overrides.get("fit_ids"),
    )
    with _captured():
        _cli(*arguments)
    return json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))


def _policy_cell_predictions(env: Env, policy: str, k: int) -> dict[str, str]:
    """Read one run's predictions straight from the store, for an independent check."""
    path = env.base / _fit_id(policy, k) / "predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(zip(payload["image_ids"], payload["predictions"], strict=True))


def test_diagnostics_read_policy_predictions_and_compute_disagreement_inclusion(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    export = _run_diagnostics(env, staged, staged_repo, tmp_path / "diagnostics")

    # The median-validation run of each policy is seed 2 (val_f1 = 0.5 + 0.01 * k).
    by_policy = [_policy_cell_predictions(env, policy, 2) for policy in POLICIES]
    expected = sorted(
        image_id
        for image_id in by_policy[0]
        if len({predictions[image_id] for predictions in by_policy}) > 1
    )
    sample = export["diagnostics"]["gradcam_sample"]
    assert expected, "the fixture must contain policy disagreements"
    assert sorted(sample["selected"]["disagreement"]) == expected
    assert sample["policy_disagreement_count"] == len(expected)
    assert sample["all_policy_disagreements_included"] is True
    kept = {i for ids in sample["selected"].values() for i in ids}
    assert set(expected) <= kept
    record = json.loads(env.freeze.read_text(encoding="utf-8"))["body"]
    assert export["diagnostics"]["scope"]["pool"] == record["diagnostic"]["pool"]
    ledger = _ledger(staged_repo)
    assert [row["operation"] for row in ledger[-2:]] == [
        "image_diagnostics",
        "image_diagnostics_output",
    ]


def test_diagnostics_refuse_policy_predictions_changed_since_the_run_record(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    path = env.base / _fit_id("exact-copy", 2) / "predictions.json"
    original = path.read_bytes()
    payload = json.loads(original)
    payload["predictions"] = list(reversed(payload["predictions"]))
    output = tmp_path / "diagnostics"
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="predictions"):
            _cli(
                *_diagnostics_arguments(
                    env, staged_repo, staged.scene_tags, staged.sanity_checks, output
                )
            )
    finally:
        path.write_bytes(original)
    assert not output.exists()


def test_diagnostics_refuse_a_run_set_missing_a_policy(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    output = tmp_path / "diagnostics"
    with pytest.raises(SystemExit, match="test read refused"):
        _cli(
            *_diagnostics_arguments(
                env,
                staged_repo,
                staged.scene_tags,
                staged.sanity_checks,
                output,
                fit_ids=env.primary_ids,
            )
        )
    assert not output.exists()
    assert "image_diagnostics" not in _operations(staged_repo)


def test_diagnostics_no_longer_accept_hand_supplied_context(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    arguments = _diagnostics_arguments(
        env, staged_repo, staged.scene_tags, staged.sanity_checks, tmp_path / "out"
    )
    with pytest.raises(SystemExit) as raised:
        with contextlib.redirect_stderr(io.StringIO()):
            _cli(*arguments, "--context", tmp_path / "context.json")
    assert raised.value.code == 2


def test_diagnostics_export_lists_a_sha256_for_every_output(
    env: Env, staged: Staged, staged_repo: Path, tmp_path: Path
) -> None:
    output = tmp_path / "diagnostics"
    _run_diagnostics(env, staged, staged_repo, output)

    manifest = json.loads((output / "output-manifest.json").read_text(encoding="utf-8"))
    on_disk = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "output-manifest.json"
    }
    assert set(manifest["files"]) == on_disk
    assert any(name.startswith("gradcam/") for name in on_disk)
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    completion = _ledger(staged_repo)[-1]
    assert completion["operation"] == "image_diagnostics_output"
    assert (
        completion["details"]["output_manifest_sha256"]
        == hashlib.sha256((output / "output-manifest.json").read_bytes()).hexdigest()
    )


def test_a_crash_during_the_export_leaves_no_complete_looking_directory(
    env: Env,
    staged: Staged,
    staged_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exports" / "diagnostics"
    real_replace = os.replace

    def crash_on_publish(source: Any, destination: Any) -> None:
        if Path(destination) == output:
            raise OSError("simulated crash while publishing the export")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", crash_on_publish)
    with pytest.raises(SystemExit, match="simulated crash"):
        _cli(
            *_diagnostics_arguments(
                env, staged_repo, staged.scene_tags, staged.sanity_checks, output
            )
        )
    assert not output.exists()
    assert not list(output.parent.glob("*partial*"))
    assert "image_diagnostics_output" not in _operations(staged_repo)

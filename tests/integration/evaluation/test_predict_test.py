"""``evaluate predict-test`` through the CLI on real tiny fits of the synthetic target."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG, CURATION_RULES_PATH
from jute_disease.data import freeze_target, load_manifest, make_synthetic_data
from jute_disease.data.manifest import load_protected_test_manifest
from jute_disease.evaluation import gated_predictions as predict_module
from jute_disease.training import (
    DataInputs,
    FitKind,
    FitSpec,
    Recipe,
    RunStore,
    fit_classical,
    fit_cnn,
    load_benchmark,
)

from .test_gated_evaluator import _cli, _freeze, _write

DEEP = FitSpec(
    "cnn/tiny/seed-42",
    FitKind.TARGET,
    42,
    system="tiny",
    model="resnet18",
    micro_batch=8,
    accumulation=1,
)
CLASSICAL = FitSpec(
    "classical/lr-raw", FitKind.CLASSICAL, 42, estimator="lr", features="raw"
)
FOREIGN = FitSpec(
    "classical/lr-foreign", FitKind.CLASSICAL, 42, estimator="lr", features="raw"
)
OUTSIDE = "classical/lr-outside-the-matrix"
FITS = {"deep": DEEP, "classical": CLASSICAL}


@pytest.fixture(scope="module")
def image_root(target: Path) -> Path:
    return target.parents[1] / "images"


def _recipe() -> Recipe:
    return replace(
        load_benchmark(BENCHMARK_CONFIG).recipe,
        image_size=64,
        max_epochs=1,
        precision="32-true",
        num_workers=0,
    )


def _train_classical(
    spec: FitSpec, store: RunStore, target: Path, image_root: Path
) -> None:
    """Train and record one real classical fit of the synthetic target."""
    run = store.start(spec.fit_id, "synthetic identity")
    result = fit_classical(
        spec,
        _recipe(),
        DataInputs(image_root=image_root, target_dir=target, sources={}),
        run.attempt_dir,
        run.attempt_dir.parent / "feature-cache",
    )
    run.complete(result.results, result.artifacts)


@pytest.fixture(scope="module")
def foreign(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A store holding a classical fit trained on another frozen target."""
    base = tmp_path_factory.mktemp("foreign")
    plan = make_synthetic_data(base / "images", per_class=24, size=64)
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
    store = RunStore(base / "store")
    _train_classical(FOREIGN, store, base / "freeze" / "target", plan.root)
    return store.root


@pytest.fixture(scope="module")
def retrained(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The same fit ID retrained with another seed, so its receipt and model differ."""
    store = RunStore(tmp_path_factory.mktemp("retrained"))
    _train_classical(replace(CLASSICAL, seed=7), store, target, image_root)
    return store.root


@pytest.fixture(scope="module")
def trained(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A run store holding one real deep fit and one real classical fit."""
    recipe = _recipe()
    data = DataInputs(image_root=image_root, target_dir=target, sources={})
    store = RunStore(tmp_path_factory.mktemp("trained"))
    deep = store.start(DEEP.fit_id, "synthetic identity")
    result = fit_cnn(DEEP, recipe, data, deep.attempt_dir, pretrained=False)
    deep.complete(result.results, result.artifacts)
    classical = store.start(CLASSICAL.fit_id, "synthetic identity")
    result = fit_classical(
        CLASSICAL,
        recipe,
        data,
        classical.attempt_dir,
        classical.attempt_dir.parent / "feature-cache",
    )
    classical.complete(result.results, result.artifacts)
    return store.root


@pytest.fixture
def store(trained: Path, tmp_path: Path) -> Path:
    """A private copy of the trained store, so each test starts without predictions."""
    copy = tmp_path / "store"
    shutil.copytree(trained, copy)
    return copy


@pytest.fixture
def freeze(
    tmp_path: Path,
    target: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> Path:
    return _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [DEEP.fit_id, CLASSICAL.fit_id],
    )


def _predict(
    capsys: pytest.CaptureFixture[str],
    freeze: Path | None,
    fit_id: str,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    *extra: object,
) -> dict[str, object]:
    capsys.readouterr()
    arguments: list[object] = [
        "evaluate",
        "predict-test",
        "--fit-id",
        fit_id,
        "--run-store",
        store,
        "--target-dir",
        target,
        "--image-root",
        image_root,
        "--root",
        root,
    ]
    if freeze is not None:
        arguments += ["--freeze-record", freeze]
    _cli(*arguments, *extra)
    return json.loads(capsys.readouterr().out)


def _rows(root: Path) -> list[dict[str, object]]:
    log = root / "artifacts/evaluation/test-reads.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("kind", ["deep", "classical"])
def test_predict_test_authorises_logs_and_writes_digested_predictions(
    kind: str,
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit_id = FITS[kind].fit_id
    result = _predict(capsys, freeze, fit_id, store, target, image_root, root)

    record_hash = json.loads(freeze.read_text(encoding="utf-8"))["hash"]
    assert result["freeze_record_hash"] == record_hash
    rows = [row for row in _rows(root) if row["operation"] == "predict_test"]
    assert len(rows) == 1
    assert rows[0]["record_hash"] == record_hash
    assert rows[0]["fit_ids"] == [fit_id]

    path = store / result["predictions"]
    assert path == store / fit_id / "predictions.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["predictions_sha256"] == digest
    sidecar = path.with_name("predictions.json.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{digest}  predictions.json\n"

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = sorted(e.image_id for e in load_protected_test_manifest(target).entries)
    labels = {entry.label for entry in load_manifest(target).entries}
    assert payload["image_ids"] == expected
    assert result["images"] == len(expected)
    assert len(payload["predictions"]) == len(expected)
    assert set(payload["predictions"]) <= labels

    runs = _write(
        tmp_path / "runs.json",
        [
            {
                "fit_id": fit_id,
                "system": FITS[kind].system or "classical",
                "seed": 42,
                "predictions": f"{fit_id}/predictions.json",
            }
        ],
    )
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        runs,
        "--run-store",
        store,
        "--output",
        tmp_path / "run-record.json",
    )
    entry = json.loads((tmp_path / "run-record.json").read_text())["body"]["fits"][
        fit_id
    ]
    assert entry["predictions_sha256"] == digest


def test_a_custom_reads_log_never_replaces_the_project_ledger(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    custom = tmp_path / "elsewhere" / "reads.jsonl"
    _predict(
        capsys,
        freeze,
        CLASSICAL.fit_id,
        store,
        target,
        image_root,
        root,
        "--reads-log",
        custom,
    )
    assert len(_rows(root)) == 1
    assert len(custom.read_text().splitlines()) == 1

    later = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [DEEP.fit_id, CLASSICAL.fit_id],
        name="later-freeze.json",
    )
    assert json.loads(later.read_text(encoding="utf-8"))["body"]["post_hoc"] is True


def _run_record(
    freeze: Path, store: Path, fit_id: str, tmp_path: Path, name: str = "rr.json"
) -> None:
    system = DEEP.system if fit_id == DEEP.fit_id else "classical"
    runs = _write(
        tmp_path / f"{name}.runs.json",
        [
            {
                "fit_id": fit_id,
                "system": system,
                "seed": 42,
                "predictions": f"{fit_id}/predictions.json",
            }
        ],
    )
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        runs,
        "--run-store",
        store,
        "--output",
        tmp_path / name,
    )


def test_handwritten_predictions_cannot_enter_a_run_record(
    freeze: Path, store: Path, tmp_path: Path
) -> None:
    _write(
        store / DEEP.fit_id / "predictions.json",
        {"image_ids": [], "predictions": []},
    )
    with pytest.raises(SystemExit, match="no prediction receipt"):
        _run_record(freeze, store, DEEP.fit_id, tmp_path)
    assert not (tmp_path / "rr.json").exists()


def test_swapped_predictions_cannot_enter_a_run_record(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit_id = CLASSICAL.fit_id
    _predict(capsys, freeze, fit_id, store, target, image_root, root)
    path = store / fit_id / "predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["predictions"] = payload["predictions"][::-1]
    _write(path, payload)
    with pytest.raises(SystemExit, match="not the ones predict-test wrote"):
        _run_record(freeze, store, fit_id, tmp_path)


def test_predictions_from_another_freeze_record_are_not_accepted(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit_id = CLASSICAL.fit_id
    _predict(capsys, freeze, fit_id, store, target, image_root, root)
    later = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [DEEP.fit_id, CLASSICAL.fit_id],
        name="later-freeze.json",
    )
    with pytest.raises(SystemExit, match="no prediction receipt"):
        _run_record(later, store, fit_id, tmp_path)


def test_a_fit_trained_on_another_manifest_is_refused_before_any_read(
    tmp_path: Path,
    target: Path,
    image_root: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    foreign: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze = _freeze(
        tmp_path,
        target,
        root,
        policy_dir,
        mask_audit_report,
        capsys,
        [FOREIGN.fit_id],
    )
    with pytest.raises(SystemExit, match="not trained on the manifest"):
        _predict(capsys, freeze, FOREIGN.fit_id, foreign, target, image_root, root)
    assert _rows(root) == []
    assert not (foreign / FOREIGN.fit_id / "predictions.json").exists()


def test_a_fit_outside_the_frozen_matrix_is_refused_before_any_read(
    tmp_path: Path,
    target: Path,
    image_root: Path,
    root: Path,
    policy_dir: Path,
    mask_audit_report: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze = _freeze(
        tmp_path, target, root, policy_dir, mask_audit_report, capsys, [OUTSIDE]
    )
    store = RunStore(tmp_path / "outside-store")
    run = store.start(OUTSIDE, "synthetic identity")
    model = run.attempt_dir / "model.joblib"
    model.write_bytes(b"synthetic model")
    run.complete(
        {
            "fit": OUTSIDE,
            "seed": 42,
            "pool": "Z",
            "view": 1,
            "policy": "near-duplicate",
            "split": 0,
            "recipe": {"image_size": 64},
            "features": "raw",
            "manifest": "c" * 64,
        },
        {"model": model},
    )
    with pytest.raises(SystemExit, match="outside the frozen matrix"):
        _predict(capsys, freeze, OUTSIDE, store.root, target, image_root, root)
    assert _rows(root) == []


def test_a_repeat_with_a_different_training_receipt_is_refused(
    freeze: Path,
    store: Path,
    retrained: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit_id = CLASSICAL.fit_id
    _predict(capsys, freeze, fit_id, store, target, image_root, root)
    row = _rows(root)[0]
    receipt = hashlib.sha256((store / fit_id / "receipt.json").read_bytes()).hexdigest()
    assert row["details"]["training_receipt_sha256"] == receipt
    assert row["details"]["artifact_kind"] == "model"
    assert len(row["details"]["artifact_sha256"]) == 64

    with pytest.raises(SystemExit, match="already read under this freeze record"):
        _predict(capsys, freeze, fit_id, retrained, target, image_root, root)
    assert len(_rows(root)) == 1


def test_a_repeat_read_is_logged_again_and_writes_the_same_predictions(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fit_id = CLASSICAL.fit_id
    first = _predict(capsys, freeze, fit_id, store, target, image_root, root)
    second = _predict(capsys, freeze, fit_id, store, target, image_root, root)
    assert first["predictions_sha256"] == second["predictions_sha256"]
    assert len(_rows(root)) == 2

    path = store / fit_id / "predictions.json"
    path.write_text('{"image_ids":[],"predictions":[]}', encoding="utf-8")
    with pytest.raises(SystemExit, match="already exist and differ"):
        _predict(capsys, freeze, fit_id, store, target, image_root, root)


def test_predict_test_is_refused_without_a_frozen_record(
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for freeze in (None, tmp_path / "missing.json"):
        with pytest.raises(SystemExit, match="no frozen selection record"):
            _predict(capsys, freeze, CLASSICAL.fit_id, store, target, image_root, root)
    assert _rows(root) == []
    assert not (store / CLASSICAL.fit_id / "predictions.json").exists()


def test_predict_test_is_refused_for_a_fit_the_record_does_not_list(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="not in the freeze record"):
        _predict(capsys, freeze, "cnn/unfrozen/seed-1", store, target, image_root, root)
    assert _rows(root) == []


def test_predict_test_is_refused_without_a_verified_receipt(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (store / DEEP.fit_id / "receipt.json").unlink()
    with pytest.raises(SystemExit, match="no verified training receipt"):
        _predict(capsys, freeze, DEEP.fit_id, store, target, image_root, root)

    classical = RunStore(store).receipt(CLASSICAL.fit_id)
    model = store / CLASSICAL.fit_id / classical["artifacts"]["model"]["path"]
    model.write_bytes(b"swapped after the receipt")
    with pytest.raises(SystemExit, match="receipt"):
        _predict(capsys, freeze, CLASSICAL.fit_id, store, target, image_root, root)
    assert _rows(root) == []


def test_the_read_is_on_disk_before_any_test_image_is_read(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    original = predict_module._classical_labels

    def spy(*args: object, **kwargs: object) -> tuple[list[str], list[str]]:
        seen.append(len(_rows(root)))
        return original(*args, **kwargs)

    monkeypatch.setattr(predict_module, "_classical_labels", spy)
    _predict(capsys, freeze, CLASSICAL.fit_id, store, target, image_root, root)
    assert seen == [1]

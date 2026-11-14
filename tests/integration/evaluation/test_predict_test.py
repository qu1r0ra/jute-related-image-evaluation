"""``evaluate predict-test`` through the CLI on real tiny fits of the synthetic target."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import load_manifest
from jute_disease.data.manifest import load_protected_test_manifest
from jute_disease.evaluation import gated_predictions as predict_module
from jute_disease.training import (
    DataInputs,
    FitKind,
    FitSpec,
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
FITS = {"deep": DEEP, "classical": CLASSICAL}


@pytest.fixture(scope="module")
def image_root(target: Path) -> Path:
    return target.parents[1] / "images"


@pytest.fixture(scope="module")
def trained(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A run store holding one real deep fit and one real classical fit."""
    recipe = replace(
        load_benchmark(BENCHMARK_CONFIG).recipe,
        image_size=64,
        max_epochs=1,
        precision="32-true",
        num_workers=0,
    )
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
    _cli(*arguments)
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

"""``evaluate predict-leaf-only``: the only producer of leaf-only test predictions."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from jute_disease.data import ImageWithMask
from jute_disease.evaluation.gated_predictions import LeafOnlyTransform

from .test_gated_evaluator import _cli, _freeze, _write
from .test_predict_test import (  # noqa: F401
    CLASSICAL,
    DEEP,
    OUTSIDE,
    _predict,
    _rows,
    image_root,
    trained,
)

LEAF = "classical/lr-raw-leaf"
OTHER_LEAF = "classical/lr-raw-leaf-other"
SAMPLE = 12
PAD_FILL = (124, 116, 104)


@pytest.fixture
def store(trained: Path, tmp_path: Path) -> Path:  # noqa: F811
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
        [DEEP.fit_id, CLASSICAL.fit_id, LEAF, OTHER_LEAF],
    )


def _predict_leaf(
    capsys: pytest.CaptureFixture[str],
    freeze: Path,
    fit_id: str,
    of: str,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    *extra: object,
) -> dict[str, object]:
    capsys.readouterr()
    _cli(
        "evaluate",
        "predict-leaf-only",
        "--freeze-record",
        freeze,
        "--fit-id",
        fit_id,
        "--of",
        of,
        "--run-store",
        store,
        "--target-dir",
        target,
        "--image-root",
        image_root,
        "--root",
        root,
        "--sample-size",
        SAMPLE,
        *extra,
    )
    return json.loads(capsys.readouterr().out)


def _runs(leaf: str = LEAF, of: str = CLASSICAL.fit_id) -> list[dict[str, object]]:
    source = DEEP if of == DEEP.fit_id else CLASSICAL
    return [
        {
            "fit_id": source.fit_id,
            "system": source.system,
            "seed": 42,
            "predictions": f"{source.fit_id}/predictions.json",
        },
        {
            "fit_id": leaf,
            "system": source.system,
            "seed": 42,
            "variant": "leaf_only",
            "predictions": f"{leaf}/predictions.json",
            "of": of,
        },
    ]


def _run_record(
    freeze: Path, store: Path, tmp_path: Path, runs: list[dict[str, object]]
) -> Path:
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        freeze,
        "--runs",
        _write(tmp_path / "runs.json", runs),
        "--run-store",
        store,
        "--output",
        tmp_path / "run-record.json",
    )
    return tmp_path / "run-record.json"


def _full_prediction(
    capsys: pytest.CaptureFixture[str],
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    *fit_ids: str,
) -> None:
    for fit_id in fit_ids:
        _predict(capsys, freeze, fit_id, store, target, image_root, root)


def _sample(
    capsys: pytest.CaptureFixture[str], freeze: Path, target: Path, root: Path
) -> list[str]:
    capsys.readouterr()
    _cli(
        "evaluate",
        "leaf-sample",
        "--freeze-record",
        freeze,
        "--target-dir",
        target,
        "--root",
        root,
        "--sample-size",
        SAMPLE,
    )
    return json.loads(capsys.readouterr().out)


def test_the_producer_writes_the_leaf_sample_and_a_receipt_a_run_record_accepts(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(capsys, freeze, store, target, image_root, root, CLASSICAL.fit_id)
    result = _predict_leaf(
        capsys, freeze, LEAF, CLASSICAL.fit_id, store, target, image_root, root
    )

    path = store / str(result["predictions"])
    assert path == store / LEAF / "predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["image_ids"] == sorted(_sample(capsys, freeze, target, root))
    assert len(payload["predictions"]) == SAMPLE
    assert result["predictions_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    rows = [row for row in _rows(root) if row["operation"] == "predict_leaf_only"]
    assert [row["fit_ids"] for row in rows] == [[LEAF]]

    accepted = _run_record(freeze, store, tmp_path, _runs())
    entry = json.loads(accepted.read_text(encoding="utf-8"))["body"]["fits"][LEAF]
    assert entry["of"] == CLASSICAL.fit_id
    assert entry["leaf_receipt_sha256"] == result["leaf_receipt_sha256"]
    assert entry["predictions_sha256"] == result["predictions_sha256"]


def test_a_deep_fit_predicts_the_leaf_sample_from_its_own_checkpoint(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(capsys, freeze, store, target, image_root, root, DEEP.fit_id)
    result = _predict_leaf(
        capsys, freeze, LEAF, DEEP.fit_id, store, target, image_root, root
    )
    payload = json.loads((store / LEAF / "predictions.json").read_text("utf-8"))
    assert payload["image_ids"] == sorted(_sample(capsys, freeze, target, root))
    accepted = _run_record(freeze, store, tmp_path, _runs(LEAF, DEEP.fit_id))
    entry = json.loads(accepted.read_text(encoding="utf-8"))["body"]["fits"][LEAF]
    assert entry["leaf_receipt_sha256"] == result["leaf_receipt_sha256"]


def _picture(*, is_dl: bool) -> object:
    image = Image.new("RGB", (80, 60), (240, 240, 240))
    ImageDraw.Draw(image).ellipse((25, 15, 55, 45), fill=(30, 160, 30))
    return LeafOnlyTransform(64, is_dl=is_dl)(image)


def test_pixels_outside_the_leaf_mask_become_the_letterbox_fill() -> None:
    blanked = _picture(is_dl=False)
    assert isinstance(blanked, ImageWithMask)
    assert tuple(blanked.pixels[0, 0]) == PAD_FILL  # letterbox padding
    assert tuple(blanked.pixels[8, 0]) == PAD_FILL  # background inside the image
    red, green, blue = (int(v) for v in blanked.pixels[32, 32])
    assert green > red and green > blue  # leaf kept
    assert blanked.valid_mask[8, 0] and not blanked.valid_mask[0, 0]


def test_the_deep_view_of_a_blanked_image_normalises_the_fill_to_zero() -> None:
    tensor = _picture(is_dl=True)
    assert tuple(tensor.shape) == (3, 64, 64)
    assert float(tensor[:, 8, 0].abs().max()) < 0.05
    assert float(tensor[1, 32, 32]) > 0.5


def test_handwritten_leaf_only_predictions_are_refused(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(capsys, freeze, store, target, image_root, root, CLASSICAL.fit_id)
    ids = sorted(_sample(capsys, freeze, target, root))
    full = json.loads(
        (store / CLASSICAL.fit_id / "predictions.json").read_text(encoding="utf-8")
    )
    by_id = dict(zip(full["image_ids"], full["predictions"], strict=True))
    _write(
        store / LEAF / "predictions.json",
        {"image_ids": ids, "predictions": [by_id[i] for i in ids]},
    )
    with pytest.raises(SystemExit, match="no leaf-only receipt"):
        _run_record(freeze, store, tmp_path, _runs())
    assert not (tmp_path / "run-record.json").exists()


def test_another_leaf_fits_receipt_cannot_vouch_for_these_predictions(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(capsys, freeze, store, target, image_root, root, CLASSICAL.fit_id)
    _predict_leaf(
        capsys, freeze, LEAF, CLASSICAL.fit_id, store, target, image_root, root
    )
    (store / OTHER_LEAF).mkdir()
    for produced in (store / LEAF).glob("predictions*"):
        shutil.copy(produced, store / OTHER_LEAF / produced.name)
    with pytest.raises(SystemExit, match="leaf-only receipt of .* does not verify"):
        _run_record(freeze, store, tmp_path, _runs(OTHER_LEAF))


def test_a_receipt_made_from_another_fit_is_refused(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(
        capsys, freeze, store, target, image_root, root, CLASSICAL.fit_id, DEEP.fit_id
    )
    _predict_leaf(
        capsys, freeze, LEAF, CLASSICAL.fit_id, store, target, image_root, root
    )
    with pytest.raises(SystemExit, match="made from a different fit"):
        _run_record(freeze, store, tmp_path, _runs(LEAF, DEEP.fit_id))


def test_predictions_edited_after_the_receipt_are_refused(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _full_prediction(capsys, freeze, store, target, image_root, root, CLASSICAL.fit_id)
    _predict_leaf(
        capsys, freeze, LEAF, CLASSICAL.fit_id, store, target, image_root, root
    )
    path = store / LEAF / "predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["predictions"] = payload["predictions"][::-1]
    _write(path, payload)
    with pytest.raises(SystemExit, match="not the ones predict-leaf-only wrote"):
        _run_record(freeze, store, tmp_path, _runs())


@pytest.mark.parametrize(
    ("leaf", "of", "message"),
    [
        (LEAF, OUTSIDE, "not in the freeze record"),
        (CLASSICAL.fit_id, CLASSICAL.fit_id, "must differ"),
    ],
)
def test_the_producer_refuses_a_source_it_cannot_authorise_before_any_read(
    leaf: str,
    of: str,
    message: str,
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,  # noqa: F811
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match=message):
        _predict_leaf(capsys, freeze, leaf, of, store, target, image_root, root)
    assert [r for r in _rows(root) if r["operation"] == "predict_leaf_only"] == []

"""Probe fits trained by the engines, read by the gate and bound in a run record."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
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
from jute_disease.training.benchmark import fit_identity

from .test_gated_evaluator import _cli, _freeze, _write
from .test_predict_test import _predict
from .test_predict_test import _recipe as _base_recipe

# The 32 px frame covers a 64 px image entirely, so the fits read a 128 px
# letterbox, where the frame leaves a blank interior.
PROBE_IMAGE_SIZE = 128


def _recipe() -> Recipe:
    return replace(_base_recipe(), image_size=PROBE_IMAGE_SIZE)


BORDER_CLASSICAL = FitSpec(
    "probe/border/classical/lr-raw",
    FitKind.CLASSICAL,
    42,
    estimator="lr",
    features="raw",
    variant="border",
)
PAD_CUE = FitSpec(
    "probe/pad-cue/classical/lr",
    FitKind.CLASSICAL,
    42,
    estimator="lr",
    features="pad-cue",
    variant="pad_cue",
)
BORDER_CNN = FitSpec(
    "probe/border/cnn/tiny/seed-42",
    FitKind.TARGET,
    42,
    system="tiny",
    model="resnet18",
    micro_batch=8,
    accumulation=1,
    variant="border",
)
FULL_CLASSICAL = FitSpec(
    "classical/lr-raw", FitKind.CLASSICAL, 42, estimator="lr", features="raw"
)
PROBES = (BORDER_CLASSICAL, PAD_CUE, BORDER_CNN)
EVERY_FIT = (*PROBES, FULL_CLASSICAL)


@pytest.fixture(scope="module")
def image_root(target: Path) -> Path:
    return target.parents[1] / "images"


@pytest.fixture(scope="module")
def trained(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A run store holding every probe fit and a full fit, each trained for real."""
    recipe = _recipe()
    data = DataInputs(image_root=image_root, target_dir=target, sources={})
    store = RunStore(tmp_path_factory.mktemp("probes"))
    for spec in EVERY_FIT:
        run = store.start(spec.fit_id, "synthetic identity")
        if spec.kind is FitKind.CLASSICAL:
            result = fit_classical(
                spec,
                recipe,
                data,
                run.attempt_dir,
                run.attempt_dir.parent / "feature-cache",
            )
        else:
            result = fit_cnn(spec, recipe, data, run.attempt_dir, pretrained=False)
        run.complete(result.results, result.artifacts)
    return store.root


@pytest.fixture
def store(trained: Path, tmp_path: Path) -> Path:
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
        [spec.fit_id for spec in EVERY_FIT],
    )


def _system(spec: FitSpec) -> str:
    return spec.system or "classical"


def _record(
    freeze: Path,
    store: Path,
    spec: FitSpec,
    tmp_path: Path,
    variant: str | None = None,
) -> dict[str, object]:
    runs = _write(
        tmp_path / "runs.json",
        [
            {
                "fit_id": spec.fit_id,
                "system": _system(spec),
                "seed": 42,
                "variant": spec.variant or "full" if variant is None else variant,
                "predictions": f"{spec.fit_id}/predictions.json",
            }
        ],
    )
    output = tmp_path / "run-record.json"
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
        output,
    )
    return json.loads(output.read_text(encoding="utf-8"))["body"]["fits"][spec.fit_id]


@pytest.mark.parametrize("spec", PROBES, ids=lambda spec: spec.fit_id)
def test_a_probe_fit_enters_a_run_record_with_its_lineage_and_variant(
    spec: FitSpec,
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = RunStore(store).receipt(spec.fit_id)
    assert receipt["results"]["variant"] == spec.variant

    result = _predict(capsys, freeze, spec.fit_id, store, target, image_root, root)
    entry = _record(freeze, store, spec, tmp_path)

    assert entry["variant"] == spec.variant
    assert entry["predictions_sha256"] == result["predictions_sha256"]
    assert entry["lineage_fit_id"] == spec.fit_id
    assert len(str(entry["training_receipt_sha256"])) == 64
    assert len(str(entry["prediction_receipt_sha256"])) == 64
    kind = "checkpoint" if spec.kind is FitKind.TARGET else "model"
    assert len(str(entry[f"{kind}_sha256"])) == 64


def test_a_probe_run_cannot_claim_to_be_a_full_fit(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _predict(capsys, freeze, BORDER_CLASSICAL.fit_id, store, target, image_root, root)
    with pytest.raises(SystemExit, match="variant"):
        _record(freeze, store, BORDER_CLASSICAL, tmp_path, variant="full")


def test_a_full_fit_cannot_be_filed_as_a_probe(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _predict(capsys, freeze, FULL_CLASSICAL.fit_id, store, target, image_root, root)
    with pytest.raises(SystemExit, match="variant"):
        _record(freeze, store, FULL_CLASSICAL, tmp_path, variant="border")


def test_a_border_probe_reads_the_test_images_through_its_own_variant(
    freeze: Path,
    store: Path,
    target: Path,
    image_root: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The border fit and the full fit share features and estimator, not pixels."""
    border = _predict(
        capsys, freeze, BORDER_CLASSICAL.fit_id, store, target, image_root, root
    )
    full = _predict(
        capsys, freeze, FULL_CLASSICAL.fit_id, store, target, image_root, root
    )
    assert border["predictions_sha256"] != full["predictions_sha256"]
    assert border["images"] == full["images"]


def test_a_variant_changes_a_fit_identity_and_its_absence_changes_nothing() -> None:
    recipe = _recipe()
    manifests = {"train": "a" * 64}
    plain = fit_identity(FULL_CLASSICAL, recipe, manifests, None)
    border = fit_identity(
        replace(FULL_CLASSICAL, variant="border"), recipe, manifests, None
    )
    pad = fit_identity(
        replace(FULL_CLASSICAL, variant="pad_cue"), recipe, manifests, None
    )
    assert len({plain, border, pad}) == 3
    # Pinned before the field existed: receipts of fits made earlier stay valid.
    recorded = fit_identity(
        FULL_CLASSICAL, load_benchmark(BENCHMARK_CONFIG).recipe, manifests, None
    )
    assert recorded == (
        "4237a5bfb4aa11d6d400f4bc4af6caedbc57a5ab95d9561b5bcbe5e9cb9423e5"
    )

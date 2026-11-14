"""Gated inference reads the pixels a cached fit trained on, for images the cache shrinks."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from jute_disease.config import BENCHMARK_CONFIG, CURATION_RULES_PATH
from jute_disease.data import (
    ManifestDataset,
    PreprocessingSpec,
    build_image_cache,
    cache_root,
    freeze_target,
    load_manifest,
    make_synthetic_data,
)
from jute_disease.data.manifest import load_complete_manifest
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

from .test_gated_evaluator import _cli, _freeze

IMAGE_SIZE = 64
SOURCE_SIDE = 600
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
def target(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A frozen target whose images are larger than the cache's downsize side."""
    base = tmp_path_factory.mktemp("large")
    plan = make_synthetic_data(base / "images", per_class=16, size=SOURCE_SIDE)
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
def image_root(target: Path) -> Path:
    return target.parents[1] / "images"


def _recipe() -> Recipe:
    return replace(
        load_benchmark(BENCHMARK_CONFIG).recipe,
        image_size=IMAGE_SIZE,
        max_epochs=1,
        precision="32-true",
        num_workers=0,
    )


def _train(
    spec: FitSpec, store: RunStore, data: DataInputs, *, drop: tuple[str, ...] = ()
) -> None:
    run = store.start(spec.fit_id, "synthetic identity")
    if spec.kind is FitKind.CLASSICAL:
        result = fit_classical(
            spec,
            _recipe(),
            data,
            run.attempt_dir,
            run.attempt_dir.parent / "feature-cache",
        )
    else:
        result = fit_cnn(spec, _recipe(), data, run.attempt_dir, pretrained=False)
    results = {key: value for key, value in result.results.items() if key not in drop}
    run.complete(results, result.artifacts)


@pytest.fixture(scope="module")
def image_cache(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The train and validation image cache the cached fits read."""
    cache = tmp_path_factory.mktemp("image-cache")
    build_image_cache(
        load_manifest(target),
        image_root,
        cache,
        PreprocessingSpec(size=IMAGE_SIZE),
        workers=2,
    )
    return cache


@pytest.fixture(scope="module")
def cached_store(
    target: Path,
    image_root: Path,
    image_cache: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Real deep and classical fits trained through the image cache."""
    data = DataInputs(
        image_root=image_root,
        target_dir=target,
        sources={},
        image_cache_dir=image_cache,
    )
    store = RunStore(tmp_path_factory.mktemp("cached"))
    for spec in FITS.values():
        _train(spec, store, data)
    return store.root


@pytest.fixture(scope="module")
def uncached_store(
    target: Path, image_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The same fits trained from the original images, with no cache."""
    data = DataInputs(image_root=image_root, target_dir=target, sources={})
    store = RunStore(tmp_path_factory.mktemp("uncached"))
    for spec in FITS.values():
        _train(spec, store, data)
    return store.root


@pytest.fixture(scope="module")
def unrecorded_store(
    target: Path,
    image_root: Path,
    image_cache: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Cached fits whose receipts do not say how their pixels were prepared."""
    data = DataInputs(
        image_root=image_root,
        target_dir=target,
        sources={},
        image_cache_dir=image_cache,
    )
    store = RunStore(tmp_path_factory.mktemp("unrecorded"))
    for spec in FITS.values():
        _train(spec, store, data, drop=("preprocessing",))
    return store.root


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
    freeze: Path, fit_id: str, store: Path, target: Path, image_root: Path, root: Path
) -> None:
    _cli(
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
        "--freeze-record",
        freeze,
    )


def _copy(source: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "store"
    shutil.copytree(source, copy)
    return copy


def _reference(
    target: Path, image_root: Path, cache: Path, transform: Any
) -> ManifestDataset:
    """The test images read through a cache, as a cached fit would read them."""
    spec = PreprocessingSpec(size=IMAGE_SIZE)
    manifest = load_complete_manifest(target)
    build_image_cache(
        manifest, image_root, cache, spec, ("test",), workers=2, allow_test=True
    )
    return ManifestDataset(
        manifest,
        "test",
        image_root,
        transform=transform,
        cache_root=cache_root(cache, spec),
        spec=spec,
    )


def _same_pixels(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def test_the_synthetic_images_are_larger_than_the_cache_downsize_side(
    target: Path, image_root: Path
) -> None:
    entry = load_manifest(target).partition("train")[0]
    with Image.open(image_root / entry.path) as image:
        assert min(image.size) > PreprocessingSpec().cache_shortest_side


@pytest.mark.parametrize("kind", ["deep", "classical"])
def test_a_cached_fit_is_read_at_inference_through_the_same_downsize(
    kind: str,
    cached_store: Path,
    freeze: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _copy(cached_store, tmp_path)
    seen: list[Any] = []
    if kind == "classical":
        real_extract = predict_module.extract_features

        def spy_extract(dataset: Any, *arguments: Any, **keywords: Any) -> Any:
            seen.append(dataset)
            return real_extract(dataset, *arguments, **keywords)

        monkeypatch.setattr(predict_module, "extract_features", spy_extract)
    else:
        real_module = predict_module.DataModule

        def spy_module(*arguments: Any, **keywords: Any) -> Any:
            module = real_module(*arguments, **keywords)
            seen.append(module)
            return module

        monkeypatch.setattr(predict_module, "DataModule", spy_module)

    _predict(freeze, FITS[kind].fit_id, store, target, image_root, root)

    assert len(seen) == 1
    if kind == "classical":
        inferred = seen[0]
        transform = inferred.transform
    else:
        inferred = seen[0].jute_predict
        transform = seen[0]._eval_transform()
    reference = _reference(target, image_root, tmp_path / "reference", transform)
    assert inferred.image_ids == reference.image_ids
    assert len(reference) > 0
    for index in range(len(reference)):
        assert _same_pixels(inferred[index][0], reference[index][0]), index


@pytest.mark.parametrize("kind", ["deep", "classical"])
def test_an_uncached_fit_is_still_read_from_the_original_images(
    kind: str,
    uncached_store: Path,
    freeze: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
) -> None:
    store = _copy(uncached_store, tmp_path)
    receipt = RunStore(store).receipt(FITS[kind].fit_id)
    assert receipt["results"]["preprocessing"] == {"source": "original"}

    _predict(freeze, FITS[kind].fit_id, store, target, image_root, root)

    assert (store / FITS[kind].fit_id / "predictions.json").is_file()


@pytest.mark.parametrize("kind", ["deep", "classical"])
def test_a_cached_fit_records_its_preprocessing_in_its_receipt(
    kind: str, cached_store: Path
) -> None:
    receipt = RunStore(cached_store).receipt(FITS[kind].fit_id)
    recorded = receipt["results"]["preprocessing"]
    spec = PreprocessingSpec(size=IMAGE_SIZE)
    assert recorded["source"] == "cache"
    assert recorded["spec_digest"] == spec.digest()
    assert recorded["spec"]["cache_shortest_side"] == spec.cache_shortest_side


@pytest.mark.parametrize("kind", ["deep", "classical"])
def test_a_cached_fit_that_records_no_preprocessing_is_refused_before_any_read(
    kind: str,
    unrecorded_store: Path,
    freeze: Path,
    target: Path,
    image_root: Path,
    root: Path,
    tmp_path: Path,
) -> None:
    store = _copy(unrecorded_store, tmp_path)

    with pytest.raises(SystemExit, match="preprocessing"):
        _predict(freeze, FITS[kind].fit_id, store, target, image_root, root)

    assert not (root / "artifacts/evaluation/test-reads.jsonl").exists()
    assert not (store / FITS[kind].fit_id / "predictions.json").exists()

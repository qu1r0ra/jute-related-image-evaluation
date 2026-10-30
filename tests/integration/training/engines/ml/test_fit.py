"""Classical fits on tiny frozen fixtures: train-only scaling, no test pixels."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import DataModule, ManifestDataset, file_sha256
from jute_disease.models.ml import FEATURE_EXTRACTORS
from jute_disease.training import (
    DataInputs,
    FitKind,
    FitSpec,
    Recipe,
    fit_classical,
    load_benchmark,
    prepare_features,
)

from ...conftest import Inputs


@pytest.fixture(scope="module")
def recipe() -> Recipe:
    return replace(load_benchmark(BENCHMARK_CONFIG).recipe, image_size=64)


@pytest.fixture(scope="module")
def data(inputs: Inputs) -> DataInputs:
    return DataInputs(
        image_root=inputs.image_root,
        target_dir=inputs.target_dir,
        sources=inputs.sources,
    )


def _fit(estimator: str, features: str) -> FitSpec:
    return FitSpec(
        f"classical/{estimator}-{features}",
        FitKind.CLASSICAL,
        42,
        estimator=estimator,
        features=features,
    )


PREPARE = FitSpec("classical/prepare", FitKind.PREPARE, 0)


def test_prepare_extracts_both_feature_sets_for_train_and_validation_only(
    recipe: Recipe, data: DataInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    original = ManifestDataset.__init__

    def spy(self, manifest, partition, *args, **kwargs) -> None:
        opened.append(partition)
        original(self, manifest, partition, *args, **kwargs)

    monkeypatch.setattr(ManifestDataset, "__init__", spy)
    result = prepare_features(PREPARE, recipe, data, tmp_path / "cache")

    assert set(opened) == {"train", "val"}
    assert result.results["test_accessed"] is False
    assert set(result.results["features"]) == {"raw", "crafted"}
    for entry in result.results["features"].values():
        assert set(entry) == {"train", "val"}
        assert entry["train"]["rows"] > 0
        assert len(entry["train"]["x_sha256"]) == 64
    assert len(list((tmp_path / "cache").glob("*_X.npy"))) == 4


def test_fit_after_prepare_reads_only_the_cache(
    recipe: Recipe, data: DataInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    prepare_features(PREPARE, recipe, data, cache)

    def forbidden(self, img) -> None:
        raise AssertionError("features must come from the prepared cache")

    for extractor in FEATURE_EXTRACTORS.values():
        monkeypatch.setattr(extractor, "extract", forbidden)
    result = fit_classical(_fit("lr", "raw"), recipe, data, tmp_path / "fit", cache)
    assert result.artifacts["model"].is_file()


@pytest.mark.parametrize("estimator", ["gnb", "knn", "lr", "rf", "svm"])
def test_every_estimator_fits_both_feature_sets(
    estimator: str, recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    for features in ("raw", "crafted"):
        result = fit_classical(
            _fit(estimator, features), recipe, data, tmp_path / features, cache
        )
        results = result.results
        assert results["estimator"] == estimator
        assert results["features"] == features
        assert 0.0 <= results["metrics"]["val_accuracy"] <= 1.0
        assert 0.0 <= results["metrics"]["val_macro_f1"] <= 1.0
        assert results["model"]["sha256"] == file_sha256(result.artifacts["model"])
        assert results["calibration"] is False
        assert results["sweep"] is False
        assert results["test_accessed"] is False
        assert not any("test" in key for key in results["metrics"])


def test_scaler_sees_training_rows_only(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    import joblib

    result = fit_classical(
        _fit("lr", "raw"), recipe, data, tmp_path / "fit", tmp_path / "cache"
    )
    pipeline = joblib.load(result.artifacts["model"])
    scaler = pipeline.named_steps["scaler"]
    assert scaler.n_samples_seen_ == result.results["rows"]["train"]
    assert result.results["rows"]["val"] != scaler.n_samples_seen_
    assert result.results["scaling"] == "train-only"


def test_weighting_is_explicit_and_unsupported_is_recorded(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    knn = fit_classical(_fit("knn", "raw"), recipe, data, tmp_path / "knn", cache)
    lr = fit_classical(_fit("lr", "raw"), recipe, data, tmp_path / "lr", cache)
    assert knn.results["weighting"] == {
        "requested": "balanced",
        "status": "unsupported",
    }
    assert lr.results["weighting"] == {"requested": "balanced", "status": "applied"}


def test_svm_has_no_probability_calibration(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    import joblib

    result = fit_classical(
        _fit("svm", "raw"), recipe, data, tmp_path / "fit", tmp_path / "cache"
    )
    assert (
        joblib.load(result.artifacts["model"]).named_steps["clf"].probability is False
    )


def test_a_fit_is_deterministic_for_its_seed(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    first = fit_classical(_fit("rf", "crafted"), recipe, data, tmp_path / "a", cache)
    second = fit_classical(_fit("rf", "crafted"), recipe, data, tmp_path / "b", cache)
    assert first.results["metrics"] == second.results["metrics"]
    assert first.results["predictions_sha256"] == second.results["predictions_sha256"]


def test_classical_fits_never_open_the_test_partition(
    recipe: Recipe, data: DataInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    original = DataModule._manifest_dataset

    def spy(self, partition, train) -> object:
        opened.append(partition)
        return original(self, partition, train)

    monkeypatch.setattr(DataModule, "_manifest_dataset", spy)
    fit_classical(
        _fit("gnb", "raw"), recipe, data, tmp_path / "fit", tmp_path / "cache"
    )
    assert "test" not in opened


def test_features_must_match_a_declared_extractor(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="features"):
        fit_classical(_fit("lr", "bogus"), recipe, data, tmp_path / "x", tmp_path / "c")
    with pytest.raises(ValueError, match="estimator"):
        fit_classical(
            _fit("bogus", "raw"), recipe, data, tmp_path / "x", tmp_path / "c"
        )


def test_cached_features_equal_fresh_extraction(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    prepared = prepare_features(PREPARE, recipe, data, cache)
    again = prepare_features(PREPARE, recipe, data, tmp_path / "other")
    for name in ("raw", "crafted"):
        for partition in ("train", "val"):
            assert (
                prepared.results["features"][name][partition]["x_sha256"]
                == again.results["features"][name][partition]["x_sha256"]
            )
    assert all(np.load(x).dtype == np.float32 for x in cache.glob("*_X.npy"))

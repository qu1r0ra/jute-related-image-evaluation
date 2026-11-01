import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.data.masks import (
    ColorThresholdBackend,
    SceneFeatures,
    SceneObservation,
    SceneThresholds,
    classify_scene,
    color_uniformity,
    extract_scene_features,
    fit_scene_thresholds,
    save_mask_artifacts,
)


def test_color_mask_keeps_only_largest_excess_green_component() -> None:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:] = (25, 35, 100)
    pixels[8:48, 8:42] = (20, 180, 30)
    pixels[55:58, 55:58] = (20, 180, 30)

    result = ColorThresholdBackend().predict(Image.fromarray(pixels))

    assert result.mask.dtype == np.bool_
    assert result.mask.shape == (64, 64)
    assert result.mask.sum() == 40 * 34
    assert result.mask[20, 20]
    assert not result.mask[56, 56]
    assert result.model == "excess-green-otsu-largest-component"


def test_color_mask_returns_empty_component_when_no_green_signal() -> None:
    result = ColorThresholdBackend().predict(Image.new("RGB", (20, 16), (80, 50, 120)))

    assert result.mask.shape == (16, 20)
    assert not result.mask.any()
    assert result.diagnostics["component_count"] == 0


def test_background_features_and_scene_tags_use_calibrated_cutoffs() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = (245, 245, 245)
    image[5:15, 5:15] = (20, 170, 30)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    features = extract_scene_features(image, mask)
    thresholds = SceneThresholds(
        background_fraction_min=0.5,
        color_uniformity_min=0.9,
        development_ids=tuple(f"dev-{index}" for index in range(100)),
        development_sources=tuple(
            (f"dev-{index}", hashlib.sha256(f"source-{index}".encode()).hexdigest())
            for index in range(100)
        ),
        development_digest="a" * 64,
        score=1.0,
    )

    assert features.background_fraction == pytest.approx(0.75)
    assert features.color_uniformity == pytest.approx(1.0)
    assert classify_scene(features, thresholds) == "clean-background"
    assert classify_scene(SceneFeatures(0.8, 0.2), thresholds) == "field"
    assert classify_scene(SceneFeatures(0.2, 1.0), thresholds) == "unknown"


def test_color_uniformity_is_zero_for_empty_background() -> None:
    assert color_uniformity(np.empty((0, 3), dtype=np.uint8)) == 0.0


def test_scene_calibration_requires_100_and_records_source_holdout() -> None:
    observations = []
    for index in range(100):
        category = index % 3
        tag = ("clean-background", "field", "unknown")[category]
        features = (
            SceneFeatures(0.9, 0.99)
            if category == 0
            else SceneFeatures(0.9, 0.5)
            if category == 1
            else SceneFeatures(0.2, 0.99)
        )
        observations.append(
            SceneObservation(
                f"dev-{index}",
                features,
                tag,
                hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            )
        )

    thresholds = fit_scene_thresholds(observations)
    restored = SceneThresholds.from_dict(thresholds.to_dict())

    assert thresholds.score == 1.0
    assert len(thresholds.development_ids) == 100
    assert len(thresholds.development_sources) == 100
    assert restored == thresholds
    with pytest.raises(ValueError, match="exactly 100"):
        fit_scene_thresholds(observations[:99])


def test_mask_artifacts_store_provenance_and_overlay(tmp_path: Path) -> None:
    image = Image.new("RGB", (12, 10), (0, 180, 0))
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    result = ColorThresholdBackend().predict(image)
    result = result.with_mask(mask)

    artifacts = save_mask_artifacts(
        image,
        result,
        tmp_path,
        image_id="class/a leaf.png",
        model_version="fixture-1",
        rubric_version="leaf-mask-v1",
    )

    assert artifacts.mask_path.exists()
    assert artifacts.overlay_path.exists()
    assert artifacts.metadata_path.exists()
    metadata = artifacts.read_metadata()
    assert metadata["model"] == result.model
    assert metadata["model_version"] == "fixture-1"
    assert metadata["rubric_version"] == "leaf-mask-v1"
    assert Image.open(artifacts.mask_path).size == image.size
    with pytest.raises(FileExistsError):
        save_mask_artifacts(
            image,
            result,
            tmp_path,
            image_id="class/a leaf.png",
            model_version="fixture-1",
            rubric_version="leaf-mask-v1",
        )

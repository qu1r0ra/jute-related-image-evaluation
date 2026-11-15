import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.data.masks import (
    MAX_DEVELOPMENT_IMAGES,
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


def test_scene_calibration_records_source_holdout() -> None:
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


def _observation(index: int, tag: str, features: SceneFeatures) -> SceneObservation:
    return SceneObservation(
        f"dev-{index:03d}",
        features,
        tag,
        hashlib.sha256(f"source-{index}".encode()).hexdigest(),
    )


def _two_tag_observations() -> list[SceneObservation]:
    groups = (
        ("clean-background", 40, SceneFeatures(0.9, 0.9)),
        ("clean-background", 10, SceneFeatures(0.1, 0.6)),
        ("field", 45, SceneFeatures(0.8, 0.3)),
        ("field", 5, SceneFeatures(0.05, 0.7)),
    )
    observations: list[SceneObservation] = []
    for tag, count, features in groups:
        for _ in range(count):
            observations.append(_observation(len(observations), tag, features))
    return observations


def _three_tag_observations() -> list[SceneObservation]:
    cycle = (
        ("clean-background", SceneFeatures(0.9, 0.99)),
        ("field", SceneFeatures(0.9, 0.5)),
        ("unknown", SceneFeatures(0.2, 0.99)),
    )
    return [_observation(index, *cycle[index % 3]) for index in range(100)]


def test_two_tag_development_set_fits_with_no_unknown_class() -> None:
    observations = _two_tag_observations()

    thresholds = fit_scene_thresholds(observations)

    assert thresholds.background_fraction_min == 0.0
    assert thresholds.development_tags == ("clean-background", "field")
    assert all(
        classify_scene(item.features, thresholds) != "unknown" for item in observations
    )
    assert SceneThresholds.from_dict(thresholds.to_dict()) == thresholds


def test_two_tag_uniformity_cutoff_is_the_hand_computed_macro_f1_optimum() -> None:
    # Uniformity values are 0.3 (45 field), 0.6 (10 clean), 0.7 (5 field) and
    # 0.9 (40 clean), so the candidate cutoffs are 0, 0.45, 0.65, 0.8 and 1.
    # Two-tag macro-F1 at each: 1/3, 0.9499 (clean 100/105, field 90/95),
    # 0.8496 (80/95, 90/105), 0.8990 (80/90, 100/110) and 1/3.
    thresholds = fit_scene_thresholds(_two_tag_observations())

    assert thresholds.color_uniformity_min == pytest.approx(0.45)
    assert thresholds.score == pytest.approx((100 / 105 + 90 / 95) / 2)


@pytest.mark.parametrize(
    "tags",
    [("field", "unknown"), ("clean-background", "unknown"), ("field",)],
)
def test_development_set_without_clean_background_or_field_is_refused(
    tags: tuple[str, ...],
) -> None:
    observations = [
        _observation(index, tags[index % len(tags)], SceneFeatures(0.5, 0.5))
        for index in range(100)
    ]

    with pytest.raises(ValueError, match="clean-background and field"):
        fit_scene_thresholds(observations)


def test_three_tag_fit_keeps_its_cutoffs_and_digest_material() -> None:
    observations = _three_tag_observations()
    material = [
        {
            "image_id": item.image_id,
            "scene_tag": item.scene_tag,
            "source_sha256": item.source_sha256,
            "background_fraction": item.features.background_fraction,
            "color_uniformity": item.features.color_uniformity,
        }
        for item in sorted(observations, key=lambda item: item.image_id)
    ]
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    thresholds = fit_scene_thresholds(observations)
    stored = thresholds.to_dict()

    assert thresholds.background_fraction_min == pytest.approx(0.55)
    assert thresholds.color_uniformity_min == pytest.approx(0.745)
    assert thresholds.score == 1.0
    assert thresholds.development_digest == digest
    assert thresholds.method == "100-image grid search maximizing scene macro-F1"
    assert stored.pop("development_tags") == [
        "clean-background",
        "field",
        "unknown",
    ]
    assert set(stored) == {
        "version",
        "background_fraction_min",
        "color_uniformity_min",
        "development_ids",
        "development_sources",
        "development_digest",
        "development_count",
        "score",
        "method",
        "mask_model",
        "mask_version",
        "categories",
    }


def test_thresholds_stored_before_the_tag_record_still_load() -> None:
    legacy = fit_scene_thresholds(_three_tag_observations()).to_dict()
    del legacy["development_tags"]

    restored = SceneThresholds.from_dict(legacy)

    assert restored.development_tags is None
    assert restored.background_fraction_min == legacy["background_fraction_min"]
    assert restored.to_dict() == legacy


def test_thresholds_refuse_a_tag_record_outside_the_scene_tags() -> None:
    stored = fit_scene_thresholds(_two_tag_observations()).to_dict()

    for tags in (["clean-background", "sky"], ["field", "unknown"]):
        with pytest.raises(ValueError, match="development tags"):
            SceneThresholds.from_dict({**stored, "development_tags": tags})


def _stored_digest(thresholds: SceneThresholds) -> str:
    stored = json.dumps(thresholds.to_dict(), sort_keys=True)
    return hashlib.sha256(stored.encode("utf-8")).hexdigest()


def test_development_limit_is_100_images() -> None:
    assert MAX_DEVELOPMENT_IMAGES == 100


def test_90_image_development_set_fits_and_records_its_count() -> None:
    observations = _three_tag_observations()[:90]

    thresholds = fit_scene_thresholds(observations)
    stored = thresholds.to_dict()

    assert stored["development_count"] == 90
    assert len(thresholds.development_ids) == 90
    assert len(thresholds.development_sources) == 90
    assert len({digest for _, digest in thresholds.development_sources}) == 90
    assert thresholds.method == "90-image grid search maximizing scene macro-F1"
    assert thresholds.score == 1.0
    assert SceneThresholds.from_dict(json.loads(json.dumps(stored))) == thresholds


def test_90_image_two_tag_set_names_its_count_in_the_method() -> None:
    observations = [
        item
        for item in _two_tag_observations()
        if item.features.color_uniformity != 0.6
    ]

    thresholds = fit_scene_thresholds(observations)

    assert len(observations) == 90
    assert thresholds.to_dict()["development_count"] == 90
    assert thresholds.method.startswith("90-image grid search")
    assert thresholds.background_fraction_min == 0.0


def test_one_image_of_each_required_tag_is_enough() -> None:
    observations = _three_tag_observations()[:2]

    thresholds = fit_scene_thresholds(observations)

    assert thresholds.to_dict()["development_count"] == 2
    assert SceneThresholds.from_dict(thresholds.to_dict()) == thresholds


@pytest.mark.parametrize("count", [0, 101])
def test_development_set_outside_1_to_100_images_is_refused(count: int) -> None:
    cycle = _three_tag_observations()
    observations = [
        _observation(index, cycle[index % 3].scene_tag, cycle[index % 3].features)
        for index in range(count)
    ]

    with pytest.raises(ValueError, match="between 1 and 100 development images"):
        fit_scene_thresholds(observations)


def test_development_set_with_a_repeated_id_is_refused() -> None:
    observations = _three_tag_observations()[:90]
    observations[1] = SceneObservation(
        observations[0].image_id,
        observations[1].features,
        observations[1].scene_tag,
        observations[1].source_sha256,
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        fit_scene_thresholds(observations)


def test_development_set_with_a_repeated_source_hash_is_refused() -> None:
    observations = _three_tag_observations()[:90]
    observations[1] = SceneObservation(
        observations[1].image_id,
        observations[1].features,
        observations[1].scene_tag,
        observations[0].source_sha256,
    )

    with pytest.raises(ValueError, match="unique source content"):
        fit_scene_thresholds(observations)


def _stored_with(count: int) -> dict:
    stored = fit_scene_thresholds(_three_tag_observations()[:90]).to_dict()
    extra = [
        _observation(index, "field", SceneFeatures(0.5, 0.5)) for index in range(101)
    ]
    stored["development_ids"] = [item.image_id for item in extra[:count]]
    stored["development_sources"] = [
        {"image_id": item.image_id, "sha256": item.source_sha256}
        for item in extra[:count]
    ]
    return stored


@pytest.mark.parametrize("count", [0, 101])
def test_stored_thresholds_outside_1_to_100_images_are_refused(count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        SceneThresholds.from_dict(_stored_with(count))


def test_stored_thresholds_with_a_repeated_id_are_refused() -> None:
    stored = _stored_with(90)
    stored["development_ids"][1] = stored["development_ids"][0]

    with pytest.raises(ValueError, match="unique development IDs"):
        SceneThresholds.from_dict(stored)


def test_stored_thresholds_with_a_repeated_source_hash_are_refused() -> None:
    stored = _stored_with(90)
    sources = stored["development_sources"]
    sources[1]["sha256"] = sources[0]["sha256"]

    with pytest.raises(ValueError, match="unique source content"):
        SceneThresholds.from_dict(stored)


def test_stored_thresholds_with_fewer_sources_than_ids_are_refused() -> None:
    stored = _stored_with(90)
    stored["development_sources"].pop()

    with pytest.raises(ValueError, match="one source per development ID"):
        SceneThresholds.from_dict(stored)


def test_100_image_sets_keep_the_record_they_had_under_the_exact_count() -> None:
    # Digests of the stored records fitted at 48423d6, when the count was fixed.
    three_tag = fit_scene_thresholds(_three_tag_observations())
    two_tag = fit_scene_thresholds(_two_tag_observations())

    assert _stored_digest(three_tag) == (
        "bbbf7de23e021671de4630605ef445082334a9a5ccdd6d3b508b8ebf4648a465"
    )
    assert _stored_digest(two_tag) == (
        "035dae33cb15a55561b4e785de2eb52a09f21e31e949a8c54ae020909aae15f2"
    )
    assert three_tag.to_dict()["development_count"] == 100

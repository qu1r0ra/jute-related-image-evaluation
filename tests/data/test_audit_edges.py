import pytest

from jute_disease.data import sample_edges


def test_pooled_edges_stay_inside_each_pool_with_binned_strata() -> None:
    images = {
        "A/x/1.jpg": (0b0, "x", "A"),
        "A/x/2.jpg": (0b111, "x", "A"),
        "A/y/3.jpg": (0b11_1111, "y", "A"),
        "B/x/1.jpg": (0b111, "x", "B"),
    }

    bands = sample_edges(images, count=10)["bands"]

    pairs = [pair for band in bands.values() for pair in band["pairs"]]
    assert all(pair["image_a"][0] == pair["image_b"][0] for pair in pairs)
    assert set(bands["merged"]["strata"]) == {"A:3:same_label", "A:3:cross_label"}
    assert set(bands["near_miss"]["strata"]) == {"A:5-8:cross_label"}


def test_pool_tag_must_be_given_for_every_image() -> None:
    with pytest.raises(ValueError, match="every image or for none"):
        sample_edges({"a": (0, "x", "A"), "b": (1, "x")})

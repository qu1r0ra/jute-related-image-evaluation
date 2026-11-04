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


def test_threshold_moves_the_merged_and_near_miss_bands() -> None:
    images = {
        "a": (0b0, "x"),
        "b": (0b11_1111, "x"),
        "c": (0b111_1111, "x"),
    }

    wide = sample_edges(images, count=10, threshold=6)
    narrow = sample_edges(images, count=10)

    assert wide["threshold"] == 6
    assert set(wide["bands"]["merged"]["strata"]) == {"6:same_label"}
    assert set(wide["bands"]["near_miss"]["strata"]) == {"7:same_label"}
    assert not narrow["bands"]["merged"]["strata"]
    assert set(narrow["bands"]["near_miss"]["strata"]) == {
        "6:same_label",
        "7:same_label",
    }
    with pytest.raises(ValueError, match="threshold"):
        sample_edges(images, threshold=2)


def test_pool_tag_must_be_given_for_every_image() -> None:
    with pytest.raises(ValueError, match="every image or for none"):
        sample_edges({"a": (0, "x", "A"), "b": (1, "x")})

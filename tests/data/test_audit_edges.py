import random
from typing import Any

import pytest

from jute_disease.data import DEFAULT_DESIGN, sample_edges

CENSUS_C = [{"pool": "C", "band": "merged", "relation": "cross_label"}]


def design(budget_c: int = 30, allocation: str = "neyman") -> dict[str, Any]:
    return {
        "pools": {
            "A": {"budget": 12, "allocation": "proportional"},
            "C": {"budget": budget_c, "allocation": allocation},
        },
        "census": CENSUS_C,
        "neyman_prior": {"merged": 0.05, "near_miss": 0.5},
    }


def random_images(count: int = 60, bits: int = 12, seed: int = 1) -> dict[str, tuple]:
    rng = random.Random(seed)
    return {
        f"{pool}/{i}.jpg": (rng.randrange(1 << bits), rng.choice("xy"), pool)
        for pool in "AC"
        for i in range(count)
    }


def test_grouping_version_sets_the_merge_threshold_and_merged_band_starts_at_zero() -> (
    None
):
    images = {
        "C/a": (0b0, "x", "C"),
        "C/b": (0b0, "x", "C"),
        "C/c": (0b11, "x", "C"),
        "C/d": (0b11_1111, "x", "C"),
    }

    v1 = sample_edges(images, design(), "dhash-v1")
    v2 = sample_edges(images, design(), "dhash-v2")

    assert (v1["threshold"], v2["threshold"]) == (4, 6)
    bins_v1 = {s["bin"] for s in v1["strata"].values() if s["band"] == "merged"}
    bins_v2 = {s["bin"] for s in v2["strata"].values() if s["band"] == "merged"}
    assert bins_v1 == {"0-2", "3-4"}
    assert bins_v2 == {"0-2", "3-4", "5-6"}
    assert {s["bin"] for s in v1["strata"].values() if s["band"] == "near_miss"} == {
        "5-8"
    }


def test_unknown_grouping_version_is_refused() -> None:
    with pytest.raises(ValueError, match="grouping version"):
        sample_edges(random_images(), design(), "dhash-v9")


def test_every_pool_c_cross_label_merged_pair_is_in_the_sample() -> None:
    images = random_images()
    result = sample_edges(images, design(budget_c=20), "dhash-v2")

    census = {
        key: s
        for key, s in result["strata"].items()
        if s["pool"] == "C" and s["band"] == "merged" and s["relation"] == "cross_label"
    }
    assert census
    assert all(s["census"] and s["sampled"] == s["population"] for s in census.values())
    sampled = {
        (p["image_a"], p["image_b"]) for p in result["pairs"] if p["stratum"] in census
    }
    expected = {
        (a, b)
        for a in images
        for b in images
        if a < b
        and a.startswith("C/")
        and b.startswith("C/")
        and images[a][1] != images[b][1]
        and (images[a][0] ^ images[b][0]).bit_count() <= 6
    }
    assert sampled == expected
    # Distance 0..2 pairs are part of the census, not skipped.
    assert any((images[a][0] ^ images[b][0]).bit_count() <= 2 for a, b in expected)
    assert not any(s["census"] for s in result["strata"].values() if s["pool"] == "A")


def test_each_pair_records_stratum_inclusion_probability_and_family() -> None:
    result = sample_edges(random_images(), design(), "dhash-v2")

    strata = result["strata"]
    for pair in result["pairs"]:
        stratum = strata[pair["stratum"]]
        assert pair["inclusion_probability"] == pytest.approx(
            stratum["sampled"] / stratum["population"]
        )
        assert pair["family"] == stratum["family"] in {"same_family", "split"}
        assert pair["pool"] == pair["image_a"][0] == pair["image_b"][0]
    for key, stratum in strata.items():
        taken = [p for p in result["pairs"] if p["stratum"] == key]
        assert len(taken) == stratum["sampled"] >= 1
        assert key == ":".join(
            stratum[name] for name in ("pool", "band", "bin", "relation", "family")
        )
    assert result["design"]["census"] == CENSUS_C


def test_family_is_derived_from_the_threshold_graph_without_dropping_scenes() -> None:
    images = {
        # One chain a-b-c at distance 4 and 4; a and c are 8 apart: same family.
        "C/a": (0b0, "x", "C"),
        "C/b": (0b1111, "x", "C"),
        "C/c": (0b1111_1111, "y", "C"),
        # An isolated image 9 bits from a: a different family.
        "C/d": (0b1_1111_1111 << 20, "x", "C"),
    }

    result = sample_edges(images, design(), "dhash-v1")

    near = {
        (p["image_a"], p["image_b"]): p["family"]
        for p in result["pairs"]
        if p["band"] == "near_miss"
    }
    assert near[("C/a", "C/c")] == "same_family"
    assert near[("C/a", "C/d")] == "split"
    quality = result["dataset_quality"]
    assert quality["pool_c_mixed_label_scenes"] == 1
    assert quality["pools"]["C"] == {
        "images": 4,
        "families": 2,
        "mixed_label_scenes": 1,
        "dropped_families": 0,
    }


def test_proportional_and_neyman_allocation_follow_their_weights() -> None:
    images = random_images(count=70, bits=12, seed=4)
    spread = {"merged": (0.05 * 0.95) ** 0.5, "near_miss": 0.5}

    def allocate(allocation: str) -> dict[str, Any]:
        no_census = design(300, allocation) | {"census": []}
        return sample_edges(images, no_census, "dhash-v2")

    for allocation in ("proportional", "neyman"):
        strata = [
            s for s in allocate(allocation)["strata"].values() if s["pool"] == "C"
        ]
        weights = [
            s["population"] * (spread[s["band"]] if allocation == "neyman" else 1)
            for s in strata
        ]
        for stratum, weight in zip(strata, weights, strict=True):
            target = 300 * weight / sum(weights)
            assert stratum["sampled"] >= 1
            if target >= 10:
                assert stratum["sampled"] == pytest.approx(target, abs=2)
        assert sum(s["sampled"] for s in strata) == 300

    def near_miss_share(allocation: str) -> float:
        strata = [
            s for s in allocate(allocation)["strata"].values() if s["pool"] == "C"
        ]
        return sum(s["sampled"] for s in strata if s["band"] == "near_miss") / 300

    assert near_miss_share("neyman") > near_miss_share("proportional")


def test_pairs_stay_inside_each_pool_and_sampling_is_seeded() -> None:
    images = random_images()

    first = sample_edges(images, design(), "dhash-v2", seed=3)
    again = sample_edges(images, design(), "dhash-v2", seed=3)
    other = sample_edges(images, design(), "dhash-v2", seed=4)

    assert first == again
    assert first["pairs"] != other["pairs"]
    assert all(p["image_a"][0] == p["image_b"][0] for p in first["pairs"])


def test_default_design_targets_about_two_hundred_forty_pairs() -> None:
    budgets = [spec["budget"] for spec in DEFAULT_DESIGN["pools"].values()]
    assert sum(budgets) == 240
    assert DEFAULT_DESIGN["census"] == CENSUS_C


def test_image_needs_a_pool_and_design_needs_a_budget_for_it() -> None:
    with pytest.raises(ValueError, match="pool"):
        sample_edges({"a": (0, "x"), "b": (1, "x")}, design(), "dhash-v2")
    with pytest.raises(ValueError, match="budget for pool"):
        sample_edges({"a": (0, "x", "Z"), "b": (1, "x", "Z")}, design(), "dhash-v2")

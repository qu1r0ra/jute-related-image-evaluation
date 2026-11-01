import numpy as np
import pytest

from jute_disease.evaluation import PredictionSet, resample_metrics


def test_nested_shared_family_weights_and_replicate_rankings() -> None:
    def make(ids, predictions) -> PredictionSet:
        return PredictionSet(
            ("a", "b"),
            tuple(ids),
            ("a",) * len(ids),
            tuple(ids),
            ("s",) * len(ids),
            {1: tuple(predictions)},
        )

    # Different native sets: both systems agree on the common family and have
    # identical outcomes on the shared family, despite distinct native sets.
    a = make(["shared", "only-a"], ["a", "b"])
    b = make(["shared", "only-b"], ["a", "b"])
    result = resample_metrics(
        {"a": {0: a}, "b": {0: b}}, "accuracy", resamples=500, seed=3
    )
    # In the event that both systems have one sampled observation, the shared
    # family must be present for both or absent for both.
    assert np.all((result.samples["a"] == 0) == (result.samples["b"] == 0))
    # Three union-family draws admit only these score pairs. In particular,
    # (2/3, 1/2) would require different shared-family multiplicities.
    possible = {
        (1.0, 1.0),
        (2 / 3, 1.0),
        (1.0, 2 / 3),
        (1 / 3, 1.0),
        (1.0, 1 / 3),
        (0.5, 0.5),
        (0.0, 0.0),
    }
    assert set(zip(result.samples["a"], result.samples["b"], strict=True)) <= possible
    assert result.rankings.shape == (500, 2)
    assert np.any(result.samples["a"] != result.samples["b"])
    assert result.difference("a", "b").lower < 0 < result.difference("a", "b").upper
    for row, left, right in zip(
        result.rankings, result.samples["a"], result.samples["b"], strict=True
    ):
        if left > right:
            assert tuple(row) == ("a", "b")
        elif right > left:
            assert tuple(row) == ("b", "a")


def test_route_weights_checkpoints_equally_with_17_17_16_seeds() -> None:
    predictions = {i: ("a" if i < 17 else "b",) for i in range(50)}
    system = PredictionSet(("a", "b"), ("i",), ("a",), ("f",), ("s",), predictions)
    assignment = {
        i: "first" if i < 17 else "second" if i < 34 else "third" for i in range(50)
    }
    result = resample_metrics(
        {"route": {0: system}}, "accuracy", 30, checkpoints={0: assignment}
    )
    assert result.estimates["route"] == pytest.approx(1 / 3)
    assert result.samples["route"] == pytest.approx(np.full(30, 1 / 3))


def test_route_checkpoints_pool_runs_across_splits_before_weighting() -> None:
    # Checkpoint A has three runs (two in split 0, one in split 1) and checkpoint
    # B has three runs. Each run is scored by its image accuracy.
    split0 = PredictionSet(
        ("a", "b"),
        ("i0", "i1"),
        ("a", "b"),
        ("f0", "f1"),
        ("s", "s"),
        {1: ("a", "b"), 2: ("a", "b"), 3: ("b", "a")},
    )
    split1 = PredictionSet(
        ("a", "b"),
        ("i0", "i1"),
        ("a", "b"),
        ("f0", "f1"),
        ("s", "s"),
        {4: ("b", "a"), 5: ("a", "b"), 6: ("a", "b")},
    )
    checkpoints = {
        0: {1: "A", 2: "A", 3: "B"},
        1: {4: "A", 5: "B", 6: "B"},
    }
    result = resample_metrics(
        {"route": {0: split0, 1: split1}}, "accuracy", 1, checkpoints=checkpoints
    )
    # Pooled checkpoint means are A = (1 + 1 + 0) / 3 and B = (0 + 1 + 1) / 3.
    assert result.estimates["route"] == pytest.approx(2 / 3)


def test_one_image_per_family_sensitivity_uses_stable_image_id() -> None:
    from jute_disease.evaluation import one_image_per_family, score_run

    predictions = PredictionSet(
        ("a", "b"),
        ("z", "a", "b"),
        ("a", "a", "a"),
        ("f", "f", "g"),
        ("s", "s", "s"),
        {1: ("a", "b", "a")},
    )
    selected = one_image_per_family(predictions)
    assert selected.image_ids == ("a", "b")
    assert score_run(selected, 1).accuracy == 0.5
    assert score_run(predictions, 1).accuracy == pytest.approx(2 / 3)


def test_split_and_seed_draws_are_shared_between_identical_systems() -> None:
    def make(predictions) -> PredictionSet:
        return PredictionSet(("a", "b"), ("i",), ("a",), ("f",), ("s",), predictions)

    split_sets = {0: make({1: ("a",), 2: ("b",)}), 1: make({1: ("a",), 2: ("a",)})}
    result = resample_metrics(
        {"a": split_sets, "b": split_sets}, "accuracy", 200, seed=4
    )
    assert result.estimates["a"] == 0.75
    assert np.array_equal(result.samples["a"], result.samples["b"])
    assert set(result.samples["a"]) >= {0.0, 0.5, 1.0}
    assert result.difference("a", "b").lower == result.difference("a", "b").upper == 0

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from jute_disease.evaluation import (
    PredictionSet,
    require_class_support,
    resample_metrics,
)


def _family_set(
    families: dict[str, tuple[str, str]],
    classes: tuple[str, ...] = ("a", "b"),
) -> PredictionSet:
    """One image per family; each family maps to (true class, predicted class)."""
    ids = tuple(families)
    return PredictionSet(
        classes,
        ids,
        tuple(truth for truth, _ in families.values()),
        ids,
        ("s",) * len(ids),
        {1: tuple(pred for _, pred in families.values())},
    )


def test_shared_resampling_gives_every_system_the_same_family_multiplicities() -> None:
    # View 2: one protected test set. System a is right only on "x", system b
    # only on "y", so their accuracies are k/2 and (2 - k)/2 for the same k
    # copies of "x" and sum to one in every replicate. Independent draws would
    # not satisfy that identity.
    a = _family_set({"x": ("a", "a"), "y": ("a", "b")})
    b = _family_set({"x": ("a", "b"), "y": ("a", "a")})
    result = resample_metrics(
        {"a": {0: a}, "b": {0: b}},
        "accuracy",
        resamples=500,
        seed=3,
        family_resampling="shared",
    )
    assert result.samples["a"] + result.samples["b"] == pytest.approx(np.ones(500))
    assert set(result.samples["a"]) == {0.0, 0.5, 1.0}
    assert result.rankings.shape == (500, 2)
    assert result.difference("a", "b").lower < 0 < result.difference("a", "b").upper
    for row, left, right in zip(
        result.rankings, result.samples["a"], result.samples["b"], strict=True
    ):
        if left > right:
            assert tuple(row) == ("a", "b")
        elif right > left:
            assert tuple(row) == ("b", "a")
    assert "shared" in result.method


def test_shared_resampling_requires_the_same_test_families_in_every_system() -> None:
    a = _family_set({"shared": ("a", "a"), "only-a": ("a", "b")})
    b = _family_set({"shared": ("a", "a"), "only-b": ("a", "b")})
    with pytest.raises(ValueError, match="same test families"):
        resample_metrics({"a": {0: a}, "b": {0: b}}, "accuracy", 10)


def test_per_system_resampling_draws_each_policys_own_families_independently() -> None:
    # View 1: each policy has its own test families. Policy x has a right and a
    # wrong family in class "a"; policy y has only right ones, so its score can
    # never depend on x's draw.
    x = _family_set({"x1": ("a", "a"), "x2": ("a", "b")})
    y = _family_set({"y1": ("a", "a"), "y2": ("a", "a"), "y3": ("a", "a")})
    result = resample_metrics(
        {"x": {0: x}, "y": {0: y}},
        "accuracy",
        resamples=400,
        seed=5,
        family_resampling="per-system",
    )
    assert set(result.samples["x"]) == {0.0, 0.5, 1.0}
    assert set(result.samples["y"]) == {1.0}
    assert result.estimates == pytest.approx({"x": 0.5, "y": 1.0})
    assert "per-system" in result.method


def test_per_system_draws_are_independent_across_identical_systems() -> None:
    x = _family_set({"x1": ("a", "a"), "x2": ("a", "b")})
    y = _family_set({"y1": ("a", "a"), "y2": ("a", "b")})
    per_system = resample_metrics(
        {"x": {0: x}, "y": {0: y}},
        "accuracy",
        300,
        seed=2,
        family_resampling="per-system",
    )
    assert not np.array_equal(per_system.samples["x"], per_system.samples["y"])
    # Mean differences are centred on zero because the systems are identical.
    assert abs(per_system.difference("x", "y").estimate) < 1e-12


def test_class_stratified_draws_never_change_the_macro_f1_estimand() -> None:
    # Class "b" has one family and is always mispredicted; class "a" has many
    # families and is always right. Macro-F1 over the two supported classes is
    # (1 + 0) / 2 in every draw. An unstratified draw would lose class "b" and
    # report a macro-F1 of 1 over the single remaining class.
    families = {f"a{i}": ("a", "a") for i in range(8)} | {"b0": ("b", "a")}
    system = _family_set(families)
    for mode in ("shared", "per-system"):
        result = resample_metrics(
            {"s": {0: system}},
            "macro_f1",
            resamples=300,
            seed=11,
            family_resampling=mode,
        )
        # Each stratum keeps its family count: eight "a" images (all right),
        # nine predicted "a", one "b" image (wrong). F1_a = 16/17, F1_b = 0.
        assert result.samples["s"] == pytest.approx(np.full(300, 8 / 17))


def test_class_support_assertion_names_the_lost_class() -> None:
    require_class_support(("a", "b"), np.array([3, 1]), np.array([2, 4]))
    with pytest.raises(ValueError, match="'b'"):
        require_class_support(("a", "b"), np.array([3, 1]), np.array([5, 0]))
    # A class absent from the native set carries no obligation.
    require_class_support(("a", "b"), np.array([3, 0]), np.array([5, 0]))
    with pytest.raises(ValueError):
        require_class_support(("a", "b"), np.array([3]), np.array([5, 0]))


def test_mixed_label_families_are_resampled_as_their_own_stratum() -> None:
    # Pool C keeps scenes that appear under two labels. A family that spans two
    # classes is its own stratum, so both of its classes stay supported.
    system = PredictionSet(
        ("a", "b"),
        ("i0", "i1", "i2", "i3"),
        ("a", "b", "a", "a"),
        ("mixed", "mixed", "solo", "solo2"),
        ("s",) * 4,
        {1: ("a", "a", "a", "a")},
    )
    result = resample_metrics(
        {"s": {0: system}}, "macro_f1", 200, seed=1, family_resampling="shared"
    )
    # The mixed family's "b" image is always mispredicted, and the stratum keeps
    # that family in every draw, so class "b" never drops out of the macro mean.
    assert np.all(result.samples["s"] < 1.0)


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


def test_checkpoint_bootstrap_is_reproducible_across_hash_seeds() -> None:
    # Checkpoint labels are strings, so set order would vary with PYTHONHASHSEED.
    script = textwrap.dedent(
        """
        from jute_disease.evaluation import PredictionSet, resample_metrics

        patterns = [("a", "b"), ("b", "b"), ("b", "a")]

        def make(seed_ids):
            predictions = {s: patterns[s % 3] for s in seed_ids}
            return PredictionSet(
                ("a", "b"), ("i0", "i1"), ("a", "b"), ("f0", "f1"), ("s", "s"),
                predictions,
            )

        # Unequal checkpoint sizes, so which checkpoint receives each draw matters.
        split0 = make(range(1, 9))
        split1 = make(range(9, 17))
        checkpoints = {
            0: {s: ("alpha" if s <= 4 else "beta" if s <= 6 else "gamma") for s in range(1, 9)},
            1: {s: ("delta" if s <= 12 else "beta" if s <= 14 else "gamma") for s in range(9, 17)},
        }
        result = resample_metrics(
            {"route": {0: split0, 1: split1}}, "accuracy", 40, seed=42,
            checkpoints=checkpoints,
        )
        print(result.samples["route"].tobytes().hex())
        """
    )
    outputs = set()
    for hash_seed in ("1", "3"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1


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

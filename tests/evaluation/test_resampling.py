import numpy as np
import pytest

from jute_disease.evaluation import (
    CONFIDENCE_LEVEL,
    EVAL_SEED,
    RESAMPLES,
    evaluate_predictions,
    resample_metrics,
)

from .conftest import SetFactory

# Class "a" has two groups of two images, class "b" one group of one image.
#   g1: a a   g2: a a   g3: b
# Resampling draws two "a" groups with replacement and always keeps g3, so
# with k copies of g1 the correct count under a prediction that gets g1 right
# and g2 wrong is 2k + 1 of 5 images.
TRUTH = ["a", "a", "a", "a", "b"]
GROUPS = ["g1", "g1", "g2", "g2", "g3"]
G1_RIGHT = ["a", "a", "b", "b", "b"]
G2_RIGHT = ["b", "b", "a", "a", "b"]
PERFECT = TRUTH


def test_defaults_follow_the_declared_protocol() -> None:
    assert RESAMPLES == 10_000
    assert EVAL_SEED == 42
    assert CONFIDENCE_LEVEL == 0.95


def test_groups_move_together_and_strata_are_kept(make_set: SetFactory) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))

    result = resample_metrics({"m": system}, "accuracy", resamples=4000)

    values = {round(v, 6) for v in result.samples["m"]}
    # Per-image resampling would also give 0.4 and 0.8; a pooled resample
    # could lose the lone "b" group and reach 0.0.
    assert values == {0.2, 0.6, 1.0}
    share = np.mean(np.isclose(result.samples["m"], 0.6))
    assert share == pytest.approx(0.5, abs=0.03)


def test_interval_is_the_percentile_interval(make_set: SetFactory) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))

    result = resample_metrics({"m": system}, "accuracy")
    interval = result.interval("m")

    assert result.resamples == 10_000
    assert result.seed == 42
    assert interval.estimate == pytest.approx(0.6)
    assert interval.lower == pytest.approx(0.2)
    assert interval.upper == pytest.approx(1.0)
    assert interval.level == 0.95
    low, high = np.percentile(result.samples["m"], [2.5, 97.5], method="linear")
    assert (interval.lower, interval.upper) == pytest.approx((low, high))


def test_resampling_is_deterministic_and_seed_dependent(
    make_set: SetFactory,
) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))

    first = resample_metrics({"m": system}, "accuracy", resamples=500)
    again = resample_metrics({"m": system}, "accuracy", resamples=500)
    other = resample_metrics({"m": system}, "accuracy", resamples=500, seed=7)

    assert np.array_equal(first.samples["m"], again.samples["m"])
    assert not np.array_equal(first.samples["m"], other.samples["m"])


def test_seeds_are_resampled_jointly_before_averaging(
    make_set: SetFactory,
) -> None:
    # Seed 1 is right on g1 and seed 2 on g2, so on any one resample their
    # accuracies are (2k+1)/5 and (2(2-k)+1)/5 and the seed mean is always 3/5.
    system = make_set(
        TRUTH,
        {1: G1_RIGHT, 2: G2_RIGHT},
        groups=GROUPS,
        classes=("a", "b"),
    )

    result = resample_metrics({"m": system}, "accuracy", resamples=1000)

    assert np.allclose(result.samples["m"], 0.6)
    interval = result.interval("m")
    assert (interval.lower, interval.upper) == pytest.approx((0.6, 0.6))


def test_identical_systems_have_zero_paired_difference(
    make_set: SetFactory,
) -> None:
    a = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))
    b = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))

    diff = resample_metrics({"a": a, "b": b}, "accuracy", resamples=500).difference(
        "a", "b"
    )

    assert diff.estimate == 0.0
    assert (diff.lower, diff.upper) == (0.0, 0.0)


def test_paired_difference_uses_the_same_resamples(make_set: SetFactory) -> None:
    # "a" gets the lone "b" image right and "b" gets it wrong; both are right
    # on g1 only. Accuracies are (2k+1)/5 vs 2k/5 on every resample, so the
    # paired difference is exactly 0.2 and shows no spread.
    right = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))
    wrong = make_set(
        TRUTH, {1: ["a", "a", "b", "b", "a"]}, groups=GROUPS, classes=("a", "b")
    )

    result = resample_metrics({"right": right, "wrong": wrong}, "accuracy", 800)
    diff = result.difference("right", "wrong")

    assert diff.estimate == pytest.approx(0.2)
    assert (diff.lower, diff.upper) == pytest.approx((0.2, 0.2))
    assert result.interval("right").upper > result.interval("right").lower


def test_paired_difference_averages_seeds_per_system(make_set: SetFactory) -> None:
    perfect = make_set(
        TRUTH, {1: PERFECT, 2: PERFECT}, groups=GROUPS, classes=("a", "b")
    )
    mixed = make_set(
        TRUTH, {1: G1_RIGHT, 2: G2_RIGHT}, groups=GROUPS, classes=("a", "b")
    )

    diff = resample_metrics(
        {"perfect": perfect, "mixed": mixed}, "accuracy", resamples=400
    ).difference("perfect", "mixed")

    assert diff.estimate == pytest.approx(0.4)
    assert (diff.lower, diff.upper) == pytest.approx((0.4, 0.4))


def test_macro_f1_estimate_matches_the_seed_mean(make_set: SetFactory) -> None:
    truth = ["a", "a", "a", "b", "b", "b"]
    system = make_set(
        truth,
        {1: ["a", "a", "b", "b", "b", "c"], 2: truth},
        groups=["g1", "g1", "g2", "g3", "g3", "g4"],
    )

    result = resample_metrics({"m": system}, "macro_f1", resamples=300)

    expected = evaluate_predictions(system).summary["macro_f1"].mean
    assert result.interval("m").estimate == pytest.approx(expected)
    assert np.isfinite(result.samples["m"]).all()


def test_receipt_records_groups_and_degenerate_strata(make_set: SetFactory) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b", "c"))

    result = resample_metrics({"m": system}, "accuracy", resamples=50)

    assert result.groups_per_class == {"a": 2, "b": 1, "c": 0}
    assert result.degenerate_classes == ("b",)
    assert result.unsupported_classes == ("c",)
    assert "percentile" in result.method
    assert "within class" in result.method


def test_systems_must_share_images_and_groups(make_set: SetFactory) -> None:
    a = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))
    regrouped = make_set(
        TRUTH,
        {1: G1_RIGHT},
        groups=["g1", "g2", "g2", "g1", "g3"],
        classes=("a", "b"),
    )

    with pytest.raises(ValueError, match="share"):
        resample_metrics({"a": a, "b": regrouped}, "accuracy", resamples=10)


def test_group_spanning_classes_is_rejected(make_set: SetFactory) -> None:
    mixed = make_set(["a", "b"], {1: ["a", "b"]}, groups=["g", "g"], classes=("a", "b"))

    with pytest.raises(ValueError, match="more than one class"):
        resample_metrics({"m": mixed}, "accuracy", resamples=10)


@pytest.mark.parametrize(
    ("metric", "resamples", "level"),
    [("auc", 10, 0.95), ("accuracy", 0, 0.95), ("accuracy", 10, 1.0)],
)
def test_invalid_arguments_are_rejected(
    make_set: SetFactory, metric: str, resamples: int, level: float
) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))

    with pytest.raises(ValueError):
        resample_metrics({"m": system}, metric, resamples=resamples, level=level)


def test_unknown_system_name_is_rejected(make_set: SetFactory) -> None:
    system = make_set(TRUTH, {1: G1_RIGHT}, groups=GROUPS, classes=("a", "b"))
    result = resample_metrics({"m": system}, "accuracy", resamples=10)

    with pytest.raises(ValueError, match="system"):
        result.interval("other")
    with pytest.raises(ValueError, match="system"):
        result.difference("m", "other")


def test_no_systems_are_rejected() -> None:
    with pytest.raises(ValueError):
        resample_metrics({}, "accuracy", resamples=10)

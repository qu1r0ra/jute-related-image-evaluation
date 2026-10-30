import math

import pytest

from jute_disease.evaluation import (
    PredictionSet,
    evaluate_predictions,
    score_run,
)

from .conftest import SetFactory

# Hand-checkable fixture. Class "c" has no true images (unsupported).
#   img:   0  1  2  3  4  5
#   truth: a  a  a  b  b  b
#   pred:  a  a  b  b  b  c
# Confusion (rows true a,b,c; columns predicted a,b,c):
#   a: [2, 1, 0]   b: [0, 2, 1]   c: [0, 0, 0]
TRUTH = ["a", "a", "a", "b", "b", "b"]
SEED1 = ["a", "a", "b", "b", "b", "c"]
SOURCES = ["x", "x", "x", "y", "y", "y"]


def test_single_run_scores_match_hand_calculation(make_set: SetFactory) -> None:
    scores = score_run(make_set(TRUTH, {1: SEED1}), seed=1)

    assert scores.confusion == ((2, 1, 0), (0, 2, 1), (0, 0, 0))
    assert scores.accuracy == pytest.approx(4 / 6)
    a, b, c = scores.classes
    assert (a.label, a.support) == ("a", 3)
    assert a.precision == pytest.approx(1.0)
    assert a.recall == pytest.approx(2 / 3)
    assert a.f1 == pytest.approx(4 / 5)
    assert b.precision == pytest.approx(2 / 3)
    assert b.recall == pytest.approx(2 / 3)
    assert b.f1 == pytest.approx(2 / 3)
    assert scores.macro_f1 == pytest.approx((4 / 5 + 2 / 3) / 2)
    assert c.support == 0


def test_unsupported_class_is_reported_not_scored(make_set: SetFactory) -> None:
    scores = score_run(make_set(TRUTH, {1: SEED1}), seed=1)

    c = scores.classes[2]
    assert (c.precision, c.recall, c.f1) == (None, None, None)
    assert scores.unsupported == ("c",)
    assert not math.isnan(scores.macro_f1)


def test_supported_class_never_predicted_scores_zero_recall(
    make_set: SetFactory,
) -> None:
    scores = score_run(make_set(["a", "a", "b"], {1: ["a", "a", "a"]}), seed=1)

    b = scores.classes[1]
    assert b.support == 1
    assert b.precision is None
    assert b.recall == 0.0
    assert b.f1 == 0.0
    assert scores.macro_f1 == pytest.approx((0.8 + 0.0) / 2)


def test_seed_summary_uses_mean_and_sample_sd(make_set: SetFactory) -> None:
    seed2 = ["a", "a", "a", "b", "b", "c"]
    seed3 = TRUTH
    evaluation = evaluate_predictions(
        make_set(TRUTH, {1: SEED1, 2: seed2, 3: seed3}, sources=SOURCES)
    )

    assert sorted(evaluation.overall) == [1, 2, 3]
    accuracy = evaluation.summary["accuracy"]
    assert accuracy.n == 3
    assert accuracy.mean == pytest.approx(5 / 6)
    assert accuracy.sd == pytest.approx(1 / 6)


def test_single_seed_has_no_sample_sd(make_set: SetFactory) -> None:
    evaluation = evaluate_predictions(make_set(TRUTH, {42: SEED1}))

    assert evaluation.summary["accuracy"].n == 1
    assert evaluation.summary["accuracy"].sd is None


def test_per_source_scores_report_support(make_set: SetFactory) -> None:
    evaluation = evaluate_predictions(make_set(TRUTH, {1: SEED1}, sources=SOURCES))

    assert sorted(evaluation.by_source) == ["x", "y"]
    x = evaluation.by_source["x"][1]
    assert x.n == 3
    assert x.accuracy == pytest.approx(2 / 3)
    assert x.unsupported == ("b", "c")
    assert x.classes[0].support == 3
    assert x.macro_f1 == pytest.approx(4 / 5)
    y = evaluation.by_source["y"][1]
    assert y.accuracy == pytest.approx(2 / 3)
    assert y.classes[1].recall == pytest.approx(2 / 3)
    assert evaluation.source_summary["x"]["accuracy"].mean == pytest.approx(2 / 3)


def test_class_summary_averages_defined_seeds_only(make_set: SetFactory) -> None:
    truth = ["a", "a", "b"]
    evaluation = evaluate_predictions(
        make_set(truth, {1: ["a", "a", "a"], 2: ["a", "a", "b"]})
    )

    precision_b = evaluation.class_summary["b"]["precision"]
    assert precision_b.n == 1
    assert precision_b.mean == pytest.approx(1.0)
    assert evaluation.class_summary["b"]["recall"].mean == pytest.approx(0.5)
    assert evaluation.class_summary["c"]["f1"].n == 0
    assert evaluation.class_summary["c"]["f1"].mean is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: PredictionSet(
            ("a", "b"), ("i",), ("a", "b"), ("g",), ("s",), {1: ("a",)}
        ),
        lambda: PredictionSet(("a", "b"), ("i",), ("z",), ("g",), ("s",), {1: ("a",)}),
        lambda: PredictionSet(("a", "b"), ("i",), ("a",), ("g",), ("s",), {1: ("z",)}),
        lambda: PredictionSet(
            ("a", "b"), ("i",), ("a",), ("g",), ("s",), {1: ("a", "b")}
        ),
        lambda: PredictionSet(
            ("a", "b"), ("i", "i"), ("a", "a"), ("g", "g"), ("s", "s"), {1: ("a", "a")}
        ),
        lambda: PredictionSet(("a", "b"), ("i",), ("a",), ("g",), ("s",), {}),
        lambda: PredictionSet(("a", "b"), (), (), (), (), {1: ()}),
        lambda: PredictionSet(("a", "a"), ("i",), ("a",), ("g",), ("s",), {1: ("a",)}),
    ],
    ids=[
        "truth-length",
        "unknown-truth-label",
        "unknown-predicted-label",
        "prediction-length",
        "duplicate-image-id",
        "no-seeds",
        "no-images",
        "duplicate-class",
    ],
)
def test_malformed_prediction_sets_are_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_score_run_rejects_unknown_seed(make_set: SetFactory) -> None:
    with pytest.raises(ValueError, match="seed"):
        score_run(make_set(TRUTH, {1: SEED1}), seed=2)

"""Unit tests for all SklearnClassifier subclasses."""

# ruff: noqa: N803, N806
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jute_disease.config import DEFAULT_SEED
from jute_disease.models.ml import (
    GaussianNaiveBayes,
    KNearestNeighbors,
    LogisticRegression,
    RandomForest,
    SupportVectorMachine,
)


@pytest.fixture
def xy() -> tuple[np.ndarray, np.ndarray]:
    """Tiny 2-class dataset."""
    rng = np.random.default_rng(DEFAULT_SEED)
    X = np.abs(rng.random((30, 8))).astype(np.float32)
    y = np.array([0, 1] * 15)
    return X, y


@pytest.mark.parametrize(
    "cls",
    [LogisticRegression, RandomForest, SupportVectorMachine, GaussianNaiveBayes],
)
def test_classifier_fit_predict(cls: Any, xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    model = cls()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("cls", [LogisticRegression, RandomForest, GaussianNaiveBayes])
def test_classifier_predict_proba(cls: Any, xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    model = cls()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_knn_fit_predict(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    model = KNearestNeighbors(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_knn_ignores_sample_weight(
    xy: tuple[np.ndarray, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    """KNN does not support sample_weight — must log a warning, not crash."""
    X, y = xy
    sw = np.ones(30)
    model = KNearestNeighbors(n_neighbors=3)
    model.fit(X, y, sample_weight=sw)
    assert "does not support sample_weight" in caplog.text


def test_svm_is_never_calibrated(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """The fixed SVC has no Platt calibration: no internal CV, no probabilities."""
    X, y = xy
    model = SupportVectorMachine()
    model.fit(X, y)
    assert model.model.named_steps["clf"].probability is False
    with pytest.raises(AttributeError):
        model.predict_proba(X)


def test_weighting_is_recorded_as_applied_or_unsupported(
    xy: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = xy
    weights = np.ones(30)
    for cls in (
        LogisticRegression,
        RandomForest,
        SupportVectorMachine,
        GaussianNaiveBayes,
    ):
        model = cls()
        model.fit(X, y, sample_weight=weights)
        assert model.weighting == "applied"
    knn = KNearestNeighbors(n_neighbors=3)
    knn.fit(X, y, sample_weight=weights)
    assert knn.weighting == "unsupported"
    plain = LogisticRegression()
    plain.fit(X, y)
    assert plain.weighting == "none"


def test_scaler_is_fit_on_training_rows_only(xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = xy
    model = LogisticRegression()
    model.fit(X, y)
    scaler = model.model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    model.predict(X * 100)
    assert np.allclose(scaler.mean_, X.mean(axis=0))


def test_classifiers_with_sample_weight(xy: tuple[np.ndarray, np.ndarray]) -> None:
    """Classifiers that support sample_weight must accept it without error."""
    X, y = xy
    sw = np.ones(30)
    for cls in [LogisticRegression, RandomForest, SupportVectorMachine]:
        model = cls()
        model.fit(X, y, sample_weight=sw)


def test_classifier_save_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xy: tuple[np.ndarray, np.ndarray]
) -> None:
    from jute_disease.models.ml import classifiers

    monkeypatch.setattr(classifiers, "ML_MODELS_DIR", tmp_path)

    X, y = xy
    model = RandomForest()
    model.fit(X, y)
    model.save()

    assert (tmp_path / "randomforest.joblib").exists()

    loaded = RandomForest.load()
    assert loaded is not None
    preds = loaded.predict(X)
    assert preds.shape == (30,)


def test_classifiers_deterministic_seed() -> None:
    """Verifies that passing a random_state yields identical models."""
    from jute_disease.models.ml import RandomForest

    rng = np.random.default_rng(DEFAULT_SEED)
    X = np.abs(rng.random((50, 8))).astype(np.float32)
    y = np.random.randint(0, 2, size=50)

    # Without seeding, random forests are non-deterministic between instances.
    # With a seed, their parameters should be identical after fitting.
    rf1 = RandomForest(random_state=DEFAULT_SEED)
    rf2 = RandomForest(random_state=DEFAULT_SEED)

    rf1.fit(X, y)
    rf2.fit(X, y)

    preds1 = rf1.predict_proba(X)
    preds2 = rf2.predict_proba(X)

    assert np.allclose(preds1, preds2)


def test_classifiers_unsupported_random_state_ignored() -> None:
    """Classifiers that do not support random_state shouldn't crash if passed one."""
    from jute_disease.models.ml import GaussianNaiveBayes, KNearestNeighbors

    knn = KNearestNeighbors(random_state=DEFAULT_SEED)
    gnb = GaussianNaiveBayes(random_state=DEFAULT_SEED)

    assert knn is not None
    assert gnb is not None

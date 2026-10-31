# ruff: noqa: N803, N806
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from jute_disease.models.ml import SklearnClassifier


class MockEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, check_sample_weight: bool = False) -> None:
        self.check_sample_weight = check_sample_weight
        self.sample_weight_passed: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> MockEstimator:
        self.sample_weight_passed = sample_weight
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 2))


def test_sklearn_classifier_adapter_passes_weight() -> None:
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    sw = np.random.rand(10)

    adapter = SklearnClassifier(MockEstimator)
    adapter.fit(X, y, sample_weight=sw)

    clf_step = adapter.model.named_steps["clf"]
    assert isinstance(clf_step, MockEstimator)
    assert clf_step.sample_weight_passed is not None
    assert np.array_equal(clf_step.sample_weight_passed, sw)


def test_sklearn_classifier_warns_no_weight(caplog: pytest.LogCaptureFixture) -> None:
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    sw = np.random.rand(10)

    # Force supports_sample_weight to False for testing warning
    class NoWeightAdapter(SklearnClassifier):
        supports_sample_weight = False

    adapter = NoWeightAdapter(MockEstimator)
    adapter.fit(X, y, sample_weight=sw)

    clf_step = adapter.model.named_steps["clf"]
    assert "does not support sample_weight" in caplog.text
    assert isinstance(clf_step, MockEstimator)
    assert clf_step.sample_weight_passed is None


def test_sklearn_classifier_predict() -> None:
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    adapter = SklearnClassifier(MockEstimator)
    adapter.fit(X, y)

    y_pred = adapter.predict(X)
    assert len(y_pred) == 10
    assert isinstance(y_pred, np.ndarray)


def test_sklearn_classifier_save_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jute_disease.models.ml import classifiers

    monkeypatch.setattr(classifiers, "ML_MODELS_DIR", tmp_path)

    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)

    class PersistentClassifier(SklearnClassifier):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(MockEstimator, **kwargs)

    adapter = PersistentClassifier()
    adapter.fit(X, y)
    adapter.save()

    # Verify file exists (lowercase class name)
    expected_path = tmp_path / "persistentclassifier.joblib"
    assert expected_path.exists()

    # Load and verify
    loaded = PersistentClassifier.load()
    assert loaded is not None
    loaded_clf = loaded.model.named_steps["clf"]
    assert isinstance(loaded_clf, MockEstimator)
    # Check if fit attributes survived
    assert loaded_clf.sample_weight_passed is None

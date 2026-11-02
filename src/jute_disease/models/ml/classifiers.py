from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from jute_disease.config import ML_MODELS_DIR
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


class FeatureBlockWeights(TransformerMixin, BaseEstimator):
    """Apply equal L2 normalization across fixed descriptor blocks."""

    def __init__(self, block_sizes: tuple[int, ...]) -> None:
        self.block_sizes = block_sizes

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> FeatureBlockWeights:
        if sum(self.block_sizes) != x.shape[1]:
            raise ValueError("Feature block sizes do not match the feature matrix")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        weighted = x.astype(np.float64, copy=True)
        start = 0
        for size in self.block_sizes:
            weighted[:, start : start + size] /= np.sqrt(size)
            start += size
        return weighted


class ScaleRelativeRbfSVC(ClassifierMixin, BaseEstimator):
    """RBF-SVC whose gamma is a multiplier of training-fold ``scale``."""

    def __init__(
        self,
        C: float = 1.0,
        gamma_scale_multiplier: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.gamma_scale_multiplier = gamma_scale_multiplier
        self.random_state = random_state

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> ScaleRelativeRbfSVC:
        from sklearn.svm import SVC

        variance = float(np.var(x))
        scale = 1.0 if variance == 0.0 else 1.0 / (x.shape[1] * variance)
        self.gamma_ = self.gamma_scale_multiplier * scale
        self.estimator_ = SVC(
            C=self.C,
            gamma=self.gamma_,
            kernel="rbf",
            probability=False,
            random_state=self.random_state,
        )
        self.estimator_.fit(x, y, sample_weight=sample_weight)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(x)


def classical_configurations(estimator: str) -> tuple[dict[str, object], ...]:
    """Return the prespecified nine configurations for a classical baseline."""
    if estimator == "knn":
        return tuple(
            {"n_neighbors": value} for value in (1, 3, 5, 7, 11, 15, 21, 31, 51)
        )
    if estimator == "lr":
        return tuple({"C": 10.0**power} for power in range(-4, 5))
    if estimator == "svm":
        return tuple(
            {"C": c_value, "gamma_scale_multiplier": gamma}
            for c_value in (1, 10, 100)
            for gamma in (0.1, 1.0, 10.0)
        )
    if estimator == "rf":
        return tuple(
            {"max_features": features, "min_samples_leaf": leaf}
            for features in ("sqrt", 0.1, 0.3)
            for leaf in (1, 3, 10)
        )
    if estimator == "gnb":
        return tuple({"var_smoothing": 10.0**power} for power in range(-12, -3))
    raise ValueError(f"Unknown classical estimator {estimator!r}")


def build_classical_classifier(
    estimator: str,
    seed: int,
    features: str,
    parameters: dict[str, object] | None = None,
    *,
    dimensions: int | None = None,
) -> SklearnClassifier:
    """Build a train-scaled classifier with the selected fixed settings."""
    values = dict(parameters or classical_configurations(estimator)[0])
    fixed = {
        "knn": {"metric": "euclidean"},
        "lr": {"penalty": "l2"},
        "rf": {"n_estimators": 500},
        "svm": {},
        "gnb": {},
    }
    if estimator not in ML_CLASSIFIERS:
        raise ValueError(f"Unknown classical estimator {estimator!r}")
    if estimator == "svm":
        multiplier = float(values.pop("gamma_scale_multiplier"))
        classifier = SklearnClassifier(
            ScaleRelativeRbfSVC,
            random_state=seed,
            gamma_scale_multiplier=multiplier,
            **values,
        )
    else:
        classifier = ML_CLASSIFIERS[estimator](
            random_state=seed, **fixed[estimator], **values
        )
    if features == "crafted":
        if dimensions is not None and dimensions != sum((512, 28, 8)):
            raise ValueError("Handcrafted descriptor dimensions are not 548")
        classifier.model = Pipeline(
            [
                *classifier.model.steps[:1],
                ("block_weights", FeatureBlockWeights((512, 28, 8))),
                *classifier.model.steps[1:],
            ]
        )
    return classifier


def select_classical_configuration(
    estimator: str,
    features: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_reserve: np.ndarray,
    y_reserve: np.ndarray,
) -> dict[str, object]:
    """Choose the highest macro-F1 configuration using only train and reserve."""
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight

    best: dict[str, object] | None = None
    best_score = float("-inf")
    rows: list[dict[str, object]] = []
    sample_weight = compute_sample_weight("balanced", y_train)
    for parameters in classical_configurations(estimator):
        model = build_classical_classifier(
            estimator, seed, features, parameters, dimensions=x_train.shape[1]
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        score = float(f1_score(y_reserve, model.predict(x_reserve), average="macro"))
        rows.append({"parameters": parameters, "reserve_macro_f1": score})
        if score > best_score:
            best, best_score = parameters, score
    if best is None:
        raise ValueError("No classical configurations were evaluated")
    return {
        "estimator": estimator,
        "features": features,
        "metric": "macro_f1",
        "configurations": rows,
        "selected": best,
        "selected_macro_f1": best_score,
        "test_accessed": False,
    }


class SklearnClassifier:
    """Adapter that wraps any scikit-learn classifier into a consistent interface."""

    supports_sample_weight: bool = True
    scaler_cls: type = StandardScaler
    weighting: str = "none"

    def __init__(self, sklearn_cls: type[ClassifierMixin], **kwargs: object) -> None:
        if not issubclass(sklearn_cls, ClassifierMixin):
            raise TypeError(f"{sklearn_cls} must be a scikit-learn classifier.")

        steps = []
        if self.scaler_cls is not None:
            steps.append(("scaler", self.scaler_cls()))
        steps.append(("clf", sklearn_cls(**kwargs)))

        self.model = Pipeline(steps)

    def __repr__(self) -> str:
        clf_step = self.model.named_steps["clf"]
        params_str = ", ".join(f"{k}={v!r}" for k, v in clf_step.get_params().items())
        return f"{self.__class__.__name__}({params_str})"

    def __str__(self) -> str:
        return f"Classical ML Wrapper: {self.__class__.__name__}"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None:
        fit_kwargs = {}
        self.weighting = "none"
        if sample_weight is not None:
            if not self.supports_sample_weight:
                self.weighting = "unsupported"
                logger.warning(
                    f"{self.__class__.__name__} does not support sample_weight. "
                    "Weighting will be ignored."
                )
            else:
                self.weighting = "applied"
                fit_kwargs["clf__sample_weight"] = sample_weight

        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, name: str | None = None) -> None:
        ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_name = name or self.__class__.__name__.lower()
        path = str(ML_MODELS_DIR / f"{model_name}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, name: str | None = None) -> SklearnClassifier | None:
        model_name = name or cls.__name__.lower()
        path = ML_MODELS_DIR / f"{model_name}.joblib"
        if not path.exists():
            return None
        instance = cls.__new__(cls)
        instance.model = joblib.load(str(path))
        return instance


class KNearestNeighbors(SklearnClassifier):
    supports_sample_weight = False

    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(KNeighborsClassifier, **kwargs)


class SupportVectorMachine(SklearnClassifier):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        kwargs["probability"] = False
        if random_state is not None:
            kwargs["random_state"] = random_state
        super().__init__(SVC, **kwargs)


class LogisticRegression(SklearnClassifier):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        kwargs.setdefault("max_iter", 1000)
        if random_state is not None:
            kwargs["random_state"] = random_state
        super().__init__(SKLogisticRegression, **kwargs)


class RandomForest(SklearnClassifier):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        if random_state is not None:
            kwargs["random_state"] = random_state
        super().__init__(RandomForestClassifier, **kwargs)


class GaussianNaiveBayes(SklearnClassifier):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(GaussianNB, **kwargs)


ML_CLASSIFIERS: dict[str, Any] = {
    "gnb": GaussianNaiveBayes,
    "knn": KNearestNeighbors,
    "lr": LogisticRegression,
    "rf": RandomForest,
    "svm": SupportVectorMachine,
}

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
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


class SklearnClassifier:
    """Adapter that wraps any scikit-learn classifier into a consistent interface."""

    supports_sample_weight: bool = True
    scaler_cls: type = StandardScaler

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
        if sample_weight is not None:
            if not self.supports_sample_weight:
                logger.warning(
                    f"{self.__class__.__name__} does not support sample_weight. "
                    "Weighting will be ignored."
                )
            else:
                fit_kwargs["clf__sample_weight"] = sample_weight

        self.model.fit(X, y, **fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, name: str | None = None, directory: Path | None = None) -> Path:
        target = ML_MODELS_DIR if directory is None else directory
        target.mkdir(parents=True, exist_ok=True)
        model_name = name or self.__class__.__name__.lower()
        path = target / f"{model_name}.joblib"
        joblib.dump(self.model, str(path))
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, name: str | None = None) -> "SklearnClassifier | None":
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
        kwargs.setdefault("probability", True)
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

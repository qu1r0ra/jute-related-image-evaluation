import torch
from sklearn.metrics import classification_report

from jute_disease.config import DEFAULT_SEED
from jute_disease.data import classical_pipeline, open_partition
from jute_disease.evaluation.metrics import EVAL_METRICS, format_metrics
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    ML_CLASSIFIERS,
    extract_features,
)
from jute_disease.training.logger import get_logger, write_evaluation
from jute_disease.training.seed import seed_everything

logger = get_logger(__name__)


def test_ml(
    classifier: str = "rf",
    feature_type: str = "crafted",
    seed: int = DEFAULT_SEED,
) -> None:
    seed_everything(seed)

    # Load Model
    classifier_cls = ML_CLASSIFIERS[classifier]
    model = classifier_cls.load(f"{classifier}_{feature_type}")
    if model is None:
        logger.error(f"No saved model found for {classifier}. Please train it first.")
        return

    logger.info(f"Loaded {classifier} for evaluation.")

    # Setup Feature Extractor
    extractor_cls = FEATURE_EXTRACTORS[feature_type]
    extractor = extractor_cls()

    # Load Test Data
    try:
        test_ds = open_partition("test", classical_pipeline())
    except FileNotFoundError as error:
        logger.error(f"No completed freeze to evaluate against: {error}")
        return
    class_names = test_ds.classes

    # Extract Features
    X_test, y_test = extract_features(test_ds, extractor=extractor)

    # Predict and Evaluate
    logger.info("Running predictions on test set...")
    y_pred = model.predict(X_test)

    evaluator = EVAL_METRICS.clone()
    test_out = evaluator(torch.tensor(y_pred), torch.tensor(y_test))
    test_metrics = format_metrics(test_out, prefix="test_")

    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['test_f1']:.4f}")
    logger.info(
        "\nClassification Report:\n"
        + classification_report(y_test, y_pred, target_names=class_names)
    )

    write_evaluation(
        f"{classifier}_{feature_type}",
        test_metrics,
        y_test,
        y_pred,
        class_names,
        {"seed": seed, "classifier": classifier, "feature_type": feature_type},
    )

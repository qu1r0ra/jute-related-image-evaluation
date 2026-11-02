import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from jute_disease.config import MANIFEST_DIR, SOURCE_INVENTORY_DIR
from jute_disease.data import REPORT_FILE, load_manifest
from jute_disease.models.ml import (
    BaseFeatureExtractor,
    CraftedFeatureExtractor,
    GaussianNaiveBayes,
    KNearestNeighbors,
    LogisticRegression,
    RandomForest,
    RawPixelFeatureExtractor,
    SupportVectorMachine,
)
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)

ML_CLASSIFIERS: dict[str, Any] = {
    "gnb": GaussianNaiveBayes,
    "knn": KNearestNeighbors,
    "lr": LogisticRegression,
    "rf": RandomForest,
    "svm": SupportVectorMachine,
}


def _guard_target_image_path(
    image_path: Path, target_dir: Path, image_root: Path | None
) -> None:
    """Refuse direct prediction reads of target files outside train/validation."""
    report_path = target_dir.parent / REPORT_FILE
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report = None
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"Prediction blocked: cannot verify freeze report: {error}"
        ) from error
    if report is not None and not isinstance(report, dict):
        raise SystemExit("Prediction blocked: freeze report is invalid")
    recorded_root = report.get("image_root") if report is not None else None
    if not isinstance(recorded_root, str) and report is not None:
        raise SystemExit("Prediction blocked: freeze report has an invalid image root")
    if image_root is not None and recorded_root is not None:
        if image_root.resolve(strict=False) != Path(recorded_root).resolve(
            strict=False
        ):
            raise SystemExit(
                "Prediction blocked: image root differs from the freeze report"
            )
    if image_root is None:
        if recorded_root is not None:
            image_root = Path(recorded_root)
        elif target_dir.resolve(strict=False) != MANIFEST_DIR.resolve(strict=False):
            raise SystemExit(
                "Prediction blocked: custom target freezes require "
                "a verified image root"
            )
        else:
            image_root = SOURCE_INVENTORY_DIR

    lexical_path = Path(os.path.abspath(image_path))
    lexical_root = Path(os.path.abspath(image_root))
    try:
        lexical_relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        lexical_relative = None

    try:
        resolved_path = image_path.resolve(strict=False)
        resolved_root = image_root.resolve(strict=False)
        resolved_relative = resolved_path.relative_to(resolved_root)
    except OSError, RuntimeError, ValueError:
        resolved_relative = None

    if lexical_relative is not None and resolved_relative is None:
        raise SystemExit(
            "Prediction blocked: target image path resolves outside its frozen root"
        )
    if lexical_relative is None and resolved_relative is None:
        return
    try:
        manifest = load_manifest(target_dir)
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"Prediction blocked: cannot verify target image access: {error}"
        ) from error
    allowed = {
        os.path.normcase(Path(entry.path).as_posix())
        for entry in manifest.entries
        if entry.partition in {"train", "val"}
    }
    candidate_relatives = [
        relative for relative in (lexical_relative, resolved_relative) if relative
    ]
    if any(
        os.path.normcase(relative.as_posix()) not in allowed
        for relative in candidate_relatives
    ):
        raise SystemExit(
            "Prediction blocked: target test images can only be read "
            "by the gated evaluator"
        )


def predict_ml(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Jute Classical ML Inference")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to classify",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to use for prediction",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="crafted",
        choices=["crafted", "raw"],
        help="Type of features the model was trained on",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Frozen target manifest used to guard held-out image paths.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root containing paths recorded in the frozen target manifest.",
    )

    args = parser.parse_args(argv)
    img_path = Path(args.image_path)
    _guard_target_image_path(img_path, args.target_dir, args.image_root)

    # 1. Load Model
    classifier_cls = ML_CLASSIFIERS[args.classifier]
    model = classifier_cls.load(f"{args.classifier}_{args.feature_type}")
    if model is None:
        logger.error(
            f"No saved model found for {args.classifier}. Please train it first."
        )
        return

    # 2. Setup Feature Extractor
    extractor: BaseFeatureExtractor
    if args.feature_type == "crafted":
        extractor = CraftedFeatureExtractor()
    else:
        extractor = RawPixelFeatureExtractor()

    # 3. Load and Process Image
    if not img_path.exists():
        logger.error(f"Image not found at {img_path}")
        return

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    # 4. Extract Features
    logger.info(f"Extracting features from {img_path.name}...")
    feats = extractor.extract(img)
    X = feats.reshape(1, -1)

    # 5. Predict
    prediction = model.predict(X)[0]

    # Try to get class names from the training directory if it exists
    class_names: list[str] | None = None
    try:
        class_names = list(load_manifest(args.target_dir).labels)
    except FileNotFoundError:
        class_names = None

    if class_names and prediction < len(class_names):
        result = class_names[prediction]
        logger.info(f"Prediction: {result} (index: {prediction})")
    else:
        logger.info(f"Prediction: index {prediction}")

    # Optional: Proba
    try:
        proba = model.predict_proba(X)[0]
        if class_names:
            logger.info("Class Probabilities:")
            for name, p in zip(class_names, proba, strict=True):
                logger.info(f"  {name}: {p:.4f}")
        else:
            logger.info(f"Probabilities: {proba}")
    except Exception:
        pass

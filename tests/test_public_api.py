"""Validate package APIs and local artifact output without experiment fits."""

import csv
from pathlib import Path


def test_root_package_exposes_subpackages() -> None:
    from jute_disease import config, data, evaluation, models, reporting, training

    assert config is not None
    assert data is not None
    assert evaluation is not None
    assert models is not None
    assert reporting is not None
    assert training is not None


def test_data_api() -> None:
    from jute_disease.data import (
        DECISION_FILE,
        DEFAULT_BUDGET_SECONDS,
        DEFAULT_MAX_BYTES,
        INVENTORY_FILE,
        PLANTDOC,
        PLANTVILLAGE,
        REPORT_FILE,
        SOURCES,
        TARGET_DIR,
        DataModule,
        FreezeBlockedError,
        ManifestContentVerifier,
        ManifestDataset,
        OverlapReport,
        SourceMismatchError,
        SourceSpec,
        SourceSplit,
        SyntheticPlan,
        build_inventory,
        build_manifest,
        check_overlap,
        classical_pipeline,
        create_pipeline,
        derive_validation,
        drop_overlap,
        file_sha256,
        find_candidates,
        find_source_root,
        freeze_target,
        intake_source,
        inventory_digest,
        leaf_constraints,
        leaf_key,
        letterbox_dimensions,
        letterbox_pad_fraction,
        load_manifest,
        load_source_manifest,
        observed_counts,
        probe_cache,
        provenance,
        readiness_blockers,
        scan_source,
        verify_counts,
        verify_manifest_content,
        write_manifest,
    )

    for name in (
        DEFAULT_BUDGET_SECONDS,
        DEFAULT_MAX_BYTES,
        DECISION_FILE,
        DataModule,
        FreezeBlockedError,
        ManifestContentVerifier,
        INVENTORY_FILE,
        ManifestDataset,
        OverlapReport,
        PLANTDOC,
        PLANTVILLAGE,
        REPORT_FILE,
        SOURCES,
        SourceMismatchError,
        SourceSpec,
        SourceSplit,
        SyntheticPlan,
        TARGET_DIR,
        build_manifest,
        build_inventory,
        check_overlap,
        classical_pipeline,
        create_pipeline,
        derive_validation,
        drop_overlap,
        file_sha256,
        find_candidates,
        find_source_root,
        freeze_target,
        intake_source,
        inventory_digest,
        letterbox_dimensions,
        letterbox_pad_fraction,
        leaf_constraints,
        leaf_key,
        load_source_manifest,
        load_manifest,
        observed_counts,
        probe_cache,
        provenance,
        readiness_blockers,
        scan_source,
        verify_counts,
        verify_manifest_content,
        write_manifest,
    ):
        assert name is not None

    from jute_disease import data

    assert all(hasattr(data, name) for name in data.__all__)
    assert len(set(data.__all__)) == len(data.__all__)


def test_models_dl_api() -> None:
    from jute_disease.models.dl import Classifier, TimmBackbone

    assert Classifier is not None
    assert TimmBackbone is not None


def test_models_ml_api() -> None:
    from jute_disease.models.ml import (
        BaseFeatureExtractor,
        CraftedFeatureExtractor,
        GaussianNaiveBayes,
        KNearestNeighbors,
        LogisticRegression,
        RandomForest,
        RawPixelFeatureExtractor,
        SklearnClassifier,
        SupportVectorMachine,
        extract_features,
    )

    assert SklearnClassifier is not None
    assert GaussianNaiveBayes is not None
    assert KNearestNeighbors is not None
    assert LogisticRegression is not None
    assert RandomForest is not None
    assert SupportVectorMachine is not None
    assert BaseFeatureExtractor is not None
    assert CraftedFeatureExtractor is not None
    assert RawPixelFeatureExtractor is not None
    assert extract_features is not None


def test_evaluation_api() -> None:
    from jute_disease.evaluation import EVAL_METRICS, TRAIN_METRICS, format_metrics

    assert EVAL_METRICS is not None
    assert TRAIN_METRICS is not None
    assert format_metrics is not None


def test_training_api() -> None:
    from jute_disease.config import (
        BATCH_SIZE,
        DATA_DIR,
        DEFAULT_SEED,
        IMAGE_SIZE,
        ML_MODELS_DIR,
        ML_SPLIT_DIR,
        ROOT_DIR,
        SPLITS,
    )
    from jute_disease.training import (
        experiment_loggers,
        fit_classical,
        fit_cnn,
        get_logger,
        run_queue,
        seed_everything,
        verify_stack,
    )

    assert ROOT_DIR is not None
    assert DATA_DIR is not None
    assert ML_MODELS_DIR is not None
    assert ML_SPLIT_DIR is not None
    assert isinstance(IMAGE_SIZE, int)
    assert isinstance(BATCH_SIZE, int)
    assert isinstance(DEFAULT_SEED, int)
    assert isinstance(SPLITS, dict)
    assert get_logger is not None
    assert seed_everything is not None
    assert experiment_loggers is not None
    assert fit_classical is not None
    assert fit_cnn is not None
    assert run_queue is not None
    assert verify_stack is not None


def test_pad_fraction_api_reports_letterbox_geometry() -> None:
    from jute_disease.data import letterbox_dimensions, letterbox_pad_fraction

    assert letterbox_dimensions(400, 200, 64) == (64, 32)
    assert letterbox_pad_fraction(400, 200, 64) == 0.5
    assert letterbox_pad_fraction(200, 200, 64) == 0.0


def test_lightning_csv_logger_writes_local_metrics_without_a_fit(
    tmp_path: Path,
) -> None:
    from lightning.pytorch.loggers import CSVLogger

    logger = CSVLogger(save_dir=tmp_path, name="synthetic")
    logger.log_hyperparams({"seed": 42, "classes": ["healthy", "mosaic"]})
    logger.log_metrics({"val_macro_f1": 0.75}, step=1)
    logger.save()

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["val_macro_f1"] == "0.75"
    assert rows[0]["step"] == "1"

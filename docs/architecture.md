# Project Architecture

This document describes the architectural design and directory structure of the `jute-disease-detection` project. It serves as a guide for developers and contributors to understand the core components and their interactions.

## Directory Structure Overview

```text
.
├── configs/
│   └── historical/     # Lightning CLI and grid search configs for the reported runs
├── data/               # Dataset storage (git-ignored) and the tracked class list
├── docs/               # Architecture notes and the submitted paper
├── notebooks/
│   └── historical/     # Notebooks and Jupytext pairs as run for the paper
├── results/
│   └── submitted/      # Committed logs and figures behind the reported numbers
├── src/
│   └── jute_disease/   # Library package and the jute-disease console entry point
├── tests/              # Unit and API tests
├── justfile            # Task runner: one recipe per study command
└── artifacts/          # Runtime output (git-ignored): checkpoints, features, logs
```

## Core Design Principles

### 1. Unified Public API

Each subpackage in `src/jute_disease/` (e.g., `models.ml`, `data`, `training`, `reporting`) uses `__init__.py` to expose a clean, flattened public API.

- **Internal developers** use relative imports where appropriate or `from jute_disease.x import y` to avoid circular dependencies.
- **External consumers** (scripts, tests, notebooks) use the clean package-level imports (e.g., `from jute_disease.models.ml import RandomForest`).

### 2. Command Line Interface (CLI)

The project exposes a single console entry point, `jute-disease` (`jute_disease.cli:main`), and a `justfile` whose recipes each wrap one of its commands:

- **`jute-disease train dl fit|test`**: The deep learning engine. Arguments are forwarded to the **Lightning CLI**, which drives training and testing from configuration files.
- **`jute-disease train ml --all`**: Classical ML training over every classifier and feature combination. It drives the pipeline in `src/jute_disease/training/engines/ml/`.
- **Other commands**: `data` (download, synthetic), `freeze` (target, cache, source), `train` (baselines, cv, grid, pretrain), and `evaluate` (aggregate, ml, predict). `jute-disease --help` lists them all.
- Commands planned in open issues (`queue`, `profile`, and `evaluate` on its own) exit non-zero and name the issue that owns them.

### 3. Deep Learning Service (Lightning + Timm)

The DL pipeline is built using **PyTorch Lightning** for state-of-the-art reproducibility and boilerplate reduction.

- **LightningModule (`Classifier`)**: The core class handling the training loop, optimization, logging, and metrics.
- **Backbone System**: Uses a generic `TimmBackbone` to wrap any model from the `timm` library, allowing for easy experimentation with different architectures.
- **Transfer Learning Strategy**: Generic **ImageNet-1K pre-training** provides the most robust baseline. Our final champion configuration for Jute Disease Detection utilizes **MobileNet V2** at **256x256** resolution, with **Learning Rate: 0.01**, **Weight Decay: 0.05**, and **Dropout: 0.1**. This configuration achieves the optimal balance of features without overfitting, even while hitting the dataset's inherent "single-label" performance ceiling.
- **Lightning CLI**: Training is driven by configuration files in `configs/`, promoting "Configuration as Code".

### 4. Machine Learning Framework (Scikit-learn Adapters)

Classical ML models are integrated using a custom adapter pattern to unify them with the DL workflow.

- **Feature Extractors**: Classes like `CraftedFeatureExtractor` convert raw images into numerical vectors (HSV, LBP, HOG). `RawPixelFeatureExtractor` handles pixel flattening.
- **Adapters**: The `SklearnClassifier` base class wraps standard scikit-learn estimators to provide a consistent `fit`/`predict`/`save`/`load` interface across the project.
- **Implementations**: Currently supports Logistic Regression, SVM, Random Forest, KNN, and Gaussian Naive Bayes.

### 5. Data Management & Reproducibility

- **Freeze** (`jute-disease freeze`): fingerprints the original images (byte, decoded-pixel and perceptual hashes), applies the curation rules and the identity policy, and writes a seed-42 group-aware 70/15/15 manifest. A receipt written last hashes every dependency (inventory, rules, reviews, identity, preprocessing) with the manifest and partition digests and per-class support. A manifest without a matching receipt is never loaded, and a different completed freeze is only replaced with `--force`.
- **Readers**: `ManifestDataset` / `open_partition` read frozen partitions in stable image-id order, from the originals or from the lossless image cache. A requested cache never falls back to the originals.
- **Sources**: PlantVillage and PlantDoc are recorded with tracked version and inventory SHA-256 pins, exact partition counts and license caveats. Intake builds grouped validation and removes source images matching frozen target validation or test images. The owner authorized the downloaded packages for internal research use; this does not establish publisher identity or resolve license terms, and redistribution is not authorized. The report keeps these caveats visible while its internal-use readiness gate checks the tracked content pin and owner authorization again at execution.
- **DataModule**:
  - Serves manifest partitions (or a synthetic tree), with fixed seeds.
  - Supports **K-Fold Cross-Validation** via `set_fold()`.
  - Implements **Weighted Random Sampling** to handle class imbalance.
- **Transforms**: Uses a factory pattern (`create_pipeline`) via **Albumentations** for robust and deterministic data augmentation. Supports both global/fixed preprocessing for ML and resolution-parameterized preprocessing for DL.
- **Seed Everything**: A centralized `seed_everything` utility ensures deterministic behavior across Python, Numpy, specific libraries, and PyTorch.

### 6. Unified Diagnostics & Metrics

Both pipelines are evaluated identically using a combined **Experiment Aggregation Utility** (`jute-disease evaluate aggregate`, backed by `jute_disease.reporting.tables`). We completely deprecated `scikit-learn` metrics in favor of native PyTorch `torchmetrics.MetricCollection`.

- **EVAL_METRICS**: Computes Accuracy, Macro-F1, Precision, and Recall uniformly for both classical and neural models.
- **Aggregation**: Metrics from `WandbLogger` (cloud) and `CSVLogger` (local) are automatically synthesized into tabular CSV files for all Grid Searches and manual experiment targets.

### 7. Code Quality & Standards

- **Type Safety**: Public functions carry type annotations and use modern Python 3.10+ syntax (e.g., `list[str] | None`). Mypy is not part of the CI or lint pipeline.
- **Formatting**: Code is formatted and linted using `ruff` to ensure PEP 8 compliance.
- **CLI Patterns**: Automation functions raise on failure and are called from `jute_disease.cli`, which owns argument parsing and exit codes. The functions stay usable from Python and remain unit-testable.
- **Testing**: A comprehensive test suite (`tests/`) covers unit tests (logic verification) and slower integration tests (end-to-end pipeline).

### 8. Artifact Management

The project uses a structured approach to saving experiment outputs in the `artifacts/` directory.

- **Checkpoints**: PyTorch Lightning automatically saves the best `.ckpt` files based on validation loss/accuracy.
- **Metrics**: Local experiments save metrics to `experiment_name/metrics.csv`. These are prioritized for reproducibility when cloud logging (WandB) is unavailable.
- **Model Exports**: Classical ML models are serialized as `.joblib` files within the `artifacts/` hierarchy, ensuring they are versioned alongside the feature extraction pipelines that generated them.

## Tools & Dependencies

- **uv**: Package management and environment isolation.
- **PyTorch Lightning**: Deep learning framework.
- **timm**: Pretrained computer vision models.
- **Scikit-learn**: Classical machine learning algorithms.
- **Albumentations**: Fast image augmentation library.
- **WandB**: Experiment tracking and visualization.

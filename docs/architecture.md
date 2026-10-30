# Project Architecture

This document describes the architectural design and directory structure of the `jute-disease-detection` project. It serves as a guide for developers and contributors to understand the core components and their interactions.

## Directory Structure Overview

```text
.
├── configs/            # benchmark.json (recipe, systems, queue limits) and data rules
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
└── artifacts/          # Runtime output (git-ignored): run receipts, features, smoke receipts
```

## Core Design Principles

### 1. Unified Public API

Each subpackage in `src/jute_disease/` (e.g., `models.ml`, `data`, `training`, `reporting`) uses `__init__.py` to expose a clean, flattened public API.

- **Internal developers** use relative imports where appropriate or `from jute_disease.x import y` to avoid circular dependencies.
- **External consumers** (scripts, tests, notebooks) use the clean package-level imports (e.g., `from jute_disease.models.ml import RandomForest`).

### 2. Command Line Interface (CLI)

The project exposes a single console entry point, `jute-disease` (`jute_disease.cli:main`), and a `justfile` whose recipes each wrap one of its commands:

- **`jute-disease train fit`**: Runs one declared fit (classical preparation, a classical estimator, a source stage or a target CNN) in the current process. The queue launches it as a child, so a crash or timeout ends only that fit.
- **`jute-disease queue plan|status|run`**: Plans stages, shows run states, and runs or resumes the whole benchmark under a cutoff. It is the only path that enforces stage order, the eight-hour reserve and the time limits.
- **`jute-disease stack`**: Reports the measured library versions and CUDA device, and fails when CUDA is required and unavailable.
- **Other commands**: `data` (download, synthetic), `freeze` (target, cache, source), and `evaluate` (aggregate, predict). `jute-disease --help` lists them all.
- Commands planned in open issues (`profile`, and `evaluate` on its own) exit non-zero and name the issue that owns them.

The per-model, per-resolution and check variants of the old scripts are values in `configs/benchmark.json`: the recipe, the CNN systems with their micro-batch and source chain, the classical estimator and feature grid, and the queue limits.

### 3. Deep Learning Service (Lightning + Timm)

The DL pipeline is built using **PyTorch Lightning** for state-of-the-art reproducibility and boilerplate reduction.

- **LightningModule (`Classifier`)**: The core class handling the training loop, optimization, logging, and metrics.
- **Backbone System**: Uses a generic `TimmBackbone` to wrap any model from the `timm` library, allowing for easy experimentation with different architectures.
- **Recipe** (`configs/benchmark.json`): AdamW at 3e-4 with weight decay 0.01, reduce-on-plateau on validation loss, at most 50 epochs with patience-8 early stopping, 256 px inputs, a weighted sampler, mixed precision on CUDA, and compilation off. The full backbone trains, and the checkpoint is the lowest validation loss.
- **Systems**: MobileNetV2 and EfficientNet-B5 trained directly, plus MobileNetV2 chained through PlantVillage and through PlantVillage then PlantDoc. A source stage trains once and its checkpoint is reused by every seed. A chained stage keeps the backbone, resets the head and the training state, and records which tensors moved.
- **Run receipts** (`training/receipts.py`): each fit writes an atomic state and a receipt that hashes its artifacts. Resume skips a fit only when the receipt verifies and its identity (recipe, manifests, upstream checkpoint, initialization) matches, so a partial or incompatible output is never reused.
- **Queue** (`training/queue/`): builds the stage plan, forecasts each stage conservatively from a bounded timing probe, admits optional stages only with the reserve left and all three seeds fitting, and records any mandatory shortfall as an escalation.

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

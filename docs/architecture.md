# Project Architecture

The repository contains active tooling for a revised jute related-image evaluation and an archive of the earlier classification study. The tooling does not itself authorize or report revised-study experiment fits.

## Repository layout

```text
.
├── configs/baselines/       # Current baseline entry points
├── configs/historical/      # Configurations used for submitted-study runs
├── data/                    # Local datasets and tracked class definitions
├── docs/adr/                # Accepted architecture and methodology decisions
├── notebooks/historical/    # Original notebooks and their outputs
├── results/submitted/       # Artifacts backing the original paper
├── scripts/                 # Dependency audit and Linux lab helpers
├── src/jute_disease/        # Python package and CLI
├── tests/                   # Unit, public API and synthetic integration coverage
└── artifacts/               # Ignored runtime output
```

## Command and library interfaces

`jute-disease` is the console entry point. Its subcommands cover dataset preparation, freezing, classical ML, Lightning training and evaluation. Public package imports are exposed from the relevant `jute_disease` subpackages. The `justfile` provides short wrappers for common commands; it does not define another application interface.

Historical study files are retained as records. Active dependency configuration, baseline entry points and run output use the revised tooling. W&B is not an active logger or dependency. Lightning writes CSV logs locally; evaluation writes JSON with metrics, class order, confusion matrix and run settings.

## Data identity and preprocessing

Dataset freezing inventories image bytes and decoded pixels, applies the tracked curation and identity rules, and writes grouped partitions with a receipt binding the manifest to its inputs. Manifest version 2 records per-image letterbox pad fraction and reports a source-by-class summary. Version 1 manifests remain readable. A new freeze is explicit; completed freeze outputs are not silently replaced.

ADR 0002 defines shared preprocessing. Both feature routes resize to a 256-pixel letterboxed image with bilinear interpolation and rounded uint8 ImageNet-mean padding. The classical route receives deterministic unnormalized RGB; the deep-learning route converts to tensors and normalizes with ImageNet statistics. Training adds the fixed crop, flips and brightness/contrast jitter from the ADR. The preprocessing specification contributes to cache identity, so a changed specification invalidates matching image caches.

## Modeling and local evidence

The deep-learning path uses PyTorch Lightning and `timm`; the classical path uses scikit-learn estimators and image-feature extractors. Both paths expose consistent evaluation outputs. Local logs and evaluation records are the active evidence format. Downloaded SAM2 weights and inference are outside tooling setup; the optional `masks` extra is only used to check package installation and imports.

## Environment and validation

Python 3.14.10 is pinned in `.python-version` and constrained in `pyproject.toml`. `uv.lock` locks the supported environment. PyTorch and TorchVision form a single pinned pair, with official CUDA 13.0 wheels on Windows and CUDA 12.6 wheels on Linux. See [tooling.md](tooling.md) for installation, update, audit and helper-command details.

Continuous integration checks locked installs on Windows and Linux, package imports, dependency advisories, lint and the synthetic test suite. Windows CI checks the CUDA wheel build; a small GPU operation is also checked on the development RTX 5060. Synthetic fixtures exercise public interfaces. Linux CI checks notification transport without sending messages, process locking, timeouts, detachment and copy-back integrity. CI does not fit an experiment, access the lab, or download model weights.

## Historical study

The original notebooks, W&B run references, configs and results are retained as historical evidence under `notebooks/historical/`, `configs/historical/` and `results/submitted/`. Reproduction uses the `submitted-study` tag and its lockfile. See [legacy-names.md](legacy-names.md) for the old repository, package and W&B names.

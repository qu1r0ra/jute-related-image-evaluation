# jute-related-image-evaluation <!-- omit from toc -->

<div align="center">
  <img src="./results/submitted/figures/dl/grad_cam.png" width="800px" alt="Grad-CAM Interpretability">
  <p><i>Grad-CAM: Visualizing model focus on disease symptoms.</i></p>

  <br>

  <img src="./results/submitted/figures/dl/finetuned_training_history.png" width="800px" alt="Training History">
  <p><i>Model Convergence: Training and Validation metrics.</i></p>

  <br>

  <img src="./results/submitted/figures/dl/part2_confusion_matrix_comparison.png" width="800px" alt="Confusion Matrix">
  <p><i>Performance Benchmark: Multi-class Confusion Matrix evaluation.</i></p>
</div>

<!-- Refer to <https://shields.io/badges> for usage -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23414231.svg)](https://doi.org/10.5281/zenodo.23414231) ![Year, Term, Course](https://img.shields.io/badge/AY2526--T2-CSC713M-blue)

This repository now prepares tooling for a revised jute related-image evaluation. It also preserves the earlier six-class disease-classification study and its submitted paper, which was withdrawn before publication ([PDF](docs/hnicem2026-paper.pdf), [full paper](https://drive.google.com/file/d/1DbaQ4TWEHChkLTLvpzntbUxhkBkSmQ5o/view?usp=drive_link)).

> [!NOTE]
> This README documents the original course study. A revised study of how related-image controls affect measured performance is being planned in [issue 98](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/98); its results are not reported here.

> [!IMPORTANT]
> The active tooling follows the revised direction tracked by [issue 116](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/116). It does not report results for the revised experiment. Use Python 3.14 and the committed uv lock. Historical reproduction uses the `submitted-study` tag and its lockfile.

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Getting Started](#3-getting-started)
  - [3.1. Technical Prerequisites](#31-technical-prerequisites)
  - [3.2. Data \& Platform Credentials](#32-data--platform-credentials)
  - [3.3. Installation](#33-installation)
- [4. Reproducing the Original Results](#4-reproducing-the-original-results)
- [5. Current Tooling and Readiness](#5-current-tooling-and-readiness)
- [6. Historical Benchmarked Performance](#6-historical-benchmarked-performance)
- [7. Citation](#7-citation)
- [8. License](#8-license)
- [9. References](#9-references)

## 1. Introduction

The revised work focuses on evaluating how related-image controls affect measured performance. The active codebase prepares versioned dataset freezes, consistent preprocessing, classical and deep-learning APIs, and local evaluation records. The tooling issue does not authorize experiment fits or establish revised-study results.

The original course study classified six jute leaf conditions—**Cercospora Leaf Spot, Dieback, General Damage, Healthy, Mosaic, and Stem Rot**—using handcrafted-feature ML and transfer-learning models. Its methods and results are retained as historical context below and in the archive.

The original short paper, **"Benchmarking Classical Machine Learning and Transfer Learning for Data-Constrained Jute Leaf Disease Classification"**, was submitted to IEEE HNICEM 2026 and withdrawn before publication ([PDF](docs/hnicem2026-paper.pdf)). Its findings are not results of the revised study.

## 2. Project Structure

A high-level overview of the repository organization:

```text
.
├── configs/
│   ├── benchmark.json  # Declared fits, shared recipe and per-step timeouts
│   └── historical/     # Lightning CLI and grid search configs for the reported runs (superseded record)
├── data/               # Dataset storage (git-ignored) and the tracked class list
├── docs/               # Architecture notes and the submitted paper
├── notebooks/          # Local training notebooks and the freeze provenance demo
│   └── historical/     # Submitted notebooks for the withdrawn paper
├── results/
│   └── submitted/      # Committed logs and figures behind the reported numbers
│       ├── figures/    # dl/ and ml/ figures
│       └── logs/       # CSV logs for baselines, grid searches and resolution runs
├── src/
│   └── jute_disease/   # Library package and the jute-disease console entry point
├── tests/              # Unit and API tests
├── .env.example        # Environment variables template
├── justfile            # Task runner: one recipe per study command
└── artifacts/          # Runtime output (git-ignored): run receipts, features, smoke receipts
```

For a detailed look at the internal design, public APIs, and architectural decisions, see [architecture.md](docs/architecture.md).

## 3. Getting Started

### 3.1. Technical Prerequisites

Ensure you have the following installed on your local machine:

1. **Git:** Used to clone this repository.
2. **Python `3.14.10`:** (Managed by `uv` automatically if not present).
3. **uv:** Our unified Python package and project manager. Installation instructions: <https://docs.astral.sh/uv/getting-started/installation/>.

### 3.2. Data & Platform Credentials

This project uses third-party services only for optional data acquisition:

1. **Jute dataset:** The unified six-class dataset is distributed as `data.zip` in this [Google Drive folder](https://drive.google.com/drive/folders/1WoQ-Xzy0Prl9lInHW5JpGX4tpE9YDUua?usp=sharing). Unzip it into the project root so that `data/by_class/` exists. See [References](#9-references) for the three source datasets it merges.
2. **Kaggle API:** Only needed to download the PlantVillage (Hughes & Salathé, 2015; Mohanty et al., 2016) and PlantDoc (Singh et al., 2020) pretraining datasets (`just data-download`, which pins the Kaggle package versions). Ensure you have a `kaggle.json` in `~/.kaggle/`.

The unified jute dataset merges three public sources. The repository code is Apache-2.0; the data keeps its original terms:

| Source                                                                  | Terms                                      |
| :---------------------------------------------------------------------- | :----------------------------------------- |
| Islam & Sheikh, Harvard Dataverse (`doi:10.7910/DVN/FJ1DM1`)            | CC0 1.0                                    |
| Mridha, Mendeley Data (`z87b9hnkh7`)                                    | CC BY 4.0 (attribute the author)           |
| Jannat et al., Kaggle (`mdsaimunalam/jute-leaf-disease-detection`)      | Publicly available on Kaggle, no licence stated; used with attribution (Jannat et al., 2025) |

### 3.3. Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/qu1r0ra/jute-related-image-evaluation
   ```

2. Navigate to the project root and install all dependencies:

   ```bash
   cd jute-related-image-evaluation
   uv sync --locked --all-groups
   ```

   uv selects CUDA 13.0 PyTorch wheels on Windows and CUDA 12.6 wheels on Linux. Optional SAM2 support is installed with `uv sync --locked --extra masks --all-groups`. Setup does not include or download a model checkpoint. Optional SAM 2 refinement requires an existing local checkpoint and loads it offline.

3. Optional: install [`just`](https://just.systems) to run the recipes shown below. `just --list` shows every recipe, and each one wraps a single `uv run jute-disease` command. `uv run jute-disease --help` lists the commands themselves. The Linux lab helpers are under `scripts/`; ntfy topic and token are supplied through the host's environment secret mechanism, never command arguments.

## 4. Reproducing the Original Results

The submitted notebooks and results are preserved under `notebooks/historical/` and `results/submitted/`. They belong to the original course study and use the original 3.11-era environment and tracking setup. To reproduce that study, check out the `submitted-study` tag and install its committed lock; the active Python 3.14 environment is for the revised study.

```bash
git switch --detach submitted-study
uv sync --locked --all-groups
```

The notebooks are preserved in numerical order:

1. **`01_Exploratory_Data_Analysis.ipynb`**
   Consolidates the unified jute dataset and visualizes initial class distributions.
2. **`02_Model_Selection_Training_DL.ipynb`**
   Executes the baseline deep learning benchmarks (EfficientNet, MobileNetV2, etc.).
3. **`02_Model_Selection_Training_ML.ipynb`**
   Extracts handcrafted features (HOG, LBP, Color Hist) for classical ML training.
4. **`03_Model_Analysis_Tuning_DL.ipynb`**
   Performs deep learning hyperparameter tuning and generates Grad-CAM heatmaps.
5. **`03_Model_Analysis_Tuning_ML.ipynb`**
   Performs ML error analysis and hyperparameter tuning (Grid Search).
6. **`04_Model_Evaluation.ipynb`**
   Finalizes evaluation on the held-out test set and identifies top failure modes.

> [!NOTE]
> The notebooks were run on Google Colab (CPU for classical ML, L4 GPU for deep learning). Use the submitted-study tag for reproduction; the active CLI and configurations implement the revised study tooling.

## 5. Current Tooling and Readiness

The current environment installs from the committed lock and uses Python 3.14.10. Set it up and run the local checks with:

```bash
uv sync --locked --all-groups
uv run --locked pytest -m ""
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/audit_dependencies.py
```

Training runs through a serial, resumable queue declared in `configs/benchmark.json`: `just queue-plan` shows the fits, `just queue-run` runs them, and running it again skips completed fits and resumes an interrupted CNN fit from its last saved epoch. [Architecture](docs/architecture.md#training-queue) defines the identity, retry and resume rules.

For optional local SAM2 package imports, use `uv sync --locked --all-groups --extra masks`. No model weights are included or downloaded by setup.

This tooling work does not authorize experiment fits or establish readiness of the lab A100. The synthetic freeze and Linux CI checks validate the development environment. The validation-only lab pilots and experiment readiness gates remain in [issue 114](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/114). See [Tooling](docs/tooling.md) for update cadence, local evidence, audit limits, preprocessing identity, and helper commands.

## 6. Historical Benchmarked Performance

The table records results from the earlier course study on its held-out test split (360 images). It does not describe the revised experiment. The [full paper](https://drive.google.com/file/d/1DbaQ4TWEHChkLTLvpzntbUxhkBkSmQ5o/view?usp=drive_link) discusses learning rates, input resolutions, and feature-space analysis in detail.

| Model Class       | Architecture / Classifier                  | Features / Training       | Val Acc (%) | Val Macro-F1 (%) | Test Acc (%) |
| :---------------- | :----------------------------------------- | :------------------------ | :---------- | :--------------- | :----------- |
| **Deep Learning** | MobileNetV2 (LR = 0.01)                    | Fine-tuned ImageNet       | 88.76       | 89.11            | **91.39**    |
| **Deep Learning** | EfficientNet-B5 (baseline)                 | Fine-tuned ImageNet       | 90.44       | 90.41            | 88.61        |
| **Classical ML**  | Random Forest (tuned)                      | HOG + LBP + GLCM + Gabor + HSV | -      | 77.73            | 72.22        |

Model selection used validation metrics. Results come from a single stratified 70/15/15 split (seed 42) without near-duplicate removal, so differences of a point or two are within noise.

### Historical Pretrained Weights

The original MobileNetV2 checkpoint (learning rate 0.01, weight decay 0.05, frozen backbone, 91.39% test accuracy) is attached to the [v1.0.0 release](https://github.com/qu1r0ra/jute-related-image-evaluation/releases/tag/v1.0.0) as `mobilenet_v2_jute.pt` (SHA-256 `f172c2668378f6a59376928f3d55e1562e289dd954c29b74e5054c978edcad9d`). It is the historical W&B `best` checkpoint of run `af60c7b9` as a plain `state_dict`; class indices follow `data/disease_classes.txt`. The code below shows the original checkpoint preprocessing and is not the revised experiment pipeline.


```python
import torch
from PIL import Image
from torchvision import transforms as T

from jute_disease.models.dl.backbone import TimmBackbone
from jute_disease.models.dl.classifier import Classifier

classes = [c.strip() for c in open("data/disease_classes.txt") if c.strip()]
model = Classifier(
    TimmBackbone("mobilenetv2_100.ra_in1k", pretrained=False),
    num_classes=len(classes),
    compile_model=False,
)
model.load_state_dict(torch.load("mobilenet_v2_jute.pt", weights_only=True))
model.eval()

preprocess = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
image = preprocess(Image.open("leaf.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    print(classes[model(image).argmax(1).item()])
```

## 7. Citation

If you find this research or codebase useful in your own work, please consider citing our paper. The code and weights are archived on Zenodo ([10.5281/zenodo.23414231](https://doi.org/10.5281/zenodo.23414231), which always resolves to the latest version):

```bibtex
@inproceedings{bunyi2026jute,
  title={Benchmarking Classical Machine Learning and Transfer Learning for Data-Constrained Jute Leaf Disease Classification},
  author={Bunyi, Christian Joseph and Umali, Immanuel},
  booktitle={IEEE 2026 International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM)},
  year={2026},
  note={Submitted to IEEE HNICEM 2026; withdrawn before publication},
  url={https://github.com/qu1r0ra/jute-related-image-evaluation}
}
```

## 8. License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full text.

## 9. References

- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. _arXiv preprint_. <https://arxiv.org/abs/1511.08060>

- Islam, M. M., & Sheikh, M. R. (2026). A comprehensive image dataset of jute diseases. _Data in Brief, 64_, 112334. <https://doi.org/10.1016/j.dib.2025.112334>

- Jannat, M., Uddin, M. S., Hasan, M. A., Alam, M. S., Paul, A., Chowdhury, M. E. H., & Haider, J. (2025). Real-time jute leaf disease classification using an explainable lightweight CNN via a supervised and semi-supervised self-training approach. _Frontiers in Plant Science, 16_. <https://doi.org/10.3389/fpls.2025.1647177>

- Mridha, M. H. (2024). _Jute plant leaves_ (Version 1) [Data set]. Mendeley Data. <https://doi.org/10.17632/z87b9hnkh7.1>

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. _Frontiers in Plant Science, 7_. <https://doi.org/10.3389/fpls.2016.01419>

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ (pp. 4510–4520). <https://doi.org/10.1109/CVPR.2018.00474>

- Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). _PlantDoc: A dataset for visual plant disease detection_. In _Proceedings of the 7th ACM IKDD CoDS and 25th COMAD_ (pp. 249–253). <https://doi.org/10.1145/3371158.3371196>

- Wightman, R. (2019). _PyTorch Image Models_ (timm) [Software]. GitHub. <https://doi.org/10.5281/zenodo.4414861>

For the active environment, package choices, and lab helpers, see [Tooling](docs/tooling.md).

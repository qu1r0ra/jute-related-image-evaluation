# Tooling

## Supported environment

- Python 3.14.10 is selected by `.python-version`; `pyproject.toml` supports Python 3.14 only.
- `uv.lock` is the dependency source of truth. Use `uv sync --locked --all-groups` for development and add `--extra masks` when validating the optional mask package imports.
- PyTorch and TorchVision are pinned as one version pair. uv selects the official [PyTorch CUDA 12.6 wheels on Linux and CUDA 13.0 wheels on Windows](https://pytorch.org/get-started/previous-versions/). CPU-only use remains supported by the same builds.
- The optional `masks` extra installs Transformers for [Segment Anything Model 2](https://github.com/facebookresearch/sam2) code and API checks. It does not download checkpoints or run segmentation.
- Lightning remains the deep-learning training framework. Active run evidence is written locally as Lightning CSV logs and JSON evaluation records with metrics, class order, confusion matrix, and relevant settings. Active W&B integration has been removed; historical configs, notebooks, run records and artifacts remain available under the `submitted-study` tag and historical folders.

## Updating dependencies

Update dependencies deliberately, regenerate `uv.lock`, and review the full platform-specific diff. Keep PyTorch and TorchVision on the same release pair and verify resolution for both Linux and Windows. Run the lock audit, import checks, complete test suite and the Windows CUDA smoke check before merging. Recheck the official PyTorch wheel indexes when changing the pair. Security advisories for the locked requirements should be handled as soon as practical; review general dependency updates at least quarterly.

`just audit` runs `scripts/audit_dependencies.py`. It exports the exact locked requirements, scans all non-PyTorch packages through pip-audit, then scans the pinned PyTorch and TorchVision base release numbers. The CUDA wheel build suffixes (`+cu126` and `+cu130`) are platform-specific and are not published as PyPI releases, so the advisory lookup uses their shared upstream release numbers. The check does not install or change packages.

## Preprocessing and freezes

Preprocessing identity revision 2 covers the shared 256-pixel letterbox, bilinear resampling, rounded uint8 ImageNet-mean padding, tensor normalization and cache encoding. The classical route uses the same geometry and pad values without tensor normalization. Freeze manifest version 2 records each image's letterbox pad fraction and summarizes it by source and class. Version 1 manifests remain readable, but new freezes use version 2. A preprocessing identity change invalidates matching image caches; it does not rewrite or delete existing freeze artifacts. Re-freeze explicitly when the source, curation policy or split must change.

## Lab helpers

The scripts in `scripts/` are Linux-only command-line helpers for the validation work owned by issue 114:

- `notify_run.py` sends start, hourly heartbeat and completion messages to an `ntfy` topic. `NTFY_TOPIC` and `NTFY_TOKEN` come from the environment. The token is sent only in the HTTPS Authorization header, never in process arguments; transport errors do not print credentials.
- `lab_run.sh` holds a `flock` lock for the full process lifetime, applies a timeout to the process group, and can detach a run while recording its PID and log path. Example: `scripts/lab_run.sh /path/to/run.lock 7200 uv run jute-disease train baselines`.
- `verify_transfer.py record ROOT MANIFEST` writes a SHA-256 inventory outside the batch. Run `verify_transfer.py verify ROOT MANIFEST` after copy-back; missing, changed, unexpected, unsafe, hard-linked, non-regular and colliding paths reject the batch.

These helpers do not run real lab sessions or experiments as part of tooling CI. Lab validation and A100 readiness are handled separately under issue 114.

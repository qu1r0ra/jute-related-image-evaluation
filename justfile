# Task runner for the study workflow. Each recipe wraps one `jute-disease` command.
# List recipes with `just --list`; the commands behind them are in `jute-disease --help`.

default:
    @just --list

# Environment

setup:
    uv sync --all-groups

# Data

data-download:
    uv run jute-disease data download

# Freeze the target dataset: curation, identity, grouped 70/15/15 manifest and receipt
freeze *args:
    uv run jute-disease freeze target {{args}}

# Bounded train/val image-cache probe and decision (reads the frozen manifest)
freeze-cache *args:
    uv run jute-disease freeze cache {{args}}

# Record a pinned source package and check it against the frozen target
freeze-source name root *args:
    uv run jute-disease freeze source {{name}} {{root}} {{args}}

synthetic:
    uv run jute-disease data synthetic

# Pretraining on external data

pretrain *args:
    uv run jute-disease train pretrain {{args}}

# Classical ML

train-ml:
    uv run jute-disease train ml --all

# Corrected training queue: plan writes stages and starts nothing; run needs --admit
queue-check *args:
    uv run jute-disease queue check {{args}}

queue-plan *args:
    uv run jute-disease queue plan {{args}}

queue-run *args:
    uv run jute-disease queue run {{args}}

# Deep learning

train-dl:
    uv run jute-disease train baselines

train-dl-single model="mobilenet_v2":
    uv run jute-disease train baselines --config configs/historical/baselines/{{model}}.yaml

train-dl-check:
    uv run jute-disease train dl fit --config configs/historical/baselines/mobilenet_v2.yaml --trainer.fast_dev_run=True

train-cv folds="5":
    uv run jute-disease train cv configs/historical/baselines/mobilenet_v2.yaml --folds {{folds}}

train-dl-512:
    uv run jute-disease train dl fit --config configs/historical/experiments/mobilenet_v2_512.yaml

# Grid search

grid-search:
    uv run jute-disease train grid configs/historical/grid/mobilenet_v2_grid.yaml

grid-search-check:
    uv run jute-disease train grid configs/historical/grid/mobilenet_v2_grid.yaml --fast-dev-run

grid-search-finetune:
    uv run jute-disease train grid configs/historical/grid/mobilenet_v2_finetune_grid.yaml --base-config configs/historical/baselines/mobilenet_v2.yaml

# Aggregation

aggregate exps:
    uv run jute-disease evaluate aggregate --exp-names {{exps}}

# Quality and tests

test:
    uv run pytest

test-all:
    uv run pytest -m ""

lint:
    uv run ruff check .

format:
    uv run ruff check --fix . && uv run ruff format .

sync-nb:
    uv run jupytext --sync "notebooks/*.py"

# Cleanup

clean:
    rm -rf .coverage htmlcov lightning_logs wandb .pytest_cache .ruff_cache

clean-artifacts:
    rm -rf artifacts/ml_models artifacts/checkpoints artifacts/features

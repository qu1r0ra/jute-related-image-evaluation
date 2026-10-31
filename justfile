# Task runner for the study workflow. Each recipe wraps one `jute-disease` command.
# List recipes with `just --list`; the commands behind them are in `jute-disease --help`.

default:
    @just --list

# Environment

setup:
    uv sync --locked --all-groups

setup-masks:
    uv sync --locked --all-groups --extra masks

audit:
    uv run --locked python scripts/audit_dependencies.py

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

test-ml classifier="rf" feature="crafted":
    uv run jute-disease evaluate ml --classifier {{classifier}} --feature_type {{feature}}

# Deep learning

train-dl:
    uv run jute-disease train baselines

train-dl-single model="mobilenet_v2":
    uv run jute-disease train baselines --config configs/baselines/{{model}}.yaml

# Aggregation

aggregate exps:
    uv run jute-disease evaluate aggregate --exp-names {{exps}}

# Quality and tests

test:
    uv run --locked pytest

test-all:
    uv run --locked pytest -m ""

lint:
    uv run --locked ruff check .
    uv run --locked ruff format --check .

format:
    uv run --locked ruff check --fix . && uv run --locked ruff format .

sync-nb:
    uv run jupytext --sync "notebooks/*.py"

# Cleanup

clean:
    rm -rf .coverage htmlcov lightning_logs .pytest_cache .ruff_cache

clean-artifacts:
    rm -rf artifacts/ml_models artifacts/checkpoints artifacts/features

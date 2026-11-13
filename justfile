# Task runner for the study workflow. Each recipe wraps one `jute-disease` command.
# List recipes with `just --list`; the commands behind them are in `jute-disease --help`.

default:
    @just --list

# Environment

setup:
    uv sync --locked --all-groups

audit:
    uv run --locked python scripts/audit_dependencies.py

# Data

data-download:
    uv run jute-disease data download

# Freeze the target dataset: curation, identity, grouped 70/15/15 manifest and receipt
freeze *args:
    uv run jute-disease freeze target {{args}}

# Generate policy and view manifests from frozen, audited pools
freeze-policies *args:
    uv run jute-disease freeze policies {{args}}

# Bounded train/val image-cache probe and decision (reads the frozen manifest)
freeze-cache *args:
    uv run jute-disease freeze cache {{args}}

# Record a pinned source package and check it against the frozen target
freeze-source name root *args:
    uv run jute-disease freeze source {{name}} {{root}} {{args}}

synthetic:
    uv run jute-disease data synthetic

# End-to-end synthetic manuscript export, including the mechanical macro check
manuscript-fixture output:
    uv run jute-disease manuscript fixture --output {{output}}

manuscript-check export_dir:
    uv run jute-disease manuscript check {{export_dir}}

manuscript-audit-check export_dir manuscript:
    uv run jute-disease manuscript audit-check {{export_dir}} --manuscript {{manuscript}}

rebuild-splits kit_manifest image_root output:
    uv run --locked python scripts/rebuild_splits.py --kit-manifest {{kit_manifest}} --image-root {{image_root}} --output {{output}}

# Training: the queue runs every declared fit; resume by running it again

# Verify the measured CUDA stack
stack *args:
    uv run jute-disease stack {{args}}

# Show stages and each fit's run state
queue-plan *args:
    uv run jute-disease queue plan {{args}}

queue-status *args:
    uv run jute-disease queue status {{args}}

# Run or resume the benchmark; running it again skips completed fits
queue-run *args:
    uv run jute-disease queue run {{args}}

# Launch a detached lab queue invocation with notifications and transfer inventory
queue-unattended *args:
    bash scripts/queue_unattended.sh {{args}}

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

# Cleanup

clean:
    rm -rf lightning_logs .pytest_cache .ruff_cache

clean-artifacts:
    rm -rf artifacts/runs artifacts/features

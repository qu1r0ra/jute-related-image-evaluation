#!/usr/bin/env bash
# Start a detached, locked queue invocation with notifications and a transfer inventory.
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd)
cd "$repository"

timeout=${1:-259200}
[[ $timeout =~ ^[1-9][0-9]*$ ]] || {
    echo 'Timeout must be a positive integer' >&2
    exit 2
}

command -v uv >/dev/null || { echo 'uv is required' >&2; exit 127; }
command -v just >/dev/null || { echo 'just is required' >&2; exit 127; }

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir=artifacts/queue
mkdir -p "$run_dir" artifacts/transfer-manifests

# Check the secret-backed notification environment before detaching.
uv run --locked python scripts/notify_run.py --test

export JUTE_RUN_LOG="$run_dir/$run_id.log"
export JUTE_RUN_PID="$run_dir/$run_id.pid"

exec bash scripts/lab_run.sh --detach "$run_dir/queue.lock" "$timeout" \
    uv run --locked python scripts/notify_run.py -- \
    uv run --locked python scripts/queue_batch.py \
    --transfer-root artifacts/runs \
    --manifest-dir artifacts/transfer-manifests \
    -- just queue-run

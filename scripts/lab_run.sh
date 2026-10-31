#!/usr/bin/env bash
# Linux only: a single lock holder covers the entire timed process group.
set -euo pipefail
if [[ ${1:-} == --detach ]]; then
    shift
    log=${JUTE_RUN_LOG:?Set JUTE_RUN_LOG to the detached log path}
    pid_file=${JUTE_RUN_PID:?Set JUTE_RUN_PID to the detached PID path}
    setsid nohup bash "$0" "$@" </dev/null >>"$log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    printf '%s\n' "$pid"
    exit 0
fi
lock=${1:?Usage: lab_run.sh [--detach] LOCK SECONDS COMMAND...}
seconds=${2:?Missing timeout in seconds}
shift 2
[[ $seconds =~ ^[1-9][0-9]*$ ]] || { echo 'Timeout must be positive' >&2; exit 2; }
[[ $# -gt 0 ]] || { echo 'Missing command' >&2; exit 2; }
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 127; }
command -v timeout >/dev/null || { echo 'GNU timeout is required' >&2; exit 127; }
exec 9>"$lock"
flock -n 9 || exit 75
exec timeout --signal=TERM --kill-after=5 "$seconds" "$@"

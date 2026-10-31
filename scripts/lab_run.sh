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
command -v setsid >/dev/null || { echo 'setsid is required' >&2; exit 127; }
exec 9>"$lock"
flock -n 9 || exit 75

setsid "$@" &
command_pid=$!
timed_out=0
trap 'timed_out=1' USR1

setsid bash -c '
    sleep "$1"
    kill -USR1 "$2" 2>/dev/null || true
    kill -TERM -- "-$3" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$3" 2>/dev/null || true
' bash "$seconds" "$BASHPID" "$command_pid" &
timer_pid=$!

if wait "$command_pid"; then
    command_status=0
else
    command_status=$?
fi

if ((timed_out)); then
    wait "$timer_pid" || true
    exit 124
fi

kill -TERM -- "-$timer_pid" 2>/dev/null || true
wait "$timer_pid" 2>/dev/null || true
exit "$command_status"

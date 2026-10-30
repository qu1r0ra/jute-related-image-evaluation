"""Deadline admission through the queue command on simulated times."""

import json
import subprocess
import sys
from pathlib import Path


def run_check(config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "queue",
            "check",
            "--config",
            str(config),
            "--now",
            "2026-10-30T12:00:00+00:00",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_queue_refuses_missing_verified_cutoff(tmp_path: Path) -> None:
    config = tmp_path / "queue.json"
    config.write_text(
        json.dumps({"mandatory_seconds": 3600, "optional_seconds": 0}), encoding="utf-8"
    )
    result = run_check(config)
    assert result.returncode != 0
    assert "verified cutoff" in result.stderr.lower()


def test_queue_forecast_reserves_two_hours(tmp_path: Path) -> None:
    config = tmp_path / "queue.json"
    config.write_text(
        json.dumps(
            {
                "cutoff": "2026-10-31T00:00:00+00:00",
                "timezone": "UTC",
                "verified": True,
                "verification": "owner operational cutoff",
                "mandatory_seconds": 7200,
                "optional_seconds": 28801,
            }
        ),
        encoding="utf-8",
    )
    result = run_check(config)
    assert result.returncode != 0
    assert "finalization" in result.stderr.lower()
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["optional_seconds"] = 28800
    config.write_text(json.dumps(payload), encoding="utf-8")
    admitted = run_check(config)
    assert admitted.returncode == 0, admitted.stderr
    assert json.loads(admitted.stdout)["forecast_fits"] is True

    payload["finalization_seconds"] = 7199
    config.write_text(json.dumps(payload), encoding="utf-8")
    too_small = run_check(config)
    assert too_small.returncode != 0
    assert "at least two hours" in too_small.stderr

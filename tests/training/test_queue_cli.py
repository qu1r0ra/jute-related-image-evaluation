"""The queue command: planning starts nothing and a run needs explicit admission."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.training.queue import Stage, plan_to_json

HOUR = 3600


def config_file(tmp_path: Path, hours: float = 100.0, **extra: object) -> Path:
    path = tmp_path / "queue.json"
    cutoff = datetime.now(UTC) + timedelta(hours=hours)
    path.write_text(
        json.dumps(
            {
                "cutoff": cutoff.isoformat(),
                "timezone": "UTC",
                "verified": True,
                "verification": "owner operational cutoff",
                "core_seconds": 2 * HOUR,
                "plantvillage_seconds": HOUR,
                "plantdoc_seconds": HOUR,
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


def trivial_plan(tmp_path: Path, code: str = "pass") -> Path:
    stages = [
        Stage(
            name=f"core-{index}",
            group="core",
            command=(sys.executable, "-c", code),
            timeout_seconds=None,
            estimate_seconds=1,
            mandatory=True,
        )
        for index in range(2)
    ]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_to_json(stages)), encoding="utf-8")
    return path


@pytest.fixture
def recipes(tmp_path: Path) -> Path:
    root = tmp_path / "recipes"
    for relative in ("core/a.yaml", "plantvillage/target.yaml", "plantdoc/target.yaml"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("seed_everything: 42\n", encoding="utf-8")
    return root


def test_queue_plan_writes_the_plan_and_starts_nothing(
    tmp_path: Path, recipes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state"
    output = tmp_path / "plan.json"
    main(
        [
            "queue",
            "plan",
            "--config",
            str(config_file(tmp_path)),
            "--recipes-dir",
            str(recipes),
            "--state-dir",
            str(state),
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text("utf-8"))
    assert payload["seed"] == 42
    assert payload["stages"][0]["name"] == "core-a"
    assert len(payload["stages"]) == 1 + 10 + 4
    assert not state.exists()
    assert "planned stages" in capsys.readouterr().out


def test_queue_plan_without_estimates_is_blocked(tmp_path: Path, recipes: Path) -> None:
    config = tmp_path / "queue.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["queue", "plan", "--config", str(config), "--recipes-dir", str(recipes)])
    assert "core_seconds" in str(excinfo.value)


def test_queue_run_without_admit_is_refused(tmp_path: Path) -> None:
    state = tmp_path / "state"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "queue",
                "run",
                "--plan",
                str(trivial_plan(tmp_path)),
                "--config",
                str(config_file(tmp_path)),
                "--state-dir",
                str(state),
            ]
        )
    assert "--admit" in str(excinfo.value)
    assert not state.exists()


def run_args(tmp_path: Path, plan: Path, hours: float = 100.0) -> list[str]:
    return [
        "queue",
        "run",
        "--plan",
        str(plan),
        "--config",
        str(config_file(tmp_path, hours)),
        "--state-dir",
        str(tmp_path / "state"),
        "--admit",
    ]


def test_queue_run_admitted_writes_receipts_then_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = trivial_plan(tmp_path)
    main(run_args(tmp_path, plan))
    first = json.loads(capsys.readouterr().out)
    assert first["completed"] == ["core-0", "core-1"]
    receipt = json.loads((tmp_path / "state" / "receipts" / "core-0.json").read_text())
    assert receipt["status"] == "completed"
    main(run_args(tmp_path, plan))
    second = json.loads(capsys.readouterr().out)
    assert second["skipped"] == ["core-0", "core-1"]
    assert second["completed"] == []


def test_queue_run_escalates_a_failed_mandatory_stage(tmp_path: Path) -> None:
    plan = trivial_plan(tmp_path, code="raise SystemExit(2)")
    with pytest.raises(SystemExit) as excinfo:
        main(run_args(tmp_path, plan))
    assert "queue escalation" in str(excinfo.value)
    assert "core-0" in str(excinfo.value)


def test_queue_run_escalates_when_the_reserve_would_be_consumed(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    stage = Stage(
        name="core-0",
        group="core",
        command=(sys.executable, "-c", "pass"),
        timeout_seconds=None,
        estimate_seconds=3 * HOUR,
        mandatory=True,
    )
    plan.write_text(json.dumps(plan_to_json([stage])), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(run_args(tmp_path, plan, hours=4))
    assert "queue escalation" in str(excinfo.value)
    assert not (tmp_path / "state" / "receipts").exists()

"""The child-process boundary that enforces a fit's time limit."""

import sys
import time
from pathlib import Path

import pytest

from jute_disease.training import run_child


def _python(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def test_a_finished_child_reports_its_exit_code_and_seconds(tmp_path: Path) -> None:
    result = run_child(_python("print('hello')"), 30, tmp_path / "child.log")
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.seconds > 0
    assert "hello" in (tmp_path / "child.log").read_text(encoding="utf-8")


def test_a_failing_child_is_reported_not_raised(tmp_path: Path) -> None:
    result = run_child(
        _python("import sys; print('boom', file=sys.stderr); sys.exit(3)"),
        30,
        tmp_path / "child.log",
    )
    assert result.returncode == 3
    assert result.timed_out is False
    assert "boom" in result.tail


def test_a_child_past_its_limit_is_killed_with_its_children(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    result = run_child(_python(code), 2, tmp_path / "child.log")
    assert time.monotonic() - started < 30
    assert result.timed_out is True
    assert result.returncode != 0

    import psutil

    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not psutil.pid_exists(pid)


def test_no_limit_waits_for_the_child(tmp_path: Path) -> None:
    result = run_child(
        _python("import time; time.sleep(0.2)"), None, tmp_path / "c.log"
    )
    assert result.returncode == 0 and result.timed_out is False


def test_an_interrupted_wait_kills_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    original = subprocess.Popen.wait
    calls = {"n": 0}

    def interrupt(self, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return original(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_child(_python("import time; time.sleep(60)"), 60, tmp_path / "c.log")

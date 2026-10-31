"""Exercise the lab helper commands through their user-facing CLI surfaces."""

import importlib.util
import io
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
TRANSFER_COMMAND = REPOSITORY / "scripts" / "verify_transfer.py"
NOTIFIER_COMMAND = REPOSITORY / "scripts" / "notify_run.py"
RUN_COMMAND = REPOSITORY / "scripts" / "lab_run.sh"
BASH = shutil.which("bash")


def _transfer(
    action: str, root: Path, manifest: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRANSFER_COMMAND), action, str(root), str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )


def _verify_entries(
    tmp_path: Path, manifest_text: str
) -> subprocess.CompletedProcess[str]:
    root = tmp_path / "batch"
    root.mkdir()
    (root / "result.txt").write_text("data", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(manifest_text, encoding="utf-8")
    return _transfer("verify", root, manifest)


def _notifier_module() -> object:
    spec = importlib.util.spec_from_file_location("jute_notify_run", NOTIFIER_COMMAND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notifier_connectivity_command_hides_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _notifier_module()
    requests: list[object] = []
    sentinel = "secret-token-sentinel"

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO()

    monkeypatch.setenv("NTFY_TOPIC", "jute-test-topic")
    monkeypatch.setenv("NTFY_TOKEN", sentinel)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "argv", [str(NOTIFIER_COMMAND), "--test"])

    assert module.main() == 0
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") == f"Bearer {sentinel}"
    assert sentinel not in " ".join(sys.argv)
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err


@pytest.mark.parametrize(("child", "expected"), [("pass", 0), ("fail", 7)])
def test_notifier_reports_child_completion(
    monkeypatch: pytest.MonkeyPatch, child: str, expected: int
) -> None:
    module = _notifier_module()
    messages: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        messages.append(request.data.decode())
        return io.BytesIO()

    monkeypatch.setenv("NTFY_TOPIC", "jute-test-topic")
    monkeypatch.setenv("NTFY_TOKEN", "secret-token-sentinel")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(NOTIFIER_COMMAND),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({expected})",
        ],
    )

    assert module.main() == expected
    assert messages[0] == "Jute run started"
    assert messages[-1] == (
        "Jute run succeeded" if expected == 0 else "Jute run failed"
    )


def test_notifier_cli_rejects_missing_credentials_without_echoing_them() -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"NTFY_TOPIC", "NTFY_TOKEN"}
    }
    result = subprocess.run(
        [sys.executable, str(NOTIFIER_COMMAND), "--test"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 2
    assert "NTFY_TOPIC" not in result.stderr
    assert "NTFY_TOKEN" not in result.stderr


def test_notifier_cli_redacts_transport_errors_and_keeps_token_out_of_argv() -> None:
    sentinel = "transport-secret-sentinel"
    wrapper = (
        "import os, runpy, sys, urllib.request; "
        "urllib.request.urlopen = lambda *args, **kwargs: "
        "(_ for _ in ()).throw(OSError(os.environ['NTFY_TOKEN'])); "
        "runpy.run_path(sys.argv[1], run_name='__main__')"
    )
    environment = {
        **os.environ,
        "NTFY_TOPIC": "jute-test-topic",
        "NTFY_TOKEN": sentinel,
    }
    command = [sys.executable, "-c", wrapper, str(NOTIFIER_COMMAND), "--test"]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 2
    assert sentinel not in result.stdout + result.stderr
    assert sentinel not in " ".join(command)


def test_transfer_command_records_and_verifies_file_contents(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    (root / "result.json").write_text('{"score": 0.7}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    recorded = _transfer("record", root, manifest)
    assert recorded.returncode == 0, recorded.stderr
    assert _transfer("verify", root, manifest).returncode == 0

    (root / "result.json").write_text('{"score": 0.8}', encoding="utf-8")
    assert _transfer("verify", root, manifest).returncode != 0


def test_transfer_verify_rejects_missing_and_unexpected_files(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    result_file = root / "result.txt"
    result_file.write_text("data", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    assert _transfer("record", root, manifest).returncode == 0

    result_file.unlink()
    assert _transfer("verify", root, manifest).returncode != 0

    result_file.write_text("data", encoding="utf-8")
    (root / "unexpected.txt").write_text("extra", encoding="utf-8")
    assert _transfer("verify", root, manifest).returncode != 0


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "./escape.txt",
        "/absolute.txt",
        "C:/outside.txt",
        "\\\\server\\share\\file.txt",
        "dir\\file.txt",
        "CON",
        "nul.txt",
        "PRN",
        "AUX",
        "COM9.log",
        "COM³.log",
        "LPT1.txt",
        "name.",
        "name ",
        "stream:secret",
        "bad<name",
        "bad?name",
    ],
)
def test_transfer_record_rejects_paths_unsafe_on_windows(
    tmp_path: Path, name: str
) -> None:
    result = _verify_entries(tmp_path, json.dumps({name: "0" * 64}))

    assert result.returncode != 0


@pytest.mark.parametrize(
    ("first", "second"),
    [("Leaf.PNG", "leaf.png"), ("é.png", "e\u0301.png")],
)
def test_transfer_record_rejects_normalized_name_collisions(
    tmp_path: Path, first: str, second: str
) -> None:
    digest = "0" * 64
    result = _verify_entries(tmp_path, json.dumps({first: digest, second: digest}))

    assert result.returncode != 0


def test_transfer_verify_rejects_duplicate_manifest_entries(tmp_path: Path) -> None:
    result = _verify_entries(
        tmp_path,
        '{"result.txt":"' + "0" * 64 + '","result.txt":"' + "0" * 64 + '"}',
    )

    assert result.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may be restricted")
@pytest.mark.parametrize("action", ["record", "verify"])
def test_transfer_rejects_symlinks(tmp_path: Path, action: str) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    (root / "source.txt").write_text("data", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    if action == "verify":
        assert _transfer("record", root, manifest).returncode == 0
    (root / "link.txt").symlink_to(root / "source.txt")

    result = _transfer(action, root, manifest)

    assert result.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="FIFO creation requires POSIX")
@pytest.mark.parametrize("action", ["record", "verify"])
def test_transfer_rejects_non_regular_files(tmp_path: Path, action: str) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    (root / "source.txt").write_text("data", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    if action == "verify":
        assert _transfer("record", root, manifest).returncode == 0
    os.mkfifo(root / "pipe")

    result = _transfer(action, root, manifest)

    assert result.returncode != 0


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX hard-link semantics are checked in CI"
)
@pytest.mark.parametrize("action", ["record", "verify"])
def test_transfer_rejects_hardlinked_files(tmp_path: Path, action: str) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    first = root / "first.txt"
    first.write_text("shared content", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    if action == "verify":
        assert _transfer("record", root, manifest).returncode == 0
    os.link(first, root / "second.txt")

    result = _transfer(action, root, manifest)

    assert result.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="case-distinct directories require POSIX")
def test_transfer_record_rejects_colliding_directory_names(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    (root / "Reports").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "Reports" / "first.txt").write_text("a", encoding="utf-8")
    (root / "reports" / "second.txt").write_text("b", encoding="utf-8")

    result = _transfer("record", root, tmp_path / "manifest.json")

    assert result.returncode != 0


@pytest.mark.skipif(BASH is None, reason="Linux shell runner requires bash")
@pytest.mark.skipif(os.name == "nt", reason="Linux runner behavior is checked in CI")
def test_lab_runner_rejects_a_second_process_holding_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    first = subprocess.Popen(
        [
            BASH,
            str(RUN_COMMAND),
            str(lock),
            "10",
            sys.executable,
            "-c",
            "import time; print('lock-held', flush=True); time.sleep(1.5)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert first.stdout is not None
        assert select.select([first.stdout], [], [], 5)[0]
        assert first.stdout.readline().strip() == "lock-held"
        second = subprocess.run(
            [BASH, str(RUN_COMMAND), str(lock), "10", sys.executable, "-c", "pass"],
            capture_output=True,
            check=False,
        )
    finally:
        first.wait(timeout=10)

    assert first.returncode == 0
    assert second.returncode == 75


@pytest.mark.skipif(BASH is None, reason="Linux shell runner requires bash")
@pytest.mark.skipif(os.name == "nt", reason="Linux runner behavior is checked in CI")
def test_lab_runner_bounds_runtime_and_records_detached_pid_and_log(
    tmp_path: Path,
) -> None:
    timed = subprocess.run(
        [
            BASH,
            str(RUN_COMMAND),
            str(tmp_path / "timeout.lock"),
            "1",
            sys.executable,
            "-c",
            "import time; time.sleep(20)",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert timed.returncode == 124

    log = tmp_path / "detached.log"
    pid_file = tmp_path / "detached.pid"
    environment = {
        **os.environ,
        "JUTE_RUN_LOG": str(log),
        "JUTE_RUN_PID": str(pid_file),
    }
    detached = subprocess.run(
        [
            BASH,
            str(RUN_COMMAND),
            "--detach",
            str(tmp_path / "detached.lock"),
            "10",
            sys.executable,
            "-c",
            "import time; print('detached-ready', flush=True); time.sleep(1.5)",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=10,
    )
    assert detached.returncode == 0, detached.stderr
    assert pid_file.read_text(encoding="utf-8").strip() == detached.stdout.strip()
    for _ in range(50):
        if log.exists() and "detached-ready" in log.read_text(encoding="utf-8"):
            break
        time.sleep(0.1)
    assert "detached-ready" in log.read_text(encoding="utf-8")

    while_locked = subprocess.run(
        [
            BASH,
            str(RUN_COMMAND),
            str(tmp_path / "detached.lock"),
            "10",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        check=False,
    )
    assert while_locked.returncode == 75

    for _ in range(50):
        available = subprocess.run(
            [
                BASH,
                str(RUN_COMMAND),
                str(tmp_path / "detached.lock"),
                "10",
                sys.executable,
                "-c",
                "pass",
            ],
            capture_output=True,
            check=False,
        )
        if available.returncode == 0:
            break
        time.sleep(0.1)
    assert available.returncode == 0


@pytest.mark.skipif(BASH is None, reason="Linux shell runner requires bash")
@pytest.mark.skipif(
    os.name == "nt", reason="Linux process-group behavior is checked in CI"
)
def test_lab_runner_timeout_terminates_descendant_processes(tmp_path: Path) -> None:
    marker = tmp_path / "child-terminated.txt"
    pid_file = tmp_path / "child.pid"
    child_code = (
        "import pathlib, signal, sys, time; "
        "marker = pathlib.Path(sys.argv[1]); "
        "signal.signal(signal.SIGTERM, lambda *_: marker.write_text('terminated')); "
        "time.sleep(30)"
    )
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
        "pathlib.Path(sys.argv[3]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    timed = subprocess.run(
        [
            BASH,
            str(RUN_COMMAND),
            str(tmp_path / "timeout.lock"),
            "1",
            sys.executable,
            "-c",
            parent_code,
            child_code,
            str(marker),
            str(pid_file),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert timed.returncode == 124
    for _ in range(30):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.read_text(encoding="utf-8") == "terminated"

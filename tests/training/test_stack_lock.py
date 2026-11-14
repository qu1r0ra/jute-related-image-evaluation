"""The environment check compares installed versions with the lock."""

import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from jute_disease.config import ROOT_DIR
from jute_disease.training import StackError, verify_stack


def _lock(path: Path, versions: dict[str, str], python: str = "==3.14.*") -> Path:
    blocks = [f'version = 1\nrequires-python = "{python}"\n']
    for name, version in versions.items():
        blocks.append(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
    path.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    return path


def _installed() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in ("timm", "lightning", "scikit-learn", "torchvision")
    } | {"torch": torch.__version__}


def test_stack_report_records_the_checked_lock(tmp_path: Path) -> None:
    lock = _lock(tmp_path / "uv.lock", _installed())

    report = verify_stack(require_cuda=False, lock=lock)

    assert report["lock"]["checked_packages"] == len(_installed())
    assert report["lock"]["path"] == "uv.lock"
    assert len(report["lock"]["sha256"]) == 64


def test_stack_rejects_an_installed_version_that_differs_from_the_lock(
    tmp_path: Path,
) -> None:
    versions = _installed() | {"timm": "0.0.1"}

    with pytest.raises(StackError, match=r"timm.*0\.0\.1"):
        verify_stack(require_cuda=False, lock=_lock(tmp_path / "uv.lock", versions))


def test_stack_accepts_any_locked_build_of_a_platform_split_package(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "uv.lock"
    _lock(lock, _installed())
    with lock.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            '\n[[package]]\nname = "torch"\nversion = "0.0.1"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )

    assert verify_stack(require_cuda=False, lock=lock)["lock"]["path"] == "uv.lock"


def test_stack_rejects_a_python_outside_the_lock(tmp_path: Path) -> None:
    lock = _lock(tmp_path / "uv.lock", _installed(), python="==2.7.*")

    with pytest.raises(StackError, match="Python"):
        verify_stack(require_cuda=False, lock=lock)


def test_stack_rejects_a_missing_lock(tmp_path: Path) -> None:
    with pytest.raises(StackError, match="uv.lock"):
        verify_stack(require_cuda=False, lock=tmp_path / "uv.lock")


def test_stack_ignores_a_locked_package_that_is_not_installed(tmp_path: Path) -> None:
    versions = _installed() | {"not-installed-on-this-platform": "1.0"}
    lock = _lock(tmp_path / "uv.lock", versions)

    report = verify_stack(require_cuda=False, lock=lock)

    assert report["lock"]["not_installed"] == ["not-installed-on-this-platform"]


def test_stack_records_the_observed_precision_on_cpu() -> None:
    report = verify_stack(require_cuda=False, accelerator="cpu")

    assert report["precision"] == "32-true"
    assert report["observed_precision"] == {
        "requested": "32-true",
        "compute_dtype": "torch.float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def test_stack_rejects_a_requested_precision_the_device_does_not_deliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jute_disease.training.stack as stack

    monkeypatch.setattr(
        stack, "_observed_compute_dtype", lambda uses_cuda: "torch.float32"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(StackError, match="precision"):
        stack.verify_stack(require_cuda=True, precision="bf16-mixed")


def test_stack_command_reports_the_lock_and_observed_precision() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", "stack", "--allow-cpu"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT_DIR,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["lock"]["checked_packages"] > 100
    assert report["observed_precision"]["compute_dtype"]

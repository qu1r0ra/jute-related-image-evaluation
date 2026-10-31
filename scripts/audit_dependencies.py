"""Audit locked requirements and CUDA builds via their public base versions."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTORCH_DISTRIBUTIONS = {"torch", "torchvision"}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )


def _pytorch_base_requirements() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements: dict[str, str] = {}
    for entry in project["dependencies"]:
        match = re.fullmatch(r"(torch|torchvision)==([0-9][A-Za-z0-9.!+-]*)", entry)
        if match:
            requirements[match.group(1)] = match.group(2)
    if requirements.keys() != PYTORCH_DISTRIBUTIONS:
        raise ValueError("pyproject.toml must pin torch and torchvision exactly")
    return [f"{name}=={requirements[name]}" for name in sorted(requirements)]


def _filter_cuda_builds(exported: str) -> str:
    blocks: list[list[str]] = []
    for line in exported.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            blocks.append([line])
        elif not blocks:
            raise ValueError(f"Unexpected uv export line: {line.strip()}")
        else:
            blocks[-1].append(line)

    requirements: list[str] = []
    for block in blocks:
        first_line = block[0].strip()
        match = re.match(r"([A-Za-z0-9_.-]+)", first_line)
        if match is None:
            raise ValueError(f"Unexpected uv export line: {first_line}")
        normalized_name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if normalized_name not in PYTORCH_DISTRIBUTIONS:
            requirements.append("\n".join(block))
    return "\n".join(requirements) + "\n"


def _audit(requirements_file: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--requirement",
        str(requirements_file),
        "--no-deps",
        "--disable-pip",
        "--strict",
        "--format",
        "columns",
    ]
    result = _run(command)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def main() -> int:
    """Export exact lock pins and audit CUDA package releases by their base versions."""
    export = _run(
        [
            "uv",
            "export",
            "--locked",
            "--all-groups",
            "--extra",
            "masks",
            "--format",
            "requirements-txt",
            "--no-emit-project",
            "--no-annotate",
            "--no-header",
        ]
    )
    if export.returncode:
        if export.stdout:
            sys.stdout.write(export.stdout)
        if export.stderr:
            sys.stderr.write(export.stderr)
        return export.returncode
    if export.stderr:
        sys.stderr.write(export.stderr)

    try:
        base_requirements = _pytorch_base_requirements()
        runtime_requirements = _filter_cuda_builds(export.stdout)
    except (KeyError, ValueError) as error:
        sys.stderr.write(f"Dependency audit setup failed: {error}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="jute-dependency-audit-") as directory:
        temporary = Path(directory)
        general_file = temporary / "locked-requirements.txt"
        pytorch_file = temporary / "pytorch-base-requirements.txt"
        general_file.write_text(runtime_requirements, encoding="utf-8")
        pytorch_file.write_text("\n".join(base_requirements) + "\n", encoding="utf-8")

        result = _audit(general_file)
        if result:
            return result
        return _audit(pytorch_file)


if __name__ == "__main__":
    raise SystemExit(main())

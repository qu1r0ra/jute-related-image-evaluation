import json
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import DataInputs, Workspace, load_assignment_table


def _workspace(target_dir: Path, tmp_path: Path) -> Workspace:
    return Workspace(
        benchmark_path=BENCHMARK_CONFIG,
        data=DataInputs(image_root=tmp_path, target_dir=target_dir, sources={}),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        pretrained=False,
        require_cuda=False,
        accelerator="cpu",
    )


def test_assignment_table_is_discovered_and_validated_without_image_reads(
    tmp_path: Path,
) -> None:
    target = tmp_path / "freeze" / "target"
    target.mkdir(parents=True)
    rows = [
        {"split": split, "seed": seed, "checkpoint": (seed + split) % 3}
        for split in range(10)
        for seed in range(5)
    ]
    table_path = target.parent / "assignments.json"
    table_path.write_text(json.dumps(rows), encoding="utf-8")

    table = load_assignment_table(_workspace(target, tmp_path))

    assert table["present"] is True
    assert table["validated"] is True
    assert table["sha256"]
    assert table["assignments"] == rows


def test_invalid_rotating_assignment_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "freeze" / "target"
    target.mkdir(parents=True)
    table_path = target.parent / "assignments.json"
    table_path.write_text(
        json.dumps([{"split": 2, "seed": 1, "checkpoint": 1}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rotating checkpoint rule"):
        load_assignment_table(_workspace(target, tmp_path))

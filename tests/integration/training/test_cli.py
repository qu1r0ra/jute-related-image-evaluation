"""The public CLI runs a tiny assigned source chain on synthetic data."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import jute_disease.cli as cli
from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import ManifestEntry, load_manifest
from jute_disease.training import ChildResult, Job, RunStore, Workspace, run_job

from .conftest import Inputs


def _write_policy_fixture(inputs: Inputs, policy_dir: Path) -> None:
    manifest = load_manifest(inputs.target_dir)
    by_label: dict[str, dict[str, list]] = {}
    for entry in manifest.entries:
        if entry.partition == "train":
            by_label.setdefault(entry.label, {}).setdefault(entry.group, []).append(
                entry
            )

    reserve_rows = []
    final_rows = []
    for _label, families in sorted(by_label.items()):
        family_ids = sorted(families)
        assert len(family_ids) >= 4
        reserve_families, final_families = family_ids[:2], family_ids[2:]
        for reserve_family in reserve_families:
            for entry in families[reserve_family]:
                reserve_rows.append(_policy_entry(entry))
        for index, family in enumerate(final_families):
            partition = "val" if index == len(final_families) - 1 else "train"
            final_rows.extend(
                _policy_entry(entry, partition=partition) for entry in families[family]
            )

    assignment_rows = [{"split": 0, "seed": 0, "checkpoint": 0}]
    artifacts = {
        "A/reserve.json": {
            "pool": "A",
            "seed": 137,
            "fraction": 0.2,
            "config_digest": "synthetic-fixture",
            "entries": reserve_rows,
        },
        "A/view1/image-random/draw-0.json": {
            "pool": "A",
            "view": 1,
            "policy": "image-random",
            "draw": 0,
            "draw_seed": 137,
            "config_digest": "synthetic-fixture",
            "entries": final_rows,
        },
    }
    files: dict[str, str] = {}
    for relative, payload in artifacts.items():
        destination = policy_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        destination.write_bytes(encoded)
        files[relative] = hashlib.sha256(encoded).hexdigest()
    assignment_path = policy_dir / "assignments.json"
    encoded = (json.dumps(assignment_rows, sort_keys=True, indent=2) + "\n").encode()
    assignment_path.write_bytes(encoded)
    files["assignments.json"] = hashlib.sha256(encoded).hexdigest()
    policy_dir.joinpath("receipt.json").write_text(
        json.dumps(
            {"version": 1, "config_digest": "synthetic-fixture", "files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _policy_entry(
    entry: ManifestEntry, *, partition: str | None = None
) -> dict[str, object]:
    row = {
        "path": entry.path,
        "sha256": entry.image_id,
        "label": entry.label,
        "family": entry.group,
        "source": entry.source,
        "width": entry.width,
        "height": entry.height,
    }
    if partition is not None:
        row["partition"] = partition
    return row


def test_cli_trains_one_assigned_chain_and_classical_fit(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixture registers its synthetic sources in this process only, so the
    # queue's children run here instead of in a fresh interpreter.
    def in_process(
        workspace_file: Path,
    ) -> Callable[[Job, float | None, Path], ChildResult]:
        workspace = Workspace.load(workspace_file)

        def launch(job: Job, timeout: float | None, log: Path) -> ChildResult:
            run_job(workspace, job)
            return ChildResult(0, False, 1.0, "")

        return launch

    monkeypatch.setattr(cli, "subprocess_launcher", in_process)
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["recipe"].update(
        image_size=64,
        seeds=[42],
        source_seeds=[42],
        max_epochs=1,
        effective_batch=8,
    )
    raw["cnn_systems"] = {
        "tiny_chain": {
            "model": "resnet18",
            "micro_batch": 8,
            "chain": ["plantvillage"],
        }
    }
    raw["classical"] = {"estimators": ["lr"], "features": ["crafted"], "seed": 42}
    raw["matrix"] = {
        "pools": ["A"],
        "policies": ["image-random"],
        "views": [1],
        "splits": [0],
        "primary_split": 0,
    }
    raw["assignments"] = [{"split": 0, "seed": 0, "checkpoint": 0}]
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(raw), encoding="utf-8")
    policy_dir = tmp_path / "policies"
    _write_policy_fixture(inputs, policy_dir)
    runs_dir = tmp_path / "runs"
    argv = [
        "queue",
        "run",
        "--benchmark",
        str(benchmark_path),
        "--freeze-dir",
        str(inputs.freeze_dir),
        "--image-root",
        str(inputs.image_root),
        "--policy-dir",
        str(policy_dir),
        "--pool-a-root",
        str(inputs.image_root),
        "--plantvillage-root",
        str(inputs.sources["plantvillage"][1]),
        "--plantdoc-root",
        str(inputs.sources["plantdoc"][1]),
        "--runs-dir",
        str(runs_dir),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--accelerator",
        "cpu",
        "--allow-cpu",
        "--no-pretrained",
    ]
    main(argv)

    store = RunStore(runs_dir)
    source_tune = store.receipt("source/tune/plantvillage")
    source = store.receipt("source/plantvillage")
    target_tune = store.receipt("cnn/tune/tiny_chain/pool-A")
    target = store.receipt("cnn/tiny_chain/seed-42")
    classical_tune = store.receipt("classical/tune/A/lr-crafted")
    classical = store.receipt("classical/lr-crafted")

    source_rate = source_tune["results"]["selected_learning_rate"]
    target_rate = target_tune["results"]["selected_learning_rate"]
    assert len(source_tune["results"]["candidates"]) == 3
    assert len(target_tune["results"]["candidates"]) == 3
    assert source["results"]["recipe"]["learning_rate"] == source_rate
    assert target["results"]["recipe"]["learning_rate"] == target_rate
    assert target["results"]["lineage"]["upstream_fit"] == source["fit_id"]
    assert target["metadata"]["assignment_table"]["present"] is True
    assert len(classical_tune["results"]["configurations"]) == 9
    assert (
        classical["results"]["selected_parameters"]
        == classical_tune["results"]["selected_parameters"]
    )
    assert all(
        receipt["results"]["test_accessed"] is False
        for receipt in (
            source_tune,
            source,
            target_tune,
            target,
            classical_tune,
            classical,
        )
    )

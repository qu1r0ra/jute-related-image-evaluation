"""The content-hashed image cache and the prepared feature cache, via the CLI."""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import jute_disease.cli as cli
from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import (
    PreprocessingSpec,
    cache_root,
    load_manifest,
    verify_image_cache,
    write_manifest,
)
from jute_disease.training import (
    ChildResult,
    DataInputs,
    Job,
    RunStore,
    Workspace,
    open_workspace,
    run_job,
)

from .conftest import Inputs

IMAGE_SIZE = 64
PREPARE_RESERVE = "classical/prepare/pool-A/reserve"
PREPARE_DRAW = "classical/prepare/pool-A/view-1/image-random/split-0"
CLASSICAL_FIT = "classical/lr-crafted"


def _benchmark(tmp_path: Path) -> Path:
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["recipe"].update(
        image_size=IMAGE_SIZE,
        seeds=[42],
        source_seeds=[42],
        max_epochs=1,
        effective_batch=8,
    )
    raw["cnn_systems"] = {}
    raw["classical"] = {"estimators": ["lr"], "features": ["crafted"], "seed": 42}
    raw["matrix"] = {
        "pools": ["A"],
        "policies": ["image-random"],
        "views": [1],
        "splits": [0],
        "primary_split": 0,
    }
    raw["assignments"] = [{"split": 0, "seed": 0, "checkpoint": 0}]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _image_copy(inputs: Inputs, tmp_path: Path) -> Path:
    images = tmp_path / "images"
    shutil.copytree(inputs.image_root, images)
    return images


def _cache_argv(
    inputs: Inputs, images: Path, cache: Path, output: Path | None = None
) -> list[str]:
    return [
        "freeze",
        "cache",
        "--output",
        str(inputs.freeze_dir if output is None else output),
        "--image-root",
        str(images),
        "--cache-dir",
        str(cache),
        "--image-size",
        str(IMAGE_SIZE),
        "--workers",
        "2",
    ]


def _workspace(
    inputs: Inputs, images: Path, tmp_path: Path, image_cache: Path | None
) -> Path:
    workspace = open_workspace(
        benchmark_path=_benchmark(tmp_path),
        data=DataInputs(
            image_root=images,
            target_dir=inputs.target_dir,
            sources=inputs.sources,
            image_cache_dir=image_cache,
        ),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "features",
        pretrained=False,
        require_cuda=False,
        accelerator="cpu",
    )
    path = tmp_path / "workspace.json"
    workspace.save(path)
    return path


def _train_fit(workspace: Path, fit_id: str, tmp_path: Path) -> None:
    attempt = tmp_path / "attempt" / fit_id.replace("/", "-")
    attempt.mkdir(parents=True, exist_ok=True)
    main(
        [
            "train",
            "fit",
            "--workspace",
            str(workspace),
            "--fit-id",
            fit_id,
            "--attempt-dir",
            str(attempt),
            "--result",
            str(attempt / "result.json"),
        ]
    )


def test_cli_refuses_a_cache_built_before_an_image_changed(
    inputs: Inputs, tmp_path: Path
) -> None:
    images = _image_copy(inputs, tmp_path)
    cache = tmp_path / "image-cache"
    main(_cache_argv(inputs, images, cache))
    workspace = _workspace(inputs, images, tmp_path, cache)

    train = next(
        entry
        for entry in load_manifest(inputs.target_dir).entries
        if entry.partition == "train"
    )
    with (images / train.path).open("ab") as stream:
        stream.write(b"\0")

    with pytest.raises(SystemExit, match="image cache"):
        _train_fit(workspace, PREPARE_RESERVE, tmp_path)


def test_cli_refuses_a_cache_whose_entry_was_altered(
    inputs: Inputs, tmp_path: Path
) -> None:
    images = _image_copy(inputs, tmp_path)
    cache = tmp_path / "image-cache"
    main(_cache_argv(inputs, images, cache))
    workspace = _workspace(inputs, images, tmp_path, cache)

    root = cache_root(cache, PreprocessingSpec(size=IMAGE_SIZE))
    victim = sorted(root.glob("*.png"))[0]
    victim.write_bytes(victim.read_bytes() + b"\0")

    with pytest.raises(SystemExit, match="image cache"):
        _train_fit(workspace, PREPARE_RESERVE, tmp_path)


@pytest.mark.serial
def test_cli_builds_one_valid_cache_from_two_concurrent_builders(
    inputs: Inputs, tmp_path: Path
) -> None:
    images = _image_copy(inputs, tmp_path)
    cache = tmp_path / "image-cache"
    argv = [
        sys.executable,
        "-m",
        "jute_disease.cli",
        *_cache_argv(inputs, images, cache),
    ]
    builders = [
        subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [builder.communicate(timeout=600)[0] for builder in builders]

    assert [builder.returncode for builder in builders] == [0, 0], outputs
    spec = PreprocessingSpec(size=IMAGE_SIZE)
    root = cache_root(cache, spec)
    assert not list(root.glob("*.tmp"))
    manifest = load_manifest(inputs.target_dir)
    identity = verify_image_cache(cache, spec, manifest, ("train", "val"), images)
    assert len(identity) == 64
    report = json.loads((root / "cache_report.json").read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["images_done"] == report["images_total"]


def _slice_freeze(inputs: Inputs, tmp_path: Path, index: int, count: int) -> Path:
    """A freeze directory holding every ``count``-th image of each label and split.

    Slices have disjoint images, so builders of different slices share a cache
    directory and a spec but each owns different index rows, and every slice
    still supports every label in train and validation.
    """
    output = tmp_path / f"slice-{index}"
    manifest = load_manifest(inputs.target_dir)
    seen: dict[tuple[str, str], int] = {}
    entries = []
    for entry in manifest.entries:
        key = (entry.label, entry.partition)
        rank = seen.get(key, 0)
        seen[key] = rank + 1
        if rank % count == index:
            entries.append(entry)
    write_manifest(
        replace(manifest, entries=tuple(entries), constraints=()), output / "target"
    )
    grouping = inputs.freeze_dir / "grouping.json"
    if grouping.is_file():
        shutil.copy(grouping, output / "grouping.json")
    return output


SLOW_INDEX_READ = """
import sys
import time

import jute_disease.data.cache as cache
from jute_disease.cli import main

read = cache._read_index


def slow(root, spec):
    rows = read(root, spec)
    time.sleep(2.0)
    return rows


cache._read_index = slow
main(sys.argv[1:])
"""


@pytest.mark.serial
def test_cli_concurrent_builders_of_different_manifests_keep_every_entry(
    inputs: Inputs, tmp_path: Path
) -> None:
    """Every builder holds its read of the index for two seconds before merging.

    Without one lock around the whole read, merge and publish, all builders read
    the same index, so each publishes only its own additions and the last
    replace erases the others.
    """
    driver = tmp_path / "slow_index_builder.py"
    driver.write_text(SLOW_INDEX_READ, encoding="utf-8")
    images = _image_copy(inputs, tmp_path)
    cache = tmp_path / "image-cache"
    count = 3
    slices = [_slice_freeze(inputs, tmp_path, index, count) for index in range(count)]
    builders = [
        subprocess.Popen(
            [
                sys.executable,
                str(driver),
                *_cache_argv(inputs, images, cache, output=output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for output in slices
    ]
    outputs = [builder.communicate(timeout=600)[0] for builder in builders]

    assert [builder.returncode for builder in builders] == [0] * count, outputs
    spec = PreprocessingSpec(size=IMAGE_SIZE)
    root = cache_root(cache, spec)
    assert not list(root.glob("*.tmp"))
    index = json.loads((root / "cache_index.json").read_text(encoding="utf-8"))
    expected = {entry.image_id for entry in load_manifest(inputs.target_dir).entries}
    assert set(index["images"]) == expected
    for output in slices:
        manifest = load_manifest(output / "target")
        identity = verify_image_cache(cache, spec, manifest, ("train", "val"), images)
        assert len(identity) == 64


def test_cli_classical_fit_refuses_to_run_without_its_feature_cache(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def in_process(
        workspace_file: Path,
    ) -> Callable[[Job, float | None, Path], ChildResult]:
        workspace = Workspace.load(workspace_file)

        def launch(job: Job, timeout: float | None, log: Path) -> ChildResult:
            run_job(workspace, job)
            return ChildResult(0, False, 1.0, "")

        return launch

    monkeypatch.setattr(cli, "subprocess_launcher", in_process)
    images = _image_copy(inputs, tmp_path)
    image_cache = tmp_path / "image-cache"
    main(_cache_argv(inputs, images, image_cache))
    workspace = _workspace(inputs, images, tmp_path, image_cache)

    # Nothing was prepared: no feature cache and no receipt for it.
    with pytest.raises((ValueError, FileNotFoundError), match="feature cache"):
        _train_fit(workspace, CLASSICAL_FIT, tmp_path)

    benchmark = _benchmark(tmp_path)
    main(
        [
            "queue",
            "run",
            "--benchmark",
            str(benchmark),
            "--freeze-dir",
            str(inputs.freeze_dir),
            "--image-root",
            str(images),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--cache-dir",
            str(tmp_path / "features"),
            "--image-cache-dir",
            str(image_cache),
            "--accelerator",
            "cpu",
            "--allow-cpu",
            "--no-pretrained",
        ]
    )
    store = RunStore(tmp_path / "runs")
    for prepare_id in (PREPARE_RESERVE, PREPARE_DRAW):
        prepared = store.receipt(prepare_id)["results"]
        assert prepared["seconds"] > 0
        assert prepared["extraction_seconds"] >= 0
        assert len(prepared["image_cache"]) == 64
    classical = store.receipt(CLASSICAL_FIT)["results"]
    assert classical["image_cache"] == prepared["image_cache"]
    assert classical["feature_cache"]["train"]["rows"] > 0

    # The receipts are intact but the matrices they pin are gone.
    for stale in (tmp_path / "features").iterdir():
        stale.unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="feature cache"):
        _train_fit(workspace, CLASSICAL_FIT, tmp_path)

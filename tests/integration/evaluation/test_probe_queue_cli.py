"""The probe stage of the queue, driven through the CLI on the synthetic target."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import jute_disease.cli as cli
from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import ChildResult, Job, RunStore, Workspace, run_job

# The 32 px frame covers a 64 px image entirely, so the queue reads a 128 px
# letterbox, where the frame leaves a blank interior.
IMAGE_SIZE = 128
BORDER = "probe/border/classical/lr-crafted"
PAD_CUE = "probe/pad-cue/pool-A/view-1/image-random/seed-42"
TUNE = "classical/tune/A/lr-crafted"
RESERVE = "classical/prepare/pool-A/reserve"
PROBES = (BORDER, PAD_CUE)


def _benchmark(tmp_path: Path) -> Path:
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["recipe"].update(
        image_size=IMAGE_SIZE,
        seeds=[42],
        source_seeds=[42],
        max_epochs=1,
        effective_batch=8,
        num_workers=0,
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


def _queue(verb: str, target: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "queue",
        verb,
        "--benchmark",
        str(_benchmark(tmp_path)),
        "--freeze-dir",
        str(target.parent),
        "--image-root",
        str(target.parents[1] / "images"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--cache-dir",
        str(tmp_path / "features"),
        "--accelerator",
        "cpu",
        "--allow-cpu",
        "--no-pretrained",
        *extra,
    ]


def test_queue_plan_lists_the_probes_after_what_they_depend_on(
    target: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    main(_queue("plan", target, tmp_path, "--probes"))
    stages = json.loads(capsys.readouterr().out)

    planned = [fit["fit"] for stage in stages for fit in stage["fits"]]
    assert set(PROBES) <= set(planned)
    # The border probe reuses its base fit's tuning, so the queue must run that
    # first or omit the probe as having an incomplete dependency.
    assert planned.index(RESERVE) < planned.index(TUNE) < planned.index(BORDER)

    capsys.readouterr()
    main(_queue("plan", target, tmp_path))
    plain = [
        fit["fit"]
        for stage in json.loads(capsys.readouterr().out)
        for fit in stage["fits"]
    ]
    assert not set(PROBES) & set(plain)


def test_queue_trains_the_probes_into_receipts_that_read_the_image_cache(
    target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    image_cache = tmp_path / "image-cache"
    main(
        [
            "freeze",
            "cache",
            "--output",
            str(target.parent),
            "--image-root",
            str(target.parents[1] / "images"),
            "--cache-dir",
            str(image_cache),
            "--image-size",
            str(IMAGE_SIZE),
            "--workers",
            "2",
        ]
    )

    cache_options = ("--probes", "--image-cache-dir", str(image_cache))
    main(_queue("run", target, tmp_path, *cache_options))
    store = RunStore(tmp_path / "runs")

    for fit_id in (RESERVE, TUNE, *PROBES):
        assert store.is_complete(fit_id), fit_id
    border = store.receipt(BORDER)
    pad_cue = store.receipt(PAD_CUE)
    assert border["results"]["variant"] == "border"
    assert pad_cue["results"]["variant"] == "pad_cue"
    # Both read the content-hashed image cache the freeze step built.
    assert len(border["results"]["image_cache"]) == 64
    assert border["results"]["image_cache"] == pad_cue["results"]["image_cache"]
    # Each probe is a fit of its own, so its receipt binds its own identity.
    assert border["identity"] != pad_cue["identity"]

    # A second run finds every fit complete and trains nothing again.
    main(_queue("run", target, tmp_path, *cache_options))
    report = json.loads((tmp_path / "runs" / "queue.json").read_text("utf-8"))
    assert set(report["skipped"]) == {RESERVE, TUNE, *PROBES}

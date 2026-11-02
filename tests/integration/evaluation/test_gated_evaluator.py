"""The gated evaluator through the ``jute-disease`` CLI on the synthetic fixture."""

import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import load_manifest
from jute_disease.training import load_benchmark

LABEL_COUNT = 2
FIT = "fixture/mobilenet_v2_direct/seed-{}"
ANALYSES = [
    "background-probe",
    "border-probe",
    "gradcam-test-pass",
    "leaf-only",
    "scene-cells",
]


def _cli(*args: object) -> None:
    main([str(a) for a in args])


def _declared_fits() -> list[str]:
    benchmark = load_benchmark(BENCHMARK_CONFIG)
    return [fit.fit_id for fit in benchmark.fits()]


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _spec(tmp_path: Path, extra_fits: list[str], probes: list[str]) -> Path:
    return _write(
        tmp_path / "spec.json",
        {
            "fit_ids": _declared_fits() + probes + extra_fits,
            "analysis_ids": ANALYSES,
            "selections": {"cnn/tune/pool-A": 1.0},
            "letterbox_decision": {"choice": "letterbox"},
            "mask_audit": {"passed": True},
            "border_probe": {
                "start": {"width_px": 32, "leaf_cutoff": 0.05},
                "final": {"width_px": 32, "leaf_cutoff": 0.05},
            },
            "budget": {"a100_gpu_hours": 10},
        },
    )


def _outcomes(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "outcomes.json",
        {
            "partition": "val",
            "letterbox_comparison": {"letterbox": 0.9, "resize": 0.8},
            "selections": {"cnn/tune/pool-A": {"selected": 1.0, "val_macro_f1": 0.9}},
            "mask_audit": {"passed": True},
        },
    )


def _probe_ids(capsys: pytest.CaptureFixture[str]) -> list[str]:
    capsys.readouterr()
    _cli("evaluate", "probe-plan")
    return json.loads(capsys.readouterr().out)


def _freeze(
    tmp_path: Path,
    root: Path,
    capsys: pytest.CaptureFixture[str],
    extra_fits: list[str],
    name: str = "freeze-record.json",
    reads_log: Path | None = None,
) -> Path:
    arguments: list[object] = [
        "evaluate",
        "freeze-record",
        "--spec",
        _spec(tmp_path, extra_fits, _probe_ids(capsys)),
        "--outcomes",
        _outcomes(tmp_path),
        "--output",
        tmp_path / name,
        "--root",
        root,
    ]
    if reads_log is not None:
        arguments += ["--reads-log", reads_log]
    _cli(*arguments)
    return tmp_path / name


def _predictions(target: Path, path: Path, flip: int, ids: list[str] | None) -> None:
    entries = sorted(load_manifest(target).partition("test"), key=lambda e: e.image_id)
    labels = sorted({e.label for e in load_manifest(target).entries})
    chosen = [e for e in entries if ids is None or e.image_id in ids]
    predicted = [
        e.label
        if (i % 4 != flip)
        else labels[(labels.index(e.label) + 1) % len(labels)]
        for i, e in enumerate(chosen)
    ]
    _write(path, {"image_ids": [e.image_id for e in chosen], "predictions": predicted})


def test_test_read_refused_without_records_and_permitted_after(
    tmp_path: Path, target: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = (
        "evaluate",
        "test",
        "--target-dir",
        target,
        "--root",
        root,
        "--output",
        tmp_path / "out" / "test.json",
        "--resamples",
        20,
        "--permutations",
        20,
    )
    with pytest.raises(SystemExit, match="no frozen selection record"):
        _cli(*arguments, "--freeze-record", tmp_path / "missing.json")
    assert not (tmp_path / "out").exists()

    probes = _probe_ids(capsys)
    _cli(
        "evaluate",
        "freeze-record",
        "--spec",
        _spec(tmp_path, [FIT.format(0), FIT.format(1)], probes),
        "--outcomes",
        _outcomes(tmp_path),
        "--output",
        tmp_path / "freeze-record.json",
        "--root",
        root,
    )
    for seed in (0, 1):
        _predictions(target, tmp_path / "runs" / f"{seed}.json", seed + 8, None)
    _write(
        tmp_path / "runs" / "runs.json",
        [
            {
                "fit_id": FIT.format(seed),
                "system": "mobilenet_v2_direct",
                "seed": seed,
                "predictions": f"{seed}.json",
            }
            for seed in (0, 1)
        ],
    )
    with pytest.raises(SystemExit, match="no run record"):
        _cli(*arguments, "--freeze-record", tmp_path / "freeze-record.json")
    _cli(
        "evaluate",
        "run-record",
        "--freeze-record",
        tmp_path / "freeze-record.json",
        "--runs",
        tmp_path / "runs" / "runs.json",
        "--output",
        tmp_path / "runs" / "run-record.json",
    )
    _cli(
        *arguments,
        "--freeze-record",
        tmp_path / "freeze-record.json",
        "--run-record",
        tmp_path / "runs" / "run-record.json",
    )
    report = json.loads((tmp_path / "out" / "test.json").read_text(encoding="utf-8"))
    assert report["post_hoc"] is False
    assert report["systems"]["mobilenet_v2_direct"]["evaluation"]["summary"]

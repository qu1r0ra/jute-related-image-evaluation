"""Command-shell checks: every command resolves and planned commands fail loudly."""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from jute_disease import cli
from jute_disease.cli import build_parser, main

# evaluate predict is forwarded by main() without argparse.
FIT_ARGS = [
    "--workspace",
    "w.json",
    "--fit-id",
    "cnn/x/seed-42",
    "--attempt-dir",
    "a",
    "--result",
    "r.json",
]

RESOLVABLE_COMMANDS = [
    ["data", "download"],
    ["data", "synthetic", "--per-class", "2", "--size", "8"],
    ["train", "fit", *FIT_ARGS],
    ["stack"],
    ["queue", "plan"],
    ["queue", "status"],
    ["queue", "run"],
    ["evaluate", "aggregate", "--exp-names", "a"],
    ["freeze", "target"],
    ["freeze", "cache", "--budget-seconds", "5"],
    [
        "freeze",
        "source",
        "plantdoc",
        "some/dir",
        "--package-version",
        "7",
        "--package-digest",
        "0" * 64,
    ],
]

PLANNED_COMMANDS = [
    (["profile"], "#721"),
    (["evaluate"], "#721"),
]


@pytest.mark.parametrize("argv", RESOLVABLE_COMMANDS, ids=lambda a: " ".join(a[:3]))
def test_command_resolves_to_handler(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.run)


def test_evaluate_predict_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_predict = MagicMock()
    monkeypatch.setattr(cli, "predict_ml", mock_predict)
    main(["evaluate", "predict", "leaf.jpg", "--classifier", "knn"])
    mock_predict.assert_called_once_with(["leaf.jpg", "--classifier", "knn"])


@pytest.mark.parametrize(
    ("argv", "owner"),
    PLANNED_COMMANDS,
    ids=lambda v: " ".join(v) if isinstance(v, list) else v,
)
def test_planned_command_exits_nonzero_naming_owner(
    argv: list[str], owner: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert isinstance(excinfo.value.code, str)
    assert owner in excinfo.value.code


def test_missing_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("missing", ["--package-version", "--package-digest"])
def test_source_intake_requires_both_package_identity_inputs(missing: str) -> None:
    argv = ["freeze", "source", "plantdoc", "source-root"]
    if missing != "--package-version":
        argv.extend(["--package-version", "7"])
    if missing != "--package-digest":
        argv.extend(["--package-digest", "0" * 64])
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_module_entry_point_prints_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jute-disease" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "ml"],
        ["train", "baselines"],
        ["train", "cv", "x.yaml"],
        ["train", "grid", "x.yaml"],
        ["train", "pretrain"],
        ["evaluate", "ml"],
    ],
    ids=lambda a: " ".join(a),
)
def test_legacy_training_variants_are_gone(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_train_fit_runs_the_job_body(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(cli.Workspace, "load", lambda path: ("workspace", path))
    monkeypatch.setattr(
        cli, "run_job", lambda workspace, job: seen.update(workspace=workspace, job=job)
    )
    main(
        [
            "train",
            "fit",
            *FIT_ARGS,
            "--upstream-checkpoint",
            "u.ckpt",
            "--bounds",
            "8,4",
        ]
    )
    job = seen["job"]
    assert job.fit_id == "cnn/x/seed-42"
    assert job.bounds == (8, 4)
    assert str(job.upstream_checkpoint) == "u.ckpt"


def test_train_fit_rejects_malformed_bounds() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "fit", *FIT_ARGS, "--bounds", "eight"])
    assert excinfo.value.code == 2


def test_queue_run_takes_no_deadline() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["queue", "run", "--cutoff", "2026-10-31T14:00"])
    assert excinfo.value.code == 2


def _stub_queue(monkeypatch: pytest.MonkeyPatch, tmp_path, fake_queue) -> None:
    monkeypatch.setattr(
        cli, "_workspace_from", lambda args: MagicMock(runs_dir=tmp_path)
    )
    monkeypatch.setattr(cli, "make_identity", lambda *a: None)
    monkeypatch.setattr(cli, "make_executor", lambda *a: None)
    monkeypatch.setattr(cli, "subprocess_launcher", lambda *a: None)
    monkeypatch.setattr(cli, "run_queue", fake_queue)


def test_queue_run_hands_the_queue_its_executor_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def fake_queue(benchmark, store, **kwargs) -> cli.QueueReport:
        captured.update(kwargs)
        return cli.QueueReport()

    _stub_queue(monkeypatch, tmp_path, fake_queue)
    main(["queue", "run"])
    assert set(captured) == {"clock", "execute", "identity_for", "report_path"}
    assert captured["report_path"] == tmp_path / "queue.json"


def test_an_aborted_queue_exits_with_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_queue(benchmark, store, **kwargs) -> cli.QueueReport:
        raise cli.QueueAbortedError(["a", "b", "c"])

    _stub_queue(monkeypatch, tmp_path, fake_queue)
    with pytest.raises(SystemExit, match="consecutive persistent failures: a, b, c"):
        main(["queue", "run"])


def test_stack_command_reports_the_measured_stack(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "verify_stack", lambda require_cuda: {"cuda_available": not require_cuda}
    )
    main(["stack", "--allow-cpu"])
    assert '"cuda_available": true' in capsys.readouterr().out

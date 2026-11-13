"""The coverage simulation of the descriptive nested bootstrap."""

import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.evaluation import CoverageReport, simulate_coverage


def test_simulated_truth_is_known_and_the_report_is_labelled_descriptive() -> None:
    report = simulate_coverage(replicates=12, resamples=60, seed=3)
    assert isinstance(report, CoverageReport)
    # Balanced classes, symmetric errors and a mean-preserving run offset make
    # the population accuracy exactly E[q] = (0.6 + 0.9) / 2.
    assert report.truth == pytest.approx(0.75)
    assert report.nominal == 0.95
    assert report.replicates == 12
    assert 0 <= report.covered <= 12
    assert report.coverage == pytest.approx(report.covered / 12)
    assert report.status == "descriptive"
    assert report.family_resampling == "shared"
    low, high = report.coverage_interval
    assert 0 <= low <= report.coverage <= high <= 1


def test_the_simulation_is_deterministic_for_a_seed() -> None:
    first = simulate_coverage(replicates=6, resamples=40, seed=9)
    again = simulate_coverage(replicates=6, resamples=40, seed=9)
    other = simulate_coverage(replicates=6, resamples=40, seed=10)
    assert first == again
    assert first != other


def test_simulation_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        simulate_coverage(replicates=0)
    with pytest.raises(ValueError):
        simulate_coverage(replicates=5, resamples=0)
    with pytest.raises(ValueError):
        simulate_coverage(replicates=5, metric="auroc")


@pytest.mark.slow
def test_bootstrap_reaches_its_nominal_level_within_tolerance() -> None:
    # 200 replicates give a Monte Carlo standard error of about 0.015 at 95%.
    # The tolerance is 0.05 (three such errors): the intervals must cover the
    # known truth at close to the nominal rate, not merely often. About 25 s.
    report = simulate_coverage(replicates=200, resamples=200, seed=20261113)
    assert report.coverage == pytest.approx(report.nominal, abs=0.05)


def test_the_coverage_command_prints_the_report_and_writes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "coverage.json"
    main(
        [
            "evaluate",
            "bootstrap-coverage",
            "--replicates",
            "5",
            "--resamples",
            "30",
            "--seed",
            "4",
            "--output",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["status"] == "descriptive"
    assert written["replicates"] == 5
    assert written["truth"] == pytest.approx(0.75)
    assert 0 <= written["coverage"] <= 1

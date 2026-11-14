"""The coverage simulation of the descriptive nested bootstrap."""

import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.evaluation import (
    GRID_TOLERANCE,
    CoverageReport,
    coverage_grid,
    simulate_coverage,
)


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


def test_the_report_names_the_configuration_it_covers() -> None:
    single = simulate_coverage(replicates=4, resamples=20, seed=1)
    assert single.target == "system mean"
    assert single.n_systems == 1
    assert single.quality_range == (0.6, 0.9)
    assert single.n_families == 40
    narrow = simulate_coverage(
        replicates=4, resamples=20, seed=1, quality_range=(0.74, 0.76), n_families=400
    )
    assert narrow.quality_range == (0.74, 0.76)
    assert narrow.n_families == 400
    assert narrow.truth == pytest.approx(0.75)


def test_two_system_difference_in_per_system_mode_has_the_effect_as_its_truth() -> None:
    report = simulate_coverage(
        replicates=6,
        resamples=40,
        seed=2,
        n_families=12,
        effect=0.04,
        family_resampling="per-system",
    )
    assert report.target == "two-system difference"
    assert report.n_systems == 2
    assert report.truth == pytest.approx(0.04)
    assert report.family_resampling == "per-system"
    assert report.status == "descriptive"
    assert 0 <= report.covered <= 6


def test_simulation_rejects_a_quality_range_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="quality"):
        simulate_coverage(replicates=2, quality_range=(0.9, 0.6))
    with pytest.raises(ValueError, match="quality"):
        simulate_coverage(replicates=2, quality_range=(0.5, 1.2))
    with pytest.raises(ValueError, match="quality"):
        simulate_coverage(replicates=2, quality_range=(0.74, 0.76), effect=0.8)


def test_the_grid_covers_family_count_split_spread_and_a_per_system_difference() -> (
    None
):
    grid = coverage_grid(replicates=3, resamples=15, seed=5)
    shapes = {
        (r.n_families, r.quality_range, r.target, r.family_resampling) for r in grid
    }
    assert {r.n_families for r in grid} >= {20, 40, 400}
    assert {r.quality_range for r in grid} >= {(0.74, 0.76), (0.6, 0.9)}
    assert len(shapes) == len(grid)
    assert any(
        r.target == "two-system difference" and r.family_resampling == "per-system"
        for r in grid
    )
    assert all(r.status == "descriptive" and r.replicates == 3 for r in grid)
    assert coverage_grid(replicates=3, resamples=15, seed=5) == grid


def test_the_grid_command_prints_every_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "grid.json"
    main(
        [
            "evaluate",
            "bootstrap-coverage",
            "--grid",
            "--replicates",
            "2",
            "--resamples",
            "10",
            "--output",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert len(printed) == len(coverage_grid(replicates=2, resamples=10))
    assert {entry["target"] for entry in printed} == {
        "system mean",
        "two-system difference",
    }


@pytest.mark.slow
def test_coverage_under_the_default_synthetic_configuration() -> None:
    # One configuration only: one system, shared family resampling, no
    # checkpoints, 40 two-image families and split quality U(0.6, 0.9). Here the
    # family variance counted twice (once through the split-level test sets and
    # once through the family draws) offsets the undercoverage of a percentile
    # interval on ten splits, which is why the figure sits near nominal. It is
    # not a general result: see the grid test. 200 replicates give a Monte Carlo
    # standard error of about 0.015, and the tolerance is 0.05. About 25 s.
    report = simulate_coverage(replicates=200, resamples=200, seed=20261113)
    assert report.coverage == pytest.approx(report.nominal, abs=0.05)


@pytest.mark.slow
def test_the_grid_is_reported_and_never_collapses() -> None:
    # The grid is reported, not tuned to nominal. Coverage moves with the family
    # count and the split spread (1.00 when the splits barely differ, 0.90 to 0.95
    # at this size otherwise), so the only assertion is the documented floor: no
    # configuration may fall more than GRID_TOLERANCE below nominal. About 150 s.
    grid = coverage_grid(replicates=100, resamples=100, seed=20261113)
    for report in grid:
        assert report.coverage >= report.nominal - GRID_TOLERANCE, report


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

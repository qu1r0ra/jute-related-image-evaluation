import csv
import json
import subprocess
import sys
from pathlib import Path


def test_fixture_rating_round_is_blind_and_evaluated(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "data",
            "audit-fixture",
            "--output",
            str(tmp_path / "audit"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "audit"
    with (output / "owner.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120
    assert "agent_rating" not in rows[0]
    report = json.loads((output / "agreement.json").read_text())
    assert report["pass"] is True
    assert report["kappa_interval"][0] >= 0.6
    assert report["weighted_error_rate"] == 0
    assert report["very_similar"]
    samples = json.loads((output / "samples.json").read_text())
    population_pairs = {
        (p["image_a"], p["image_b"])
        for band in samples["edges"]["bands"].values()
        for p in band["pairs"]
    }
    assert all((r["image_a"], r["image_b"]) in population_pairs for r in rows)
    assert report["weighted_algorithm_error_rate"] == 0
    failed = json.loads((output / "failed_rounds.json").read_text())
    assert failed[0]["next_action"] == "revise_rubric_and_rerun"
    assert failed[1]["next_action"] == "owner_escalation"


def test_fixture_exercises_audit_samples(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "data",
            "audit-fixture",
            "--output",
            str(tmp_path / "audit"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "audit" / "samples.json").read_text())
    assert report["edges"]["bands"]["near_miss"]["population"] > 0
    assert report["edges"]["bands"]["merged"]["population"] > 0
    assert len(report["edges"]["bands"]["near_miss"]["pairs"]) == 200
    assert len(report["edges"]["bands"]["merged"]["pairs"]) == 200
    assert report["features"]
    assert report["feature_provenance"]["kind"] == "synthetic"
    assert len(report["residual"]["pools"]["fixture"]["rows"]) == 200
    assert report["residual"]["pools"]["fixture"]["population"] == 220
    assert report["residual"]["pools"]["fixture"]["rows"][0]["hash_neighbors"]
    assert report["residual"]["pools"]["fixture"]["rows"][0]["feature_neighbors"]
    assert {row["transform"] for row in report["recall"]["transforms"]} == {
        "brightness",
        "crop",
        "mirror",
    }
    assert report["recall"]["unrelated"]["total"] == 6


def test_owner_rating_import_reports_weighted_error_and_escalation(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "pair_id": str(i),
            "image_a": f"a{i}",
            "image_b": f"b{i}",
            "agent_rating": "exact" if i % 2 else "different",
            "algorithm_merge": bool(i % 2 == 0),
            "stratum": "small" if i < 20 else "large",
        }
        for i in range(40)
    ]
    source = tmp_path / "agent.json"
    source.write_text(
        json.dumps({"pairs": rows, "denominators": {"small": 10, "large": 90}}),
        encoding="utf-8",
    )
    sheet = tmp_path / "owner.csv"
    base = [sys.executable, "-m", "jute_disease.cli", "data"]
    exported = subprocess.run(
        base + ["audit-export", str(source), str(sheet)], capture_output=True, text=True
    )
    assert exported.returncode == 0, exported.stderr
    with sheet.open(newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle))
    for row in ratings:
        i = int(row["pair_id"])
        row["rating"] = "different" if i % 2 else "exact"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_a", "image_b", "rating"]
        )
        writer.writeheader()
        writer.writerows(ratings)
    first = subprocess.run(
        base + ["audit-evaluate", str(source), str(sheet)],
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    report = json.loads(first.stdout)
    assert report["weighted_error_rate"] == 1
    assert report["weighted_algorithm_error_rate"] == 0
    assert report["next_action"] == "revise_rubric_and_rerun"
    previous = tmp_path / "first.json"
    previous.write_text(first.stdout, encoding="utf-8")
    second = subprocess.run(
        base
        + [
            "audit-evaluate",
            str(source),
            str(sheet),
            "--previous",
            str(previous),
            "--revision",
            "clarified examples",
        ],
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["next_action"] == "owner_escalation"

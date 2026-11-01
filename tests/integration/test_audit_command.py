import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
        "resize",
        "crop",
        "jpeg",
    }
    recall = report["recall"]
    assert recall["source_images"] == 100
    assert recall["unrelated"]["total"] == 4950
    assert {(r["transform"], r["strength"]) for r in recall["transforms"]} == {
        ("resize", 0.5),
        ("resize", 0.75),
        ("crop", 0.05),
        ("crop", 0.1),
        ("jpeg", 70),
        ("jpeg", 85),
    }
    assert all(row["total"] == 100 and row["mild"] for row in recall["transforms"])
    assert all(
        row["actual_removed_area_max"] <= row["strength"]
        for row in recall["transforms"]
        if row["transform"] == "crop"
    )
    assert all(
        len(row["actual_removed_area"]) == 100
        and max(row["actual_removed_area"]) <= row["strength"]
        for row in recall["transforms"]
        if row["transform"] == "crop"
    )
    assert recall["threshold_change"]["condition_met"] == (
        recall["threshold_change"]["mild_recall_failed"]
        or recall["threshold_change"]["unrelated_false_merge_failed"]
    )


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
    assert report["positive_agreement"] == 0
    assert report["negative_agreement"] == 0
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
    assert json.loads(second.stdout)["cohort_digest"] == report["cohort_digest"]
    for mutation in ("denominator", "pair"):
        changed = json.loads(source.read_text(encoding="utf-8"))
        if mutation == "denominator":
            changed["denominators"]["small"] += 1
        else:
            changed["pairs"][0]["image_a"] = "replacement"
        altered = tmp_path / f"altered-{mutation}.json"
        altered.write_text(json.dumps(changed), encoding="utf-8")
        retry = subprocess.run(
            base
            + [
                "audit-evaluate",
                str(altered),
                str(sheet),
                "--previous",
                str(previous),
                "--revision",
                "clarified examples",
            ],
            capture_output=True,
            text=True,
        )
        assert retry.returncode != 0
        assert "cohort" in retry.stderr


def test_prepare_real_cohort_from_edge_samples(tmp_path: Path) -> None:
    bands = {}
    for band, distance in [("merged", 3), ("near_miss", 5)]:
        pairs = [
            {
                "image_a": f"{band}-a{i}",
                "image_b": f"{band}-b{i}",
                "distance": distance,
                "stratum": f"{distance}:same_label",
            }
            for i in range(75)
        ]
        bands[band] = {
            "population": 100,
            "strata": {f"{distance}:same_label": {"population": 100, "sampled": 75}},
            "pairs": pairs,
        }
    source = tmp_path / "edges.json"
    source.write_text(json.dumps({"seed": 42, "bands": bands}), encoding="utf-8")
    output = tmp_path / "cohort"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "data",
            "audit-cohort",
            str(source),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    agent = json.loads((output / "agent.json").read_text(encoding="utf-8"))
    with (output / "owner.csv").open(newline="", encoding="utf-8") as handle:
        owner = list(csv.DictReader(handle))
    assert len(agent["pairs"]) == len(owner) == 120
    assert sum(row["algorithm_merge"] for row in agent["pairs"]) == 60
    assert {row["algorithm_merge"] for row in agent["pairs"][:60]} == {True, False}
    assert all(row["agent_rating"] == "" for row in agent["pairs"])
    assert all(
        set(row) == {"pair_id", "image_a", "image_b", "rating"} and row["rating"] == ""
        for row in owner
    )
    assert agent["denominators"] == {
        "merged:3:same_label": 100,
        "near_miss:5:same_label": 100,
    }


def test_residual_sampler_rejects_short_pool_before_feature_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jute_disease import cli

    source = tmp_path / "short.json"
    source.write_text(
        json.dumps(
            {
                "protected_test": {"short": {"missing-test": 0}},
                "training": {"short": {"missing-train": 1}},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def forbidden_extractor() -> None:
        calls.append(True)
        raise AssertionError("pretrained extractor must not be constructed")

    monkeypatch.setattr(cli, "ImageNetExtractor", forbidden_extractor)
    with pytest.raises(SystemExit, match="short.*200"):
        cli.main(["data", "audit-sample", "residual", str(source)])
    assert not calls


def test_undefined_kappa_fails_then_escalates_after_revision(tmp_path: Path) -> None:
    rows = [
        {
            "pair_id": str(i),
            "image_a": f"a{i}",
            "image_b": f"b{i}",
            "agent_rating": "exact",
            "algorithm_merge": True,
            "stratum": "all",
        }
        for i in range(20)
    ]
    source = tmp_path / "agent.json"
    source.write_text(
        json.dumps({"pairs": rows, "denominators": {"all": 20}}), encoding="utf-8"
    )
    sheet = tmp_path / "owner.csv"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_a", "image_b", "rating"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "pair_id": r["pair_id"],
                "image_a": r["image_a"],
                "image_b": r["image_b"],
                "rating": "exact",
            }
            for r in rows
        )
    base = [
        sys.executable,
        "-m",
        "jute_disease.cli",
        "data",
        "audit-evaluate",
        str(source),
        str(sheet),
    ]
    gate = tmp_path / "gate.json"
    first = subprocess.run(
        base + ["--gate-output", str(gate)], capture_output=True, text=True, check=False
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(gate.read_text(encoding="utf-8")) == {
        "kappa_interval": None,
        "threshold": 0.6,
        "pass": False,
        "method": "Cohen kappa undefined: both raters used one category",
    }
    report = json.loads(first.stdout)
    assert report["kappa"] is None and report["kappa_interval"] is None
    assert (
        report["pass"] is False and report["next_action"] == "revise_rubric_and_rerun"
    )
    assert report["positive_agreement"] == 1 and report["negative_agreement"] is None
    prior = tmp_path / "prior.json"
    prior.write_text(first.stdout, encoding="utf-8")
    second = subprocess.run(
        base + ["--previous", str(prior), "--revision", "clarify examples"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["next_action"] == "owner_escalation"

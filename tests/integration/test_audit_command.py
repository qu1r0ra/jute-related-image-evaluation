import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from jute_disease import cli

BASE = [sys.executable, "-m", "jute_disease.cli", "data"]
AGENT = ["--agent-name", "test-agent", "--agent-version", "t1"]


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> Any:
    cli.main(["data", *argv])
    return json.loads(capsys.readouterr().out)


def write_hashes(path: Path, per_pool: int = 40, seed: int = 7) -> dict[str, list]:
    rng = random.Random(seed)
    images: dict[str, list] = {}
    for pool in "ABC":
        values = [rng.randrange(1 << 20) for _ in range(per_pool)]
        for i in range(per_pool // 4):
            # Near copies, 0 to 3 bits apart, populate the closest merged bins.
            values[per_pool - 1 - i] = values[i] ^ ((1 << (i % 4)) - 1)
        for i, value in enumerate(values):
            images[f"{pool}/{i}.jpg"] = [value, rng.choice("xy"), pool]
    path.write_text(json.dumps(images), encoding="utf-8")
    return images


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_cohort(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, version: str = "dhash-v1"
) -> tuple[Path, dict[str, Any]]:
    hashes = tmp_path / "hashes.json"
    write_hashes(hashes)
    sample = tmp_path / "edges.json"
    sample.write_text(
        json.dumps(
            run(
                capsys,
                "audit-sample",
                "edges",
                str(hashes),
                "--grouping-version",
                version,
            )
        ),
        encoding="utf-8",
    )
    cohort_dir = tmp_path / "cohort"
    run(capsys, "audit-cohort", str(sample), str(cohort_dir), *AGENT)
    return cohort_dir, json.loads((cohort_dir / "agent.json").read_text("utf-8"))


def rate_pairs(cohort: dict[str, Any], flip_every: int = 4) -> dict[str, str]:
    """Agent rates by the algorithm decision; the owner flips every nth pair."""
    owner = {}
    for row in cohort["pairs"]:
        row["agent_rating"] = (
            "near-duplicate" if row["algorithm_merge"] else "different"
        )
        flipped = int(row["pair_id"]) % flip_every == 0
        merge = row["algorithm_merge"] != flipped
        owner[row["pair_id"]] = "near-duplicate" if merge else "different"
    return owner


def test_audit_tools_refuse_to_run_without_a_grouping_version(
    tmp_path: Path,
) -> None:
    hashes = tmp_path / "hashes.json"
    write_hashes(hashes)

    fixture = subprocess.run(
        [*BASE, "audit-fixture", "--output", str(tmp_path / "audit")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fixture.returncode != 0
    assert "--grouping-version" in fixture.stderr
    assert not (tmp_path / "audit").exists()

    for kind in ("edges", "recall"):
        sampler = subprocess.run(
            [*BASE, "audit-sample", kind, str(hashes)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert sampler.returncode != 0
        assert "--grouping-version" in sampler.stderr

    unknown = subprocess.run(
        [*BASE, "audit-sample", "edges", str(hashes), "--grouping-version", "dhash-v9"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unknown.returncode != 0
    assert "grouping version" in unknown.stderr


def test_edge_sampler_takes_every_pool_c_cross_label_merged_pair(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    hashes = tmp_path / "hashes.json"
    images = write_hashes(hashes)

    v2 = run(
        capsys, "audit-sample", "edges", str(hashes), "--grouping-version", "dhash-v2"
    )
    v1 = run(
        capsys, "audit-sample", "edges", str(hashes), "--grouping-version", "dhash-v1"
    )

    assert (v1["threshold"], v2["threshold"]) == (4, 6)
    paths = [p for p in images if p.startswith("C/")]
    expected = {
        tuple(sorted((a, b)))
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
        if images[a][1] != images[b][1]
        and (images[a][0] ^ images[b][0]).bit_count() <= 6
    }
    census = {
        (p["image_a"], p["image_b"])
        for p in v2["pairs"]
        if v2["strata"][p["stratum"]]["census"]
    }
    assert census == expected
    assert any((images[a][0] ^ images[b][0]).bit_count() <= 2 for a, b in expected)
    for pair in v2["pairs"]:
        stratum = v2["strata"][pair["stratum"]]
        assert pair["family"] in {"same_family", "split"}
        assert pair["inclusion_probability"] == pytest.approx(
            stratum["sampled"] / stratum["population"]
        )
    quality = v2["dataset_quality"]
    assert (
        quality["pool_c_mixed_label_scenes"]
        == quality["pools"]["C"]["mixed_label_scenes"]
    )
    assert all(pool["dropped_families"] == 0 for pool in quality["pools"].values())
    assert sum(pool["images"] for pool in quality["pools"].values()) == len(images)


def test_cohort_records_grouping_version_and_agent_rater(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cohort_dir, cohort = make_cohort(capsys, tmp_path, "dhash-v2")

    assert cohort["grouping_version"] == "dhash-v2"
    assert cohort["threshold"] == 6
    assert cohort["agent"] == {"name": "test-agent", "version": "t1"}
    assert {"pool", "family", "inclusion_probability", "stratum"} <= set(
        cohort["pairs"][0]
    )
    owner = read_sheet(cohort_dir / "owner.csv")
    assert len(owner) == len(cohort["pairs"])
    assert list(owner[0]) == ["pair_id", "image_a", "image_b", "rating"]

    with pytest.raises(SystemExit):
        cli.main(
            ["data", "audit-cohort", str(tmp_path / "edges.json"), str(tmp_path / "x")]
        )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"seed": 1, "bands": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="grouping version"):
        cli.main(["data", "audit-cohort", str(legacy), str(tmp_path / "y"), *AGENT])


def test_evaluation_reports_exact_bounds_per_pool_and_names_the_agent(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cohort_dir, cohort = make_cohort(capsys, tmp_path)
    ratings = rate_pairs(cohort)
    (cohort_dir / "agent.json").write_text(json.dumps(cohort), encoding="utf-8")
    sheet = cohort_dir / "owner.csv"
    rows = read_sheet(sheet)
    for row in rows:
        row["rating"] = ratings[row["pair_id"]]
    write_sheet(sheet, rows)
    gate = tmp_path / "gate.json"

    report = run(
        capsys,
        "audit-evaluate",
        str(cohort_dir / "agent.json"),
        str(sheet),
        "--gate-output",
        str(gate),
    )

    assert report["grouping_version"] == "dhash-v1"
    assert report["hamming_threshold"] == 4
    assert report["agent"] == {"name": "test-agent", "version": "t1"}
    assert set(report["specific_agreement_by_pool"]) == {"A", "B", "C"}
    assert report["kappa_interval"][0] < report["kappa"] < report["kappa_interval"][1]
    assert set(json.loads(gate.read_text("utf-8"))) == {
        "kappa_interval",
        "threshold",
        "pass",
        "method",
    }
    for kind in ("wrong_merge", "false_split"):
        overall = report[kind]["overall"]
        assert overall["rate"] <= overall["upper"] and "lower" not in overall
        assert set(report[kind]["by_pool"]) == {"A", "B", "C"}
    # Owner flips every fourth pair, so some merged pairs are wrong merges.
    assert report["wrong_merge"]["overall"]["events"] > 0
    assert report["wrong_merge"]["overall"]["upper"] > 0
    for stratum in report["per_stratum"].values():
        assert stratum["lower"] <= stratum["rate"] <= stratum["upper"]
        assert stratum["kind"] in {"wrong_merge", "false_split", "same_family"}
    assert report["dataset_quality"]["pool_c_mixed_label_scenes"] >= 0
    assert "Clopper-Pearson" in report["bound_methods"]["per_stratum"]
    assert report["bound_methods"]["design_weighted"].startswith(
        "conservative one-sided"
    )


def test_evaluation_refuses_a_cohort_without_grouping_version_or_agent(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cohort_dir, cohort = make_cohort(capsys, tmp_path)
    sheet = cohort_dir / "owner.csv"
    for missing in ("grouping_version", "agent"):
        broken = {key: value for key, value in cohort.items() if key != missing}
        path = tmp_path / f"no-{missing}.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(SystemExit, match=missing):
            cli.main(["data", "audit-evaluate", str(path), str(sheet)])


def test_retry_is_bound_to_the_cohort_and_escalates(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cohort_dir, cohort = make_cohort(capsys, tmp_path)
    rate_pairs(cohort)
    source = cohort_dir / "agent.json"
    source.write_text(json.dumps(cohort), encoding="utf-8")
    sheet = cohort_dir / "owner.csv"
    rows = read_sheet(sheet)
    for row in rows:
        # The owner reverses every agent decision, so the raters disagree.
        agent = cohort["pairs"][int(row["pair_id"])]["agent_rating"]
        row["rating"] = "different" if agent == "near-duplicate" else "exact"
    write_sheet(sheet, rows)

    first = run(capsys, "audit-evaluate", str(source), str(sheet))
    assert first["pass"] is False
    assert first["next_action"] == "revise_rubric_and_rerun"
    previous = tmp_path / "first.json"
    previous.write_text(json.dumps(first), encoding="utf-8")
    retry = [
        "audit-evaluate",
        str(source),
        str(sheet),
        "--previous",
        str(previous),
        "--revision",
        "clarified examples",
    ]
    second = run(capsys, *retry)
    assert second["next_action"] == "owner_escalation"
    assert second["cohort_digest"] == first["cohort_digest"]

    changed = json.loads(source.read_text("utf-8"))
    changed["pairs"][0]["image_a"] = "replacement"
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="cohort"):
        cli.main(["data", *retry[:1], str(altered), *retry[2:]])


def test_fixture_round_samples_rates_and_reports_through_the_cli(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            *BASE,
            "audit-fixture",
            "--output",
            str(tmp_path / "audit"),
            "--grouping-version",
            "dhash-v2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "audit"
    samples = json.loads((output / "samples.json").read_text())
    edges = samples["edges"]
    assert edges["grouping_version"] == "dhash-v2" and edges["threshold"] == 6
    assert 200 <= len(edges["pairs"]) <= 260
    owner = read_sheet(output / "owner.csv")
    assert len(owner) == len(edges["pairs"])
    assert "agent_rating" not in owner[0]
    sampled = {(p["image_a"], p["image_b"]) for p in edges["pairs"]}
    assert all((r["image_a"], r["image_b"]) in sampled for r in owner)
    report = json.loads((output / "agreement.json").read_text())
    assert report["pass"] is True
    assert report["kappa_interval"][0] >= 0.6
    assert report["agent"]["name"] == "synthetic-fixture-agent"
    assert report["wrong_merge"]["overall"]["rate"] == 0
    assert report["wrong_merge"]["overall"]["upper"] > 0
    assert report["very_similar"]
    recall = samples["recall"]
    assert recall["source_images"] == 100 and recall["unrelated"]["total"] == 4950
    assert {row["transform"] for row in recall["transforms"]} == {
        "resize",
        "crop",
        "jpeg",
    }
    assert samples["features"] and samples["feature_provenance"]["kind"] == "synthetic"
    failed = json.loads((output / "failed_rounds.json").read_text())
    assert failed[0]["next_action"] == "revise_rubric_and_rerun"
    assert failed[1]["next_action"] == "owner_escalation"


def test_residual_sampler_rejects_short_pool_before_feature_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

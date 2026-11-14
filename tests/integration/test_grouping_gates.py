"""The grouping gates and the Pool C re-freeze, through the command line.

Every freeze here is a real ``freeze target`` run on the synthetic dataset. The gate
records come from the real audit commands (sample, cohort, evaluate) run on a small
hash set built so that each scenario's rates are known: the audit takes every pair
(a census design), so the design-weighted bounds equal the observed rates. The
owner's ratings are synthetic; the blind sitting itself is not part of this work.
"""

import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from jute_disease import cli
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import make_synthetic_data

from .gate_records import gate_record

pytestmark = pytest.mark.slow

CENSUS_ALL = {
    "pools": {"C": {"budget": 1000, "allocation": "proportional"}},
    "census": [{"pool": "C"}],
    "neyman_prior": {"merged": 0.05, "near_miss": 0.15},
}
PAIRS = 8
AGENT = ["--agent-name", "test-agent", "--agent-version", "t1"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("grouping-gates")
    make_synthetic_data(base / "images", per_class=20, quarantine_count=4)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (base / "reviews.json").write_text(
        json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
    )
    # Pool C hashes: merged pairs 2 bits apart, split pairs 8 apart (a near miss
    # under dhash-v2 and dhash-v3), and far near-misses 14 apart.
    rng = random.Random(11)
    images: dict[str, list] = {}
    for kind, bits in (("merge", 0b11), ("split", 0xFF), ("far", 0x3FFF)):
        for index in range(PAIRS):
            value = rng.getrandbits(64)
            images[f"C/{kind}-{index}-a.jpg"] = [value, "x", "C"]
            images[f"C/{kind}-{index}-b.jpg"] = [value ^ bits, "x", "C"]
    (base / "hashes.json").write_text(json.dumps(images), encoding="utf-8")
    (base / "design.json").write_text(json.dumps(CENSUS_ALL), encoding="utf-8")
    return {"base": base, "images": base / "images"}


def _freeze(
    world: dict[str, Path], name: str, *extra: str
) -> tuple[Path, subprocess.CompletedProcess]:
    out = world["base"] / name
    result = _run(
        "freeze",
        "target",
        "--image-root",
        str(world["images"]),
        "--rules",
        str(world["base"] / "rules.json"),
        "--reviews",
        str(world["base"] / "reviews.json"),
        "--output",
        str(out),
        "--workers",
        "2",
        *(
            (
                "--plantvillage-root",
                str(world["images"]),
                "--plantdoc-root",
                str(world["images"]),
            )
            if "--grouping-version" in extra
            else ()
        ),
        *extra,
    )
    return out, result


@pytest.fixture(scope="module")
def inventory(world: dict[str, Path]) -> str:
    """The inventory digest of the synthetic images, from an ungated freeze."""
    _, result = _freeze(world, "probe")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["receipt"]["dependencies"]["inventory"]


def _owner_rating(row: dict[str, Any], scenario: dict[str, int | bool]) -> str:
    """Rate a pair the way the scenario's owner would; the agent agrees."""
    name = row["image_a"].split("/")[1]
    index = int(name.split("-")[1])
    if name.startswith("merge"):
        wrong = index < scenario.get("wrong_merges", 0)
        return "different" if wrong else "near-duplicate"
    if name.startswith("split"):
        return "near-duplicate" if scenario.get("split_is_duplicate") else "different"
    return "different"


def _sheet(path: Path, ratings: dict[str, str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["rating"] = ratings[row["pair_id"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohort_gate(
    world: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    name: str,
    inventory: str,
    version: str,
    seed: int = 42,
    **scenario: int | bool,
) -> Path:
    """Sample, rate and evaluate one cohort; return the gate record it wrote."""
    base = world["base"] / name
    base.mkdir()

    def run(*argv: str) -> None:
        cli.main(["data", *argv])

    run(
        "audit-sample",
        "edges",
        str(world["base"] / "hashes.json"),
        "--grouping-version",
        version,
        "--design",
        str(world["base"] / "design.json"),
        "--inventory-digest",
        f"C={inventory}",
        "--seed",
        str(seed),
    )
    (base / "edges.json").write_text(capsys.readouterr().out, encoding="utf-8")
    run("audit-cohort", str(base / "edges.json"), str(base / "cohort"), *AGENT)
    capsys.readouterr()
    cohort = json.loads((base / "cohort" / "agent.json").read_text("utf-8"))
    ratings = {}
    for row in cohort["pairs"]:
        ratings[row["pair_id"]] = row["agent_rating"] = _owner_rating(row, scenario)
    (base / "cohort" / "agent.json").write_text(json.dumps(cohort), encoding="utf-8")
    _sheet(base / "cohort" / "owner.csv", ratings)
    run(
        "audit-evaluate",
        str(base / "cohort" / "agent.json"),
        str(base / "cohort" / "owner.csv"),
        "--gate-output",
        str(base / "gate.json"),
    )
    capsys.readouterr()
    return base / "gate.json"


def _outcomes(out: Path) -> dict[str, str]:
    return json.loads((out / "target" / "grouping.json").read_text())["gate_outcomes"]


def _policies(freeze: Path, output: Path) -> subprocess.CompletedProcess:
    return _run(
        "freeze",
        "policies",
        "--pool",
        f"C={freeze}",
        "--output",
        str(output),
        "--reserve-fraction",
        "0.1",
        "--train-fraction",
        "0.7",
        "--val-fraction",
        "0.15",
        "--test-fraction",
        "0.15",
    )


def test_gate_record_binds_the_cohort_and_a_matching_freeze_accepts_it(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    gate = cohort_gate(world, capsys, "bound", inventory, "dhash-v2")
    record = json.loads(gate.read_text())
    assert record["grouping_version"] == "dhash-v2"
    assert record["hamming_threshold"] == 6
    assert record["inventory_digests"] == {"C": inventory}
    assert len(record["cohort_digest"]) == 64
    assert record["threshold"] == 0.6
    assert (record["wrong_merge"]["upper"], record["false_split"]["upper"]) == (0, 0)
    out, result = _freeze(
        world,
        "bound-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    assert _outcomes(out) == {
        "agreement": "pass",
        "wrong_merge": "pass",
        "false_split": "pass",
        "threshold_change": "applied",
    }
    assert json.loads((out / "target" / "grouping.json").read_text())[
        "agreement_results"
    ] == json.loads(gate.read_text())


@pytest.mark.parametrize(
    ("edit", "version", "pool", "message"),
    [
        ("other_inventory", "dhash-v2", "C", "inventory digest"),
        ("none", "dhash-v1", "C", "grouping version"),
        ("none", "dhash-v2", "A", "pool"),
        ("threshold", "dhash-v2", "C", "bindings"),
        ("other_cohort_version", "dhash-v2", "C", "bindings"),
    ],
)
def test_freeze_target_refuses_a_gate_whose_bindings_differ(
    world: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    inventory: str,
    edit: str,
    version: str,
    pool: str,
    message: str,
) -> None:
    other = hashlib.sha256(b"another inventory").hexdigest()
    gate = cohort_gate(
        world,
        capsys,
        f"mismatch-{edit}-{version}-{pool}",
        other if edit == "other_inventory" else inventory,
        "dhash-v2",
    )
    record = json.loads(gate.read_text())
    if edit == "threshold":
        record["hamming_threshold"] = 4
    if edit == "other_cohort_version":
        record["grouping_version"] = "dhash-v9"
    gate.write_text(json.dumps(record), encoding="utf-8")
    out, result = _freeze(
        world,
        f"refused-{edit}-{version}-{pool}",
        "--grouping-version",
        version,
        "--pool",
        pool,
        "--agreement-results",
        str(gate),
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not (out / "receipt.json").exists()


def test_freeze_policies_refuses_a_pool_until_every_gate_has_passed(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    ungated, result = _freeze(world, "ungated", "--grouping-version", "dhash-v2")
    assert result.returncode == 0, result.stderr
    refused = _policies(ungated, world["base"] / "ungated-policies")
    assert refused.returncode != 0
    assert "gate" in refused.stderr

    for wrong_merges, gate_name in ((2, "wrong-merge"), (0, "passing")):
        gate = cohort_gate(
            world,
            capsys,
            f"policies-{gate_name}",
            inventory,
            "dhash-v2",
            wrong_merges=wrong_merges,
        )
        out, result = _freeze(
            world,
            f"policies-{gate_name}-freeze",
            "--grouping-version",
            "dhash-v2",
            "--pool",
            "C",
            "--agreement-results",
            str(gate),
        )
        assert result.returncode == 0, result.stderr
        policies = _policies(out, world["base"] / f"policies-{gate_name}-out")
        if wrong_merges:
            assert _outcomes(out)["wrong_merge"] == "fail"
            assert policies.returncode != 0
            assert "wrong_merge" in policies.stderr
        else:
            assert policies.returncode == 0, policies.stderr


def test_kappa_threshold_is_a_constant_and_not_an_input(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    for flag in ("--threshold", "--kappa-threshold"):
        refused = _run("data", "audit-evaluate", "agent.json", "owner.csv", flag, "0.3")
        assert refused.returncode != 0
        assert "unrecognized arguments" in refused.stderr
    record = gate_record(inventory, kappa_lower=0.4)
    record["threshold"] = 0.3
    record["pass"] = True
    gate = world["base"] / "loosened-kappa.json"
    gate.write_text(json.dumps(record), encoding="utf-8")
    _, result = _freeze(
        world,
        "loosened-kappa-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode != 0
    assert "kappa threshold is fixed" in result.stderr
    # At the fixed 0.6 the same lower bound fails; it is not loosened.
    record = gate_record(inventory, kappa_lower=0.59)
    gate.write_text(json.dumps(record), encoding="utf-8")
    out, result = _freeze(
        world,
        "kappa-below-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    assert _outcomes(out)["agreement"] == "fail"
    forged = gate_record(inventory, kappa_lower=0.59) | {"pass": True}
    gate.write_text(json.dumps(forged), encoding="utf-8")
    _, result = _freeze(
        world,
        "kappa-forged-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    ("upper", "expected"), [(0.05, "pass"), (0.0501, "fail"), (0.4, "fail")]
)
def test_wrong_merge_gate_requires_a_pool_c_upper_bound_of_at_most_five_percent(
    world: dict[str, Path], inventory: str, upper: float, expected: str
) -> None:
    gate = world["base"] / f"wrong-merge-{upper}.json"
    gate.write_text(
        json.dumps(gate_record(inventory, wrong_merge=upper)), encoding="utf-8"
    )
    out, result = _freeze(
        world,
        f"wrong-merge-{upper}-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    assert _outcomes(out)["wrong_merge"] == expected
    forged = gate_record(inventory, wrong_merge=upper)
    forged["wrong_merge"]["pass"] = True
    gate.write_text(json.dumps(forged), encoding="utf-8")
    _, forged_result = _freeze(
        world,
        f"wrong-merge-{upper}-forged",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert (forged_result.returncode == 0) == (expected == "pass")


def _tripped_freeze(
    world: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    inventory: str,
    tag: str,
) -> tuple[Path, Path]:
    gate = cohort_gate(
        world, capsys, f"tripped-{tag}", inventory, "dhash-v2", split_is_duplicate=True
    )
    out, result = _freeze(
        world,
        f"tripped-{tag}-freeze",
        "--grouping-version",
        "dhash-v2",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    return out, gate


def test_false_split_rule_refreezes_pool_c_at_the_next_threshold_on_a_new_cohort(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    tripped, gate = _tripped_freeze(world, capsys, inventory, "refreeze")
    record = json.loads(gate.read_text())
    assert record["false_split"]["upper"] > 0.20
    assert _outcomes(tripped)["false_split"] == "refreeze_required"
    assert _policies(tripped, world["base"] / "tripped-out").returncode != 0

    second = cohort_gate(world, capsys, "second", inventory, "dhash-v3", seed=43)
    refreeze = (
        "--grouping-version",
        "dhash-v3",
        "--pool",
        "C",
        "--agreement-results",
        str(second),
    )
    _, bare = _freeze(world, "refreeze-bare", *refreeze)
    assert bare.returncode != 0
    _, same_cohort = _freeze(
        world,
        "refreeze-same-cohort",
        "--grouping-version",
        "dhash-v3",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
        "--refreeze-from",
        str(tripped),
    )
    assert same_cohort.returncode != 0
    out, result = _freeze(world, "refrozen", *refreeze, "--refreeze-from", str(tripped))
    assert result.returncode == 0, result.stderr
    grouping = json.loads((out / "target" / "grouping.json").read_text())
    assert grouping["algorithm_version"] == "dhash-v3"
    assert grouping["parameters"]["threshold"] == 7
    assert grouping["refreeze"]["from_version"] == "dhash-v2"
    assert grouping["refreeze"]["from_cohort_digest"] == record["cohort_digest"]
    assert (
        grouping["agreement_results"]["cohort_digest"]
        != grouping["refreeze"]["from_cohort_digest"]
    )
    assert _outcomes(out)["false_split"] == "pass"
    accepted = _policies(out, world["base"] / "refrozen-out")
    assert accepted.returncode == 0, accepted.stderr


def test_a_second_false_split_failure_escalates_and_no_second_refreeze_is_made(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    tripped, _ = _tripped_freeze(world, capsys, inventory, "escalate")
    gate = cohort_gate(
        world,
        capsys,
        "second-failure",
        inventory,
        "dhash-v3",
        seed=44,
        split_is_duplicate=True,
    )
    assert json.loads(gate.read_text())["false_split"]["upper"] > 0.20
    _, failed = _freeze(
        world,
        "second-failure-freeze",
        "--grouping-version",
        "dhash-v3",
        "--pool",
        "C",
        "--agreement-results",
        str(gate),
        "--refreeze-from",
        str(tripped),
    )
    assert failed.returncode != 0
    assert "escalate to the owner" in failed.stderr

    passing = cohort_gate(world, capsys, "third", inventory, "dhash-v3", seed=45)
    refrozen, result = _freeze(
        world,
        "refrozen-once",
        "--grouping-version",
        "dhash-v3",
        "--pool",
        "C",
        "--agreement-results",
        str(passing),
        "--refreeze-from",
        str(tripped),
    )
    assert result.returncode == 0, result.stderr
    again = cohort_gate(world, capsys, "fourth", inventory, "dhash-v3", seed=46)
    _, second = _freeze(
        world,
        "refrozen-twice",
        "--grouping-version",
        "dhash-v3",
        "--pool",
        "C",
        "--agreement-results",
        str(again),
        "--refreeze-from",
        str(refrozen),
    )
    assert second.returncode != 0
    assert "escalate to the owner" in second.stderr


def test_false_split_is_evaluated_once_on_the_owners_imported_ratings(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], inventory: str
) -> None:
    base = world["base"] / "once"
    base.mkdir()
    cli.main(
        [
            "data",
            "audit-sample",
            "edges",
            str(world["base"] / "hashes.json"),
            "--grouping-version",
            "dhash-v2",
            "--design",
            str(world["base"] / "design.json"),
            "--inventory-digest",
            f"C={inventory}",
        ]
    )
    (base / "edges.json").write_text(capsys.readouterr().out, encoding="utf-8")
    cli.main(
        ["data", "audit-cohort", str(base / "edges.json"), str(base / "cohort"), *AGENT]
    )
    capsys.readouterr()
    cohort = json.loads((base / "cohort" / "agent.json").read_text("utf-8"))
    owner = {}
    for row in cohort["pairs"]:
        owner[row["pair_id"]] = _owner_rating(row, {"split_is_duplicate": True})
        # The agent rates every pair the opposite way, so kappa fails in round 1.
        row["agent_rating"] = (
            "different" if owner[row["pair_id"]] == "near-duplicate" else "exact"
        )
    (base / "cohort" / "agent.json").write_text(json.dumps(cohort), encoding="utf-8")
    _sheet(base / "cohort" / "owner.csv", owner)
    evaluate = [
        "data",
        "audit-evaluate",
        str(base / "cohort" / "agent.json"),
        str(base / "cohort" / "owner.csv"),
    ]
    cli.main(evaluate)
    first = json.loads(capsys.readouterr().out)
    assert first["pass"] is False
    (base / "first.json").write_text(json.dumps(first), encoding="utf-8")

    # Round 2: the agent is revised and rates with the owner; the owner sheet is
    # changed to say the split pairs are different. The false-split rule is not
    # evaluated again, so the recorded bound stays the round-1 bound.
    for row in cohort["pairs"]:
        row["agent_rating"] = _owner_rating(row, {})
    (base / "cohort" / "agent.json").write_text(json.dumps(cohort), encoding="utf-8")
    _sheet(
        base / "cohort" / "owner.csv",
        {row["pair_id"]: _owner_rating(row, {}) for row in cohort["pairs"]},
    )
    cli.main(
        [
            *evaluate,
            "--previous",
            str(base / "first.json"),
            "--revision",
            "clarified",
            "--gate-output",
            str(base / "gate.json"),
        ]
    )
    second = json.loads(capsys.readouterr().out)
    record = json.loads((base / "gate.json").read_text())
    assert second["pass"] is True
    assert second["false_split"] == first["false_split"]
    assert (
        record["false_split"]["upper"] == first["false_split"]["by_pool"]["C"]["upper"]
    )
    assert record["false_split"]["evaluated_round"] == 1
    assert record["false_split"]["exceeded"] is True

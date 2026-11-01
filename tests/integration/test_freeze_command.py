"""The shared freeze command, run as a user would, on the synthetic fixture."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    PreprocessingSpec,
    inventory_digest,
    load_manifest,
    make_synthetic_data,
    read_inventory,
)

pytestmark = pytest.mark.slow


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("freeze-command")
    make_synthetic_data(base / "images", per_class=20, quarantine_count=4)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (base / "reviews.json").write_text(
        json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
    )
    return {
        "images": base / "images",
        "rules": base / "rules.json",
        "reviews": base / "reviews.json",
        "base": base,
    }


def _freeze(
    inputs: dict[str, Path], name: str, *extra: str
) -> tuple[Path, subprocess.CompletedProcess]:
    out = inputs["base"] / name
    if "--grouping-version" in extra and "--plantvillage-root" not in extra:
        extra = (
            *extra,
            "--plantvillage-root",
            str(inputs["images"]),
            "--plantdoc-root",
            str(inputs["images"]),
        )
    result = _run(
        "freeze",
        "target",
        "--image-root",
        str(inputs["images"]),
        "--rules",
        str(inputs["rules"]),
        "--reviews",
        str(inputs["reviews"]),
        "--output",
        str(out),
        "--workers",
        "2",
        *extra,
    )
    return out, result


def test_freeze_command_writes_a_verifiable_receipt(inputs: dict[str, Path]) -> None:
    out, result = _freeze(inputs, "first")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    manifest = load_manifest(out / "target")
    manifest_data = json.loads((out / "target" / "manifest.json").read_text())
    receipt = report["receipt"]
    assert receipt["manifest_digest"] == manifest.digest()
    assert receipt["seed"] == 42
    assert receipt["fractions"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    dependencies = receipt["dependencies"]
    assert {"inventory", "rules", "reviews", "identity", "preprocessing"} <= set(
        dependencies
    )
    assert dependencies["inventory"] == inventory_digest(
        read_inventory(out / "inventory.jsonl")
    )
    assert dependencies["preprocessing"] == PreprocessingSpec().digest()
    assert manifest_data["version"] == 2
    assert all("pad_fraction" in entry for entry in manifest_data["entries"])
    assert report["pad_fraction_by_source_and_class"]


def test_lineage_a_second_process_reproduces_the_same_freeze(
    inputs: dict[str, Path],
) -> None:
    first_out, first = _freeze(inputs, "lineage-a")
    second_out, second = _freeze(inputs, "lineage-b")
    assert first.returncode == second.returncode == 0
    assert json.loads(first.stdout)["receipt"] == json.loads(second.stdout)["receipt"]
    for name in ("manifest.json", "receipt.json"):
        assert (first_out / "target" / name).read_bytes() == (
            second_out / "target" / name
        ).read_bytes()


def test_regression_a_different_freeze_is_not_replaced_without_force(
    inputs: dict[str, Path],
) -> None:
    out, first = _freeze(inputs, "guarded")
    assert first.returncode == 0
    before = (out / "target" / "receipt.json").read_bytes()
    _, refused = _freeze(inputs, "guarded", "--seed", "7")
    assert refused.returncode != 0
    assert "--force" in refused.stderr
    assert (out / "target" / "receipt.json").read_bytes() == before
    _, forced = _freeze(inputs, "guarded", "--seed", "7", "--force")
    assert forced.returncode == 0
    assert (out / "target" / "receipt.json").read_bytes() != before


def test_regression_a_changed_input_changes_its_dependency_hash(
    inputs: dict[str, Path],
) -> None:
    _, base = _freeze(inputs, "dep-base")
    rules = json.loads(inputs["rules"].read_text(encoding="utf-8"))
    rules["min_support"] = 3
    changed = inputs["base"] / "rules-changed.json"
    changed.write_text(json.dumps(rules), encoding="utf-8")
    other = {**inputs, "rules": changed}
    _, moved = _freeze(other, "dep-moved")
    old = json.loads(base.stdout)["receipt"]["dependencies"]
    new = json.loads(moved.stdout)["receipt"]["dependencies"]
    assert old["rules"] != new["rules"]
    assert old["inventory"] == new["inventory"]


def _source(inputs: dict[str, Path], *extra: str) -> subprocess.CompletedProcess:
    return _run(
        "freeze",
        "source",
        "plantdoc",
        str(inputs["base"] / "no-such-package"),
        "--output",
        str(inputs["base"] / "no-such-freeze"),
        *extra,
    )


def test_source_command_requires_both_package_pins(inputs: dict[str, Path]) -> None:
    no_digest = _source(inputs, "--package-version", "7")
    no_version = _source(inputs, "--package-digest", "0" * 64)
    assert no_digest.returncode == no_version.returncode == 2
    assert "--package-digest" in no_digest.stderr
    assert "--package-version" in no_version.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--package-version", "6", "--package-digest", "0" * 64), "pinned version"),
        (("--package-version", "7", "--package-digest", "xyz"), "digest"),
        (
            ("--package-version", "7", "--package-digest", "0" * 64),
            "tracked content pin",
        ),
    ],
)
def test_source_command_rejects_invalid_pins_before_scanning(
    inputs: dict[str, Path], args: tuple[str, ...], message: str
) -> None:
    result = _source(inputs, *args)
    assert result.returncode != 0
    assert "source intake blocked" in result.stderr
    assert message in result.stderr
    assert not (inputs["base"] / "no-such-freeze").exists()


def test_dhash_freeze_receipt_binds_groups_and_cannot_replace_existing_freeze(
    inputs: dict[str, Path],
) -> None:
    out, first = _freeze(inputs, "dhash", "--grouping-version", "dhash-v1")
    assert first.returncode == 0, first.stderr
    report = json.loads(first.stdout)
    grouping = json.loads((out / "target" / "grouping.json").read_text())
    assert grouping["algorithm_version"] == "dhash-v1"
    assert report["receipt"]["dependencies"]["grouping"] == grouping["digest"]
    assert grouping["denominators"]["images"] > 0
    manifest = load_manifest(out / "target")
    groups = {entry.path: entry.group for entry in manifest.entries}
    assert any(len(family) > 1 for family in grouping["families"])
    for family in grouping["families"]:
        assert len({groups[path] for path in family if path in groups}) <= 1
    _, repeat = _freeze(inputs, "dhash", "--grouping-version", "dhash-v1")
    assert repeat.returncode == 0
    assert json.loads(repeat.stdout)["receipt"] == report["receipt"]
    _, refused = _freeze(
        inputs, "dhash", "--grouping-version", "dhash-v1", "--seed", "7", "--force"
    )
    assert refused.returncode != 0


def test_dhash_grouping_receipt_is_verified_before_loading(
    inputs: dict[str, Path],
) -> None:
    from jute_disease.data import FreezeBlockedError

    out, result = _freeze(inputs, "dhash-tamper", "--grouping-version", "dhash-v1")
    assert result.returncode == 0, result.stderr
    path = out / "target" / "grouping.json"
    receipt = json.loads(path.read_text())
    receipt["parameters"]["threshold"] = 5
    path.write_text(json.dumps(receipt))
    with pytest.raises(FreezeBlockedError, match="grouping"):
        load_manifest(out / "target")


def test_grouped_freeze_records_ratings_and_agreement(inputs: dict[str, Path]) -> None:
    ratings = inputs["base"] / "ratings.csv"
    ratings.write_text("image_a,image_b,verdict\n", encoding="utf-8")
    agreement = inputs["base"] / "agreement.json"
    agreement.write_text(
        json.dumps(
            {
                "kappa_interval": [0.7, 0.9],
                "threshold": 0.6,
                "pass": True,
                "method": "bootstrap",
            }
        ),
        encoding="utf-8",
    )
    out, result = _freeze(
        inputs,
        "grouped-inputs",
        "--grouping-version",
        "dhash-v1",
        "--no-merge-ratings",
        str(ratings),
        "--agreement-results",
        str(agreement),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "target" / "grouping.json").read_text())
    assert receipt["no_merge_ratings"] == []
    assert receipt["agreement_results"]["method"] == "bootstrap"
    assert receipt["gate_outcomes"]["agreement"] == "pass"


def test_grouped_freeze_rejects_invalid_agreement(inputs: dict[str, Path]) -> None:
    agreement = inputs["base"] / "invalid-agreement.json"
    agreement.write_text('{"pass": true}', encoding="utf-8")
    _, result = _freeze(
        inputs,
        "bad-agreement",
        "--grouping-version",
        "dhash-v1",
        "--agreement-results",
        str(agreement),
    )
    assert result.returncode != 0
    assert "freeze blocked:" in result.stderr


def test_grouped_freeze_requires_and_records_source_screening(
    inputs: dict[str, Path],
) -> None:
    out, result = _freeze(
        inputs,
        "screened",
        "--grouping-version",
        "dhash-v1",
        "--plantvillage-root",
        str(inputs["images"]),
        "--plantdoc-root",
        str(inputs["images"]),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "target" / "grouping.json").read_text())
    assert receipt["source_screening"]["plantvillage"]["exclusions"]
    assert receipt["source_screening"]["plantdoc"]["exclusions"]


def test_freeze_command_reports_image_io_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from jute_disease import cli

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("image disappeared")

    monkeypatch.setattr(cli, "freeze_target", fail)
    with pytest.raises(SystemExit, match="freeze blocked: image disappeared"):
        cli.main(["freeze", "target"])


def test_grouped_freeze_removes_a_rated_edge(inputs: dict[str, Path]) -> None:
    import shutil

    out, first = _freeze(inputs, "rating-base", "--grouping-version", "dhash-v1")
    assert first.returncode == 0, first.stderr
    receipt = json.loads((out / "target" / "grouping.json").read_text())
    original = receipt["families"][0][0]
    images = inputs["base"] / "rated-images"
    shutil.copytree(inputs["images"], images)
    original_path = images / original
    variant = original_path.with_name(
        f"{original_path.stem}_rated{original_path.suffix}"
    )
    variant.write_bytes(original_path.read_bytes() + b"rating fixture")
    pair = sorted((original, variant.relative_to(images).as_posix()))
    ratings = inputs["base"] / "actual-ratings.csv"
    ratings.write_text(
        f"image_a,image_b,verdict\n{pair[0]},{pair[1]},no_merge\n", encoding="utf-8"
    )
    out, result = _freeze(
        {**inputs, "images": images},
        "rating-repaired",
        "--grouping-version",
        "dhash-v1",
        "--no-merge-ratings",
        str(ratings),
    )
    assert result.returncode == 0, result.stderr
    repaired = json.loads((out / "target" / "grouping.json").read_text())
    assert list(pair) in repaired["no_merge_ratings"]
    assert any(edge[:2] == list(pair) for edge in repaired["removed_edges"])


def test_grouped_freeze_requires_source_roots(inputs: dict[str, Path]) -> None:
    result = _run(
        "freeze",
        "target",
        "--image-root",
        str(inputs["images"]),
        "--rules",
        str(inputs["rules"]),
        "--reviews",
        str(inputs["reviews"]),
        "--output",
        str(inputs["base"] / "no-sources"),
        "--grouping-version",
        "dhash-v1",
    )
    assert result.returncode != 0
    assert "requires both source roots" in result.stderr


def test_grouped_manifest_rejects_missing_grouping_dependency(
    inputs: dict[str, Path],
) -> None:
    from dataclasses import replace

    from jute_disease.data import FreezeBlockedError
    from jute_disease.data.manifest import write_manifest

    out, result = _freeze(inputs, "missing-grouping", "--grouping-version", "dhash-v1")
    assert result.returncode == 0, result.stderr
    manifest = load_manifest(out / "target")
    altered = replace(
        manifest,
        dependencies=tuple(
            (key, value) for key, value in manifest.dependencies if key != "grouping"
        ),
    )
    write_manifest(altered, out / "target")
    with pytest.raises(FreezeBlockedError, match="grouping"):
        load_manifest(out / "target")


def test_freeze_accepts_audit_cli_gate_projection(inputs: dict[str, Path]) -> None:
    audit_dir = inputs["base"] / "rating-fixture"
    fixture = _run("data", "audit-fixture", "--output", str(audit_dir))
    assert fixture.returncode == 0, fixture.stderr
    gate = inputs["base"] / "projected-gate.json"
    rating = _run(
        "data",
        "audit-evaluate",
        str(audit_dir / "agent.json"),
        str(audit_dir / "synthetic_ratings.csv"),
        "--gate-output",
        str(gate),
    )
    assert rating.returncode == 0, rating.stderr
    assert set(json.loads(gate.read_text())) == {
        "kappa_interval",
        "threshold",
        "pass",
        "method",
    }
    out, result = _freeze(
        inputs,
        "projected-gate",
        "--grouping-version",
        "dhash-v1",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "target" / "grouping.json").read_text())
    assert receipt["gate_outcomes"]["agreement"] == "pass"


def test_freeze_records_undefined_kappa_as_failed_gate(inputs: dict[str, Path]) -> None:
    gate = inputs["base"] / "undefined-gate.json"
    gate.write_text(
        json.dumps(
            {
                "kappa_interval": None,
                "threshold": 0.6,
                "pass": False,
                "method": "Cohen kappa undefined",
            }
        ),
        encoding="utf-8",
    )
    out, result = _freeze(
        inputs,
        "undefined-gate",
        "--grouping-version",
        "dhash-v1",
        "--agreement-results",
        str(gate),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "target" / "grouping.json").read_text())
    assert receipt["gate_outcomes"]["agreement"] == "fail"

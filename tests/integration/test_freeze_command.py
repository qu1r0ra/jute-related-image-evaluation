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

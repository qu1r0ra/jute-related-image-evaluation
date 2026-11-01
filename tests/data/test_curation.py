"""Curation rules: exclusions, quarantine and provenance records."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    CurationRules,
    InventoryRecord,
    SyntheticPlan,
    curate,
    load_rules,
)
from jute_disease.data.curation import (
    UNASSIGNED_SOURCE,
    source_of,
)


def test_shipped_rules_load_and_hash_is_stable() -> None:
    rules = load_rules(CURATION_RULES_PATH)
    assert len(rules.supported_labels) == 6
    assert rules.digest() == load_rules(CURATION_RULES_PATH).digest()
    assert rules.quarantine[0].filename_regex == "^IMG_20240502"


def test_quarantine_files_are_excluded_with_a_reason_and_preserved(
    synthetic: SyntheticPlan, inventory: list[InventoryRecord], rules: CurationRules
) -> None:
    result = curate(inventory, rules)
    excluded = {e.path: e for e in result.excluded}
    assert set(excluded) == set(synthetic.quarantine_files)
    assert all(e.kind == "quarantine" for e in excluded.values())
    assert all("Mridha" in e.source for e in excluded.values())
    assert len(result.admitted) + len(result.excluded) == len(inventory)
    assert all((synthetic.root / p).is_file() for p in excluded)


def test_unsupported_and_excluded_labels_are_recorded(
    inventory: list[InventoryRecord], rules: CurationRules
) -> None:
    extra = replace(inventory[0], path="Alien/x.png", label="Alien", sha256="a" * 64)
    flagged = replace(
        inventory[1], path="Flagged/y.png", label="Flagged", sha256="b" * 64
    )
    custom = replace(rules, excluded_labels=(("Flagged", "label audit pending"),))
    result = curate([*inventory, extra, flagged], custom)
    kinds = {e.path: (e.kind, e.reason) for e in result.excluded}
    assert kinds["Alien/x.png"][0] == "unsupported-label"
    assert kinds["Flagged/y.png"] == ("excluded-label", "label audit pending")


def test_dropping_a_whole_class_is_refused(
    inventory: list[InventoryRecord], rules: CurationRules
) -> None:
    without = [r for r in inventory if r.label != "Dieback"]
    with pytest.raises(ValueError, match="Dieback"):
        curate(without, rules)


def test_sources_follow_the_first_matching_rule(rules: CurationRules) -> None:
    def record(path: str, label: str) -> InventoryRecord:
        return InventoryRecord(path, label, None, 1, 1, 1, "0" * 64, "0" * 64, (0,) * 8)

    assert "Jannat" in source_of(record("Healthy/Healthy (3).jpg", "Healthy"), rules)
    assert "Islam" in source_of(record("Healthy/Fresh-001.jpg", "Healthy"), rules)
    assert "Mridha" in source_of(record("Healthy/IMG_20240502_1.jpg", "Healthy"), rules)
    assert source_of(record("Mosaic/other.jpg", "Mosaic"), rules) == UNASSIGNED_SOURCE
    # a rule scoped to a label does not leak into another label
    assert (
        source_of(record("Dieback/Fresh-1.jpg", "Dieback"), rules) == UNASSIGNED_SOURCE
    )


def test_summary_counts_balance(
    inventory: list[InventoryRecord], rules: CurationRules
) -> None:
    summary = curate(inventory, rules).summary()
    assert summary["admitted"] + summary["excluded"] == len(inventory)
    assert sum(summary["admitted_by_label"].values()) == summary["admitted"]


def test_bad_rules_files_raise(tmp_path: Path) -> None:
    def write(**overrides: object) -> Path:
        data = {"version": 1, "supported_labels": ["A"], **overrides}
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(data))
        return path

    with pytest.raises(ValueError, match="version"):
        load_rules(write(version=9))
    with pytest.raises(ValueError, match="at least one"):
        load_rules(write(supported_labels=[]))
    with pytest.raises(ValueError, match="both supported and excluded"):
        load_rules(write(excluded_labels={"A": "why"}))
    with pytest.raises(Exception, match="unterminated|missing|nothing"):
        load_rules(
            write(
                quarantine=[
                    {"label": "A", "filename_regex": "(", "source": "s", "reason": "r"}
                ]
            )
        )

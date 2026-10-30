"""Shared synthetic fixtures for the data-layer tests (no real data is touched)."""

from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    CurationRules,
    InventoryRecord,
    Manifest,
    SyntheticPlan,
    build_inventory,
    build_manifest,
    load_rules,
    make_synthetic_data,
)


@pytest.fixture(scope="session")
def inventory(synthetic: SyntheticPlan) -> list[InventoryRecord]:
    return build_inventory(synthetic.root, workers=2)


@pytest.fixture(scope="session")
def rules() -> CurationRules:
    loaded = load_rules(CURATION_RULES_PATH)
    return CurationRules(
        supported_labels=loaded.supported_labels,
        excluded_labels=loaded.excluded_labels,
        quarantine=loaded.quarantine,
        sources=loaded.sources,
        min_support=2,
    )


@pytest.fixture(scope="session")
def frozen(inventory: list[InventoryRecord], rules: CurationRules) -> Manifest:
    return build_manifest(inventory, rules, seed=42)[0]


def build_tree(root: Path, **kwargs: object) -> SyntheticPlan:
    return make_synthetic_data(root, **kwargs)  # type: ignore[arg-type]

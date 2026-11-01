"""Shared fixtures for tests that use the synthetic source dataset."""

import pytest

from jute_disease.data import SyntheticPlan, make_synthetic_data


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPlan:
    root = tmp_path_factory.mktemp("synthetic") / "tree"
    return make_synthetic_data(root, per_class=20, size=64, quarantine_count=4)

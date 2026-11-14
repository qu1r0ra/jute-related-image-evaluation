"""Shared fixtures for tests that use the synthetic source dataset."""

import os

# Under pytest-xdist each worker would otherwise start one multithreaded Torch/BLAS
# pool per core and the workers oversubscribe the machine. This runs before the
# first Torch import in the worker, on CI and locally alike. A value the caller
# already set wins.
if "PYTEST_XDIST_WORKER" in os.environ:
    for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_threads, "1")

import pytest  # noqa: E402

from jute_disease.data import SyntheticPlan, make_synthetic_data  # noqa: E402


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPlan:
    root = tmp_path_factory.mktemp("synthetic") / "tree"
    return make_synthetic_data(root, per_class=20, size=64, quarantine_count=4)

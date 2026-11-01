# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: jute-disease-detection (3.11.14)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Classical Training
#
# This notebook is a thin front for the ten classical fits: five estimators on raw
# pixels and on crafted features. Each fit is the same `jute-disease train fit`
# child the queue launches, so extraction, train-only scaling and weighting come
# from the shared code.
#
# - **Declared fits:** the estimator and feature grid from the benchmark config.
# - **Shared preparation:** one pass extracts and caches the train and validation
#   features that every fit reads.
# - **Selected fits:** `SELECT` names the fits to run here. The default is a short
#   smoke subset; `just queue-run` runs all ten under the queue's time limits.
#
# The test partition is never opened, and there are no sweeps or calibration.

# %%
import json
from pathlib import Path

import pandas as pd

from jute_disease.config import (
    ARTIFACTS_DIR,
    BENCHMARK_CONFIG,
    FREEZE_DIR,
    ML_FEATURES_DIR,
    PLANTDOC_DIR,
    PLANTVILLAGE_DIR,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data import TARGET_DIR
from jute_disease.training import (
    WORKSPACE_FILE,
    DataInputs,
    Job,
    open_workspace,
    subprocess_launcher,
    verify_stack,
)

SMOKE_DIR = ARTIFACTS_DIR / "smoke"
SELECT = ("lr-raw", "lr-crafted")

workspace = open_workspace(
    benchmark_path=BENCHMARK_CONFIG,
    data=DataInputs(
        image_root=SOURCE_INVENTORY_DIR,
        target_dir=FREEZE_DIR / TARGET_DIR,
        sources={
            "plantvillage": (FREEZE_DIR / "source_plantvillage", PLANTVILLAGE_DIR),
            "plantdoc": (FREEZE_DIR / "source_plantdoc", PLANTDOC_DIR),
        },
    ),
    runs_dir=SMOKE_DIR / "runs",
    cache_dir=ML_FEATURES_DIR,
    require_cuda=False,
)
workspace_file = SMOKE_DIR / WORKSPACE_FILE
workspace.save(workspace_file)
launch = subprocess_launcher(workspace_file)
benchmark = workspace.benchmark
pd.Series(verify_stack(require_cuda=False)).to_frame("measured")

# %% [markdown]
# ## 1. Declared fits

# %%
pd.DataFrame(
    [
        {"fit": f.fit_id, "estimator": f.estimator, "features": f.features}
        for f in benchmark.fits()
        if f.fit_id.startswith("classical/") and f.estimator
    ]
).set_index("fit")


# %% [markdown]
# ## 2. Shared preparation and selected fits
#
# Preparation times the extraction. A fit reads the cache it wrote, so the fit
# seconds below exclude extraction.


# %%
def run(fit_id: str) -> dict:
    """Run one fit as a child, bounded by the classical limit, and return its results."""
    name = fit_id.replace("/", "_")
    attempt = SMOKE_DIR / name
    attempt.mkdir(parents=True, exist_ok=True)
    job = Job(fit_id, attempt, attempt / "result.json")
    limit = benchmark.limits.classical_fit_seconds
    outcome = launch(job, limit, attempt / "child.log")
    if outcome.timed_out:
        raise RuntimeError(f"{fit_id} exceeded the {limit:.0f} s limit")
    if outcome.returncode:
        raise RuntimeError(f"{fit_id} failed:\n{outcome.tail}")
    return json.loads(job.result_path.read_text(encoding="utf-8"))["results"]


prepared = run("classical/prepare")
fits = {name: run(f"classical/{name}") for name in SELECT}
pd.DataFrame(
    {
        name: {
            "train rows": r["rows"]["train"],
            "val rows": r["rows"]["val"],
            "dimensions": r["feature_cache"]["train"]["dimensions"],
            "fit seconds": r["fit_seconds"],
            "val accuracy": r["metrics"]["val_accuracy"],
            "weighting": r["weighting"]["status"],
            "test accessed": r["test_accessed"],
        }
        for name, r in fits.items()
    }
).T

# %% [markdown]
# ## 3. Receipt
#
# The raw 256 px matrix is the capacity concern: one float32 row is 786 KB. The
# estimate below is rows times dimensions times four bytes.

# %%
raw = next((r for n, r in fits.items() if n.endswith("raw")), None)
receipt = {
    "kind": "classical-smoke",
    "note": "Validation only; not a result.",
    "preparation_seconds": prepared["seconds"],
    "fits": {
        name: {
            "fit_seconds": r["fit_seconds"],
            "rows": r["rows"],
            "dimensions": r["feature_cache"]["train"]["dimensions"],
            "test_accessed": r["test_accessed"],
        }
        for name, r in fits.items()
    },
    "raw_train_matrix_gib": None
    if raw is None
    else raw["rows"]["train"] * raw["feature_cache"]["train"]["dimensions"] * 4 / 2**30,
}
assert not any(r["test_accessed"] for r in fits.values())
(SMOKE_DIR / "classical_smoke.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
)
print("wrote", Path(SMOKE_DIR / "classical_smoke.json"))
receipt

# %% [markdown]
# ## 4. Full run
#
# ```bash
# just queue-run
# ```
#
# The queue runs the classical stage after the core CNN fits. Feature preparation
# and each classical fit run under their configured timeouts, and the queue
# resumes by skipping fits whose receipt is complete and matches.

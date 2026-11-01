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
# # CNN and Domain-Transfer Training
#
# This notebook is a thin front for the target CNN systems and their shared source
# stages. Every fit is the same `jute-disease train fit` child process the queue
# launches, so nothing here duplicates training logic.
#
# - **Declared systems:** what the benchmark config asks for.
# - **Bounded smoke:** a few real train and validation batches per system, which
#   checks memory, throughput and the recipe without fitting a model.
# - **Source transfer:** a bounded source stage hands its checkpoint to a bounded
#   target stage, and the receipt shows which backbone tensors moved and that the
#   head was reset.
# - **Full run:** `just queue-run` runs and resumes the whole benchmark.
#
# The test partition is never opened. A bounded run is not a fit and its numbers are
# never reported as results.

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
BOUNDS = (8, 4)  # train and validation batches

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
)
workspace_file = SMOKE_DIR / WORKSPACE_FILE
workspace.save(workspace_file)
launch = subprocess_launcher(workspace_file)
pd.Series(verify_stack()).to_frame("measured")

# %% [markdown]
# ## 1. Declared systems
#
# Each target system runs at three seeds. A chained system starts from the shared
# source stage named in its `chain`, which is trained once and reused.

# %%
benchmark = workspace.benchmark
pd.DataFrame(
    [
        {
            "system": name,
            "model": definition["model"],
            "chain": " -> ".join(definition["chain"]) or "direct",
            "micro_batch": definition["micro_batch"],
            "accumulation": benchmark.recipe.effective_batch
            // definition["micro_batch"],
        }
        for name, definition in benchmark.systems.items()
    ]
).set_index("system")


# %% [markdown]
# ## 2. Bounded smoke
#
# One bounded child per distinct model and batch shape. `peak_cuda_bytes` is the
# memory the real micro-batch needs on this GPU.


# %%
def bounded(fit_id: str, name: str, upstream: Path | None = None) -> dict:
    """Run one bounded child and return its results."""
    attempt = SMOKE_DIR / name
    attempt.mkdir(parents=True, exist_ok=True)
    job = Job(fit_id, attempt, attempt / "result.json", upstream, BOUNDS)
    outcome = launch(job, 1800.0, attempt / "child.log")
    if outcome.returncode or outcome.timed_out:
        raise RuntimeError(f"{fit_id} failed:\n{outcome.tail}")
    return json.loads(job.result_path.read_text(encoding="utf-8"))


smoke = {
    "mobilenet_v2": bounded("cnn/mobilenet_v2_direct/seed-42", "mobilenet_v2"),
    "efficientnet_b5": bounded("cnn/efficientnet_b5_direct/seed-42", "efficientnet_b5"),
}
pd.DataFrame(
    {
        name: {
            "micro_batch": run["results"]["micro_batch"],
            "accumulation": run["results"]["accumulate_grad_batches"],
            "precision": run["results"]["precision"],
            "seconds_per_bounded_epoch": run["results"]["seconds_per_epoch"],
            "peak_cuda_gib": run["results"]["peak_cuda_bytes"] / 2**30,
            "train_batches_per_epoch": run["results"]["epoch_batches"]["train"],
            "test_accessed": run["results"]["test_accessed"],
        }
        for name, run in smoke.items()
    }
).T

# %% [markdown]
# ## 3. Source transfer
#
# The bounded source stage writes a checkpoint. The bounded target stage loads its
# backbone, and the receipt records which tensors were copied, that the head was
# re-initialized, and that no optimizer or trainer state came across.

# %%
source = bounded("source/plantvillage", "source_plantvillage")
checkpoint = Path(source["artifacts"]["checkpoint"])
target = bounded(
    "cnn/mobilenet_v2_plantvillage/seed-42", "mobilenet_v2_plantvillage", checkpoint
)
transfer = target["results"]
assert transfer["head"]["fresh"] and not transfer["head"]["equals_upstream"]
assert transfer["training_state"] == {"resumed": False, "optimizer_loaded": False}
assert not transfer["pretrained"] and not transfer["test_accessed"]
pd.Series(
    {
        "source classes": source["results"]["num_classes"],
        "target classes": transfer["num_classes"],
        "backbone tensors": transfer["backbone"]["total"],
        "backbone trainable": transfer["backbone"]["trainable"],
        "backbone updated": transfer["backbone"]["updated"],
        "transfer": json.dumps(transfer["transfer"], sort_keys=True)[:120],
    }
).to_frame("value")

# %% [markdown]
# ## 4. Receipt
#
# The smoke receipt lives under ignored `artifacts/` and states what it is.

# %%
receipt = {
    "kind": "bounded-smoke",
    "bounds": {"train_batches": BOUNDS[0], "val_batches": BOUNDS[1]},
    "note": "Throughput and memory only; not a fit and not a result.",
    "systems": {
        name: {
            key: run["results"][key]
            for key in ("seconds_per_epoch", "peak_cuda_bytes", "micro_batch")
        }
        for name, run in smoke.items()
    },
    "transfer": {
        "head": transfer["head"],
        "backbone": transfer["backbone"],
        "test_accessed": transfer["test_accessed"],
    },
}
(SMOKE_DIR / "cnn_smoke.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
)
print("wrote", SMOKE_DIR / "cnn_smoke.json")

# %% [markdown]
# ## 5. Full run
#
# ```bash
# just queue-run
# ```
#
# The queue runs fits serially in stage order, each under its configured timeout.
# A failed fit is retried once with the same seed, and three persistent failures in
# a row stop the queue. Running it again resumes: a fit whose receipt is complete
# and matches is skipped.

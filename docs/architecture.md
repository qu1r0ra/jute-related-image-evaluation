# Architecture

This page records cross-cutting behavior that is easy to break during changes. Setup and command reference are in [tooling.md](tooling.md), preprocessing is specified in [ADR 0002](adr/0002-measure-background-confound-with-fixed-transforms.md), and historical study names are in [legacy-names.md](legacy-names.md).

## Frozen data

An explicit target freeze inventories image bytes and decoded pixels, applies the tracked curation and identity rules, and writes grouped partitions with a receipt binding the manifest to its inputs. Existing freeze outputs are not silently replaced. Cache identity includes the preprocessing specification, so changing it invalidates matching image caches.

## Training queue

`configs/benchmark.json` declares the fits, shared recipe, per-fit seeds, and step timeouts. The queue runs one child process per fit and can resume when invoked again. The official CUDA profile uses `bf16-mixed`, TF32 matmul and cuDNN, `cudnn.benchmark` off, and eight DataLoader workers. CPU smoke work uses zero workers. `torch.compile` stays off, and the CLI entry point keeps its `__main__` guard so forkserver workers can safely re-import it.

- A fit is reusable only when its versioned receipt parses, its identity matches the frozen train/validation data, seed, recipe, preprocessing, assignment table, source-checkpoint ancestry, config, code commit, environment lock and run-affecting options, and every recorded artifact still matches its recorded bytes. Backend settings and installed version observations are recorded in receipts outside the identity. Each attempt keeps its own provenance and outcome. Other attempts remain disclosed partial runs and are never reused.
- A clean checkout records its commit and may be marked `official: true`. A dirty checkout is marked `official: false`. Confirmatory analysis checks every planned fit for a completed compatible receipt, a clean official provenance record, a validated rotating assignment table, and a complete attempt history; any missing or non-completed fit blocks analysis.
- Fixed seeds are used without forcing deterministic algorithms. Receipts record deterministic, backend, library, CUDA, driver, and GPU observations. Resumed training is not bit-identical to uninterrupted training.
- Infrastructure faults, including killed processes, out-of-memory and I/O errors, and timeouts retry once with the same seed. A non-finite loss is recorded as divergence and is not retried. Persistent failures omit dependent fits; three consecutive persistent failures abort the queue. A new invocation starts a fresh failure count.
- CNN training resumes from `last.ckpt` only when `owner.json` identifies the same run and receipt format, training is unfinished, and the checkpoint matches the bytes recorded in `last.json`. The checkpoint and progress hash are saved after the early-stopping callback updates. Otherwise, old checkpoints move under `superseded/` and training starts fresh.
- Training code never names the test partition. `tests/training/test_held_out_guard.py` checks this boundary statically and at the command line.

## Decided, not yet implemented

These items are decided but are not current behavior. Each bullet states the target. When its issue lands, move the rule into the section above it belongs to and delete the bullet.

- Benchmark config shape ([ADR 0003](adr/0003-ten-split-draws-with-rotating-source-checkpoints.md); [issue 137](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/137)): ten split draws, five seeds, and three rotating source checkpoints per stage. Target fit (seed s, split k) uses checkpoint (s + k) mod 3.
- Test separation ([issue 139](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/139)): the freeze writes the test manifest where only the evaluator reads it, and the trainer never receives it. The evaluator runs only with an immutable selection record committed with its hash. The static test-name guard stays as a second check.
- Artifact transfer ([issue 107](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/107), owner-confirmed 2026-11-02): keep artifacts under `artifacts/` inside the checkout on both machines, gitignored. Never run `git clean -x` on the lab. A batch is everything finished since the last accepted copy-back. Copy it back after every completed queue invocation, and accept it only after `scripts/verify_transfer.py` matches every hash. Delete nothing on the lab until its PC copy is verified.

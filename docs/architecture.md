# Architecture

This page records cross-cutting behavior that is easy to break during changes. Setup and command reference are in [tooling.md](tooling.md), preprocessing is specified in [ADR 0002](adr/0002-measure-background-confound-with-fixed-transforms.md), and historical study names are in [legacy-names.md](legacy-names.md).

## Frozen data

An explicit target freeze inventories image bytes and decoded pixels, applies the tracked curation and identity rules, and writes grouped partitions with a receipt binding the manifest to its inputs. Existing freeze outputs are not silently replaced. Cache identity includes the preprocessing specification, so changing it invalidates matching image caches.

## Training queue

`configs/benchmark.json` declares the fits, shared recipe, per-fit seeds, and step timeouts. The queue runs one child process per fit and can resume when invoked again.

- A fit is reusable only when its receipt parses, its identity matches the frozen inputs, recipe, and upstream checkpoint bytes, and every recorded artifact still matches its recorded bytes. Other attempts remain as disclosed partial runs and are never reused.
- Failed or timed-out fits retry once with the same seed. Persistent failures omit dependent fits; three persistent failures in a row abort the queue. A new invocation starts a fresh failure count.
- CNN training resumes from `last.ckpt` only when `owner.json` identifies the same run, training is unfinished, and the checkpoint matches the bytes recorded in `last.json`. Otherwise, the old checkpoints move under `superseded/` and training starts fresh. A resumed fit revalidates its interrupted epoch, so early-stopping patience may count that epoch twice; resumed training is not bit-identical to uninterrupted training.
- Training code never names the test partition. `tests/training/test_held_out_guard.py` checks this boundary statically and at the command line.

## Decided, not yet implemented

These items are decided but are not current behavior. Each bullet states the target. When its issue lands, move the rule into the section above it belongs to and delete the bullet.

- Benchmark config shape ([ADR 0003](adr/0003-ten-split-draws-with-rotating-source-checkpoints.md); [issue 136](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/136), [issue 137](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/137)): ten split draws, five seeds, and three rotating source checkpoints per stage. Target fit (seed s, split k) uses checkpoint (s + k) mod 3. Receipts record the source-checkpoint ancestry chain and the seed/split-to-checkpoint table.
- Resume ([issue 136](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/136)): save `last.ckpt` after the early-stopping state updates, so a resume at an epoch boundary does not count the interrupted epoch twice.
- Retry and missing fits ([issue 136](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/136)): retry once only for infrastructure faults (killed process, out-of-memory, I/O error, timeout). Record a divergence (NaN or infinite loss) as failed without retrying. Block confirmatory analysis on any planned fit that has not completed, until it is rerun or the owner records an approved deviation. Drop no fit silently.
- Test separation ([issue 139](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/139)): the freeze writes the test manifest where only the evaluator reads it, and the trainer never receives it. The evaluator runs only with an immutable selection record committed with its hash. The static test-name guard stays as a second check.
- Provenance ([issue 136](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/136)): run official fits from a clean working tree at a recorded git commit, stored in every receipt. Stamp dirty-tree runs `official: false` and refuse them for confirmatory analysis. Run identity covers data, config, seed, code commit and upstream checkpoint bytes. Record environment versions (driver, CUDA, package builds) in the receipt, outside identity.
- Execution contract ([issue 107](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/107), owner-confirmed 2026-11-02): run the recipe at `bf16-mixed` precision on CUDA with TF32 matmuls allowed. Keep `cudnn.benchmark` off and do not force deterministic algorithms. Record torch, CUDA and driver versions, GPU name and backend flags in the receipt. Use `num_workers` 8 on the lab and 0 for PC smoke runs, and include the worker count in run identity. Python 3.14 starts workers by forkserver, which re-imports the entry module, so keep loader construction under a `__main__` guard. Keep `torch.compile` off. Keep artifacts under `artifacts/` inside the checkout on both machines, gitignored. Never run `git clean -x` on the lab. A batch is everything finished since the last accepted copy-back. Copy it back after every completed queue invocation, and accept it only after `scripts/verify_transfer.py` matches every hash. Delete nothing on the lab until its PC copy is verified.

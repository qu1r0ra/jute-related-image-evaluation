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

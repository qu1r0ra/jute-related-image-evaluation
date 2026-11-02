# Architecture

This page records cross-cutting behavior that is easy to break during changes. Setup and command reference are in [tooling.md](tooling.md), preprocessing is specified in [ADR 0002](adr/0002-measure-background-confound-with-fixed-transforms.md), and historical study names are in [legacy-names.md](legacy-names.md).

## Frozen data

An explicit target freeze inventories image bytes and decoded pixels, applies the tracked curation and identity rules, and writes grouped partitions with a receipt binding the manifest to its inputs. Existing freeze outputs are not silently replaced. Cache identity includes the preprocessing specification, so changing it invalidates matching image caches.

## Test evaluation

The freeze writes a source-bound test-only manifest under `target/evaluator/`. The evaluator is its only reader; manifest readers, the cache builder and the data module reject the held-out test partition, while the static training guard protects training code. `evaluate freeze-matrix` requires recorded #114 validation outcomes and writes an immutable, hash-bound selection containing planned fit and analysis IDs, every validation-selected hyperparameter, the letterbox and mask-audit outcomes, border settings, post-pilot budget, pipeline-code tree hash and environment-lock hash. The owner commits the exact selection file only after explicit approval and before a production test read. A later selection is a new record. A selection created after a test read is post-hoc and its results must be reported as such. Before opening any test image, `evaluate score-test` durably appends a started event to the target's test-read journal; it appends a completion event after saving the report, so an interrupted or failed read remains visible.

`evaluate score-test` verifies the selection and completed run record before loading the frozen test partition or opening test images. Each artifact index row must match its frozen training view, model configuration and preprocessing; the run record binds each model to its bytes and a complete #107 receipt from a clean, official run. Classical and CNN fit specifications include the exact #114 selected parameters or learning rate, and the receipt must report the same choice used for fitting. It also binds the Grad-CAM test-pass analysis receipt without implementing a Grad-CAM score. Production evaluation requires an existing local SAM 2 checkpoint and uses the color-threshold, largest-component mask as its prompt. The evaluator runs those models on the frozen views itself and reuses prepared test pixels and masks for each model route. It scores background-only views for the primary split, every pool and policy, classical feature/model routes and the MobileNetV2 ImageNet route over five seeds. The border probe evaluates the nine declared classical configurations per estimator on frame-only views, plus the MobileNetV2 and pad-fraction and aspect-ratio baselines, over the primary split, every pool and policy, and three seeds. Every final checkpoint is scored on a deterministic class-and-scene stratified leaf-only sample up to 2,000 images. Class-by-scene cells include support and accuracy with Wilson intervals; scene macro-F1 appears only when every class has at least 30 test images.

The evaluator follows the frozen #114 letterbox or resize-and-center-crop decision for every model view. The border probe uses the frozen width (32 px starting value) inside resized image content and masks letterbox padding. It reports seed-then-family hierarchical accuracy and macro-F1 intervals with family multiplicities paired across systems, a family-level 1,000-permutation null, majority-class accuracy, pad-fraction and aspect-ratio baselines, and each frame's leaf-pixel share. The symptom-survival headline uses the frozen #114 validation-selected leaf-share cutoff (5% starting value, reported with the results), requires at least 50 retained test images per class and a passed mask audit, and reports class and scene support. If either gate fails, the report withholds the symptom-survival scores and intervals and states why; only the unrestricted result stands. Border-probe costs are counted in the post-pilot budget and the #141 A100 GPU-hours report; because the budget is set after the pilot, there is no pre-pilot estimate. It states that the 32 px strip is more sensitive to mask-edge error than whole-leaf mask quality. No Grad-CAM leaf score is implemented.

## Training queue

`configs/benchmark.json` declares the fits, shared recipe, per-fit seeds, and step timeouts. The queue runs one child process per fit and can resume when invoked again. The official CUDA profile uses `bf16-mixed`, TF32 matmul and cuDNN, `cudnn.benchmark` off, and eight DataLoader workers. CPU smoke work uses zero workers. `torch.compile` stays off, and the CLI entry point keeps its `__main__` guard so forkserver workers can safely re-import it.

- A fit is reusable only when its versioned receipt parses, its identity matches the frozen train/validation data, seed, recipe, preprocessing, assignment table, source-checkpoint ancestry, config, code commit, environment lock and run-affecting options, and every recorded artifact still matches its recorded bytes. Backend settings and installed version observations are recorded in receipts outside the identity. Each attempt keeps its own provenance and outcome. Other attempts remain disclosed partial runs and are never reused.
- A clean checkout records its commit and may be marked `official: true`. A dirty checkout is marked `official: false`. Confirmatory analysis checks every planned fit for a completed compatible receipt, a clean official provenance record, a validated rotating assignment table, and a complete attempt history; any missing or non-completed fit blocks analysis.
- Fixed seeds are used without forcing deterministic algorithms. Receipts record deterministic, backend, library, CUDA, driver, and GPU observations. Resumed training is not bit-identical to uninterrupted training.
- Infrastructure faults, including killed processes, out-of-memory and I/O errors, and timeouts retry once with the same seed. A non-finite loss is recorded as divergence and is not retried. Persistent failures omit dependent fits; three consecutive persistent failures abort the queue. A new invocation starts a fresh failure count.
- CNN training resumes from `last.ckpt` only when `owner.json` identifies the same run and receipt format, training is unfinished, and the checkpoint matches the bytes recorded in `last.json`. The checkpoint and progress hash are saved after the early-stopping callback updates. Otherwise, old checkpoints move under `superseded/` and training starts fresh.
- The matrix is three pools, two views, three policies and ten split draws (primary draw 0), with five seeds per draw. Each source stage trains three seeded checkpoints; PlantDoc checkpoint j starts from PlantVillage checkpoint j. A target fit (seed index s, split k) resolves its source checkpoint only through the assignment table (checkpoint (s + k) mod 3, per [ADR 0003](adr/0003-ten-split-draws-with-rotating-source-checkpoints.md)); the table is generated once and stored with the policy manifests, and its hash enters every fit identity. Overlap exclusions live in the frozen source manifests, so every chain trains on the same excluded set.
- Tuning runs first and is frozen. Each classical estimator searches nine configurations per pool on the family-disjoint tuning reserve; each CNN route and pool tries three learning rates (0.1, 1 and 10 times the base rate) on that reserve, and the source stage selects its rate once on source validation. A final fit reads the selection from the tuning receipt, the selection enters its identity, and a failed tuning fit omits the finals that depend on it. A CNN tuning fit gets one timeout per candidate.
- CNN fits use a 50-epoch ceiling, early stopping and validation-only checkpoint selection. Classical classifiers fit inside each training pipeline (features, then fold-fitted z-scores and block weights, then the classifier): deterministic estimators run once per split and random forest runs with five seeds. EfficientNet-B5 (ImageNet route, primary split, five seeds, both views, every pool and policy) is declared as a secondary analysis.
- Handcrafted features are an HSV 8x8x8 histogram, riu2 LBP at (8,1) and (16,2), and a four-angle GLCM at distances 1 and 2 on 32 gray levels. Letterbox padding is excluded from every descriptor, and each block is weighted by 1/sqrt(block size). The single transform choice lives in `ImagePipeline`, so a different #114 outcome changes only that choice.
- Training code never names the test partition. `tests/training/test_held_out_guard.py` checks this boundary statically and at the command line.

## Evaluation

`jute-disease evaluate` routes every test read through `evaluation/evaluator.py` or `evaluation/gated.py`; training code never names the test partition. The target freeze stores test assignments only in the evaluator manifest under `target/evaluator/`, outside the public target manifest. Manifest readers, cache builders and the data module reject test access.

- The matrix-freeze record (`evaluate freeze-record`) is written only from recorded #114 validation outcomes: a validation-partition letterbox comparison of at least two candidates that includes the chosen one, each frozen selection with its chosen value, and the mask-audit pass/fail result. The values frozen in the record must match those recorded outcomes. It lists every planned fit (the declared matrix and the probe fits), the analyses, selections, border-probe start and final settings, the budget, and hashes of the pipeline tree (src, configs, uv.lock) and the environment lock. Its hash is the SHA-256 of the canonical body. A later freeze lists prior test reads as invalidated and is flagged `post_hoc`. Every test read is logged at `artifacts/evaluation/test-reads.jsonl` under the project root; an optional custom log receives an additional copy. This includes the label read used by `evaluate leaf-sample`.
- A run record (`evaluate run-record`) binds each fit to the freeze record hash and to the SHA-256 of its prediction file. A fit not listed in the freeze record cannot enter it.
- `evaluate test` refuses the read unless the freeze record hash matches, the pipeline and environment trees are unchanged, a run record is bound to that freeze record, and every run is listed in both. `evaluate leaf-sample` needs only the freeze record, so the leaf-only sample is known before leaf-only checkpoints are scored.
- Background-only and border probes use final classical models and the MobileNetV2 ImageNet route on the primary split, every pool and policy, with five matrix seeds (three for the border probe). Tuning-reserve fits are excluded from test probe plans. `evaluate probe-plan` lists the fits, including pad-cue baseline fits. A probe reports a detection only when the lower CI bound of macro-F1 exceeds the 95th percentile of a family-level label-permutation null (1,000 permutations); otherwise it reports no detectable information at that setting, never absence. Majority-class accuracy is reported alongside.
- The border probe uses a 32 px frame of the resized content. Symptom survival: the headline is the result on images with at most 5% frame leaf share. It is withheld when any class has fewer than 50 retained test images or the mask audit failed, and the unrestricted result then stands alone with a statement that the audit rates whole-leaf masks, not strip-level leaf share.
- Leaf-only evaluation covers every final checkpoint on a stratified sample of about 2,000 test images (largest-remainder class quotas, ranked by seeded SHA-256), compared with the full-image result on the same sample. The evaluator refuses to report a full-checkpoint set with incomplete leaf-only coverage.
- Class-by-scene cells report counts and accuracy with Wilson intervals. Scene macro-F1 is calculated over the classes present in each scene, and reported only where each present class has at least 30 test images. Scenes and per-image frame leaf shares come from a features file.
- One `evaluate test` call evaluates one frozen target manifest; looping over pools and policies belongs to the caller. The Grad-CAM leaf score is not part of this evaluator.

## Artifact recovery

Runtime artifacts stay under the gitignored `artifacts/` directory inside the checkout on both machines. Never run `git clean -x` on the lab. `just queue-unattended` detaches the queue behind the lab lock and timeout wrapper, checks ntfy before launch, and sends start, hourly heartbeat, success or failure notifications using `NTFY_TOPIC` and `NTFY_TOKEN` from the environment. Each invocation writes a SHA-256 inventory of the full `artifacts/runs/` store to `artifacts/transfer-manifests/`, including when the queue exits with an error. A batch is all work since the last accepted copy-back; the full-store manifest includes earlier receipts and checkpoints needed for recovery.

Closing the browser terminal does not stop a detached process. The 72-hour JupyterLab cull or a host restart can stop it without a final failure notification; a missed heartbeat signals interruption. After relaunch, run `just queue-unattended` again to resume from compatible receipts. After every completed invocation, copy `artifacts/runs/` and its new transfer manifest to the PC, then run `uv run --locked python scripts/verify_transfer.py verify artifacts/runs artifacts/transfer-manifests/queue-<batch-id>.json` there. Accept the batch only when every hash matches. Delete nothing on the lab until the PC copy is verified.
## Manuscript exports

`jute-disease manuscript export` reads a receipt-verified freeze and a JSON
configuration snapshot. Each export has an ID and writes support and compute
tables, a figure, `macros.json`, and `macros.tex`. Every exported number,
including A100 GPU-hours by training stage and the classical tuning count per
family, is keyed by export ID in both macro files. Compute totals use verified
training receipts and their recorded attempt times. The classical tuning
budget is nine configurations per family. No latency or edge-device benchmarks
are reported. `export.json` records the command, embedded configuration and
its SHA-256, plus the manifest and receipt paths and hashes.
`jute-disease manuscript check EXPORT_DIR` recomputes the counts from the
authoritative frozen manifest and rejects changed macro values, source
artifacts, table values, training receipt totals, or generated files.

The export also writes `rebuild-kit.json` with the source, URL, version, path,
SHA-256, family ID, split and label for every retained image. Run
`uv run --locked python scripts/rebuild_splits.py --kit-manifest EXPORT_DIR/rebuild-kit.json --image-root IMAGE_ROOT --output rebuilt-splits.json`
to verify train and validation source bytes and reconstruct family-level split
assignments. Test entries contribute their recorded IDs and labels to the split
rebuild, but the source image files are not opened.
`jute-disease manuscript fixture --output DIR` exercises synthetic image
creation, target freeze, table/figure and macro export, split reconstruction,
and the mechanical macro check without real-data training or scoring. Its
compute totals stay pending because the fixture does not train a model.

An optional `--evaluation-metrics PATH` imports an existing aggregate CSV from
the validation metrics workflow. It does not run evaluation or read partition
images. Each row must reproduce from the latest validation metrics file listed
in a completed, verified training receipt; the receipt's configuration digest
and the metrics artifact digest are recorded. Every numeric validation or
training metric cell is emitted as an export-ID macro, and the original CSV is
copied into the export. The check revalidates the receipts, configuration and
metrics artifacts and recomputes the aggregate. Unsupported columns, including
held-out metrics, are rejected.

Already-produced evaluation JSON can be imported with
`--evaluation-artifacts BUNDLE_DIR`. The bundle contains `receipt.json`,
`metrics.json`, `predictions.json` and the run receipts named by the evaluation
receipt. It must describe validation (`val`) artifacts. The receipt binds each
file to the export configuration SHA-256 and frozen manifest digest; run
receipts must agree on their identities and partition. Every finite numeric
leaf beneath the metrics object's `metrics` field becomes an export-ID macro.
This import checks hashes and receipt lineage without scoring or opening
partition images. Existing bootstrap summaries can
be imported with `--statistics-artifacts BUNDLE_DIR`; its receipt and report
must match the same configuration, manifest and evaluation-receipt hash. Every
numeric leaf beneath the `statistics` field becomes a macro. Both bundle
receipts use schema version 1 and are rechecked by `manuscript check`.

`claim-audit.json` starts as a pending review record linked to the accepted
[#108 may-report and may-not-claim lists](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108#issuecomment-6545452441)
and the [#108 independent audit decision](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108#issuecomment-6572818142).
A reviewer in a fresh context from a different provider than the writer records
pass, fail or unsupported for every macro and flagged claim, plus any unflagged
numeric statements. The audit record must explicitly report when the manuscript
has no flagged claims or no unflagged numeric statements. Any fail or unsupported
row blocks submission; changed manuscript claims require a re-audit. The owner
records a random 10% spot check and signs after the reviewer timestamp. Audit
timestamps use ISO 8601 with a timezone. The synthetic fixture does not perform
or imply either review.
`jute-disease manuscript audit-check EXPORT_DIR --manuscript PATH` is the
separate submission gate: it binds the review to the current manuscript SHA-256
and blocks until each macro and claim row passes, the numeric scan and claim
inventory are complete, the owner records the random spot check, and the owner
signs off. The fixture deliberately leaves this gate pending.

External figure images are admitted only with CC0, CC BY or CC BY-SA licenses
and nonempty source, URL and attribution fields in `figure-sources.json`.
Generated plots from project artifacts need no external image attribution.


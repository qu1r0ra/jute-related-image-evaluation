# Tooling

See [Continuous integration](ci.md) for pull-request and main-branch checks and
the response to a failing main run.

## Supported environment

- Python 3.14.10 is selected by `.python-version`; `pyproject.toml` supports Python 3.14 only.
- `uv.lock` is the dependency source of truth. Use `uv sync --locked --all-groups` for development.
- PyTorch and TorchVision are pinned as one version pair. uv selects the official [PyTorch CUDA 12.6 wheels on Linux and CUDA 13.0 wheels on Windows](https://pytorch.org/get-started/previous-versions/). CI validates Linux; run Windows-specific checks locally when changing Windows dependency selection.
- Lightning remains the deep-learning training framework. Active run evidence is written locally as Lightning CSV logs and JSON evaluation records with metrics, class order, confusion matrix, and relevant settings. Historical W&B configs, notebooks, run records and artifacts live under the `submitted-study` tag and in historical folders.

## Updating dependencies

Update dependencies deliberately, regenerate `uv.lock`, and review the full platform-specific diff. Keep PyTorch and TorchVision on the same release pair. Linux CI runs lock resolution, the dependency audit, import checks, the full suite on code PRs and on `main`, and focused tests on docs-only PRs. Run Windows resolution and the CUDA smoke check locally only when a change affects Windows-specific dependency selection. Recheck the official PyTorch wheel indexes when changing the pair. Security advisories for the locked requirements should be handled as soon as practical; review general dependency updates at least quarterly.

`just audit` runs `scripts/audit_dependencies.py`. It exports the exact locked requirements, scans all non-PyTorch packages through pip-audit, then scans the pinned PyTorch and TorchVision base release numbers. The CUDA wheel build suffixes (`+cu126` and `+cu130`) are platform-specific and are not published as PyPI releases, so the advisory lookup uses their shared upstream release numbers. The check does not install or change packages.

## Preprocessing and freezes

Preprocessing identity revision 2 covers the shared 256-pixel letterbox, bilinear resampling, rounded uint8 ImageNet-mean padding, tensor normalization and cache encoding. The classical route uses the same geometry and pad values without tensor normalization. Freeze manifest version 2 records each image's letterbox pad fraction and summarizes it by source and class. Version 1 manifests remain readable, but new freezes use version 2. A preprocessing identity change invalidates matching image caches; it does not rewrite or delete existing freeze artifacts. Re-freeze explicitly when the source, curation policy or split must change.

## Benchmark queue

`jute-disease queue plan` and `queue status` list every declared fit and need no policy directory. `queue run` over the full matrix requires `--policy-dir`, the directory of receipted policy manifests (with `assignments.json`), and reads each pool's images from `--pool-a-root`, `--pool-b-root` and `--pool-c-root` (falling back to `--image-root`). The same options reach `just queue-run` and `just queue-unattended` (an optional leading timeout in seconds, then the options). Tuning fits (`classical/tune/...`, `cnn/tune/...`, `source/tune/...`) run before the finals that reuse their selection and write a `tuning.json` that records `test_accessed: false`.

`queue run --probes` replaces the plan with the probe stage only (the tuning fits the probes reuse, then the pad-cue, border and background-only fits), so it does not repeat the main plan. Run the plain `queue run` and then `queue run --probes` before a confirmatory `evaluate test`: the test read refuses when any scored fit of the freeze record has no lineage-bound run, and it names the missing fit IDs.

### Manuscript export fixture

Run `just manuscript-fixture artifacts/manuscript-fixture` to exercise a
synthetic freeze, support table and figure export, export-ID keyed TeX/JSON
macros, rebuild kit, split reconstruction and macro-to-artifact check. The
fixture makes no real-data training or scoring claims. For a real frozen
manifest, run `jute-disease manuscript export --freeze-dir artifacts/freeze/target
--config CONFIG.json --output EXPORT_DIR --source-metadata SOURCE_METADATA.json
--training-runs artifacts/runs`,
then `jute-disease manuscript check EXPORT_DIR`. Use
`just rebuild-splits EXPORT_DIR/rebuild-kit.json IMAGE_ROOT rebuilt-splits.json`
to validate train and validation source bytes and reproduce the split
assignment. It records the test entries from the kit but does not open test
image files. The export's
`claim-audit.json` is a pending record: the independent claim review, owner
spot check and owner sign-off remain human gates. Complete the final gate with
`jute-disease manuscript audit-check EXPORT_DIR --manuscript PATH`; it checks
the manuscript SHA-256 recorded in the review. Compute tables report measured
A100 GPU-hours per stage from verified training receipts and nine classical
tuning configurations per family. They include no latency or edge-device
benchmarks. The synthetic fixture leaves its compute report pending because it
does not run training.

To include an existing validation summary, pass
`--evaluation-metrics artifacts/experiment_summary.csv` to the export command.
It reads the aggregate CSV only; it does not run evaluation or access partition
images. Each row must reproduce from metrics covered by a completed, verified
training receipt, including its configuration digest. Numeric validation and
training cells become export-ID macros, and `manuscript check` revalidates the
receipts and recomputes the aggregate before accepting the copied source CSV.

To import an already-produced evaluation and bootstrap summary, pass
`--evaluation-artifacts BUNDLE_DIR --statistics-artifacts STATS_DIR`. The
evaluation bundle has `receipt.json`, `metrics.json`, `predictions.json` and the
run receipt files referenced by `receipt.json`. Its version 1 receipt records
the configuration SHA-256, frozen manifest digest, partition, and hashes for
each artifact and run receipt. The metrics and predictions JSON repeat the
configuration and manifest digests, and predictions list the same run
identities as the receipts. Only `val` evaluation receipts are accepted; test
reports and predictions are outside this export path. The statistics bundle
has `receipt.json` and a
report JSON containing `config_digest`, `manifest_digest`,
`evaluation_receipt_sha256` and a `statistics` object. Its receipt records the
report hash and must bind to the evaluation receipt hash. Every finite numeric
leaf under `metrics` or `statistics` is emitted as an export-ID macro;
`manuscript check` verifies the receipts, lineage and copied files again.
These options import existing results only; they do not score data or open
partition images.

### Gated image diagnostics

`jute-disease evaluate diagnostics` reads one authorized test manifest after
checking the freeze and run records. Repeat it for each primary-split target
manifest and policy group, passing all five seed fits with repeated `--fit-id`
options. `--assignment-table` must be the table whose digest is recorded in
every fit receipt. The inputs are outputs of two logged commands, run first
under the same freeze record:

- `jute-disease evaluate sanity-checks` computes parameter randomization, plus a
  head-refit label-randomization variant (not Adebayo data-randomization
  retraining), on the validation-median primary-cell checkpoint. The pass rule is
  fixed in code and its thresholds (60 sampled images, median absolute Spearman
  correlation below 0.5, at least half defined) are prespecified heuristics, not a
  significance test; they apply to the sampled images only. The pass is a
  checkpoint-level heuristic precondition for the leaf score, which is computed on
  a separate, larger set of images: it does not certify those images
  individually. The record carries, per stage, the count of constant randomized
  maps and of non-finite randomized maps (both leave a correlation undefined or
  scored as changed, as stated above) and the final-stage per-image
  correlations, so a reader can see the thresholds were applied to defined values
  only. The label variant is descriptive, reported beside a true-label control
  refit and the share of labels its permutation changed, and does not gate the
  leaf score.
- `jute-disease evaluate scene-tags` computes a scene tag and frame leaf share
  for every test image with the thresholds and backend of the frozen mask audit.

Pass their files as `--sanity-checks` and `--scene-tags`. Both are refused unless
the ledger holds the completion row of the command that wrote those exact bytes,
so a hand-written file is rejected. The image-reading entry point repeats this
check itself rather than trusting the caller, and appends its own
`image_diagnostics` intent row before it opens any test image. The policy
predictions come from the run
record, which must hold the runs of all three policies for the frozen cell; pass
every fit with `--fit-id`. The freeze record, run record, lockfile and project
configuration are hashed into the export. The command writes `diagnostics.json`,
aligned coordinate CSV files and prespecified Grad-CAM figures into a staging
directory and renames it into place with `output-manifest.json`, which lists a
SHA-256 for every file. The command itself does not train models or
benchmark hardware; reported compute totals come from verified training
receipts.

Classical estimators and their nine-configuration grids: k-NN with k in {1, 3, 5, 7, 11, 15, 21, 31, 51}; L2 logistic regression with C from 1e-4 to 1e4 in nine log-spaced values; RBF-SVC with C in {1, 10, 100} and gamma in {0.1, 1, 10} times scale; random forest with 500 trees, max_features in {sqrt, 0.1, 0.3} and min_samples_leaf in {1, 3, 10}. Gaussian naive Bayes searches `var_smoothing` over 1e-12 to 1e-4 in nine decades. The crafted feature vector has 548 values; z-scores are fitted on the training fold only.

## Lab helpers

The scripts in `scripts/` are Linux-only command-line helpers for lab runs:

- `notify_run.py` sends start, hourly heartbeat, success and failure messages to an `ntfy` topic. `NTFY_TOPIC` and `NTFY_TOKEN` come from the environment. `NTFY_HEARTBEAT_SEC` sets the heartbeat interval in seconds (default `3600`; `0` disables periodic heartbeats). The token is sent only in the HTTPS Authorization header, never in process arguments; transport errors do not print credentials. A cull or other abrupt kill can prevent a final failure message; a missed heartbeat signals interruption.
- `lab_run.sh` holds a `flock` lock until every process in the run's process group has exited, applies a timeout to the process group, and can detach a run while recording its PID and log path. On SIGTERM or SIGINT it signals the group, waits out a grace period, kills what remains, and only then releases the lock. `queue run` turns SIGTERM into an exit so each fit child, which runs in its own session, is killed too. Example: `scripts/lab_run.sh /path/to/run.lock 7200 uv run jute-disease queue run`.
- `queue_unattended.sh` (through `just queue-unattended`) checks notification connectivity, launches a detached locked queue, and records a transfer manifest after each invocation. Re-run it after a JupyterLab cull or host restart; the queue reuses only fits whose receipts still match.
- `queue_batch.py` runs one queue command and records the full `artifacts/runs/` tree after it exits, including a failed invocation, so completed receipts and partial-run evidence can be copied together.
- `pilot_fit.py` runs one validation-only MobileNetV2 fit through `fit_cnn` on the frozen target manifest, for the #114 timing, VRAM and forced-kill resume pilots. `--resume-from CHECKPOINT` continues an interrupted fit; `--image-size`, `--max-epochs`, `--seed` and `--summary` set the run. It never opens the test partition.
- `verify_transfer.py record ROOT MANIFEST` writes a SHA-256 inventory outside the batch. Run `verify_transfer.py verify ROOT MANIFEST` after copy-back; missing, changed, unexpected, unsafe, hard-linked, non-regular and colliding paths reject the batch.

On the lab, run `just queue-unattended --policy-dir <dir> --pool-a-root <dir> --pool-b-root <dir> --pool-c-root <dir>` from the checkout. The ntfy topic and token must already be supplied through the host's environment secret mechanism. After the detached invocation ends, copy `artifacts/runs/` and the newly written manifest from `artifacts/transfer-manifests/` to the PC. Verify the PC copy, for example:

```powershell
uv run --locked python scripts/verify_transfer.py verify artifacts/runs artifacts/transfer-manifests/queue-<batch-id>.json
```

Repeat the copy and verification after every queue invocation. Keep the lab copy until verification succeeds; the manifest snapshots the full run store so earlier receipts needed for resume remain included.

These helpers do not run real lab sessions or experiments as part of tooling CI. The stand-in pilots are recorded in [issue 114](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/114); the pilots on the real reserve are tracked in [issue 193](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/193).

### Grouped target freezes

`--grouping-version` selects the Hamming threshold: `dhash-v1` merges at distance 4, `dhash-v2` at 6, and `dhash-v3` at 7. `dhash-v3` exists only for the one Pool C re-freeze described under the gate record below. `dhash-v2` applies the one prespecified threshold change: the synthetic 5% centered-crop recall was 0.83 at threshold 4 (bar 0.90) and is 0.93 at 6, while the unrelated false-merge rate stays at 0.0002. Threshold 6 is the smallest value that clears the bar; family counts per pool and class confirmed it (larger thresholds chain families and exclude more cross-label images in pool A). The receipt records `gate_outcomes.threshold_change: applied` and a `threshold_change_record`. Source screening keeps its distance-4 rule. Every grouping audit tool (`data audit-sample edges` and `recall`, `audit-fixture`) requires `--grouping-version` and derives the Hamming threshold from it; none has a threshold default or a `--threshold` option. The merged band is distance 0 to the threshold and the near-miss band runs from the threshold plus one to 16. `audit-cohort` reads the version from the sample file.

`freeze target --grouping-version dhash-v1` (or `dhash-v2`) requires `--plantvillage-root` and
`--plantdoc-root`, each pointing to an image inventory in the supported
`<label>/<file>` or `<train|val|test>/<label>/<file>` layout. The freeze screens
all inventoried target images against both supplied source inventories, records
source inventory digests and all matching source paths, target paths and match
methods in `target/grouping.json`, and does not silently discard matches. Verify
that the supplied roots are the intended source packages. Source intake is a
separate operation and continues to apply its own quarantine rules.

Pool freezes run once per pool. `scripts/stage_pools.py --archives data/source-archives --output data/pools` copies each pool's original-release files, unchanged, to `data/pools/<A|B|C>/<label>/<file>`. Staging validates each archive first and rejects an empty result, an empty image, a pool missing a label, and paths that differ only by case. It records the archive SHA-256 in `stage-receipt.json` and refuses to write into a pool directory that already has files. Pass that folder as `--image-root` with `--rules configs/data/pools/pool-<a|b|c>.json`. Point `--plantvillage-root` at `data/external/plantvillage/PlantVillage`, the folder that holds `train` and `val`. Each run re-fingerprints both source roots, so the three freezes took about 25 minutes together.

One combined audit covers all three pools. Give `data audit-sample edges` a JSON object of `"<pool>/<path>": [hash, label, pool]` entries, `--grouping-version`, and one `--inventory-digest POOL=SHA256` per pool (the `inventory` dependency of that pool's freeze; the sample is refused when the pools differ). Pairs form only inside a pool. The sampler draws about 240 pairs from a recorded stratum design (`--design PATH` replaces the default): every Pool C cross-label merged pair is taken whole, including Hamming 0–2, and the remaining budget is spread over the other strata of each pool, proportionally or by Neyman allocation, with at least one pair per nonempty stratum. A stratum reads `pool:band:distance-bin:relation:family`, where family is `same` or `split`, meaning whether the two images fall in one connected family at the threshold. Each sampled pair records its stratum, inclusion probability (sampled over population in the stratum) and family membership. The sample file also reports the count of Pool C scenes that mix labels as a dataset-quality output; no family is dropped.

`audit-cohort SAMPLE --agent-name NAME --agent-version VERSION` writes the agent-rated cohort; every audit record names the agent rater. `audit-evaluate` reports per-stratum and design-weighted wrong-merge and false-split rates with 95% bounds: each stratum gets an exact two-sided Clopper–Pearson interval, and the design-weighted bound is conservative rather than exact (it is a one-sided 95% upper bound that sums exact one-sided hypergeometric stratum bounds taken at tail 0.05 divided by the number of sampled strata, with census strata counted exactly, so its coverage is at least 95% and it loosens as the number of sampled strata grows; the report names the method under `bound_methods`), and Cohen's κ on the combined pools with a bootstrap stratified by pool and band and clustered on images that recur across pairs. Specific agreement is reported per pool. `--gate-output` writes the gate record described below.

The blind owner kit joins the grouping pairs with the owner subset of the mask-audit overlays (`data masks-audit`), mask items first, with no agent ratings, algorithm decisions, strata or predicted scenes:

```sh
uv run jute-disease data audit-kit artifacts/audit-cohort artifacts/mask-audit artifacts/audit-kit
uv run jute-disease data audit-sitting artifacts/audit-kit --limit 60
uv run jute-disease data audit-collect artifacts/audit-kit
```

Each `audit-sitting` call writes `sitting-NNN.csv` holding only items not yet rated, so a later sitting never re-shows a rated item; run it again after filling a sheet to resume. `audit-collect` refuses while any item is unrated, or if two sittings rate one item differently, then writes `owner.csv` for `audit-evaluate` and `owner-masks.csv` for `masks-evaluate`.

`audit-kit --held-out PATH` takes the `held-out.json` of `data scene-sets` (below) and adds one `scene` item per image, placed after the mask items and before the pairs and shuffled among themselves with the kit seed. A scene item is the image alone, copied into `scenes/` under its image ID, with no overlay, pool, label, source path or automatic tag; the kit refuses an image whose content does not match its ID. A sitting sheet shows a scene row with the image in `image_a` and only `scene_tag` to fill (any other rating on that row is refused), and `audit-sitting` prints the scene rubric with every sheet that holds scene rows. `audit-collect` then also writes `owner-scenes.csv` (`image_id`, `scene_tag`, `rubric_version`), the owner's first-pass scene tags. A kit with scene items is written as `jute-audit-kit-v2`; a kit without them keeps `jute-audit-kit-v1` and its former item order, and both remain readable.

Optionally, `--no-merge-ratings PATH` reads a UTF-8 CSV with the exact header
`image_a,image_b,verdict`. Paths are relative to the target image root and must
identify two different admitted images. Every row must have verdict `no_merge`;
other verdicts and malformed rows are rejected. Duplicate or reversed pairs are
collapsed. Exact byte-identical copies remain inseparable.

Optionally, `--pool A|B|C --agreement-results PATH` reads the gate record that `data audit-evaluate --gate-output` writes. The record carries `cohort_digest`, `grouping_version`, `hamming_threshold`, `inventory_digests` (per pool), `kappa_interval` (two ordered numbers in [-1, 1], or null for undefined kappa), `threshold`, `pass`, `method`, and the Pool C `wrong_merge` and `false_split` bounds with their limits. `freeze target` and `freeze policies` refuse a record whose grouping version, Hamming threshold, pool or inventory digest differs from the freeze, and recompute every pass flag from the recorded bounds, so a hand-edited flag is refused. The limits are module constants in `jute_disease.data.gates`, not inputs: the κ lower bound must reach 0.6, the Pool C wrong-merge 95% upper bound must be at most 5%, and the false-split rule fires when the Pool C 95% upper bound exceeds 20%. A null interval always fails. The freeze records `gate_outcomes` for `agreement`, `wrong_merge` and `false_split` (`pass`, `fail`, `pending`, `not_required`, `refreeze_required` or `escalate_to_owner`). Wrong merge and false split are Pool C gates; Pools A and B record `not_required`. Without the option every outcome stays pending. `freeze policies` refuses a pool until each outcome is `pass` or `not_required` for that pool.

The false-split rule is evaluated once, on the owner's imported ratings: a κ retry (`audit-evaluate --previous`) carries the first evaluation's false-split bound rather than re-reading it. When the rule fires at `dhash-v2`, the freeze records `false_split: refreeze_required`, and Pool C is re-frozen at the next higher threshold on a newly drawn cohort, before any tuning or target fit:

```sh
uv run jute-disease freeze target ... --pool C --grouping-version dhash-v3     --refreeze-from <the dhash-v2 freeze> --agreement-results <gate of the new cohort>
```

The re-freeze needs the digest-verified receipt of the freeze it replaces, the same inventory, and a cohort that differs from the one that tripped. It is allowed once. A `dhash-v3` gate that exceeds the limit, or a `--refreeze-from` pointing at a freeze that is already a re-freeze, is refused with an escalation to the owner. These grouping inputs are unavailable on the legacy freeze path.

### Leaf-mask and scene audit

The mask workflow implements the dated amendment to [ADR 0002](adr/0002-measure-background-confound-with-fixed-transforms.md): calibrate scene cutoffs on at most 100 development images, sample about 300 audit images, and use one blind owner sitting on 120 overlays for both mask quality and scene tag. The earlier 50-overlay paragraph in the original decision is superseded by that amendment.

The development JSON is an array of at most 100 records (at least one) with `image_id`, `path`, and `scene_tag`; the three tags are `clean-background`, `field`, and `unknown`. The separate audit inventory is an array of `image_id` and `path` records. Paths may be relative to their JSON file. Calibration stores the development IDs and source SHA-256 values, and records their number as `development_count`. Calibration and audit preparation take `--target-dir` (a frozen target), which may be repeated, and refuse when the development set or the audit inventory shares image content with the union of the named protected test partitions; the audit manifest lists every partition's manifest digest under `partition_check.target_manifest_digests`. Audit preparation also rejects any ID or image-content overlap with development data. Only the colour-threshold backend is audited: its name and version are recorded in the thresholds and the audit manifest, and any other recorded backend is refused.

```powershell
uv run jute-disease data masks-calibrate development.json --output artifacts/scene-thresholds.json --target-dir data/target
uv run jute-disease data masks-audit audit-inventory.json artifacts/scene-thresholds.json --output artifacts/mask-audit --target-dir data/target
# Complete agent.csv, then complete both owner.csv rating columns in one blind sitting.
uv run jute-disease data masks-evaluate artifacts/mask-audit/audit.json artifacts/mask-audit/agent.csv artifacts/mask-audit/owner.csv --agent-name "<rater name>" --agent-version "<rater version>" --output artifacts/mask-audit/evaluation.json
```

Calibration uses a deterministic grid search for the background-fraction and color-uniformity cutoffs. The development set must hold both `clean-background` and `field` images; `unknown` images are optional. With no `unknown` image, calibration sets `background_fraction_min` to `0.0`, so no image is classed `unknown`, and fits only `color_uniformity_min` by macro-F1 over the two tags present; with at least one, it fits both cutoffs by macro-F1 over all three tags. The thresholds record the tags the development set held under `development_tags`, which `masks-calibrate` prints with the rest of the thresholds; thresholds stored before that record still load. The owner comparison still scores all three tags, so an image the owner tags `unknown` counts as an error under two-tag thresholds. Audit sampling balances the nonempty predicted-scene strata, which oversamples rare scenes, and records each stratum's population and sample counts for the weighted mask-failure estimate. The owner sheet contains only an audit ID, overlay path, and blank mask/scene ratings; automatic scene tags and model outputs remain in `audit.json`. Rate the owner sample before opening the grouping-pair worksheet. Evaluation reports three-category Cohen's κ with a paired-bootstrap 95% interval, requires the lower bound to reach 0.6, reports the full owner-versus-automatic three-category confusion matrix, and passes the mask-failure gate only when the exact one-sided 95% upper bound of the population fail rate is within 10%. The bound combines per-stratum Clopper-Pearson upper bounds, Bonferroni-split across sampled strata; a stratum audited in full contributes its observed rate. A point estimate never passes the gate. The evaluation report records the backend name and version and carries a `digest`, the SHA-256 of its canonical JSON, for the freeze record to bind; no downstream step reads a pass flag from it. Production evaluation requires the external agent-rater name and version; no rater is selected by this code.

The scene rubric is `scene-tag-v2`, recorded in the audit manifest, the kit and the owner scene sheet. It rules that a leaf photographed against other leaves or plants is `field`, and that `unknown` is only for a tight crop, a blurred image, or an image with no readable background.

`data scene-sets` draws the scene-tag sets and the audit inventory in one seeded step, reading manifests only:

```powershell
uv run jute-disease data scene-sets --target-dir A=data/target-a --target-dir B=data/target-b --target-dir C=data/target-c --image-root A=data/pool-a --image-root B=data/pool-b --image-root C=data/pool-c --development development.json --exclude excluded-ids.json --output artifacts/scene-sets
uv run jute-disease data masks-audit artifacts/scene-sets/audit-inventory.json artifacts/scene-thresholds.json --output artifacts/mask-audit --target-dir data/target-a --target-dir data/target-b --target-dir data/target-c
```

It takes one `--target-dir POOL=PATH` and one `--image-root POOL=PATH` for each of pools A, B and C (the root is the directory the manifest paths are relative to; it only makes the written paths absolute, and no image is opened), the development manifest, an optional `--exclude` JSON array of further image IDs, and `--seed`. Eligible images are the train and validation entries of each fit manifest, less quarantined images, images whose ID appears in more than one pool, and every family with a member in a protected-test partition, in the development set (matched by image ID and, where the file is present, by content SHA-256) or among the excluded IDs. A development or excluded image that matches no manifest entry is refused, because its family could not be excluded. Each pool's eligible images are shuffled once with the seed: `held-out.json` takes the first 60, 60 and 180 images from A, B and C, a simple random sample; the reserve of 270, `reserve-test.json` (20, 20, 60) then `reserve-fit.json` (34, 34, 102), takes the next images whose family no earlier set holds, so the reserve is conditional on the held-out draw and reaches large families less often; `audit-inventory.json` holds every eligible image of the families left untouched, in the `image_id` and `path` format `masks-audit` reads. A family is kept inside one output, so the four files share no image and no family; unchosen members of a family claimed by a set are left out and counted as `unused_family_members`. `receipt.json` records the seed, each pool's grouping version, threshold and digest, the fit and protected-test manifest digests, the development manifest digest and image count, counts per pool and set, exclusion counts by reason, and the SHA-256 of each output. The command refuses an existing output directory and a pool with too few eligible families, and the same inputs and seed give the same files.

`data scene-evaluate` applies the scene-tag protocol to the held-out set in two steps, so the owner's later sittings are prepared from the first:

```powershell
uv run jute-disease data scene-evaluate compare --held-out artifacts/scene-sets/held-out.json --owner artifacts/audit-kit/owner-scenes.csv --judge haiku=haiku-scenes.csv --judge luna=luna-scenes.csv --thresholds artifacts/scene-thresholds.json --output artifacts/scene-compare
uv run jute-disease data scene-evaluate test --held-out artifacts/scene-sets/held-out.json --thresholds artifacts/scene-thresholds.json --compare artifacts/scene-compare --owner artifacts/audit-kit/owner-scenes.csv --final final-scenes.csv --recheck recheck-scenes.csv
```

Every tag sheet is a CSV in the `owner-scenes.csv` format (`image_id`, `scene_tag`, `rubric_version`; a `path` column is allowed and ignored). A sheet is refused when it misses a held-out image, repeats one, names an image outside the set, carries a tag outside the rubric (a blank included) or another rubric version.

`compare` takes the owner's first-pass sheet from `audit-collect`, one `--judge NAME=PATH` sheet per agent judge and the frozen thresholds as `--thresholds`. It refuses an existing output directory and a held-out image that is in the thresholds' development set. It writes `first-pass.csv` (the first tag of the owner and of each judge, kept apart from the final tags), `disagreements.csv` (every image where any judge differs from the owner's first tag), `recheck.csv` and `compare.json`. `compare.json` records the seed, the SHA-256 of the held-out set, of each sheet and of the thresholds, the disagreement IDs, and each judge's three-category Cohen's κ against the owner with a 95% paired-bootstrap interval. The recheck set is a simple random sample of `--recheck-count` images (default 30) drawn with `--seed` from the images every judge agreed with the owner on; when fewer are unanimous, all of them are taken. `recheck.csv` lists them in draw order with their paths and a blank `scene_tag` for the owner to fill without the first tags in view.

The owner then resolves the disagreements. The final sheet tags every held-out image: copy the first-pass sheet and change only disagreement rows. A final tag that differs from the first pass on an image that was not a disagreement is refused.

`test` checks that the held-out set, the first-pass sheet and the thresholds are the files `compare` read, so thresholds changed after the compare step are refused. It checks each image's content against its ID, classifies every image with the colour-threshold backend and the frozen thresholds, and scores the result against the final tags.

The held-out test is run once. Its record is always `test.json` inside the compare directory; the command takes no output path. Once the inputs are checked, `test` creates that file exclusively, before it classifies any image, and fills it when the scores are known. Any later run is refused while the file exists, whatever its arguments. A run that stops after the claim leaves `test.json` empty: report that crashed claim with the compare directory as it stands, and do not delete the file to run the test again. The refusal covers one compare directory, so a second `compare` into a new directory would open a second test: run `compare` once per held-out set, and record the thresholds digest in the study record before the held-out images are tagged.

The record scores the thresholds as follows:

- Each image weighs its pool's population (A 920, B 1,390, C 9,574 of 11,884) over the pool's held-out images. Pooled accuracy is therefore the sum over pools of pool share times pool accuracy, and pooled clean-background recall is a ratio of weighted confusion counts, never a mean of pool recalls.
- Pool accuracy and recall carry Wilson 95% intervals. The pooled intervals are Wilson intervals at the Kish effective sample size, (sum of weights)² / (sum of squared weights). They are an approximation, because a Wilson interval at the Kish size does not use the stratified variance of the pooled estimate, and they do not gate the test.
- The test passes when pooled accuracy is at least 90% and pooled clean-background recall is at least 80%, both on the point estimate, compared exactly. With no clean-background image in the set the recall is undefined and the test fails.
- A pool with fewer than 10 clean-background images reports `insufficient evidence` for its clean recall, with its counts; a pool with none reports `none`. Those images still count in the pooled recall.
- Fewer than 10 `unknown` images in the held-out set reports `insufficient evidence` for unknown; otherwise unknown recall is reported with its support, overall and per pool. Thresholds fitted on a development set with no `unknown` image class nothing as `unknown`; owner-`unknown` images are then scored as errors.
- With `--recheck`, the record reports how many recheck images kept the owner's first tag and lists the ones that changed; without it the recheck is recorded as `not supplied`.

The record holds the per-pool and pooled results, the weighted and per-pool confusion matrices, each image's final tag, predicted tag and features, and `inputs`: the SHA-256 of the held-out set, the thresholds, `compare.json`, the first-pass sheet, each judge sheet, the final sheet and the recheck sheet. Its `digest` is the SHA-256 of its canonical JSON. The held-out test is run once: `test` refuses to run when the record already exists, and never replaces it.

`data masks-fixture --output PATH` runs this workflow end to end with local synthetic images, the colour-threshold backend, a synthetic protected-test partition, and explicitly synthetic ratings. Its output is fixture evidence only; it contains no real threshold, owner rating, or audit result.

### Rating app persistence and recovery

The rating app stores a backup for each opened session and a versioned change
journal under `.rating-applet/<sitting filename>/` beside the sitting. These
private files contain item identities and rating changes; keep the directory
private and retain it with the sitting. It is ignored by Git and lies outside
the collector's top-level sitting-sheet glob. Do not edit the sheet or run
collection while the app is writing.

An interrupted save is reconciled on the next access: if the sheet matches the
previous revision, the attempted change is abandoned; if it matches the new
revision, the change is recorded as committed. If neither matches, or the
journal is incomplete, the app reports that recovery is required. Stop editing
and preserve the sheet, journal and backups for inspection; do not delete or
modify the journal to bypass the check. A failed save response does not confirm
whether a replacement reached disk, so reload and check the sitting before
continuing.

Finishing a complete sitting makes it read-only. A genuine entry correction
requires a deliberate external call to `reopen_sitting(sheet, reason)` from
`jute_disease.rating_applet`; the nonempty reason is recorded in the private
journal. Reopening is refused after collection, and collection itself also makes
the sitting read-only. Generate the next sitting and collect using the existing
commands above, outside the app.

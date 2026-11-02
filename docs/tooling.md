# Tooling

See [Continuous integration](ci.md) for pull-request and main-branch checks and
the response to a failing main run.

## Supported environment

- Python 3.14.10 is selected by `.python-version`; `pyproject.toml` supports Python 3.14 only.
- `uv.lock` is the dependency source of truth. Use `uv sync --locked --all-groups` for development and add `--extra masks` when validating the optional mask package imports.
- PyTorch and TorchVision are pinned as one version pair. uv selects the official [PyTorch CUDA 12.6 wheels on Linux and CUDA 13.0 wheels on Windows](https://pytorch.org/get-started/previous-versions/). CI validates Linux; run Windows-specific checks locally when changing Windows dependency selection.
- The optional `masks` extra installs Transformers for [Segment Anything Model 2](https://github.com/facebookresearch/sam2) code and API checks. It does not download checkpoints or run segmentation.
- Lightning remains the deep-learning training framework. Active run evidence is written locally as Lightning CSV logs and JSON evaluation records with metrics, class order, confusion matrix, and relevant settings. Historical W&B configs, notebooks, run records and artifacts live under the `submitted-study` tag and in historical folders.

## Updating dependencies

Update dependencies deliberately, regenerate `uv.lock`, and review the full platform-specific diff. Keep PyTorch and TorchVision on the same release pair. Linux CI runs lock resolution, the dependency audit, import checks, focused PR tests and the full suite on `main`. Run Windows resolution and the CUDA smoke check locally only when a change affects Windows-specific dependency selection. Recheck the official PyTorch wheel indexes when changing the pair. Security advisories for the locked requirements should be handled as soon as practical; review general dependency updates at least quarterly.

`just audit` runs `scripts/audit_dependencies.py`. It exports the exact locked requirements, scans all non-PyTorch packages through pip-audit, then scans the pinned PyTorch and TorchVision base release numbers. The CUDA wheel build suffixes (`+cu126` and `+cu130`) are platform-specific and are not published as PyPI releases, so the advisory lookup uses their shared upstream release numbers. The check does not install or change packages.

## Preprocessing and freezes

Preprocessing identity revision 2 covers the shared 256-pixel letterbox, bilinear resampling, rounded uint8 ImageNet-mean padding, tensor normalization and cache encoding. The classical route uses the same geometry and pad values without tensor normalization. Freeze manifest version 2 records each image's letterbox pad fraction and summarizes it by source and class. Version 1 manifests remain readable, but new freezes use version 2. A preprocessing identity change invalidates matching image caches; it does not rewrite or delete existing freeze artifacts. Re-freeze explicitly when the source, curation policy or split must change.

## Benchmark queue

`jute-disease queue plan` and `queue status` list every declared fit and need no policy directory. `queue run` over the full matrix requires `--policy-dir`, the directory of receipted policy manifests (with `assignments.json`), and reads each pool's images from `--pool-a-root`, `--pool-b-root` and `--pool-c-root` (falling back to `--image-root`). The same options reach `just queue-run` and `just queue-unattended` (an optional leading timeout in seconds, then the options). Tuning fits (`classical/tune/...`, `cnn/tune/...`, `source/tune/...`) run before the finals that reuse their selection and write a `tuning.json` that records `test_accessed: false`.

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
identities as the receipts. The statistics bundle has `receipt.json` and a
report JSON containing `config_digest`, `manifest_digest`,
`evaluation_receipt_sha256` and a `statistics` object. Its receipt records the
report hash and must bind to the evaluation receipt hash. Every finite numeric
leaf under `metrics` or `statistics` is emitted as an export-ID macro;
`manuscript check` verifies the receipts, lineage and copied files again.
These options import existing results only; they do not score data or open
partition images.

Classical estimators and their nine-configuration grids: k-NN with k in {1, 3, 5, 7, 11, 15, 21, 31, 51}; L2 logistic regression with C from 1e-4 to 1e4 in nine log-spaced values; RBF-SVC with C in {1, 10, 100} and gamma in {0.1, 1, 10} times scale; random forest with 500 trees, max_features in {sqrt, 0.1, 0.3} and min_samples_leaf in {1, 3, 10}. Gaussian naive Bayes searches `var_smoothing` over 1e-12 to 1e-4 in nine decades. The crafted feature vector has 548 values; z-scores are fitted on the training fold only.

## Lab helpers

The scripts in `scripts/` are Linux-only command-line helpers for lab runs:

- `notify_run.py` sends start, hourly heartbeat, success and failure messages to an `ntfy` topic. `NTFY_TOPIC` and `NTFY_TOKEN` come from the environment. The token is sent only in the HTTPS Authorization header, never in process arguments; transport errors do not print credentials. A cull or other abrupt kill can prevent a final failure message; a missed heartbeat signals interruption.
- `lab_run.sh` holds a `flock` lock for the full process lifetime, applies a timeout to the process group, and can detach a run while recording its PID and log path. Example: `scripts/lab_run.sh /path/to/run.lock 7200 uv run jute-disease queue run`.
- `queue_unattended.sh` (through `just queue-unattended`) checks notification connectivity, launches a detached locked queue, and records a transfer manifest after each invocation. Re-run it after a JupyterLab cull or host restart; the queue reuses only fits whose receipts still match.
- `queue_batch.py` runs one queue command and records the full `artifacts/runs/` tree after it exits, including a failed invocation, so completed receipts and partial-run evidence can be copied together.
- `verify_transfer.py record ROOT MANIFEST` writes a SHA-256 inventory outside the batch. Run `verify_transfer.py verify ROOT MANIFEST` after copy-back; missing, changed, unexpected, unsafe, hard-linked, non-regular and colliding paths reject the batch.

On the lab, run `just queue-unattended --policy-dir <dir> --pool-a-root <dir> --pool-b-root <dir> --pool-c-root <dir>` from the checkout. The ntfy topic and token must already be supplied through the host's environment secret mechanism. After the detached invocation ends, copy `artifacts/runs/` and the newly written manifest from `artifacts/transfer-manifests/` to the PC. Verify the PC copy, for example:

```powershell
uv run --locked python scripts/verify_transfer.py verify artifacts/runs artifacts/transfer-manifests/queue-<batch-id>.json
```

Repeat the copy and verification after every queue invocation. Keep the lab copy until verification succeeds; the manifest snapshots the full run store so earlier receipts needed for resume remain included.

These helpers do not run real lab sessions or experiments as part of tooling CI. Lab validation and A100 readiness are tracked separately in [issue 114](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/114).

### Grouped target freezes

`freeze target --grouping-version dhash-v1` requires `--plantvillage-root` and
`--plantdoc-root`, each pointing to an image inventory in the supported
`<label>/<file>` or `<train|val|test>/<label>/<file>` layout. The freeze screens
all inventoried target images against both supplied source inventories, records
source inventory digests and all matching source paths, target paths and match
methods in `target/grouping.json`, and does not silently discard matches. Verify
that the supplied roots are the intended source packages. Source intake is a
separate operation and continues to apply its own quarantine rules.

Optionally, `--no-merge-ratings PATH` reads a UTF-8 CSV with the exact header
`image_a,image_b,verdict`. Paths are relative to the target image root and must
identify two different admitted images. Every row must have verdict `no_merge`;
other verdicts and malformed rows are rejected. Duplicate or reversed pairs are
collapsed. Exact byte-identical copies remain inseparable.

Optionally, `--agreement-results PATH` reads a JSON object with exactly
`kappa_interval` (two ordered numbers in [-1, 1], or null for undefined
kappa), `threshold` (a number in [-1, 1]), `pass` (a boolean), and `method`
(a nonempty string). A numeric interval passes only when its lower bound meets
the threshold; a null interval always fails. `data audit-evaluate --gate-output`
writes this record separately from its full rating report. The receipt records
the input and the resulting pass/fail gate. Without the option, agreement
remains pending. These grouping inputs are unavailable on the legacy freeze path.

### Leaf-mask and scene audit

The mask workflow implements the dated amendment to [ADR 0002](adr/0002-measure-background-confound-with-fixed-transforms.md): calibrate scene cutoffs on 100 development images, sample about 300 audit images, and use one blind owner sitting on 120 overlays for both mask quality and scene tag. The earlier 50-overlay paragraph in the original decision is superseded by that amendment.

The development JSON is an array of exactly 100 records with `image_id`, `path`, and `scene_tag`; the three tags are `clean-background`, `field`, and `unknown`. The separate audit inventory is an array of `image_id` and `path` records. Paths may be relative to their JSON file. Calibration stores the development IDs and source SHA-256 values. Audit preparation rejects any ID or image-content overlap with development data and requires the same mask backend that calibrated the cutoffs.

```powershell
uv run jute-disease data masks-calibrate development.json --output artifacts/scene-thresholds.json
uv run jute-disease data masks-audit audit-inventory.json artifacts/scene-thresholds.json --output artifacts/mask-audit
# Complete agent.csv, then complete both owner.csv rating columns in one blind sitting.
uv run jute-disease data masks-evaluate artifacts/mask-audit/audit.json artifacts/mask-audit/agent.csv artifacts/mask-audit/owner.csv --agent-name "<rater name>" --agent-version "<rater version>" --output artifacts/mask-audit/evaluation.json
```

Calibration uses a deterministic grid search for the background-fraction and color-uniformity cutoffs. Audit sampling balances the nonempty predicted-scene strata, which oversamples rare scenes, and records each stratum's population and sample counts for the weighted mask-failure estimate. The owner sheet contains only an audit ID, overlay path, and blank mask/scene ratings; automatic scene tags and model outputs remain in `audit.json`. Rate the owner sample before opening the grouping-pair worksheet. Evaluation reports three-category Cohen's κ with a paired-bootstrap 95% interval, requires the lower bound to reach 0.6, reports the full owner-versus-automatic three-category confusion matrix, and gates the population-weighted estimated fail rate below 10%. Production evaluation requires the external agent-rater name and version; no rater is selected by this code.

`data masks-fixture --output PATH` runs this workflow end to end with local synthetic images, a deterministic stub mask backend, and explicitly synthetic ratings. Its output is fixture evidence only; it contains no checkpoint, real threshold, owner rating, or audit result.

SAM 2 refinement is optional and uses the `masks` extra. Supply an existing local checkpoint directory to both calibration and audit commands with `--checkpoint PATH`; the loader sets `local_files_only=True` and never fetches weights. It requires a local model-weight file, includes the weight digest in the backend version, and stores the config digest in per-mask diagnostics. Mask sidecars retain the binary mask, overlay, model/version, and fixed mask rubric. SAM output also records intersection-over-union against the color-threshold mask. The locked Transformers 5.22.1 package was not indexed in Context7, so implementation used the official [Hugging Face SAM2 documentation](https://huggingface.co/docs/transformers/model_doc/sam2) as the API reference.

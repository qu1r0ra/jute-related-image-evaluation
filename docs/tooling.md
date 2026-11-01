# Tooling

See [Continuous integration](ci.md) for pull-request and main-branch checks and
the response to a failing main run.

## Supported environment

- Python 3.14.10 is selected by `.python-version`; `pyproject.toml` supports Python 3.14 only.
- `uv.lock` is the dependency source of truth. Use `uv sync --locked --all-groups` for development and add `--extra masks` when validating the optional mask package imports.
- PyTorch and TorchVision are pinned as one version pair. uv selects the official [PyTorch CUDA 12.6 wheels on Linux and CUDA 13.0 wheels on Windows](https://pytorch.org/get-started/previous-versions/). CI validates Linux; run Windows-specific checks locally when changing Windows dependency selection.
- The optional `masks` extra installs Transformers for [Segment Anything Model 2](https://github.com/facebookresearch/sam2) code and API checks. It does not download checkpoints or run segmentation.
- Lightning remains the deep-learning training framework. Active run evidence is written locally as Lightning CSV logs and JSON evaluation records with metrics, class order, confusion matrix, and relevant settings. Active W&B integration has been removed; historical configs, notebooks, run records and artifacts remain available under the `submitted-study` tag and historical folders.

## Updating dependencies

Update dependencies deliberately, regenerate `uv.lock`, and review the full platform-specific diff. Keep PyTorch and TorchVision on the same release pair. Linux CI runs lock resolution, the dependency audit, import checks, focused PR tests and the full suite on `main`. Run Windows resolution and the CUDA smoke check locally only when a change affects Windows-specific dependency selection. Recheck the official PyTorch wheel indexes when changing the pair. Security advisories for the locked requirements should be handled as soon as practical; review general dependency updates at least quarterly.

`just audit` runs `scripts/audit_dependencies.py`. It exports the exact locked requirements, scans all non-PyTorch packages through pip-audit, then scans the pinned PyTorch and TorchVision base release numbers. The CUDA wheel build suffixes (`+cu126` and `+cu130`) are platform-specific and are not published as PyPI releases, so the advisory lookup uses their shared upstream release numbers. The check does not install or change packages.

## Preprocessing and freezes

Preprocessing identity revision 2 covers the shared 256-pixel letterbox, bilinear resampling, rounded uint8 ImageNet-mean padding, tensor normalization and cache encoding. The classical route uses the same geometry and pad values without tensor normalization. Freeze manifest version 2 records each image's letterbox pad fraction and summarizes it by source and class. Version 1 manifests remain readable, but new freezes use version 2. A preprocessing identity change invalidates matching image caches; it does not rewrite or delete existing freeze artifacts. Re-freeze explicitly when the source, curation policy or split must change.

## Lab helpers

The scripts in `scripts/` are Linux-only command-line helpers for the validation work owned by issue 114:

- `notify_run.py` sends start, hourly heartbeat and completion messages to an `ntfy` topic. `NTFY_TOPIC` and `NTFY_TOKEN` come from the environment. The token is sent only in the HTTPS Authorization header, never in process arguments; transport errors do not print credentials.
- `lab_run.sh` holds a `flock` lock for the full process lifetime, applies a timeout to the process group, and can detach a run while recording its PID and log path. Example: `scripts/lab_run.sh /path/to/run.lock 7200 uv run jute-disease queue run`.
- `verify_transfer.py record ROOT MANIFEST` writes a SHA-256 inventory outside the batch. Run `verify_transfer.py verify ROOT MANIFEST` after copy-back; missing, changed, unexpected, unsafe, hard-linked, non-regular and colliding paths reject the batch.

These helpers do not run real lab sessions or experiments as part of tooling CI. Lab validation and A100 readiness are handled separately under issue 114.

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

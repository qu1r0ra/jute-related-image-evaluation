# Legacy names

The repository was renamed on 2026-10-31 when the study moved from the original course benchmark to the related-image evaluation (issue #98). Tracked source, configs, docs and links use the current names; this file is the only place that maps them to the old ones.

| Item | Current | Legacy |
| --- | --- | --- |
| GitHub repository | `qu1r0ra/jute-related-image-evaluation` | `qu1r0ra/jute-disease-detection` (GitHub redirects old issue, PR and clone URLs) |
| Python distribution (`pyproject.toml`) | `jute-related-image-evaluation` | `jute-disease-detection` |
| Import package | `jute_disease` | unchanged |
| W&B project (`WANDB_PROJECT`, historical configs and notebooks) | `jute-related-image-evaluation` | `jute-disease-detection` under entities `grade-descent` and `immanumali12-de-la-salle-university` |
| Jupyter kernel display name | `jute-related-image-evaluation (3.11.14)` | `jute-disease-detection (3.11.14)` |

## Reproducing the original study

- Recorded notebook outputs under `notebooks/historical/` (tracebacks, run logs, W&B run URLs) keep the legacy name because they record what ran.
- The W&B project was not renamed on the W&B side. Historical runs, W&B links and artifacts such as `svm-champion` stay under the legacy project path permanently. No W&B-side rename or project creation will happen. Reproduce the original study from tag `submitted-study` and its lockfile.
- Zenodo record `10.5281/zenodo.23414231` keeps its DOI and its original title and repository link until the next release.

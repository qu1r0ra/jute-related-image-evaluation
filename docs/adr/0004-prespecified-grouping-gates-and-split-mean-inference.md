# 4. Prespecify grouping gates and test the primary contrast on split means

Status: accepted, 2026-11-13. Partly supersedes [ADR 0003](0003-ten-split-draws-with-rotating-source-checkpoints.md) (uncertainty). Canonical protocol: [issue 172](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/172) and the revised-protocol spec under [issue 98](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/98).

## Context

Independent plan, code, statistics and lab-operations reviews of the study, after the dhash-v2 freezes of pools A, B and C, found four problems:

- One grouping threshold does not fit every pool. At dhash-v2 threshold 6, Pool A merges unrelated images: about 20% of its same-label merged pairs are wrong. Pool C under-merges: in the agent's provisional ratings, 8 of 9 same-label near-miss pairs (Hamming 7–16) are near-duplicates, and 5 of those 8 sit in different families.
- Pool C contains the same scene under different labels. 10 of 13 merged cross-label pairs were rated near-duplicate.
- A 120-pair blind audit cannot bound a wrong-merge rate of 5% with useful power.
- The nested bootstrap of ADR 0003 covers about 90% at its nominal 95%. With 2/2/1 seed strata it undercounts seed variance, and it counts family variance twice.

These decisions are recorded before the owner's blind ratings and before any target fit, so that neither grouping nor inference is tuned on the results.

## Decision

- **One threshold.** Keep `dhash-v2` at threshold 6 for every pool. Pools A and B are negative controls. Pool A's wrong-merge rate is reported as a limitation.
- **Audit size and gates.**
  - The blind audit grows to about 240 pairs. It includes every cross-label merged pair in Pool C and audits Hamming distances 0–2.
  - Rates and bounds are stratified, using exact (Clopper–Pearson) bounds.
  - The κ gate (lower bound ≥ 0.6) is computed on the pools combined, and specific agreement is reported per pool.
  - The wrong-merge gate requires a 95% upper bound of at most 5%, in Pool C only.
- **Pool C false-split rule.** A false split is a near-miss pair rated duplicate whose images sit in different families. If the 95% upper bound of Pool C's false-split rate exceeds 20%, Pool C is re-frozen at a higher threshold before any training, and the audit is re-run. Otherwise Pool C's leakage effect is reported as a lower bound, and the inference-only strict View 2 subset carries the main claim.
- **Cross-label duplicates stay.** Mixed-label scenes stay in Pool C, outside the existing #105 drop rule for identical conflicting duplicates. Their count is reported as a dataset-quality finding.
- **Primary inference on split means.**
  - The single primary test is Pool C, View 1, MobileNetV2 ImageNet route, image-random minus near-duplicate macro-F1.
  - It is a Welch test on the ten split means.
  - The smallest effect of interest, 2 macro-F1 points, is fixed in advance.
  - At most five secondary tests are Holm-adjusted. Pools A and B get equivalence tests. Everything else is exploratory.
  - The nested bootstrap stays descriptive, with a reported coverage simulation. In View 1 it resamples per policy, stratified by class, and it asserts class support instead of rejecting draws.

## Alternatives considered

- **Per-pool thresholds.** These fit each pool better, but add a degree of freedom chosen after looking at each pool's images.
- **No re-freeze rule.** This keeps one freeze, but leaves known under-merging in Pool C unaddressed.
- **Dropping mixed-label families.** This gives cleaner labels, but the pool would no longer be the public dataset others use.
- **A 120-pair descriptive audit.** This costs less owner time, but cannot support a gate.
- **The nested bootstrap as primary inference.** Its intervals undercover with this many seeds and splits.

## Consequences

- Owner rating time roughly doubles, to about 240 pairs. It can be split across sittings.
- Pool C may be frozen twice. Any re-freeze happens before tuning or target fits, so no fit is rerun.
- A View 1 result is unpaired across policies. Paired claims use View 2.

# 3. Use ten split draws with rotating source checkpoints

Status: accepted, 2026-11-01. Canonical protocol: [issue 106](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/106).

## Context

The earlier matrix used one primary split plus three redraws per policy, and trained each plant-pretraining source stage once. Four splits give the top level of the hierarchical bootstrap too few draws for stable intervals, and split-to-split variation usually exceeds seed variation. A single source checkpoint per stage makes every route finding depend on one source run, so a route effect cannot be separated from a lucky or unlucky source fit. Lab A100 access is persistent, so compute is not the binding constraint.

## Decision

- Use one primary split plus nine redraws per policy, each with five seeds, in both views (about 2,700 final MobileNetV2 fits).
- Train each source stage with three independently seeded checkpoints. PlantDoc checkpoint *j* is trained from PlantVillage checkpoint *j*. The source learning rate is selected once on source-stage validation and reused.
- A target fit with seed *s* on split *k* uses checkpoint (*s* + *k*) mod 3, giving 17/17/16 fits per checkpoint. The assignment table is identical across policies and views, so comparisons stay paired.
- Route estimands weight checkpoints equally: average within each checkpoint, then across checkpoints.
- Uncertainty uses one nested bootstrap (splits, then seeds, then test families), with draws and family multiplicities shared across all systems in a comparison. Intervals are conditional on the three retained source checkpoints.

## Alternatives considered

- One primary split plus three redraws: cheaper, but split-level uncertainty rests on four draws.
- One source checkpoint per stage: the route claim stays conditional on a single source run, and source variability is unestimated.
- Seed *s* using checkpoint *s* mod 3: weights the checkpoints 2:2:1 with five seeds.
- A full cross of source checkpoints with target seeds: triples the route fits for little added information over the rotation.

## Consequences

- Roughly 110–220 additional GPU-hours, including the prespecified EfficientNet-B5 capacity check; the [issue 114](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/114) pilots replace the estimate.
- Route findings are disclosed as conditional on MobileNetV2 and three source checkpoints per stage.
- Changing the split count or the assignment after fits begin would require rerunning them.

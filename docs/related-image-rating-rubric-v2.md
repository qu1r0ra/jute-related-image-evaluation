# Blind rating rubric: `rating-bundle-v2`

The machine-readable source is [`rating_rubric.py`](../src/jute_disease/data/rating_rubric.py). The local applet and independent-rater instructions use those definitions. This page is the inspectable guide for preparing a future pass. It does not authorize or conduct real ratings.

## Pair identity and resemblance: `related-image-v2`

Apply the first matching category in this order:

1. **Visually identical** (`exact`; merge): The images have indistinguishable visible content, framing and orientation under inspection. A copy or re-save qualifies if it introduces no perceptible change. This does not establish byte identity.
2. **Near-duplicate** (`near-duplicate`; merge): The images visibly differ, but corresponding distinctive details establish that they depict the same physical leaf or overlapping content from the same captured scene. Differences may include another capture, a crop, rotation, reflection or image processing. A shared generic background alone is insufficient.
3. **Visually similar** (`very similar`; no merge): Neither identity category applies, but the images strongly resemble one another overall. For leaf-focused images, judge resemblance primarily from the leaf's shape, venation, markings and damage; background is secondary.
4. **Visually dissimilar** (`different`; no merge): Neither identity category applies, and overall resemblance is limited. Shared generic features, such as green color or similar placement, do not by themselves establish strong resemblance.

The display names map to existing stored values. The first two values retain merge semantics; the last two retain no-merge semantics. File-hash exact-copy grouping is separate from visual identity. These categories form an identity-first decision sequence, not a validated interval or ordinal similarity scale. Flag identity uncertainty when resolving it could change the merge decision.

## Mask quality: `leaf-mask-v2`

- **Target leaf:** The visually prominent leaf that appears to be the intended subject. Position and apparent closeness may help identify it, but neither determines the target.
- **Background:** Everything outside the target leaf, including other leaves, stems and surrounding objects.
- **Pass:** The target leaf and its visible lesions are covered, with no more than negligible clipping or background inclusion.
- **Minor defect:** Localized clipping or background inclusion is visible, but most target-leaf tissue and visible lesions remain covered.
- **Fail:** Substantial target-leaf or lesion clipping, or substantial background inclusion, is present.

Consider target-leaf tissue coverage, visible-lesion coverage and background inclusion (mask leakage) separately, then apply the most severe applicable defect. A small clipped region containing most visible lesion evidence may warrant **Fail**. Flag an ambiguous target leaf for review instead of forcing a mask rating.

## Scene tags: unchanged `scene-tag-v2`

- **Clean background:** Most visible background is a uniform, uncluttered surface.
- **Field:** The background contains substantial outdoor or field structure, including other leaves or plants.
- **Unknown:** The image provides insufficient readable background, for example because of a tight crop or blur.

Keep existing scene labels and their provenance. Adopting the current instrument does not require rerating unchanged scene categories; do not mark historical records as though they were rated with revised wording.

## Synthetic calibration examples

Use these examples to check consistent use of the definitions. They are guidance, not study samples or software assertions about what a real rater must see.

| Synthetic case | Expected guidance |
| --- | --- |
| The visible content, framing and orientation are unchanged after a file re-save. | Visually identical. |
| A crop differs, while matching distinctive venation and damage identify the same leaf. | Near-duplicate. |
| A visible crop or rotation changes framing, while distinctive details still identify the same leaf. | Near-duplicate. |
| The focal leaf changes, but distinctive overlapping details establish the same captured scene. | Near-duplicate. |
| A generic background matches while leaf shape and markings differ. | The background alone does not establish identity; judge resemblance from visible content. |
| Different leaves strongly match in shape, venation, markings and damage, with no identity evidence. | Visually similar. |
| Images share only green color and similar placement. | Visually dissimilar. |
| Distinctive details are unreadable and identity could change the merge decision. | Flag identity uncertainty for review. |
| Several leaves are equally prominent and the intended mask target is unclear. | Flag the ambiguous target for review. |
| A small clipped area contains most visible lesion evidence. | Fail may apply because lesion loss is considered separately from total area. |

## Preparing a future independent pass

Real owner and agent ratings, collection, evaluation and policy freezing remain under [issue #172](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/172). Before that work starts:

1. Preserve the earlier sheets, agent prompts where available, instrument versions and outputs. Keep each source pass intact and identify its rubric and instrument provenance.
2. Use the same affected pair cohort for the fresh independent agent pass. Give the agent the pair images, opaque item identifiers and these definitions. Do not provide earlier owner or agent answers, grouping decisions, predicted labels, algorithm outcomes, rating distributions or gate outcomes.
3. Record the new rubric and instrument versions with the pass. Treat the owner's revised pass as a review of prior evidence, not as a new naive first observation. Keep the original independent first pass separately identifiable.
4. Leave open review flags unresolved until a reviewer records an explicit resolution note. The flag is review metadata, not a fifth rating value. Preserve all flag and resolution history.
5. Reconcile adoption with the existing evaluator contract before any revised gate run. Keep the cohort fixed, retain original false-split evidence, and apply the existing retry allowance only when its stated conditions hold. Do not reset a gate or extend the retry allowance. Surface any protocol conflict before execution.
6. Resolve mask rerating scope under #172. The unchanged scene categories stay run-once; do not retag them solely because the pair and mask rubric bundle was adopted.

No study images, ratings or gates were used to prepare this guide. Synthetic fixtures were used for software behavior only.

## Methods-ready wording

Pair judgments used the ordered, identity-first `related-image-v2` rubric: visually identical and near-duplicate pairs retained merge semantics, while visually similar and visually dissimilar pairs retained no-merge semantics. Visual identity was judged separately from file-hash identity; the categories were not treated as an ordinal scale. Mask review considered target-leaf tissue coverage, visible-lesion coverage and background inclusion separately, applying the most severe applicable defect and flagging ambiguous targets. Uncertain pair identity and ambiguous mask targets were retained for explicit review and resolution. Scene tags retained their existing `scene-tag-v2` definitions and run-once treatment. Each pass retained its rubric and instrument provenance.

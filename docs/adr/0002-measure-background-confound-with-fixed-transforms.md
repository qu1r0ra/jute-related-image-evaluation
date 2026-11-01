# 2. Measure the background confound under fixed transforms

Status: accepted, 2026-10-31. Canonical protocol: [issue 106](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/106#issuecomment-6545535815).

## Context

Reviewers of the prior submission suspected that one class was learned from clean backgrounds, because that class held many clean-background images. Published work shows that backgrounds alone can carry label information (Noyan 2022 on PlantVillage; Xiao et al. 2020) and that background removal does not reliably improve plant-disease accuracy (Craze et al. 2022). The study's question is whether related images inflate evaluation, so any preprocessing or augmentation that differs between arms would confound that comparison.

## Decision

Measure the background confound; do not mitigate it with a masked-training arm.

- Report a scene-by-class support table and scene-stratified metrics.
- Train a background-only probe: an automatic leaf mask (colour threshold plus largest component, refined by SAM) blanks the leaf, then the classical models and MobileNetV2 on the ImageNet route are trained on the remainder on the primary split for every pool and policy, with five seeds.
- Evaluate every final checkpoint on a stratified, leaf-only sample of the test images.
- Gate the probe and leaf-only results on a mask audit: a named vision-language model rates a stratified overlay sample against a fixed rubric, colour-threshold and SAM masks are compared, and the owner blind-rates a random 50 overlays. The audit passes at Cohen's κ ≥ 0.6 and an estimated fail rate under 10%.

Fix preprocessing and augmentation before any fit, identically across policies, views and routes, and tune only the learning rate. Images are letterboxed to 256 × 256 with ImageNet-mean padding and ImageNet normalization. Training adds RandomResizedCrop with scale 0.5–1.0, horizontal and vertical flips, and brightness and contrast jitter of 0.2, with no hue or saturation jitter. Evaluation applies the letterbox only. A validation-only pilot may replace the letterbox with resize plus center crop only if the letterbox loses by more than 2 macro-F1 points.

## Alternatives considered

- Masked or background-swapped training: changes the input distribution between training and the raw test images, needs mask labels or retraining, and the evidence does not show it helps. It is disclosed as a limitation.
- Per-arm tuned transforms: would mix augmentation effects with split-policy effects.
- Center crop as the default: can remove edge lesions on non-square images.

## Consequences

- About 45 extra CNN fits for the probe; [issue 114](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/114) times them before the [issue 106](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/106) budget is approved.
- The methods section names the mask-rating model. The project's no-AI-attribution rule covers authorship and prose, not disclosed research tools.
- Letterbox, flips and jitter values have no direct ablation; the manuscript labels them prespecified judgments. Pad fraction is recorded per class and source because padding can become a source cue.
- Claims about background reliance stay within the boundaries in [issue 108](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108#issuecomment-6545452441).

## Amendment, 2026-11-01

The owner's blind check grows from 50 to about 120 overlays, and the agreement gate moves from the point estimate to the lower bound of the 95% confidence interval for Cohen's κ (≥ 0.6). With 50 items the interval is roughly ±0.2, too wide to show the threshold is met. The estimated fail-rate condition (under 10%) and the decision to measure rather than mitigate the confound are unchanged. The owner rates mask quality and scene tag on the same images in one sitting. Recorded in [issue 106](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/106).

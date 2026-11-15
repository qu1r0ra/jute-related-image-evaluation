# Synthetic design review

This file records synthetic interface evidence. `PRODUCT.md` owns product constraints, `DESIGN.md` summarizes the current CSS, and `README.md` owns operation and required owner-viewer provenance. No real study images or ratings were read or used for this review.

## Fable recheck

- Finish now arms a warning and exposes a separate confirmation button. A double-click or repeated Enter on Finish cannot confirm; Enter on the separately focused confirmation can. Escape or a changed rating disarms the confirmation.
- Pair panels are equal-sized and taller at desktop widths. The browser regression checks 1440 × 900 and 1920 × 1080; at 1000 × 700, it checks that both panels, all four rating buttons, and sticky navigation fit in the viewport. Definitions may extend below the fold at that compact size.
- The shared 1×–4× slider scales both images in fit preview and full-resolution modes. Zoom retains the normalized viewed point where the new scroll range allows it, and pair panning remains synchronized. The optional 1:1 image-pixel view remains distinct.
- Fit preview may enlarge small previews to fill each panel according to the same aspect-preserving rule. This does not add detail; the explicit 1:1 option remains available for source-pixel inspection. The tradeoff is documented in `README.md` and `DESIGN.md`.
- A failed save stores a one-time recovery position. Once reload has fetched and rendered that row's current value from the sitting, the marker is cleared and the user sees a review message. Navigation or another reload no longer repeats a stale recovery warning.
- Successful save/navigation returns focus to the initiating control when it remains available and focus has not moved. A keyboard rating shortcut from the page body moves focus to the selected rating. Navigation moves to the item heading when the previous control no longer exists. Image, save, item-load and finish failures identify the failed action and the next recovery step; a confirmed finish remains visibly read-only, including after reload.

## Verification evidence

- `uv run python -m pytest tests/data/test_rating_applet.py -q -p no:cacheprovider`: **33 passed**. The suite uses generated images, a temporary synthetic kit and sitting, and Playwright Chromium 159.0.8154.6. Browser assertions cover the finish confirmation regression, Escape/rating-change cancellation, save and image failures, stale recovery clearing, focus restoration, preview zoom, pan retention, equal panels, and the three viewport sizes above.
- `uv run ruff check src/jute_disease/rating_applet/__init__.py tests/data/test_rating_applet.py`: passed.
- `node --check src/jute_disease/rating_applet/static/app.js`: passed.
- The applet-local Impeccable detector, `.agents/skills/impeccable/scripts/impeccable.cmd detect --json static/`, returned no findings. The design sidecar JSON also parsed successfully.

## Limits

The detector is a static scan and does not certify accessibility or interaction quality. The browser suite uses Chromium at its test viewport and synthetic assets; it does not establish screen-reader behavior, other browser rendering, physical monitor scaling, or how a real owner sitting will look under its actual viewing conditions. For each real sitting, capture the instrument version, browser and version, display resolution, viewport, browser zoom and operating-system display scaling in study methods, as required by [#235](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/235) and [#237](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/237).

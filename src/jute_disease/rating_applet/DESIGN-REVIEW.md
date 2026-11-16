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

- The browser suite uses generated images, a temporary synthetic kit and sitting, and Playwright Chromium. Pair tests cover the ordered display labels, keyboard adoption, keyboard ratings, image inspection and transformations. The review flow is flagged and resolved by keyboard; the resolution textarea has an accessible label. Existing browser checks cover finish confirmation, save and image failures, recovery, focus behavior, synchronized zoom and pan, equal panels, and desktop and compact viewport layouts.
- Focused synthetic tests across the applet, audit kit collection, and audit CLI: **59 passed** (`uv run python -m pytest tests/data/test_rating_applet.py tests/data/test_audit_kit_scenes.py tests/integration/test_audit_command.py tests/integration/test_audit_kit_command.py -q -p no:cacheprovider --tb=short`).
- The 1000 × 700 applet layout check asserts both pair panels, all four rating buttons, and sticky navigation fit in the viewport. Definitions remain available below the fold.
- Applet-local Impeccable detector (`detect --json static/index.html static/app.js static/style.css`): **no findings** (`[]`), run once after UI edits.
- The design sidecar is documentation of the current stylesheet, not an accessibility certification.

## Limits

The detector is a static scan and does not certify accessibility or interaction quality. The browser suite uses Chromium at its test viewport and synthetic assets; it does not establish screen-reader behavior, other browser rendering, physical monitor scaling, or how a real owner sitting will look under its actual viewing conditions. No real ratings or study images were used. For each real sitting, capture the instrument version, browser and version, display resolution, viewport, browser zoom and operating-system display scaling in study methods, as required by [#235](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/235) and [#237](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/237).

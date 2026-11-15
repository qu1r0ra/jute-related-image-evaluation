# Synthetic design review

This file owns evidence from synthetic interface review. `PRODUCT.md` owns product constraints, `DESIGN.md` summarizes the current CSS, and `README.md` owns operation and required owner-viewer provenance. No real study images, ratings or protected data were used.

## Review findings

- Pair panels have equal sizing, opaque gray wells and independent scrolling. Choice labels, rubric text and sitting navigation remain readable at the reviewed desktop viewport.
- Keyboard review covered visible focus, named choices, pressed states, save/progress status, image-load gating, and failed-save recovery.
- Synthetic browser behavior covered keyboard choices, undo, full-resolution inspection, synchronized zoom, actual-pixel sizing, swap, reset on item change, mixed pair/scene/mask rows, finish and read-only reopening.
- HTTP checks covered rating allowlists, opaque image routes, path containment, no-store responses and invalid requests.
- A Windows CLI smoke test used a synthetic pair sitting and pool root; it checked the page, choice API, image routes, no-store headers, save and finish.

## Detector and browser evidence

- Impeccable 4.9.1 `detect` on `static/` reported no findings. The `npm run design-check` target scans the two CSS/JS files with regex rules; its output says parsed-page rules were not run. Treat this as a scoped static check, not a complete accessibility or usability audit.
- Synthetic browser interaction review: Chromium 159.0.8154.6 at 1200 × 900. It checked keyboard shortcuts, accessible choice names, pressed states, focus, save/progress status, readable-image gating and failed-save recovery.
- Windows visual review: Chrome 157 at 1200 × 900. Pair frames fit their columns and sitting navigation remained in view. The accessibility snapshot showed image and rating group names, generic image alt text, labeled zoom and sitting navigation; no console errors were observed.
- Python Playwright browser coverage and the Windows CLI smoke use generated fixtures. These checks do not establish the viewing conditions of a real owner sitting.

The review aid is installed under this applet. No repository-wide Impeccable hook or automatic worker dispatch is configured. For each real owner sitting, capture instrument version, browser and version, display resolution, browser viewport size, browser zoom and operating-system display scaling in study methods as required by [#235](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/235) and [#237](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/237).

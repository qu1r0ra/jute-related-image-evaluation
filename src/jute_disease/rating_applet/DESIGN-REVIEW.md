# Synthetic usability and accessibility review

Impeccable 4.9.1 (npm `impeccable`, engine 0.1.24) was inspected in an isolated temporary directory with `--no-hooks`. Context and the new-work, craft-floor and Operate references were used for this applet. No hooks, automatic dispatch, runtime dependencies, or repository-wide configuration were added. Run a scoped manual check from this directory with `npx impeccable@4.9.1 detect static/`. The scoped detector returned no findings for `static/`. `npm run design-check` scanned two CSS/JS files with regex-only checks and warned that parsed-page rules were not run; this result is not treated as a complete audit. Do not provide real study images or ratings to design tools.

## Findings and changes

The owner delegated routine design decisions in the task brief. The interface uses neutral surroundings and equal-weight choices; saving feedback sits beside the choices. Synthetic browser checks cover keyboard choices, undo, zoom, failed image loading, delayed and failed saves, recovery by reload, finishing, and read-only reopening. The image element is created only after the opaque route is known, with a generic alt label, so the initial document does not expose a guessed item URL or render a broken-image box.

Impeccable 4.9.1 scanned the running app with a synthetic scene and image. Its first static scan flagged the image with no initial `src`; moving image creation until after the sitting item loads resolved that finding. The running-page scan and the scoped static scan then reported no findings. The browser interaction review used synthetic fixtures in Chromium 159.0.8154.6 at 1200 × 900. It covered keyboard shortcuts, visible choice names, pressed states, focus outline, live save/progress status, readable-image gating, and failed-save recovery. No real images or ratings were used. No Fable recommendation artifact was present in this checkout or in issue #235, so the review followed the tracked `PRODUCT.md` and the issue's confirmed interaction and blinding constraints.

The first pair-layout browser review found that the hidden single-image frame still reserved height because `.image-frame { display: flex }` overrode the browser's hidden styling. At 1200 × 900 this pushed the pair content and navigation below the viewport. It also found that fixed panel height plus `aspect-ratio` could make frames wider than their columns and overlap the rating area. The app now honors `[hidden]` and sizes desktop pair frames from their available width. Browser assertions cover both hidden states and frame containment.

After those fixes, a Windows Chrome 157 review at 1200 × 900 showed two equal, opaque gray image panels (321 × 240/241 CSS pixels), readable controls and rubric, and navigation within the viewport. No console errors were present. The browser accessibility snapshot showed a named image region, a named rating group, generic image alt text, visible choice names and pressed states, a labeled zoom slider, and sitting navigation. Text and controls use dark foregrounds on the neutral gray surround or white surfaces; buttons are 46 px high and the focus ring is visible. The pair flow keeps identity, path, pool, and distribution metadata out of the page.

The synthetic 1440 × 900 browser view was previously inspected: the initial image frame pushed navigation below the viewport, so its height was reduced. The current 1200 × 900 view keeps the choice and navigation controls visible without scrolling. Context7 documentation for Flask 3.1.2 and Playwright Python informed the local-request and browser-routing checks. The spec's separate finish reviewer was not launched because the worker was explicitly given no descendants and no separate review; the integration owner retains that review.

Synthetic browser behavior checks cover the four keyboard-selectable ratings, save failure and reload recovery, undo, full-resolution loading, synchronized zoom, 1:1 display, swap, reset on item change, transition from pair to scene to mask items, required-field gating, finish, and read-only reopening. HTTP checks cover the rating allowlist, opaque image routes, path containment, no-store responses, and invalid requests. No study data or protected ratings were read or used.

## Evidence

- `impeccable detect --json static/`: `[]`.
- `npm run design-check`: two CSS/JS files scanned; tool reports that parsed-page rules were not run.
- Python Playwright Chromium browser test: keyboard choice, save failure/reload, undo, zoom/swap, mixed pair/scene/mask sitting, finish, and read-only reopening passed.
- Windows CLI smoke: launched a synthetic pair sitting with `--pool-root A=...`; HTML, four-choice API, both image routes, no-store headers, save, and finish passed.
- Browser inspection: Chrome 157, Windows, 1200 × 900 CSS viewport; pair panels fit their columns and navigation stayed in view after the layout fix.
- All fixtures used locally generated synthetic images and ratings.

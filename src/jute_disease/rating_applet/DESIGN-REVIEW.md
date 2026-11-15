# Synthetic usability and accessibility review

Impeccable 4.9.1 (engine 0.1.24) was installed and run only from this applet directory. No hooks, automatic dispatch, runtime dependencies, or repository-wide configuration were added. The scoped detector returned no findings for `static/`. `npm run design-check` scanned two CSS/JS files with regex-only checks and warned that parsed-page rules were not run; this result is not treated as a complete audit.

## Findings and changes

The first synthetic browser review found that the hidden single-image frame still reserved height because `.image-frame { display: flex }` overrode the browser's hidden styling. At 1200 × 900 this pushed the pair content and navigation below the viewport. It also found that fixed panel height plus `aspect-ratio` could make frames wider than their columns and overlap the rating area. The app now honors `[hidden]` and sizes desktop pair frames from their available width. Browser assertions cover both hidden states and frame containment.

After those fixes, a Windows Chrome 157 review at 1200 × 900 showed two equal, opaque gray image panels (321 × 240/241 CSS pixels), readable controls and rubric, and navigation within the viewport. No console errors were present. The browser accessibility snapshot showed a named image region, a named rating group, generic image alt text, visible choice names and pressed states, a labeled zoom slider, and sitting navigation. Text and controls use dark foregrounds on the neutral gray surround or white surfaces; buttons are 46 px high and the focus ring is visible. The pair flow keeps identity, path, pool, and distribution metadata out of the page.

Synthetic browser behavior checks cover the four keyboard-selectable ratings, save failure and reload recovery, undo, full-resolution loading, synchronized zoom, 1:1 display, swap, reset on item change, transition from pair to scene to mask items, required-field gating, finish, and read-only reopening. HTTP checks cover the rating allowlist, opaque image routes, path containment, no-store responses, and invalid requests. No study data or protected ratings were read or used.

The provided Fable UX guidance was applied: neutral gray surround, equal-weight choices, blind presentation, stable layout, no-store assets, and swap/zoom reset on each item. Decisions about the remaining usability details followed the applet's `PRODUCT.md` and established workflow.

## Evidence

- `impeccable detect --json static/`: `[]`.
- `npm run design-check`: two CSS/JS files scanned; tool reports that parsed-page rules were not run.
- Python Playwright Chromium browser test: keyboard choice, save failure/reload, undo, zoom/swap, mixed pair/scene/mask sitting, finish, and read-only reopening passed.
- Windows CLI smoke: launched a synthetic pair sitting with `--pool-root A=...`; HTML, four-choice API, both image routes, no-store headers, save, and finish passed.
- Browser inspection: Chrome 157, Windows, 1200 × 900 CSS viewport; pair panels fit their columns and navigation stayed in view after the layout fix.
- All fixtures used locally generated synthetic images and ratings.

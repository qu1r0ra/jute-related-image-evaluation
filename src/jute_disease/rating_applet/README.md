# Local blind rating instrument

The app opens an existing sitting sheet in a local browser. It does not create sheets, collect ratings, or run statistical gates. Use synthetic data for development and review; for study use, the owner opens the existing sheet in the reviewed instrument and retains its prior values.

`PRODUCT.md` owns product and protocol constraints. `DESIGN.md` summarizes the current CSS, and `DESIGN-REVIEW.md` records synthetic review evidence and its limits. Applet developer instructions are in [AGENTS.md](AGENTS.md).

## Launch

From the repository root, open a sitting created by `jute-disease data audit-sitting`:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv
```

Pair images are resolved relative to the pool roots configured for that kit. Add one `--pool-root POOL=PATH` argument for every pool represented in the sitting; repeat the option as needed:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv `
  --pool-root A=synthetic/pools/A `
  --pool-root B=synthetic/pools/B `
  --pool-root C=synthetic/pools/C
```

The example roots are placeholders for directories containing the kit-listed synthetic pair images. The `data masks-fixture` workflow creates synthetic mask-audit evidence; it does not create an applet sitting or pair-image roots. Applet tests build their own temporary synthetic kits and roots in `tests/data/test_rating_applet.py`.

The command reports the local URL. Open it on the same Windows computer and leave the terminal running. The app binds to `127.0.0.1`, uses the bundled Python assets, and makes no runtime network requests. `--port` selects another local port if 8765 is occupied. Relative `--pool-root` paths resolve from the launch directory. Configure each pair pool named by the sitting; the app serves only the two paths listed on each pair row, after checking containment under that pool's root.

## Rate a sitting

The item order comes from the sheet. The header shows only its position, total, completed count and remaining count. The instrument does not sort, suggest answers, show rating distributions or expose paths, item IDs, pool names or labels.

- A mask item has independent **Mask quality** (`pass`, `minor`, `fail`) and **Scene** (`clean-background`, `field`, `unknown`) choices, with the tracked rubric anchors shown beside the options.
- A scene item shows only its scene choices and anchors.
- A pair item shows two equal-sized image panels and only the four choices **Exact**, **Near-duplicate**, **Very similar** and **Different**. Its agreed boundary is visible: near-duplicate means “the same physical leaf or scene captured again, such as a burst frame, a small shift or zoom, a crop or a re-save”; very similar means “a different leaf or scene that merely looks alike.” Exact and near-duplicate mean merge; very similar and different mean no merge.

Buttons and shortcuts are equally available. Keys `1`–`3` select mask quality, `Q`/`W`/`E` select scene, and `1`–`4` select pair ratings. A shortcut is ignored while a zoom control or text entry has focus. Each selection is saved immediately. Wait for **Saved** before continuing; **Saving** blocks more input, and a failed save blocks navigation until the page is reloaded and its confirmed state is checked. The app never auto-advances.

Use **Inspect full resolution** to load the original, then choose **Actual pixels**. It sets each image to the browser's intrinsic `naturalWidth` at scale 1, in CSS pixels. This is a density-corrected CSS size; browser zoom, device scale and display scaling affect its physical on-screen size. Both panels remain independently scrollable. The pair **Zoom** slider scales both images together. **Swap image order** changes only which panel is presented first. The pair identity, routes, and saved rating do not change. Swap and zoom reset when the next item opens.

Use **Previous** to revisit a row and **Undo last choice** to reverse the latest saved choice on the current row. **Next** is available only after all required fields are saved and every required image loads. The progress indicator reports counts, never rating distributions. Reopening an unfinished sheet starts at its first incomplete row. A failed or missing image blocks advancing.

## Finish, recover, and collect

After reviewing corrections, choose **Finish sitting**. It becomes read-only and the app displays a break message. An existing collected owner file also makes a sitting read-only. The app records a private backup, sheet digest, prior complete-item identities, instrument version, and append-only change history before adopting values. A save uses atomic sheet replacement and a private transaction journal; restart reconciles a pending transaction or presents a recovery error without claiming an unconfirmed choice was saved.

If a genuine entry correction is needed after finishing, stop the app and explicitly authorize it from PowerShell:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv `
  --reopen-reason "Correct an entry error"
```

The reason is recorded. This cannot reopen a sitting after collection. Resume the same unfinished sheet by launching it again. Use the existing `jute-disease data audit-sitting` command to create the next sheet and `jute-disease data audit-collect` to collect completed sheets. The app does neither operation.

Backups, journals, finish records and preview caches live under `.rating-applet/<sitting filename>/`, outside the collector's `sitting-*.csv` discovery pattern. Keep those records with the sitting. Previews are derived, content-addressed images capped at 1400 × 1400 pixels; originals are preserved. Image routes and bundled app responses use `Cache-Control: no-store`.

## Viewer provenance

This instrument is version **1.1.0**. Pair comparison uses equal panels, aspect-preserving previews capped at 1400 × 1400 pixels, full-resolution inspection, synchronized 1×–4× zoom, and an explicit one-source-pixel-per-CSS-pixel option.

For every real owner sitting, record the instrument version, browser and version, display resolution, browser viewport size, browser zoom and operating-system display scaling in study methods. Record any other viewer configuration that changes how image detail appears. This capture is required by issues [#235](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/235) and [#237](https://github.com/qu1r0ra/jute-related-image-evaluation/issues/237); synthetic browser review does not substitute for it. Review evidence and detector limits are in [DESIGN-REVIEW.md](DESIGN-REVIEW.md).

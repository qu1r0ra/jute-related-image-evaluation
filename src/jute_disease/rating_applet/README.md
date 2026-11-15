# Local blind rating instrument

The app opens an existing sitting sheet in a local browser. It does not create sheets, collect ratings, or run statistical gates. Work only from the synthetic fixtures during development; for study use, the owner opens the existing sheet in this reviewed instrument and keeps its prior values.

## Launch

From the repository root, open a sitting created by `jute-disease data audit-sitting`:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv
```

Pair images are resolved relative to the pool roots configured for that kit. Add one `--pool-root POOL=PATH` argument for every pool represented in the sitting; repeat the option as needed:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv `
  --pool-root A=data/pools/A `
  --pool-root B=data/pools/B `
  --pool-root C=data/pools/C
```

The command reports the local URL. Open it on the same Windows computer and leave the terminal running. The app binds to `127.0.0.1`, uses the bundled Python assets, and makes no runtime network requests. `--port` selects another local port if 8765 is occupied. Do not configure a root from a different pool than its kit metadata; the app serves only the two paths listed on each pair row, after checking containment under that pool's root.

## Rate a sitting

The item order comes from the sheet. The header shows only its position, total, completed count and remaining count. The instrument does not sort, suggest answers, show rating distributions or expose paths, item IDs, pool names or labels.

- A mask item has independent **Mask quality** (`pass`, `minor`, `fail`) and **Scene** (`clean-background`, `field`, `unknown`) choices, with the tracked rubric anchors shown beside the options.
- A scene item shows only its scene choices and anchors.
- A pair item shows two equal-sized image panels and only the four choices **Exact**, **Near-duplicate**, **Very similar** and **Different**. Its agreed boundary is visible: near-duplicate means “the same physical leaf or scene captured again, such as a burst frame, a small shift or zoom, a crop or a re-save”; very similar means “a different leaf or scene that merely looks alike.” Exact and near-duplicate mean merge; very similar and different mean no merge.

Buttons and shortcuts are equally available. Keys `1`–`3` select mask quality, `Q`/`W`/`E` select scene, and `1`–`4` select pair ratings. A shortcut is ignored while a zoom control or text entry has focus. Each selection is saved immediately. Wait for **Saved** before continuing; **Saving** blocks more input, and a failed save blocks navigation until the page is reloaded and its confirmed state is checked. The app never auto-advances.

Use **Inspect full resolution** to load the original at its actual dimensions. The pair **Zoom** slider scales both images together. **Actual pixels** sets a one-to-one display scale; both panels remain independently scrollable. **Swap image order** changes only which panel is presented first. The pair identity, routes, and saved rating do not change. Swap and zoom reset when the next item opens.

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

This tracked instrument is version **1.1.0**. Pair comparison uses equal 4:3 panels, aspect-preserving previews capped at 1400 × 1400 pixels, full-resolution inspection, synchronized 1×–4× zoom, and an explicit 1:1 actual-pixel option. The browser version and display resolution used for an actual owner sitting can be recorded with study methods; synthetic browser review alone does not claim identical viewing conditions.

The scoped design aid is Impeccable 4.9.1 (engine 0.1.24), with no hooks or automatic worker dispatch. The synthetic accessibility and usability review, including its browser and viewport, is recorded in [DESIGN-REVIEW.md](DESIGN-REVIEW.md). No real study images or ratings were used in that review.

# Local rating sitting

Open an existing mask or scene sitting on Windows from the repository's Python environment:

```powershell
uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv
```

Open `http://127.0.0.1:8765/` and keep the terminal running. Use `--port` if necessary. The instrument binds to loopback and requires no external service. Pair editing is not supported in this slice.

Each choice saves immediately; wait for **Saved** before advancing. The image must load and all fields must be saved before Next becomes available. Use Previous to correct earlier entries and Undo to reverse the latest choice on the current item. Keys 1–3 select mask quality and Q/W/E select scene; shortcuts do not run while a zoom or text-entry control has focus. Inspect full resolution and the zoom slider provide detail. Closing and reopening resumes the first incomplete item.

Finish only after checking entries. The explicit Finish action makes the sitting read-only. To authorize a genuine entry correction after finishing, stop the app and run `uv run python -m jute_disease.rating_applet path/to/kit/sitting-001.csv --reopen-reason "reason for correction"`. This records the reason and cannot reopen collected output. Create the next sitting and collect finished sheets with the existing external `jute-disease data audit-sitting` and `audit-collect` commands.

Backups, journal, finish state and content-addressed 1400-pixel preview caches live under the kit's `.rating-applet` directory, outside the collector's sitting glob. Preserve them with the sheet. Preview caches are derived from image bytes and regenerate when the source changes; source images are never modified. If a save fails or recovery is requested, reload to verify the confirmed state; if recovery remains required, stop and inspect the private journal and backup before continuing. Do not delete evidence or assume a failed choice was saved. Adopt real sheets only in the reviewed instrument; development and checks use synthetic data.


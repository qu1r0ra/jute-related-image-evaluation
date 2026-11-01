# Coding and delivery

- Annotate every function return; use `|` unions and built-in generics such as `list[str]`.
- Library code raises; command entry points translate errors to exit codes.
- Ruff output is the formatting authority.
- Keep shared experiment logic in the package or CLI; notebooks are wrappers. Run changed notebooks in JupyterLab.
- Import through a subpackage's `__init__.py` public surface.
- Use snake_case modules, noun packages, and verb commands. Avoid catch-all helper modules and variant-suffix modules; represent variants as configs.
- Preserve submitted results and historical notebooks. Put corrected published evidence under `results/corrected/`.
- Ignore runtime artifacts.
- Follow the owning native spec/ticket when available; missing plans use the tracker gate in `docs/agents/issue-tracker.md`.
- Use one scoped branch and PR per implementation ticket. Require all PR checks before merge. CI runs focused tests on PRs and the full suite on Linux pushes to `main`; run relevant tests and public API checks locally for the change. Never force-push; report checks that failed or were not run.
- Merge PRs with a normal merge commit (`gh pr merge --merge`) to preserve branch commits.
- Write commits, PRs, comments, and public docs as direct professional prose credited to the owner alone, without AI attribution or emojis.
- Track guidance and ADRs. Keep private context, correspondence, editable manuscripts, local data, secrets, and runtime artifacts ignored.

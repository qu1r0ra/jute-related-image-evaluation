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
- Direct ticket work uses one scoped branch and PR per ticket. A spec run uses one integration branch and one PR: workers push scoped branches and open no PR, the run owner merges them into the integration branch, and the PR body lists each ticket as `Closes #N` once its acceptance evidence is verified, otherwise `Refs #N`. Require all PR checks and the one consolidated review before merge; a ticket closes only when that PR lands. CI runs the full suite on PRs that touch anything beyond documentation (`pyproject.toml` and `justfile` count), focused tests on docs-only PRs, and the full suite on Linux pushes to `main`. Never force-push; report checks that failed or were not run.
- Run tests as `uv run python -m pytest` in worktrees; plain `uv run pytest` can fail with a uv trampoline error. Iterate with focused tests (`uv run python -m pytest tests/<path> -q -p no:cacheprovider`; the default run skips `slow`). Run the full suite and the public API checks once before opening a PR with `just test-all` (parallel, `slow` included), then let CI be the gate rather than rerunning the full suite after each fix. A spec-run worker runs focused tests only, pushes its branch, and ends its turn with branch, head SHA, and focused-test result; the run owner runs the full suite once on the integration branch and sends failures back to the owning worker. Re-run a failing test alone to tell load from a defect. Mark a test `serial` when it depends on wall-clock timing, concurrent builders, locks, or signals; `serial` tests run after the parallel pass. Pytest keeps `tmp_path` directories only for failed tests (`tmp_path_retention_policy = "failed"`), because retained training-test output has filled the disk.
- Pass the pool, view, and policy explicitly in every test fixture. A default such as `frozen_pool` freezing Pool C passes until a gate binds the pool, then breaks every fixture that relied on it.
- Merge PRs with a normal merge commit (`gh pr merge --merge`) to preserve branch commits.
- After a PR merges, remove its clean worktree with `git worktree remove` (never `--force`), then delete its local branch with `git branch -d` once no worktree holds it. A worktree with ignored or untracked content needs the owner's direction first. Wait for the removal command's final result before judging the path: a large `.venv` makes removal slow, and a second delete started meanwhile collides with it. Report the removed path and merged PR.
- Write commits, PRs, comments, and public docs as direct professional prose credited to the owner alone, without AI attribution or emojis.
- Track guidance and ADRs. Keep private context, correspondence, editable manuscripts, local data, secrets, and runtime artifacts ignored.

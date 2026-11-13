# Continuous integration

Pull requests to `main` always run lint and formatting checks, the Linux helper
and lock-resolution checks, and the synthetic-freeze smoke. A change-detection
job then picks the test path. A PR whose changed files are all documentation
(`docs/`, top-level Markdown files and the pull request template) runs a focused
set of behavior tests: the fast path. Any other PR, including source, tests,
configs, scripts, dependencies and workflow changes, runs the full test suite,
slow tests included, and the test job fails when any test fails. Both paths check
CPU imports and audit the locked dependencies with `pip-audit`. CI uses Linux
runners only. The fast path is intended to finish within two minutes, with three
minutes as the ceiling; the full-suite path is not bound by that ceiling. Measure
elapsed CI time from the first job start through the last job finish; report
queue time separately.

Pushes to `main` run the complete test suite on Linux. The run also verifies CPU
imports and audits the locked dependencies. The repository's GitGuardian check
remains enabled independently of this workflow. Windows support remains in the
dependency configuration; Windows checks are run locally when a change affects
that platform.

When a push to `main` has a failing CI run, pause merges, identify and fix the
failure, and resume merges after `main` is green. The focused suite is a quick
behavior gate for docs-only PRs; the full suite runs on code PRs and on `main`.

The previous five-run PR duration median was 10m28s. For a CI change, record the
before and after elapsed durations in the pull request, excluding queue time,
and state whether the three-minute ceiling was met. If the post-change GitHub
Actions run has not finished, report its duration as pending.

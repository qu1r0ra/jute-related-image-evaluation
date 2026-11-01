# Continuous integration

Pull requests to `main` run a focused set of behavior tests, lint and formatting
checks, the Linux helper and lock-resolution checks, and the synthetic-freeze
smoke. The Linux test job checks CPU imports and audits the locked dependencies
with `pip-audit`. CI uses Linux runners only. These jobs are intended to finish
within two minutes, with three minutes as the ceiling. Measure elapsed CI time
from the first job start through the last job finish; report queue time
separately.

Pushes to `main` run the complete test suite on Linux. The run also verifies CPU
imports and audits the locked dependencies. The repository's GitGuardian check
remains enabled independently of this workflow. Windows support remains in the
dependency configuration; Windows checks are run locally when a change affects
that platform.

When a push to `main` has a failing CI run, pause merges, identify and fix the
failure, and resume merges after `main` is green. The focused PR suite is a quick
behavior gate; the full suite runs on `main`.

The previous five-run PR duration median was 10m28s. For a CI change, record the
before and after elapsed durations in the pull request, excluding queue time,
and state whether the three-minute ceiling was met. If the post-change GitHub
Actions run has not finished, report its duration as pending.

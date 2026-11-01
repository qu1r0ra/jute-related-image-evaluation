# Continuous integration

Pull requests to `main` run a focused set of behavior tests, lint and formatting
checks, the Linux helper and lock-resolution checks, the synthetic-freeze smoke,
and the Windows CUDA 13.0 wheel smoke. The Linux PR job also checks CPU imports
and audits the locked dependencies with `pip-audit`. These jobs are intended to
finish within two minutes, with three minutes as the ceiling. Measure elapsed CI
time from the first job start through the last job finish; report queue time
separately.

Pushes to `main` run the complete test suite on Linux and Windows. The Linux run
also verifies CPU imports and audits the locked dependencies. The repository's
GitGuardian check remains enabled independently of this workflow.

When a push to `main` has a failing CI run, pause merges, identify and fix the
failure, and resume merges after `main` is green. Do not treat the focused PR
suite as a replacement for the full main-branch test matrix.

The previous five-run PR duration median was 10m28s. For a CI change, record the
before and after elapsed durations in the pull request, excluding queue time,
and state whether the three-minute ceiling was met. If the post-change GitHub
Actions run has not finished, report its duration as pending.

# Triage labels

- `needs-triage`: scope and route are not established.
- `needs-info`: required information is missing.
- `ready-for-agent`: scoped work is ready for an agent.
- `ready-for-human`: human judgment or action is required.
- `wontfix`: explicitly declined work.

- Use `status:blocked` separately for blocked work. The run owner removes it, with a comment, when the last blocker merges. `wayfinder:map` marks the live index; the other `wayfinder:*` labels mark decision children of a planning map and stay off execution tickets. Configure missing labels during approved tracker setup or migration; keep routing local to this repository.

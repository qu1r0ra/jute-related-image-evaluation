"""The grouping gates: fixed thresholds, the gate record and its binding to a freeze.

The audit writes one gate record per evaluated cohort. The record names the cohort,
the grouping version and Hamming threshold it was drawn under, and the inventory
digest of every pool in it. ``freeze target`` and ``freeze policies`` both read the
record back through :func:`gate_outcomes`, which recomputes every pass flag from the
recorded bounds against the constants below, so an operator-written file cannot
loosen a gate and a record cannot be reused for a freeze it was not drawn from.

The thresholds are module constants. No command accepts them as input.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jute_disease.data.audit_sampling import CONFIRMATORY_POOL
from jute_disease.data.grouping import GROUPING_THRESHOLDS

KAPPA_THRESHOLD = 0.6
WRONG_MERGE_LIMIT = 0.05
FALSE_SPLIT_LIMIT = 0.20
# The single prespecified re-freeze: the next higher threshold, for Pool C only.
REFREEZE_VERSIONS = {"dhash-v2": "dhash-v3"}
REFREEZE_TARGETS = frozenset(REFREEZE_VERSIONS.values())
ESCALATION = "escalate_to_owner"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = {
    "cohort_digest",
    "grouping_version",
    "hamming_threshold",
    "inventory_digests",
    "kappa_interval",
    "threshold",
    "pass",
    "method",
    "wrong_merge",
    "false_split",
}
_WRONG_MERGE_KEYS = {"pool", "upper", "limit", "pass"}
_FALSE_SPLIT_KEYS = {"pool", "upper", "limit", "exceeded", "evaluated_round"}


def _number(value: object, low: float, high: float) -> bool:
    return type(value) in (int, float) and low <= value <= high


def _bound(value: object) -> bool:
    return value is None or _number(value, 0, 1)


def build_gate_record(report: dict[str, Any]) -> dict[str, Any]:
    """Return the gate record for an evaluated cohort report."""
    digests = report.get("inventory_digests")
    if not digests:
        raise ValueError(
            "the cohort carries no inventory digests; resample its edges with "
            "--inventory-digest POOL=SHA256 for every pool"
        )

    def confirmatory(kind: str) -> float | None:
        entry = report[kind]["by_pool"].get(CONFIRMATORY_POOL)
        return None if entry is None else entry["upper"]

    wrong_merge = confirmatory("wrong_merge")
    false_split = confirmatory("false_split")
    return {
        "cohort_digest": report["cohort_digest"],
        "grouping_version": report["grouping_version"],
        "hamming_threshold": report["hamming_threshold"],
        "inventory_digests": dict(digests),
        "kappa_interval": report["kappa_interval"],
        "threshold": KAPPA_THRESHOLD,
        "pass": report["pass"],
        "method": report["method"],
        "wrong_merge": {
            "pool": CONFIRMATORY_POOL,
            "upper": wrong_merge,
            "limit": WRONG_MERGE_LIMIT,
            "pass": wrong_merge is not None and wrong_merge <= WRONG_MERGE_LIMIT,
        },
        "false_split": {
            "pool": CONFIRMATORY_POOL,
            "upper": false_split,
            "limit": FALSE_SPLIT_LIMIT,
            "exceeded": false_split is not None and false_split > FALSE_SPLIT_LIMIT,
            "evaluated_round": report["false_split_evaluated_round"],
        },
    }


def _validate(record: object) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise ValueError(
            "Invalid gate record schema; write it with audit-evaluate --gate-output"
        )
    version = record["grouping_version"]
    digests = record["inventory_digests"]
    interval = record["kappa_interval"]
    wrong_merge, false_split = record["wrong_merge"], record["false_split"]
    if (
        not isinstance(record["cohort_digest"], str)
        or not _SHA256.fullmatch(record["cohort_digest"])
        or version not in GROUPING_THRESHOLDS
        or record["hamming_threshold"] != GROUPING_THRESHOLDS[version]
        or type(record["hamming_threshold"]) is not int
        or not isinstance(digests, dict)
        or not digests
        or any(
            not isinstance(pool, str)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            for pool, digest in digests.items()
        )
    ):
        raise ValueError("Invalid gate record bindings")
    if record["threshold"] != KAPPA_THRESHOLD or type(record["pass"]) is not bool:
        raise ValueError("Invalid gate record: the kappa threshold is fixed")
    if interval is None:
        valid = record["pass"] is False
    else:
        valid = (
            isinstance(interval, list)
            and len(interval) == 2
            and all(_number(x, -1, 1) for x in interval)
            and interval[0] <= interval[1]
            and record["pass"] == (interval[0] >= KAPPA_THRESHOLD)
        )
    if (
        not valid
        or not isinstance(record["method"], str)
        or not record["method"].strip()
    ):
        raise ValueError("Invalid gate record agreement values")
    if (
        not isinstance(wrong_merge, dict)
        or set(wrong_merge) != _WRONG_MERGE_KEYS
        or wrong_merge["pool"] != CONFIRMATORY_POOL
        or wrong_merge["limit"] != WRONG_MERGE_LIMIT
        or not _bound(wrong_merge["upper"])
        or wrong_merge["pass"]
        != (
            wrong_merge["upper"] is not None
            and wrong_merge["upper"] <= WRONG_MERGE_LIMIT
        )
    ):
        raise ValueError("Invalid gate record wrong-merge values")
    if (
        not isinstance(false_split, dict)
        or set(false_split) != _FALSE_SPLIT_KEYS
        or false_split["pool"] != CONFIRMATORY_POOL
        or false_split["limit"] != FALSE_SPLIT_LIMIT
        or not _bound(false_split["upper"])
        or false_split["evaluated_round"] not in (1, 2)
        or false_split["exceeded"]
        != (
            false_split["upper"] is not None
            and false_split["upper"] > FALSE_SPLIT_LIMIT
        )
    ):
        raise ValueError("Invalid gate record false-split values")
    return record


def gate_outcomes(
    record: object,
    *,
    pool: str,
    grouping_version: str,
    threshold: int,
    inventory: str,
) -> dict[str, str]:
    """Bind a gate record to one pool's freeze and return what each gate decided.

    Raises ``ValueError`` when the record is malformed or was drawn under a different
    grouping version, Hamming threshold or inventory than the freeze. The outcomes
    are ``pass``, ``fail``, ``pending`` (no evidence), ``not_required`` (a gate that
    applies to Pool C only), ``refreeze_required`` (the false-split rule fired and a
    re-freeze is still open) or ``escalate_to_owner`` (it fired and none is left).
    """
    record = _validate(record)
    mismatched = [
        name
        for name, ok in (
            ("grouping version", record["grouping_version"] == grouping_version),
            ("Hamming threshold", record["hamming_threshold"] == threshold),
            ("pool", pool in record["inventory_digests"]),
            (
                "inventory digest",
                record["inventory_digests"].get(pool) == inventory,
            ),
        )
        if not ok
    ]
    if mismatched:
        raise ValueError(
            f"gate record bindings differ from the freeze: {', '.join(mismatched)}"
        )
    outcomes = {"agreement": "pass" if record["pass"] else "fail"}
    if pool != CONFIRMATORY_POOL:
        return {
            **outcomes,
            "wrong_merge": "not_required",
            "false_split": "not_required",
        }
    wrong_merge, false_split = record["wrong_merge"], record["false_split"]
    if wrong_merge["upper"] is None:
        outcomes["wrong_merge"] = "pending"
    else:
        outcomes["wrong_merge"] = "pass" if wrong_merge["pass"] else "fail"
    if false_split["upper"] is None:
        outcomes["false_split"] = "pending"
    elif not false_split["exceeded"]:
        outcomes["false_split"] = "pass"
    elif grouping_version in REFREEZE_VERSIONS:
        outcomes["false_split"] = "refreeze_required"
    else:
        outcomes["false_split"] = ESCALATION
    return outcomes


def read_grouping(target_dir: Path) -> dict[str, Any]:
    """Read a freeze's grouping receipt, checking its digest."""
    grouping = json.loads((Path(target_dir) / "grouping.json").read_text("utf-8"))
    claimed = grouping.get("digest")
    body = {key: value for key, value in grouping.items() if key != "digest"}
    recomputed = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != recomputed:
        raise ValueError("grouping receipt digest does not match its contents")
    return grouping


def refreeze_record(
    prior: dict[str, Any],
    *,
    grouping_version: str,
    record: dict[str, Any],
    pool: str,
    inventory: str,
) -> dict[str, Any]:
    """Check a Pool C re-freeze against the freeze it replaces and describe it.

    The re-freeze is allowed once: it must follow a freeze whose false-split rule
    fired at the version just below, on the same inventory, and it must be judged on
    a new cohort. A freeze that is itself a re-freeze has no further version, so a
    second failure is an escalation to the owner and a second re-freeze is refused.
    """
    version = prior.get("algorithm_version")
    if pool != CONFIRMATORY_POOL or prior.get("pool") != CONFIRMATORY_POOL:
        raise ValueError("only Pool C is re-frozen")
    if version not in REFREEZE_VERSIONS:
        raise ValueError(
            "the false-split re-freeze fires at most once and has been used: "
            "escalate to the owner"
        )
    if REFREEZE_VERSIONS[version] != grouping_version:
        raise ValueError(
            f"{version} re-freezes at {REFREEZE_VERSIONS[version]}, not "
            f"{grouping_version}"
        )
    outcomes = prior.get("gate_outcomes", {})
    if outcomes.get("false_split") != "refreeze_required":
        raise ValueError("the freeze to replace did not trip the false-split rule")
    before = prior["agreement_results"]
    if before["inventory_digests"].get(pool) != inventory:
        raise ValueError("re-freeze must keep the inventory of the freeze it replaces")
    if before["cohort_digest"] == record["cohort_digest"]:
        raise ValueError("re-freeze needs a new cohort, not the one that tripped")
    return {
        "from_grouping_digest": prior["digest"],
        "from_version": version,
        "from_cohort_digest": before["cohort_digest"],
        "false_split_upper": before["false_split"]["upper"],
    }

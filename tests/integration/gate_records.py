"""Gate records for tests, in the shape ``audit-evaluate --gate-output`` writes."""

import hashlib
from typing import Any

from jute_disease.data.gates import (
    FALSE_SPLIT_LIMIT,
    KAPPA_THRESHOLD,
    WRONG_MERGE_LIMIT,
)
from jute_disease.data.grouping import GROUPING_THRESHOLDS


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def gate_record(
    inventory: str,
    pool: str = "C",
    version: str = "dhash-v2",
    cohort: str = "cohort-1",
    kappa_lower: float = 0.7,
    wrong_merge: float | None = 0.02,
    false_split: float | None = 0.10,
) -> dict[str, Any]:
    """Build a self-consistent record; every pass flag follows from the bounds."""
    return {
        "cohort_digest": sha(cohort),
        "grouping_version": version,
        "hamming_threshold": GROUPING_THRESHOLDS[version],
        "inventory_digests": {pool: inventory},
        "kappa_interval": [kappa_lower, 0.95],
        "threshold": KAPPA_THRESHOLD,
        "pass": kappa_lower >= KAPPA_THRESHOLD,
        "method": "bootstrap",
        "wrong_merge": {
            "pool": "C",
            "upper": wrong_merge,
            "limit": WRONG_MERGE_LIMIT,
            "pass": wrong_merge is not None and wrong_merge <= WRONG_MERGE_LIMIT,
        },
        "false_split": {
            "pool": "C",
            "upper": false_split,
            "limit": FALSE_SPLIT_LIMIT,
            "exceeded": false_split is not None and false_split > FALSE_SPLIT_LIMIT,
            "evaluated_round": 1,
        },
    }

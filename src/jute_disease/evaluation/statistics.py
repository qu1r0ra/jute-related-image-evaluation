"""Agreement and interval statistics for evaluation reports."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import kendalltau

from jute_disease.evaluation.resampling import Interval


@dataclass(frozen=True)
class Agreement:
    """Unweighted Cohen kappa, paired percentile interval and binary specific agreement.

    Positive and negative agreement are reported only for 0/1 ratings.
    """

    kappa: float
    interval: Interval
    positive_agreement: float | None
    negative_agreement: float | None
    threshold: float
    passed: bool
    method: str


def wilson_interval(successes: int, total: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if total < 1 or not 0 <= successes <= total or not 0 < level < 1:
        raise ValueError("invalid binomial counts or confidence level")
    p = successes / total
    z = NormalDist().inv_cdf((1 + level) / 2)
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return Interval(
        p, float(max(0, center - radius)), float(min(1, center + radius)), level
    )


def kendall_tau(first: Sequence[float], second: Sequence[float]) -> float:
    """Kendall tau-b, with ties accounted for; reject undefined rankings."""
    result = float(kendalltau(first, second, variant="b").statistic)
    if not np.isfinite(result):
        raise ValueError("Kendall tau is undefined")
    return result


def agreement_gate(
    first: Sequence[int],
    second: Sequence[int],
    threshold: float = 0.6,
    resamples: int = 10_000,
    seed: int = 42,
) -> Agreement:
    """Unweighted kappa with a 95% paired percentile bootstrap.

    Undefined bootstrap draws are discarded.
    """
    a, b = np.asarray(first), np.asarray(second)
    if (
        a.ndim != 1
        or len(a) == 0
        or a.shape != b.shape
        or not np.issubdtype(a.dtype, np.integer)
        or not np.issubdtype(b.dtype, np.integer)
        or resamples < 1
        or not -1 <= threshold <= 1
    ):
        raise ValueError("invalid integer ratings, threshold or resample count")
    labels = np.unique(np.concatenate([a, b]))
    binary = labels.tolist() == [0, 1]
    positive = int(np.sum((a == 1) & (b == 1)))
    negative = int(np.sum((a == 0) & (b == 0)))
    a, b = np.searchsorted(labels, a), np.searchsorted(labels, b)
    categories = len(labels)

    def kappa(x, y) -> np.ndarray:
        observed = np.mean(x == y, axis=-1)
        expected = np.zeros_like(observed, dtype=float)
        for category in range(categories):
            expected += np.mean(x == category, axis=-1) * np.mean(
                y == category, axis=-1
            )
        return np.divide(
            observed - expected,
            1 - expected,
            out=np.full_like(observed, np.nan, dtype=float),
            where=expected != 1,
        )

    value = float(kappa(a, b))
    if not np.isfinite(value):
        raise ValueError("kappa is undefined")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(resamples, len(a)))
    values = kappa(a[draws], b[draws])
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("all bootstrap kappas are undefined")
    low, high = np.percentile(values, [2.5, 97.5])
    positive_agreement = negative_agreement = None
    if binary:
        discordant = len(a) - positive - negative
        pos_denom, neg_denom = 2 * positive + discordant, 2 * negative + discordant
        positive_agreement = 2 * positive / pos_denom if pos_denom else None
        negative_agreement = 2 * negative / neg_denom if neg_denom else None
    interval = Interval(value, float(low), float(high), 0.95)
    return Agreement(
        value,
        interval,
        positive_agreement,
        negative_agreement,
        threshold,
        bool(low >= threshold),
        f"95% paired percentile bootstrap; {len(values)}/{resamples} defined draws; "
        f"seed {seed}",
    )

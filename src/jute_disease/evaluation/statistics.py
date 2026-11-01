"""Agreement and interval statistics for evaluation reports."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import kendalltau

from jute_disease.evaluation.resampling import Interval


@dataclass(frozen=True)
class Agreement:
    """Binary Cohen kappa, paired percentile interval and specific agreement."""

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
    """Binary kappa and 95% paired percentile bootstrap, discarding undefined draws."""
    a, b = np.asarray(first), np.asarray(second)
    if (
        a.ndim != 1
        or len(a) == 0
        or a.shape != b.shape
        or not np.isin(a, [0, 1]).all()
        or not np.isin(b, [0, 1]).all()
        or resamples < 1
        or not -1 <= threshold <= 1
    ):
        raise ValueError("invalid binary ratings, threshold or resample count")

    def kappa(x, y) -> np.ndarray:
        observed = np.mean(x == y, axis=-1)
        px, py = np.mean(x, axis=-1), np.mean(y, axis=-1)
        expected = px * py + (1 - px) * (1 - py)
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
    positive = int(np.sum((a == 1) & (b == 1)))
    negative = int(np.sum((a == 0) & (b == 0)))
    discordant = len(a) - positive - negative
    pos_denom, neg_denom = 2 * positive + discordant, 2 * negative + discordant
    interval = Interval(value, float(low), float(high), 0.95)
    return Agreement(
        value,
        interval,
        2 * positive / pos_denom if pos_denom else None,
        2 * negative / neg_denom if neg_denom else None,
        threshold,
        bool(low >= threshold),
        f"95% paired percentile bootstrap; {len(values)}/{resamples} defined draws; "
        f"seed {seed}",
    )

"""Agreement and interval statistics for evaluation reports."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import beta, kendalltau

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


def clopper_pearson_interval(
    successes: float, total: float, level: float = 0.95
) -> Interval:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion.

    Counts may be fractional so the design-weighted interval can pass an
    effective sample size.
    """
    if total <= 0 or not 0 <= successes <= total or not 0 < level < 1:
        raise ValueError("invalid binomial counts or confidence level")
    tail = (1 - level) / 2
    lower = 0.0 if successes == 0 else beta.ppf(tail, successes, total - successes + 1)
    upper = (
        1.0
        if successes == total
        else beta.ppf(1 - tail, successes + 1, total - successes)
    )
    return Interval(successes / total, float(lower), float(upper), level)


@dataclass(frozen=True)
class StratumCounts:
    """Audited events in one design stratum: its population, sample and event count."""

    population: int
    sampled: int
    events: int

    def __post_init__(self) -> None:
        if not 1 <= self.sampled <= self.population or not (
            0 <= self.events <= self.sampled
        ):
            raise ValueError(
                "stratum needs 1 <= sampled <= population, events <= sampled"
            )


def design_weighted_interval(
    strata: Sequence[StratumCounts], level: float = 0.95
) -> Interval:
    """Design-weighted rate, sum(N_h p_h) / sum(N_h), with an exact interval.

    The interval is the Clopper-Pearson interval at the design's effective sample
    size (Korn and Graubard 1998). The effective size divides the Bernoulli variance
    by the stratified variance with finite-population correction, evaluated at
    add-half stratum rates so that all-zero strata still carry sampling error. A
    design that is a census has no sampling error, so its interval is the point.
    """
    if not strata or not 0 < level < 1:
        raise ValueError("need at least one stratum and a confidence level in (0, 1)")
    population = sum(s.population for s in strata)
    weights = [s.population / population for s in strata]
    estimate = sum(
        w * s.events / s.sampled for w, s in zip(weights, strata, strict=True)
    )
    smoothed = [(s.events + 0.5) / (s.sampled + 1) for s in strata]
    variance = sum(
        w * w * (1 - s.sampled / s.population) * q * (1 - q) / s.sampled
        for w, s, q in zip(weights, strata, smoothed, strict=True)
    )
    if variance == 0:
        return Interval(estimate, estimate, estimate, level)
    overall = sum(w * q for w, q in zip(weights, smoothed, strict=True))
    effective = min(overall * (1 - overall) / variance, sum(s.sampled for s in strata))
    exact = clopper_pearson_interval(estimate * effective, effective, level)
    return Interval(estimate, exact.lower, exact.upper, level)


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
    strata: Sequence[Hashable] | None = None,
    clusters: Sequence[Hashable] | None = None,
) -> Agreement:
    """Unweighted kappa with a 95% paired percentile bootstrap.

    Undefined bootstrap draws are discarded. With ``strata`` and ``clusters``,
    each row's cluster (images that recur across pairs) is resampled whole and
    only within its stratum, so the interval keeps the design's stratum mix and
    does not treat rows that share an image as independent.
    """
    if (strata is None) != (clusters is None):
        raise ValueError("strata and clusters must be given together")
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
    if strata is None:
        draws = rng.integers(0, len(a), size=(resamples, len(a)))
        values = kappa(a[draws], b[draws])
        method = "paired percentile bootstrap"
    else:
        if len(strata) != len(a) or len(clusters) != len(a):
            raise ValueError("strata and clusters must match the ratings")
        values = _clustered_kappas(a, b, categories, strata, clusters, resamples, rng)
        method = "paired percentile bootstrap stratified by stratum, clustered"
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
        f"95% {method}; {len(values)}/{resamples} defined draws; seed {seed}",
    )


def _clustered_kappas(
    a: np.ndarray,
    b: np.ndarray,
    categories: int,
    strata: Sequence[Hashable],
    clusters: Sequence[Hashable],
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Kappa of each resample that redraws whole clusters within every stratum."""
    cells: dict[Hashable, dict[Hashable, np.ndarray]] = {}
    for x, y, stratum, cluster in zip(a, b, strata, clusters, strict=True):
        matrix = cells.setdefault(stratum, {}).setdefault(
            cluster, np.zeros((categories, categories))
        )
        matrix[x, y] += 1
    total = np.zeros((resamples, categories, categories))
    for by_cluster in cells.values():
        matrices = np.stack(list(by_cluster.values()))
        draws = rng.integers(0, len(matrices), size=(resamples, len(matrices)))
        total += matrices[draws].sum(axis=1)
    count = total.sum(axis=(1, 2))
    observed = np.trace(total, axis1=1, axis2=2) / count
    rows, columns = (
        total.sum(axis=2) / count[:, None],
        total.sum(axis=1) / count[:, None],
    )
    expected = (rows * columns).sum(axis=1)
    return np.divide(
        observed - expected,
        1 - expected,
        out=np.full_like(observed, np.nan),
        where=expected != 1,
    )

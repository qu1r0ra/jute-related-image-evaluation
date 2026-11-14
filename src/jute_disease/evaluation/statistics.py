"""Agreement and interval statistics for evaluation reports."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import beta, hypergeom, kendalltau

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
    successes: int, total: int, level: float = 0.95
) -> Interval:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion."""
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


@dataclass(frozen=True)
class UpperBound:
    """A point estimate with a one-sided upper confidence bound on a rate."""

    estimate: float
    upper: float
    level: float


def _upper_event_bound(stratum: StratumCounts, tail: float) -> int:
    """Exact upper bound on a stratum's population event count.

    The sample is drawn without replacement, so the sampled event count X is
    hypergeometric. The bound is the largest population count D for which
    P(X <= x) is at least ``tail``. That probability falls as D rises, so
    bisection finds it. The true count exceeds the bound with probability at
    most ``tail``: this is the one-sided Clopper-Pearson construction with the
    finite population kept.
    """
    population, sampled, events = (
        stratum.population,
        stratum.sampled,
        stratum.events,
    )
    low, high = events, population - (sampled - events)
    while low < high:
        middle = (low + high + 1) // 2
        if hypergeom.cdf(events, population, middle, sampled) >= tail:
            low = middle
        else:
            high = middle - 1
    return low


def design_weighted_upper_bound(
    strata: Sequence[StratumCounts], level: float = 0.95
) -> UpperBound:
    """Design-weighted rate, sum(N_h p_h) / sum(N_h), and a conservative upper bound.

    The bound is one-sided and conservative, not exact: its coverage is at least
    ``level`` and is usually higher. Within a stratum the audited pairs are a
    simple random sample without replacement, so the sampled event count is
    hypergeometric and ``_upper_event_bound`` is an exact one-sided bound on the
    stratum's population event count D_h at any tail probability. Let m be the
    number of strata that are not censuses and alpha = 1 - level. Each such
    stratum gets tail alpha / m. By the union bound, with probability at least
    ``level`` no sampled stratum's true D_h exceeds its upper bound, and a census
    stratum has D_h equal to its sampled count exactly. On that event the true
    rate sum(D_h) / sum(N_h) is at most the sum of the upper stratum bounds
    divided by sum(N_h). The strata are independent samples, so this needs no
    variance estimate or normal approximation. The price of the union bound is
    width, which grows with m. Only the upper bound is formed: the audit asks
    whether a wrong-merge or false-split rate is small, and a one-sided bound
    spends all of alpha on that side.
    """
    if not strata or not 0 < level < 1:
        raise ValueError("need at least one stratum and a confidence level in (0, 1)")
    population = sum(s.population for s in strata)
    estimate = sum(s.population * s.events / s.sampled for s in strata) / population
    sampled_strata = sum(s.sampled < s.population for s in strata)
    tail = (1 - level) / max(sampled_strata, 1)
    upper = sum(_upper_event_bound(s, tail) for s in strata) / population
    return UpperBound(estimate, upper, level)


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

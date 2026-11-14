"""Agreement and interval statistics for evaluation reports.

The split-mean tests follow ADR 0004: every test runs on one macro-F1 value per
split, never on images, so the ten splits are the unit of replication.
"""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import beta, hypergeom, kendalltau

from jute_disease.evaluation.resampling import Interval

SMALLEST_EFFECT_OF_INTEREST = 0.02
"""The smallest macro-F1 difference worth detecting, on the 0-1 scale (2 points)."""

EQUIVALENCE_MARGIN = SMALLEST_EFFECT_OF_INTEREST
"""The recorded equivalence margin; it defaults to the smallest effect of interest."""

TEST_LEVEL = 0.95
"""Confidence level of the Welch and paired intervals; alpha is 1 - TEST_LEVEL."""


@dataclass(frozen=True)
class MeanDifference:
    """A two-sided t test of a mean difference, with its confidence interval."""

    estimate: float
    statistic: float
    df: float
    p_value: float
    interval: Interval
    method: str


@dataclass(frozen=True)
class EquivalenceResult:
    """Two one-sided tests (TOST) of a mean difference against +-margin."""

    estimate: float
    margin: float
    alpha: float
    lower_statistic: float
    upper_statistic: float
    lower_p: float
    upper_p: float
    p_value: float
    df: float
    interval: Interval
    equivalent: bool


@dataclass(frozen=True)
class PowerStatement:
    """Power of the two-sided Welch test to detect ``effect`` at the observed spread."""

    effect: float
    alpha: float
    sd_first: float
    sd_second: float
    n_first: int
    n_second: int
    df: float
    power: float
    sd_factor_interval: tuple[float, float]
    """Multipliers (low, high) of the standard error across the spread's interval."""
    power_interval: tuple[float, float]
    """Power (worst, best) at the high and low standard-error factors."""
    note: str


POWER_NOTE = (
    "The spread of the ten validation split means is treated as known in the "
    "headline power. It is an estimate on a few degrees of freedom, so power is "
    "also reported across the confidence interval for the standard deviation "
    "(chi-square interval on the Welch df); read the interval, not the point value."
)


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


def _sample(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} needs at least two finite values")
    return array


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError("confidence level must lie strictly between 0 and 1")


def _welch_parts(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float]:
    """Return the mean difference, its standard error and the Welch df."""
    a, b = _sample(first, "first group"), _sample(second, "second group")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        raise ValueError("Welch test is undefined when neither group has spread")
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return float(a.mean() - b.mean()), math.sqrt(va + vb), float(df)


def welch_test(
    first: Sequence[float], second: Sequence[float], level: float = TEST_LEVEL
) -> MeanDifference:
    """Welch's unequal-variance t test of ``mean(first) - mean(second)``.

    With sample variances s1^2, s2^2 and sizes n1, n2 (Welch 1947),

        t = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2),
        df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)),

    the Welch-Satterthwaite approximation. The p-value is two-sided and the
    interval is ``estimate +- t_{df, (1+level)/2} * se``. Each group needs two
    values and at least one needs spread.
    """
    _check_level(level)
    estimate, se, df = _welch_parts(first, second)
    return _difference(estimate, se, df, level, "Welch t test on split means")


def paired_test(
    first: Sequence[float], second: Sequence[float], level: float = TEST_LEVEL
) -> MeanDifference:
    """Paired t test: a one-sample t test of the per-split differences.

    With differences d_k = first_k - second_k, mean d, sample sd s and n pairs,
    t = d / (s / sqrt(n)) on n - 1 df. Zero spread in the differences leaves the
    statistic undefined and is rejected.
    """
    _check_level(level)
    a, b = _sample(first, "first group"), _sample(second, "second group")
    if a.shape != b.shape:
        raise ValueError("paired groups must have the same length")
    differences = a - b
    se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    if se == 0:
        raise ValueError("paired test is undefined when every difference is equal")
    return _difference(
        float(differences.mean()),
        se,
        len(differences) - 1,
        level,
        "paired t test on split means",
    )


def _difference(
    estimate: float, se: float, df: float, level: float, method: str
) -> MeanDifference:
    statistic = estimate / se
    p_value = float(2 * scipy_stats.t.sf(abs(statistic), df))
    radius = float(scipy_stats.t.ppf((1 + level) / 2, df)) * se
    return MeanDifference(
        estimate,
        statistic,
        float(df),
        p_value,
        Interval(estimate, estimate - radius, estimate + radius, level),
        method,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (Holm 1979).

    With the m p-values sorted ascending as p(1) <= ... <= p(m), the adjusted
    value is max over j <= i of min(1, (m - j + 1) * p(j)). Rejecting adjusted
    values below alpha controls the family-wise error rate at alpha without any
    assumption on the dependence between the tests.
    """
    if not p_values or any(not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be nonempty and lie in [0, 1]")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def equivalence_test(
    first: Sequence[float],
    second: Sequence[float],
    margin: float = EQUIVALENCE_MARGIN,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two one-sided Welch tests of equivalence within +-``margin`` (TOST).

    Schuirmann (1987), Lakens (2017). With difference d = m1 - m2 and Welch
    standard error se, test H0: d <= -margin with t_L = (d + margin) / se and
    H0: d >= +margin with t_U = (d - margin) / se, each on the Welch df. The
    p-value is the larger one-sided p-value; equivalence is declared when it is
    below ``alpha``, which is the same as the (1 - 2 alpha) interval lying inside
    (-margin, +margin).

    Limits. With ten split means per group the standard error is large unless
    the split spread is small, so the test is weak: it often fails to establish
    equivalence even when there is no effect, and that failure is not evidence
    of an effect. Called once per pool, the tests are unadjusted across pools.
    """
    if margin <= 0 or not 0 < alpha < 0.5:
        raise ValueError("margin must be positive and alpha in (0, 0.5)")
    estimate, se, df = _welch_parts(first, second)
    lower_statistic = (estimate + margin) / se
    upper_statistic = (estimate - margin) / se
    lower_p = float(scipy_stats.t.sf(lower_statistic, df))
    upper_p = float(scipy_stats.t.cdf(upper_statistic, df))
    radius = float(scipy_stats.t.ppf(1 - alpha, df)) * se
    p_value = max(lower_p, upper_p)
    return EquivalenceResult(
        estimate,
        margin,
        alpha,
        lower_statistic,
        upper_statistic,
        lower_p,
        upper_p,
        p_value,
        df,
        Interval(estimate, estimate - radius, estimate + radius, 1 - 2 * alpha),
        bool(p_value < alpha),
    )


def split_power(
    first: Sequence[float],
    second: Sequence[float],
    effect: float = SMALLEST_EFFECT_OF_INTEREST,
    alpha: float = 0.05,
    sd_level: float = 0.95,
) -> PowerStatement:
    """Power of the two-sided Welch test to detect ``effect``, at the split spread.

    The standard error and Welch df come from the observed split means (for
    example the validation draws). Under a true difference ``effect`` the
    statistic is noncentral t with noncentrality effect / se, so

        power = P(T > t_crit) + P(T < -t_crit),   t_crit = t_{df, 1 - alpha/2}.

    Cohen (1988); G*Power gives 0.5620 for sd 1, n = 10 per group, effect 1.

    The headline value treats the spread of the validation means as known. With
    ten means per group it is an estimate, so the statement also carries the
    power when the standard error is scaled by the ``sd_level`` chi-square
    interval for a standard deviation on ``df`` degrees of freedom,
    sqrt(df / chi2_{1 - (1 - level)/2}) to sqrt(df / chi2_{(1 - level)/2}); the
    smaller spread gives the best case and the larger the worst.
    """
    if effect <= 0 or not 0 < alpha < 1 or not 0 < sd_level < 1:
        raise ValueError("effect must be positive, alpha and sd_level in (0, 1)")
    _, se, df = _welch_parts(first, second)
    critical = float(scipy_stats.t.ppf(1 - alpha / 2, df))

    def power_at(error: float) -> float:
        noncentrality = effect / error
        return float(
            scipy_stats.nct.sf(critical, df, noncentrality)
            + scipy_stats.nct.cdf(-critical, df, noncentrality)
        )

    tail = (1 - sd_level) / 2
    factors = (
        float(math.sqrt(df / scipy_stats.chi2.ppf(1 - tail, df))),
        float(math.sqrt(df / scipy_stats.chi2.ppf(tail, df))),
    )
    power = power_at(se)
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return PowerStatement(
        effect,
        alpha,
        float(a.std(ddof=1)),
        float(b.std(ddof=1)),
        len(a),
        len(b),
        df,
        power,
        factors,
        (power_at(se * factors[1]), power_at(se * factors[0])),
        POWER_NOTE,
    )


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

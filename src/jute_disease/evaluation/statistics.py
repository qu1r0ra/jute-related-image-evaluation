"""Agreement and interval statistics for evaluation reports.

The split-mean tests follow ADR 0004: every test runs on one macro-F1 value per
split, never on images, so the ten splits are the unit of replication.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import kendalltau

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

"""Statistical primitives for the Central Limit Theorem demonstration.

This module is intentionally free of any plotting or I/O so that the
functions remain pure, testable, and reusable. Everything here operates on
NumPy arrays and returns either arrays or plain Python values.

Concepts implemented:
    * Population construction
    * The sampling distribution of the mean
    * Confidence intervals for the mean
    * A frequentist coverage experiment for confidence intervals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# A single shared default generator keeps results reproducible across the
# whole package. Callers may pass their own generator to override it.
DEFAULT_RNG = np.random.default_rng(seed=42)


def build_population(
    size: int = 10_000,
    low: int = 0,
    high: int = 1_000,
    rng: np.random.Generator = DEFAULT_RNG,
) -> np.ndarray:
    """Create a deliberately non-normal (uniform) population.

    A uniform population is used on purpose: it lets us *see* the Central
    Limit Theorem's "regardless of population shape" claim, because the
    distribution of sample means still converges to a bell curve.

    Args:
        size: Number of items in the population.
        low: Inclusive lower bound of the uniform integer range.
        high: Inclusive upper bound of the uniform integer range.
        rng: NumPy random generator (injected for reproducibility/testing).

    Returns:
        A 1-D integer array of length ``size``.
    """
    # ``high + 1`` because numpy's ``integers`` upper bound is exclusive.
    return rng.integers(low, high + 1, size=size)


def sampling_distribution_of_mean(
    population: np.ndarray,
    sample_size: int,
    n_samples: int = 2_000,
    rng: np.random.Generator = DEFAULT_RNG,
) -> np.ndarray:
    """Build the sampling distribution of the mean by repeated sampling.

    We draw ``n_samples`` independent samples (with replacement) of size
    ``sample_size`` from ``population`` and record each sample's mean. The
    distribution of those means is the *sampling distribution of the mean*.

    Args:
        population: The population to draw from.
        sample_size: Number of observations in each individual sample (``n``).
        n_samples: How many samples (and therefore means) to draw.
        rng: NumPy random generator.

    Returns:
        A 1-D array of length ``n_samples`` containing the sample means.
    """
    means = np.empty(n_samples)
    for i in range(n_samples):
        sample = rng.choice(population, size=sample_size, replace=True)
        means[i] = sample.mean()
    return means


def standard_error(sigma: float, n: int) -> float:
    """Return the standard error of the mean, ``sigma / sqrt(n)``."""
    return sigma / np.sqrt(n)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval for a population mean.

    Attributes:
        mean: The sample mean (the interval's center).
        low: Lower bound of the interval.
        high: Upper bound of the interval.
        confidence: The confidence level used to build it (e.g. 0.95).
    """

    mean: float
    low: float
    high: float
    confidence: float

    def contains(self, value: float) -> bool:
        """Return ``True`` if ``value`` lies within the interval."""
        return self.low <= value <= self.high


def confidence_interval(
    sample: np.ndarray,
    confidence: float = 0.95,
    sigma: float | None = None,
) -> ConfidenceInterval:
    """Construct a confidence interval for the mean of ``sample``.

    Uses the normal approximation that the Central Limit Theorem justifies::

        CI = x_bar +/- z * (sigma / sqrt(n))

    The critical value ``z`` is computed from the normal distribution, so any
    confidence level in ``(0, 1)`` is supported (not just 90/95/99%).

    Args:
        sample: The observed sample.
        confidence: Confidence level in the open interval ``(0, 1)``.
        sigma: Population standard deviation. If ``None``, the sample standard
            deviation (with Bessel's correction, ``ddof=1``) is used instead.

    Returns:
        A :class:`ConfidenceInterval`.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    n = len(sample)
    x_bar = float(sample.mean())
    sigma = float(sample.std(ddof=1)) if sigma is None else sigma

    # Two-sided critical value: e.g. 95% confidence -> z at the 0.975 quantile.
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    margin = z * standard_error(sigma, n)

    return ConfidenceInterval(
        mean=x_bar,
        low=x_bar - margin,
        high=x_bar + margin,
        confidence=confidence,
    )


@dataclass(frozen=True)
class CoverageResult:
    """Outcome of a confidence-interval coverage experiment.

    Attributes:
        intervals: Every interval constructed during the experiment.
        n_contained: How many of them contained the true mean.
        true_mean: The population mean the intervals were checked against.
    """

    intervals: list[ConfidenceInterval]
    n_contained: int
    true_mean: float

    @property
    def n_trials(self) -> int:
        return len(self.intervals)

    @property
    def empirical_coverage(self) -> float:
        """Fraction of intervals that actually contained the true mean."""
        return self.n_contained / self.n_trials


def coverage_experiment(
    population: np.ndarray,
    sample_size: int = 100,
    n_trials: int = 1_000,
    confidence: float = 0.95,
    sigma: float | None = None,
    rng: np.random.Generator = DEFAULT_RNG,
) -> CoverageResult:
    """Repeatedly build CIs and measure how often they cover the true mean.

    This demonstrates the *frequentist* interpretation of confidence: a 95%
    confidence interval does not mean "there is a 95% chance the mean is in
    this particular interval", but rather "if we repeat this procedure many
    times, ~95% of the constructed intervals will contain the true mean".

    Args:
        population: The population to sample from.
        sample_size: Size of each sample used to build one interval.
        n_trials: Number of intervals to construct.
        confidence: Confidence level for each interval.
        sigma: Population standard deviation (defaults to the true population
            standard deviation if ``None``).
        rng: NumPy random generator.

    Returns:
        A :class:`CoverageResult` summarizing the experiment.
    """
    true_mean = float(population.mean())
    sigma = float(population.std()) if sigma is None else sigma

    intervals: list[ConfidenceInterval] = []
    n_contained = 0
    for _ in range(n_trials):
        sample = rng.choice(population, size=sample_size, replace=True)
        ci = confidence_interval(sample, confidence=confidence, sigma=sigma)
        intervals.append(ci)
        if ci.contains(true_mean):
            n_contained += 1

    return CoverageResult(
        intervals=intervals,
        n_contained=n_contained,
        true_mean=true_mean,
    )

"""Visualization helpers for the Central Limit Theorem demonstration.

Each function takes the data it needs plus an optional Matplotlib ``Axes``
(or builds its own figure) and returns the ``Axes``/``Figure`` so callers can
either display the result interactively or save it to disk.

Keeping plotting separate from :mod:`clt.core` means the statistics can be
unit-tested and reused without importing a plotting backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .core import (
    ConfidenceInterval,
    sampling_distribution_of_mean,
    standard_error,
)

# Apply a clean, consistent theme to every figure in this module.
sns.set_theme(style="whitegrid")


def plot_population(population: np.ndarray) -> plt.Figure:
    """Plot the population distribution with its mean marked.

    The point is to show that the population is clearly *not* normal, so that
    the emergence of normality in the sampling distribution is striking.
    """
    pop_mean = population.mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(population, bins=50, ax=ax, color="steelblue")
    ax.axvline(pop_mean, color="red", linestyle="--", label=f"μ = {pop_mean:.2f}")
    ax.set_title("Population distribution (uniform — clearly not normal)")
    ax.set_xlabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_single_sample(ax: plt.Axes, sample: np.ndarray, title: str) -> None:
    """Draw one sample's density with its mean and ±1 std bands."""
    mean = np.mean(sample)
    std = np.std(sample)
    sns.kdeplot(sample, ax=ax, fill=True, color="skyblue")
    ax.axvline(mean, color="red", linestyle="--", label=f"sample mean = {mean:.2f}")
    ax.axvline(mean + std, color="green", linestyle=":", label="+1 std")
    ax.axvline(mean - std, color="green", linestyle=":", label="-1 std")
    ax.set_title(title)
    ax.legend()


def plot_single_samples(
    population: np.ndarray,
    sample_sizes: Sequence[int] = (5, 10, 30, 100),
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    """Show how individual samples become more representative as ``n`` grows.

    With very few observations a single sample's mean can land far from the
    population mean; larger samples already look more representative.
    """
    if rng is None:
        rng = np.random.default_rng()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, n in zip(axes.flat, sample_sizes):
        sample = rng.choice(population, size=n, replace=False)
        _plot_single_sample(ax, sample, f"Single sample distribution (n = {n})")
    fig.tight_layout()
    return fig


def plot_sampling_distributions(
    population: np.ndarray,
    sample_sizes: Sequence[int] = (2, 5, 30, 100),
    n_samples: int = 2_000,
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    """Plot the sampling distribution of the mean for several sample sizes.

    Annotates each panel with the observed standard error and the value the
    Central Limit Theorem predicts (``σ / sqrt(n)``) so the two can be
    compared directly. As ``n`` grows the histograms become visibly bell
    shaped even though the population is uniform.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_mean = population.mean()
    pop_std = population.std()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, n in zip(axes.flat, sample_sizes):
        means = sampling_distribution_of_mean(
            population, sample_size=n, n_samples=n_samples, rng=rng
        )
        predicted_se = standard_error(pop_std, n)

        sns.histplot(means, bins=40, kde=True, ax=ax, color="orange", stat="density")
        ax.axvline(pop_mean, color="red", linestyle="--", label=f"μ = {pop_mean:.1f}")
        ax.set_title(
            f"Sampling distribution of the mean  (n = {n})\n"
            f"observed mean = {means.mean():.2f}, observed SE = {means.std():.2f}, "
            f"predicted SE = {predicted_se:.2f}"
        )
        ax.set_xlabel("Sample mean")
        ax.legend()

    fig.tight_layout()
    return fig


def plot_confidence_intervals(
    intervals: Sequence[ConfidenceInterval],
    true_mean: float,
    max_intervals: int = 100,
) -> plt.Figure:
    """Plot a stack of confidence intervals against the true mean.

    Intervals that miss the true mean are drawn in red, illustrating that a
    small fraction of correctly constructed 95% intervals are *expected* to
    miss — that is exactly what "95% coverage" means.
    """
    shown = intervals[:max_intervals]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ci in enumerate(shown):
        color = "steelblue" if ci.contains(true_mean) else "crimson"
        ax.plot([ci.low, ci.high], [i, i], color=color, alpha=0.7)
    ax.axvline(true_mean, color="black", linestyle="--", label=f"True μ = {true_mean:.1f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Trial index")
    ax.set_title(
        f"First {len(shown)} confidence intervals\n(red = does not contain μ)"
    )
    ax.legend()
    fig.tight_layout()
    return fig

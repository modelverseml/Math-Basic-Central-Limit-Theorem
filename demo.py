"""Runnable demonstration of the Central Limit Theorem and confidence intervals.

This script runs, as a plain Python program, the four experiments that build up
the Central Limit Theorem and confidence intervals:

1. Build a deliberately non-normal (uniform) population.
2. Inspect single samples of varying sizes.
3. Build the sampling distribution of the mean and watch it become normal.
4. Construct a confidence interval and verify its frequentist coverage.

Usage::

    # Show the figures interactively:
    python demo.py

    # Save the figures to a directory instead of displaying them:
    python demo.py --save-dir figures

    # Make the run reproducible:
    python demo.py --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clt import (
    build_population,
    confidence_interval,
    coverage_experiment,
)
from clt.plots import (
    plot_confidence_intervals,
    plot_population,
    plot_sampling_distributions,
    plot_single_samples,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10_000,
        help="Number of items in the synthetic population (default: 10000).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If given, save figures here as PNGs instead of showing them.",
    )
    return parser.parse_args()


def _output(fig: plt.Figure, name: str, save_dir: Path | None) -> None:
    """Either save ``fig`` to ``save_dir`` or display it, then free it."""
    if save_dir is None:
        plt.show()
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)

    # 1. Population ---------------------------------------------------------
    population = build_population(size=args.population_size, rng=rng)
    pop_mean = population.mean()
    pop_std = population.std()
    print("Population")
    print(f"  mean (μ): {pop_mean:.2f}")
    print(f"  std  (σ): {pop_std:.2f}")
    _output(plot_population(population), "population", args.save_dir)

    # 2. Single samples -----------------------------------------------------
    _output(
        plot_single_samples(population, rng=rng),
        "single_samples",
        args.save_dir,
    )

    # 3. Sampling distribution of the mean ----------------------------------
    _output(
        plot_sampling_distributions(population, rng=rng),
        "sampling_distributions",
        args.save_dir,
    )

    # 4. A single confidence interval ---------------------------------------
    one_sample = rng.choice(population, size=100, replace=True)
    ci = confidence_interval(one_sample, confidence=0.95, sigma=pop_std)
    inside = "inside" if ci.contains(pop_mean) else "OUTSIDE"
    print("\n95% confidence interval from one sample (n = 100)")
    print(f"  sample mean: {ci.mean:.2f}")
    print(f"  95% CI:      ({ci.low:.2f}, {ci.high:.2f})")
    print(f"  true μ:      {pop_mean:.2f}  ->  {inside} the CI")

    # 5. Coverage experiment ------------------------------------------------
    result = coverage_experiment(
        population,
        sample_size=100,
        n_trials=1_000,
        confidence=0.95,
        sigma=pop_std,
        rng=rng,
    )
    print("\nCoverage experiment (1000 intervals, 95% confidence)")
    print(f"  intervals containing μ: {result.n_contained}/{result.n_trials}")
    print(f"  empirical coverage:     {result.empirical_coverage:.1%}")
    _output(
        plot_confidence_intervals(result.intervals, result.true_mean),
        "confidence_intervals",
        args.save_dir,
    )


if __name__ == "__main__":
    main()

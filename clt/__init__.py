"""Central Limit Theorem & confidence intervals — a hands-on demonstration.

The package is split into two modules:

* :mod:`clt.core`  — pure statistical functions (no plotting, no I/O).
* :mod:`clt.plots` — Matplotlib/Seaborn visualizations built on top of them.

See :mod:`central_limit_theorem` (the top-level script) for a runnable demo
that ties everything together.
"""

from __future__ import annotations

from .core import (
    ConfidenceInterval,
    CoverageResult,
    build_population,
    confidence_interval,
    coverage_experiment,
    sampling_distribution_of_mean,
    standard_error,
)

__all__ = [
    "ConfidenceInterval",
    "CoverageResult",
    "build_population",
    "confidence_interval",
    "coverage_experiment",
    "sampling_distribution_of_mean",
    "standard_error",
]

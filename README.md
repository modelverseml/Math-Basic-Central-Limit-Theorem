# ML Mathematics Basics

A focused, code-first walkthrough of the statistical foundations behind machine
learning inference — built around the **Central Limit Theorem (CLT)** and
**confidence intervals**, with runnable simulations that let you *see* the
theory hold.

> Before reasoning about any ML model's reliability, you need a working grasp of
> **inferential statistics**: how a small sample lets us make quantified,
> confidence-bounded claims about a population we can never measure in full.

---

## Table of Contents

- [Why this repo](#why-this-repo)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Using the package as a library](#using-the-package-as-a-library)
- [The statistics, briefly](#the-statistics-briefly)
  - [Population vs. sample](#population-vs-sample)
  - [Sampling distribution of the mean](#sampling-distribution-of-the-mean)
  - [Central Limit Theorem](#central-limit-theorem)
  - [Confidence intervals](#confidence-intervals)
- [What the simulation shows](#what-the-simulation-shows)
- [Summary](#summary)

---

## Why this repo

Most ML tutorials jump straight into models without anchoring the reader in the
math that makes them work. This repo takes the opposite approach: build the
**statistical intuition** first, then the formulas, then code that demonstrates
the theory empirically rather than asserting it.

The code is organized as a small, importable Python package (`clt`) with a
clear separation between **pure statistics** and **visualization**, plus a
single command-line entry point that reproduces every experiment end to end.

---

## Quick start

**1. Clone and enter the repo**

```bash
git clone https://github.com/<your-username>/ml_math_basics.git
cd ml_math_basics
```

**2. Create a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the demonstration**

```bash
# Show the figures interactively
python demo.py

# Or save them to disk instead of displaying them
python demo.py --save-dir figures

# Make the run fully reproducible
python demo.py --seed 42
```

The script prints the population parameters, a single confidence interval, and
the empirical coverage of 1,000 intervals, while producing four figures.

---

## Project layout

```
ml_math_basics/
├── demo.py                     # CLI entry point — runs the full demo
├── clt/                        # Importable package
│   ├── __init__.py             # Public API re-exports
│   ├── core.py                 # Pure statistics (no plotting, no I/O)
│   └── plots.py                # Matplotlib/Seaborn visualizations
├── images/                     # Static figures used in this README
├── requirements.txt            # Runtime dependencies
└── README.md
```

The split is deliberate: `clt.core` contains only NumPy/SciPy logic, so it is
trivial to test and reuse, while `clt.plots` and the entry-point script handle
everything visual.

---

## Using the package as a library

Every experiment is exposed as a plain function, so you can drop the pieces
into your own analysis:

```python
import numpy as np
from clt import build_population, confidence_interval, coverage_experiment

rng = np.random.default_rng(seed=42)

# Build a non-normal (uniform) population.
population = build_population(size=10_000, rng=rng)

# A 95% confidence interval from a single sample.
sample = rng.choice(population, size=100, replace=True)
ci = confidence_interval(sample, confidence=0.95, sigma=population.std())
print(ci.mean, ci.low, ci.high, ci.contains(population.mean()))

# Verify the frequentist coverage of that procedure.
result = coverage_experiment(population, n_trials=1_000, confidence=0.95, rng=rng)
print(result.empirical_coverage)   # ~0.95
```

`confidence_interval` derives its critical value from `scipy.stats.norm`, so any
confidence level in `(0, 1)` is supported — not just the usual 90/95/99%.

---

## The statistics, briefly

### Population vs. sample

In practice "mean" and "variance" almost always refer to the **sample**
statistics, because computing population parameters requires the entire
population — usually infeasible in cost, memory, and time. Sampling lets us
*estimate* those parameters with a known level of confidence.

**Population statistics**

| Term                | Notation | Formula                            |
|---------------------|----------|------------------------------------|
| Population size     | N        | Number of items in the population  |
| Population mean     | μ        | (1 / N) · Σ xᵢ                     |
| Population variance | σ²       | (1 / N) · Σ (xᵢ − μ)²              |

**Sample statistics**

| Term            | Notation | Formula                             |
|-----------------|----------|-------------------------------------|
| Sample size     | n        | Number of sampled items             |
| Sample mean     | x̄        | (1 / n) · Σ xᵢ                      |
| Sample variance | s²       | (1 / (n − 1)) · Σ (xᵢ − x̄)²        |

> Sample variance divides by **(n − 1)** — *Bessel's correction* — which makes
> it an unbiased estimator of the population variance.

### Sampling distribution of the mean

A *sampling distribution* is the probability distribution of a statistic (here,
the mean) computed from **many repeated** random samples drawn from the same
population.

![Sampling distribution example](images/sampling_distribution.png)

Two properties fall out of this construction:

1. **Its mean equals the population mean:** μ<sub>x̄</sub> = μ
2. **Its standard deviation is the *standard error*:** σ<sub>x̄</sub> = σ / √n

### Central Limit Theorem

The CLT states that, **regardless of the shape of the population**, the sampling
distribution of the mean:

1. is **centered** at μ,
2. has **standard error** σ / √n, and
3. becomes approximately **normal** as the sample size grows (a common rule of
   thumb is *n ≳ 30*).

This is the result that makes inferential statistics work: we can reason about
means using the normal distribution even when the underlying data is far from
normal. Quadrupling the sample size halves the standard error (the `1/√n` law).

### Confidence intervals

Because the sampling distribution of the mean is approximately normal, we can
turn "confidence" into a range:

> **CI = x̄ ± z · (σ / √n)**

| Confidence | z      |
|------------|--------|
| 90%        | 1.645  |
| 95%        | 1.960  |
| 99%        | 2.576  |

![Confidence intervals](images/confidence_intervals.webp)

A higher confidence level widens the interval — there is a direct trade-off
between *certainty* and *precision*.

**Worked example.** For n = 100, x̄ = 3.5, σ ≈ 0.3, the standard error is
0.3 / √100 = 0.03, giving:

- **90% CI:** 3.5 ± 1.645 · 0.03 = **(3.4507, 3.5493)**
- **95% CI:** 3.5 ± 1.960 · 0.03 = **(3.4412, 3.5588)**

The 95% interval is wider — being more confident costs precision.

> **Interpretation matters.** A 95% confidence interval does *not* mean "there
> is a 95% chance μ lies in this particular interval". It means "if we repeat
> this procedure many times, ~95% of the constructed intervals will contain μ".

---

## What the simulation shows

Running `python demo.py` produces four figures:

1. **Population distribution** — a uniform population, deliberately non-normal.
2. **Single samples** — how individual samples become more representative as
   `n` grows.
3. **Sampling distributions of the mean** — for `n ∈ {2, 5, 30, 100}`; each
   panel compares the *observed* standard error against the CLT's predicted
   `σ / √n`, and the histograms visibly turn into bell curves as `n` rises.
4. **Confidence-interval coverage** — 1,000 independent 95% intervals plotted
   against the true mean; roughly 5% miss it (drawn in red), confirming the
   frequentist interpretation empirically.

---

## Summary

| Concept                | What it tells us                                                             |
|------------------------|------------------------------------------------------------------------------|
| Population vs. sample  | Why we sample at all, and how the two relate                                 |
| Sampling distribution  | How a statistic (e.g. the mean) varies across random samples                 |
| Central Limit Theorem  | Why the sampling distribution of the mean is approximately normal for n ≳ 30 |
| Confidence interval    | A range expressing how confidently a sample statistic estimates a parameter  |

Together these ideas let us make rigorous claims about a population from a
comparatively tiny sample — the foundation of nearly every ML evaluation,
A/B test, and hypothesis test.

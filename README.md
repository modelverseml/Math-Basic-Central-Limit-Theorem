# ML Mathematics Basics

A beginner-friendly walkthrough of the statistical foundations every ML practitioner should know — with worked examples and intuition first, formulas second.


> Before understanding any machine learning model, it is essential to understand some basic mathematical concepts. Two of the most important topics are **inferential statistics** and **hypothesis testing**.

---

## Table of Contents

- [ML Mathematics Basics](#ml-mathematics-basics)
  - [Table of Contents](#table-of-contents)
  - [Why this repo](#why-this-repo)
  - [Getting Started](#getting-started)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Set up a virtual environment (recommended)](#2-set-up-a-virtual-environment-recommended)
    - [3. Install dependencies](#3-install-dependencies)
    - [4. Launch the notebook](#4-launch-the-notebook)
  - [Inferential Statistics](#inferential-statistics)
    - [Population vs. Sample](#population-vs-sample)
    - [A motivating example](#a-motivating-example)
  - [Sampling Distribution](#sampling-distribution)
  - [Central Limit Theorem (CLT)](#central-limit-theorem-clt)
  - [Intuition Behind Sampling Distributions and the CLT](#intuition-behind-sampling-distributions-and-the-clt)
  - [Confidence Intervals](#confidence-intervals)
    - [Worked example](#worked-example)
  - [Summary](#summary)
  - [Repository Structure](#repository-structure)

---

## Why this repo

Most ML tutorials jump straight into models without anchoring the reader in the math that makes those models work. This repo takes the opposite approach: build up the **statistical intuition** first, then the formulas, then code that lets you *see* the theory in action.


---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ml_math_basics.git
cd ml_math_basics
```

### 2. Set up a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook central_limit_theorem.ipynb
```

---

## Inferential Statistics

When we talk about *mean* and *variance* in practice, we almost always mean **sample** mean and **sample** variance — not the population's. Why?

Because computing population parameters requires access to the entire population, which is usually impractical: it costs memory, computation, and time. Sampling lets us estimate those parameters *with a known level of confidence*, which is good enough for most real-world decisions.

### Population vs. Sample

**Population Statistics**

| Term                | Notation | Formula                        |
|---------------------|----------|--------------------------------|
| Population Size     | N        | Number of items in the population |
| Population Mean     | μ        | (1 / N) × Σ (xᵢ)               |
| Population Variance | σ²       | (1 / N) × Σ (xᵢ − μ)²          |

**Sample Statistics**

| Term              | Notation | Formula                            |
|-------------------|----------|-------------------------------------|
| Sample Size       | n        | Number of sampled items             |
| Sample Mean       | x̄        | (1 / n) × Σ (xᵢ)                    |
| Sample Variance   | s²       | (1 / (n − 1)) × Σ (xᵢ − x̄)²        |

> The sample variance divides by **(n − 1)** rather than n. This is *Bessel's correction*, and it makes the sample variance an unbiased estimator of the population variance.

### A motivating example

Imagine a food inspection team visits a factory for a surprise check. They want to verify that the ingredient quantities on the packaging match what's actually in each product. Because the factory produces thousands of items, testing every single one is infeasible.

So the team picks a **random sample**, tests it, and uses the results to make a judgment about the entire batch. If the sample meets the standards with sufficient confidence, the factory passes. Otherwise, it's flagged.

This raises two natural questions:

1. What do we mean by *"confidence"*?
2. How do we know a sample actually reflects the population?

To answer them, we need three building blocks: **sampling distributions**, the **Central Limit Theorem**, and **confidence intervals**.

---

## Sampling Distribution

A *sampling distribution* is the probability distribution of a statistic (such as the mean, standard deviation, or variance) obtained from **repeated** random samples drawn from the same population.

**Example.** Suppose we want to estimate the average height of all students at a university. Instead of measuring every student, we repeatedly draw random samples of, say, 30 students each. For each sample we compute the mean height. The distribution of *all those sample means* is the sampling distribution of the mean.

![Sampling Distribution Example](images/sampling_distribution.png)

From this construction, two important properties fall out:

1. **Mean of the sampling distribution equals the population mean**
   μ<sub>x̄</sub> = μ

2. **Standard deviation of the sampling distribution (a.k.a. Standard Error)**
   σ<sub>x̄</sub> = σ / √n

---

## Central Limit Theorem (CLT)

The Central Limit Theorem states that **regardless of the shape of the original population distribution**, if we take a sufficiently large number of random samples, the sampling distribution of the sample mean will have the following properties:

1. **Mean:** μ<sub>x̄</sub> = μ
2. **Standard Error:** σ<sub>x̄</sub> = σ / √n
3. **Shape:** For **n > 30**, the sampling distribution of the sample mean approaches a **normal distribution**, regardless of the population's shape.

This is the result that makes most of inferential statistics work. We can reason about means using the normal distribution even when the underlying data is decidedly not normal.

---

## Intuition Behind Sampling Distributions and the CLT

Suppose the population has 1,000 items. From it, we draw 5 samples of 950 observations each. Since each sample covers almost the whole population (missing only ~50 items), every sample mean will land very close to the true population mean μ. The variation across these sample means is tiny — and that variation *is* the standard error.

Put differently:

> **The spread of the sampling distribution of the mean is the error in our estimate of the population mean.**

- Large samples (e.g., n = 950) → very small error.
- Small samples (e.g., n = 1 or 2) → much larger variation across sample means.

Through repeated experiments, as the sample size increases, the sampling distribution stabilizes. Specifically, for **n > 30**, the sampling distribution of the mean looks like a normal distribution that is:

- **Centered at** the population mean (μ),
- **Spread by** the standard error (σ / √n).

This is what the CLT formalizes — and it is exactly what the notebook in this repo demonstrates with random data.

---

## Confidence Intervals

We have a sample mean. How confident are we that it actually represents the population mean — and *within what range*?

Because the sampling distribution of the mean is approximately normal for n > 30 (by the CLT), we can convert "confidence" into a range using the normal distribution's well-known properties.

![Confidence Intervals](images/confidence_intervals.webp)

**Key points:**

- A **confidence interval (CI)** is a range of values that is likely to contain the true population parameter.
- The **confidence level** (e.g., 90%, 95%, 99%) quantifies how sure we are that the interval covers the population mean.
- Higher confidence ⇒ wider interval. There is a tradeoff between *certainty* and *precision*.

The general formula (when σ is known or n is large enough that s ≈ σ):

> **CI = x̄ ± z · (σ / √n)**

Common z-values:

| Confidence | z      |
|------------|--------|
| 90%        | 1.645  |
| 95%        | 1.960  |
| 99%        | 2.576  |

### Worked example

**Given**
- Sample size (n): 100
- Sample mean (x̄): 3.5
- Sample standard deviation (s ≈ σ): 0.3

**Step 1 — Standard Error**

SE = σ / √n = 0.3 / √100 = 0.03

**Step 2 — 90% Confidence Interval (z = 1.645)**

CI = 3.5 ± 1.645 × 0.03
CI = 3.5 ± 0.04935
**CI = (3.4507, 3.5493)**

**Step 3 — 95% Confidence Interval (z = 1.960)**

CI = 3.5 ± 1.960 × 0.03
CI = 3.5 ± 0.0588
**CI = (3.4412, 3.5588)**

**Interpretation**

- **90% CI:** The population mean lies in **(3.4507, 3.5493)** with 90% confidence.
- **95% CI:** The population mean lies in **(3.4412, 3.5588)** with 95% confidence.

Notice how the 95% interval is *wider* — being more confident costs us precision.

---

## Summary

| Concept | What it tells us |
|---------|------------------|
| Population vs. Sample | Why we sample at all, and how the two relate |
| Sampling Distribution | How a *statistic* (e.g., the mean) varies across random samples |
| Central Limit Theorem | Why the sampling distribution of the mean is approximately normal for n > 30 |
| Confidence Interval | A range expressing how confidently a sample statistic estimates the population parameter |

Together, these ideas let us make rigorous claims about a population from a comparatively tiny sample — which is what almost every ML evaluation, A/B test, and hypothesis test ultimately relies on.

---

---

## Repository Structure

```
ml_math_basics/
├── README.md                       # You are here
├── requirements.txt                # Python dependencies
├── central_limit_theorem.ipynb     # Sampling + CLT simulations
└── images/
    ├── sampling_distribution.png
    └── confidence_intervals.webp
```

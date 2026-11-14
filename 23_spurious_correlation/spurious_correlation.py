"""Pitfall 23: correlating two independent but slowly changing signals.

Two synthetic signals are generated completely independently of each other. If
they are white noise, a significance test at p < 0.05 flags about 5 % of the
pairs, as it should. If they are slow (autocorrelated) or random walks, the
test flags a large share of pairs, because neighbouring samples are not
independent and the effective sample size is far smaller than the number of
samples. Differencing a random walk removes the problem.

Run:  python spurious_correlation.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import lfilter

N, N_PAIRS, ALPHA = 500, 2000, 0.05


def white(rng):
    return rng.standard_normal(N)


def ar1(rng, phi=0.95):
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(N))


def walk(rng):
    return np.cumsum(rng.standard_normal(N))


def false_positive_rate(make, rng, diff=False):
    hits = 0
    for _ in range(N_PAIRS):
        a, b = make(rng), make(rng)
        if diff:
            a, b = np.diff(a), np.diff(b)
        hits += pearsonr(a, b)[1] < ALPHA
    return hits / N_PAIRS


def main():
    rng = np.random.default_rng(0)
    print(f"{N} samples, {N_PAIRS} pairs of independent signals, test at p < {ALPHA}")
    cases = [("white noise", white, False), ("AR(1), phi 0.95", ar1, False),
             ("random walk", walk, False), ("random walk, differenced", walk, True)]
    rates = []
    for name, make, diff in cases:
        r = false_positive_rate(make, rng, diff)
        rates.append(r)
        print(f"{name:<26} flagged as correlated: {100 * r:5.1f} %")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(cases)), [100 * r for r in rates])
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[0] for c in cases], rotation=20, ha="right")
    ax.axhline(100 * ALPHA, c="k", ls=":")
    ax.set_ylabel("independent pairs flagged (%)")
    fig.tight_layout()
    fig.savefig("spurious_correlation.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 26: high correlation between two methods is not agreement.

Two methods measure the same quantity on 200 synthetic cases. In the first
setting method B only adds noise. In the second it also has a constant offset
and a proportional bias. Pearson r is about the same in both. Lin's concordance
coefficient and the Bland-Altman limits of agreement show that the second
method cannot replace the first. A second part repeats the biased setting on
500 new datasets to show the result is not a lucky draw.

Run:  python agreement_vs_correlation.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, NOISE, TOLERANCE, N_REPEATS = 200, 1.5, 5.0, 500


def make_methods(rng, slope, offset):
    truth = rng.uniform(30, 70, N)
    a = truth + rng.normal(0, NOISE, N)
    b = slope * truth + offset + rng.normal(0, NOISE, N)
    return a, b


def lin_ccc(a, b):
    cov = np.cov(a, b, ddof=0)[0, 1]
    return 2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def report(name, a, b):
    d, m = b - a, (a + b) / 2
    bias, half = d.mean(), 1.96 * d.std(ddof=1)
    trend = np.polyfit(m, d, 1)[0]
    within = np.mean(np.abs(d) <= TOLERANCE)
    r = np.corrcoef(a, b)[0, 1]
    print(f"{name}: r {r:.3f}, Lin CCC {lin_ccc(a, b):.3f}, "
          f"bias {bias:+.2f}, limits {bias - half:+.2f} to {bias + half:+.2f}, "
          f"difference changes {trend:+.3f} per unit, "
          f"within +-{TOLERANCE:g}: {100 * within:.1f} %")
    return m, d, bias, half


def main():
    rng = np.random.default_rng(0)
    settings = [("B only noisy", 1.0, 0.0), ("B biased", 1.15, 5.0)]
    print(f"{N} cases, noise sd {NOISE} in each method, tolerance +-{TOLERANCE:g}")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, (name, slope, offset) in enumerate(settings):
        a, b = make_methods(rng, slope, offset)
        m, d, bias, half = report(name, a, b)
        ax = axes[row, 0]
        ax.scatter(a, b, s=8)
        ax.plot([25, 85], [25, 85], "k--", lw=1)
        ax.set_xlabel("method A")
        ax.set_ylabel("method B")
        ax.set_title(f"{name}: r = {np.corrcoef(a, b)[0, 1]:.3f}")
        ax = axes[row, 1]
        ax.scatter(m, d, s=8)
        for y in (bias, bias - half, bias + half):
            ax.axhline(y, color="r", lw=1)
        ax.set_xlabel("mean of A and B")
        ax.set_ylabel("B - A")
        ax.set_title("Bland-Altman")
    fig.tight_layout()
    fig.savefig("agreement_vs_correlation.png", dpi=130)

    rs, ccs = [], []
    for s in range(N_REPEATS):
        a, b = make_methods(np.random.default_rng(100 + s), 1.15, 5.0)
        rs.append(np.corrcoef(a, b)[0, 1])
        ccs.append(lin_ccc(a, b))
    print(f"\n{N_REPEATS} new datasets, biased method: "
          f"r mean {np.mean(rs):.3f} (lowest {np.min(rs):.3f}), "
          f"Lin CCC mean {np.mean(ccs):.3f} (highest {np.max(ccs):.3f})")


if __name__ == "__main__":
    main()

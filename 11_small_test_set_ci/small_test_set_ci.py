"""Pitfall 11: a single accuracy number from a small test set.

Synthetic setup: a classifier whose true accuracy is exactly 80 %. We draw
many independent test sets of different sizes and look at what accuracy we
would have reported each time, and how often a 95 % confidence interval
(Wilson) actually contains the true value. A bootstrap interval is shown for
one example test set.

Run:  python small_test_set_ci.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUE_ACC = 0.80
SIZES = [20, 50, 100, 500, 2000]
N_REPEATS = 5000
Z = 1.96
rng = np.random.default_rng(0)


def wilson(k, n, z=Z):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def bootstrap_ci(correct, n_boot=2000):
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    print(f"True accuracy {TRUE_ACC:.0%}, {N_REPEATS} independent test sets per size")
    print(f"{'n':>5} {'mean':>7} {'2.5-97.5 % of reported acc':>28} {'Wilson coverage':>16}")
    spreads = []
    for n in SIZES:
        k = rng.binomial(n, TRUE_ACC, size=N_REPEATS)
        acc = k / n
        lo, hi = np.percentile(acc, [2.5, 97.5])
        cover = np.mean([wilson(ki, n)[0] <= TRUE_ACC <= wilson(ki, n)[1] for ki in k])
        spreads.append(acc)
        print(f"{n:>5} {acc.mean():>7.1%} {lo:>14.1%} to {hi:<9.1%} {cover:>15.1%}")

    # one concrete small test set, n = 50
    correct = (rng.random(50) < TRUE_ACC).astype(float)
    k = int(correct.sum())
    w = wilson(k, 50)
    b = bootstrap_ci(correct)
    print(f"One test set, n=50: accuracy {k / 50:.1%}, Wilson {w[0]:.1%} to {w[1]:.1%}, "
          f"bootstrap {b[0]:.1%} to {b[1]:.1%}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([a * 100 for a in spreads], tick_labels=[str(n) for n in SIZES], whis=(2.5, 97.5), showfliers=False)
    ax.axhline(TRUE_ACC * 100, color="gray", ls="--", lw=1)
    ax.set_xlabel("test set size")
    ax.set_ylabel("reported accuracy (%)")
    ax.set_title("Same model (true accuracy 80 %), different test sets")
    fig.tight_layout()
    fig.savefig("small_test_set_ci.png", dpi=130)


if __name__ == "__main__":
    main()

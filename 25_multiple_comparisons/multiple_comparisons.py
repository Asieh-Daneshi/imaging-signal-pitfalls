"""Pitfall 25: testing many channels (or pixels, time points) without correction.

Two synthetic groups have exactly the same distribution in every channel. A
t-test is run in each channel. With one channel, about 5 % of studies report a
"significant" channel, as designed. With 64 channels, most studies report at
least one. Bonferroni and Benjamini-Hochberg corrections bring the rate of
studies with a false finding back down. A second part adds a real effect in a
few channels to show what each correction costs in detection.

Run:  python multiple_comparisons.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

N_PER_GROUP, ALPHA, N_SIM = 20, 0.05, 2000
CHANNEL_COUNTS = [1, 4, 16, 64, 256]


def benjamini_hochberg(p, alpha):
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
    keep = np.zeros(len(p), dtype=bool)
    if passed.any():
        keep[order[: np.max(np.nonzero(passed)[0]) + 1]] = True
    return keep


def flagged(p):
    return {
        "uncorrected": p < ALPHA,
        "bonferroni": p < ALPHA / len(p),
        "bh": benjamini_hochberg(p, ALPHA),
    }


def one_study(rng, n_channels, n_real=0, effect=1.0):
    a = rng.normal(size=(N_PER_GROUP, n_channels))
    b = rng.normal(size=(N_PER_GROUP, n_channels))
    b[:, :n_real] += effect
    return flagged(ttest_ind(a, b, axis=0)[1])


def main():
    rng = np.random.default_rng(0)
    methods = ["uncorrected", "bonferroni", "bh"]
    print(f"No true difference anywhere, {N_PER_GROUP} per group, {N_SIM} studies per row")
    print("Share of studies with at least one significant channel (p < 0.05):")
    fwer = {m: [] for m in methods}
    for n_ch in CHANNEL_COUNTS:
        hits = {m: 0 for m in methods}
        for _ in range(N_SIM):
            res = one_study(rng, n_ch)
            for m in methods:
                hits[m] += res[m].any()
        for m in methods:
            fwer[m].append(hits[m] / N_SIM)
        print(f"{n_ch:>4} channels: " + ", ".join(
            f"{m} {100 * hits[m] / N_SIM:5.1f} %" for m in methods))

    n_ch, n_real = 64, 6
    print(f"\n{n_ch} channels, a real shift of 1 SD in the first {n_real}:")
    found = {m: 0 for m in methods}
    false = {m: 0 for m in methods}
    for _ in range(N_SIM):
        res = one_study(rng, n_ch, n_real=n_real)
        for m in methods:
            found[m] += res[m][:n_real].sum()
            false[m] += res[m][n_real:].sum()
    for m in methods:
        print(f"{m:>12}: real channels found {found[m] / (N_SIM * n_real):.3f}, "
              f"false channels per study {false[m] / N_SIM:.2f}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m in methods:
        ax.plot(CHANNEL_COUNTS, [100 * v for v in fwer[m]], "o-", label=m)
    ax.axhline(100 * ALPHA, c="k", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("channels tested")
    ax.set_ylabel("studies with a false finding (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("multiple_comparisons.png", dpi=130)


if __name__ == "__main__":
    main()

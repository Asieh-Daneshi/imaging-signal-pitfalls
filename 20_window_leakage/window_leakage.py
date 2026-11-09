"""Pitfall 20: overlapping windows from one recording, split at random.

One long synthetic recording is cut into overlapping windows. The label of a
window (state 0 or 1) comes from long segments, and it has NO relation to the
signal: the signal is a slowly wandering baseline plus noise. A random split of
windows puts near-identical neighbours in train and test, so a nearest-neighbour
classifier "recognises" the time point and looks good. A split in contiguous
blocks, with a gap, shows the real accuracy (chance).

Run:  python window_leakage.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

N_SAMPLES, WIN, N_BINS, SEG, N_REPEATS = 60000, 200, 20, 1500, 20
STRIDES = (200, 100, 50, 20)  # 200 = no overlap, 20 = 90 % overlap


def make_windows(stride, rng):
    base = np.cumsum(rng.standard_normal(N_SAMPLES)) * 0.05  # slow wandering baseline
    sig = base + rng.standard_normal(N_SAMPLES)
    seg_labels = rng.integers(0, 2, N_SAMPLES // SEG + 1)  # unrelated to the signal
    starts = np.arange(0, N_SAMPLES - WIN, stride)
    x = np.stack([sig[s:s + WIN].reshape(N_BINS, -1).mean(1) for s in starts])
    y = seg_labels[(starts + WIN // 2) // SEG]
    return x, y


def knn_acc(x, y, tr, te):
    return KNeighborsClassifier(1).fit(x[tr], y[tr]).score(x[te], y[te])


def run(stride, seed):
    rng = np.random.default_rng(seed)
    x, y = make_windows(stride, rng)
    n = len(y)
    perm = rng.permutation(n)
    rand_tr, rand_te = perm[: n // 2], perm[n // 2:]
    gap = WIN // stride  # drop windows that overlap the boundary
    cut = n // 2
    return (knn_acc(x, y, rand_tr, rand_te),
            knn_acc(x, y, np.arange(0, cut - gap), np.arange(cut, n)))


def main():
    print(f"{N_SAMPLES} samples, window {WIN}, labels in segments of {SEG} samples, "
          f"no link between label and signal, 1-NN, {N_REPEATS} repeats")
    res = {}
    for stride in STRIDES:
        r = np.array([run(stride, s) for s in range(N_REPEATS)]).mean(0)
        res[stride] = r
        print(f"stride {stride:>3} (overlap {100 * (1 - stride / WIN):3.0f} %): "
              f"random split {r[0]:.3f}, blocked split {r[1]:.3f}")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(STRIDES, [res[s][0] for s in STRIDES], "o-", label="random split")
    ax.plot(STRIDES, [res[s][1] for s in STRIDES], "s-", label="blocked split")
    ax.axhline(0.5, c="k", ls=":")
    ax.set_xlabel("stride (samples)")
    ax.set_ylabel("test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig("window_leakage.png", dpi=130)


if __name__ == "__main__":
    main()

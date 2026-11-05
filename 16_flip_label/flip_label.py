"""Pitfall 16: an augmentation that changes the label.

Synthetic 16x16 images with a bright blob and noise. The label says whether
the blob sits in the left half (0) or the right half (1). A horizontal flip
looks like a harmless augmentation, but here it swaps the class while the
label stays the same. A vertical flip keeps the label. We train a logistic
regression on pixels with a small training set and flip a fraction p of the
training images, then score on clean test images. Mean and standard deviation
over repeated runs.

Run:  python flip_label.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

SIZE, N_TRAIN, N_TEST, N_RUNS = 16, 60, 2000, 20
NOISE = 0.5
P_FLIP = [0.0, 0.1, 0.25, 0.5]
yy, xx = np.mgrid[0:SIZE, 0:SIZE]


def make_images(n, rng):
    y = rng.integers(0, 2, n)
    cx = np.where(y == 0, rng.uniform(2, 6, n), rng.uniform(9, 13, n))
    cy = rng.uniform(3, 12, n)
    blob = np.exp(-((xx[None] - cx[:, None, None]) ** 2 +
                    (yy[None] - cy[:, None, None]) ** 2) / (2 * 1.5 ** 2))
    return blob + NOISE * rng.standard_normal((n, SIZE, SIZE)), y


def run(axis, p, seed):
    rng = np.random.default_rng(seed)
    x_tr, y_tr = make_images(N_TRAIN, rng)
    x_te, y_te = make_images(N_TEST, rng)
    flip = rng.random(N_TRAIN) < p
    x_tr = x_tr.copy()
    x_tr[flip] = np.flip(x_tr[flip], axis=axis)  # axis 2: left-right, axis 1: up-down; label unchanged
    clf = LogisticRegression(C=0.1, max_iter=2000)
    clf.fit(x_tr.reshape(N_TRAIN, -1), y_tr)
    return clf.score(x_te.reshape(N_TEST, -1), y_te)


def main():
    print(f"{N_TRAIN} training images, {N_TEST} test images, {N_RUNS} runs, noise {NOISE}")
    print(f"{'flip share':>10} {'horizontal':>14} {'vertical':>14}")
    res = {0: [], 1: []}  # 0: left-right flip, 1: up-down flip
    for p in P_FLIP:
        h = np.array([run(2, p, s) for s in range(N_RUNS)])  # flip columns
        v = np.array([run(1, p, s) for s in range(N_RUNS)])  # flip rows
        res[0].append(h)
        res[1].append(v)
        print(f"{p:10.2f} {h.mean():8.3f}+-{h.std():.3f} {v.mean():8.3f}+-{v.std():.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for k, name in ((0, "horizontal flip (changes label)"), (1, "vertical flip (keeps label)")):
        ax.errorbar(P_FLIP, [r.mean() for r in res[k]], [r.std() for r in res[k]],
                    marker="o", capsize=3, label=name)
    ax.axhline(0.5, c="k", ls=":")
    ax.set_xlabel("share of training images flipped")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("flip_label.png", dpi=130)


if __name__ == "__main__":
    main()

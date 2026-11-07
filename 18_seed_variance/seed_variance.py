"""Pitfall 18: seed variance in a small network.

Synthetic 12x12 images: a bright blob at a random position, and the label says
whether it is elongated horizontally (0) or vertically (1).
The data split is fixed. Only the random initialisation of a small neural
network changes between runs. We report the spread of the test accuracy over
seeds, and compare two ways of reporting a single number: the best seed
chosen on the test set versus the seed chosen on a separate validation set.

Run:  python seed_variance.py
"""
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SIZE, N_TRAIN, N_VAL, N_TEST, N_SEEDS = 12, 60, 200, 200, 30
NOISE = 0.3
yy, xx = np.mgrid[0:SIZE, 0:SIZE]


def make_images(n, rng):
    y = rng.integers(0, 2, n)
    cx, cy = rng.uniform(3, 9, n), rng.uniform(3, 9, n)
    sx = np.where(y == 0, 3.0, 1.2)
    sy = np.where(y == 0, 1.2, 3.0)
    blob = np.exp(-((xx[None] - cx[:, None, None]) ** 2 / (2 * sx[:, None, None] ** 2) +
                    (yy[None] - cy[:, None, None]) ** 2 / (2 * sy[:, None, None] ** 2)))
    x = blob + NOISE * rng.standard_normal((n, SIZE, SIZE))
    return x.reshape(n, -1), y


def main():
    rng = np.random.default_rng(0)  # one fixed data split for every seed
    x_tr, y_tr = make_images(N_TRAIN, rng)
    x_va, y_va = make_images(N_VAL, rng)
    x_te, y_te = make_images(N_TEST, rng)
    val, test = [], []
    for seed in range(N_SEEDS):
        net = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=300,
                            random_state=seed)
        net.fit(x_tr, y_tr)
        val.append(net.score(x_va, y_va))
        test.append(net.score(x_te, y_te))
    val, test = np.array(val), np.array(test)
    print(f"{N_TRAIN} train, {N_VAL} validation, {N_TEST} test images, {N_SEEDS} seeds, fixed split")
    print(f"test accuracy: mean {test.mean():.3f}  std {test.std():.3f}  "
          f"min {test.min():.3f}  max {test.max():.3f}")
    print(f"best seed picked on the test set:       reported {test.max():.3f}")
    print(f"best seed picked on the validation set: reported {test[val.argmax()]:.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(test, bins=12, color="C0")
    ax.axvline(test.mean(), c="k", ls=":")
    ax.set_xlabel("test accuracy")
    ax.set_ylabel("number of seeds")
    fig.tight_layout()
    fig.savefig("seed_variance.png", dpi=130)


if __name__ == "__main__":
    main()

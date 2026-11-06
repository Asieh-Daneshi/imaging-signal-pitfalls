"""Pitfall 17: a change in image resolution at test time.

Synthetic 32x32 textures. Class 0 is smoothed noise with a small blur, class 1
is smoothed noise with a larger blur, so the classes differ in spatial scale.
A logistic regression on two gradient-energy features is trained at one
resolution. At test time the same images are resampled (zoomed) by a factor,
as happens when a scanner or an export step changes the pixel size. Scale
augmentation during training is compared with none. Mean and standard
deviation over repeated runs.

Run:  python resolution_shift.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from sklearn.linear_model import LogisticRegression

SIZE, N_TRAIN, N_TEST, N_RUNS = 32, 200, 400, 15
SIGMAS = (1.0, 1.6)
ZOOMS = [0.6, 0.8, 1.0, 1.25, 1.6]


def make_images(n, rng):
    y = rng.integers(0, 2, n)
    x = np.stack([gaussian_filter(rng.standard_normal((SIZE * 2, SIZE * 2)), SIGMAS[c])
                  for c in y])
    return x, y


def resample(img, f):
    """Zoom by f, then crop or pad (reflect) back to the central SIZE x SIZE."""
    big = zoom(img, f, order=1)
    h = big.shape[0]
    if h < SIZE:
        pad = SIZE - h
        big = np.pad(big, ((pad // 2, pad - pad // 2),) * 2, mode="reflect")
        h = SIZE
    s = (h - SIZE) // 2
    return big[s:s + SIZE, s:s + SIZE]


def features(img):
    img = (img - img.mean()) / img.std()
    gx, gy = np.diff(img, axis=1), np.diff(img, axis=0)
    lap = img[1:-1, 1:-1] * 4 - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]
    return [np.log((gx ** 2).mean() + (gy ** 2).mean()), np.log((lap ** 2).mean())]


def feats(x, zooms, rng):
    return np.array([features(resample(im, rng.choice(zooms))) for im in x])


def run(seed):
    rng = np.random.default_rng(seed)
    x_tr, y_tr = make_images(N_TRAIN, rng)
    x_te, y_te = make_images(N_TEST, rng)
    plain = LogisticRegression(max_iter=1000).fit(feats(x_tr, [1.0], rng), y_tr)
    aug = LogisticRegression(max_iter=1000).fit(feats(x_tr, ZOOMS, rng), y_tr)
    out = []
    for f in ZOOMS:
        ft = feats(x_te, [f], rng)
        out.append((plain.score(ft, y_te), aug.score(ft, y_te)))
    return out


def main():
    print(f"{N_TRAIN} training, {N_TEST} test images, {N_RUNS} runs; train at zoom 1.0")
    res = np.array([run(s) for s in range(N_RUNS)])  # runs x zooms x (plain, aug)
    print(f"{'test zoom':>9} {'trained at 1.0':>16} {'scale-augmented':>16}")
    for i, f in enumerate(ZOOMS):
        a, b = res[:, i, 0], res[:, i, 1]
        print(f"{f:9.2f} {a.mean():9.3f}+-{a.std():.3f} {b.mean():9.3f}+-{b.std():.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for k, name in ((0, "trained at zoom 1.0"), (1, "scale-augmented")):
        ax.errorbar(ZOOMS, res[:, :, k].mean(0), res[:, :, k].std(0),
                    marker="o", capsize=3, label=name)
    ax.axhline(0.5, c="k", ls=":")
    ax.set_xlabel("test-time zoom factor")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("resolution_shift.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 21: a shortcut that works in training and breaks in deployment.

Synthetic 16x16 images. The real class signal is a faint blob in the centre
(class 1) buried in noise. A bright 2x2 corner tag (think scanner label or
burned-in annotation) is also present. In the training data the tag matches the
class in 95 % of images, so a classifier mostly learns the tag. On a test set
where the tag is unrelated to the class, or reversed, accuracy drops.

Run:  python shortcut_marker.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

SIZE, N_TRAIN, N_TEST, N_REPEATS = 16, 600, 2000, 20
BLOB_AMP, NOISE = 0.5, 1.0


def make_images(n, tag_agree, rng):
    y = rng.integers(0, 2, n)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    blob = np.exp(-((yy - 7.5) ** 2 + (xx - 7.5) ** 2) / (2 * 2.5 ** 2))
    x = rng.standard_normal((n, SIZE, SIZE)) * NOISE
    x += y[:, None, None] * BLOB_AMP * blob
    tag = np.where(rng.random(n) < tag_agree, y, 1 - y)
    x[:, :2, :2] += tag[:, None, None] * 3.0  # bright corner tag
    return x.reshape(n, -1), y


def run(seed):
    rng = np.random.default_rng(seed)
    xtr, ytr = make_images(N_TRAIN, 0.95, rng)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, ytr)
    out = {}
    for name, agree in (("tag matches class (0.95)", 0.95),
                        ("tag unrelated (0.50)", 0.50),
                        ("tag reversed (0.05)", 0.05)):
        xte, yte = make_images(N_TEST, agree, rng)
        out[name] = clf.score(xte, yte)
    # reference: trained without any tag correlation, tested on the unrelated set
    xtr2, ytr2 = make_images(N_TRAIN, 0.50, rng)
    ref = LogisticRegression(C=1.0, max_iter=2000).fit(xtr2, ytr2)
    xte, yte = make_images(N_TEST, 0.50, rng)
    out["trained with tag unrelated, tested unrelated"] = ref.score(xte, yte)
    return out


def main():
    print(f"{SIZE}x{SIZE} synthetic images, blob amplitude {BLOB_AMP}, noise {NOISE}, "
          f"{N_TRAIN} training images, {N_REPEATS} repeats")
    runs = [run(s) for s in range(N_REPEATS)]
    res = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for k, v in res.items():
        print(f"{k:<48} accuracy {v:.3f}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(list(res), list(res.values()))
    ax.axvline(0.5, c="k", ls=":")
    ax.set_xlabel("test accuracy")
    fig.tight_layout()
    fig.savefig("shortcut_marker.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 07: pixel accuracy hides failure on small structures.

Synthetic setup: 128x128 images with one small round "lesion" (radius 3-5 px,
about 0.1-0.5 % of the pixels). We score four hypothetical segmenters:
empty (predicts only background), shifted by 3 px, over-segmenting (dilated)
and a good one (boundary noise only). Pixel accuracy is compared with Dice.

Run:  python dice_vs_accuracy.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

SEED = 0
SIZE = 128
N_IMAGES = 200


def make_mask(rng):
    r = rng.uniform(3, 5)
    cy, cx = rng.uniform(20, SIZE - 20, size=2)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    return 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 1.0


def pixel_acc(a, b):
    return (a == b).mean()


def predictors(mask, rng):
    shifted = ndi.shift(mask.astype(float), (3, 3), order=0) > 0.5
    dilated = ndi.binary_dilation(mask, iterations=3)
    edge = ndi.binary_dilation(mask) & ~ndi.binary_erosion(mask)
    flip = edge & (rng.random(mask.shape) < 0.3)
    good = np.logical_xor(mask, flip)
    return {
        "empty (all background)": np.zeros_like(mask),
        "shifted by 3 px": shifted,
        "over-segmented (+3 px)": dilated,
        "good (boundary noise)": good,
    }


def main():
    rng = np.random.default_rng(SEED)
    acc, dsc = {}, {}
    fracs = []
    for _ in range(N_IMAGES):
        m = make_mask(rng)
        fracs.append(m.mean())
        for name, p in predictors(m, rng).items():
            acc.setdefault(name, []).append(pixel_acc(m, p))
            dsc.setdefault(name, []).append(dice(m, p))
    print(f"Lesion covers {np.mean(fracs):.2%} of the pixels on average")
    print(f"{'predictor':26s} {'pixel acc':>10s} {'Dice':>8s}")
    for name in acc:
        print(f"{name:26s} {np.mean(acc[name]):10.1%} {np.mean(dsc[name]):8.2f}")

    names = list(acc)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [np.mean(acc[n]) for n in names], 0.4, label="pixel accuracy", color="#f59e0b")
    ax.bar(x + 0.2, [np.mean(dsc[n]) for n in names], 0.4, label="Dice", color="#38bdf8")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.set_title("Small lesion: accuracy looks great for every predictor")
    fig.tight_layout()
    fig.savefig("dice_vs_accuracy.png", dpi=130)


if __name__ == "__main__":
    main()

"""Dice score vs. object size for the same 1-pixel boundary error (synthetic data).

A circular "lesion" is the ground truth. The prediction is the same mask dilated by
one pixel. Only the size of the object changes. Run: python dice_vs_object_size.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


def disk(n, r):
    y, x = np.ogrid[:n, :n]
    c = n // 2
    return (x - c) ** 2 + (y - c) ** 2 <= r * r


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


STRUCT = np.ones((3, 3), bool)


def dice_for_radius(r, n=161):
    truth = disk(n, r)
    pred = binary_dilation(truth, STRUCT)  # prediction exactly 1 px too wide
    return dice(truth, pred)


if __name__ == "__main__":
    res = {r: dice_for_radius(r) for r in [3, 5, 10, 20, 40]}
    for r, d in res.items():
        print(f"radius {r:>2} px -> Dice {d:.2f}")

    bg, fg = "#0f172a", "#e2e8f0"
    fig, ax = plt.subplots(1, 3, figsize=(12, 6.27), dpi=100, facecolor=bg,
                           gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for a, r in zip(ax[:2], [40, 5]):
        g = disk(161, r)
        p = binary_dilation(g, STRUCT)
        img = np.zeros((161, 161, 3))
        img[g] = [.2, .6, .9]
        img[p & ~g] = [.95, .4, .3]
        a.imshow(img)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"radius {r} px\nDice = {res[r]:.2f}", color=fg, fontsize=14)
        for s in a.spines.values():
            s.set_color("#334155")
    a = ax[2]
    a.set_facecolor(bg)
    rs = np.arange(2, 61)
    a.plot(rs, [dice_for_radius(r) for r in rs], color="#38bdf8", lw=2.5)
    for r in [5, 40]:
        a.scatter([r], [res[r]], color="#f87171", zorder=3, s=50)
    a.set_xlabel("object radius (pixels)", color=fg, fontsize=12)
    a.set_ylabel("Dice", color=fg, fontsize=12)
    a.set_title("Same 1-pixel boundary error", color=fg, fontsize=14)
    a.tick_params(colors=fg)
    for s in a.spines.values():
        s.set_color("#334155")
    a.grid(alpha=.15)
    fig.suptitle("Dice depends on object size, not only on error size  (synthetic, no real data)",
                 color=fg, fontsize=15, y=.97)
    fig.text(.02, .02, "Blue: ground truth   Red: prediction dilated by 1 px", color="#94a3b8", fontsize=11)
    plt.tight_layout(rect=[0, .04, 1, .93])
    plt.savefig("dice_vs_object_size.png", facecolor=bg)

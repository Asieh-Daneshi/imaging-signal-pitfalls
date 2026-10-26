"""More Gaussian smoothing removes noise, and small structures with it.

A synthetic image with noise and one small bright spot is blurred with increasing sigma.
We track how much noise is left and how much of the spot's contrast survives.
Run: python smoothing.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(2)
size = 128
yy, xx = np.mgrid[:size, :size]
spot = np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 1.0 ** 2))   # tiny spot, sigma = 1 px
noise = rng.normal(0, 0.3, (size, size))
image = spot + noise

sigmas = [0, 1, 2, 4]
rows = []
fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 3.4))
for ax, s in zip(axes, sigmas):
    smooth = gaussian_filter(image, s) if s else image
    noise_left = (gaussian_filter(noise, s) if s else noise).std() / noise.std()
    spot_left = (gaussian_filter(spot, s) if s else spot).max() / spot.max()
    rows.append((s, noise_left, spot_left))
    ax.imshow(smooth, cmap="gray")
    ax.set_title(f"sigma={s}\nnoise {noise_left:.0%}, spot {spot_left:.0%}", fontsize=9)
    ax.axis("off")
fig.suptitle("Gaussian smoothing: noise left vs. spot contrast left")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("smoothing.png", dpi=150)

print("sigma | noise left | spot peak left")
for s, n, p in rows:
    print(f"{s:5} | {n:10.0%} | {p:14.0%}")

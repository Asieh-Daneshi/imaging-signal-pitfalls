"""Experiment 08: accuracy is not calibration.

A synthetic classifier is given overconfident scores. Temperature scaling (one parameter,
fitted on held-out data) fixes the confidence without changing the ranking of predictions.
Synthetic data only.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

SEED = 7
N = 6000          # samples per split
OVERCONFIDENCE = 2.6

rng = np.random.default_rng(SEED)
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def make_split(n):
    z = rng.normal(0, 1.6, n)                      # true log-odds
    y = (rng.random(n) < sigmoid(z)).astype(int)   # labels drawn from the true probabilities
    return OVERCONFIDENCE * z, y                   # model logits are too large


def reliability(logit, y, bins=10):
    p = sigmoid(logit)
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1, bins + 1)
    mc, ma, w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 20:
            mc.append(conf[m].mean()); ma.append(correct[m].mean()); w.append(m.mean())
    mc, ma, w = map(np.array, (mc, ma, w))
    return mc, ma, float(np.sum(w * np.abs(mc - ma)))   # last value: expected calibration error


cal_logit, cal_y = make_split(N)     # held-out split used to fit the temperature
test_logit, test_y = make_split(N)   # separate test split used for all reported numbers


def nll(T):
    p = sigmoid(cal_logit / T)
    return -np.mean(cal_y * np.log(p + 1e-12) + (1 - cal_y) * np.log(1 - p + 1e-12))


T = minimize_scalar(nll, bounds=(0.5, 6), method="bounded").x
raw = reliability(test_logit, test_y)
scaled = reliability(test_logit / T, test_y)
acc_raw = ((sigmoid(test_logit) > 0.5) == test_y).mean()
acc_scaled = ((sigmoid(test_logit / T) > 0.5) == test_y).mean()
print(f"Temperature T = {T:.2f}")
print(f"Accuracy  raw {acc_raw:.3f} | scaled {acc_scaled:.3f} (unchanged: scaling does not reorder predictions)")
print(f"ECE       raw {raw[2]:.3f} | scaled {scaled[2]:.3f}")

bg, fg, mut, c1, c2 = "#0f1720", "#e8eef5", "#8a99a8", "#ff7a59", "#3ddc97"
plt.rcParams.update({"text.color": fg, "axes.labelcolor": fg, "xtick.color": mut, "ytick.color": mut})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=100, facecolor=bg)
for ax, r, col, title in [(axes[0], raw, c1, "Raw model"), (axes[1], scaled, c2, f"Temperature scaled (T={T:.2f})")]:
    ax.set_facecolor(bg)
    ax.plot([0.5, 1], [0.5, 1], "--", color=mut, label="perfect calibration")
    ax.bar(r[0], r[1], width=0.04, color=col, alpha=0.85, label="observed accuracy")
    ax.plot(r[0], r[1], "o-", color=col)
    ax.set(xlim=(0.5, 1), ylim=(0.4, 1.02), xlabel="predicted confidence")
    ax.set_title(title, color=col, loc="left", weight="bold")
    ax.text(0.52, 0.93, f"ECE = {r[2]:.3f}", fontsize=14, weight="bold")
    ax.legend(loc="lower right", frameon=False, labelcolor=fg)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(mut)
axes[0].set_ylabel("observed accuracy")
fig.suptitle("Accurate is not the same as trustworthy (synthetic data)", color=fg, weight="bold")
fig.tight_layout()
fig.savefig("calibration.png", facecolor=bg)

"""How often is a positive screening result real? PPV depends on prevalence.

A classifier with 90 % sensitivity and 90 % specificity is applied to populations
with different disease prevalence. The model never changes; only the population does.
Synthetic data only.
Run: python screening_ppv.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
sensitivity, specificity = 0.90, 0.90


def ppv_exact(prevalence):
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)


def simulate(prevalence, n):
    """Draw n people, apply the classifier, return (TP, FP, flagged)."""
    sick = rng.random(n) < prevalence
    flagged = np.where(sick, rng.random(n) < sensitivity, rng.random(n) > specificity)
    tp = int(np.sum(flagged & sick))
    fp = int(np.sum(flagged & ~sick))
    return tp, fp, tp + fp


# 1) The example from the post: 1,000 people, 1 % prevalence (expected counts)
n, prev = 1000, 0.01
sick = round(n * prev)
tp = round(sick * sensitivity)
fp = round((n - sick) * (1 - specificity))
print(f"1,000 people, 1 % prevalence: {sick} sick, {tp} caught, {fp} healthy people flagged")
print(f"flagged: {tp + fp}, really sick: {tp}, PPV = {tp / (tp + fp):.1%}")

# 2) Same model, other populations (simulated with 1,000,000 people each)
print("\nprevalence   PPV (exact)   PPV (simulated)")
for p in [0.5, 0.1, 0.01, 0.001]:
    tp_s, fp_s, flagged_s = simulate(p, 1_000_000)
    print(f"{p:>9.1%}   {ppv_exact(p):>10.1%}   {tp_s / flagged_s:>14.1%}")

# 3) Figure
prevalences = np.logspace(-4, np.log10(0.5), 200)
marks = [0.5, 0.1, 0.01, 0.001]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(prevalences * 100, ppv_exact(prevalences) * 100, lw=2)
for p in marks:
    ax.plot(p * 100, ppv_exact(p) * 100, "o", color="k")
    ax.annotate(f"{ppv_exact(p):.1%}", (p * 100, ppv_exact(p) * 100),
                textcoords="offset points", xytext=(-42, -4) if p == 0.5 else (6, -12), fontsize=9)
ax.set_xscale("log")
ax.set_xticks([0.01, 0.1, 1, 10, 50], ["0.01", "0.1", "1", "10", "50"])
ax.set_xlabel("disease prevalence in the screened population (%)")
ax.set_ylabel("positive predictive value (%)")
ax.set_title("Same model (90 % sensitivity, 90 % specificity), different populations")
ax.grid(True, which="both", lw=0.3)
fig.tight_layout()
fig.savefig("screening_ppv.png", dpi=150)

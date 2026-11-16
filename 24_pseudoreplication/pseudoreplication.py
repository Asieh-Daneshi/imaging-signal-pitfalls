"""Pitfall 24: many trials per subject treated as independent samples.

Two synthetic groups of subjects have exactly the same true mean. Every subject
has a personal offset (some people simply have larger values) plus trial-to-trial
noise. If all trials are pooled and compared with a t-test, the number of trials
is mistaken for the number of independent samples, and the test flags a group
difference far more often than the nominal 5 %. Averaging the trials within each
subject first, and testing the subject means, restores the expected rate.

Run:  python pseudoreplication.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

SUBJECTS, SIGMA_SUBJECT, SIGMA_TRIAL = 10, 1.0, 1.0
N_SIM, ALPHA = 2000, 0.05


def simulate_group(rng, n_trials):
    offsets = rng.normal(0, SIGMA_SUBJECT, size=(SUBJECTS, 1))
    return offsets + rng.normal(0, SIGMA_TRIAL, size=(SUBJECTS, n_trials))


def false_positive_rates(rng, n_trials):
    pooled = per_subject = 0
    for _ in range(N_SIM):
        a, b = simulate_group(rng, n_trials), simulate_group(rng, n_trials)
        pooled += ttest_ind(a.ravel(), b.ravel())[1] < ALPHA
        per_subject += ttest_ind(a.mean(axis=1), b.mean(axis=1))[1] < ALPHA
    return pooled / N_SIM, per_subject / N_SIM


def main():
    rng = np.random.default_rng(0)
    print(f"{SUBJECTS} subjects per group, true group difference 0, "
          f"{N_SIM} simulated studies per row, test at p < {ALPHA}")
    trial_counts = [1, 5, 20, 100]
    rows = []
    for n_trials in trial_counts:
        pooled, per_subject = false_positive_rates(rng, n_trials)
        rows.append((pooled, per_subject))
        print(f"{n_trials:>4} trials per subject: pooled trials flagged "
              f"{100 * pooled:5.1f} %, subject means flagged {100 * per_subject:5.1f} %")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(trial_counts, [100 * r[0] for r in rows], "o-", label="all trials pooled")
    ax.plot(trial_counts, [100 * r[1] for r in rows], "s-", label="subject means")
    ax.axhline(100 * ALPHA, c="k", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("trials per subject")
    ax.set_ylabel("false positives (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pseudoreplication.png", dpi=130)


if __name__ == "__main__":
    main()

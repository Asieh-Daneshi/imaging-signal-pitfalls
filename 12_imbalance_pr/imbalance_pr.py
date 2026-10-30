"""Pitfall 12: ROC AUC looks fine under class imbalance, precision does not.

Synthetic setup: one fixed scoring model (two Gaussian score distributions,
separation d' = 2). Only the share of positives in the test set changes.
ROC AUC does not depend on prevalence; average precision (area under the
precision-recall curve) and precision at a fixed sensitivity of 90 % do.

Run:  python imbalance_pr.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

D_PRIME = 2.0
N_NEG = 100_000
PREVALENCES = [0.5, 0.1, 0.01, 0.001]
TARGET_SENS = 0.90
rng = np.random.default_rng(0)


def make_test_set(prevalence):
    n_pos = max(int(round(N_NEG * prevalence / (1 - prevalence))), 1)
    scores = np.concatenate([rng.normal(0, 1, N_NEG), rng.normal(D_PRIME, 1, n_pos)])
    labels = np.concatenate([np.zeros(N_NEG), np.ones(n_pos)])
    return labels, scores


def precision_at_sensitivity(labels, scores, sens):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return precision[recall >= sens].max()


def main():
    print(f"Same model (d' = {D_PRIME}), {N_NEG} negatives, only the number of positives changes")
    print(f"{'prevalence':>10} {'positives':>10} {'ROC AUC':>8} {'avg precision':>14} {'precision @90% sens':>20}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for prev in PREVALENCES:
        y, s = make_test_set(prev)
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        p90 = precision_at_sensitivity(y, s, TARGET_SENS)
        print(f"{prev:>10.1%} {int(y.sum()):>10} {auc:>8.3f} {ap:>14.3f} {p90:>20.3f}")
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"prevalence {prev:.1%} (AUC {auc:.2f})")
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision")
    ax.set_title("Same model, different prevalence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("imbalance_pr.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 19: normalisation statistics fitted on the test data.

Synthetic feature vectors (think: intensity features from an image) from two
classes. The test data come from a "different scanner": every feature has a
gain and an offset that the training data do not have. We compare how the
features are standardised before a logistic regression:

  A. mean and std from the training set only (what a deployed model can do)
  B. mean and std from the whole test set (needs the full test batch)
  C. mean and std from only k test cases at a time (a small batch in use)

Run:  python norm_leakage.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

N_FEAT, N_TRAIN, N_TEST, N_REPEATS = 20, 300, 1000, 50
GAIN, OFFSET = 1.8, 1.5


def make(n, rng, shifted):
    y = rng.integers(0, 2, n)
    x = rng.standard_normal((n, N_FEAT))
    x[:, :5] += 0.8 * y[:, None]  # only five features carry the class
    if shifted:
        x = GAIN * x + OFFSET
    return x, y


def standardise(x, ref):
    return (x - ref.mean(0)) / ref.std(0)


def run(seed):
    rng = np.random.default_rng(seed)
    x_tr, y_tr = make(N_TRAIN, rng, False)
    x_te, y_te = make(N_TEST, rng, True)
    clf = LogisticRegression(max_iter=1000).fit(standardise(x_tr, x_tr), y_tr)
    acc = {"A": clf.score(standardise(x_te, x_tr), y_te),
           "B": clf.score(standardise(x_te, x_te), y_te)}
    for k in (5, 20, 100):
        correct = 0
        for i in range(0, N_TEST, k):
            batch = x_te[i:i + k]
            correct += (clf.predict(standardise(batch, batch)) == y_te[i:i + k]).sum()
        acc[k] = correct / N_TEST
    return acc


def main():
    runs = [run(s) for s in range(N_REPEATS)]
    mean = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
    print(f"{N_TRAIN} train cases, {N_TEST} test cases, {N_FEAT} features, "
          f"test gain {GAIN} and offset {OFFSET}, {N_REPEATS} repeats")
    print(f"A  statistics from train only:            accuracy {mean['A']:.3f}")
    print(f"B  statistics from the whole test set:    accuracy {mean['B']:.3f}")
    for k in (5, 20, 100):
        print(f"C  statistics from batches of {k:>3} test cases: accuracy {mean[k]:.3f}")
    labels = ["train\nonly", "batch 5", "batch 20", "batch 100", "whole\ntest set"]
    vals = [mean["A"], mean[5], mean[20], mean[100], mean["B"]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, vals, color=["C3", "C1", "C1", "C1", "C0"])
    ax.axhline(0.5, c="k", ls=":")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1)
    fig.tight_layout()
    fig.savefig("norm_leakage.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 13: choosing the decision threshold on the test set.

Synthetic setup: a scoring model with two Gaussian score distributions
(negatives N(0, 1), positives N(d', 1)), balanced classes. A threshold is
picked by maximising accuracy, either on the test set itself or on a separate
validation set. We then compare the accuracy reported on the test set with the
accuracy on a very large fresh sample from the same distribution.

Run:  python threshold_on_test.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_TEST = 100          # cases per split (half positive)
N_TRIALS = 2000
N_FRESH = 200_000     # large sample used as the "true" accuracy
D_PRIMES = [0.0, 0.5, 1.0]
rng = np.random.default_rng(0)


def draw(n, d):
    y = np.repeat([0, 1], n // 2)
    s = rng.normal(0, 1, n) + d * y
    return y, s


def best_threshold(y, s):
    cand = np.sort(s)
    acc = [np.mean((s > t) == y) for t in cand]
    return cand[int(np.argmax(acc))]


def acc_at(y, s, t):
    return np.mean((s > t) == y)


def main():
    print(f"{N_TEST} cases per split, {N_TRIALS} repetitions, threshold = best accuracy")
    print(f"{'d prime':>8} {'true acc':>9} {'test-picked (reported)':>23} {'val-picked (reported)':>22} {'val-picked (true)':>18}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in D_PRIMES:
        yf, sf = draw(N_FRESH, d)
        true_best = acc_at(yf, sf, d / 2)
        rep_test, rep_val, true_val = [], [], []
        for _ in range(N_TRIALS):
            yv, sv = draw(N_TEST, d)
            yt, st = draw(N_TEST, d)
            t_test = best_threshold(yt, st)
            rep_test.append(acc_at(yt, st, t_test))
            t_val = best_threshold(yv, sv)
            rep_val.append(acc_at(yt, st, t_val))
            true_val.append(acc_at(yf, sf, t_val))
        print(f"{d:>8.1f} {true_best:>9.3f} {np.mean(rep_test):>23.3f} {np.mean(rep_val):>22.3f} {np.mean(true_val):>18.3f}")
        ax.scatter([d], [np.mean(rep_test)], c="C3", label="threshold picked on test" if d == 0 else None)
        ax.scatter([d], [np.mean(rep_val)], c="C0", label="threshold picked on validation" if d == 0 else None)
        ax.scatter([d], [true_best], c="k", marker="_", s=300, label="true accuracy" if d == 0 else None)
    ax.set_xlabel("model strength d'")
    ax.set_ylabel("reported test accuracy (mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("threshold_on_test.png", dpi=130)


if __name__ == "__main__":
    main()

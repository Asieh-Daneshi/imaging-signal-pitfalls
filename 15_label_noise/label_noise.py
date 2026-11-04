"""Pitfall 15: label noise in the test set.

Synthetic setup: two scoring models on the same balanced test cases. Model A is
a bit better than model B (score separation d' = 2.0 vs 1.8, correlated errors).
Test labels are flipped at random with probability p, as happens when the
reference annotation is imperfect. We look at the accuracy that gets reported
and at how often the better model is also reported as the better one.

Run:  python label_noise.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_TEST = 500          # test cases (half positive)
N_TRIALS = 4000
D_A, D_B, RHO = 2.0, 1.8, 0.8
NOISE = [0.0, 0.1, 0.2, 0.3]
rng = np.random.default_rng(0)


def accuracy(score, d, label):
    return np.mean((score > d / 2) == label, axis=-1)


def one_round(p):
    y = np.repeat([0, 1], N_TEST // 2)
    z, z2 = rng.normal(size=(2, N_TRIALS, N_TEST))
    s_a = z + D_A * y
    s_b = RHO * z + np.sqrt(1 - RHO**2) * z2 + D_B * y
    flip = rng.random((N_TRIALS, N_TEST)) < p
    y_noisy = np.where(flip, 1 - y, y).astype(bool)
    return (accuracy(s_a, D_A, y_noisy), accuracy(s_b, D_B, y_noisy),
            accuracy(s_a, D_A, y.astype(bool)), accuracy(s_b, D_B, y.astype(bool)))


def main():
    print(f"{N_TEST} test cases, {N_TRIALS} repetitions, d' A={D_A}, B={D_B}")
    print(f"{'noise':>6} {'A reported':>11} {'B reported':>11} {'gap A-B':>8} {'P(A>B)':>7}")
    rows = []
    for p in NOISE:
        a, b, a_clean, b_clean = one_round(p)
        rows.append((p, a.mean(), b.mean(), np.mean(a > b)))
        print(f"{p:6.1f} {a.mean():11.3f} {b.mean():11.3f} {a.mean() - b.mean():8.3f} "
              f"{np.mean(a > b):7.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    p_, a_, b_, w_ = zip(*rows)
    ax[0].plot(p_, a_, "o-", label="model A")
    ax[0].plot(p_, b_, "s-", label="model B")
    ax[0].set_xlabel("test label noise rate")
    ax[0].set_ylabel("reported accuracy")
    ax[0].legend()
    ax[1].plot(p_, w_, "o-", c="C3")
    ax[1].axhline(0.5, c="k", ls=":")
    ax[1].set_xlabel("test label noise rate")
    ax[1].set_ylabel("P(better model reported better)")
    fig.tight_layout()
    fig.savefig("label_noise.png", dpi=130)


if __name__ == "__main__":
    main()

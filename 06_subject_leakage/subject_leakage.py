"""Pitfall 06: slice-level splits leak subject identity.

Synthetic setup: each subject contributes many slices. Every subject has its own
"fingerprint" (anatomy, scanner quirks, ...) that is much stronger than the weak
disease signal. A random split over slices lets the model recognise subjects it
has already seen, so accuracy looks great. Splitting by subject removes that
shortcut and shows the real performance.

Run:  python subject_leakage.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 0
N_SUBJECTS = 60        # half healthy, half diseased
SLICES = 30            # slices per subject
N_FEATURES = 40
SUBJECT_STRENGTH = 2.0  # size of the per-subject fingerprint
DISEASE_STRENGTH = 0.15  # weak real signal
NOISE = 1.0


def make_data(rng):
    labels = np.repeat([0, 1], N_SUBJECTS // 2)
    rng.shuffle(labels)
    disease_axis = rng.normal(size=N_FEATURES)
    disease_axis /= np.linalg.norm(disease_axis)
    X, y, g = [], [], []
    for s in range(N_SUBJECTS):
        fingerprint = rng.normal(size=N_FEATURES) * SUBJECT_STRENGTH
        for _ in range(SLICES):
            x = fingerprint + rng.normal(size=N_FEATURES) * NOISE
            x += labels[s] * DISEASE_STRENGTH * disease_axis * np.sqrt(N_FEATURES)
            X.append(x)
            y.append(labels[s])
            g.append(s)
    return np.array(X), np.array(y), np.array(g)


def evaluate(X, y, train_idx, test_idx):
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    return clf.score(X[test_idx], y[test_idx])


def main():
    rng = np.random.default_rng(SEED)
    X, y, g = make_data(rng)
    idx = np.arange(len(y))

    # 1) wrong: random split over slices
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)
    acc_slice = evaluate(X, y, tr, te)
    shared = len(set(g[tr]) & set(g[te]))

    # 2) right: split by subject
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=g))
    acc_subject = evaluate(X, y, tr, te)

    print(f"Random split over slices : accuracy {acc_slice:.1%} "
          f"({shared} subjects appear in both train and test)")
    print(f"Split by subject         : accuracy {acc_subject:.1%} (chance is 50%)")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["random split\n(slices)", "split by\nsubject"],
                  [acc_slice * 100, acc_subject * 100], color=["#f59e0b", "#38bdf8"])
    for b, v in zip(bars, [acc_slice, acc_subject]):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%}", ha="center")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Same data, same model, different split")
    fig.tight_layout()
    fig.savefig("subject_leakage.png", dpi=130)


if __name__ == "__main__":
    main()

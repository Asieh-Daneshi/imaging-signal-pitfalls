"""Pitfall 07: feature selection on all the data before cross-validation.

Synthetic setup: the features are pure noise and the labels are random, so no
model can do better than 50 %. If you pick the "best" features using ALL
samples and only then cross-validate, the selection has already seen the test
folds and the score looks well above chance. Doing the selection inside each
training fold (a Pipeline) gives the honest answer.

Run:  python selection_leakage.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

N_SAMPLES = 60       # small cohort, typical for imaging studies
N_FEATURES = 2000    # e.g. radiomics or spectral features
K = 20               # features kept
N_REPEATS = 20       # different random datasets
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def one_dataset(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    y = np.repeat([0, 1], N_SAMPLES // 2)  # labels unrelated to X
    clf = LogisticRegression(max_iter=1000)

    # wrong: select on everything, then cross-validate
    X_sel = SelectKBest(f_classif, k=K).fit_transform(X, y)
    wrong = cross_val_score(clf, X_sel, y, cv=CV).mean()

    # right: selection is part of the pipeline, refit inside each training fold
    pipe = make_pipeline(SelectKBest(f_classif, k=K), clf)
    right = cross_val_score(pipe, X, y, cv=CV).mean()
    return wrong, right


def main():
    res = np.array([one_dataset(s) for s in range(N_REPEATS)])
    wrong, right = res[:, 0], res[:, 1]
    print(f"{N_REPEATS} random datasets, {N_SAMPLES} samples, {N_FEATURES} noise features, keep {K}")
    print(f"Select on all data, then CV : mean accuracy {wrong.mean():.1%} "
          f"(min {wrong.min():.1%}, max {wrong.max():.1%})")
    print(f"Select inside each fold     : mean accuracy {right.mean():.1%} "
          f"(min {right.min():.1%}, max {right.max():.1%})  chance = 50%")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot([wrong * 100, right * 100], widths=0.5)
    ax.set_xticklabels(["selection on\nall data", "selection inside\nCV folds"])
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_ylabel("CV accuracy (%)")
    ax.set_title("Pure noise: only one of these can be right")
    fig.tight_layout()
    fig.savefig("selection_leakage.png", dpi=130)


if __name__ == "__main__":
    main()

"""Pitfall 22: ICA on rank-deficient EEG (after average reference).

Synthetic setup: 8 channels = 4 brain-like sources + 1 eye-blink source mixed by
a random matrix (the blink is strong on frontal channels), plus sensor noise.
We know the true clean signal. After re-referencing to the average, the 8
channels sum to zero, so the data only have rank 7. Asking ICA for 8
components then asks it to find a component that does not exist.

Blink component = the independent component most correlated with a frontal
channel (a stand-in for an EOG channel). It is zeroed and the signal rebuilt.
Error = RMS(cleaned - true clean) / RMS(true clean), both average-referenced.

Run:  python ica_rank.py
"""
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

warnings.filterwarnings("ignore")
FS = 250
N_CH = 8
DURATION = 60
N_REPEATS = 40


def make_data(rng):
    n = DURATION * FS
    t = np.arange(n) / FS

    def rhythm(freq, mod):
        return np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.3)) * (
            1 + 0.5 * np.sin(2 * np.pi * mod * t + rng.uniform(0, 6.3)))

    pink = np.cumsum(rng.normal(size=n))
    pink = pink - np.convolve(pink, np.ones(FS) / FS, mode="same")
    brain = np.vstack([rhythm(10, 0.2), rhythm(20, 0.35), rhythm(6, 0.15), pink])
    brain = brain / brain.std(axis=1, keepdims=True)
    blink = np.zeros(n)
    for s in rng.choice(n - FS, DURATION // 3, replace=False):
        w = np.hanning(int(0.3 * FS))
        blink[s:s + len(w)] += w * rng.uniform(0.8, 1.2)
    blink /= blink.std()
    clean = rng.normal(size=(N_CH, 4)) @ brain
    a_blink = np.array([4.0, 3.5, 3.0, 1.2, 0.8, 0.6, 0.4, 0.3]) * 2.0
    X = clean + np.outer(a_blink, blink) + 0.3 * rng.normal(size=(N_CH, n))
    return X, clean, blink


def avg_ref(x):
    return x - x.mean(axis=0, keepdims=True)


def clean_with_ica(Xr, n_components, seed):
    ica = FastICA(n_components=n_components, whiten="unit-variance", max_iter=1000, tol=1e-3, random_state=seed)
    S = ica.fit_transform(Xr.T).T
    k = int(np.argmax([abs(np.corrcoef(S[i], Xr[0])[0, 1]) for i in range(n_components)]))
    S_clean = S.copy()
    S_clean[k] = 0
    return ica.inverse_transform(S_clean.T).T, S[k]


def main():
    rng = np.random.default_rng(0)
    err = {"no cleaning": [], "ICA, 8 components": [], "ICA, 7 components (= rank)": []}
    corr = {"ICA, 8 components": [], "ICA, 7 components (= rank)": []}
    last_sv = []
    sv_example = None
    for r in range(N_REPEATS):
        X, clean, blink = make_data(rng)
        Xr, cr = avg_ref(X), avg_ref(clean)
        sv = np.linalg.svd(Xr, compute_uv=False)
        sv_example = sv if sv_example is None else sv_example
        last_sv.append(sv[-1] / sv[0])
        rms = np.sqrt((cr ** 2).mean())
        err["no cleaning"].append(np.sqrt(((Xr - cr) ** 2).mean()) / rms)
        for name, nc in [("ICA, 8 components", N_CH), ("ICA, 7 components (= rank)", N_CH - 1)]:
            Xc, ic = clean_with_ica(Xr, nc, r)
            err[name].append(np.sqrt(((Xc - cr) ** 2).mean()) / rms)
            corr[name].append(abs(np.corrcoef(ic, blink)[0, 1]))

    print(f"{N_CH} channels, average reference, {DURATION} s at {FS} Hz, {N_REPEATS} repeats")
    print(f"smallest / largest singular value after average reference: {np.mean(last_sv):.1e}")
    print(f"{'':<28}{'rel. error':>11}{'|corr| blink IC vs true blink (mean / worst)':>48}")
    for name in err:
        c = corr.get(name)
        cs = f"{np.mean(c):.3f} / {np.min(c):.3f}" if c else "-"
        print(f"{name:<28}{np.mean(err[name]):>11.3f}{cs:>48}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].semilogy(range(1, N_CH + 1), sv_example / sv_example[0], "o-")
    ax[0].set_xlabel("component")
    ax[0].set_ylabel("singular value (relative)")
    ax[0].set_title("After average reference\nthe 8th value is ~0")
    names = list(err)
    ax[1].bar(range(3), [np.mean(err[n]) for n in names], color=["grey", "#c4511f", "#2a4d8f"])
    ax[1].set_xticks(range(3), ["none", "8 comp.", "7 comp."])
    ax[1].set_ylabel("relative error vs true clean")
    ax[1].set_title("Error after cleaning")
    ax[2].boxplot([corr[n] for n in names[1:]], tick_labels=["8 comp.", "7 comp."])
    ax[2].set_ylabel("|corr| blink IC vs true blink")
    ax[2].set_title("Is the blink isolated?")
    fig.tight_layout()
    fig.savefig("ica_rank.png", dpi=130)


if __name__ == "__main__":
    main()

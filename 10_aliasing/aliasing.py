"""Pitfall 10: downsampling without an anti-aliasing filter.

A 1 kHz recording contains a slow 5 Hz rhythm and a 40 Hz component (for example
mains-related or muscle noise). We reduce the sampling rate to 50 Hz, so the new
Nyquist frequency is 25 Hz and the 40 Hz component cannot be represented.

- Naive decimation (keep every 20th sample): the 40 Hz component does not vanish,
  it folds back to |40 - 50| = 10 Hz and shows up as a fake rhythm.
- Filter first, then decimate (scipy.signal.decimate): the 40 Hz component is
  removed and only the real 5 Hz rhythm remains.

Run:  python aliasing.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

FS = 1000          # original sampling rate (Hz)
FACTOR = 20        # downsampling factor -> 50 Hz
DURATION = 10      # seconds
SLOW_HZ, FAST_HZ = 5, 40
SEED = 0


def spectrum(x, fs):
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    amp = np.abs(np.fft.rfft(x * np.hanning(n))) / n * 4
    return freqs, amp


def peaks(x, fs, min_amp=0.15):
    f, a = spectrum(x, fs)
    idx, _ = signal.find_peaks(a, height=min_amp)
    return [(round(float(f[i]), 1), round(float(a[i]), 2)) for i in idx]


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(0, DURATION, 1 / FS)
    x = (np.sin(2 * np.pi * SLOW_HZ * t)
         + 0.8 * np.sin(2 * np.pi * FAST_HZ * t)
         + 0.1 * rng.normal(size=t.size))

    naive = x[::FACTOR]
    proper = signal.decimate(x, FACTOR, ftype="fir", zero_phase=True)
    fs_new = FS // FACTOR

    print(f"Original ({FS} Hz)        peaks (Hz, amplitude): {peaks(x, FS)}")
    print(f"Naive every {FACTOR}th sample ({fs_new} Hz): {peaks(naive, fs_new)}")
    print(f"Filter + decimate ({fs_new} Hz):    {peaks(proper, fs_new)}")

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=False)
    for ax, (name, sig, fs) in zip(axes, [("Original, 1000 Hz", x, FS),
                                          ("Naive decimation, 50 Hz", naive, fs_new),
                                          ("Filter, then decimate, 50 Hz", proper, fs_new)]):
        f, a = spectrum(sig, fs)
        ax.plot(f, a)
        ax.set_xlim(0, 60)
        ax.set_ylabel("amplitude")
        ax.set_title(name)
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig("aliasing.png", dpi=130)


if __name__ == "__main__":
    main()

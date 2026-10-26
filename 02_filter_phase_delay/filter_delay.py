"""Same low-pass filter, two ways of applying it: causal (lfilter) vs zero-phase (filtfilt).

Shows how a causal filter shifts the timing of a peak. Synthetic data only.
Run: python filter_delay.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt

rng = np.random.default_rng(1)
fs = 1000                                  # Hz
t = np.arange(0, 1, 1 / fs)                # 1 second
true_peak_ms = 300

clean = np.exp(-((t - true_peak_ms / 1000) ** 2) / (2 * 0.03 ** 2))   # Gaussian event
noisy = clean + rng.normal(0, 0.3, t.size)

b, a = butter(4, 8, btype="low", fs=fs)    # 4th-order Butterworth, 8 Hz cut-off
causal = lfilter(b, a, noisy)              # only looks backwards in time
zero_phase = filtfilt(b, a, noisy)         # forward + backward pass

peak = lambda x: t[np.argmax(x)] * 1000
print(f"true peak:          {true_peak_ms} ms")
print(f"causal (lfilter):   {peak(causal):.0f} ms")
print(f"zero-phase (filtfilt): {peak(zero_phase):.0f} ms")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t * 1000, noisy, color="0.8", lw=0.6, label="noisy signal")
ax.plot(t * 1000, clean, "k--", lw=1, label="clean event")
ax.plot(t * 1000, causal, lw=2, label=f"causal: peak at {peak(causal):.0f} ms")
ax.plot(t * 1000, zero_phase, lw=2, label=f"zero-phase: peak at {peak(zero_phase):.0f} ms")
ax.axvline(true_peak_ms, color="k", lw=0.5)
ax.set_xlabel("time (ms)")
ax.set_title("8 Hz Butterworth low-pass, applied causally vs zero-phase")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("filter_delay.png", dpi=150)

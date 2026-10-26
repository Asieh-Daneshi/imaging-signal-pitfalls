"""Turn a noisy 1-D signal into an image (spectrogram) so a hidden event becomes visible.

Synthetic data only. Run: python spectrogram_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, stft

rng = np.random.default_rng(0)
fs = 250                      # sampling rate in Hz (typical for EEG)
t = np.arange(0, 15, 1 / fs)

# Background: broadband noise plus a slow drift
signal = rng.normal(0, 1.0, t.size) + 0.5 * np.sin(2 * np.pi * 0.2 * t)

# Hidden event: a weak frequency sweep from 20 Hz to 45 Hz between 6 s and 10 s
event = (t >= 6) & (t < 10)
signal[event] += 0.9 * chirp(t[event] - 6, f0=20, t1=4, f1=45)

# Short-time Fourier transform. Window length decides the time/frequency trade-off.
nperseg = 128
f, tt, Z = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
power_db = 20 * np.log10(np.abs(Z) + 1e-12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(t, signal, lw=0.5, color="k")
ax1.set_ylabel("amplitude")
ax1.set_title("Raw signal: can you see the event?")
im = ax2.pcolormesh(tt, f, power_db, shading="gouraud", cmap="magma")
ax2.set_ylim(0, 80)
ax2.set_ylabel("frequency (Hz)")
ax2.set_xlabel("time (s)")
ax2.set_title(f"Spectrogram (STFT, window = {nperseg} samples)")
fig.colorbar(im, ax=ax2, label="power (dB)")
fig.tight_layout()
fig.savefig("spectrogram_demo.png", dpi=150)

# Simple check: mean power in the 20-45 Hz band inside vs outside the event window
band = (f >= 20) & (f <= 45)
inside = (tt >= 6) & (tt < 10)
p = np.abs(Z[band]) ** 2
ratio = p[:, inside].mean() / p[:, ~inside].mean()
print(f"20-45 Hz power during the event is {ratio:.1f}x the power outside it")

# 10 - Aliasing when downsampling

Reducing the sampling rate by keeping every n-th sample looks harmless, but any component above the new Nyquist frequency does not disappear: it folds back into the band you keep and looks like a real signal.

## What the script does

- Builds a 10 s synthetic recording at 1000 Hz with a real 5 Hz rhythm and a 40 Hz component (think mains interference or muscle noise).
- Downsamples to 50 Hz (Nyquist 25 Hz) in two ways: keeping every 20th sample, and `scipy.signal.decimate`, which low-pass filters first.
- Prints the spectral peaks of each version.

## Result (seed 0)

| Signal | Peaks (Hz, amplitude) |
|---|---|
| Original, 1000 Hz | 5.0 (1.0), 40.0 (0.8) |
| Every 20th sample, 50 Hz | 5.0 (0.99), **10.0 (0.79)** |
| Filter, then decimate, 50 Hz | 5.0 (1.0) |

The 40 Hz component folds to |40 - 50| = 10 Hz, so naive decimation invents a 10 Hz rhythm that was never in the recording. Filtering first removes it.

## Takeaway

Low-pass filter below the new Nyquist frequency before you downsample (or use `scipy.signal.decimate`, `resample_poly` or the equivalent in your toolbox). The same applies to images: shrink them with an anti-aliased resize, not by dropping pixels.

## Run

```bash
pip install -r ../requirements.txt
python aliasing.py
```

The figure is written to `aliasing.png` next to the script.

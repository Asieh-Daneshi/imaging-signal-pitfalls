# 22 - ICA on rank-deficient EEG (after average reference)

Re-referencing EEG to the average makes the channels sum to zero, so N channels carry only N-1 independent signals. If ICA is still asked for N components, it has to invent one, and the artifact component it is meant to isolate comes out blurred.

## What the script does

- Builds 8 synthetic channels from 4 brain-like sources (10 Hz, 20 Hz, 6 Hz rhythms and slow noise) and one eye-blink source, mixed with a random matrix. The blink is strong on the frontal channels. Sensor noise is added.
- Because the clean signal is known, the cleaning error can be measured.
- Re-references to the average (the data now have rank 7), runs FastICA with 8 components and with 7 components, picks the component most correlated with a frontal channel as the blink, zeroes it and rebuilds the signal.
- Repeats this 40 times with 60 s recordings at 250 Hz.

## Result (seed 0)

| | Relative error vs true clean signal | \|corr\| of blink component with true blink (mean / worst) |
|---|---|---|
| No cleaning | 1.607 | - |
| ICA, 8 components | 1.079 | 0.869 / 0.390 |
| ICA, 7 components (= rank) | 0.435 | 0.997 / 0.981 |

After the average reference the smallest singular value is about 1e-15 of the largest. With 8 components the blink is sometimes not isolated at all (correlation 0.39 in the worst repeat), with 7 it is always isolated and the cleaning error is about 2.5 times smaller.

## Takeaway

Check the rank of the data before ICA (after average reference, interpolating channels or removing bad channels) and set the number of components to the rank, for example with a PCA step first. EEGLAB's `pop_runica` has a `'pca'` option for this.

## Run

```bash
pip install -r ../requirements.txt
python ica_rank.py
```

The figure is written to `ica_rank.png` next to the script.

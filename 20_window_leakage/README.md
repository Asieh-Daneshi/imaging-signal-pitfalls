# 20 - Overlapping windows split at random

A long recording is often cut into overlapping windows, and the windows are then
split at random into train and test. Neighbouring windows share most of their
samples, and even without overlap they share the same slowly changing baseline.
A model can then recognise the time point instead of learning anything about the
label, and the test score is too high.

## What the script does

- Builds one synthetic recording of 60000 samples: a slowly wandering baseline
  plus noise. Nothing here is clinical data.
- Gives each stretch of 1500 samples a random label (0 or 1). The label has no
  relation to the signal, so the true accuracy is 50 %.
- Cuts windows of 200 samples with four strides (200, 100, 50 and 20, which is
  0, 50, 75 and 90 % overlap) and reduces each window to 20 bin means.
- Scores a 1-nearest-neighbour classifier with two splits: windows split at
  random in half, and a blocked split (first half of the recording for training,
  second half for testing, with the windows at the boundary removed).
- Repeats this 20 times with different random draws and prints the mean accuracy.

## Result (mean of 20 repeats, true accuracy 50 %)

| Stride (overlap) | Random split | Blocked split |
|---|---|---|
| 200 (0 %) | 0.589 | 0.510 |
| 100 (50 %) | 0.590 | 0.516 |
| 50 (75 %) | 0.609 | 0.521 |
| 20 (90 %) | 0.632 | 0.520 |

With no signal at all, the random split reports 0.589 to 0.632, and the score
rises as the windows overlap more. The blocked split stays close to chance
(0.510 to 0.521). The effect here is moderate because the model is a plain
1-nearest-neighbour on a noisy baseline; a model that can memorise more would
gain more from the leak.

## Takeaway

Split by contiguous blocks of the recording (or by subject or session), leave a
gap at least as long as the window, and cut windows only after the split.

## Run

```bash
pip install -r ../requirements.txt
python window_leakage.py
```

The figure is written to `window_leakage.png` next to the script.

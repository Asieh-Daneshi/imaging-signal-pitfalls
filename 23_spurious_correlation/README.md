# 23 - Correlating two independent slow signals

Two signals with nothing to do with each other can still look significantly
correlated. If each signal changes slowly, neighbouring samples are not
independent, so the effective sample size is far smaller than the number of
samples, and the usual p-value is much too optimistic. This matters for any
analysis that correlates two recordings (EEG power and pupil size, a metric and
time, two drifting sensors).

## What the script does

- Generates 2000 pairs of completely independent synthetic signals, 500 samples
  each. Nothing here is clinical data.
- Tries four kinds of signal: white noise, a slow AR(1) process (phi 0.95), a
  random walk, and a random walk after taking first differences.
- Runs a Pearson correlation test on each pair and counts how often it reports
  p < 0.05. For independent signals, that should happen in about 5 % of pairs.

## Result (2000 pairs, true correlation 0)

| Signal | Pairs flagged as correlated |
|---|---|
| White noise | 4.8 % |
| AR(1), phi 0.95 | 66.9 % |
| Random walk | 89.4 % |
| Random walk, differenced | 4.3 % |

White noise behaves as the test assumes. With slow or drifting signals most
independent pairs are reported as correlated, and taking first differences of
the random walk brings the rate back to the expected level.

## Takeaway

Before correlating two time series, check their autocorrelation. Remove trends
or take differences, use a test or a permutation scheme that respects
autocorrelation (for example shifting one signal circularly), and do not read a
p-value computed for independent samples as evidence.

## Run

```bash
pip install -r ../requirements.txt
python spurious_correlation.py
```

The figure is written to `spurious_correlation.png` next to the script.

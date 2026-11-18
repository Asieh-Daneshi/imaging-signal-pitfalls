# 25 - Testing many channels without correction

A recording has many channels, an image has many pixels, a trace has many time
points. Running one test in each and reporting whatever comes out below
p = 0.05 almost guarantees a "finding", even when nothing differs between the
groups. This matters for channel-by-channel EEG comparisons, voxel-wise maps and
time-point-wise tests.

## What the script does

- Simulates two groups of 20 synthetic samples with exactly the same
  distribution in every channel. Nothing here is clinical data.
- Runs a t-test in each channel and counts the studies (2000 per row) that report
  at least one significant channel, for 1, 4, 16, 64 and 256 channels.
- Compares three ways of deciding: raw p < 0.05, Bonferroni (p < 0.05 divided by
  the number of channels) and Benjamini-Hochberg (controls the share of false
  discoveries).
- Repeats with 64 channels where the first 6 have a real shift of 1 standard
  deviation, to show what each correction costs in detection.

## Result (2000 simulated studies per row)

No true difference anywhere, share of studies with at least one significant
channel:

| Channels | Uncorrected | Bonferroni | Benjamini-Hochberg |
|---|---|---|---|
| 1 | 5.2 % | 5.2 % | 5.2 % |
| 4 | 19.5 % | 5.2 % | 5.2 % |
| 16 | 56.5 % | 4.7 % | 4.7 % |
| 64 | 96.7 % | 4.5 % | 4.8 % |
| 256 | 100.0 % | 5.3 % | 5.5 % |

64 channels, real shift in 6 of them:

| Method | Real channels found | False channels per study |
|---|---|---|
| Uncorrected | 0.865 | 2.90 |
| Bonferroni | 0.331 | 0.05 |
| Benjamini-Hochberg | 0.490 | 0.20 |

## Takeaway

The more places you look, the more likely something looks significant by
chance, so decide how many tests you run before you read the results and correct
for them. Corrections are not free: they remove false findings and also some real
ones, and the less strict Benjamini-Hochberg rule keeps more of the real effects
than Bonferroni. Report how many channels, pixels or time points were tested, not
only the ones that passed.

## Run

```bash
pip install -r ../requirements.txt
python multiple_comparisons.py
```

The figure is written to `multiple_comparisons.png` next to the script.

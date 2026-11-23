# 26 - High correlation is not agreement

Two methods that measure the same quantity (two software tools, two operators, a
new device and a reference) are often compared with a correlation coefficient. A
high r only says that one method goes up when the other goes up. It says nothing
about whether the numbers are the same, so a method with a constant offset or a
proportional bias still looks excellent.

## What the script does

- Simulates 200 synthetic cases with a true value between 30 and 70 (arbitrary
  units, nothing here is clinical data) and two methods that each add noise with
  a standard deviation of 1.5.
- Setting 1: method B is the true value plus noise only.
- Setting 2: method B also reads 15 % high and adds a constant 5 units.
- Reports Pearson r, Lin's concordance correlation coefficient (which penalises
  offset and scale differences), the mean difference with Bland-Altman limits of
  agreement, how the difference changes with the size of the measurement, and the
  share of cases where the two methods differ by at most 5 units.
- Repeats setting 2 on 500 new datasets to check the result is not one lucky draw.

## Result

| Setting | Pearson r | Lin CCC | Mean difference (B - A) | Limits of agreement | Within +-5 |
|---|---|---|---|---|---|
| B only noisy | 0.985 | 0.985 | +0.16 | -3.92 to +4.24 | 98.5 % |
| B biased | 0.987 | 0.643 | +12.82 | +7.73 to +17.90 | 0.0 % |

In the biased setting the difference also grows by 0.134 units per unit of
measurement, so no single correction number fixes it.

Over 500 new datasets with the biased method: r averages 0.986 (lowest 0.980),
Lin CCC averages 0.650 (highest 0.694).

## Takeaway

When the question is whether one method can replace another, do not stop at r.
Plot the difference against the mean (Bland-Altman), report the bias and the
limits of agreement, and decide beforehand how large a difference is acceptable.
Look at whether the difference depends on the size of the measurement, because a
proportional bias cannot be removed by subtracting a constant.

## Run

```bash
pip install -r ../requirements.txt
python agreement_vs_correlation.py
```

The figure is written to `agreement_vs_correlation.png` next to the script.

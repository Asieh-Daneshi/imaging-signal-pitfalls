# 24 - Many trials per subject are not many samples

Two groups of people with the same true mean can look significantly different
if every trial is treated as an independent sample. Trials from one subject
share that subject's personal level, so they are correlated, and the effective
sample size is closer to the number of subjects than to the number of trials.
This matters for any study that records many epochs, trials or images per
person (EEG epochs, repeated visual trials, several slices or frames per
patient).

## What the script does

- Simulates two groups of 10 synthetic subjects with exactly the same true
  mean. Each subject has a personal offset (standard deviation 1) and trial
  noise (standard deviation 1). Nothing here is clinical data.
- Varies the number of trials per subject: 1, 5, 20 and 100.
- For each setting, runs 2000 simulated studies and compares the groups two
  ways with a t-test at p < 0.05: once with all trials pooled, once with one
  mean per subject.

## Result (2000 simulated studies per row, true group difference 0)

| Trials per subject | Pooled trials flagged | Subject means flagged |
|---|---|---|
| 1 | 5.4 % | 5.4 % |
| 5 | 26.1 % | 4.7 % |
| 20 | 53.8 % | 5.1 % |
| 100 | 78.2 % | 5.7 % |

With one trial per subject the two analyses are the same. As trials are added,
the pooled test finds a group difference more and more often although there is
none, while the test on subject means stays close to the expected 5 %.

## Takeaway

Count independent units, not rows. Average within subject first, or use a model
that includes subject as a random effect, and keep the split by subject in the
design of the study. More trials per subject make each subject's estimate more
precise, but they do not add subjects.

## Run

```bash
pip install -r ../requirements.txt
python pseudoreplication.py
```

The figure is written to `pseudoreplication.png` next to the script.

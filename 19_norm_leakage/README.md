# 19 - Normalisation statistics fitted on the test data

Standardising features with the mean and standard deviation of the test set
looks like routine preprocessing, but it changes what the reported number
means. The statistics come from data that a deployed model may not have: a
single new case has no batch to take a mean from. If the test set differs from
the training set (a different scanner, for example), test-set statistics can
hide that shift, and the reported accuracy then describes a pipeline that
needs the whole test batch at prediction time.

## What the script does

- Draws synthetic feature vectors (20 features, 5 of them carry the class).
  Nothing here is clinical data.
- Gives the test data a gain of 1.8 and an offset of 1.5 on every feature,
  which the training data do not have.
- Trains a logistic regression on features standardised with training
  statistics, then scores the same test set in three ways: standardised with
  training statistics, with statistics from the whole test set, and with
  statistics from small batches of test cases only.
- Repeats this 50 times with different random draws and prints the mean
  accuracy.

## Result (300 train cases, 1000 test cases, mean of 50 repeats)

| Statistics used to standardise the test features | Accuracy |
|---|---|
| Training set only | 0.558 |
| Whole test set (1000 cases) | 0.796 |
| Batches of 100 test cases | 0.794 |
| Batches of 20 test cases | 0.783 |
| Batches of 5 test cases | 0.743 |

The same model scores 0.558 or 0.796 on the same test set depending only on
where the normalisation statistics come from. In this setup the test-set
statistics are not wrong, they remove the shift; the point is that they are a
different pipeline. The accuracy drops as the batch gets smaller, and with one
case at a time the statistics cannot be computed at all.

## Takeaway

Fit every scaler on the training data only and apply it unchanged to the test
data. If you want test-time adaptation, state it, report how many cases the
statistics were computed from, and evaluate it with the batch size you will
actually have when the model is used.

## Run

```bash
pip install -r ../requirements.txt
python norm_leakage.py
```

The figure is written to `norm_leakage.png` next to the script.

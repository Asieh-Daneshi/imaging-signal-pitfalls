# 13 - Picking the threshold on the test set

A model outputs a score, and somebody has to choose the threshold that turns it into a yes/no decision. If that threshold is chosen to maximise accuracy on the test set, the test set has been used for fitting, and the reported accuracy is optimistic. It looks like a small effect, and it is largest exactly when the test set is small and the model is weak.

## What the script does

- Uses synthetic scores on balanced classes: negatives N(0, 1), positives N(d', 1), with d' = 0 (no signal), 0.5 and 1.0.
- Draws a validation set and a test set of 100 cases each, repeated 2000 times.
- Compares two ways of choosing the threshold: the one that maximises accuracy on the test set itself, and the one that maximises accuracy on the validation set and is then applied unchanged to the test set.
- Estimates the true accuracy on a fresh sample of 200000 cases from the same distribution.

## Result (seed 0)

| d' | True accuracy | Reported, threshold picked on test | Reported, threshold picked on validation |
|---|---|---|---|
| 0.0 | 0.502 | 0.558 | 0.501 |
| 0.5 | 0.599 | 0.639 | 0.588 |
| 1.0 | 0.692 | 0.723 | 0.682 |

With no signal at all, choosing the threshold on the test set reports 55.8 % accuracy on average. Choosing it on a separate validation set reports 50.1 %, which is honest, and is slightly pessimistic for the stronger models because the validation set is also small. (The "true accuracy" of 0.502 for d' = 0 is sampling noise in the large fresh sample; the real value is 0.5.)

## Takeaway

Fix the threshold using validation data (or cross-validation on the training data) before the test set is touched, and report the test result for that fixed threshold. Threshold-free metrics such as ROC AUC avoid the issue, but any reported sensitivity or specificity needs a threshold that was chosen without looking at the test data. Compare with experiment 09, which shows the same kind of leakage through feature selection.

## Run

```bash
pip install -r ../requirements.txt
python threshold_on_test.py
```

The figure is written to `threshold_on_test.png` next to the script.

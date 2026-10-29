# 08 - Accurate is not the same as trustworthy (calibration)

A classifier can rank cases well and still be wrong about how sure it is. In clinical settings the confidence value is what a person acts on, so it should match how often the model is actually right.

## What the script does

- Simulates a binary task in which labels are drawn from known probabilities, then gives the "model" logits that are 2.6x too large (overconfident). No real data is used.
- Fits a single temperature `T` on a held-out calibration split (6,000 samples), then reports everything on a separate test split (6,000 samples).
- Plots reliability diagrams (observed accuracy vs predicted confidence) before and after, and prints the expected calibration error (ECE).

## Result (seed 7)

| | Accuracy | ECE |
|---|---|---|
| Raw model | 74.6 % | 0.128 |
| After temperature scaling (T = 2.68) | 74.6 % | 0.015 |

Accuracy does not change at all: dividing the logits by one number never reorders the predictions. Only the stated confidence becomes honest.

The LinkedIn post that goes with this experiment showed a figure where the temperature was fitted and evaluated on the same samples (ECE 0.134 to 0.006). The version here uses separate splits, which is the correct protocol, so the "after" number is a bit higher.

## Takeaway

Plot a reliability diagram next to ROC and accuracy, calibrate on data the model has not seen, and re-check calibration whenever the data source or scanner changes.

## Run

```bash
pip install -r ../requirements.txt
python calibration.py
```

The figure is written to `calibration.png` next to the script.

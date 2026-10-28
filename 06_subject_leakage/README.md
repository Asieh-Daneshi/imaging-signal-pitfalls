# 06 - Subject leakage

Medical datasets usually contain many slices, frames or recordings per person. If you split **slices** at random, slices from the same person end up in both the training and the test set. The model can then score well by recognising the person, not the disease.

## What the script does

- Builds synthetic data for 60 subjects with 30 slices each. Every subject has a strong personal "fingerprint" and only a weak real disease signal.
- Trains the same random forest twice: once on a random split over slices, once on a split by subject (`GroupShuffleSplit`).

## Result (seed 0)

| Split | Test accuracy |
|---|---|
| Random over slices (all 60 subjects appear in train and test) | 99.8 % |
| By subject | 50.9 % (chance level) |

The data and the model are identical. Only the split changed.

## Takeaway

Split by subject (or by site, scanner or time) before anything else, and keep every preprocessing step that learns from data (scaling, feature selection, augmentation statistics) inside the training fold.

## Run

```bash
pip install -r ../requirements.txt
python subject_leakage.py
```

The figure is written to `subject_leakage.png` next to the script.

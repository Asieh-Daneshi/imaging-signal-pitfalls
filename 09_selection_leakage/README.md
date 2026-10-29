# 09 - Feature selection before cross-validation

With few samples and many features (radiomics, spectral bands, pixel statistics), it is tempting to first pick the features that best separate the classes and then cross-validate a model on them. But the selection step has already looked at every sample, including the ones that later serve as test folds. The score is then inflated even when the data contain no signal at all.

## What the script does

- Builds 20 synthetic datasets of 60 samples and 2000 features. The features are pure noise and the labels are unrelated to them, so the true accuracy is 50 %.
- Keeps the 20 best features (`SelectKBest`, ANOVA F-score) and runs 5-fold cross-validation with logistic regression, in two ways: selecting on all data first, or selecting inside each training fold with a `Pipeline`.

## Result (seeds 0-19)

| Procedure | Mean CV accuracy (min - max) |
|---|---|
| Select on all data, then cross-validate | 90.2 % (76.7 % - 98.3 %) |
| Select inside each training fold | 50.4 % (40.0 % - 65.0 %) |

Same data, same model, same folds. Only the place of the selection step changed.

## Takeaway

Anything that learns from data (feature selection, scaling, PCA, augmentation statistics, threshold tuning) belongs inside the training fold. Wrapping the steps in a `Pipeline` makes this automatic.

## Run

```bash
pip install -r ../requirements.txt
python selection_leakage.py
```

The figure is written to `selection_leakage.png` next to the script.

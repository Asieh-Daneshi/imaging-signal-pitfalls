# 12 - ROC AUC hides what happens when positives are rare

ROC AUC does not depend on how common the positive class is, so a model can keep a respectable AUC while almost every flagged case is a false alarm. Precision and the precision-recall curve do change with prevalence, and they are what a reader of the results actually experiences.

## What the script does

- Uses one fixed scoring model on synthetic data: negatives score N(0, 1), positives score N(2, 1).
- Keeps 100000 negatives and changes only the number of positives, so prevalence goes from 50 % down to 0.1 %.
- Reports ROC AUC, average precision (area under the precision-recall curve) and the best precision reachable at 90 % sensitivity.

## Result (seed 0)

| Prevalence | Positives | ROC AUC | Average precision | Precision at 90 % sensitivity |
|---|---|---|---|---|
| 50.0 % | 100000 | 0.921 | 0.922 | 0.791 |
| 10.0 % | 11111 | 0.921 | 0.669 | 0.293 |
| 1.0 % | 1010 | 0.927 | 0.306 | 0.038 |
| 0.1 % | 100 | 0.919 | 0.062 | 0.004 |

The model is the same in every row. ROC AUC stays around 0.92, while average precision drops from 0.922 to 0.062. The AUC wiggles slightly at low prevalence because only a few hundred or a hundred positives are drawn.

## Takeaway

When the positive class is rare, report precision-recall metrics (or precision at a clinically relevant sensitivity) next to ROC AUC, and state the prevalence of the test set. Compare it with experiment 05, which shows the same effect through PPV.

## Run

```bash
pip install -r ../requirements.txt
python imbalance_pr.py
```

The figure is written to `imbalance_pr.png` next to the script.

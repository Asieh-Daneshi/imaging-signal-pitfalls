# 18 - Seed variance in a small network

A single training run of a small network gives one number, but that number
depends on the random initialisation. With a small training set the spread
between seeds can be as large as the difference between two methods that are
compared in a paper or a report. Picking the best seed on the test set then
adds an optimistic bias on top.

## What the script does

- Builds synthetic 12 x 12 images with one bright blob at a random position.
  The label is 0 if the blob is elongated horizontally and 1 if it is
  elongated vertically.
- Fixes one split: 60 training, 200 validation and 200 test images.
- Trains a small neural network (one hidden layer with 16 units, scikit-learn)
  30 times. Only the random seed of the initialisation changes.
- Prints the mean, standard deviation, minimum and maximum of the test
  accuracy, and what would be reported if the best seed were chosen on the
  test set or on the validation set.

## Result (30 seeds, same data)

| Quantity | Test accuracy |
|---|---|
| Mean over seeds | 0.842 |
| Standard deviation | 0.019 |
| Worst seed | 0.795 |
| Best seed | 0.875 |
| Seed chosen on the validation set | 0.860 |

The same data and the same model give accuracies from 79.5 % to 87.5 %,
depending only on the seed. Reporting the best seed chosen on the test set
gives 87.5 %, which is 3.3 points above the mean of 84.2 %.

## Takeaway

Report the mean and spread over several seeds, not a single run, and do not
claim that one method beats another unless the difference is clearly larger
than the seed-to-seed spread. If one model has to be selected, select it on a
validation set, never on the test set.

## Run

```bash
pip install -r ../requirements.txt
python seed_variance.py
```

The figure is written to `seed_variance.png` next to the script.

# 11 - A single accuracy number from a small test set

Imaging studies often have a few dozen test cases. The accuracy computed on them is one draw from a wide distribution, yet it is usually reported as a single number with no interval. Two models, or two runs of the same model, can differ by ten points purely because of which cases ended up in the test set.

## What the script does

- Simulates a classifier whose true accuracy is exactly 80 %, and draws 5000 independent test sets for each size from 20 to 2000 cases (synthetic, no real data).
- Reports the spread of the accuracy you would have reported, and how often a 95 % Wilson interval contains the true value.
- Shows one concrete test set of 50 cases with its Wilson and bootstrap intervals.

## Result (seed 0)

| Test set size | Reported accuracy, 2.5 to 97.5 % of draws | Wilson interval coverage |
|---|---|---|
| 20 | 60.0 % to 95.0 % | 95.6 % |
| 50 | 68.0 % to 90.0 % | 95.4 % |
| 100 | 71.0 % to 88.0 % | 93.5 % |
| 500 | 76.4 % to 83.4 % | 94.8 % |
| 2000 | 78.2 % to 81.8 % | 94.8 % |

One test set of 50 cases gave 88.0 % accuracy, with a Wilson interval of 76.2 % to 94.4 % and a bootstrap interval of 78.0 % to 96.0 %. The true value is 80 %.

## Takeaway

With 50 test cases the same 80 % model can look like anything from 68 % to 90 %. Report an interval (Wilson or bootstrap) next to every test metric, and do not rank models whose intervals overlap heavily.

## Run

```bash
pip install -r ../requirements.txt
python small_test_set_ci.py
```

The figure is written to `small_test_set_ci.png` next to the script.

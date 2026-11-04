# 15 - Label noise in the test set

Reference labels are rarely perfect: annotators disagree, a diagnosis is revised, a threshold on a measurement is somewhat arbitrary. If a fraction of the test labels is wrong, every model is scored against the wrong answer on those cases. Reported accuracy drops towards chance, and the difference between a better and a worse model shrinks, so a real improvement is harder to detect and the ranking can flip.

## What the script does

- Uses synthetic scores on balanced test cases: negatives N(0, 1), positives N(d', 1), with d' = 2.0 for model A and 1.8 for model B. The two models share most of their noise, as two models on the same images would.
- Flips each test label at random with probability 0, 0.1, 0.2 and 0.3.
- Draws 500 test cases, repeated 4000 times, and reports the mean accuracy of each model and the fraction of repetitions in which model A is reported better than model B.

## Result (seed 0)

| Label noise | A reported | B reported | Gap A-B | P(A reported better) |
|---|---|---|---|---|
| 0.0 | 0.841 | 0.816 | 0.025 | 0.938 |
| 0.1 | 0.773 | 0.753 | 0.020 | 0.886 |
| 0.2 | 0.705 | 0.690 | 0.015 | 0.810 |
| 0.3 | 0.636 | 0.626 | 0.010 | 0.711 |

With clean labels, model A is reported better in 93.8 % of the repetitions. With 30 % noisy labels the same pair of models gives 63.6 % and 62.6 % accuracy, the gap has fallen from 0.025 to 0.010, and the better model is reported better in only 71.1 % of the repetitions.

## Takeaway

Measure how reliable the reference labels are (repeat annotations, inter-rater agreement) and report it next to the test metric. Low accuracy on a noisy test set does not necessarily mean a weak model, and a small difference between two models on such a set says little about which is better. Compare with experiment 11, which shows the sampling side of the same problem for small test sets.

## Run

```bash
pip install -r ../requirements.txt
python label_noise.py
```

The figure is written to `label_noise.png` next to the script.

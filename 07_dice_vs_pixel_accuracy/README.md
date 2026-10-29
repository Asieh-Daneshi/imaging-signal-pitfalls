# 07 - Dice vs pixel accuracy

When the structure you care about is small (a lesion, a vessel, a spot), almost every pixel is background. Pixel accuracy is then dominated by the background, so even a model that segments nothing looks excellent. Dice only counts the overlap with the structure itself.

## What the script does

- Builds 200 synthetic 128x128 masks, each with one round "lesion" of radius 3-5 px (0.31 % of the pixels on average). No real images are used.
- Scores four hypothetical predictors against the true mask with both pixel accuracy and Dice: an empty prediction, a prediction shifted by 3 px, an over-segmented (dilated by 3 px) one, and a good one with boundary noise only.

## Result (seed 0)

| Predictor | Pixel accuracy | Dice |
|---|---|---|
| Empty (all background) | 99.7 % | 0.00 |
| Shifted by 3 px | 99.6 % | 0.35 |
| Over-segmented (+3 px) | 99.5 % | 0.54 |
| Good (boundary noise) | 99.9 % | 0.86 |

Pixel accuracy spans only 99.5-99.9 % across all four, so it cannot tell a useless model from a good one. Dice goes from 0.00 to 0.86.

## Takeaway

For small structures report an overlap metric (Dice, IoU) or a boundary metric, per image, and look at the empty-prediction baseline before trusting any accuracy number.

## Run

```bash
pip install -r ../requirements.txt
python dice_vs_accuracy.py
```

The figure is written to `dice_vs_accuracy.png` next to the script.

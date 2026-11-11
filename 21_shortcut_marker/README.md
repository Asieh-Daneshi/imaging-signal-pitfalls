# 21 - A shortcut that breaks in deployment

Images often carry something besides the finding: a scanner tag, a burned-in
annotation, a ruler, a site-specific border. If that mark happens to match the
label in the training data, a model can use it instead of the anatomy. It looks
excellent on a test set drawn the same way and fails when the mark stops
matching the label.

## What the script does

- Builds synthetic 16x16 images. Class 1 has a faint Gaussian blob in the centre
  (amplitude 0.5) in noise of standard deviation 1.0. Nothing here is clinical data.
- Adds a bright 2x2 tag in one corner. In the training set the tag matches the
  class in 95 % of the images.
- Trains a logistic regression on 600 images and tests it on 2000 images in three
  settings: tag matches the class as in training (0.95), tag unrelated to the
  class (0.50), and tag reversed (0.05).
- As a reference, trains a second model on data where the tag is unrelated to the
  class and tests it on unrelated-tag data.
- Repeats this 20 times with different random draws and prints the mean accuracy.

## Result (mean of 20 repeats)

| Setting | Accuracy |
|---|---|
| Tag matches class (0.95), as in training | 0.934 |
| Tag unrelated to class (0.50) | 0.603 |
| Tag reversed (0.05) | 0.282 |
| Reference: trained with unrelated tag, tested unrelated | 0.773 |

The test set that looks like the training data gives 0.934. With the tag
unrelated to the class the same model gives 0.603, and with the tag reversed it
falls to 0.282, well below chance, because it confidently follows the tag. A
model that never had the shortcut available reaches 0.773 on the unrelated-tag
data. The synthetic blob is weak on purpose, which makes the shortcut very
attractive; with a stronger real signal the drop would be smaller.

## Takeaway

A high score on a test set that shares the acquisition quirks of the training
set says little. Check what the model uses (crop or mask the tag, look at
saliency), and evaluate on data where the nuisance feature is unrelated to the
label, for example another site or scanner.

## Run

```bash
pip install -r ../requirements.txt
python shortcut_marker.py
```

The figure is written to `shortcut_marker.png` next to the script.

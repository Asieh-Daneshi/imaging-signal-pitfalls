# 16 - Augmentation that changes the label

Flips, rotations and crops are standard augmentations, but each one assumes
that the label survives the transformation. For anything where left and right
(or up and down) carry meaning, such as laterality of an organ, the side of a
lesion, or the direction of a movement, a horizontal flip turns an example of
one class into an example of the other and keeps the old label. The model is
then trained on a share of wrong labels that nobody wrote down.

## What the script does

- Builds synthetic 16 x 16 images with one bright blob and Gaussian noise. The
  label is 0 if the blob is in the left half and 1 if it is in the right half.
- Trains a logistic regression on the pixels with 60 training images and tests
  on 2000 clean images.
- Flips a share p of the training images (0, 0.1, 0.25, 0.5) and keeps their
  labels: once left-right, once up-down. Up-down keeps the label here,
  left-right does not.
- Repeats everything for 20 seeds and prints mean and standard deviation of the
  test accuracy.

## Result (20 runs)

| Share flipped | Left-right flip | Up-down flip |
|---|---|---|
| 0.00 | 0.902 +- 0.050 | 0.902 +- 0.050 |
| 0.10 | 0.840 +- 0.070 | 0.901 +- 0.048 |
| 0.25 | 0.719 +- 0.073 | 0.898 +- 0.046 |
| 0.50 | 0.509 +- 0.072 | 0.901 +- 0.043 |

With no augmentation the model reaches 90.2 % accuracy. When half of the
training images are flipped left-right with their old label, the accuracy is
50.9 %, which is chance. The same amount of up-down flipping changes nothing
(90.1 %).

## Takeaway

Before adding an augmentation, ask whether the label would change if you
applied it by hand. If a human annotator would give a different answer, either
change the label together with the image or leave the augmentation out. A
quick check is to train with and without it and compare on an untouched
validation set.

## Run

```bash
pip install -r ../requirements.txt
python flip_label.py
```

The figure is written to `flip_label.png` next to the script.

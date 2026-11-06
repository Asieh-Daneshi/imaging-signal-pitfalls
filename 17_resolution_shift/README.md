# 17 - A change in resolution at test time

A model trained on images with one pixel size can fail quietly when the test
images come with another one: a different scanner, a different export setting,
or a resize step that was added later. If the structure that carries the label
lives at a particular spatial scale, resampling moves it to a different scale
and the model reads the wrong class.

## What the script does

- Builds synthetic 32 x 32 textures from Gaussian noise smoothed with a small
  blur (class 0, sigma 1.0) or a larger blur (class 1, sigma 1.6). The classes
  differ only in spatial scale.
- Trains a logistic regression on two gradient-energy features (image gradient
  and Laplacian, after normalising each image) with 200 training images, all at
  zoom 1.0.
- Tests on 400 images that are resampled by a zoom factor between 0.6 and 1.6.
- Compares this with a model trained on images drawn at random from all of the
  zoom factors (scale augmentation).
- Repeats everything for 15 seeds and prints mean and standard deviation of the
  test accuracy.

## Result (15 runs, synthetic data)

| Test zoom | Trained at 1.0 | Scale-augmented |
|---|---|---|
| 0.60 | 0.500 +- 0.014 | 0.500 +- 0.014 |
| 0.80 | 0.913 +- 0.027 | 0.800 +- 0.150 |
| 1.00 | 1.000 +- 0.000 | 1.000 +- 0.000 |
| 1.25 | 0.543 +- 0.021 | 0.642 +- 0.136 |
| 1.60 | 0.500 +- 0.014 | 0.500 +- 0.014 |

At the training resolution the model is perfect (1.000). At a zoom of 1.25 it
falls to 54.3 %, and at 0.6 and 1.6 it is at chance (50.0 %). Scale
augmentation does not rescue it here: it gives 64.2 % at zoom 1.25 and still
50.0 % at the extremes, and it costs accuracy at zoom 0.8 (80.0 % against
91.3 %), with a large spread between seeds.

## Takeaway

Augmenting over scale only helps when scale is not what defines the label. In
this setup it is, so the zoom factor and the class are confounded and no
classifier can separate them. Record the pixel size or resolution with every
image, resample everything to one common spacing before training and testing,
and report accuracy on a deliberately resampled test set to see how fragile
the model is.

## Run

```bash
pip install -r ../requirements.txt
python resolution_shift.py
```

The figure is written to `resolution_shift.png` next to the script.

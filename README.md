# Imaging and signal pitfalls

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Data](https://img.shields.io/badge/data-synthetic%20only-lightgrey) ![Experiments](https://img.shields.io/badge/experiments-22-2ea44f) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Small, self-contained experiments on things that quietly go wrong when you process biomedical images and signals. Each one runs in a few seconds on a laptop and uses synthetic data only, so nothing here is clinical or patient data.

These are clean, reproducible versions of the demos behind a series of my LinkedIn posts. The exact numbers can differ a little from the posts because the parameters and random seeds differ, but the effects are the same. Each one is a problem I have run into in real research pipelines (EEG, retinal imaging, behavioural video).

| # | Experiment | What it shows | Result (from the scripts) |
|---|---|---|---|
| 01 | [Signal to spectrogram](01_signal_to_spectrogram) | A weak frequency sweep that is invisible in the raw trace stands out once the signal becomes an image (STFT). | Event band power about 3x the background |
| 02 | [Filter phase delay](02_filter_phase_delay) | The same 8 Hz Butterworth filter shifts a peak when applied causally (`lfilter`) but not zero-phase (`filtfilt`). | Peak at 300 ms moves to 359 ms with `lfilter` |
| 03 | [Smoothing vs small structures](03_smoothing_small_structures) | Stronger Gaussian blur removes noise, and a 1-pixel-scale spot with it. | At sigma 4: 8 % noise left, 6 % of the spot left |
| 04 | [Dataset shift with a CNN](04_dataset_shift_cnn) | A small PyTorch CNN is perfect on its own "site" and drops to chance on another, while AUC stays high. Per-image normalisation fixes most of it. | Site B accuracy 50 % (AUC 0.975), 96 % after normalisation |
| 05 | [Screening and PPV](05_screening_ppv) | A classifier with 90 % sensitivity and 90 % specificity is mostly wrong when it flags someone in a low-prevalence population. The model stays the same; the population changes. | At 1 % prevalence, PPV 8.3 % (9 of 108 flags are real) |
| 06 | [Subject leakage](06_subject_leakage) | Splitting over slices instead of subjects lets a model recognise patients it has already seen, so test accuracy looks near perfect. | Random slice split 99.8 %, split by subject 50.9 % (chance) |
| 07 | [Dice vs pixel accuracy](07_dice_vs_pixel_accuracy) | On a tiny lesion, pixel accuracy is dominated by background, so even an empty prediction scores like a good model. Dice tells them apart. | Empty prediction: 99.7 % accuracy, Dice 0.00; good one: 99.9 %, Dice 0.86 |
| 08 | [Calibration](08_calibration) | A model can rank well and still be overconfident. One temperature parameter, fitted on held-out data, fixes the confidence without changing accuracy. | ECE 0.128 to 0.015, accuracy unchanged at 74.6 % |
| 09 | [Feature selection before cross-validation](09_selection_leakage) | Picking the best features on all data and only then cross-validating inflates the score, even when the data are pure noise. | Pure noise (true accuracy 50 %): 90.2 % if selected on all data, 50.4 % if selected inside each fold |
| 10 | [Aliasing when downsampling](10_aliasing) | Keeping every n-th sample without a low-pass filter folds fast components back into the band and creates a rhythm that was never there. | 40 Hz folds to a fake 10 Hz rhythm (amplitude 0.79) at 50 Hz; filtering first leaves only the real 5 Hz |
| 11 | [Small test set intervals](11_small_test_set_ci) | One accuracy number from a few dozen test cases is a single draw from a wide distribution. An interval shows how much it can move. | True accuracy 80 %: with 50 test cases the reported value ranges from 68 % to 90 % (2.5 to 97.5 % of draws) |
| 12 | [Class imbalance and PR curves](12_imbalance_pr) | ROC AUC stays the same when positives become rare, but precision collapses. The same model looks very different once prevalence is low. | Same model: ROC AUC about 0.92 throughout; average precision 0.922 at 50 % prevalence, 0.306 at 1 %, 0.062 at 0.1 % |
| 13 | [Threshold chosen on the test set](13_threshold_on_test) | Choosing the decision threshold that maximises accuracy on the test set uses the test set for fitting. With 100 test cases and no signal at all, it reports well above chance. | No signal (true accuracy 50 %): 55.8 % reported if the threshold is picked on the test set, 50.1 % if picked on a separate validation set |
| 14 | [Dice vs object size](14_dice_vs_object_size) | The same 1-pixel boundary error gives a very different Dice score depending on object size. | 1 px too wide: Dice 0.97 at radius 40, 0.77 at radius 5, 0.64 at radius 3 |
| 15 | [Label noise in the test set](15_label_noise) | Wrong reference labels in the test set pull every reported accuracy towards chance and shrink the gap between models, so a real improvement becomes harder to see. | Noise 0 to 0.3: A-B gap 0.025 to 0.010; chance that the better model is reported better 0.938 to 0.711 |
| 16 | [Augmentation that changes the label](16_flip_label) | A horizontal flip looks harmless, but if the label is left or right, it swaps the class while the label stays the same. A vertical flip does not. | Training images flipped (label unchanged): 0 % gives 90.2 % accuracy, 50 % gives 50.9 % with horizontal flips and 90.1 % with vertical flips |
| 17 | [Resolution change at test time](17_resolution_shift) | A model trained at one pixel size fails when test images are resampled, if the label lives at a particular spatial scale. Scale augmentation does not help when scale defines the class. | Trained at zoom 1.0: accuracy 1.000, 0.543 at zoom 1.25, 0.500 (chance) at zoom 0.6 and 1.6; scale augmentation gives 0.642 at zoom 1.25 |
| 18 | [Seed variance in a small network](18_seed_variance) | The same data and the same model give different test accuracies for different random seeds. Choosing the best seed on the test set adds an optimistic bias. | 30 seeds: mean 0.842, std 0.019, range 0.795 to 0.875; best seed on the test set 0.875, seed chosen on validation 0.860 |
| 19 | [Normalisation fitted on the test data](19_norm_leakage) | Standardising test features with test-set statistics changes the pipeline: it needs a whole batch at prediction time and can hide a shift between training and test data. | Same model and test set: accuracy 0.558 with training statistics, 0.796 with whole-test-set statistics, 0.743 with batches of 5 |
| 20 | [Overlapping windows split at random](20_window_leakage) | Cutting one recording into overlapping windows and splitting them at random puts near-identical neighbours in train and test, so a model with no real signal scores above chance. | No signal (true accuracy 50 %): random split 0.589 at 0 % overlap and 0.632 at 90 %, blocked split 0.510 and 0.520 |
| 21 | [Shortcut learning from a corner tag](21_shortcut_marker) | A mark that matches the label in the training data (scanner tag, annotation) can be used instead of the real finding. The test set looks great until the mark stops matching. | Tag matches class in 95 % of training images: accuracy 0.934 on the same kind of test set, 0.603 with the tag unrelated, 0.282 with it reversed; trained without the shortcut 0.773 |
| 22 | [ICA on rank-deficient EEG](22_ica_rank) | After re-referencing to the average, EEG channels sum to zero and have one fewer independent signal than channels. Asking ICA for as many components as channels blurs the artifact component. | 8 channels, 60 s: relative error 1.079 with 8 components, 0.435 with 7 (= rank); blink component correlation with true blink 0.869 on average (worst 0.390) vs 0.997 (worst 0.981) |

## Run

```bash
pip install -r requirements.txt
cd 02_filter_phase_delay
python filter_delay.py
```

Each folder writes its figure next to the script.

## Contact

Asieh Daneshi, PhD · Berlin · [LinkedIn](https://www.linkedin.com/in/asieh-daneshi)

# Imaging and signal pitfalls

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Data](https://img.shields.io/badge/data-synthetic%20only-lightgrey) ![Experiments](https://img.shields.io/badge/experiments-9-2ea44f) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## Run

```bash
pip install -r requirements.txt
cd 02_filter_phase_delay
python filter_delay.py
```

Each folder writes its figure next to the script.

## Contact

Asieh Daneshi, PhD · Berlin · [LinkedIn](https://www.linkedin.com/in/asieh-daneshi)

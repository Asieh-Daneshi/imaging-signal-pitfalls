# Imaging and signal pitfalls

Small, self-contained experiments on things that quietly go wrong when you process biomedical images and signals. Each one runs in a few seconds on a laptop and uses synthetic data only, so nothing here is clinical or patient data.

These are clean, reproducible versions of the demos behind a series of my LinkedIn posts. The exact numbers can differ a little from the posts because the parameters and random seeds differ, but the effects are the same. Each one is a problem I have run into in real research pipelines (EEG, retinal imaging, behavioural video).

| # | Experiment | What it shows | Result (from the scripts) |
|---|---|---|---|
| 01 | [Signal to spectrogram](01_signal_to_spectrogram) | A weak frequency sweep that is invisible in the raw trace stands out once the signal becomes an image (STFT). | Event band power about 3x the background |
| 02 | [Filter phase delay](02_filter_phase_delay) | The same 8 Hz Butterworth filter shifts a peak when applied causally (`lfilter`) but not zero-phase (`filtfilt`). | Peak at 300 ms moves to 359 ms with `lfilter` |
| 03 | [Smoothing vs small structures](03_smoothing_small_structures) | Stronger Gaussian blur removes noise, and a 1-pixel-scale spot with it. | At sigma 4: 8 % noise left, 6 % of the spot left |
| 04 | [Dataset shift with a CNN](04_dataset_shift_cnn) | A small PyTorch CNN is perfect on its own "site" and drops to chance on another, while AUC stays high. Per-image normalisation fixes most of it. | Site B accuracy 50 % (AUC 0.975), 96 % after normalisation |

## Run

```bash
pip install -r requirements.txt
cd 02_filter_phase_delay
python filter_delay.py
```

Each folder writes its figure next to the script.

## Contact

Asieh Daneshi, PhD · Berlin · [LinkedIn](https://www.linkedin.com/in/asieh-daneshi)

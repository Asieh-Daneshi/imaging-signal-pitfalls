# Dataset shift with a small CNN (PyTorch)

Toy task: does a 32 x 32 image contain a small bright blob? A two-layer CNN is trained on "site A". "Site B" has the same task but a different gain, offset, noise level and shading, roughly what a different scanner would do.

| | Site A (held-out) | Site B (external) |
|---|---|---|
| No normalisation | 100 % acc, AUC 1.000 | **50 % acc**, AUC 0.975 |
| Per-image intensity normalisation | 99.7 % acc, AUC 1.000 | 96.0 % acc, AUC 0.993 |

The interesting part is the first row. Accuracy on site B is at chance, but AUC is still high, so the model still ranks the images correctly. What broke is the decision threshold, which was learned on site A's intensity range.

Takeaways:
- A random train/test split only tests the data you already have. Validate on another site, device or time period.
- Report AUC next to accuracy. The gap tells you whether the model is wrong or just miscalibrated.
- Normalise intensities, or recalibrate per site, before blaming the architecture.

Runs on CPU in about 15 seconds.

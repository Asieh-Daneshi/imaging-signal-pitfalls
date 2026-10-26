"""A small PyTorch CNN that looks perfect on its own "site" and fails on another.

Task: does a toy image contain a small bright blob? Site A and site B have the same task,
but site B images have different brightness, contrast, noise and shading (like another scanner).
We report accuracy and ROC AUC on both sites, then show that per-image intensity
normalisation closes most of the gap. Synthetic data only.
Run: python dataset_shift.py   (CPU, about a minute)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
rng = np.random.default_rng(0)
SIZE = 32


def make_images(n, gain, offset, noise, shading):
    """n images, half with a blob. gain/offset/noise/shading describe the 'scanner'."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    X = np.zeros((n, 1, SIZE, SIZE), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        img = rng.normal(0, noise, (SIZE, SIZE))
        img += shading * (xx / SIZE)                       # left-to-right shading
        if i % 2:
            cx, cy = rng.integers(6, SIZE - 6, 2)
            img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2))
            y[i] = 1
        X[i, 0] = gain * img + offset
    return torch.tensor(X), torch.tensor(y)


def per_image_norm(X):
    m = X.mean(dim=(2, 3), keepdim=True)
    s = X.std(dim=(2, 3), keepdim=True) + 1e-6
    return (X - m) / s


def small_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 8 * 8, 1),
    )


def train(model, X, y, epochs=8):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]).squeeze(1), y[idx].float())
            loss.backward()
            opt.step()
    return model


def evaluate(model, X, y):
    with torch.no_grad():
        logits = model(X).squeeze(1).numpy()
    acc = ((logits > 0) == y.numpy()).mean()          # threshold 0.5 on the probability
    return acc, roc_auc_score(y.numpy(), logits)       # AUC on logits avoids ties from saturated sigmoids


site_a = dict(gain=1.0, offset=0.0, noise=0.3, shading=0.0)
site_b = dict(gain=0.7, offset=0.4, noise=0.35, shading=0.3)

Xa_tr, ya_tr = make_images(3000, **site_a)
Xa_te, ya_te = make_images(1000, **site_a)
Xb_te, yb_te = make_images(1000, **site_b)

print("Without normalisation")
m = train(small_cnn(), Xa_tr, ya_tr)
for name, X, y in [("site A (held-out)", Xa_te, ya_te), ("site B (external)", Xb_te, yb_te)]:
    acc, auc = evaluate(m, X, y)
    print(f"  {name}: accuracy {acc:.1%}, AUC {auc:.3f}")

print("With per-image intensity normalisation")
m = train(small_cnn(), per_image_norm(Xa_tr), ya_tr)
for name, X, y in [("site A (held-out)", Xa_te, ya_te), ("site B (external)", Xb_te, yb_te)]:
    acc, auc = evaluate(m, per_image_norm(X), y)
    print(f"  {name}: accuracy {acc:.1%}, AUC {auc:.3f}")

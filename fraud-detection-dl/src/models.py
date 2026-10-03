"""PyTorch models: supervised MLP and unsupervised autoencoder."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(64, 32), p: float = 0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AutoEncoder(nn.Module):
    def __init__(self, d_in: int, bottleneck: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 24), nn.ReLU(), nn.Linear(24, bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck, 24), nn.ReLU(), nn.Linear(24, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_mlp(model, X):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X))).numpy()


def train_mlp(Xtr, ytr, Xva, yva, epochs=30, bs=512, lr=1e-3, patience=5, seed=0):
    """Class-weighted BCE; early stopping on validation PR-AUC."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = MLP(Xtr.shape[1])
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for b in _batches(len(Xt), bs, rng):
            if len(b) < 2:
                continue
            opt.zero_grad(); loss_fn(model(Xt[b]), yt[b]).backward(); opt.step()
        score = average_precision_score(yva, predict_mlp(model, Xva))
        if score > best:
            best, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_autoencoder(X_legit, epochs=30, bs=512, lr=1e-3, seed=0):
    """Train on LEGITIMATE transactions only; fraud is detected via high reconstruction error."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = AutoEncoder(X_legit.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X_legit)
    for _ in range(epochs):
        model.train()
        for b in _batches(len(Xt), bs, rng):
            opt.zero_grad(); ((model(Xt[b]) - Xt[b]) ** 2).mean().backward(); opt.step()
    return model


@torch.no_grad()
def reconstruction_error(model, X):
    model.eval()
    x = torch.tensor(X)
    return ((model(x) - x) ** 2).mean(dim=1).numpy()

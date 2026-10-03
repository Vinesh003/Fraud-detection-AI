"""Data loading, synthetic fallback, and time-based splitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount_log"]


def load_creditcard(path: str) -> pd.DataFrame:
    """Load the ULB Credit Card Fraud dataset (Kaggle: mlg-ulb/creditcardfraud)."""
    df = pd.read_csv(path)
    missing = {"Time", "Amount", "Class"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def make_synthetic(n: int = 30000, fraud_rate: float = 0.004, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the same schema. ONLY for smoke tests / CI, never for reported results."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    shift = rng.normal(1.5, 0.5, size=28) * (rng.random(28) < 0.4)
    X[y == 1] += shift
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = np.where(y == 1, rng.lognormal(4.5, 1.2, n), rng.lognormal(3.5, 1.0, n))
    df["Class"] = y
    return df


def time_split(df: pd.DataFrame, train: float = 0.6, val: float = 0.2):
    """Chronological split: earlier -> train, middle -> validation, latest -> test (no shuffling)."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


def prepare(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Scale log(Amount) with a scaler fitted on the TRAIN set only (no leakage)."""
    out = []
    scaler = StandardScaler().fit(np.log1p(train[["Amount"]]))
    for d in (train, val, test):
        d = d.copy()
        d["Amount_log"] = scaler.transform(np.log1p(d[["Amount"]])).ravel()
        out.append(d)
    return out, scaler


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(dtype="float32"), df["Class"].to_numpy(), df["Amount"].to_numpy()

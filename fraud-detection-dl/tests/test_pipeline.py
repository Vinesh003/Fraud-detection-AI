import numpy as np
from src import data as D, evaluate as E, train as T


def test_time_split_is_chronological():
    df = D.make_synthetic(5000)
    tr, va, te = D.time_split(df)
    assert tr["Time"].max() <= va["Time"].min() <= va["Time"].max() <= te["Time"].min()


def test_scaler_fit_on_train_only():
    df = D.make_synthetic(5000)
    (tr, va, te), sc = D.prepare(*D.time_split(df))
    assert abs(tr["Amount_log"].mean()) < 1e-6  # train is centred; val/test are not forced to be


def test_cost_threshold_beats_flag_nothing():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    s = rng.random(5000) * 0.3 + y * 0.6
    amt = rng.lognormal(3, 1, 5000)
    thr, cost = E.best_threshold(y, s, amt, 5.0)
    assert cost <= amt[y == 1].sum()


def test_end_to_end_smoke(tmp_path):
    res = T.run(D.make_synthetic(8000, fraud_rate=0.01), out=str(tmp_path), epochs=3)
    assert len(res["results"]) == 4
    assert (tmp_path / "xgb.json").exists()

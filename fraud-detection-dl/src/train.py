"""Train and compare all models on the same chronological split; write results + artifacts.

Usage:
  python -m src.train --data data/creditcard.csv
  python -m src.train --synthetic        # smoke test only; numbers are NOT meaningful
"""
from __future__ import annotations
import argparse, json, os
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from . import data as D, evaluate as E, models as M


def run(df, review_cost=5.0, out="artifacts", epochs=30, seed=0):
    os.makedirs(out, exist_ok=True)
    tr, va, te = D.time_split(df)
    (tr, va, te), scaler = D.prepare(tr, va, te)
    Xtr, ytr, _ = D.xy(tr); Xva, yva, ava = D.xy(va); Xte, yte, ate = D.xy(te)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    scores = {}  # name -> (val_scores, test_scores)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, ytr)
    scores["Logistic Regression"] = (lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1])

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                        random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    scores["XGBoost"] = (xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1])

    mlp = M.train_mlp(Xtr, ytr, Xva, yva, epochs=epochs, seed=seed)
    scores["MLP (PyTorch)"] = (M.predict_mlp(mlp, Xva), M.predict_mlp(mlp, Xte))

    ae = M.train_autoencoder(Xtr[ytr == 0], epochs=epochs, seed=seed)
    scores["Autoencoder (unsupervised)"] = (M.reconstruction_error(ae, Xva), M.reconstruction_error(ae, Xte))

    baseline_cost = E.amount_rule_baseline(yte, ate, review_cost)
    no_model_cost = float(ate[yte == 1].sum())  # flag nothing: every fraud is missed
    rows = []
    for name, (sv, st) in scores.items():
        thr, _ = E.best_threshold(yva, sv, ava, review_cost)      # chosen on validation only
        cost = E.total_cost(yte, st >= thr, ate, review_cost)       # evaluated on test
        rows.append({
            "model": name,
            "PR-AUC": round(E.pr_auc(yte, st), 4),
            "recall@precision>=0.9": round(E.recall_at_precision(yte, st, 0.9), 4),
            "threshold": thr,
            "test_cost": round(cost, 2),
            "saving_vs_amount_rule_%": round(100 * (baseline_cost - cost) / baseline_cost, 2) if baseline_cost else None,
            "saving_vs_no_model_%": round(100 * (no_model_cost - cost) / no_model_cost, 2) if no_model_cost else None,
        })
    result = {"review_cost": review_cost, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "test_fraud_rate": float(yte.mean()), "amount_rule_cost": baseline_cost,
              "no_model_cost": no_model_cost, "results": rows}
    with open(f"{out}/results.json", "w") as f:
        json.dump(result, f, indent=2)
    xgb.save_model(f"{out}/xgb.json")
    torch.save(mlp.state_dict(), f"{out}/mlp.pt")
    joblib.dump(scaler, f"{out}/scaler.joblib")
    te.sample(min(2000, len(te)), random_state=seed).to_csv(f"{out}/sample_test.csv", index=False)
    return result


def to_markdown(result) -> str:
    rows = result["results"]
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--review-cost", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    df = D.make_synthetic() if a.synthetic else D.load_creditcard(a.data)
    res = run(df, a.review_cost, epochs=a.epochs)
    print(to_markdown(res))
    open("reports/results.md", "w").write(to_markdown(res))

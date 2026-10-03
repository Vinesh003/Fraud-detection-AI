"""Streamlit demo: fraud score, per-transaction SHAP explanation, adjustable cost threshold."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D, evaluate as E  # noqa: E402

ART = ROOT / "artifacts"
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")

if not (ART / "xgb.json").exists():
    st.error("No trained artifacts found. Run `python -m src.train --data data/creditcard.csv` first.")
    st.stop()


@st.cache_resource
def load():
    model = XGBClassifier(); model.load_model(str(ART / "xgb.json"))
    sample = pd.read_csv(ART / "sample_test.csv")
    return model, sample, shap.TreeExplainer(model)


model, sample, explainer = load()
X, y, amount = D.xy(sample)
scores = model.predict_proba(X)[:, 1]

st.sidebar.header("Business costs")
review_cost = st.sidebar.number_input("Manual review cost per flagged transaction", 0.0, 500.0, 5.0, 1.0)
thr_opt, _ = E.best_threshold(y, scores, amount, review_cost)
thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(min(max(thr_opt, 0.0), 1.0)), 0.01)
st.sidebar.caption(f"Cost-optimal threshold on this sample: {thr_opt:.3f}")

flagged = scores >= thr
c1, c2, c3 = st.columns(3)
c1.metric("Flagged transactions", int(flagged.sum()))
c2.metric("Frauds caught", f"{int((flagged & (y == 1)).sum())} / {int(y.sum())}")
c3.metric("Total cost at threshold", f"{E.total_cost(y, flagged, amount, review_cost):,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Cost vs. threshold")
    grid, costs = E.cost_curve(y, scores, amount, review_cost)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(grid, costs); ax.axvline(thr, color="r", ls="--")
    ax.set_xlabel("threshold"); ax.set_ylabel("total cost")
    st.pyplot(fig)

with right:
    st.subheader("Explain one transaction")
    idx = st.number_input("Transaction row", 0, len(sample) - 1, int(np.argmax(scores)))
    st.write(f"Fraud score: **{scores[idx]:.3f}** | Amount: {amount[idx]:.2f} | "
             f"Decision: **{'FLAG' if scores[idx] >= thr else 'allow'}** | Actual label: {int(y[idx])}")
    sv = explainer.shap_values(X[idx:idx + 1])[0]
    top = np.argsort(np.abs(sv))[-10:]
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh([D.FEATURES[i] for i in top], sv[top])
    ax2.set_xlabel("SHAP value (impact on fraud score)")
    st.pyplot(fig2)
    st.caption("V1-V28 are anonymised PCA components, so explanations show which components "
               "drove the score, not human-readable reasons.")

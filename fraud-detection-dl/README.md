# Credit Card Fraud Detection with Deep Learning

Compares a PyTorch MLP, a PyTorch autoencoder, and an XGBoost baseline on the highly imbalanced
ULB credit card fraud dataset (about 284K transactions, 0.17% fraud). Models are chosen and
thresholds are set by **business cost**, not accuracy.

## Method
- **Chronological split** (60/20/20 by `Time`): train on the past, validate, test on the future. No shuffling.
- **No leakage:** the `Amount` scaler is fitted on the training set only.
- **Imbalance:** class weighting (`pos_weight` for the MLP, `scale_pos_weight` for XGBoost/logistic regression).
- **Models:** Logistic Regression, XGBoost, MLP (batch norm, dropout, early stopping on validation PR-AUC),
  and an autoencoder trained on legitimate transactions only (anomaly score = reconstruction error).
- **Metrics:** PR-AUC and recall at precision >= 0.9. Accuracy and ROC-AUC are not used as headline metrics.
- **Cost-based threshold:** a missed fraud costs its amount; every flagged transaction costs a manual review
  (default 5, an assumption you can change with `--review-cost`). The threshold is picked on the validation
  set and the cost is reported on the test set, compared with a naive rule (review the top 1% largest amounts)
  and with flagging nothing.
- **Explainability:** SHAP (TreeExplainer) for per-transaction explanations in the app.

## Run
```bash
pip install -r requirements.txt
# download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/
python -m src.train --data data/creditcard.csv      # writes artifacts/ and reports/results.md
streamlit run app/streamlit_app.py
pytest                                              # smoke tests (use synthetic data)
```
Docker: `docker build -t fraud-dl . && docker run -p 8501:8501 fraud-dl` (run training first so `artifacts/` exists).

## Results
Fill this table from `reports/results.md` after running on the real dataset.

| model | PR-AUC | recall@precision>=0.9 | test cost | saving vs amount rule |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Limitations
- `V1-V28` are anonymised PCA components, so SHAP explanations are not human-interpretable.
- The dataset covers two days only, so concept drift cannot be assessed.
- The review cost is an assumption; conclusions depend on it.
- `--synthetic` data exists only for smoke tests; its numbers are meaningless.

"""
AuraSync — Economist Agent — churn feature store export
=========================================================

The churn model scores a customer from transaction-derived features (recency_days,
momentum_ratio, ...). Those are computed from 31.8M transactions, which is too heavy to do
inside an API request, so this script computes them once for every customer and writes
`churn_customer_features.csv`. The API's `predict_churn_risk` chat tool and the
`/economist/churn/predict` endpoint look a customer up in that file and run the model live.

It reuses the exact feature functions from economist_churn_fatigue_model.py and the
label encoders / feature order saved in churn_fatigue_model.pkl, so serving features are
built the same way as training features (including the median fill for missing values).

USAGE (run from this folder, after training the churn model):
    python export_churn_features.py                 # features as of the last transaction date (live scoring)
    python export_churn_features.py --as-of training  # features at the model's training cutoff
"""
import argparse
import os
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import economist_churn_fatigue_model as ecm  # noqa: E402


def encode_with_saved(series, encoder):
    """LabelEncoder.transform without crashing on labels unseen at training time."""
    mapping = {label: i for i, label in enumerate(encoder.classes_)}
    fallback = mapping.get("Unknown", np.nan)
    return series.fillna("Unknown").astype(str).map(lambda v: mapping.get(v, fallback))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transactions", default=ecm.TRANSACTIONS_PATH)
    ap.add_argument("--articles", default=ecm.ARTICLES_PATH)
    ap.add_argument("--customer-features", default=ecm.CUSTOMER_FEATURES_PATH)
    ap.add_argument("--model", default=os.path.join(ecm.OUTPUT_DIR, "churn_fatigue_model.pkl"))
    ap.add_argument("--out", default=os.path.join(ecm.OUTPUT_DIR, "churn_customer_features.csv"))
    ap.add_argument("--as-of", choices=["latest", "training"], default="latest",
                    help="latest = day after the last transaction (score customers now); training = model cutoff date")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "feature_cols" not in bundle:
        sys.exit("Expected the dict bundle saved by economist_churn_fatigue_model.py (with 'feature_cols').")
    model, feature_cols = bundle["model"], list(bundle["feature_cols"])
    encoders = bundle.get("label_encoders", {}) or {}

    print("Loading transactions...")
    tx = ecm.load_transactions(args.transactions)
    print("Loading articles...")
    articles = ecm.load_articles(args.articles)

    if args.as_of == "training":
        cutoff = pd.to_datetime(bundle["cutoff_date"])
    else:
        cutoff = tx["t_dat"].max() + timedelta(days=1)
    print(f"Feature date (as of): {cutoff.date()}  |  model trained at cutoff {bundle.get('cutoff_date')}")

    print("Engineering fatigue features...")
    feats = ecm.build_fatigue_features(tx, articles, cutoff, ecm.RECENT_MOMENTUM_DAYS)
    feats = ecm.load_and_merge_customer_features(args.customer_features, feats)

    if "rfm_segment" in feats.columns and "rfm_segment" in encoders:
        feats["rfm_segment_enc"] = encode_with_saved(feats["rfm_segment"], encoders["rfm_segment"])

    absent = [c for c in feature_cols if c not in feats.columns]
    if absent:
        print(f"WARNING: features missing from the export inputs, filled with the median (NaN if no data): {absent}")
        for c in absent:
            feats[c] = np.nan

    X = feats[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())  # same rule as training
    out = pd.concat([feats[["customer_id"]].reset_index(drop=True), X.reset_index(drop=True).astype("float32")], axis=1)
    out["as_of_date"] = str(cutoff.date())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.6g")

    proba = model.predict_proba(X[feature_cols])[:, 1]
    print(f"\nSaved {len(out):,} customers x {len(feature_cols)} features to {args.out}")
    print(f"Sanity check: mean churn probability {proba.mean():.3f}, share >= 0.5: {(proba >= 0.5).mean():.1%}")


if __name__ == "__main__":
    main()

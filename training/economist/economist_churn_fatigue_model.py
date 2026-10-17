"""
AuraSync — Economist Agent — Model 1: Style Fatigue / Churn Prediction
========================================================================

Adapts the classic "telco churn" methodology to H&M transaction data to
predict whether a customer is about to go quiet ("style fatigue") —
i.e. stop purchasing despite an established buying history.

DATA EXPECTED
-------------
1) transactions_train.csv (H&M Kaggle dataset)
   columns: t_dat, customer_id, article_id, price, sales_channel_id

2) articles.csv (H&M Kaggle dataset)
   columns (subset used): article_id, product_group_name, product_type_name,
   department_name, section_name, garment_group_name, colour_group_name

3) customer_features.csv (output of the Retail Strategist agent — RFM/CLTV)
   Exact column names are auto-detected (see `detect_column` below) so this
   script is robust even if the Retail Strategist's naming differs slightly
   from what's assumed here. Check the printed column-mapping report on run.

LABEL CONSTRUCTION
-------------------
A cutoff date splits history into a FEATURE WINDOW (everything before cutoff)
and a LABEL WINDOW (the PREDICTION_WINDOW_DAYS after cutoff). A customer who
purchased at least once in the feature window is "churned" (label=1) if they
have zero transactions in the label window, else "retained" (label=0).

CUTOFF_DATE defaults to (max transaction date - PREDICTION_WINDOW_DAYS).
Adjust CUTOFF_DATE manually below if you want a specific date instead
(e.g. to match the exact evaluation window used elsewhere in the project).

USAGE
-----
    python economist_churn_fatigue_model.py

Edit the CONFIG block below to point at your real file paths before running.
"""

import os
import pickle
import warnings
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit these paths / params for your environment
# ============================================================
TRANSACTIONS_PATH = "/Users/mac/Downloads/transactions_train.csv"
ARTICLES_PATH = "/Users/mac/Downloads/articles.csv"
CUSTOMER_FEATURES_PATH = "/Users/mac/Downloads/customer_features.csv"

OUTPUT_DIR = "/Users/mac/Downloads/aurasync_economist_outputs"

PREDICTION_WINDOW_DAYS = 56  # 8-week "did they come back?" window
CUTOFF_DATE = None  # set e.g. "2020-07-28" to override the auto cutoff
RECENT_MOMENTUM_DAYS = 60  # window used for the fatigue/momentum ratio
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparameter search
N_ITER_SEARCH = 40  # RandomizedSearchCV candidates; raise for a more thorough (slower) search
CV_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def detect_column(df, candidates, required=True, default=None):
    """
    Finds the first column in `df` whose name contains any of the
    substrings in `candidates` (case-insensitive). Lets this script
    survive naming drift in customer_features.csv without editing code.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lower_name, actual_name in cols_lower.items():
            if cand in lower_name:
                return actual_name
    if required:
        raise KeyError(
            f"Could not find a column matching any of {candidates} in "
            f"customer_features.csv. Available columns: {list(df.columns)}"
        )
    return default


def load_transactions(path):
    df = pd.read_csv(
        path,
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype={"customer_id": "str", "article_id": "int64", "price": "float32"},
    )
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    return df


def load_articles(path):
    use_cols = [
        "article_id",
        "product_group_name",
        "product_type_name",
        "department_name",
        "section_name",
        "garment_group_name",
        "colour_group_name",
    ]
    df = pd.read_csv(path)
    keep = [c for c in use_cols if c in df.columns]
    missing = set(use_cols) - set(keep)
    if missing:
        print(f"[articles.csv] Missing expected columns (continuing without them): {missing}")
    return df[keep]


# ============================================================
# Feature engineering
# ============================================================
def build_churn_labels(tx, cutoff_date, prediction_window_days):
    """Returns a DataFrame [customer_id, churn] for customers active pre-cutoff."""
    label_end = cutoff_date + timedelta(days=prediction_window_days)

    pre_cutoff_customers = tx.loc[tx["t_dat"] < cutoff_date, "customer_id"].unique()
    label_window_customers = set(
        tx.loc[
            (tx["t_dat"] >= cutoff_date) & (tx["t_dat"] < label_end), "customer_id"
        ].unique()
    )

    labels = pd.DataFrame({"customer_id": pre_cutoff_customers})
    labels["churn"] = (~labels["customer_id"].isin(label_window_customers)).astype(int)
    return labels


def build_fatigue_features(tx, articles, cutoff_date, recent_momentum_days):
    """
    Builds per-customer features from pre-cutoff transaction history:
    recency, frequency, monetary, tenure, purchase cadence, category
    concentration/diversity, and a momentum ratio that acts as the core
    'style fatigue' signal (declining recent activity vs. their own baseline).
    """
    hist = tx.loc[tx["t_dat"] < cutoff_date].copy()

    if not articles.empty and "product_group_name" in articles.columns:
        hist = hist.merge(articles[["article_id", "product_group_name"]], on="article_id", how="left")
        hist["product_group_name"] = hist["product_group_name"].fillna("Unknown")
    else:
        hist["product_group_name"] = "Unknown"

    grp = hist.groupby("customer_id")

    agg = grp.agg(
        frequency=("t_dat", "count"),
        monetary=("price", "sum"),
        avg_basket_value=("price", "mean"),
        first_purchase=("t_dat", "min"),
        last_purchase=("t_dat", "max"),
        category_diversity=("product_group_name", "nunique"),
    ).reset_index()

    agg["recency_days"] = (cutoff_date - agg["last_purchase"]).dt.days
    agg["tenure_days"] = (cutoff_date - agg["first_purchase"]).dt.days
    agg["avg_days_between_purchases"] = agg["tenure_days"] / agg["frequency"].clip(lower=1)

    # Top-category purchase share (concentration) — high concentration + decline = fatigue risk
    cat_counts = hist.groupby(["customer_id", "product_group_name"]).size().reset_index(name="n")
    top_cat = cat_counts.sort_values("n", ascending=False).drop_duplicates("customer_id")
    total_per_cust = cat_counts.groupby("customer_id")["n"].sum().reset_index(name="total_n")
    top_cat = top_cat.merge(total_per_cust, on="customer_id")
    top_cat["top_category_share"] = top_cat["n"] / top_cat["total_n"]
    agg = agg.merge(
        top_cat[["customer_id", "top_category_share"]], on="customer_id", how="left"
    )

    # Momentum / fatigue ratio: recent activity vs. the equivalent prior window.
    # ratio < 1 means the customer is buying less than their own baseline rate —
    # this is the direct proxy for "style fatigue" setting in.
    recent_start = cutoff_date - timedelta(days=recent_momentum_days)
    prior_start = cutoff_date - timedelta(days=2 * recent_momentum_days)

    recent_counts = (
        hist.loc[hist["t_dat"] >= recent_start]
        .groupby("customer_id")
        .size()
        .rename("recent_n")
    )
    prior_counts = (
        hist.loc[(hist["t_dat"] >= prior_start) & (hist["t_dat"] < recent_start)]
        .groupby("customer_id")
        .size()
        .rename("prior_n")
    )

    agg = agg.merge(recent_counts, on="customer_id", how="left").merge(
        prior_counts, on="customer_id", how="left"
    )
    agg["recent_n"] = agg["recent_n"].fillna(0)
    agg["prior_n"] = agg["prior_n"].fillna(0)
    # +1 smoothing so customers with a quiet earlier window don't cause div/0 or infinite ratios
    agg["momentum_ratio"] = (agg["recent_n"] + 1) / (agg["prior_n"] + 1)

    agg = agg.drop(columns=["first_purchase", "last_purchase", "recent_n", "prior_n"])
    return agg


def load_and_merge_customer_features(path, engineered_df):
    if not os.path.exists(path):
        print(
            f"\n[customer_features.csv] Not found at: {path}\n"
            "  This file isn't part of the raw H&M Kaggle download — it's the output of your\n"
            "  own Retail Strategist agent. Skipping the RFM/CLTV merge; the churn model will\n"
            "  train on transaction-derived features only (recency, frequency, monetary, tenure,\n"
            "  category concentration, momentum ratio — RFM/CLTV columns just won't be added).\n"
            "  Set CUSTOMER_FEATURES_PATH to wherever you saved that agent's output to include them.\n"
        )
        return engineered_df

    cf = pd.read_csv(path)
    cust_id_col = detect_column(cf, ["customer_id", "customer id", "cust_id"])
    recency_col = detect_column(cf, ["recency"], required=False)
    frequency_col = detect_column(cf, ["frequency"], required=False)
    monetary_col = detect_column(cf, ["monetary"], required=False)
    cltv_col = detect_column(cf, ["cltv"], required=False)
    segment_col = detect_column(cf, ["segment"], required=False)

    rename_map = {cust_id_col: "customer_id"}
    keep_cols = ["customer_id"]
    for col, new_name in [
        (recency_col, "rfm_recency"),
        (frequency_col, "rfm_frequency"),
        (monetary_col, "rfm_monetary"),
        (cltv_col, "cltv"),
        (segment_col, "rfm_segment"),
    ]:
        if col is not None:
            rename_map[col] = new_name
            keep_cols.append(new_name)

    cf = cf.rename(columns=rename_map)[keep_cols]

    print("\n[customer_features.csv] Column mapping used:")
    for original, new in rename_map.items():
        print(f"    {original!r} -> {new!r}")

    merged = engineered_df.merge(cf, on="customer_id", how="left")
    return merged


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("Loading transactions...")
    tx = load_transactions(TRANSACTIONS_PATH)

    print("Loading articles...")
    articles = load_articles(ARTICLES_PATH)

    cutoff_date = (
        pd.to_datetime(CUTOFF_DATE)
        if CUTOFF_DATE
        else tx["t_dat"].max() - timedelta(days=PREDICTION_WINDOW_DAYS)
    )
    print(f"Cutoff date: {cutoff_date.date()}  |  Prediction window: {PREDICTION_WINDOW_DAYS} days")

    print("Building churn labels...")
    labels = build_churn_labels(tx, cutoff_date, PREDICTION_WINDOW_DAYS)
    print(f"  Customers active pre-cutoff: {len(labels):,}")
    print(f"  Churn rate: {labels['churn'].mean():.3%}")

    print("Engineering fatigue features...")
    features = build_fatigue_features(tx, articles, cutoff_date, RECENT_MOMENTUM_DAYS)

    print("Merging with customer_features.csv (RFM/CLTV)...")
    features = load_and_merge_customer_features(CUSTOMER_FEATURES_PATH, features)

    data = labels.merge(features, on="customer_id", how="left")

    # Encode the optional RFM segment column if present
    label_encoders = {}
    if "rfm_segment" in data.columns:
        data["rfm_segment"] = data["rfm_segment"].fillna("Unknown").astype(str)
        le = LabelEncoder()
        data["rfm_segment_enc"] = le.fit_transform(data["rfm_segment"])
        label_encoders["rfm_segment"] = le

    feature_cols = [
        c
        for c in data.columns
        if c not in ["customer_id", "churn", "rfm_segment"]
    ]
    # Numeric-only safety net: coerce, then fill remaining NaNs with column median
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = data["churn"]

    print(f"\nFinal feature set ({len(feature_cols)} features): {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\nTraining baseline LightGBM classifier (hand-set hyperparameters)...")
    baseline_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,
    )
    baseline_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_proba)
    print(f"Baseline ROC-AUC (held-out test set): {baseline_auc:.4f}")

    print(f"\nRunning RandomizedSearchCV ({N_ITER_SEARCH} candidates x {CV_FOLDS}-fold CV)...")
    param_distributions = {
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(
            objective="binary", random_state=RANDOM_STATE, class_weight="balanced", verbose=-1
        ),
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    tuned_model = search.best_estimator_
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, tuned_proba)

    print("\n=== Baseline vs. tuned (held-out test set) ===")
    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Tuned ROC-AUC:    {tuned_auc:.4f}")
    print(f"Change:           {tuned_auc - baseline_auc:+.4f}")

    model = tuned_model  # the tuned model is what gets evaluated below and saved

    print("\n=== Evaluation (tuned model) ===")
    y_pred = model.predict(X_test)
    y_proba = tuned_proba

    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    print("\nTop feature importances:")
    print(importance.head(10).to_string())

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    model_path = os.path.join(OUTPUT_DIR, "churn_fatigue_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "label_encoders": label_encoders,
                "cutoff_date": str(cutoff_date.date()),
                "prediction_window_days": PREDICTION_WINDOW_DAYS,
                "baseline_auc": baseline_auc,
                "tuned_auc": tuned_auc,
                "best_params": search.best_params_,
            },
            f,
        )
    print(f"\nSaved model bundle to {model_path}")


if __name__ == "__main__":
    main()

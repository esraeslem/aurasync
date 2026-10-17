"""
AuraSync — Retail Strategist Agent — RFM Segmentation + CLTV Scoring
========================================================================

Produces `customer_features.csv`, consumed downstream by the Economist
Agent's churn/fatigue model (and usable for RFM-based recommendation logic
elsewhere in the project).

Two complementary analyses, kept in clearly separate columns so they don't
get confused with each other (they use different definitions of "recency"
and "frequency"):

1. CLASSIC RFM  (columns: recency, frequency, monetary, *_score, RFM_SCORE, segment)
   - recency   = days since last purchase (as of the analysis date)
   - frequency = total number of distinct purchase days
   - monetary  = total amount spent
   - Each scored 1-5 via quintiles, combined into a segment label using the
     standard 10-segment RFM map (champions, at_risk, hibernating, etc.)

2. CLTV via BG/NBD + Gamma-Gamma  (columns prefixed bgnbd_ / gg_, plus CLTV, cltv_segment)
   - This is the rigorous, probabilistic approach (Miuul's CLTV module):
     BG/NBD models purchase frequency + churn probability, Gamma-Gamma
     models expected transaction value, and the two are combined into a
     forecasted customer lifetime value over CLTV_TIME_MONTHS.
   - Note: BG/NBD's own "frequency"/"recency"/"T" are DIFFERENT quantities
     from the classic RFM ones above (repeat-purchase counts and customer
     age, not raw purchase counts and days-since-last-purchase) — that's
     why they get distinct bgnbd_/gg_ prefixes here rather than overwriting
     the classic RFM columns.

DATA EXPECTED
-------------
transactions_train.csv (H&M Kaggle dataset)
  columns: t_dat, customer_id, article_id, price, sales_channel_id

USAGE
-----
    python retail_strategist_rfm_cltv.py

Edit the CONFIG block below to point at your real file path before running.
"""

import os
import warnings

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit these paths / params for your environment
# ============================================================
TRANSACTIONS_PATH = "/Users/mac/Downloads/transactions_train.csv"
OUTPUT_DIR = "/Users/mac/Downloads/aurasync_retail_strategist_outputs"
OUTPUT_FILENAME = "customer_features.csv"

ANALYSIS_DATE = None  # set e.g. "2020-09-23" to override; defaults to max(t_dat) + 1 day
CLTV_TIME_MONTHS = 6  # forecast horizon for expected purchases / CLTV
CLTV_FREQ = "W"  # time unit lifetimes works in ("W" = weekly, standard for ~1-2yr histories)
CLTV_DISCOUNT_RATE = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Data loading
# ============================================================
def load_transactions(path):
    df = pd.read_csv(
        path,
        usecols=["t_dat", "customer_id", "price"],
        dtype={"customer_id": "str", "price": "float32"},
    )
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    return df


# ============================================================
# 1. Classic RFM
# ============================================================
def build_classic_rfm(tx, analysis_date):
    grp = tx.groupby("customer_id").agg(
        last_purchase=("t_dat", "max"),
        frequency=("t_dat", lambda s: s.nunique()),  # distinct purchase days
        monetary=("price", "sum"),
    )
    grp["recency"] = (analysis_date - grp["last_purchase"]).dt.days
    rfm = grp.reset_index()[["customer_id", "recency", "frequency", "monetary"]]

    # Quintile scores. Recency is inverted: fewer days since last purchase = better = score 5.
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # Standard 10-segment RFM map (recency_score + frequency_score as a 2-digit regex key)
    seg_map = {
        r"[1-2][1-2]": "hibernating",
        r"[1-2][3-4]": "at_risk",
        r"[1-2]5": "cant_lose",
        r"3[1-2]": "about_to_sleep",
        r"33": "need_attention",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions",
    }
    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

    return rfm


# ============================================================
# 2. CLTV via BG/NBD + Gamma-Gamma
# ============================================================
def build_cltv(tx, analysis_date, time_months, freq, discount_rate):
    cltv_df = summary_data_from_transaction_data(
        tx,
        customer_id_col="customer_id",
        datetime_col="t_dat",
        monetary_value_col="price",
        observation_period_end=analysis_date,
        freq=freq,
    )
    cltv_df = cltv_df.rename(
        columns={
            "frequency": "bgnbd_frequency",
            "recency": "bgnbd_recency",
            "T": "bgnbd_T",
            "monetary_value": "gg_monetary_value",
        }
    )

    # BG/NBD needs repeat purchasers (bgnbd_frequency > 0) to fit meaningfully;
    # Gamma-Gamma additionally needs a positive monetary value on those repeats.
    fit_mask = (cltv_df["bgnbd_frequency"] > 0) & (cltv_df["gg_monetary_value"] > 0)
    fittable = cltv_df.loc[fit_mask]

    print(f"  Customers with >=1 repeat purchase (fittable for CLTV): {len(fittable):,} / {len(cltv_df):,}")

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(fittable["bgnbd_frequency"], fittable["bgnbd_recency"], fittable["bgnbd_T"])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(fittable["bgnbd_frequency"], fittable["gg_monetary_value"])

    time_periods = time_months * 4.345  # approx weeks per month, matches CLTV_FREQ="W"

    cltv_df["expected_purchases"] = 0.0
    cltv_df["expected_average_profit"] = 0.0
    cltv_df["CLTV"] = 0.0

    cltv_df.loc[fit_mask, "expected_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        time_periods,
        fittable["bgnbd_frequency"],
        fittable["bgnbd_recency"],
        fittable["bgnbd_T"],
    )
    cltv_df.loc[fit_mask, "expected_average_profit"] = ggf.conditional_expected_average_profit(
        fittable["bgnbd_frequency"], fittable["gg_monetary_value"]
    )
    cltv_df.loc[fit_mask, "CLTV"] = ggf.customer_lifetime_value(
        bgf,
        fittable["bgnbd_frequency"],
        fittable["bgnbd_recency"],
        fittable["bgnbd_T"],
        fittable["gg_monetary_value"],
        time=time_months,
        freq=freq,
        discount_rate=discount_rate,
    )

    # Single-purchase customers get CLTV=0 here (unfit for BG/NBD) rather than NaN —
    # a conservative placeholder, not a modeled prediction. Flag them if you'd rather
    # exclude them from downstream CLTV-based logic entirely.
    cltv_df = cltv_df.reset_index()

    # 4-tier CLTV segment (A = top). Computed only over customers with CLTV > 0
    # so single-purchase customers don't distort the quartile cuts, then backfilled as "D".
    cltv_df["cltv_segment"] = "D"
    scored = cltv_df.loc[cltv_df["CLTV"] > 0, "CLTV"]
    if len(scored) >= 4:
        cltv_df.loc[scored.index, "cltv_segment"] = pd.qcut(
            scored, 4, labels=["D", "C", "B", "A"]
        ).astype(str)

    return cltv_df


# ============================================================
# Main pipeline
# ============================================================
def main():
    print("Loading transactions...")
    tx = load_transactions(TRANSACTIONS_PATH)

    analysis_date = pd.to_datetime(ANALYSIS_DATE) if ANALYSIS_DATE else tx["t_dat"].max() + pd.Timedelta(days=1)
    print(f"Analysis date: {analysis_date.date()}")

    print("\nBuilding classic RFM...")
    rfm = build_classic_rfm(tx, analysis_date)
    print(f"  Customers scored: {len(rfm):,}")
    print(rfm["segment"].value_counts())

    print("\nFitting BG/NBD + Gamma-Gamma for CLTV...")
    cltv = build_cltv(tx, analysis_date, CLTV_TIME_MONTHS, CLTV_FREQ, CLTV_DISCOUNT_RATE)

    print("\nMerging RFM + CLTV into customer_features.csv...")
    customer_features = rfm.merge(cltv, on="customer_id", how="left")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    customer_features.to_csv(output_path, index=False)
    print(f"\nSaved {len(customer_features):,} rows x {customer_features.shape[1]} columns to {output_path}")
    print(f"Columns: {list(customer_features.columns)}")


if __name__ == "__main__":
    main()

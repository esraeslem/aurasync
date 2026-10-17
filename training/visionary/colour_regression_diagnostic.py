"""
AuraSync — Visionary Agent
Colour Retrieval Regression Diagnostic
---------------------------------------------------------------------------
Investigates why Marqo-FashionSigLIP underperforms vanilla CLIP on
colour_group precision@5 (0.463 vs 0.502).

Three hypotheses tested:
  1. Type dominance   — does product_type_name separate much more cleanly
                         than colour_group in the SigLIP space? (mirrors the
                         "garment-type dominates aesthetic signal" finding
                         from the persona-clustering work)
  2. Uniform vs. concentrated — is the precision drop spread evenly across
                         colour groups, or driven by a handful of hard/
                         ambiguous ones (black/charcoal/navy-type confusions)?
  3. Failure mode     — when SigLIP's top-5 neighbors are wrong on colour,
                         are they at least right on product type? (i.e. is
                         it "same garment, wrong shade" or something messier)

Run locally:
    python colour_regression_diagnostic.py

Expects the standard AuraSync data layout:
    <DATA_DIR>/catalog_metadata.csv
    <DATA_DIR>/color_features.csv
    <DATA_DIR>/catalog_emb_marqo_siglip.npy
    <DATA_DIR>/catalog_emb_clip.npy      (optional — only needed for a
                                           direct side-by-side comparison;
                                           if missing, SigLIP-only
                                           diagnostics still run)

Override the data directory with the AURASYNC_DATA_DIR env var.

ASSUMPTIONS TO CHECK AGAINST YOUR ACTUAL SCHEMA before trusting the numbers:
  - color_features.csv has a column literally named "colour_group"
  - catalog_metadata.csv has a column literally named "product_type_name"
  - catalog_emb_marqo_siglip.npy rows are in the SAME ORDER as
    catalog_metadata.csv rows (this script re-derives alignment via
    article_id merge + positional index, but if your embedding file was
    saved in a different row order than catalog_metadata.csv, that merge
    will silently misalign things — the row-count assert below only
    catches a COUNT mismatch, not an ORDER mismatch)
  - vanilla CLIP embeddings, if you kept them, are named/pathed
    differently — adjust CLIP_EMB_PATH below
Adjust names/paths as needed; the printed output at the bottom is what's
worth sharing back for precise diagnosis.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads")

METADATA_PATH = os.path.join(DATA_DIR, "catalog_metadata.csv")
COLOR_PATH = "/Users/mac/Downloads/colour_group_lookup_coarse.csv"
SIGLIP_EMB_PATH = os.path.join(DATA_DIR, "catalog_emb_marqo_siglip.npy")
CLIP_EMB_PATH = os.path.join(DATA_DIR, "catalog_emb_clip.npy")

COLOUR_COL = "colour_group"
TYPE_COL = "product_type_name"

TOP_K = 5
SAMPLE_SIZE = 2000  # queries to sample for the nearest-neighbor diagnostics
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_data():
    meta = pd.read_csv(METADATA_PATH, dtype={"article_id": str})
    meta["article_id"] = meta["article_id"].str.zfill(10)

    colour = pd.read_csv(COLOR_PATH, dtype={"article_id": str})
    colour["article_id"] = colour["article_id"].str.zfill(10)

    siglip_emb = np.load(SIGLIP_EMB_PATH)

    clip_emb = None
    if os.path.exists(CLIP_EMB_PATH):
        clip_emb = np.load(CLIP_EMB_PATH, allow_pickle=True)
    else:
        print(
            f"[info] vanilla CLIP embeddings not found at {CLIP_EMB_PATH} "
            f"— running SigLIP-only diagnostics. Point CLIP_EMB_PATH at the "
            f"right file if you still have it saved, for a direct comparison."
        )

    return meta, colour, siglip_emb, clip_emb


# ---------------------------------------------------------------------------
# 2. Align embeddings to a merged (metadata + colour) frame
#    NOTE: this assumes emb row i corresponds to meta row i (positional).
#    If that's not true for your pipeline, this whole script's numbers
#    are meaningless — check that assumption first if results look odd.
# ---------------------------------------------------------------------------

def align(meta, colour, emb):
    assert len(meta) == emb.shape[0], (
        f"Row count mismatch: metadata has {len(meta)} rows, embeddings "
        f"have {emb.shape[0]} — fix alignment before trusting anything below."
    )
    meta = meta.reset_index(drop=True)
    meta["_row"] = np.arange(len(meta))
    merged = meta.merge(colour, on="article_id", how="inner")
    row_idx = merged["_row"].to_numpy()
    return merged.drop(columns="_row"), emb[row_idx]


# ---------------------------------------------------------------------------
# 3. Neighbor composition: precision@5 on colour AND on type, plus the
#    specific failure mode "same type, wrong colour"
# ---------------------------------------------------------------------------

def neighbor_composition(df, emb, top_k=TOP_K, sample_size=SAMPLE_SIZE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(df)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)

    sims = cosine_similarity(emb[sample_idx], emb)
    colour_labels = df[COLOUR_COL].to_numpy()
    type_labels = df[TYPE_COL].to_numpy()

    same_colour, same_type, same_type_diff_colour = [], [], []

    for row_pos, query_i in enumerate(sample_idx):
        sims[row_pos, query_i] = -np.inf  # exclude self-match
        top_idx = np.argpartition(-sims[row_pos], top_k)[:top_k]

        q_colour, q_type = colour_labels[query_i], type_labels[query_i]
        same_colour.append((colour_labels[top_idx] == q_colour).mean())
        same_type.append((type_labels[top_idx] == q_type).mean())
        same_type_diff_colour.append(
            ((type_labels[top_idx] == q_type) & (colour_labels[top_idx] != q_colour)).mean()
        )

    return {
        "precision@5_colour_group": float(np.mean(same_colour)),
        "precision@5_product_type": float(np.mean(same_type)),
        "same_type_diff_colour_rate": float(np.mean(same_type_diff_colour)),
    }


# ---------------------------------------------------------------------------
# 4. Silhouette: does the space separate cleanly by colour, or by type?
# ---------------------------------------------------------------------------

def silhouette_check(df, emb, sample_size=SAMPLE_SIZE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(df)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sub_emb = emb[sample_idx]

    colour_sil = silhouette_score(sub_emb, df[COLOUR_COL].to_numpy()[sample_idx], metric="cosine")
    type_sil = silhouette_score(sub_emb, df[TYPE_COL].to_numpy()[sample_idx], metric="cosine")

    return {"silhouette_colour_group": colour_sil, "silhouette_product_type": type_sil}


# ---------------------------------------------------------------------------
# 5. Is the drop uniform or concentrated in specific colour groups?
# ---------------------------------------------------------------------------

def per_colour_breakdown(df, emb, top_k=TOP_K, sample_size=SAMPLE_SIZE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(df)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)

    sims = cosine_similarity(emb[sample_idx], emb)
    colour_labels = df[COLOUR_COL].to_numpy()

    records = []
    for row_pos, query_i in enumerate(sample_idx):
        sims[row_pos, query_i] = -np.inf
        top_idx = np.argpartition(-sims[row_pos], top_k)[:top_k]
        q_colour = colour_labels[query_i]
        precision = (colour_labels[top_idx] == q_colour).mean()
        records.append({"colour_group": q_colour, "precision_at_5": precision})

    out = pd.DataFrame(records).groupby("colour_group")["precision_at_5"].agg(["mean", "count"])
    return out.sort_values("mean")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def report(label, df, emb):
    print("=" * 70)
    print(label)
    print("=" * 70)
    print(neighbor_composition(df, emb))
    print(silhouette_check(df, emb))
    print("\nWorst colour groups (lowest precision@5, count >= 5):")
    breakdown = per_colour_breakdown(df, emb)
    print(breakdown[breakdown["count"] >= 5].head(10))
    print()


def main():
    meta, colour, siglip_emb, clip_emb = load_data()

    df_s, emb_s = align(meta, colour, siglip_emb)
    report("SigLIP embedding space", df_s, emb_s)

    if clip_emb is not None:
        df_c, emb_c = align(meta, colour, clip_emb)
        report("Vanilla CLIP embedding space (for comparison)", df_c, emb_c)


if __name__ == "__main__":
    main()

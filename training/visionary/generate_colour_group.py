"""
AuraSync — Visionary Agent
Recover the colour_group ground-truth column
---------------------------------------------------------------------------
The colour_regression_diagnostic.py script failed with:
    KeyError: 'colour_group'

That's expected — color_features.csv holds raw extracted pixel colour data
(from the np.median-on-foreground-pixels approach), not a categorical
ground-truth label. The H&M "Personalized Fashion Recommendations" dataset
ships an OFFICIAL colour taxonomy on articles.csv:

    colour_group_name              — fine-grained (~50 values, e.g.
                                      "Dark Blue", "Light Pink")
    perceived_colour_master_name   — coarser (~20 values, e.g.
                                      "Blue", "Pink", "Black")

Whichever of these generated your original 0.463 / 0.502 numbers is the
one you need to reproduce here — inventing a new RGB-bucket scheme would
silently compare against different ground truth and invalidate the numbers.

This script searches, in priority order:
    1. articles.csv                (the original raw H&M file, if still present)
    2. catalog_metadata.csv        (your bridge file — may have carried a
                                     colour field over already)
    3. color_features.csv          (unlikely, but checked for completeness)

for a column matching known H&M colour-field names, and writes a clean
lookup file:
    colour_group_lookup.csv   [article_id, colour_group]

Run locally:
    python generate_colour_group.py

Override the data directory with the AURASYNC_DATA_DIR env var.

IMPORTANT: if colour_group_name is found, this also prints how many
distinct values it has. If that number is large (~50), that granularity
alone could be inflating the apparent regression — the per-colour-group
breakdown in colour_regression_diagnostic.py will show whether the drop is
concentrated in near-duplicate fine-grained buckets (e.g. "Dark Blue" vs
"Blue" counted as different groups). If so, rerunning against
perceived_colour_master_name instead is worth trying — this script writes
a lookup for both when both are available, so you can compare.
"""

import os
import pandas as pd

DATA_DIR = os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads")

ARTICLES_PATH = os.path.join(DATA_DIR, "articles.csv")
METADATA_PATH = os.path.join(DATA_DIR, "catalog_metadata.csv")
COLOR_FEATURES_PATH = os.path.join(DATA_DIR, "color_features.csv")

OUT_PATH_FINE = os.path.join(DATA_DIR, "colour_group_lookup.csv")
OUT_PATH_COARSE = os.path.join(DATA_DIR, "colour_group_lookup_coarse.csv")

FINE_CANDIDATES = ["colour_group_name", "colour_group", "color_group_name", "color_group"]
COARSE_CANDIDATES = ["perceived_colour_master_name", "perceived_color_master_name"]


def load_if_exists(path):
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"article_id": str})
        df["article_id"] = df["article_id"].str.zfill(10)
        return df
    return None


def find_column(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def report_colour_like_columns(label, df):
    if df is None:
        print(f"  [{label}] file not found — skipped")
        return
    hits = [c for c in df.columns if "colour" in c.lower() or "color" in c.lower()]
    print(f"  [{label}] columns: {list(df.columns)}")
    if hits:
        print(f"  [{label}] colour-like columns found: {hits}")


def main():
    articles = load_if_exists(ARTICLES_PATH)
    metadata = load_if_exists(METADATA_PATH)
    color_feat = load_if_exists(COLOR_FEATURES_PATH)

    print("=" * 70)
    print("Scanning for colour ground-truth columns")
    print("=" * 70)
    report_colour_like_columns("articles.csv", articles)
    report_colour_like_columns("catalog_metadata.csv", metadata)
    report_colour_like_columns("color_features.csv", color_feat)
    print()

    sources = [("articles.csv", articles), ("catalog_metadata.csv", metadata),
               ("color_features.csv", color_feat)]

    fine_col, fine_source, fine_df = None, None, None
    coarse_col, coarse_source, coarse_df = None, None, None

    for name, df in sources:
        if fine_col is None:
            col = find_column(df, FINE_CANDIDATES)
            if col:
                fine_col, fine_source, fine_df = col, name, df
        if coarse_col is None:
            col = find_column(df, COARSE_CANDIDATES)
            if col:
                coarse_col, coarse_source, coarse_df = col, name, df

    if fine_col is None and coarse_col is None:
        print(
            "No known colour ground-truth column found in any of the three "
            "files. Check the 'colour-like columns found' lines above for "
            "anything named differently, and tell me the exact column + "
            "file so I can point this at it directly. "
            "If articles.csv genuinely isn't on disk anymore, it can be "
            "re-downloaded from the H&M Kaggle competition page."
        )
        return

    if fine_col:
        lookup = fine_df[["article_id", fine_col]].rename(columns={fine_col: "colour_group"})
        lookup = lookup.drop_duplicates(subset="article_id")
        lookup.to_csv(OUT_PATH_FINE, index=False)
        print(f"Found fine-grained colour column '{fine_col}' in {fine_source}.")
        print(f"  -> wrote {OUT_PATH_FINE} ({len(lookup)} rows)")
        print(f"  -> {lookup['colour_group'].nunique()} distinct values")
        print(lookup["colour_group"].value_counts().head(15))
        print()

    if coarse_col:
        lookup_c = coarse_df[["article_id", coarse_col]].rename(columns={coarse_col: "colour_group"})
        lookup_c = lookup_c.drop_duplicates(subset="article_id")
        lookup_c.to_csv(OUT_PATH_COARSE, index=False)
        print(f"Found coarse colour column '{coarse_col}' in {coarse_source}.")
        print(f"  -> wrote {OUT_PATH_COARSE} ({len(lookup_c)} rows)")
        print(f"  -> {lookup_c['colour_group'].nunique()} distinct values")
        print(lookup_c["colour_group"].value_counts().head(15))
        print()

    print(
        "Next step: point COLOR_PATH in colour_regression_diagnostic.py at "
        f"'{OUT_PATH_FINE if fine_col else OUT_PATH_COARSE}' and rerun. "
        "If you're not sure which granularity matches your original "
        "0.463/0.502 numbers, try both — if the coarse version narrows the "
        "gap significantly, granularity (not embedding quality) is a "
        "meaningful part of the story."
    )


if __name__ == "__main__":
    main()

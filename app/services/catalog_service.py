"""Catalog lookups shared by every agent: article metadata, sampling by product type, item-level style persona."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from ._io import clean, find_col, norm_article_id, safe_csv

ARTICLE_COLS = [
    "article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
    "perceived_colour_master_name", "graphical_appearance_name", "department_name", "index_name",
    "section_name", "garment_group_name", "detail_desc",
]


class CatalogService:
    def __init__(self, articles_path, metadata_path=None, personas_path=None):
        df = safe_csv(articles_path, dtype={"article_id": str}, usecols=lambda c: c in ARTICLE_COLS)
        if df is None and metadata_path is not None:
            df = safe_csv(metadata_path, dtype={"article_id": str})
        if df is not None and "article_id" in df.columns:
            df["article_id"] = df["article_id"].map(norm_article_id)
            df = df.drop_duplicates("article_id").set_index("article_id", drop=False)
            self._by_type = df.groupby("product_type_name").indices if "product_type_name" in df.columns else {}
        else:
            df, self._by_type = None, {}
        self.articles = df

        self.personas = None
        p = safe_csv(personas_path, dtype=str) if personas_path is not None else None
        if p is not None:
            aid = find_col(p, "article_id")
            if aid:
                p = p.rename(columns={aid: "article_id"})
                p["article_id"] = p["article_id"].map(norm_article_id)
                self.persona_id_col = find_col(p, "style_persona", "persona_cluster", "persona", exclude=("article_id",))
                self.persona_name_col = find_col(p, "style_persona_name", "persona_label", "persona_name")
                if self.persona_id_col == self.persona_name_col:
                    self.persona_id_col = None
                self.personas = p.drop_duplicates("article_id").set_index("article_id")

        self.load_status = {"articles": self.articles is not None, "style_personas": self.personas is not None}

    # ------------------------------------------------------------------
    def get(self, article_id):
        if self.articles is None:
            raise RuntimeError("Article catalog (articles.csv / catalog_metadata.csv) is not available.")
        aid = norm_article_id(article_id)
        if aid not in self.articles.index:
            return None
        return {k: clean(v) for k, v in self.articles.loc[aid].to_dict().items()}

    def describe(self, article_id):
        """Compact, LLM-friendly view of one article (None if unknown)."""
        row = self.get(article_id)
        if row is None:
            return None
        out = {
            "article_id": row.get("article_id"),
            "name": row.get("prod_name"),
            "product_type": row.get("product_type_name"),
            "product_group": row.get("product_group_name"),
            "colour": row.get("colour_group_name"),
            "colour_family": row.get("perceived_colour_master_name"),
            "pattern": row.get("graphical_appearance_name"),
            "department": row.get("department_name"),
            "index": row.get("index_name"),
            "section": row.get("section_name"),
            "garment_group": row.get("garment_group_name"),
            "description": row.get("detail_desc"),
        }
        return {k: v for k, v in out.items() if v is not None}

    def short(self, article_id):
        row = self.get(article_id) or {}
        return {
            "article_id": norm_article_id(article_id),
            "name": row.get("prod_name"),
            "product_type": row.get("product_type_name"),
            "colour": row.get("colour_group_name"),
        }

    def sample_by_type(self, product_type, prefer_index=None, exclude=(), n=1, seed_key=""):
        """Deterministically pick real catalog articles of a product type, preferring the same index (e.g. Ladieswear)."""
        if self.articles is None:
            return []
        pos = self._by_type.get(str(product_type).strip())
        if pos is None or len(pos) == 0:
            return []
        subset = self.articles.iloc[pos]
        if exclude:
            subset = subset[~subset.index.isin([norm_article_id(e) for e in exclude])]
        if prefer_index and "index_name" in subset.columns:
            same = subset[subset["index_name"] == prefer_index]
            if len(same):
                subset = same
        if subset.empty:
            return []
        rng = np.random.default_rng(zlib.crc32(f"{seed_key}|{product_type}".encode()))
        take = rng.choice(len(subset), size=min(n, len(subset)), replace=False)
        return subset.index[np.sort(take)].tolist()

    def get_persona(self, article_id):
        if self.personas is None:
            raise RuntimeError("Style persona assignments (style_personas.csv) are not available.")
        aid = norm_article_id(article_id)
        if aid not in self.personas.index:
            return None
        row = self.personas.loc[aid]
        return {
            "article_id": aid,
            "style_persona": clean(row[self.persona_id_col]) if self.persona_id_col else None,
            "style_persona_name": clean(row[self.persona_name_col]) if self.persona_name_col else None,
        }

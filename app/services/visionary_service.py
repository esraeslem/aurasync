from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ._io import LOAD_ERRORS, norm_article_id, safe_csv


class VisionaryService:
    def __init__(self, embeddings_path, ids_path, persona_path, colors_path):
        self.embeddings = self._load_npy(embeddings_path)
        self.article_ids = self._load_ids(ids_path)
        self.personas = safe_csv(persona_path)
        self.colors = safe_csv(colors_path, dtype={"article_id": str})
        if self.colors is not None and "article_id" in self.colors.columns:
            self.colors["article_id"] = self.colors["article_id"].map(norm_article_id)
            self.colors = self.colors.drop_duplicates("article_id").set_index("article_id", drop=False)
        aligned = not (self.embeddings is not None and self.article_ids is not None and len(self.embeddings) != len(self.article_ids))
        self._unit = None
        self._pos = {}
        if self.embeddings is not None and self.article_ids is not None and aligned:
            emb = self.embeddings.astype(np.float32, copy=False)
            self._unit = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            self._pos = {aid: i for i, aid in enumerate(self.article_ids)}
            self.embeddings = self._unit  # keep a single (normalized) copy in memory
        self.load_status = {
            "embeddings": self.embeddings is not None,
            "article_ids": self.article_ids is not None,
            "embeddings_aligned": aligned,
            "customer_personas_legacy": self.personas is not None,
            "colors": self.colors is not None,
        }

    @staticmethod
    def _load_npy(path):
        path = Path(path)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception as e:
            LOAD_ERRORS[str(path)] = f"{type(e).__name__}: {e}"
            return None

    @staticmethod
    def _load_ids(path):
        df = safe_csv(path, dtype=str)
        if df is None:
            return None
        if df.empty:
            return []
        col = "article_id" if "article_id" in df.columns else df.columns[0]
        return [norm_article_id(x) for x in df[col].tolist()]

    def find_similar_articles(self, article_id, top_n=5):
        if self._unit is None:
            raise RuntimeError("Visionary embeddings and article-ID alignment are not available.")
        i = self._pos.get(norm_article_id(article_id))
        if i is None:
            return None
        top_n = max(1, min(int(top_n), 20))
        scores = self._unit @ self._unit[i]
        k = min(top_n + 1, len(scores))
        cand = np.argpartition(-scores, k - 1)[:k]
        cand = cand[np.argsort(-scores[cand])]
        recs = [{"article_id": self.article_ids[j], "similarity": round(float(scores[j]), 4)} for j in cand if j != i]
        return {"recommendations": recs[:top_n]}

    def get_color(self, article_id):
        if self.colors is None:
            raise RuntimeError("Color feature data (color_features.csv) is not available.")
        aid = norm_article_id(article_id)
        if aid not in self.colors.index:
            return None
        return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in self.colors.loc[aid].to_dict().items()}

    def get_persona(self, customer_id):
        if self.personas is None:
            raise RuntimeError("Persona assignments are not available.")
        cols = {c.lower(): c for c in self.personas.columns}
        cid = cols.get("customer_id")
        if not cid:
            return None
        rows = self.personas[self.personas[cid].astype(str) == str(customer_id)]
        if rows.empty:
            return None
        row = rows.iloc[0].to_dict()
        return {"customer_id": str(customer_id), "persona_cluster": row.get(cols.get("persona_cluster", "persona_cluster")),
                "persona_label": row.get(cols.get("persona_label", "persona_label")), "data": row}

    @staticmethod
    def extract_dominant_color(image_bytes):
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
        arr = np.asarray(im).reshape(-1, 3)
        # Median is stable for product photos and avoids requiring an extra CV dependency.
        return np.median(arr, axis=0).round().astype(int).tolist()

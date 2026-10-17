"""Economist: style-fatigue churn probability and second-hand resale price.

Both models are saved by the training code as dict bundles:
  churn  -> {"model", "feature_cols", "label_encoders", "cutoff_date", "prediction_window_days", ...}
  resale -> {"model", "artifacts", "feature_names"}   (Mercari TF-IDF + frequency-encoding pipeline)
Bare estimators are still accepted for backwards compatibility.
"""
from __future__ import annotations

import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ._io import clean, safe_csv, safe_pickle

DEFAULT_ITEM_CONDITION_ID = 2  # Mercari scale 1 (new) .. 5 (poor); 2 = "gently used", as in training notebook
DEFAULT_SHIPPING = 0  # 0 = buyer pays shipping
DEFAULT_BRAND = "H&M"


class MissingFeaturesError(LookupError):
    def __init__(self, missing, message):
        super().__init__(message)
        self.missing = missing


class EconomistService:
    def __init__(self, churn_path, resale_path, churn_features_path=None):
        churn = safe_pickle(churn_path)
        self.churn_meta = {}
        if isinstance(churn, dict):
            self.churn_model = churn.get("model")
            self.churn_feature_cols = list(churn.get("feature_cols") or getattr(self.churn_model, "feature_name_", []) or [])
            self.churn_meta = {k: clean(churn.get(k)) for k in ("cutoff_date", "prediction_window_days", "tuned_auc", "baseline_auc") if k in churn}
        else:
            self.churn_model = churn
            self.churn_feature_cols = list(getattr(churn, "feature_name_", []) or [])

        resale = safe_pickle(resale_path)
        if isinstance(resale, dict):
            self.resale_model = resale.get("model")
            self.resale_artifacts = resale.get("artifacts")
        else:
            self.resale_model, self.resale_artifacts = resale, None

        self._churn_features_path = churn_features_path
        self._churn_store = None
        self._store_lock = threading.Lock()
        store_exists = bool(churn_features_path) and Path(churn_features_path).exists()
        self.load_status = {
            "churn_model": self.churn_model is not None,
            "churn_feature_store": store_exists,
            "resale_model": self.resale_model is not None,
        }

    # ------------------------------------------------------------------ churn
    def _store(self):
        """Lazy-load the per-customer feature store (it can be large) on the first churn query."""
        if self._churn_store is None and self.load_status["churn_feature_store"]:
            with self._store_lock:
                if self._churn_store is None:
                    df = safe_csv(self._churn_features_path, dtype={"customer_id": str})
                    if df is not None:
                        self._churn_store = df.drop_duplicates("customer_id").set_index("customer_id")
                    else:
                        self.load_status["churn_feature_store"] = False
        return self._churn_store

    def predict_churn(self, payload):
        if self.churn_model is None:
            raise RuntimeError("Churn model artifact (churn_fatigue_model.pkl) is not available.")
        cid = str(payload["customer_id"]).strip()
        row, as_of = {}, None
        store = self._store()
        if store is not None and cid in store.index:
            stored = store.loc[cid].to_dict()
            as_of = stored.pop("as_of_date", None)
            row.update(stored)
        row.update({k: v for k, v in (payload.get("features") or {}).items() if v is not None})

        cols = self.churn_feature_cols or list(row)
        missing = [c for c in cols if c not in row or pd.isna(pd.to_numeric(row.get(c), errors="coerce"))]
        if missing:
            where = "is not in the churn feature store" if store is not None else "has no stored features (run 02_economist_agent/export_churn_features.py)"
            raise MissingFeaturesError(missing, f"Customer {cid} {where}; missing model features: {missing}")

        X = pd.DataFrame([{c: row[c] for c in cols}]).apply(pd.to_numeric, errors="coerce")
        p = float(self.churn_model.predict_proba(X)[:, 1][0]) if hasattr(self.churn_model, "predict_proba") else float(self.churn_model.predict(X)[0])

        drivers = {}
        imp = getattr(self.churn_model, "feature_importances_", None)
        if imp is not None and len(imp) == len(cols):
            for i in np.argsort(-np.asarray(imp))[:5]:
                drivers[cols[i]] = clean(row[cols[i]])
        return {
            "customer_id": cid,
            "churn_probability": round(p, 4),
            "prediction": int(p >= 0.5),
            "meaning": f"probability of no purchase in the next {self.churn_meta.get('prediction_window_days', 'N')} days",
            "features_as_of": clean(as_of),
            "top_feature_values": drivers,
            "model_info": self.churn_meta,
        }

    # ------------------------------------------------------------------ resale
    def predict_resale_price(self, payload, article_row=None):
        if self.resale_model is None:
            raise RuntimeError("Resale model artifact (resale_pricing_model.pkl) is not available.")
        if self.resale_artifacts and article_row is not None:
            return self._mercari_price(article_row, payload)
        if self.resale_artifacts:
            raise LookupError("This resale model scores catalog articles; the article was not found in articles.csv.")
        feature_names = getattr(self.resale_model, "feature_name_", None)
        features = payload.get("features") or {}
        if not features:
            raise LookupError("No features provided for the resale model.")
        X = pd.DataFrame([{n: features.get(n, np.nan) for n in feature_names} if feature_names else features])
        return float(self.resale_model.predict(X.apply(pd.to_numeric, errors="coerce"))[0])

    def _mercari_price(self, article_row, payload):
        """Map an H&M articles.csv row onto the Mercari feature space (mirrors build_query_row_from_article)."""
        a = self.resale_artifacts
        cond = int(payload.get("item_condition_id") or DEFAULT_ITEM_CONDITION_ID)
        ship = int(payload.get("shipping") if payload.get("shipping") is not None else DEFAULT_SHIPPING)
        brand = payload.get("brand_name") or DEFAULT_BRAND
        name = str(article_row.get("prod_name") or "")
        desc = str(article_row.get("detail_desc") or "") or "No description yet"
        gen, sub1, sub2 = (str(article_row.get(k) or "missing") for k in ("index_name", "product_group_name", "product_type_name"))

        numeric = np.array([[
            cond, ship, len(name), len(desc),
            a["brand_map"].get(brand, 0), a["gen_cat_map"].get(gen, 0),
            a["sub1_map"].get(sub1, 0), a["sub2_map"].get(sub2, 0),
        ]], dtype=float)
        X = sparse.hstack([
            a["desc_vectorizer"].transform([desc]),
            a["name_vectorizer"].transform([name]),
            sparse.csr_matrix(numeric),
        ], format="csr")
        expected = getattr(self.resale_model, "n_features_in_", X.shape[1])
        if X.shape[1] != expected:
            raise RuntimeError(f"Resale feature mismatch: built {X.shape[1]} columns, model expects {expected}.")
        with warnings.catch_warnings():  # model was fit on an unnamed sparse matrix; sklearn warns about names
            warnings.simplefilter("ignore", UserWarning)
            price = float(np.expm1(self.resale_model.predict(X)[0]))
        return {
            "estimated_price_usd": round(max(price, 0.0), 2),
            "assumptions": {"item_condition_id": cond, "condition_scale": "1=new .. 5=poor", "shipping_paid_by_seller": bool(ship), "brand": brand},
            "model_info": "LightGBM regressor trained on Mercari listings, log1p(price) target",
        }

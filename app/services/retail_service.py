"""Retail Strategist: RFM/CLTV profile lookup and ARL "Complete the Look" recommendations.

ARL rules were mined at product_type_name granularity (article level is too sparse), so a
recommendation bridges article_id -> product_type_name -> rules -> consequent types -> real articles.
"""
from __future__ import annotations

import re

import pandas as pd

from ._io import clean, find_col, safe_csv


def _parse_itemset(value) -> frozenset:
    """'Trousers, Blouse', "frozenset({'Bra'})" or "(Bra)" -> frozenset of product types."""
    s = str(value)
    s = re.sub(r"^\s*frozenset\(\{?|\}?\)\s*$", "", s)
    s = s.strip("(){}[] ")
    return frozenset(t.strip().strip("'\"") for t in s.split(",") if t.strip().strip("'\""))


class RetailStrategistService:
    def __init__(self, customer_path, rules_path):
        self.customers = safe_csv(customer_path, dtype={"customer_id": str})
        if self.customers is not None:
            cid = find_col(self.customers, "customer_id")
            self.customers = self.customers.rename(columns={cid: "customer_id"}).drop_duplicates("customer_id")
            self.customers = self.customers.set_index("customer_id", drop=False)
            self.cltv_col = find_col(self.customers, "cltv_score", "cltv", exclude=(find_col(self.customers, "cltv_segment"),))
            self.cltv_segment_col = find_col(self.customers, "cltv_segment")
            self.segment_col = find_col(self.customers, "segment", "rfm_segment", exclude=(self.cltv_segment_col,))

        self.rules = safe_csv(rules_path)
        if self.rules is not None:
            ant, con = find_col(self.rules, "antecedents"), find_col(self.rules, "consequents")
            if ant and con:
                self.rules["_ant"] = self.rules[ant].map(_parse_itemset)
                self.rules["_con"] = self.rules[con].map(_parse_itemset)
            else:
                self.rules = None
        self.load_status = {"customer_features": self.customers is not None, "arl_rules": self.rules is not None}

    # ------------------------------------------------------------------
    def get_customer_segment(self, customer_id):
        if self.customers is None:
            raise RuntimeError("Customer feature data (customer_features.csv) is not available.")
        cid = str(customer_id).strip()
        if cid not in self.customers.index:
            return None
        row = {k: clean(v) for k, v in self.customers.loc[cid].to_dict().items()}
        return {
            "customer_id": cid,
            "segment": row.get(self.segment_col) if self.segment_col else None,
            "cltv": _num(row.get(self.cltv_col)) if self.cltv_col else None,
            "cltv_segment": row.get(self.cltv_segment_col) if self.cltv_segment_col else None,
            "data": row,
        }

    def rules_for_types(self, owned_types, top_n=5):
        """Consequent product types implied by rules whose antecedent is fully covered by owned_types."""
        if self.rules is None:
            raise RuntimeError("ARL rules (arl_rules.csv) are not available.")
        owned = frozenset(str(t).strip() for t in owned_types if t)
        if not owned:
            return []
        hits = self.rules[self.rules["_ant"].map(lambda a: bool(a) and a <= owned)]
        best = {}
        for _, r in hits.iterrows():
            for target in r["_con"] - owned:
                cand = {
                    "product_type": target,
                    "because_of": sorted(r["_ant"]),
                    "support": _num(r.get("support")),
                    "confidence": _num(r.get("confidence")),
                    "lift": _num(r.get("lift")),
                }
                cur = best.get(target)
                if cur is None or (cand["lift"] or 0, cand["confidence"] or 0) > (cur["lift"] or 0, cur["confidence"] or 0):
                    best[target] = cand
        ranked = sorted(best.values(), key=lambda x: (x["lift"] or 0, x["confidence"] or 0), reverse=True)
        return ranked[: max(1, min(int(top_n), 20))]

    def complete_the_look(self, article_id, catalog, top_n=5):
        """ARL bridge with real example articles for each recommended product type."""
        item = catalog.get(article_id)
        if item is None:
            return None
        ptype = item.get("product_type_name")
        recs = self.rules_for_types([ptype], top_n)
        for rec in recs:
            ids = catalog.sample_by_type(rec["product_type"], prefer_index=item.get("index_name"),
                                         exclude=[article_id], n=1, seed_key=item.get("article_id"))
            rec["example_article"] = catalog.short(ids[0]) if ids else None
        return {"article_id": item.get("article_id"), "product_type": ptype, "recommendations": recs}

    def get_recommendations(self, article_id, top_n=5, catalog=None):
        """Article IDs for the /retail/recommendations endpoint (one example per recommended type)."""
        if catalog is None:
            raise RuntimeError("Article catalog is required to bridge article IDs to ARL product types.")
        result = self.complete_the_look(article_id, catalog, top_n)
        if result is None:
            return []
        return [r["example_article"]["article_id"] for r in result["recommendations"] if r.get("example_article")]


def _num(x):
    try:
        return float(x) if x is not None and pd.notna(x) else None
    except Exception:
        return None

"""Shared loading helpers: a broken or missing artifact disables one feature, never the whole API."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger("aurasync")

# path -> error message for artifacts that exist but failed to load (shown in /chat/status)
LOAD_ERRORS: dict = {}


def safe_pickle(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # e.g. scikit-learn version mismatch between training and serving
        LOAD_ERRORS[str(path)] = f"{type(e).__name__}: {e}"
        log.warning("Could not load %s: %s", path, e)
        return None


def safe_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        LOAD_ERRORS[str(path)] = f"{type(e).__name__}: {e}"
        log.warning("Could not read %s: %s", path, e)
        return None


def find_col(df, *candidates, exclude=()):
    """First column whose lower-cased name equals, then contains, a candidate."""
    if df is None:
        return None
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower and lower[cand] not in exclude:
            return lower[cand]
    for cand in candidates:
        for low, real in lower.items():
            if cand in low and real not in exclude:
                return real
    return None


def norm_article_id(article_id) -> str:
    """H&M article IDs lose their leading zero when read as int — always compare as 10-digit strings."""
    s = str(article_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(10)


def clean(value):
    """JSON-friendly scalar (numpy -> python, NaN -> None)."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

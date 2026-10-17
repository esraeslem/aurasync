"""AuraSync application configuration.

All artifact paths are resolved relative to AURASYNC_DATA_DIR (default: ~/Downloads).
For each artifact the first existing location wins: an explicit env override, then
the agent's output sub-folder, then the data-dir root. If none exists, the agent
sub-folder path is used and the service reports the artifact as not loaded (503).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Load the project's .env before any service creates an OpenAI client.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

BASE_DIR = Path(os.environ.get("AURASYNC_DATA_DIR", str(Path.home() / "Downloads"))).expanduser()


def _find(filename: str, *subdirs: str, env: str | None = None) -> Path:
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    candidates = [BASE_DIR / s / filename for s in subdirs] + [BASE_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_ECON = "aurasync_economist_outputs"
_RETAIL = "aurasync_retail_strategist_outputs"
_STYLIST = "aurasync_stylist_outputs"
_VISION = ("aurasync_visionary_outputs", "eda_outputs")

# Economist
ECONOMIST_OUTPUT_DIR = BASE_DIR / _ECON
CHURN_MODEL_PATH = _find("churn_fatigue_model.pkl", _ECON, env="AURASYNC_CHURN_MODEL_PATH")
RESALE_MODEL_PATH = _find("resale_pricing_model.pkl", _ECON, env="AURASYNC_RESALE_MODEL_PATH")
# Per-customer feature store written by 02_economist_agent/export_churn_features.py
CHURN_FEATURES_PATH = _find("churn_customer_features.csv", _ECON, env="AURASYNC_CHURN_FEATURES_PATH")

# Retail Strategist
CUSTOMER_FEATURES_PATH = _find("customer_features.csv", _RETAIL, env="AURASYNC_CUSTOMER_FEATURES_PATH")
ARL_RULES_PATH = _find("arl_rules.csv", _RETAIL, env="AURASYNC_ARL_RULES_PATH")

# Stylist
FIT_TFIDF_PATH = _find("fit_tfidf_vectorizer.pkl", _STYLIST)
FIT_MODEL_PATH = _find("fit_logreg_model.pkl", _STYLIST)
SENTIMENT_TFIDF_PATH = _find("sentiment_tfidf_vectorizer.pkl", _STYLIST)
SENTIMENT_MODEL_PATH = _find("sentiment_logreg_model.pkl", _STYLIST)

# Visionary + catalog
CATALOG_EMBEDDINGS_PATH = _find("catalog_emb_marqo_siglip.npy", *_VISION)
CATALOG_ARTICLE_IDS_PATH = _find("catalog_emb_article_ids.csv", *_VISION)
ARTICLES_PATH = _find("articles.csv", env="AURASYNC_ARTICLES_PATH")
CATALOG_METADATA_PATH = _find("catalog_metadata.csv", *_VISION, env="AURASYNC_CATALOG_METADATA_PATH")
COLOR_FEATURES_PATH = _find("color_features.csv", *_VISION)
STYLE_PERSONAS_PATH = _find("style_personas.csv", *_VISION)
PERSONA_ASSIGNMENTS_PATH = _find("style_persona_assignments.csv", *_VISION)

# Wardrobe page served at /app (the project's frontend HTML)
def _find_frontend():
    if os.environ.get("AURASYNC_FRONTEND_PATH"):
        return Path(os.environ["AURASYNC_FRONTEND_PATH"]).expanduser()
    pools = [list((PROJECT_ROOT / "frontend").glob("*.html")),
             list(BASE_DIR.glob("*/frontend/*wardrobe*.html")),
             list(BASE_DIR.glob("*/frontend/*.html"))]
    for pool in pools:
        if pool:
            return max(pool, key=lambda q: (("wardrobe" in q.name), q.stat().st_mtime))
    return None


FRONTEND_PATH = _find_frontend()

# LLM concierge
# AURASYNC_LLM_PROVIDER: "openai" (paid API, Responses API) or "ollama" (free local model, Chat Completions).
# Any other OpenAI-compatible server works with AURASYNC_LLM_PROVIDER=custom + AURASYNC_LLM_BASE_URL.
_PRESETS = {
    "openai": {"base_url": None, "model": "gpt-4o-mini", "api": "responses"},
    # Ollama's own /api/chat: the only documented way to switch thinking off (think=false) and set num_ctx per request
    "ollama": {"base_url": "http://localhost:11434", "model": "qwen3:4b-instruct", "api": "ollama", "reasoning_effort": "none",
               "history": 12, "router": True},
    "custom": {"base_url": None, "model": "", "api": "chat"},
}
LLM_PROVIDER = os.environ.get("AURASYNC_LLM_PROVIDER", "openai").strip().lower() or "openai"
_preset = _PRESETS.get(LLM_PROVIDER, _PRESETS["custom"])
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("AURASYNC_LLM_BASE_URL", "").strip() or _preset["base_url"]
LLM_API = (os.environ.get("AURASYNC_LLM_API", "").strip().lower() or _preset["api"])  # "responses" | "chat" | "ollama"
if LLM_API == "ollama" and LLM_BASE_URL:
    LLM_BASE_URL = re.sub(r"/v1/?$", "", LLM_BASE_URL.rstrip("/"))  # native API lives at the server root
elif LLM_API == "chat" and LLM_PROVIDER == "ollama" and LLM_BASE_URL and not LLM_BASE_URL.rstrip("/").endswith("/v1"):
    LLM_BASE_URL = LLM_BASE_URL.rstrip("/") + "/v1"  # Ollama's OpenAI-compatible endpoint
LLM_MODEL = (os.environ.get("AURASYNC_LLM_MODEL", "").strip()
             or (os.environ.get("AURASYNC_OPENAI_MODEL", "").strip() if LLM_PROVIDER == "openai" else "")
             or _preset["model"])
# Local servers ignore the key but the client library requires a non-empty value.
LLM_API_KEY = os.environ.get("AURASYNC_LLM_API_KEY", "").strip() or (OPENAI_API_KEY if LLM_PROVIDER == "openai" else "local")
OPENAI_MODEL = LLM_MODEL  # backwards-compatible name
# Chat mode only. "none" switches off the slow "thinking" step of local reasoning models (qwen3, ...).
# Set AURASYNC_LLM_REASONING_EFFORT=default to send nothing and keep the model's own default.
_effort = os.environ.get("AURASYNC_LLM_REASONING_EFFORT", "").strip().lower() or _preset.get("reasoning_effort", "")
LLM_REASONING_EFFORT = None if _effort in ("", "default") else _effort
CHAT_HISTORY_MESSAGES = int(os.environ.get("AURASYNC_CHAT_HISTORY_MESSAGES", "") or _preset.get("history", 30))  # local modes: messages resent per call


def _flag(name, default):
    v = os.environ.get(name, "").strip().lower()
    return default if v == "" else v in ("1", "true", "yes", "on")


# Router: for common questions about the selected piece, run the matching tools directly and make ONE model call
# (instead of letting the model pick tools over 2-3 calls). Default on for local models, where each call is slow.
CHAT_ROUTER = _flag("AURASYNC_CHAT_ROUTER", _preset.get("router", False))
# Ollama runtime settings
OLLAMA_NUM_CTX = int(re.sub(r"\D", "", os.environ.get("AURASYNC_OLLAMA_CONTEXT", "")) or 8192)
OLLAMA_KEEP_ALIVE = os.environ.get("AURASYNC_OLLAMA_KEEP_ALIVE", "60m").strip() or "60m"
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("AURASYNC_LLM_MAX_OUTPUT_TOKENS", "500"))
CHAT_MAX_TOOL_ROUNDS = int(os.environ.get("AURASYNC_CHAT_MAX_TOOL_ROUNDS", "6"))
# Only sent when set: reasoning models (o-series, gpt-5*) reject a temperature parameter.
_temp = os.environ.get("AURASYNC_CHAT_TEMPERATURE", "").strip()
CHAT_TEMPERATURE = float(_temp) if _temp else None

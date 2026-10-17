"""Single shared instance of every agent service.

Routers and the LLM concierge import from here so each artifact (e.g. the 105k x 768
embedding matrix) is loaded exactly once per API process.
"""
from __future__ import annotations

from .. import config
from .catalog_service import CatalogService
from .economist_service import EconomistService
from .retail_service import RetailStrategistService
from .stylist_service import StylistService
from .visionary_service import VisionaryService

catalog = CatalogService(config.ARTICLES_PATH, config.CATALOG_METADATA_PATH, config.STYLE_PERSONAS_PATH)
retail = RetailStrategistService(config.CUSTOMER_FEATURES_PATH, config.ARL_RULES_PATH)
visionary = VisionaryService(config.CATALOG_EMBEDDINGS_PATH, config.CATALOG_ARTICLE_IDS_PATH,
                             config.PERSONA_ASSIGNMENTS_PATH, config.COLOR_FEATURES_PATH)
stylist = StylistService(config.FIT_TFIDF_PATH, config.FIT_MODEL_PATH,
                         config.SENTIMENT_TFIDF_PATH, config.SENTIMENT_MODEL_PATH)
economist = EconomistService(config.CHURN_MODEL_PATH, config.RESALE_MODEL_PATH, config.CHURN_FEATURES_PATH)

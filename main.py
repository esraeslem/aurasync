"""AuraSync FastAPI entrypoint."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routers import chat, economist, frontend, retail_strategist, stylist, visionary
from app.schemas import HealthResponse
from app.services import registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="AuraSync API", version="1.2.0",
              description="Smart wardrobe API and LLM-powered multi-agent concierge.")

# The wardrobe HTML is opened from disk (origin "null") or another port, so the browser needs CORS.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

for router_module in (economist, retail_strategist, stylist, visionary, chat, frontend):
    app.include_router(router_module.router)


@app.get("/health", response_model=HealthResponse)
def health():
    agents = {
        "catalog": registry.catalog.load_status,
        "retail_strategist": registry.retail.load_status,
        "economist": registry.economist.load_status,
        "stylist": registry.stylist.load_status,
        "visionary": registry.visionary.load_status,
        "llm_concierge": {"api_key_configured": bool(config.LLM_API_KEY), "model_configured": bool(config.LLM_MODEL),
                          "local_model": config.LLM_PROVIDER != "openai"},
    }
    any_loaded = any(ok for k, v in agents.items() if k != "llm_concierge"
                     for name, ok in v.items() if name != "embeddings_aligned")
    return {"status": "ok" if any_loaded else "no artifacts loaded", "agents": agents}


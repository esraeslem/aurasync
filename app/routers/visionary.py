from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas import ColorExtractionResponse, PersonaResponse, VisualSearchResponse
from ..services.registry import catalog, visionary as service
from ..services.visionary_service import VisionaryService

router = APIRouter(prefix="/visionary", tags=["Visionary"])


@router.get("/visual-search/{article_id}", response_model=VisualSearchResponse)
def visual_search(article_id: str, top_n: int = 5):
    try:
        r = service.find_similar_articles(article_id, top_n)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, f"Article {article_id} not found.")
    return {"query_article_id": article_id, **r}


@router.get("/style-persona/{article_id}")
def style_persona(article_id: str):
    """Item-level style persona from style_personas.csv."""
    try:
        r = catalog.get_persona(article_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, f"Article {article_id} not found.")
    return r


@router.get("/persona/{customer_id}", response_model=PersonaResponse)
def persona(customer_id: str):
    try:
        r = service.get_persona(customer_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, f"Customer {customer_id} not found.")
    return r


@router.post("/color/extract", response_model=ColorExtractionResponse)
async def color(file: UploadFile = File(...)):
    try:
        return {"dominant_color_rgb": VisionaryService.extract_dominant_color(await file.read())}
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")

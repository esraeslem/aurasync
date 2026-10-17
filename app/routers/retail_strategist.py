from fastapi import APIRouter, HTTPException

from ..schemas import CustomerSegmentResponse, RecommendationResponse
from ..services.registry import catalog, retail as service

router = APIRouter(prefix="/retail", tags=["Retail Strategist"])


@router.get("/customer/{customer_id}/segment", response_model=CustomerSegmentResponse)
def segment(customer_id: str):
    try:
        r = service.get_customer_segment(customer_id)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, f"Customer {customer_id} not found.")
    return r


@router.get("/recommendations/{article_id}", response_model=RecommendationResponse)
def recommendations(article_id: str, top_n: int = 5):
    try:
        r = service.get_recommendations(article_id, top_n, catalog=catalog)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"article_id": article_id, "recommended_article_ids": r}


@router.get("/complete-the-look/{article_id}")
def complete_the_look(article_id: str, top_n: int = 5):
    """Rules behind the recommendation: product type, support/confidence/lift and an example article per type."""
    try:
        r = service.complete_the_look(article_id, catalog, top_n)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, f"Article {article_id} not found.")
    return r

from fastapi import APIRouter, HTTPException

from ..schemas import ChurnFeaturesRequest, ChurnPredictionResponse, ResaleArticleRequest, ResalePriceResponse
from ..services.registry import catalog, economist as service

router = APIRouter(prefix="/economist", tags=["Economist"])


@router.post("/churn/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: ChurnFeaturesRequest):
    """Features are read from the churn feature store; any features in the body override stored values."""
    try:
        return service.predict_churn(payload.model_dump())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except LookupError as e:
        raise HTTPException(422, str(e))


@router.post("/resale/estimate", response_model=ResalePriceResponse)
def estimate_resale(payload: ResaleArticleRequest):
    try:
        row = catalog.get(payload.article_id) if service.resale_artifacts else None
        r = service.predict_resale_price({**payload.features, "features": payload.features}, article_row=row)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except LookupError as e:
        raise HTTPException(422, str(e))
    price = r["estimated_price_usd"] if isinstance(r, dict) else r
    return {"article_id": payload.article_id, "estimated_price": price}

from fastapi import APIRouter, HTTPException

from ..schemas import FitPredictionResponse, ReviewTextRequest, SentimentPredictionResponse
from ..services.registry import stylist as service

router = APIRouter(prefix="/stylist", tags=["Stylist"])


@router.post("/fit/predict", response_model=FitPredictionResponse)
def fit(payload: ReviewTextRequest):
    try:
        return service.predict_fit(payload.review_text)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.post("/sentiment/predict", response_model=SentimentPredictionResponse)
def sentiment(payload: ReviewTextRequest):
    try:
        return service.predict_sentiment(payload.review_text)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

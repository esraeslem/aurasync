"""Pydantic request/response models for AuraSync API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReviewTextRequest(BaseModel):
    review_text: str = Field(min_length=1)


class FitPredictionResponse(BaseModel):
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


class SentimentPredictionResponse(BaseModel):
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


class ChurnFeaturesRequest(BaseModel):
    customer_id: str
    features: Dict[str, Any] = {}


class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: Optional[int] = None
    features_as_of: Optional[str] = None
    top_feature_values: Dict[str, Any] = {}


class ResaleArticleRequest(BaseModel):
    article_id: str
    features: Dict[str, Any] = {}


class ResalePriceResponse(BaseModel):
    estimated_price: float
    article_id: Optional[str] = None


class CustomerSegmentResponse(BaseModel):
    customer_id: str
    segment: Optional[str] = None
    cltv: Optional[float] = None
    cltv_segment: Optional[str] = None
    data: Dict[str, Any] = {}


class RecommendationResponse(BaseModel):
    article_id: str
    recommended_article_ids: List[str]


class VisualSearchResponse(BaseModel):
    query_article_id: str
    recommendations: List[Dict[str, Any]]


class PersonaResponse(BaseModel):
    customer_id: str
    persona_cluster: Optional[Any] = None
    persona_label: Optional[str] = None
    data: Dict[str, Any] = {}


class ColorExtractionResponse(BaseModel):
    dominant_color_rgb: List[int]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    agent: str = Field(default="AI Concierge", min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    # Sent by the wardrobe page (via /app): the pieces in the user's wardrobe and the selected one
    wardrobe: List[str] = Field(default_factory=list, max_length=200)
    selected_article_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tools_used: List[str]
    agent: str = "AI Concierge"


class HealthResponse(BaseModel):
    status: str
    agents: Dict[str, Dict[str, bool]]

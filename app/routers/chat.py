from fastapi import APIRouter, HTTPException

from ..schemas import ChatRequest, ChatResponse
from ..services import concierge_service
from ..services.concierge_service import ConciergeError

router = APIRouter(prefix="/chat", tags=["AI Multi-Agent Concierge"])


@router.post("/message", response_model=ChatResponse)
def message(req: ChatRequest):
    try:
        sid, reply, used = concierge_service.chat(req.session_id, req.message, req.customer_id, req.agent,
                                                  wardrobe=req.wardrobe, selected_article_id=req.selected_article_id)
    except ConciergeError as e:
        raise HTTPException(e.status_code, e.message)
    agent = req.agent if req.agent in concierge_service.AGENTS else "AI Concierge"
    return {"session_id": sid, "reply": reply, "tools_used": used, "agent": agent}


@router.get("/warmup")
def warmup():
    """Loads the local model into memory (used by the launcher before opening the page)."""
    try:
        return concierge_service.warmup()
    except ConciergeError as e:
        raise HTTPException(e.status_code, e.message)


@router.get("/status")
def status():
    """Model, key presence, tools per agent and which artifacts loaded — first stop when debugging."""
    return concierge_service.status()


@router.delete("/session/{session_id}")
def reset(session_id: str):
    return {"session_id": session_id, "reset": concierge_service.reset_session(session_id)}

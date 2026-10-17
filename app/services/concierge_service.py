"""AuraSync multi-agent concierge: an LLM with function calling into the four agents' trained
models and data.

Flow per user message:
  0. Router (on by default for local models): if the question clearly asks for pairings, resale
     price, similar items, colour, persona, fit/feedback, churn or the customer profile, the
     matching tools run directly and the LLM writes the answer in ONE call.
  1. Otherwise the selected specialist's instructions + allowed tools are sent with the message.
  2. The LLM decides which AuraSync tools to call; each call runs the real model/data lookup.
  3. Tool results go back to the LLM until it answers in text (max CHAT_MAX_TOOL_ROUNDS rounds).

LLM back-ends: OpenAI Responses API (memory via previous_response_id), Ollama's own /api/chat
(think=false, num_ctx and keep_alive per request; memory kept here), or any OpenAI-compatible
Chat Completions server (memory kept here).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

from .. import config
from ._io import LOAD_ERRORS, norm_article_id
from .economist_service import MissingFeaturesError
from .registry import catalog, economist, retail, stylist, visionary

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover
    openai = None
    OpenAI = None

log = logging.getLogger("aurasync.concierge")


class ConciergeError(Exception):
    def __init__(self, status_code: int, message: str, original: Exception | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.original = original


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI Responses API function-tool format)
# ---------------------------------------------------------------------------
_ARTICLE = {"type": "string", "description": "H&M article_id (10 digits; a leading zero may be missing)."}


def _fn(name, description, properties, required):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


TOOLS = [
    _fn("get_article_details",
        "Look up catalog metadata for an article: name, product type, colour, pattern, department, description. "
        "Call this before describing any article you have only an ID for.",
        {"article_id": _ARTICLE}, ["article_id"]),
    _fn("get_recommendations",
        "Retail Strategist 'Complete the Look': association rules (FP-Growth on H&M baskets, product-type level) for the "
        "article's product type, with support/confidence/lift and one real example article per recommended type.",
        {"article_id": _ARTICLE, "top_n": {"type": "integer", "minimum": 1, "maximum": 10}}, ["article_id"]),
    _fn("get_customer_profile",
        "Retail Strategist: a customer's RFM segment, CLTV score and CLTV segment from customer_features.csv.",
        {"customer_id": {"type": "string"}}, ["customer_id"]),
    _fn("get_color_info",
        "Visionary: extracted colour features for an article (dominant colours from the product image).",
        {"article_id": _ARTICLE}, ["article_id"]),
    _fn("visual_search",
        "Visionary: visually similar catalog articles via Marqo-FashionSigLIP image embeddings (cosine similarity).",
        {"article_id": _ARTICLE, "top_n": {"type": "integer", "minimum": 1, "maximum": 10}}, ["article_id"]),
    _fn("get_style_persona",
        "Visionary: the style persona cluster (hierarchical clustering of embeddings) an article belongs to.",
        {"article_id": _ARTICLE}, ["article_id"]),
    _fn("predict_fit_sentiment",
        "Stylist: classify a clothing review or feedback text for fit (small/fit/large) and sentiment, using the "
        "TF-IDF + Logistic Regression models trained on ModCloth reviews.",
        {"review_text": {"type": "string"}}, ["review_text"]),
    _fn("predict_churn_risk",
        "Economist: style-fatigue churn probability for a customer (LightGBM), i.e. probability of no purchase in the "
        "prediction window, plus the customer's values for the most important features.",
        {"customer_id": {"type": "string"}}, ["customer_id"]),
    _fn("estimate_resale_price",
        "Economist: estimated second-hand resale price (USD) for a catalog article using the LightGBM model trained "
        "on Mercari listings. Optional item condition 1=new .. 5=poor (default 2) and whether the seller pays shipping.",
        {"article_id": _ARTICLE,
         "item_condition_id": {"type": "integer", "minimum": 1, "maximum": 5},
         "seller_pays_shipping": {"type": "boolean"}},
        ["article_id"]),
]
TOOL_BY_NAME = {t["name"]: t for t in TOOLS}

AGENTS = {
    "AI Concierge": {
        "tools": list(TOOL_BY_NAME),
        "instructions": "You coordinate all AuraSync specialists. Pick whichever tools answer the question.",
    },
    "Economist": {
        "tools": ["predict_churn_risk", "estimate_resale_price", "get_customer_profile", "get_article_details"],
        "instructions": "Focus on customer economics: style-fatigue churn risk, resale value, CLTV/RFM and their commercial implications.",
    },
    "Retail Strategist": {
        "tools": ["get_recommendations", "get_customer_profile", "get_article_details", "get_color_info"],
        "instructions": "Focus on assortment, cross-sell ('Complete the Look'), RFM/CLTV segments and retail recommendations.",
    },
    "Visionary": {
        "tools": ["visual_search", "get_color_info", "get_style_persona", "get_article_details"],
        "instructions": "Focus on visual similarity, colour interpretation and style personas.",
    },
    "Stylist": {
        "tools": ["predict_fit_sentiment", "get_recommendations", "get_color_info", "get_style_persona", "get_article_details"],
        "instructions": "Focus on fit, feedback sentiment and how to style pieces into outfits. When the user gives feedback on an "
                        "item or outfit, run predict_fit_sentiment on their words and adapt your suggestion to the result.",
    },
    "CRM Concierge": {
        "tools": ["get_customer_profile", "predict_churn_risk", "get_recommendations", "get_article_details"],
        "instructions": "Focus on customer engagement and retention actions grounded in the customer's profile and churn risk.",
    },
}

SYSTEM_BASE = """You are an AuraSync specialist inside a smart-wardrobe and second-hand fashion app built on the H&M catalog.

Grounding rules:
- Use AuraSync tools for any article, customer, prediction or price. Never invent article IDs, product names, customer data, model outputs or prices.
- Trust get_article_details for what an article is (H&M names can be unusual: "10p Jersey Terry" is a pack of hair ties). Only mention article IDs that a tool returned.
- If a tool returns an error or says an artifact is unavailable, say so plainly and suggest what is needed; do not guess a value.
- Distinguish model outputs (with their numbers) from your own styling or business advice.
- Messages may start with an "App context" block (the user's selected piece and wardrobe, read from the app) or "[customer_id=...]". Those IDs are always valid: never ask the user to re-check them; use them with the tools when relevant.

Style:
- Reply in the language the user writes in.
- Be concise and concrete. Plain text only: short paragraphs or simple "-" lists, no markdown headings, bold or tables (the chat window shows raw text)."""


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def _top_n(args, default=5):
    try:
        return max(1, min(int(args.get("top_n", default)), 10))
    except (TypeError, ValueError):
        return default


def run_tool(name: str, args: dict) -> dict:
    try:
        if name == "get_article_details":
            return catalog.describe(args["article_id"]) or {"error": f"article {norm_article_id(args['article_id'])} not found in the catalog"}

        if name == "get_recommendations":
            r = retail.complete_the_look(args["article_id"], catalog, _top_n(args))
            if r is None:
                return {"error": "article not found in the catalog"}
            if not r["recommendations"]:
                covered = sorted({t for a in retail.rules["_ant"] for t in a}) if retail.rules is not None else []
                r["note"] = (f"No association rules exist for product type '{r['product_type']}'. "
                             f"Rules cover these product types: {', '.join(covered)}. "
                             "Do not invent pairings; offer visual_search for similar items instead.")
            return r

        if name == "get_customer_profile":
            r = retail.get_customer_segment(args["customer_id"])
            if r is None:
                return {"error": "customer not found in customer_features.csv"}
            return r

        if name == "get_color_info":
            return visionary.get_color(args["article_id"]) or {"error": "article not found in color features"}

        if name == "visual_search":
            r = visionary.find_similar_articles(args["article_id"], _top_n(args))
            if r is None:
                return {"error": "article not found in the embedding index"}
            for rec in r["recommendations"]:
                try:
                    rec.update({k: v for k, v in catalog.short(rec["article_id"]).items() if k != "article_id"})
                except RuntimeError:
                    break
            return {"query_article_id": norm_article_id(args["article_id"]), **r}

        if name == "get_style_persona":
            return catalog.get_persona(args["article_id"]) or {"error": "article not found in style persona assignments"}

        if name == "predict_fit_sentiment":
            text = str(args["review_text"])
            return {"fit": stylist.predict_fit(text), "sentiment": stylist.predict_sentiment(text)}

        if name == "predict_churn_risk":
            return economist.predict_churn({"customer_id": args["customer_id"]})

        if name == "estimate_resale_price":
            row = catalog.get(args["article_id"])
            if row is None:
                return {"error": "article not found in the catalog"}
            payload = {"item_condition_id": args.get("item_condition_id"),
                       "shipping": 1 if args.get("seller_pays_shipping") else 0}
            result = economist.predict_resale_price(payload, article_row=row)
            return {"article_id": row["article_id"], "name": row.get("prod_name"), **result}

        return {"error": f"unknown tool {name}"}
    except MissingFeaturesError as e:
        return {"error": str(e), "missing_features": e.missing}
    except KeyError as e:
        return {"error": f"missing argument {e}"}
    except (RuntimeError, LookupError) as e:  # artifact not loaded / record not found: expected, report as-is
        log.warning("Tool %s unavailable: %s", name, e)
        return {"error": str(e)}
    except Exception as e:  # tool failures are reported to the LLM, never fabricated around
        log.exception("Tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# LLM client + sessions
# ---------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()
LOCAL = config.LLM_PROVIDER != "openai"


def _get_client():
    global _client
    if _client is not None:
        return _client
    if config.LLM_API == "ollama":
        with _client_lock:
            if _client is None:
                _client = OllamaNative(config.LLM_BASE_URL)
        return _client
    if OpenAI is None:
        raise ConciergeError(503, "The 'openai' package is not installed. Run: pip install -r requirements.txt")
    if not config.LLM_API_KEY:
        raise ConciergeError(503, "OPENAI_API_KEY is not configured. Add it to the .env file next to main.py and restart the API "
                                  "(or set AURASYNC_LLM_PROVIDER=ollama to use a free local model).")
    with _client_lock:
        if _client is None:
            # Local models can be slow on the first call while the model loads into memory.
            _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                             timeout=180.0 if LOCAL else 60.0, max_retries=1 if LOCAL else 2)
    return _client


def set_client(client) -> None:
    """Inject a client (used by tests)."""
    global _client
    _client = client


class OllamaNative:
    """Minimal client for Ollama's own /api/chat (non-streaming)."""

    def __init__(self, base_url: str):
        import httpx  # installed with the openai package
        self._httpx = httpx
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(300.0, connect=5.0))

    def _post(self, path: str, payload: dict) -> dict:
        httpx = self._httpx
        try:
            r = self.http.post(path, json=payload)
        except httpx.TimeoutException as e:
            raise ConciergeError(504, "The local model took too long to answer. Please retry.", e)
        except httpx.HTTPError as e:
            raise ConciergeError(502, f"Could not reach the local model server at {self.base_url}. "
                                      "Is Ollama running? Open the Ollama app.", e)
        if r.status_code >= 400:
            try:
                detail = r.json().get("error") or r.text
            except ValueError:
                detail = r.text
            if r.status_code == 404 or "not found" in str(detail).lower():
                raise ConciergeError(502, f"Model '{payload.get('model')}' is not installed locally. "
                                          f"Run: ollama pull {payload.get('model')}")
            raise ConciergeError(502, f"Ollama error {r.status_code}: {str(detail)[:300]}")
        return r.json()

    no_think_param = False

    def chat(self, payload: dict) -> dict:
        if self.no_think_param:
            payload.pop("think", None)
        try:
            return self._post("/api/chat", payload)
        except ConciergeError as e:
            if "think" in payload and "think" in e.message.lower():
                log.warning("Model does not accept the think option; retrying without it")
                self.no_think_param = True
                payload.pop("think", None)
                return self._post("/api/chat", payload)
            raise

    def load(self, model: str) -> dict:
        """Load the model into memory with the same context size the chat requests use (no reload later)."""
        return self._post("/api/generate", {"model": model, "keep_alive": config.OLLAMA_KEEP_ALIVE,
                                            "options": {"num_ctx": config.OLLAMA_NUM_CTX}})


def _ollama_payload(messages: list, tools: list | None = None) -> dict:
    options = {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.LLM_MAX_OUTPUT_TOKENS}
    if config.CHAT_TEMPERATURE is not None:
        options["temperature"] = config.CHAT_TEMPERATURE
    payload = {"model": config.LLM_MODEL, "messages": messages, "stream": False,
               "keep_alive": config.OLLAMA_KEEP_ALIVE, "options": options}
    if config.LLM_REASONING_EFFORT == "none":
        payload["think"] = False
    if tools:
        payload["tools"] = tools
    return payload


def warmup() -> dict:
    """Load the local model before the first question so the user does not wait for it."""
    if config.LLM_API != "ollama":
        return {"loaded": False, "reason": "not a local Ollama model"}
    t0 = time.time()
    client = _get_client()
    client.load(config.LLM_MODEL)
    return {"loaded": True, "model": config.LLM_MODEL, "num_ctx": config.OLLAMA_NUM_CTX,
            "seconds": round(time.time() - t0, 1)}


MAX_SESSIONS = 500
_sessions: "OrderedDict[str, dict]" = OrderedDict()
_sessions_lock = threading.Lock()


def _session(sid: str) -> dict:
    with _sessions_lock:
        state = _sessions.pop(sid, None) or {"previous_response_id": None, "messages": [], "turns": 0, "created": time.time()}
        _sessions[sid] = state  # move to end (most recent)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return state


def reset_session(sid: str) -> bool:
    with _sessions_lock:
        return _sessions.pop(sid, None) is not None


def _translate(e: Exception) -> ConciergeError:
    err = _translate_message(e)
    err.original = e
    return err


def _translate_message(e: Exception) -> ConciergeError:
    model = config.LLM_MODEL
    if openai is not None:
        if LOCAL and isinstance(e, openai.APIConnectionError) and not isinstance(e, openai.APITimeoutError):
            return ConciergeError(502, f"Could not reach the local model server at {config.LLM_BASE_URL}. "
                                       "Is Ollama running? Open the Ollama app or run: ollama serve")
        if LOCAL and isinstance(e, openai.NotFoundError):
            return ConciergeError(502, f"Model '{model}' is not installed locally. Run: ollama pull {model}")
        if isinstance(e, openai.AuthenticationError):
            return ConciergeError(502, "OpenAI rejected the API key (401). Check OPENAI_API_KEY in .env.")
        if isinstance(e, openai.PermissionDeniedError):
            return ConciergeError(502, f"This API key has no access to model '{model}' (403).")
        if isinstance(e, openai.NotFoundError):
            return ConciergeError(502, f"OpenAI model '{model}' was not found. Set AURASYNC_OPENAI_MODEL in .env (e.g. gpt-4o-mini).")
        if isinstance(e, openai.RateLimitError):
            return ConciergeError(429, "OpenAI rate limit or quota exceeded. Check billing/usage on platform.openai.com, then retry.")
        if isinstance(e, openai.APITimeoutError):
            return ConciergeError(504, "The model took too long to answer. Please retry"
                                       + (" (a local model is slow on its first call while it loads)." if LOCAL else "."))
        if isinstance(e, openai.APIConnectionError):
            return ConciergeError(502, "Could not reach the OpenAI API (network/proxy problem).")
        if isinstance(e, openai.APIStatusError):
            return ConciergeError(502, f"LLM API error {e.status_code}: {getattr(e, 'message', e)}")
    return ConciergeError(500, f"{type(e).__name__}: {e}")


def _call(fn, **kwargs):
    try:
        return fn(**kwargs)
    except Exception as e:
        raise _translate(e) from e


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                parts.append(getattr(c, "text", ""))
    return "".join(parts)


_ID = re.compile(r"(?<!\d)(?<!\d[.,])\d{9,10}(?!\d)(?![.,]\d)")  # 9-10 digit IDs, not parts of larger numbers


def _collect_ids(text, into: set) -> None:
    for m in _ID.findall(str(text or "")):
        into.add(norm_article_id(m))


def _guard_ids(text: str, known: set) -> str:
    """Drop any line that cites an article ID no tool or app context provided (small models invent them)."""
    kept, dropped = [], 0
    for line in text.splitlines():
        if any(norm_article_id(m) not in known for m in _ID.findall(line)):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        log.warning("Removed %d line(s) citing unverified article IDs", dropped)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return out or "I couldn't verify the article IDs for that answer. Please ask again, mentioning the piece you mean."


def _app_context(wardrobe, selected_article_id) -> str:
    """Readable, catalog-grounded description of what the user has in the app."""
    def label(aid):
        try:
            d = catalog.short(aid)
        except RuntimeError:
            return norm_article_id(aid)
        bits = ", ".join(x for x in (d.get("product_type"), d.get("colour")) if x)
        return f"{d['article_id']} — {d.get('name') or 'unknown'}" + (f" ({bits})" if bits else "")
    lines = []
    if selected_article_id:
        lines.append(f"Selected piece: {label(selected_article_id)}")
    items = [w for w in (wardrobe or []) if str(w).strip()][:20]
    if items:
        lines.append(f"Wardrobe ({len(wardrobe)} pieces):")
        lines += [f"- {label(w)}" for w in items]
    return ("App context (read from the AuraSync app; IDs are valid):\n" + "\n".join(lines) + "\n\n") if lines else ""


def _run_call(name, raw_args, profile, agent, used, seen=None):
    if name not in used:
        used.append(name)
    if name not in profile["tools"]:
        result = {"error": f"tool {name} is not available to {agent}"}
    else:
        result = run_tool(name, _parse_args(raw_args))
    out = json.dumps(result, default=str, ensure_ascii=False)
    if seen is not None:
        _collect_ids(out, seen)
    return out


# ---------------------------------------------------------------------------
# Mode 1: OpenAI Responses API (server-side memory via previous_response_id)
# ---------------------------------------------------------------------------
def _loop_responses(client, state, sid, agent, profile, instructions, user_message, used, seen):
    base = {"model": config.LLM_MODEL, "instructions": instructions, "tools": [TOOL_BY_NAME[n] for n in profile["tools"]]}
    if config.CHAT_TEMPERATURE is not None:
        base["temperature"] = config.CHAT_TEMPERATURE
    input_items: list = [{"role": "user", "content": user_message}]
    previous = state.get("previous_response_id")
    rounds = max(1, config.CHAT_MAX_TOOL_ROUNDS)

    for i in range(rounds + 1):
        kwargs = dict(base, input=input_items)
        if i == rounds:  # out of tool budget: force a text answer from what was gathered
            kwargs["tool_choice"] = "none"
        if previous:
            kwargs["previous_response_id"] = previous
        try:
            response = _call(client.responses.create, **kwargs)
        except ConciergeError as err:
            # A stale/expired previous_response_id on the first call: drop history once and retry.
            stale = openai is not None and isinstance(err.original, (openai.BadRequestError, openai.NotFoundError))
            if i == 0 and previous and stale:
                log.warning("Retrying session %s without previous_response_id", sid)
                previous = None
                kwargs.pop("previous_response_id", None)
                response = _call(client.responses.create, **kwargs)
            else:
                raise
        previous = response.id

        calls = [x for x in (response.output or []) if getattr(x, "type", None) == "function_call"]
        if not calls:
            state["previous_response_id"] = previous
            return _output_text(response)
        input_items = [{"type": "function_call_output", "call_id": c.call_id,
                        "output": _run_call(c.name, c.arguments, profile, agent, used, seen)} for c in calls]
    raise ConciergeError(500, "The assistant did not produce a final answer.")


# ---------------------------------------------------------------------------
# Mode 2: Chat Completions (Ollama / any OpenAI-compatible server; history kept here)
# ---------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.S)


def _clean_reply(text: str) -> str:
    """Remove reasoning a local model leaked into the answer, and markdown the chat window would show raw."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:  # template opened <think> itself, so only the closing tag is in the content
        text = text.rsplit("</think>", 1)[1]
    text = _MD_BOLD.sub(r"\2", text)
    text = re.sub(r"(?m)^(\s*)[*•]\s+", r"\1- ", text)                       # "* item" -> "- item"
    text = re.sub(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])", r"\1", text)  # *italic* -> italic
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = text.replace("`", "")
    return text.strip()


def _no_think_suffix() -> str:
    """Qwen3 soft switch: '/no_think' in the user turn makes the model skip its thinking step."""
    m = (config.LLM_MODEL or "").lower()
    return "\n\n/no_think" if config.LLM_REASONING_EFFORT == "none" and m.startswith("qwen3") else ""
_effort_supported = True  # flipped off if the server rejects reasoning_effort for this model


def _create_chat(client, kwargs):
    global _effort_supported
    if config.LLM_REASONING_EFFORT and _effort_supported:
        try:
            return _call(client.chat.completions.create, **kwargs, reasoning_effort=config.LLM_REASONING_EFFORT)
        except ConciergeError as err:
            rejected = openai is not None and isinstance(err.original, (openai.BadRequestError, openai.UnprocessableEntityError))
            if not rejected:
                raise
            log.warning("Model rejected reasoning_effort=%s; continuing without it (%s)", config.LLM_REASONING_EFFORT, err.message)
            _effort_supported = False
    return _call(client.chat.completions.create, **kwargs)


def _trim(messages: list, limit: int) -> list:
    """Keep the most recent messages, starting at a user turn so tool calls never lose their results."""
    if len(messages) <= limit:
        return messages
    for i in range(len(messages) - limit, len(messages)):
        if messages[i].get("role") == "user":
            return messages[i:]
    return messages[-1:]


def _loop_chat(client, state, sid, agent, profile, instructions, user_message, used, seen):
    history = state.setdefault("messages", [])
    turn: list = [{"role": "user", "content": user_message + _no_think_suffix()}]
    tools = [{"type": "function", "function": {k: TOOL_BY_NAME[n][k] for k in ("name", "description", "parameters")}}
             for n in profile["tools"]]
    rounds = max(1, config.CHAT_MAX_TOOL_ROUNDS)

    for i in range(rounds + 1):
        kwargs = {"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": instructions}] + _trim(history, config.CHAT_HISTORY_MESSAGES) + turn}
        if i < rounds:  # last round: no tools offered, so the model has to answer in text
            kwargs["tools"] = tools
        if config.CHAT_TEMPERATURE is not None:
            kwargs["temperature"] = config.CHAT_TEMPERATURE
        response = _create_chat(client, kwargs)
        msg = response.choices[0].message
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls or i == rounds:
            text = _clean_reply(msg.content or "")
            history.extend(turn + [{"role": "assistant", "content": text}])  # commit the turn only once it succeeded
            return text
        turn.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                                    for c in calls]})
        for c in calls:
            turn.append({"role": "tool", "tool_call_id": c.id,
                         "content": _run_call(c.function.name, c.function.arguments, profile, agent, used, seen)})
    raise ConciergeError(500, "The assistant did not produce a final answer.")


# ---------------------------------------------------------------------------
# Mode 3: Ollama native /api/chat (think=false; memory kept here)
# ---------------------------------------------------------------------------
def _loop_ollama(client, state, sid, agent, profile, instructions, user_message, used, seen):
    history = state.setdefault("messages", [])
    turn: list = [{"role": "user", "content": user_message + _no_think_suffix()}]
    tools = [{"type": "function", "function": {k: TOOL_BY_NAME[n][k] for k in ("name", "description", "parameters")}}
             for n in profile["tools"]]
    rounds = max(1, config.CHAT_MAX_TOOL_ROUNDS)

    for i in range(rounds + 1):
        messages = [{"role": "system", "content": instructions}] + _trim(history, config.CHAT_HISTORY_MESSAGES) + turn
        data = client.chat(_ollama_payload(messages, tools if i < rounds else None))
        msg = data.get("message") or {}
        calls = msg.get("tool_calls") or []
        if not calls or i == rounds:
            text = _clean_reply(msg.get("content") or "")
            history.extend(turn + [{"role": "assistant", "content": text}])
            return text
        turn.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name", "")
            turn.append({"role": "tool", "tool_name": name,
                         "content": _run_call(name, fn.get("arguments") or {}, profile, agent, used, seen)})
    raise ConciergeError(500, "The assistant did not produce a final answer.")


# ---------------------------------------------------------------------------
# Router: run the obvious tools directly, then ONE model call writes the answer
# ---------------------------------------------------------------------------
_INTENTS = {
    "pair": ["goes with", "go with", "pair", "match", "outfit", "complete the look", "wear it with", "wear with",
             "combine", "kombin", "ne giy", "yakış", "yakis", "uyar", "eşleş", "esles", "tamamla", "ile giy"],
    "resale": ["resell", "resale", "sell", "price", "worth", "second-hand", "second hand", "secondhand",
               "satarsam", "satabilir", "satmak", "satış", "satis", "fiyat", "kaç para", "kac para", "ne kadar eder", "kaç eder", "kaça", "ikinci el"],
    "similar": ["similar", "like this", "alternative", "lookalike", "look-alike", "benzer", "alternatif", "muadil"],
    "color": ["colour", "color", "renk"],
    "persona": ["persona", "which style", "what style", "style type", "stil", "tarz"],
    "details": ["what is", "what's this", "tell me about", "describe", "details", "nedir", "hakkında", "hakkinda", "anlat"],
    "fit": ["tight", "loose", "small", "too big", "large", "fit", "size", "runs small", "runs large", "comfortable",
            "itchy", "didn't like", "did not like", "love it", "hate", "dar", "bol", "küçük", "kucuk", "büyük", "buyuk",
            "beden", "oturmad", "rahat", "sıktı", "beğenmed", "begenmed", "beğendim"],
    "churn": ["churn", "stop buying", "stop shopping", "leave", "at risk", "risk", "retention", "terk", "kaybet", "bırak"],
    "profile": ["segment", "rfm", "cltv", "lifetime", "profile", "profil"],
}
_ARTICLE_TOOLS = {"pair": "get_recommendations", "resale": "estimate_resale_price", "similar": "visual_search",
                  "color": "get_color_info", "persona": "get_style_persona", "details": "get_article_details"}


def _norm_text(t: str) -> str:
    """Lower-case with Turkish İ/I handled and dotless ı folded to i, so keywords match either spelling."""
    return (t or "").replace("İ", "i").replace("I", "ı").lower().replace("ı", "i")


# Keywords must start a word ("dar" matches "dar", "dardı" but not "kadar"; "fit" not "outfit")
_INTENT_RX = {k: re.compile(r"(?<![^\W_])(?:" + "|".join(re.escape(_norm_text(w)) for w in v) + ")")
              for k, v in _INTENTS.items()}


def _hits(text: str, intent: str) -> bool:
    return bool(_INTENT_RX[intent].search(text))


def _route(message, profile, selected_article_id, wardrobe, customer_id):
    """Plan of (tool, args) for clear questions; None means let the model choose tools itself."""
    text = _norm_text(message)
    ids = _ID.findall(message or "")
    article = ids[0] if ids else (selected_article_id or (wardrobe[0] if wardrobe and len(wardrobe) == 1 else None))
    allowed = set(profile["tools"])
    plan = []
    if article:
        for intent, tool in _ARTICLE_TOOLS.items():
            if tool in allowed and _hits(text, intent):
                args = {"article_id": article}
                if tool == "estimate_resale_price":
                    if re.search(r"(?<![^\W_])(new|brand new|with tags|yeni|etiketli|hic giyilmemis|hiç giyilmemiş)(?![^\W_])", text):
                        args["item_condition_id"] = 1
                plan.append((tool, args))
    if "predict_fit_sentiment" in allowed and _hits(text, "fit"):
        plan.append(("predict_fit_sentiment", {"review_text": message}))
    if customer_id:
        if "predict_churn_risk" in allowed and _hits(text, "churn"):
            plan.append(("predict_churn_risk", {"customer_id": customer_id}))
        if "get_customer_profile" in allowed and _hits(text, "profile"):
            plan.append(("get_customer_profile", {"customer_id": customer_id}))
    return plan or None


def _wardrobe_look(wardrobe, exclude=None, top_n=5):
    """'Complete the look' for the whole wardrobe: rules whose antecedents the owned product types cover."""
    types = []
    for aid in wardrobe or []:
        row = catalog.get(aid) if catalog.articles is not None else None
        if row and row.get("product_type_name"):
            types.append(row["product_type_name"])
    if not types or retail.rules is None:
        return None
    recs = retail.rules_for_types(types, top_n)
    for r in recs:
        ex = catalog.sample_by_type(r["product_type"], exclude=[exclude] if exclude else (), n=1, seed_key="wardrobe")
        r["example_article"] = catalog.short(ex[0]) if ex else None
    return {"wardrobe_product_types": sorted(set(types)), "recommendations": recs}


def _compact(obj, limit=1400) -> str:
    out = json.dumps(obj, default=str, ensure_ascii=False)
    return out if len(out) <= limit else out[:limit] + "…"


def _answer_with_router(client, state, agent, profile, instructions, user_message, plan, wardrobe, used, seen,
                        history_note=None):
    results = []
    for tool, args in plan:
        out = json.loads(_run_call(tool, args, profile, agent, used, seen))
        if tool == "get_recommendations" and not out.get("recommendations") and wardrobe:
            look = _wardrobe_look(wardrobe, exclude=args.get("article_id"))
            if look and look["recommendations"]:
                out = {"selected_piece": {k: out.get(k) for k in ("article_id", "product_type")},
                       "note": "No rules for the selected piece's product type; these come from the whole wardrobe.",
                       "wardrobe_complete_the_look": look}
                _collect_ids(json.dumps(look), seen)
        results.append(f"{tool}: {_compact(out)}")

    prompt = (user_message + "\n\nAuraSync tool results (already run for this question):\n" + "\n".join(results)
              + "\n\nAnswer the question from these results only, in at most 6 short lines.")
    sys_msg = instructions + "\nThe needed AuraSync tools have already been run; do not ask to run them again."
    history = state.setdefault("messages", [])
    msgs = _trim(history, config.CHAT_HISTORY_MESSAGES)

    if config.LLM_API == "ollama":
        data = client.chat(_ollama_payload([{"role": "system", "content": sys_msg}] + msgs + [{"role": "user", "content": prompt + _no_think_suffix()}]))
        text = (data.get("message") or {}).get("content") or ""
    elif config.LLM_API == "chat":
        kwargs = {"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": sys_msg}] + msgs
                  + [{"role": "user", "content": prompt + _no_think_suffix()}]}
        if config.CHAT_TEMPERATURE is not None:
            kwargs["temperature"] = config.CHAT_TEMPERATURE
        text = _create_chat(client, kwargs).choices[0].message.content or ""
    else:
        kwargs = {"model": config.LLM_MODEL, "instructions": sys_msg, "input": [{"role": "user", "content": prompt}]}
        if state.get("previous_response_id"):
            kwargs["previous_response_id"] = state["previous_response_id"]
        response = _call(client.responses.create, **kwargs)
        state["previous_response_id"] = response.id
        text = _output_text(response)

    text = _clean_reply(text)
    history.extend([{"role": "user", "content": history_note or user_message}, {"role": "assistant", "content": text}])
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def chat(session_id: str | None, message: str, customer_id: str | None = None, agent: str = "AI Concierge",
         wardrobe: list | None = None, selected_article_id: str | None = None):
    """Returns (session_id, reply_text, tools_used)."""
    client = _get_client()
    agent = agent if agent in AGENTS else "AI Concierge"
    sid = session_id or str(uuid.uuid4())
    state = _session(sid)
    profile = AGENTS[agent]
    instructions = f"{SYSTEM_BASE}\n\nYour current role: {agent}. {profile['instructions']}"
    user_message = _app_context(wardrobe, selected_article_id) + message
    if customer_id:
        user_message = f"[customer_id={customer_id}] {user_message}"
    seen = state.setdefault("known_ids", set())  # article IDs the user/app/tools have provided in this session
    _collect_ids(user_message, seen)
    used: list = []
    plan = _route(message, profile, selected_article_id, wardrobe, customer_id) if config.CHAT_ROUTER else None
    if plan:
        sel = next((l for l in user_message.splitlines() if l.startswith("Selected piece:")), "")
        note = (f"({sel})\n" if sel else "") + message
        text = _answer_with_router(client, state, agent, profile, instructions, user_message, plan, wardrobe, used, seen,
                                   history_note=note)
    else:
        loop = {"chat": _loop_chat, "ollama": _loop_ollama}.get(config.LLM_API, _loop_responses)
        text = loop(client, state, sid, agent, profile, instructions, user_message, used, seen)
    text = _guard_ids(_clean_reply(text or ""), seen)
    state["turns"] += 1
    return sid, text or "I couldn't generate a response.", used


def status() -> dict:
    return {
        "provider": config.LLM_PROVIDER,
        "api": config.LLM_API,
        "base_url": config.LLM_BASE_URL or "https://api.openai.com/v1",
        "model": config.LLM_MODEL,
        "reasoning_effort": config.LLM_REASONING_EFFORT if config.LLM_API in ("chat", "ollama") else None,
        "router": config.CHAT_ROUTER,
        "num_ctx": config.OLLAMA_NUM_CTX if config.LLM_API == "ollama" else None,
        "api_key_configured": bool(config.LLM_API_KEY),
        "openai_package_installed": OpenAI is not None,
        "agents": {name: p["tools"] for name, p in AGENTS.items()},
        "artifacts": {
            "catalog": catalog.load_status, "retail_strategist": retail.load_status,
            "visionary": visionary.load_status, "stylist": stylist.load_status, "economist": economist.load_status,
        },
        "artifact_load_errors": dict(LOAD_ERRORS),
        "data_dir": str(config.BASE_DIR),
        "frontend": str(config.FRONTEND_PATH) if config.FRONTEND_PATH else None,
        "active_sessions": len(_sessions),
    }

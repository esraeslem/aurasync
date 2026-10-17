"""
AuraSync transaction-confirmation email service (mock/log mode).

Fires when a purchase completes and composes a confirmation email — including
a "you might also like" cross-sell line reusing your existing ARL-rules bridge
(article_id -> category via catalog_metadata -> matching rules) — but instead
of sending it for real, logs it to a local JSONL file. This is intentional:
it's safe to demo/grade without SMTP credentials or risking a real send.

To send for real later, fill in the commented-out smtplib block in
`send_transaction_email` and set SMTP_HOST / SMTP_USER / SMTP_PASS /
AURASYNC_SENDER_EMAIL in your environment.

Integration:
    from email_service import router as email_router
    app.include_router(email_router)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("aurasync.email")

DATA_DIR = Path(os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads"))
CATALOG_METADATA_PATH = Path(os.environ.get("CATALOG_METADATA_PATH", DATA_DIR / "catalog_metadata.csv"))
ARL_RULES_PATH = DATA_DIR / "arl_rules.csv"
SENT_EMAIL_LOG = Path(os.environ.get("AURASYNC_EMAIL_LOG", "sent_emails.jsonl"))

router = APIRouter(prefix="/transactions", tags=["Transaction Emails"])


class TransactionCompleteRequest(BaseModel):
    customer_id: str
    customer_email: str
    article_id: str
    price: float
    order_id: Optional[str] = None


def _cross_sell_line(article_id: str) -> str:
    """Best-effort 'you might also like' line. Fails silently (returns '') if
    the artifact files aren't there — this is a nice-to-have, not critical path."""
    try:
        if not (CATALOG_METADATA_PATH.exists() and ARL_RULES_PATH.exists()):
            return ""
        catalog = pd.read_csv(CATALOG_METADATA_PATH, dtype={"article_id": str})
        catalog["article_id"] = catalog["article_id"].str.zfill(10)
        rules = pd.read_csv(ARL_RULES_PATH)

        aid = str(article_id).zfill(10)
        row = catalog.loc[catalog.article_id == aid]
        if row.empty:
            return ""
        category = row.iloc[0]["product_type_name"]
        matches = rules[rules["antecedents"].astype(str).str.contains(category, na=False, regex=False)]
        if matches.empty:
            return ""
        rec_categories = [c.strip() for c in str(matches.iloc[0]["consequents"]).split(",") if c.strip()]
        if not rec_categories:
            return ""
        return f"Customers who bought this also liked: {', '.join(rec_categories[:3])}."
    except Exception as exc:  # noqa: BLE001
        logger.warning("cross-sell lookup failed: %s", exc)
        return ""


def compose_email(req: TransactionCompleteRequest) -> Dict[str, str]:
    cross_sell = _cross_sell_line(req.article_id)
    subject = f"Your AuraSync order {req.order_id or req.article_id} is confirmed!"
    body = (
        "Hi there,\n\n"
        "Thanks for your purchase! Here's your confirmation:\n\n"
        f"  Article: {req.article_id}\n"
        f"  Price: ${req.price:.2f}\n"
        f"  Order ID: {req.order_id or 'N/A'}\n\n"
        + (cross_sell + "\n\n" if cross_sell else "")
        + "Thanks for shopping with AuraSync!\n"
    )
    return {"to": req.customer_email, "subject": subject, "body": body}


def send_transaction_email(req: TransactionCompleteRequest) -> Dict[str, Any]:
    email = compose_email(req)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "customer_id": req.customer_id, **email}

    logger.info("MOCK EMAIL SENT: %s", record)
    with open(SENT_EMAIL_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    # --- Real SMTP version — uncomment and configure when you're ready -------
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(email["body"])
    # msg["Subject"] = email["subject"]
    # msg["From"] = os.environ["AURASYNC_SENDER_EMAIL"]
    # msg["To"] = email["to"]
    # with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as server:
    #     server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
    #     server.send_message(msg)
    # ---------------------------------------------------------------------

    return record


@router.post("/complete")
def transaction_complete(req: TransactionCompleteRequest) -> Dict[str, Any]:
    try:
        record = send_transaction_email(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to send confirmation email: {exc}")
    return {"status": "email_sent (mock)", "email": {"to": record["to"], "subject": record["subject"]}}


@router.get("/emails/sent")
def list_sent_emails(limit: int = 20) -> Dict[str, Any]:
    """Debug helper: view the last N mock emails — handy for your demo/grading."""
    if not SENT_EMAIL_LOG.exists():
        return {"emails": []}
    lines = SENT_EMAIL_LOG.read_text().strip().splitlines()
    return {"emails": [json.loads(line) for line in lines[-limit:] if line]}

"""
PartnerScout AI — Client Error & Event Logging Router.

Endpoints:
  POST /api/v1/log/error  — receive frontend JS errors, log to jarvis_error_log
  POST /api/v1/log/event  — receive pipeline events for JARVIS audit trail

All logs stored in Supabase jarvis_error_log and jarvis_audit_log tables.
Also sends critical errors to JARVIS via webhook.
"""

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Request, status
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/log", tags=["log"])


# ── Models ────────────────────────────────────────────────────────────────────

class ErrorReport(BaseModel):
    """Frontend JS error report payload."""
    context: str              # e.g. "showDone/preview_fallback"
    message: str              # Error message
    order_id: Optional[str] = None
    ts: Optional[str] = None  # ISO timestamp from client


class EventReport(BaseModel):
    """Pipeline event report payload."""
    event: str                # e.g. "pipeline.started", "pipeline.completed"
    order_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _log_to_supabase(db_pool: Any, service: str, error: str, context: str = "") -> None:
    """Insert error record into jarvis_error_log."""
    if db_pool is None:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jarvis_error_log
                    (timestamp, service, error, retry_count)
                VALUES (NOW(), $1, $2, 0)
                """,
                f"partnerscout/{service}",
                f"[{context}] {error}"[:500],  # cap at 500 chars
            )
    except Exception as e:
        logger.warning(f"[LOG][_log_to_supabase] Could not write to jarvis_error_log: {e}")


async def _notify_jarvis(webhook_url: str, payload: dict[str, Any]) -> None:
    """Send error notification to JARVIS webhook."""
    if not webhook_url:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(webhook_url, json=payload)
    except Exception as e:
        logger.warning(f"[LOG][_notify_jarvis] Webhook failed: {e}")


# ── POST /api/v1/log/error ────────────────────────────────────────────────────

@router.post(
    "/error",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Receive frontend JS error for JARVIS logging",
)
async def log_client_error(payload: ErrorReport, request: Request) -> None:
    """
    Accept error reports from the dashboard frontend.

    Logs to:
    - Python logger (Railway logs)
    - Supabase jarvis_error_log table
    - JARVIS webhook (if configured)

    Args:
        payload: ErrorReport with context, message, and optional order_id.
        request: HTTP request (for db_pool and config access).
    """
    logger.error(
        f"[PARTNERSCOUT][CLIENT_ERROR] context={payload.context} "
        f"order={payload.order_id} error={payload.message}"
    )

    db_pool = getattr(request.app.state, "db_pool", None)
    config  = getattr(request.app.state, "config", None)

    # Non-blocking: log to DB and notify JARVIS
    asyncio.create_task(
        _log_to_supabase(db_pool, "dashboard_js", payload.message, payload.context)
    )

    if config and getattr(config, "JARVIS_WEBHOOK_URL", None):
        asyncio.create_task(
            _notify_jarvis(config.JARVIS_WEBHOOK_URL, {
                "event":    "partnerscout.client_error",
                "context":  payload.context,
                "order_id": payload.order_id,
                "message":  payload.message,
                "ts":       payload.ts,
            })
        )


# ── POST /api/v1/log/event ────────────────────────────────────────────────────

@router.post(
    "/event",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log a pipeline event to JARVIS audit trail",
)
async def log_event(payload: EventReport, request: Request) -> None:
    """
    Accept pipeline event notifications for the JARVIS audit trail.

    Args:
        payload: EventReport with event name and optional details.
        request: HTTP request.
    """
    logger.info(
        f"[PARTNERSCOUT][EVENT] event={payload.event} order={payload.order_id} "
        f"details={payload.details}"
    )

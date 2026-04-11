"""
PartnerScout AI — Webhook Router.

Endpoints:
  POST /api/v1/webhook/stripe  — handle Stripe payment events (disabled until Stripe enabled)
  GET  /api/v1/webhook/health  — JARVIS health check endpoint
"""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from loguru import logger

from api.config import Settings, get_settings
from api.db.client import get_order, update_order_status
from api.worker.pipeline import run_pipeline

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_db_pool(request: Request) -> Any:
    """Extract asyncpg pool from app state."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available.",
        )
    return pool


async def _launch_pipeline_bg(
    order_id: str,
    order_data: dict[str, Any],
    db_pool: Any,
    config: Settings,
) -> None:
    """Background task: run pipeline for a newly paid order."""
    try:
        await run_pipeline(order_id, order_data, db_pool, config, is_trial=False)
    except Exception as e:
        logger.error(
            f"[WEBHOOK][_launch_pipeline_bg] Pipeline error for order={order_id}: {e}",
            exc_info=True,
        )


# ── POST /api/v1/webhook/stripe ───────────────────────────────────────────────

@router.post(
    "/stripe",
    summary="Handle Stripe webhook events (disabled until Stripe enabled)",
    status_code=status.HTTP_200_OK,
)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Stripe webhook handler — lazy imports stripe only when called.
    Returns 503 if stripe package not installed yet.
    """
    try:
        import stripe as _stripe  # noqa: PLC0415
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment processing not yet enabled.",
        )

    config = get_settings()
    db_pool = _get_db_pool(request)

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = _stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=config.STRIPE_WEBHOOK_SECRET,
        )
    except _stripe.SignatureVerificationError as e:
        logger.error(f"[WEBHOOK][stripe_webhook] Invalid signature: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe signature.")
    except Exception as e:
        logger.error(f"[WEBHOOK][stripe_webhook] Webhook parse error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not parse webhook payload.")

    event_type = event.get("type", "")
    logger.info(f"[WEBHOOK][stripe_webhook] Received event type: {event_type}")

    if event_type == "payment_intent.succeeded":
        await _handle_payment_succeeded(event["data"]["object"], db_pool, config, background_tasks)

    return {"received": "ok"}


async def _handle_payment_succeeded(
    payment_intent: dict[str, Any],
    db_pool: Any,
    config: Settings,
    background_tasks: BackgroundTasks,
) -> None:
    """Handle payment_intent.succeeded — launch pipeline for matching order."""
    intent_id = payment_intent.get("id", "")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")

    if not order_id:
        logger.warning(f"[WEBHOOK] No order_id in metadata for intent={intent_id}")
        return

    order = await get_order(db_pool, order_id)
    if not order or order["status"] != "pending":
        logger.warning(f"[WEBHOOK] Order {order_id} not found or already processed")
        return

    await update_order_status(db_pool, order_id, "pending", 0)
    logger.info(f"[WEBHOOK] Launching pipeline for order={order_id}")
    background_tasks.add_task(_launch_pipeline_bg, order_id, dict(order), db_pool, config)


# ── GET /api/v1/webhook/health ────────────────────────────────────────────────

@router.get(
    "/health",
    summary="JARVIS health check endpoint",
    status_code=status.HTTP_200_OK,
)
async def jarvis_health_check() -> dict[str, str]:
    """Health check for JARVIS monitoring agent."""
    return {"status": "ok", "service": "partnerscout"}

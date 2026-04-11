"""
PartnerScout AI — Webhook Router.

Endpoints:
  POST /api/v1/webhook/stripe  — handle Stripe payment events
  GET  /api/v1/webhook/health  — JARVIS health check endpoint
"""

from typing import Any

import stripe
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from loguru import logger

from api.config import Settings, get_settings
from api.db.client import create_order, get_order, update_order_status
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
    """
    Background task: run pipeline for a newly paid order.

    Args:
        order_id: Order UUID string.
        order_data: Full order dict.
        db_pool: asyncpg pool.
        config: Application settings.
    """
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
    summary="Handle Stripe webhook events",
    status_code=status.HTTP_200_OK,
)
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Process incoming Stripe webhook events.

    Handles:
      - payment_intent.succeeded → locate pending order, launch pipeline
      - All other events → acknowledge with 200 (no action)

    Signature verification uses STRIPE_WEBHOOK_SECRET env var.
    Events without a matching order are logged and acknowledged.

    Args:
        request: Raw HTTP request (body used for signature verification).
        background_tasks: FastAPI background task queue.

    Returns:
        Dict with {"received": "ok"} on success.

    Raises:
        HTTPException 400: Invalid Stripe signature.
        HTTPException 503: Database unavailable.
    """
    config = get_settings()
    db_pool = _get_db_pool(request)

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=config.STRIPE_WEBHOOK_SECRET,
        )
    except stripe.SignatureVerificationError as e:
        logger.error(f"[WEBHOOK][stripe_webhook] Invalid signature: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe signature.",
        )
    except Exception as e:
        logger.error(f"[WEBHOOK][stripe_webhook] Webhook parse error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse webhook payload.",
        )

    event_type = event.get("type", "")
    logger.info(f"[WEBHOOK][stripe_webhook] Received event type: {event_type}")

    if event_type == "payment_intent.succeeded":
        await _handle_payment_succeeded(
            event["data"]["object"],
            db_pool,
            config,
            background_tasks,
        )

    return {"received": "ok"}


async def _handle_payment_succeeded(
    payment_intent: dict[str, Any],
    db_pool: Any,
    config: Settings,
    background_tasks: BackgroundTasks,
) -> None:
    """
    Handle payment_intent.succeeded Stripe event.

    Looks for a pending order with matching stripe_payment_id.
    If found, launches pipeline. If not, logs warning (may be handled
    via POST /orders flow instead).

    Args:
        payment_intent: Stripe PaymentIntent object dict.
        db_pool: asyncpg pool.
        config: Application settings.
        background_tasks: FastAPI background task queue.
    """
    intent_id = payment_intent.get("id", "")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")

    if not order_id:
        logger.warning(
            f"[WEBHOOK][_handle_payment_succeeded] "
            f"No order_id in metadata for intent={intent_id}. "
            f"Order created via POST /orders flow — skipping."
        )
        return

    order = await get_order(db_pool, order_id)
    if not order:
        logger.warning(
            f"[WEBHOOK][_handle_payment_succeeded] "
            f"Order {order_id} not found in DB for intent={intent_id}"
        )
        return

    if order["status"] != "pending":
        logger.info(
            f"[WEBHOOK][_handle_payment_succeeded] "
            f"Order {order_id} already in status={order['status']} — skipping"
        )
        return

    await update_order_status(db_pool, order_id, "pending", 0)
    logger.info(f"[WEBHOOK][_handle_payment_succeeded] Launching pipeline for order={order_id}")

    order_data = dict(order)
    background_tasks.add_task(
        _launch_pipeline_bg,
        order_id,
        order_data,
        db_pool,
        config,
    )


# ── GET /api/v1/webhook/health ────────────────────────────────────────────────

@router.get(
    "/health",
    summary="JARVIS health check endpoint",
    status_code=status.HTTP_200_OK,
)
async def jarvis_health_check() -> dict[str, str]:
    """
    Health check endpoint for JARVIS monitoring.

    Called by JARVIS QA/Health Agent to verify service is alive.
    Returns service identity and status.

    Returns:
        Dict with status and service name.
    """
    return {
        "status": "ok",
        "service": "partnerscout",
    }

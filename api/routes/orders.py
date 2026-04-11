"""
PartnerScout AI — Orders API Router.

Endpoints:
  POST /api/v1/orders/trial  — create free trial order (10 blurred leads)
  POST /api/v1/orders        — create paid order (requires Stripe payment)
  GET  /api/v1/orders/{id}  — get order status and progress
"""

import asyncio
from typing import Any
from uuid import UUID

import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from loguru import logger

from api.config import Settings, get_settings
from api.db.client import create_order, get_order, get_results
from api.engine.exporter import blur_for_trial
from api.models.order import OrderCreate, OrderStatus, OrderStatusEnum
from api.worker.pipeline import run_pipeline

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


# ── Dependency Helpers ────────────────────────────────────────────────────────

def _get_db_pool(request: Request) -> Any:
    """Extract asyncpg pool from app state."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available. Please try again shortly.",
        )
    return pool


# ── Background Task Wrapper ───────────────────────────────────────────────────

async def _launch_pipeline(
    order_id: str,
    order_data: dict[str, Any],
    db_pool: Any,
    config: Settings,
    is_trial: bool,
) -> None:
    """
    Background task wrapper for the pipeline.

    Runs in FastAPI BackgroundTasks — errors are logged, never re-raised.

    Args:
        order_id: Order UUID string.
        order_data: Full order dict.
        db_pool: asyncpg pool.
        config: Application settings.
        is_trial: Trial mode flag.
    """
    try:
        await run_pipeline(order_id, order_data, db_pool, config, is_trial)
    except Exception as e:
        logger.error(
            f"[ORDERS][_launch_pipeline] Unhandled error for order={order_id}: {e}",
            exc_info=True,
        )


# ── POST /api/v1/orders/trial ─────────────────────────────────────────────────

@router.post(
    "/trial",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a free trial order (10 blurred leads)",
)
async def create_trial_order(
    payload: OrderCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    config: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Create a trial lead generation order.

    No payment required. Returns 10 leads with blurred contact details.
    Pipeline starts immediately as a background task.

    Args:
        payload: Order creation payload.
        background_tasks: FastAPI background task queue.
        request: HTTP request (for db_pool access).
        config: Application settings.

    Returns:
        Dict with order_id, status, and user message.
    """
    db_pool = _get_db_pool(request)

    trial_payload = payload.model_copy(update={"is_trial": True})
    order_dict = trial_payload.model_dump()

    order_id = await create_order(db_pool, order_dict)
    logger.info(f"[ORDERS][create_trial_order] Trial order created: {order_id}")

    background_tasks.add_task(
        _launch_pipeline,
        str(order_id),
        order_dict,
        db_pool,
        config,
        True,
    )

    return {
        "order_id": str(order_id),
        "status": "running",
        "message": "Your 10 preview leads are being prepared. This takes 2–3 minutes.",
        "poll_url": f"/api/v1/orders/{order_id}",
    }


# ── POST /api/v1/orders/admin ────────────────────────────────────────────────

@router.post(
    "/admin",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Admin order — full results, no blur, no Stripe (owner only)",
)
async def create_admin_order(
    payload: OrderCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    config: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Create a full admin order for the owner.

    Requires X-Admin-Secret header matching ADMIN_SECRET env var.
    Returns full unblurred results, no payment required.
    Count target set to 50 companies for testing.

    Args:
        payload: Order creation payload.
        background_tasks: FastAPI background task queue.
        request: HTTP request.
        config: Application settings.

    Returns:
        Dict with order_id, status, and poll_url.

    Raises:
        HTTPException 403: If admin secret is missing or invalid.
    """
    # Verify admin secret
    admin_secret = request.headers.get("X-Admin-Secret", "")
    if not config.ADMIN_SECRET or admin_secret != config.ADMIN_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin secret.",
        )

    db_pool = _get_db_pool(request)

    order_dict = payload.model_dump()
    order_dict["is_trial"] = False
    order_dict["count_target"] = 50  # Full test batch

    order_id = await create_order(db_pool, order_dict)
    logger.info(f"[ORDERS][create_admin_order] Admin order created: {order_id}")

    background_tasks.add_task(
        _launch_pipeline,
        str(order_id),
        order_dict,
        db_pool,
        config,
        False,  # is_trial=False → full unblurred results
    )

    return {
        "order_id": str(order_id),
        "status": "running",
        "message": "Admin order started. Full unblurred results, 50 companies.",
        "poll_url": f"/api/v1/orders/{order_id}",
    }


# ── POST /api/v1/orders ───────────────────────────────────────────────────────

class PaidOrderCreate(OrderCreate):
    """Extended order payload requiring Stripe payment intent."""

    stripe_payment_intent_id: str


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a paid order (requires Stripe payment intent)",
)
async def create_paid_order(
    payload: PaidOrderCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    config: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Create a paid lead generation order after Stripe payment.

    Verifies payment intent status before creating the order.
    Pipeline starts immediately as a background task.

    Args:
        payload: Order creation payload with stripe_payment_intent_id.
        background_tasks: FastAPI background task queue.
        request: HTTP request (for db_pool access).
        config: Application settings.

    Returns:
        Dict with order_id and status.

    Raises:
        HTTPException 402: If payment intent is not succeeded.
        HTTPException 400: If payment intent ID is invalid.
    """
    db_pool = _get_db_pool(request)

    # Verify Stripe payment
    stripe.api_key = config.STRIPE_SECRET_KEY
    try:
        intent = stripe.PaymentIntent.retrieve(payload.stripe_payment_intent_id)
    except stripe.StripeError as e:
        logger.error(f"[ORDERS][create_paid_order] Stripe error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payment intent: {str(e)}",
        )

    if intent.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Payment not completed. Intent status: {intent.status}",
        )

    order_dict = payload.model_dump()
    order_dict["stripe_payment_id"] = payload.stripe_payment_intent_id
    order_dict["is_trial"] = False

    order_id = await create_order(db_pool, order_dict)
    logger.info(f"[ORDERS][create_paid_order] Paid order created: {order_id}")

    background_tasks.add_task(
        _launch_pipeline,
        str(order_id),
        order_dict,
        db_pool,
        config,
        False,
    )

    return {
        "order_id": str(order_id),
        "status": "running",
        "poll_url": f"/api/v1/orders/{order_id}",
    }


# ── GET /api/v1/orders/{order_id} ────────────────────────────────────────────

@router.get(
    "/{order_id}",
    summary="Get order status and results",
)
async def get_order_status(
    order_id: UUID,
    request: Request,
) -> dict[str, Any]:
    """
    Retrieve order status, progress, and results when completed.

    For done trial orders: returns blurred 10-company preview.
    For done paid orders: returns download URLs.

    Args:
        order_id: Order UUID.
        request: HTTP request (for db_pool access).

    Returns:
        Dict with status, progress, and optional results/download links.

    Raises:
        HTTPException 404: If order not found.
    """
    db_pool = _get_db_pool(request)

    order = await get_order(db_pool, str(order_id))
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )

    response: dict[str, Any] = {
        "id": str(order["id"]),
        "status": order["status"],
        "progress": order["progress"],
        "is_trial": order["is_trial"],
        "created_at": order["created_at"].isoformat() if order.get("created_at") else None,
        "result_url": order.get("result_url"),
        "error_msg": order.get("error_msg"),
    }

    # Attach results if done
    if order["status"] == "done":
        if order["is_trial"]:
            companies = await get_results(db_pool, str(order_id))
            response["preview"] = blur_for_trial(companies)
            response["message"] = (
                "Unlock full contact details and download all leads at partnerscout.ai"
            )
        else:
            response["download"] = {
                "csv": f"/api/v1/export/{order_id}/csv",
                "json": f"/api/v1/export/{order_id}/json",
            }

    return response

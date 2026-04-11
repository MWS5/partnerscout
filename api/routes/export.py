"""
PartnerScout AI — Export Router.

Endpoints:
  GET /api/v1/export/{order_id}/csv     — download full CSV (paid only)
  GET /api/v1/export/{order_id}/json    — download full JSON (paid only)
  GET /api/v1/export/{order_id}/preview — blurred 10-company preview (trial)
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from loguru import logger

from api.db.client import get_order, get_results
from api.engine.exporter import blur_for_trial, to_csv, to_json

router = APIRouter(prefix="/api/v1/export", tags=["export"])


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


async def _get_verified_order(
    db_pool: Any,
    order_id: str,
    require_paid: bool = True,
) -> dict[str, Any]:
    """
    Fetch and validate order for export access.

    Checks:
      - Order exists
      - Order status is 'done'
      - For paid exports: order is not a trial (is_trial=False)

    Args:
        db_pool: asyncpg connection pool.
        order_id: Order UUID string.
        require_paid: If True, rejects trial orders.

    Returns:
        Order dict from DB.

    Raises:
        HTTPException 404: Order not found.
        HTTPException 402: Trial order accessing paid export.
        HTTPException 425: Order not yet completed.
    """
    order = await get_order(db_pool, order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found.",
        )

    if order["status"] != "done":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Order not ready. Current status: {order['status']} ({order['progress']}%)",
        )

    if require_paid and order.get("is_trial", False):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Full export requires a paid order. Upgrade at partnerscout.ai",
        )

    return order


# ── GET /api/v1/export/{order_id}/csv ────────────────────────────────────────

@router.get(
    "/{order_id}/csv",
    summary="Download full leads as CSV (paid orders only)",
    response_class=Response,
)
async def export_csv(order_id: UUID, request: Request) -> Response:
    """
    Download all enriched leads as a CSV file.

    Requires a completed paid order. Trial orders are rejected.

    Args:
        order_id: Order UUID.
        request: HTTP request (for db_pool access).

    Returns:
        CSV file download response.
    """
    db_pool = _get_db_pool(request)
    await _get_verified_order(db_pool, str(order_id), require_paid=True)

    companies = await get_results(db_pool, str(order_id))
    if not companies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found for this order.",
        )

    csv_content = to_csv(companies)
    filename = f"partnerscout_{order_id}.csv"

    logger.info(f"[EXPORT][export_csv] Serving {len(companies)} companies for order={order_id}")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── GET /api/v1/export/{order_id}/json ───────────────────────────────────────

@router.get(
    "/{order_id}/json",
    summary="Download full leads as JSON (paid orders only)",
    response_class=Response,
)
async def export_json(order_id: UUID, request: Request) -> Response:
    """
    Download all enriched leads as a formatted JSON file.

    Requires a completed paid order. Trial orders are rejected.

    Args:
        order_id: Order UUID.
        request: HTTP request (for db_pool access).

    Returns:
        JSON file download response.
    """
    db_pool = _get_db_pool(request)
    await _get_verified_order(db_pool, str(order_id), require_paid=True)

    companies = await get_results(db_pool, str(order_id))
    if not companies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found for this order.",
        )

    json_content = to_json(companies)
    filename = f"partnerscout_{order_id}.json"

    logger.info(f"[EXPORT][export_json] Serving {len(companies)} companies for order={order_id}")
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── GET /api/v1/export/{order_id}/preview ────────────────────────────────────

@router.get(
    "/{order_id}/preview",
    summary="Get 10 blurred leads preview (trial orders)",
)
async def export_preview(order_id: UUID, request: Request) -> dict[str, Any]:
    """
    Return 10 blurred company records for trial order preview.

    Available for both trial and paid completed orders.
    Contact details are blurred to incentivize conversion.

    Args:
        order_id: Order UUID.
        request: HTTP request (for db_pool access).

    Returns:
        Dict with blurred companies list and conversion CTA.
    """
    db_pool = _get_db_pool(request)
    order = await _get_verified_order(db_pool, str(order_id), require_paid=False)

    companies = await get_results(db_pool, str(order_id))
    blurred = blur_for_trial(companies)

    logger.info(f"[EXPORT][export_preview] Serving {len(blurred)} blurred records for order={order_id}")
    return {
        "order_id": str(order_id),
        "is_trial": order.get("is_trial", True),
        "companies": blurred,
        "total_in_preview": len(blurred),
        "unlock_cta": "Get full contact details — upgrade at partnerscout.ai",
    }

"""
PartnerScout AI — Database Client.

Provides async database operations using asyncpg connection pool.
Configured for Supabase Postgres pooler (pgbouncer mode).

CRITICAL: statement_cache_size=0 is required for Supabase pgbouncer.
Without this, prepared statement conflicts crash the connection pool.

All functions:
  - Accept pool as first arg (dependency injection pattern)
  - Are fully async
  - Log errors with full context
  - Never expose raw exceptions to callers
"""

import json
from typing import Any, Optional

import asyncpg
from loguru import logger


# ── Pool Factory ──────────────────────────────────────────────────────────────

async def get_pool(database_url: str) -> asyncpg.Pool:
    """
    Create and return an asyncpg connection pool for Supabase.

    CRITICAL: statement_cache_size=0 prevents prepared statement
    conflicts with Supabase pgbouncer in transaction mode.

    Args:
        database_url: Postgres connection string (Supabase pooler URL).

    Returns:
        Configured asyncpg connection pool.

    Raises:
        asyncpg.PostgresError: If connection cannot be established.
    """
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=10,
        statement_cache_size=0,  # CRITICAL for Supabase pgbouncer
        command_timeout=30.0,
    )
    logger.info("[DB][get_pool] Connection pool created successfully")
    return pool


# ── Order Operations ──────────────────────────────────────────────────────────

async def create_order(pool: asyncpg.Pool, order_data: dict[str, Any]) -> str:
    """
    Insert a new order record into ps_orders and return its UUID.

    Args:
        pool: asyncpg connection pool.
        order_data: Dict with order fields matching ps_orders schema.

    Returns:
        UUID string of the newly created order.

    Raises:
        Exception: Propagated on DB error after logging.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO ps_orders
                    (email, niches, regions, segment, count_target, is_trial,
                     status, progress, stripe_payment_id)
                VALUES ($1, $2, $3, $4, $5, $6, 'pending', 0, $7)
                RETURNING id
                """,
                order_data["email"],
                order_data.get("niches", []),
                order_data.get("regions", []),
                order_data.get("segment", "luxury"),
                order_data.get("count_target", 100),
                order_data.get("is_trial", False),
                order_data.get("stripe_payment_id"),
            )
        order_id = str(row["id"])
        logger.info(f"[DB][create_order] Created order {order_id} for {order_data['email']}")
        return order_id
    except Exception as e:
        logger.error(f"[DB][create_order] Error: {e}", exc_info=True)
        raise


async def update_order_status(
    pool: asyncpg.Pool,
    order_id: str,
    status: str,
    progress: int,
    result_url: Optional[str] = None,
    error_msg: Optional[str] = None,
) -> None:
    """
    Update order status, progress, and optional result URL or error message.

    Sets completed_at timestamp when status transitions to 'done' or 'failed'.

    Args:
        pool: asyncpg connection pool.
        order_id: Order UUID string.
        status: New status ('pending', 'running', 'done', 'failed').
        progress: Progress percentage (0–100).
        result_url: Optional download URL (set when done).
        error_msg: Optional error message (set when failed).
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ps_orders
                SET
                    status = $1,
                    progress = $2,
                    result_url = COALESCE($3, result_url),
                    error_msg = COALESCE($4, error_msg),
                    completed_at = CASE
                        WHEN $1 IN ('done', 'failed') THEN NOW()
                        ELSE completed_at
                    END
                WHERE id = $5::uuid
                """,
                status,
                progress,
                result_url,
                error_msg,
                order_id,
            )
        logger.debug(f"[DB][update_order_status] order={order_id} status={status} progress={progress}%")
    except Exception as e:
        logger.error(f"[DB][update_order_status] Error for order={order_id}: {e}", exc_info=True)


async def get_order(pool: asyncpg.Pool, order_id: str) -> Optional[dict[str, Any]]:
    """
    Fetch a single order by UUID.

    Args:
        pool: asyncpg connection pool.
        order_id: Order UUID string.

    Returns:
        Order dict, or None if not found.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ps_orders WHERE id = $1::uuid",
                order_id,
            )
        if row is None:
            return None
        return dict(row)
    except Exception as e:
        logger.error(f"[DB][get_order] Error for order={order_id}: {e}", exc_info=True)
        return None


# ── Results Operations ────────────────────────────────────────────────────────

async def save_results(
    pool: asyncpg.Pool,
    order_id: str,
    companies: list[dict[str, Any]],
) -> None:
    """
    Bulk-insert enriched company records into ps_results.

    Uses executemany for efficiency. Each company dict should have
    standard CompanyRecord fields.

    Args:
        pool: asyncpg connection pool.
        order_id: Parent order UUID string.
        companies: List of enriched company dicts.
    """
    if not companies:
        logger.warning(f"[DB][save_results] No companies to save for order={order_id}")
        return

    rows = [
        (
            order_id,
            c.get("category", ""),
            c.get("company_name", ""),
            c.get("website") or c.get("url", ""),
            c.get("address", "Not found"),
            c.get("phone", "Not found"),
            c.get("email", "Not found"),
            c.get("contact_person", "Not found"),
            c.get("personal_phone", "Not found"),
            c.get("personal_email", "Not found"),
            float(c.get("luxury_score", 0.0)),
            bool(c.get("verified", False)),
            json.dumps({k: v for k, v in c.items() if k not in {
                "category", "company_name", "website", "url", "address",
                "phone", "email", "contact_person", "personal_phone",
                "personal_email", "luxury_score", "verified",
            }}),
        )
        for c in companies
    ]

    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ps_results
                    (order_id, category, company_name, website, address, phone,
                     email, contact_person, personal_phone, personal_email,
                     luxury_score, verified, raw_data)
                VALUES
                    ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
                """,
                rows,
            )
        logger.info(f"[DB][save_results] Saved {len(companies)} results for order={order_id}")
    except Exception as e:
        logger.error(f"[DB][save_results] Error for order={order_id}: {e}", exc_info=True)
        raise


async def get_results(
    pool: asyncpg.Pool,
    order_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all result records for an order, sorted by luxury_score descending.

    Args:
        pool: asyncpg connection pool.
        order_id: Parent order UUID string.

    Returns:
        List of result dicts. Empty list if none found or on error.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM ps_results
                WHERE order_id = $1::uuid
                ORDER BY luxury_score DESC
                """,
                order_id,
            )
        results = [dict(row) for row in rows]
        logger.debug(f"[DB][get_results] Found {len(results)} results for order={order_id}")
        return results
    except Exception as e:
        logger.error(f"[DB][get_results] Error for order={order_id}: {e}", exc_info=True)
        return []

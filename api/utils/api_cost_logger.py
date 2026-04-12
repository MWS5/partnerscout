"""
PartnerScout — External API Cost Logger.

Logs Google Places API and Hunter.io API usage to the shared
jarvis_search_usage Supabase table, so JARVIS cost tracker
includes PartnerScout external API spend in /balance report.

Table schema (jarvis_search_usage):
  provider, search_type, num_results, cost_usd, duration_ms, success, error

Pricing (2026):
  Google Places — Find Place:     $0.017 per call
  Google Places — Place Details:  $0.017 per call
  Google Places — per hotel:      ~$0.034 (2 calls combined)
  Hunter.io     — domain search:  $0 on Free tier (50/month)
                                  $0.068 on Paid ($34/500)
"""

import time
from typing import Optional

import asyncpg
from loguru import logger


# ── Pricing ───────────────────────────────────────────────────────────────────

GOOGLE_PLACES_COST_PER_HOTEL = 0.034   # Find Place ($0.017) + Details ($0.017)
HUNTER_COST_FREE_TIER        = 0.0     # Free plan: 50 searches/month
HUNTER_COST_PAID_PER_SEARCH  = 0.068  # $34/month / 500 searches


async def log_google_places(
    db_pool: asyncpg.Pool,
    company_name: str,
    success: bool,
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    """
    Log a Google Places API lookup to jarvis_search_usage.

    One "lookup" = 1 Find Place call + 1 Place Details call = $0.034.

    Args:
        db_pool: asyncpg connection pool.
        company_name: Company name that was searched.
        success: Whether the lookup returned data.
        duration_ms: Total time taken in milliseconds.
        error: Error message if failed.
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jarvis_search_usage
                    (provider, search_type, num_results, cost_usd, duration_ms, success, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                "google_places",
                f"hotel_lookup:{company_name[:40]}",
                1 if success else 0,
                GOOGLE_PLACES_COST_PER_HOTEL if success else 0.0,
                duration_ms,
                success,
                error,
            )
        logger.debug(
            f"[CostLogger] Google Places: '{company_name}' | "
            f"{'OK' if success else 'FAIL'} | {duration_ms}ms | ${GOOGLE_PLACES_COST_PER_HOTEL if success else 0:.4f}"
        )
    except Exception as e:
        logger.warning(f"[CostLogger][log_google_places] Failed: {e}")


async def log_hunter_io(
    db_pool: asyncpg.Pool,
    domain: str,
    success: bool,
    duration_ms: int,
    is_paid_plan: bool = False,
    error: Optional[str] = None,
) -> None:
    """
    Log a Hunter.io domain search to jarvis_search_usage.

    Free tier: $0 per search (50/month). Paid: $0.068/search.

    Args:
        db_pool: asyncpg connection pool.
        domain: Domain that was searched.
        success: Whether email was found.
        duration_ms: Total time taken in milliseconds.
        is_paid_plan: True if on paid Hunter plan (affects cost).
        error: Error message if failed.
    """
    cost = HUNTER_COST_PAID_PER_SEARCH if is_paid_plan else HUNTER_COST_FREE_TIER
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jarvis_search_usage
                    (provider, search_type, num_results, cost_usd, duration_ms, success, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                "hunter_io",
                f"domain_search:{domain[:40]}",
                1 if success else 0,
                cost,
                duration_ms,
                success,
                error,
            )
        logger.debug(
            f"[CostLogger] Hunter.io: '{domain}' | "
            f"{'OK' if success else 'FAIL'} | {duration_ms}ms | ${cost:.4f}"
        )
    except Exception as e:
        logger.warning(f"[CostLogger][log_hunter_io] Failed: {e}")

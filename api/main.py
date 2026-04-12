"""
PartnerScout AI — FastAPI Application Entry Point.

Creates and configures the FastAPI app:
  - Registers all routers (orders, export, webhook)
  - Manages asyncpg DB pool lifecycle (startup/shutdown)
  - Adds CORS middleware
  - Provides /ping health check

Startup sequence:
  1. Connect to Supabase via asyncpg pool
  2. Store pool in app.state.db_pool
  3. Log readiness

Shutdown sequence:
  1. Close all pool connections gracefully

Usage (local):
    uvicorn api.main:app --reload --port 8000

Usage (Railway):
    uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import get_settings
from api.db.client import get_pool
from api.engine.searcher import duckduckgo_search, searxng_search
from api.routes.export import router as export_router
from api.routes.log import router as log_router
from api.routes.orders import router as orders_router
from api.routes.webhook import router as webhook_router


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Handles startup (DB pool creation) and shutdown (pool close).
    Uses asynccontextmanager pattern (FastAPI 0.95+).

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    config = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("[MAIN] PartnerScout AI starting up...")

    # Store config in app state so routes can access it (for webhooks etc.)
    app.state.config = config

    try:
        pool = await get_pool(config.DATABASE_URL)
        app.state.db_pool = pool
        logger.info("[MAIN] Database pool initialized. Service is ready.")
    except Exception as e:
        logger.error(f"[MAIN] Failed to connect to database: {e}", exc_info=True)
        # Allow startup to continue — DB errors surface at request time
        app.state.db_pool = None

    yield  # Application is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[MAIN] PartnerScout AI shutting down...")
    pool = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        logger.info("[MAIN] Database pool closed.")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routers and middleware.
    """
    app = FastAPI(
        title="PartnerScout AI",
        description=(
            "B2B lead generation service for luxury business partners on the French Riviera. "
            "Searches luxury hotels, event agencies, wedding planners, concierge services, "
            "travel agencies, and venues using multi-source search + LLM enrichment."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Tighten in production to specific frontend domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(orders_router)
    app.include_router(export_router)
    app.include_router(log_router)
    app.include_router(webhook_router)

    return app


app = create_app()


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get(
    "/ping",
    tags=["health"],
    summary="Health check — returns pong within 5s",
)
async def ping() -> dict[str, str]:
    """
    Lightweight health check endpoint.

    Used by Railway healthcheckPath and JARVIS QA Agent.
    Must respond within 5 seconds.

    Returns:
        Dict with status 'pong' and service name.
    """
    return {"status": "pong", "service": "partnerscout"}


# ── Search Source Diagnostics ─────────────────────────────────────────────────

@app.get(
    "/debug/sources",
    tags=["health"],
    summary="Diagnose which search sources return results",
)
async def debug_sources() -> dict[str, Any]:
    """
    Test all configured search sources with a simple luxury hotel query.

    Returns count of results from each source so you can see which are
    working from this Railway deployment.

    No auth required — safe query only, no sensitive data exposed.
    """
    config = get_settings()
    test_query = "luxury hotel Nice 5 star"
    results: dict[str, Any] = {
        "query": test_query,
        "sources": {},
        "google_places_key_set": bool(getattr(config, "GOOGLE_PLACES_API_KEY", "")),
        "searxng_url": getattr(config, "SEARXNG_URL", "") or "NOT SET",
        "brave_key_set": bool(getattr(config, "BRAVE_API_KEY", "")),
        "serper_key_set": bool(getattr(config, "SERPER_API_KEY", "")),
        "tavily_key_set": bool(getattr(config, "TAVILY_API_KEY", "")),
    }

    # Test DDG
    try:
        ddg_results = await duckduckgo_search(test_query, num=3)
        results["sources"]["ddg"] = {"count": len(ddg_results), "ok": len(ddg_results) > 0}
    except Exception as e:
        results["sources"]["ddg"] = {"count": 0, "ok": False, "error": str(e)}

    # Test SearXNG
    searxng_url = getattr(config, "SEARXNG_URL", "") or ""
    if searxng_url:
        try:
            sx_results = await searxng_search(test_query, searxng_url, num=3)
            results["sources"]["searxng"] = {
                "url": searxng_url,
                "count": len(sx_results),
                "ok": len(sx_results) > 0,
            }
        except Exception as e:
            results["sources"]["searxng"] = {
                "url": searxng_url,
                "count": 0,
                "ok": False,
                "error": str(e),
            }
    else:
        results["sources"]["searxng"] = {"ok": False, "reason": "SEARXNG_URL not set"}

    # Test Google Places (quick check — just Text Search, 1 result)
    places_key = getattr(config, "GOOGLE_PLACES_API_KEY", "") or ""
    if places_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/place/textsearch/json",
                    params={"query": test_query, "key": places_key, "language": "en"},
                )
                data = resp.json()
                status = data.get("status", "UNKNOWN")
                count  = len(data.get("results", []))
                results["sources"]["google_places"] = {
                    "status": status,
                    "count": count,
                    "ok": status == "OK" and count > 0,
                }
        except Exception as e:
            results["sources"]["google_places"] = {"ok": False, "error": str(e)}
    else:
        results["sources"]["google_places"] = {"ok": False, "reason": "GOOGLE_PLACES_API_KEY not set"}

    return results

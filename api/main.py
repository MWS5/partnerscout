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
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import get_settings
from api.db.client import get_pool
from api.routes.export import router as export_router
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

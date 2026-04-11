"""
PartnerScout AI — Configuration Module.

Loads all environment variables using Pydantic BaseSettings.
All secrets must be set as environment variables — never hardcoded.

Usage:
    from api.config import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration loaded from environment variables.

    Follows JARVIS OS secrets naming convention: SERVICE_RESOURCE_TYPE.
    All sensitive fields are Optional where the feature can be disabled gracefully.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM / OpenRouter ──────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = Field(
        ...,
        description="OpenRouter API key for all LLM calls. Required.",
    )

    # ── Search APIs ───────────────────────────────────────────────────────────
    BRAVE_API_KEY: Optional[str] = Field(
        default=None,
        description="Brave Search API key. Optional — DDG used as fallback.",
    )
    SEARXNG_URL: Optional[str] = Field(
        default="",
        description="Base URL of self-hosted SearXNG instance. Optional.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="Supabase Postgres pooler connection string (pgbouncer).",
    )

    # ── Payments ──────────────────────────────────────────────────────────────
    STRIPE_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Stripe secret key for payment verification. Optional until payments enabled.",
    )
    STRIPE_WEBHOOK_SECRET: Optional[str] = Field(
        default=None,
        description="Stripe webhook signing secret. Optional until payments enabled.",
    )

    # ── Email Delivery ────────────────────────────────────────────────────────
    RESEND_API_KEY: Optional[str] = Field(
        default=None,
        description="Resend API key for transactional emails. Optional.",
    )

    # ── JARVIS Integration ────────────────────────────────────────────────────
    JARVIS_WEBHOOK_URL: Optional[str] = Field(
        default=None,
        description="JARVIS monitoring webhook URL. Optional — for NOTIFY events.",
    )

    # ── Model Routing (JARVIS Smart Model Router) ─────────────────────────────
    TIER_A_MODEL: str = Field(
        default="anthropic/claude-sonnet-4-6",
        description="Tier A: premium reasoning model for complex extraction tasks.",
    )
    TIER_B_MODEL: str = Field(
        default="openai/gpt-4o-mini",
        description="Tier B: mid-tier model for structured writing and analysis.",
    )
    TIER_C_MODEL: str = Field(
        default="anthropic/claude-haiku-4-5",
        description="Tier C: fast/cheap model for classification and validation.",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = Field(
        default=8000,
        description="HTTP server port. Railway injects $PORT automatically.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings singleton.

    Cached so environment is read only once per process lifetime.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()

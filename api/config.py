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
    TAVILY_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Tavily AI search API key. PRIMARY search engine — works reliably from Railway "
            "datacenter IPs. Free: 1000 searches/month. Designed for AI agents."
        ),
    )
    SERPER_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Serper.dev API key. SECONDARY search engine — Google results. "
            "Cost: $1/1000 queries. Free quota: 2500 credits."
        ),
    )
    BRAVE_API_KEY: Optional[str] = Field(
        default=None,
        description="Brave Search API key. Free 2000/month.",
    )
    SEARXNG_URL: Optional[str] = Field(
        default="",
        description=(
            "Base URL of self-hosted SearXNG instance. Optional. "
            "WARNING: must be publicly accessible — Railway internal URLs "
            "(searxng.railway.internal) only work within the SAME Railway project."
        ),
    )

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="Supabase Postgres pooler connection string (pgbouncer).",
    )

    # ── Admin ─────────────────────────────────────────────────────────────────
    ADMIN_SECRET: Optional[str] = Field(
        default=None,
        description="Secret token for owner admin access. Bypasses trial limits and Stripe.",
    )
    DEMO_SECRET: Optional[str] = Field(
        default=None,
        description=(
            "Demo secret for jares-ai.com domain visitors. "
            "Grants full unblurred 20-lead access + JSON download. "
            "Embedded in frontend JS — separate from ADMIN_SECRET (no admin dashboard access)."
        ),
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

    # ── Contact Data APIs (premium enrichment) ───────────────────────────────
    GOOGLE_PLACES_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Google Places API key. Enables Tier 1 contact extraction: "
            "verified phone numbers and addresses from Google Maps. "
            "Highly recommended for near-100% phone find rate on hotels. "
            "Free tier: $200/month credit (~40,000 Place Detail calls)."
        ),
    )
    HUNTER_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Hunter.io API key. Enables Tier 2 email extraction: "
            "domain-based email finder with ~90% accuracy. "
            "Free tier: 50 searches/month. Paid: from $34/month."
        ),
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

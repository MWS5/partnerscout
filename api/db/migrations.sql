-- PartnerScout AI — Database Schema
-- Run once on Supabase Postgres to initialize all tables.
-- Safe to re-run: uses IF NOT EXISTS.

-- ── Orders Table ──────────────────────────────────────────────────────────────
-- Tracks each lead generation request from creation to completion.

CREATE TABLE IF NOT EXISTS ps_orders (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email             TEXT NOT NULL,
    niches            TEXT[] NOT NULL,
    regions           TEXT[] NOT NULL,
    segment           TEXT DEFAULT 'luxury',
    count_target      INT DEFAULT 100,
    is_trial          BOOL DEFAULT FALSE,
    status            TEXT DEFAULT 'pending',          -- pending | running | done | failed
    progress          INT DEFAULT 0,                   -- 0-100 percent
    stripe_payment_id TEXT,
    result_url        TEXT,
    error_msg         TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);

-- Constraint: status must be one of defined lifecycle states
ALTER TABLE ps_orders
    DROP CONSTRAINT IF EXISTS ps_orders_status_check;

ALTER TABLE ps_orders
    ADD CONSTRAINT ps_orders_status_check
    CHECK (status IN ('pending', 'running', 'done', 'failed'));

-- ── Results Table ─────────────────────────────────────────────────────────────
-- Stores enriched company records, one row per company per order.

CREATE TABLE IF NOT EXISTS ps_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        UUID REFERENCES ps_orders(id) ON DELETE CASCADE,
    category        TEXT NOT NULL,
    company_name    TEXT NOT NULL,
    website         TEXT,
    address         TEXT,
    phone           TEXT,
    email           TEXT,
    contact_person  TEXT,
    personal_phone  TEXT,
    personal_email  TEXT,
    luxury_score    FLOAT DEFAULT 0.0,
    verified        BOOL DEFAULT FALSE,
    raw_data        JSONB,                              -- full pipeline metadata
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Search Cache Table ────────────────────────────────────────────────────────
-- Caches raw search results to avoid redundant API calls.
-- TTL is enforced by application logic (cache_key includes date component).

CREATE TABLE IF NOT EXISTS ps_search_cache (
    cache_key   TEXT PRIMARY KEY,
    result_json JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_ps_results_order
    ON ps_results(order_id);

CREATE INDEX IF NOT EXISTS idx_ps_orders_email
    ON ps_orders(email);

CREATE INDEX IF NOT EXISTS idx_ps_orders_status
    ON ps_orders(status);

CREATE INDEX IF NOT EXISTS idx_ps_cache_time
    ON ps_search_cache(created_at DESC);

-- ── Row Level Security ────────────────────────────────────────────────────────
-- Service role bypasses RLS. Enable RLS for future multi-tenant use.

ALTER TABLE ps_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE ps_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ps_search_cache ENABLE ROW LEVEL SECURITY;

-- Policy: service role (backend) has full access
-- (Supabase service_role key bypasses RLS automatically)

-- ── Cleanup Helper ────────────────────────────────────────────────────────────
-- Optional: scheduled cleanup of old cache entries (run via pg_cron or manual)
-- DELETE FROM ps_search_cache WHERE created_at < NOW() - INTERVAL '24 hours';

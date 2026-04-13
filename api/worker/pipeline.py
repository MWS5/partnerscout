"""
PartnerScout Pipeline Orchestrator v2.

Coordinates the full lead generation pipeline:
  query_matrix → search (with exclusions) → filter → dedup → rank →
  official-url-resolve → name-dedup → extract (4-tier) → validate → export

Each step updates order progress in the database.
Errors are caught, logged, and surfaced to the order status.

Usage:
    result = await run_pipeline(order_id, order_data, db_pool, config)
"""

import asyncio
from difflib import SequenceMatcher
from typing import Any, Optional
from uuid import UUID

import httpx
from loguru import logger

from api.config import Settings
from api.db.client import (
    get_order,
    save_results,
    update_order_status,
)
# Note: blur_for_trial is intentionally NOT imported here.
# Blurring happens only at serve time in export.py / orders.py routes.
from api.engine.extractor import extract_batch
from api.engine.places_discovery import discover_companies_via_places
from api.engine.query_matrix import CITIES, generate_queries
from api.engine.ranker import (
    AGGREGATOR_DOMAINS,
    _domain_is_travel_blog,
    _registered_domain,
    _title_is_aggregator,
    bm25_rank,
    clean_company_name,
    deduplicate,
    filter_official_sites,
)
from api.engine.searcher import (
    brave_search,
    duckduckgo_search,
    searxng_search,
    serper_search,
    tavily_search,
)
from api.engine.validator import filter_by_luxury, score_luxury
from api.engine.strict_validator import validate_and_filter

# ── Progress Constants ────────────────────────────────────────────────────────

PROGRESS_START     = 0
PROGRESS_QUERIES   = 5
PROGRESS_DISCOVERY = 20   # Google Places discovery complete
PROGRESS_SEARCH    = 35   # Web search complete
PROGRESS_RANK      = 50
PROGRESS_EXTRACT   = 70
PROGRESS_VALIDATE  = 85
PROGRESS_EXPORT    = 90
PROGRESS_DONE      = 100

# ── Site exclusion string for DDG/Brave queries ───────────────────────────────
# Adding -site:X to the query forces search engines to skip these domains.
# Use sparingly (DDG supports ~5 exclusions per query effectively).

_SEARCH_EXCLUSIONS = (
    "-site:wikipedia.org "
    "-site:booking.com "
    "-site:tripadvisor.com "
    "-site:hotels.com "
    "-site:expedia.com "
    "-site:theluxevoyager.com "
    "-site:spotlist.fr "
    "-site:week-ends-de-reve.com "
    "-site:privateupgrades.com "
    "-site:relaischateaux.com"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _notify_jarvis(webhook_url: str, payload: dict[str, Any]) -> None:
    """
    Send a NOTIFY event to the JARVIS monitoring webhook.

    Args:
        webhook_url: JARVIS webhook endpoint URL.
        payload: JSON payload dict.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(webhook_url, json=payload)
    except Exception as e:
        logger.warning(f"[PIPELINE][_notify_jarvis] Webhook error (non-fatal): {e}")


async def _send_completion_email(
    resend_key: str,
    email: str,
    order_id: str,
    company_count: int,
    is_trial: bool,
) -> None:
    """
    Send result notification email via Resend API.

    Args:
        resend_key: Resend API key.
        email: Recipient email address.
        order_id: Order UUID string.
        company_count: Number of leads found.
        is_trial: True if trial mode.
    """
    subject = (
        "Your PartnerScout preview is ready!"
        if is_trial
        else f"Your {company_count} leads are ready — PartnerScout AI"
    )
    body = (
        f"Your PartnerScout {'trial' if is_trial else 'full'} report is ready.\n\n"
        f"Leads found: {company_count}\n"
        f"Order ID: {order_id}\n\n"
        f"{'View your 10 preview leads and unlock full access at partnerscout.ai' if is_trial else 'Download your full report at partnerscout.ai'}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={
                    "from": "PartnerScout AI <noreply@partnerscout.ai>",
                    "to": [email],
                    "subject": subject,
                    "text": body,
                },
            )
        logger.info(f"[PIPELINE][_send_completion_email] Email sent to {email}")
    except Exception as e:
        logger.error(f"[PIPELINE][_send_completion_email] Email failed: {e}", exc_info=True)


async def _find_official_url(company_name: str, config: Settings) -> str:
    """
    Search for the company's official website URL.

    Used when the ranked URL is suspicious or empty.
    Runs a targeted DDG search and returns the first non-aggregator result.

    Args:
        company_name: Clean company name.
        config: Application settings.

    Returns:
        Official URL string, or empty string if not found.
    """
    # Try multiple query variants to find official domain
    queries = [
        f'"{company_name}" site officiel',
        f'"{company_name}" official website',
        f'{company_name} hotel réservations',
    ]
    for query in queries:
        try:
            results = await duckduckgo_search(query, num=5)
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "")
                if not url:
                    continue
                domain = _registered_domain(url)
                if (
                    domain not in AGGREGATOR_DOMAINS
                    and not _domain_is_travel_blog(url)
                    and not _title_is_aggregator(title)
                ):
                    logger.info(
                        f"[PIPELINE][_find_official_url] '{company_name}' → {url}"
                    )
                    return url
        except Exception as e:
            logger.warning(f"[PIPELINE][_find_official_url] Error: {e}")
            continue
    return ""


async def _run_search_batch(
    queries: list[tuple[str, str]],
    config: Settings,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """
    Run multi-source search in parallel batches.

    Each query is a (query_str, niche) tuple so results are tagged
    with the correct category from the start.

    Source priority:
      1. Serper.dev (PRIMARY) — reliable from Railway datacenter IPs, $1/1K queries
      2. Brave Search (SECONDARY) — free 2000/month, good quality
      3. DuckDuckGo (FALLBACK) — free but may be blocked on cloud IPs
      4. SearXNG (OPTIONAL) — self-hosted, must be publicly accessible

    Queries include site-exclusion suffixes to prevent aggregator
    results at the source level (before filtering).

    Args:
        queries: List of (query_string, niche) tuples.
        config: Application settings.
        batch_size: Number of queries to run in parallel per batch.

    Returns:
        Flat list of all search result dicts, each with '_niche' field.
    """
    all_results: list[dict[str, Any]] = []

    # Log which sources are active (helps diagnose 0-result issues)
    active_sources = []
    if config.TAVILY_API_KEY:
        active_sources.append("tavily")
    if config.SERPER_API_KEY:
        active_sources.append("serper")
    if config.BRAVE_API_KEY:
        active_sources.append("brave")
    active_sources.append("ddg")
    if config.SEARXNG_URL:
        active_sources.append("searxng")
    logger.info(f"[PIPELINE][_run_search_batch] Active sources: {active_sources}")

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        tasks = []
        task_niches: list[str] = []

        for query_str, niche in batch:
            # Add site exclusions to every query
            q_with_excl = f"{query_str} {_SEARCH_EXCLUSIONS}"

            # PRIMARY: Tavily — AI search, works from Railway IPs, 1000 free/month
            if config.TAVILY_API_KEY:
                tasks.append(tavily_search(q_with_excl, config.TAVILY_API_KEY, num=5))
                task_niches.append(niche)

            # SECONDARY: Serper.dev — Google results, works from Railway IPs
            if config.SERPER_API_KEY:
                tasks.append(serper_search(q_with_excl, config.SERPER_API_KEY, num=5))
                task_niches.append(niche)

            # TERTIARY: Brave Search — free quota, good quality
            if config.BRAVE_API_KEY:
                tasks.append(brave_search(q_with_excl, config.BRAVE_API_KEY, num=5))
                task_niches.append(niche)

            # FALLBACK: DuckDuckGo — free but blocked on some cloud IPs
            tasks.append(duckduckgo_search(q_with_excl, num=5))
            task_niches.append(niche)

            # OPTIONAL: SearXNG — only if publicly accessible URL is set
            if config.SEARXNG_URL:
                tasks.append(searxng_search(q_with_excl, config.SEARXNG_URL, num=5))
                task_niches.append(niche)

        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        batch_count = 0
        for result_list, niche in zip(batch_results, task_niches):
            if isinstance(result_list, list):
                for r in result_list:
                    r["_niche"] = niche  # tag each result with its source niche
                all_results.extend(result_list)
                batch_count += len(result_list)

        logger.info(
            f"[PIPELINE][_run_search_batch] Batch {i // batch_size + 1}: "
            f"{len(batch)} queries → {batch_count} results"
        )

    logger.info(
        f"[PIPELINE][_run_search_batch] Total raw results: {len(all_results)} "
        f"from {len(queries)} queries via {active_sources}"
    )
    return all_results


def _deduplicate_by_name(
    companies: list[dict[str, Any]],
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """
    Deduplicate companies by name similarity.

    When two entries have >80% similar names → they are the same company
    from different sources. Keep the entry with a shorter URL (more likely
    the official homepage, not a listing page).

    Args:
        companies: List of company dicts with 'company_name' and 'url'.
        threshold: Similarity threshold (0.0–1.0). Default 0.80.

    Returns:
        Deduplicated list.
    """
    unique: list[dict[str, Any]] = []

    for company in companies:
        name = company.get("company_name", "").lower().strip()
        is_duplicate = False

        for existing in unique:
            existing_name = existing.get("company_name", "").lower().strip()
            similarity = SequenceMatcher(None, name, existing_name).ratio()

            if similarity >= threshold:
                is_duplicate = True
                # If current entry has a shorter URL, it's more likely official
                url_current  = company.get("url", "")
                url_existing = existing.get("url", "")
                if url_current and len(url_current) < len(url_existing or "https://x"):
                    # Replace existing with better URL, keep best other fields
                    existing.update({
                        k: v for k, v in company.items()
                        if v and v != "Not found"
                    })
                break

        if not is_duplicate:
            unique.append(company)

    removed = len(companies) - len(unique)
    if removed:
        logger.info(
            f"[PIPELINE][_deduplicate_by_name] {len(companies)} → {len(unique)} "
            f"(removed {removed} name-duplicates)"
        )
    return unique


def _build_company_from_result(result: dict[str, Any], fallback_niche: str) -> dict[str, Any]:
    """
    Convert a ranked search result into a partial company dict.

    Uses '_niche' tag for correct category assignment.
    Cleans company name by stripping platform suffixes.

    Args:
        result: Ranked search result dict (must be official site, not aggregator).
        fallback_niche: Fallback category if '_niche' not present.

    Returns:
        Partial company dict with clean name, ready for contact extraction.
    """
    category = result.get("_niche") or fallback_niche
    raw_title = result.get("title", "Unknown Company")
    return {
        "category":     category,
        "company_name": clean_company_name(raw_title),
        "url":          result.get("url", ""),
        "snippet":      result.get("snippet", ""),
        "bm25_score":   result.get("bm25_score", 0.0),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(
    order_id: str,
    order_data: dict[str, Any],
    db_pool: Any,
    config: Settings,
    is_trial: bool = False,
) -> dict[str, Any]:
    """
    Run the full PartnerScout lead generation pipeline.

    Steps:
      1.  Update order → running (0%)
      2.  Generate search queries (per niche, with site exclusions)
      3.  Run multi-source search in batches (DDG + Brave + SearXNG)
      4.  Filter official sites (4-layer defense)
      5.  Deduplicate by URL + domain
      6.  BM25 rank results
      7.  Build company list, resolve missing official URLs
      8.  Deduplicate by company name similarity
      9.  Extract contacts — 4-tier parallel (Google Places, Hunter, Jina, DDG)
      10. Score luxury confidence (Haiku)
      11. Filter by luxury score ≥ 0.6
      12. Sort by (category, -luxury_score)
      13. Apply trial blurring or full results
      14. Save results to DB
      15. Mark order done
      16. Send completion email
      17. Notify JARVIS webhook

    Args:
        order_id: Order UUID string.
        order_data: Full order dict from DB.
        db_pool: asyncpg connection pool.
        config: Application settings.
        is_trial: If True, apply trial blurring and truncate to 10.

    Returns:
        Dict with 'companies', 'total_found', 'status' keys.
    """
    niches       = order_data.get("niches", [])
    regions      = order_data.get("regions", [])
    segment      = order_data.get("segment", "luxury")
    count_target = order_data.get("count_target", 100)
    email        = order_data.get("email", "")

    # Optional premium API keys (graceful degradation if not set)
    google_places_key = getattr(config, "GOOGLE_PLACES_API_KEY", "") or ""
    hunter_api_key    = getattr(config, "HUNTER_API_KEY", "") or ""

    try:
        # ── Step 1: Mark as running ───────────────────────────────────────────
        await update_order_status(db_pool, order_id, "running", PROGRESS_START)
        logger.info(
            f"[PIPELINE][run_pipeline] Started order={order_id} trial={is_trial} "
            f"google_places={'✓' if google_places_key else '✗'} "
            f"hunter={'✓' if hunter_api_key else '✗'}"
        )

        # ── Step 2: Generate tagged queries ──────────────────────────────────
        tagged_queries: list[tuple[str, str]] = []
        for niche in (niches or ["general"]):
            niche_queries = generate_queries([niche], regions, segment)
            tagged_queries.extend((q, niche) for q in niche_queries)

        await update_order_status(db_pool, order_id, "running", PROGRESS_QUERIES)
        logger.info(
            f"[PIPELINE] {len(tagged_queries)} tagged queries for niches={niches}"
        )

        # ── Step 2b: Google Places Discovery (PRIMARY — verified official sites) ─
        # Places Text Search returns ONLY active businesses with official websites.
        # No articles, no review sites, no aggregators — just real companies.
        # Results: official URL + phone + address already pre-filled.
        discovery_cities = regions if regions else CITIES
        places_companies: list[dict] = []

        if google_places_key:
            for niche in (niches or []):
                niche_discovered = await discover_companies_via_places(
                    niche=niche,
                    cities=discovery_cities,
                    api_key=google_places_key,
                    max_per_city=5,  # 5 per city × 8 cities = up to 40 per niche
                )
                places_companies.extend(niche_discovered)
            logger.info(
                f"[PIPELINE] Google Places discovery: {len(places_companies)} verified companies "
                f"(phone+address pre-filled, official websites only)"
            )
        else:
            logger.warning("[PIPELINE] GOOGLE_PLACES_API_KEY not set — Places discovery skipped")
        await update_order_status(db_pool, order_id, "running", PROGRESS_DISCOVERY)

        # ── Step 3: Web Search (SUPPLEMENT — fills gaps Places misses) ─────────
        raw_results = await _run_search_batch(tagged_queries, config, batch_size=10)
        await update_order_status(db_pool, order_id, "running", PROGRESS_SEARCH)
        logger.info(f"[PIPELINE] Web search: {len(raw_results)} raw results")

        # ── Step 4: Filter official sites (4-layer defense on web results) ───────
        official_results = filter_official_sites(raw_results)

        # ── Step 5: Deduplicate web results by URL + domain ────────────────────
        unique_results = deduplicate(official_results)

        # ── Step 6: BM25 rank web results ─────────────────────────────────────
        top_query = tagged_queries[0][0] if tagged_queries else " ".join(niches)
        ranked = bm25_rank(
            top_query, unique_results, top_k=min(200, len(unique_results))
        )

        await update_order_status(db_pool, order_id, "running", PROGRESS_RANK)
        logger.info(
            f"[PIPELINE] Web search: {len(raw_results)} raw → {len(official_results)} official "
            f"→ {len(unique_results)} unique → {len(ranked)} ranked"
        )

        # ── Step 7: Build company list from web results ────────────────────────
        fallback      = niches[0] if niches else "general"
        web_companies = [_build_company_from_result(r, fallback) for r in ranked]

        # Resolve missing URLs for web-search companies
        missing_url_companies = [c for c in web_companies if not c.get("url")]
        if missing_url_companies:
            logger.info(
                f"[PIPELINE] Resolving official URLs for {len(missing_url_companies)} web companies"
            )
            url_tasks = [
                _find_official_url(c["company_name"], config)
                for c in missing_url_companies
            ]
            resolved_urls = await asyncio.gather(*url_tasks, return_exceptions=True)
            for company, resolved_url in zip(missing_url_companies, resolved_urls):
                if isinstance(resolved_url, str) and resolved_url:
                    company["url"] = resolved_url

        # ── Step 7b: Merge Places + Web results (Places FIRST, no duplicates) ──
        # Places companies already have verified official sites → go first.
        # Web companies that share a domain with a Places company → excluded.
        from urllib.parse import urlparse as _urlparse

        def _domain_of(c: dict) -> str:
            try:
                netloc = _urlparse(c.get("url", "")).netloc.lower().replace("www.", "")
                parts = netloc.split(".")
                return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
            except Exception:
                return ""

        places_domains = {_domain_of(c) for c in places_companies if _domain_of(c)}
        web_supplement = [c for c in web_companies if _domain_of(c) not in places_domains]

        companies_raw = places_companies + web_supplement
        logger.info(
            f"[PIPELINE] Merged: {len(places_companies)} Places + "
            f"{len(web_supplement)} web supplement = {len(companies_raw)} total"
        )

        # ── Step 8: Deduplicate by company name similarity ─────────────────────
        companies_raw = _deduplicate_by_name(companies_raw, threshold=0.80)

        # ── Step 9: Extract contact data ──────────────────────────────────────
        # Extract 3× the target so strict validation has enough candidates to
        # choose from even after rejecting incomplete/partner contacts.
        extract_limit = 10 if is_trial else min(count_target * 3, len(companies_raw), 60)
        to_extract = companies_raw[:extract_limit]

        enriched = await extract_batch(
            to_extract,
            config.OPENROUTER_API_KEY,
            config.TIER_B_MODEL,
            max_concurrent=5,
            google_places_key=google_places_key,
            hunter_api_key=hunter_api_key,
            db_pool=db_pool,   # for cost logging to jarvis_search_usage
        )
        await update_order_status(db_pool, order_id, "running", PROGRESS_EXTRACT)
        logger.info(f"[PIPELINE] {len(enriched)} companies extracted")

        # ── Step 9b: Strict contact validation ───────────────────────────────
        # 3-layer gate: rules + source trust + LLM sanity (Haiku, ~$0.0001/co)
        # Only companies with official URL + official email (domain match) +
        # valid phone pass. This is the quality gate for the demo.
        if not is_trial:
            enriched = await validate_and_filter(
                enriched,
                openrouter_key=config.OPENROUTER_API_KEY,
                tier_c_model=config.TIER_C_MODEL,
                max_results=count_target,
                min_results=max(10, count_target // 2),
            )
            await update_order_status(db_pool, order_id, "running", PROGRESS_VALIDATE - 5)
            logger.info(
                f"[PIPELINE] {len(enriched)} companies after strict validation "
                f"(target={count_target})"
            )

        # ── Step 10: Score luxury confidence (parallel, max 8 concurrent) ───────
        _score_semaphore = asyncio.Semaphore(8)

        async def _score_one(company: dict[str, Any]) -> dict[str, Any]:
            # Google Places verified companies already have a pre-computed
            # luxury_score based on their Places rating — skip expensive LLM call.
            if company.get("places_verified"):
                logger.debug(
                    f"[PIPELINE][score] '{company.get('company_name')}' "
                    f"→ Places pre-scored {company['luxury_score']:.2f} (LLM skipped)"
                )
                company["verified"] = company["luxury_score"] >= 0.8
                return company

            jina_content = company.get("jina_content", "")
            content = jina_content if jina_content else (
                company.get("snippet", "") + " " + company.get("address", "")
            )
            async with _score_semaphore:
                score = await score_luxury(
                    company_name=company.get("company_name", ""),
                    website_content=content,
                    niche=company.get("category", ""),
                    openrouter_key=config.OPENROUTER_API_KEY,
                    model=config.TIER_C_MODEL,
                )
            company["luxury_score"] = score
            company["verified"]     = score >= 0.8
            return company

        scored: list[dict[str, Any]] = list(
            await asyncio.gather(*[_score_one(c) for c in enriched])
        )

        # ── Step 11: Filter by luxury score ───────────────────────────────────
        qualified   = filter_by_luxury(scored, min_score=0.6)
        total_found = len(qualified)
        await update_order_status(db_pool, order_id, "running", PROGRESS_VALIDATE)
        logger.info(f"[PIPELINE] {total_found} qualified after luxury filter")

        # ── Step 12: Sort by (category, -luxury_score) ────────────────────────
        qualified.sort(
            key=lambda x: (x.get("category", ""), -float(x.get("luxury_score", 0)))
        )

        # ── Step 13: Select companies to save ────────────────────────────────
        # IMPORTANT: Always save RAW (unblurred) data to DB.
        # Blurring happens at serve time (export.py / orders.py), NOT here.
        # Saving blurred data would cause double-blurring when export endpoints
        # call blur_for_trial again on the already-blurred DB records.
        #
        # Demo mode: strict cap 10-20 strictly validated contacts.
        if is_trial:
            final_companies = qualified[:10]   # max 10 for trial, unblurred
        else:
            # count_target is already capped at 20 by model; respect it.
            final_companies = qualified[:min(count_target, 20)]

        # ── Step 14: Save results ─────────────────────────────────────────────
        await save_results(db_pool, order_id, final_companies)
        await update_order_status(db_pool, order_id, "running", PROGRESS_EXPORT)

        # ── Step 15: Mark done ────────────────────────────────────────────────
        result_url = f"/api/v1/export/{order_id}/{'preview' if is_trial else 'csv'}"
        await update_order_status(db_pool, order_id, "done", PROGRESS_DONE, result_url)
        logger.info(
            f"[PIPELINE] Order {order_id} completed. Companies: {len(final_companies)}"
        )

        # ── Step 16: Email notification ───────────────────────────────────────
        if config.RESEND_API_KEY and email:
            await _send_completion_email(
                config.RESEND_API_KEY, email, order_id,
                len(final_companies), is_trial,
            )

        # ── Step 17: JARVIS webhook ───────────────────────────────────────────
        if config.JARVIS_WEBHOOK_URL:
            await _notify_jarvis(config.JARVIS_WEBHOOK_URL, {
                "event":           "partnerscout.order.completed",
                "order_id":        order_id,
                "is_trial":        is_trial,
                "companies_found": len(final_companies),
                "email":           email,
            })

        return {
            "status":      "done",
            "companies":   final_companies,
            "total_found": total_found,
        }

    except Exception as e:
        logger.error(
            f"[PIPELINE][run_pipeline] Fatal error for order={order_id}: {e}",
            exc_info=True,
        )
        await update_order_status(
            db_pool, order_id, "failed", 0, error_msg=str(e)
        )
        return {"status": "failed", "companies": [], "total_found": 0}

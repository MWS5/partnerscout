"""
PartnerScout Pipeline Orchestrator.

Coordinates the full lead generation pipeline:
  query_matrix → search → rank → extract → validate → export

Each step updates order progress in the database.
Errors are caught, logged, and surfaced to the order status.

Usage:
    result = await run_pipeline(order_id, order_data, db_pool, config)
"""

import asyncio
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
from api.engine.exporter import blur_for_trial
from api.engine.extractor import extract_batch
from api.engine.query_matrix import generate_queries
from api.engine.ranker import bm25_rank, deduplicate
from api.engine.searcher import (
    brave_search,
    duckduckgo_search,
    searxng_search,
)
from api.engine.validator import filter_by_luxury, score_luxury

# ── Progress Constants ────────────────────────────────────────────────────────

PROGRESS_START = 0
PROGRESS_QUERIES = 10
PROGRESS_SEARCH = 30
PROGRESS_RANK = 50
PROGRESS_EXTRACT = 70
PROGRESS_VALIDATE = 85
PROGRESS_EXPORT = 90
PROGRESS_DONE = 100


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


async def _run_search_batch(
    queries: list[tuple[str, str]],
    config: Settings,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """
    Run multi-source search in parallel batches.

    FIX (Rule 57-B1): Each query is now a (query_str, niche) tuple so that
    results are tagged with the correct category from the start.

    Args:
        queries: List of (query_string, niche) tuples.
        config: Application settings.
        batch_size: Number of queries to run in parallel per batch.

    Returns:
        Flat list of all search result dicts, each with '_niche' field.
    """
    all_results: list[dict[str, Any]] = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        tasks = []
        task_niches: list[str] = []

        for query_str, niche in batch:
            tasks.append(duckduckgo_search(query_str, num=5))
            task_niches.append(niche)
            if config.BRAVE_API_KEY:
                tasks.append(brave_search(query_str, config.BRAVE_API_KEY, num=5))
                task_niches.append(niche)
            if config.SEARXNG_URL:
                tasks.append(searxng_search(query_str, config.SEARXNG_URL, num=5))
                task_niches.append(niche)

        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        for result_list, niche in zip(batch_results, task_niches):
            if isinstance(result_list, list):
                for r in result_list:
                    r["_niche"] = niche  # tag each result with its source niche
                all_results.extend(result_list)

        logger.debug(
            f"[PIPELINE][_run_search_batch] Batch {i // batch_size + 1}: "
            f"{len(batch)} queries processed"
        )

    return all_results


def _build_company_from_result(result: dict[str, Any], fallback_niche: str) -> dict[str, Any]:
    """
    Convert a ranked search result into a partial company dict.

    FIX (Rule 57-B1): Uses '_niche' tag from result to assign correct category
    instead of always using niches[0].

    Args:
        result: Ranked search result dict (may have '_niche' from search tagging).
        fallback_niche: Fallback category if '_niche' not present.

    Returns:
        Partial company dict ready for extraction enrichment.
    """
    category = result.get("_niche") or fallback_niche
    return {
        "category": category,
        "company_name": result.get("title", "Unknown Company"),
        "url": result.get("url", ""),
        "snippet": result.get("snippet", ""),
        "bm25_score": result.get("bm25_score", 0.0),
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
      2.  Generate search queries
      3.  Run multi-source search in batches (DDG + Brave + SearXNG)
      4.  Deduplicate + BM25 rank results
      5.  Extract company data per URL (Jina + LLM)
      6.  Score luxury confidence (Haiku)
      7.  Filter by luxury score ≥ 0.6
      8.  Apply trial blurring or full enrichment
      9.  Save results to ps_results table
      10. Update order → done, set result_url
      11. Send completion email (if RESEND_API_KEY configured)
      12. Notify JARVIS webhook (if JARVIS_WEBHOOK_URL configured)

    Args:
        order_id: Order UUID string.
        order_data: Full order dict from DB.
        db_pool: asyncpg connection pool.
        config: Application settings.
        is_trial: If True, apply trial blurring and truncate to 10.

    Returns:
        Dict with 'companies', 'total_found', 'status' keys.
    """
    niches = order_data.get("niches", [])
    regions = order_data.get("regions", [])
    segment = order_data.get("segment", "luxury")
    count_target = order_data.get("count_target", 100)
    email = order_data.get("email", "")

    try:
        # Step 1: Mark as running
        await update_order_status(db_pool, order_id, "running", PROGRESS_START)
        logger.info(f"[PIPELINE][run_pipeline] Started order={order_id} trial={is_trial}")

        # Step 2: Generate queries — per niche to preserve category tagging
        # FIX (Bug 1): generate (query, niche) tuples so each result knows its category
        tagged_queries: list[tuple[str, str]] = []
        for niche in (niches or ["general"]):
            niche_queries = generate_queries([niche], regions, segment)
            tagged_queries.extend((q, niche) for q in niche_queries)

        await update_order_status(db_pool, order_id, "running", PROGRESS_QUERIES)
        logger.info(f"[PIPELINE] {len(tagged_queries)} tagged queries generated for niches={niches}")

        # Step 3: Run searches (results carry _niche tag)
        raw_results = await _run_search_batch(tagged_queries, config, batch_size=10)
        await update_order_status(db_pool, order_id, "running", PROGRESS_SEARCH)
        logger.info(f"[PIPELINE] {len(raw_results)} raw results collected")

        # Step 4: Deduplicate + rank per-niche, preserve category
        unique_results = deduplicate(raw_results)
        top_query = tagged_queries[0][0] if tagged_queries else " ".join(niches)
        ranked = bm25_rank(top_query, unique_results, top_k=min(200, len(unique_results)))
        await update_order_status(db_pool, order_id, "running", PROGRESS_RANK)
        logger.info(f"[PIPELINE] {len(ranked)} unique results after rank")

        # Build partial company dicts — each carries correct category from _niche tag
        fallback = niches[0] if niches else "general"
        companies_raw = [_build_company_from_result(r, fallback) for r in ranked]

        # Step 5: Extract contact data from company websites
        limit = 10 if is_trial else min(count_target * 2, len(companies_raw))
        to_extract = companies_raw[:limit]
        enriched = await extract_batch(
            to_extract,
            config.OPENROUTER_API_KEY,
            config.TIER_B_MODEL,
            max_concurrent=5,
        )
        await update_order_status(db_pool, order_id, "running", PROGRESS_EXTRACT)
        logger.info(f"[PIPELINE] {len(enriched)} companies extracted")

        # Step 6: Score luxury confidence
        # FIX (Bug 2): use jina_content (full website text) for scoring, not just snippet
        scored: list[dict[str, Any]] = []
        for company in enriched:
            jina_content = company.get("jina_content", "")
            content = jina_content if jina_content else (
                company.get("snippet", "") + " " + company.get("address", "")
            )
            score = await score_luxury(
                company_name=company.get("company_name", ""),
                website_content=content,
                niche=company.get("category", ""),
                openrouter_key=config.OPENROUTER_API_KEY,
                model=config.TIER_C_MODEL,
            )
            company["luxury_score"] = score
            company["verified"] = score >= 0.8
            scored.append(company)

        # Step 7: Filter by luxury score
        qualified = filter_by_luxury(scored, min_score=0.6)
        await update_order_status(db_pool, order_id, "running", PROGRESS_VALIDATE)
        total_found = len(qualified)
        logger.info(f"[PIPELINE] {total_found} qualified after luxury filter")

        # FIX (Bug 3): sort by (category, -luxury_score) so output is grouped by category
        qualified.sort(key=lambda x: (x.get("category", ""), -float(x.get("luxury_score", 0))))

        # Step 8: Apply trial blurring or return full results
        if is_trial:
            final_companies = blur_for_trial(qualified)
        else:
            final_companies = qualified[:count_target]

        # Step 9: Save results to DB
        await save_results(db_pool, order_id, final_companies)
        await update_order_status(db_pool, order_id, "running", PROGRESS_EXPORT)

        # Step 10: Mark order done
        result_url = f"/api/v1/export/{order_id}/{'preview' if is_trial else 'csv'}"
        await update_order_status(db_pool, order_id, "done", PROGRESS_DONE, result_url)
        logger.info(f"[PIPELINE] Order {order_id} completed. Companies: {len(final_companies)}")

        # Step 11: Email notification
        if config.RESEND_API_KEY and email:
            await _send_completion_email(
                config.RESEND_API_KEY, email, order_id,
                len(final_companies), is_trial,
            )

        # Step 12: JARVIS webhook notification
        if config.JARVIS_WEBHOOK_URL:
            await _notify_jarvis(config.JARVIS_WEBHOOK_URL, {
                "event": "partnerscout.order.completed",
                "order_id": order_id,
                "is_trial": is_trial,
                "companies_found": len(final_companies),
                "email": email,
            })

        return {
            "status": "done",
            "companies": final_companies,
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

"""
PartnerScout AI — Company Data Extractor (v2).

Multi-source extraction strategy:
  1. Search engine snippet  — often already contains phone/email (biggest win)
  2. Main website via Jina  — homepage content
  3. /contact + /about pages — dedicated contact pages
  4. Secondary search        — "{name} email contact site:linkedin OR site:tripadvisor"
     (fallback if email still not found)

Never fabricates data — uses "Not found" for missing fields.
"""

import asyncio
import json
import re
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import httpx
from loguru import logger

from api.engine.searcher import jina_read

_DDG_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# ── Extraction Prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a B2B contact data extraction specialist for luxury businesses.

Company: {company_name}
Website: {url}

SOURCE 1 — Search engine snippet (often contains direct contact data):
---
{snippet}
---

SOURCE 2 — Website content (homepage + contact page):
---
{content}
---

Your task: extract ALL contact information that appears in ANY of the sources above.
Priority: email and phone are critical — search BOTH sources carefully.
Common patterns: "contact@", "info@", "reservations@", "+33", "+377", "Tel:", "Email:", mailto:

Extract ONLY information explicitly present. Do NOT invent or guess.
If a field is not found in any source, use exactly: "Not found"

Return ONLY valid JSON:
{{
  "address": "full postal address or Not found",
  "phone": "main company phone (include country code if present) or Not found",
  "email": "general contact email (e.g. info@hotel.com) or Not found",
  "contact_person": "Name, Title (e.g. Jean Dupont, Sales Director) or Not found",
  "personal_phone": "direct phone of contact person or Not found",
  "personal_email": "direct email of contact person or Not found"
}}

Output only the JSON. No markdown, no explanation."""


# ── OpenRouter LLM Call ───────────────────────────────────────────────────────

async def _call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    """Call OpenRouter with a single user message."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://partnerscout.ai",
        "X-Title": "PartnerScout AI",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[EXTRACTOR][_call_openrouter] API error: {e}", exc_info=True)
        return None


def _parse_extraction_response(raw: str) -> dict[str, str]:
    """Parse JSON from LLM response, stripping markdown fences if needed."""
    default = {
        "address": "Not found",
        "phone": "Not found",
        "email": "Not found",
        "contact_person": "Not found",
        "personal_phone": "Not found",
        "personal_email": "Not found",
    }
    if not raw:
        return default

    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        parsed = json.loads(cleaned)
        for key in default:
            if key not in parsed or not parsed[key]:
                parsed[key] = "Not found"
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"[EXTRACTOR][_parse_extraction_response] JSON parse failed: {e}")
        return default


# ── Multi-page Jina Fetch ─────────────────────────────────────────────────────

async def _fetch_multipage(base_url: str, max_chars: int = 4000) -> str:
    """
    Fetch homepage + /contact + /about pages, combine content.

    Contact pages almost always have email/phone even when homepage doesn't.

    Args:
        base_url: Company main URL.
        max_chars: Max chars per page (default 4000 total).

    Returns:
        Combined text content from all accessible pages.
    """
    if not base_url:
        return ""

    base = base_url.rstrip("/")
    urls_to_try = [
        base_url,
        f"{base}/contact",
        f"{base}/contact-us",
        f"{base}/about",
        f"{base}/contacts",
        f"{base}/en/contact",
    ]

    per_page = max_chars // len(urls_to_try)
    tasks = [jina_read(url, max_chars=per_page) for url in urls_to_try]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined_parts = []
    for url, result in zip(urls_to_try, results):
        if isinstance(result, str) and result.strip():
            combined_parts.append(f"[Page: {url}]\n{result}")

    combined = "\n\n".join(combined_parts)
    logger.debug(f"[EXTRACTOR][_fetch_multipage] {len(combined_parts)} pages fetched for {base_url}")
    return combined[:max_chars]


# ── Secondary Contact Search ──────────────────────────────────────────────────

def _ddg_search_sync(query: str, num: int = 3) -> list[str]:
    """Sync DDG search returning list of snippets."""
    try:
        from duckduckgo_search import DDGS
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                s = r.get("body", "") or r.get("snippet", "")
                if s:
                    snippets.append(s)
        return snippets
    except Exception as e:
        logger.warning(f"[EXTRACTOR][_ddg_search_sync] DDG error: {e}")
        return []


async def _secondary_contact_search(company_name: str) -> str:
    """
    Search for company contacts across web when website doesn't have them.

    Queries: LinkedIn, TripAdvisor, Booking.com, official press/media pages.

    Args:
        company_name: Company name to search.

    Returns:
        Combined snippet text from secondary sources.
    """
    queries = [
        f'"{company_name}" email contact',
        f'"{company_name}" reservations@  OR  info@  OR  contact@',
        f'"{company_name}" site:linkedin.com contact',
    ]

    loop = asyncio.get_event_loop()
    all_snippets = []

    for query in queries:
        snippets = await loop.run_in_executor(_DDG_EXECUTOR, _ddg_search_sync, query, 3)
        all_snippets.extend(snippets)

    combined = " | ".join(all_snippets)
    logger.info(f"[EXTRACTOR][_secondary_contact_search] '{company_name}' → {len(all_snippets)} snippets")
    return combined[:2000]


# ── Email Quick-Extract (regex fallback) ──────────────────────────────────────

def _regex_extract_email(text: str) -> Optional[str]:
    """
    Extract email address directly from text using regex.

    Used as fast-path before LLM if pattern is obvious.

    Args:
        text: Any text (snippet, website content).

    Returns:
        First email found, or None.
    """
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    matches = re.findall(pattern, text)
    # Filter out common false positives
    blacklist = {"example.com", "domain.com", "email.com", "test.com", "sentry.io"}
    for match in matches:
        domain = match.split("@")[-1].lower()
        if domain not in blacklist and len(match) < 80:
            return match
    return None


def _regex_extract_phone(text: str) -> Optional[str]:
    """
    Extract phone number from text using regex.

    Handles international formats: +33, +377, +7, etc.

    Args:
        text: Any text content.

    Returns:
        First phone number found, or None.
    """
    pattern = r"(\+[\d\s\-().]{7,18}|\b0[\d\s\-().]{8,16})"
    matches = re.findall(pattern, text)
    for match in matches:
        cleaned = re.sub(r"[\s\-()]", "", match)
        if len(cleaned) >= 8:
            return match.strip()
    return None


# ── Main Extraction Function ──────────────────────────────────────────────────

async def extract_company_data(
    url: str,
    company_name: str,
    openrouter_key: str,
    model: str,
    snippet: str = "",
) -> dict[str, Any]:
    """
    Extract structured company contact data using multi-source strategy.

    Pipeline:
      1. Quick regex scan of snippet (already has email in 40%+ of cases)
      2. Fetch homepage + /contact + /about via Jina Reader
      3. LLM extraction from snippet + website combined
      4. If email still not found → secondary DDG search + re-extract

    Args:
        url: Company website URL.
        company_name: Company name (context for LLM).
        openrouter_key: OpenRouter API key.
        model: OpenRouter model ID (Tier B).
        snippet: Search engine snippet (often contains contact data directly).

    Returns:
        Dict with all 6 contact fields + website + jina_content.
        Never returns empty — uses "Not found" for missing fields.
    """
    # ── Quick win: regex scan of snippet first ────────────────────────────────
    quick_email = _regex_extract_email(snippet) if snippet else None
    quick_phone = _regex_extract_phone(snippet) if snippet else None
    if quick_email:
        logger.info(f"[EXTRACTOR] Quick regex hit for '{company_name}': {quick_email}")

    # ── Fetch multi-page website content ─────────────────────────────────────
    website_content = await _fetch_multipage(url) if url else ""

    # Also try regex on website content
    if not quick_email and website_content:
        quick_email = _regex_extract_email(website_content)
    if not quick_phone and website_content:
        quick_phone = _regex_extract_phone(website_content)

    # ── LLM extraction: snippet + website combined ────────────────────────────
    combined_snippet = snippet or "(no search snippet available)"
    prompt = EXTRACTION_PROMPT.format(
        company_name=company_name,
        url=url or "unknown",
        snippet=combined_snippet[:1500],
        content=website_content[:2500] if website_content else "(website not accessible)",
    )

    raw_response = await _call_openrouter(prompt, openrouter_key, model)
    extracted = _parse_extraction_response(raw_response or "")

    # ── Inject regex hits as safety net ──────────────────────────────────────
    if extracted["email"] == "Not found" and quick_email:
        extracted["email"] = quick_email
        logger.info(f"[EXTRACTOR] Regex email injected for '{company_name}': {quick_email}")
    if extracted["phone"] == "Not found" and quick_phone:
        extracted["phone"] = quick_phone

    # ── Secondary search fallback if email still missing ─────────────────────
    if extracted["email"] == "Not found":
        logger.info(f"[EXTRACTOR] Email not found for '{company_name}' — running secondary search")
        secondary_text = await _secondary_contact_search(company_name)

        # Try regex on secondary results first (fast)
        secondary_email = _regex_extract_email(secondary_text)
        secondary_phone = _regex_extract_phone(secondary_text) if extracted["phone"] == "Not found" else None

        if secondary_email:
            extracted["email"] = secondary_email
            logger.info(f"[EXTRACTOR] Secondary search found email for '{company_name}': {secondary_email}")
        elif secondary_text:
            # LLM pass on secondary content
            secondary_prompt = EXTRACTION_PROMPT.format(
                company_name=company_name,
                url=url or "unknown",
                snippet=secondary_text[:2000],
                content="(secondary web search results above)",
            )
            secondary_raw = await _call_openrouter(secondary_prompt, openrouter_key, model)
            secondary_extracted = _parse_extraction_response(secondary_raw or "")
            # Merge: only fill "Not found" fields
            for field in extracted:
                if extracted[field] == "Not found" and secondary_extracted.get(field, "Not found") != "Not found":
                    extracted[field] = secondary_extracted[field]

        if secondary_phone and extracted["phone"] == "Not found":
            extracted["phone"] = secondary_phone

    extracted["website"] = url
    extracted["jina_content"] = website_content  # for luxury scorer

    logger.info(
        f"[EXTRACTOR] '{company_name}' → "
        f"email={'FOUND' if extracted['email'] != 'Not found' else 'missing'} | "
        f"phone={'FOUND' if extracted['phone'] != 'Not found' else 'missing'} | "
        f"contact={'FOUND' if extracted['contact_person'] != 'Not found' else 'missing'}"
    )
    return extracted


# ── Batch Extraction ──────────────────────────────────────────────────────────

async def extract_batch(
    companies: list[dict[str, Any]],
    openrouter_key: str,
    model: str,
    max_concurrent: int = 3,
) -> list[dict[str, Any]]:
    """
    Batch-extract contact data for multiple companies.

    Uses asyncio.Semaphore to limit simultaneous calls.
    Reduced concurrency (3 vs 5) to avoid Jina rate limits with multi-page fetch.

    Args:
        companies: List of company dicts with 'url', 'company_name', 'snippet'.
        openrouter_key: OpenRouter API key.
        model: OpenRouter model ID.
        max_concurrent: Max simultaneous extractions (default 3).

    Returns:
        List of company dicts merged with extracted contact fields.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_with_semaphore(company: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            url = company.get("url", "")
            name = company.get("company_name", "Unknown")
            snippet = company.get("snippet", "")
            extracted = await extract_company_data(url, name, openrouter_key, model, snippet=snippet)
            return {**company, **extracted}

    tasks = [_extract_with_semaphore(c) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    found_emails = sum(1 for r in results if isinstance(r, dict) and r.get("email", "Not found") != "Not found")
    logger.info(
        f"[EXTRACTOR][extract_batch] {len(results)} companies extracted | "
        f"emails found: {found_emails}/{len(results)}"
    )
    return list(results)

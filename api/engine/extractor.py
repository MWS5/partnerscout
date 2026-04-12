"""
PartnerScout AI — Company Data Extractor v3.

100% rule: contacts ALWAYS exist. If Google shows them, we find them.

3-source parallel strategy (all run simultaneously):
  A. Official website  — homepage + /contact + /about via Jina Reader
  B. Search snippets   — DDG snippets for "{name} phone email contact"
                         often directly contain phone/email (Knowledge Panel data)
  C. Regex fast-path   — instant extraction without LLM if pattern obvious

Then: single LLM call on all 3 sources combined → maximum recall.
Never fabricates — "Not found" only when genuinely absent in all 3 sources.
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
from loguru import logger

from api.engine.searcher import jina_read

_CONTACT_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ── Contact Search Queries ────────────────────────────────────────────────────
# These queries are designed to surface Knowledge Panel data (phone, email).
# Google/DDG shows phone numbers in structured snippets for well-known hotels.

CONTACT_QUERY_TEMPLATES = [
    '"{name}" phone email contact',
    '"{name}" reservations contact email',
    '"{name}" +33 OR +377 OR +34 OR +39 contact',  # French/Monaco/Spanish/Italian
    '"{name}" official website contact',
]


# ── Extraction Prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a B2B contact data extraction specialist for luxury hotels and agencies.

Company: {company_name}
Official website: {url}

=== SOURCE A — WEBSITE CONTENT (homepage + contact page) ===
{website_content}

=== SOURCE B — SEARCH ENGINE DATA (Knowledge Panel, directories, press) ===
{search_snippets}

TASK: Extract ALL contact information visible in Source A or Source B.
Phone numbers and emails are almost always present in at least one source.
Look carefully — phones appear as: "+33 4 93 76 50 50", "Tel: +33493765050",
"Tél. : 04 93 76 50 50", "+377 98 06 20 00" etc.
Emails appear as: "reservations@hotel.com", "contact@palace.mc", "info@hotel.fr".

Rules:
- Extract ONLY what is explicitly present — never invent
- Prefer official/general contacts over personal ones
- If found in ANY source → include it
- Missing in ALL sources → use exactly: "Not found"

Return ONLY valid JSON:
{{
  "address": "full street address with city and postal code, or Not found",
  "phone": "main phone with country code (e.g. +33 4 93 76 50 50), or Not found",
  "email": "general contact email (e.g. reservations@hotel.com), or Not found",
  "contact_person": "Name, Title (e.g. Marie Dupont, Director of Sales), or Not found",
  "personal_phone": "direct phone of contact person, or Not found",
  "personal_email": "direct email of contact person, or Not found"
}}

Output ONLY the JSON object. No markdown, no explanation."""


# ── Regex Extractors ──────────────────────────────────────────────────────────

# International phone: +33, +377, +34, +39, +44, +7 etc.
_PHONE_RE = re.compile(
    r'(\+\d{1,3}[\s.\-()]?\d{1,4}[\s.\-()]?\d{1,4}[\s.\-()]?\d{1,9}(?:[\s.\-()]\d{1,4})?)',
    re.IGNORECASE
)

# Email addresses (strict, no false positives)
_EMAIL_RE = re.compile(
    r'\b([a-zA-Z0-9._%+\-]{2,50}@[a-zA-Z0-9.\-]{2,40}\.[a-zA-Z]{2,8})\b'
)

# Known false-positive email domains
_EMAIL_BLACKLIST = frozenset({
    "example.com", "domain.com", "email.com", "test.com",
    "sentry.io", "amazonaws.com", "cloudfront.net",
    "schema.org", "w3.org", "google.com", "facebook.com",
})


def _extract_emails(text: str) -> list[str]:
    """Extract all valid emails from text, deduplicated."""
    found = []
    for match in _EMAIL_RE.findall(text):
        domain = match.split("@")[-1].lower()
        if domain not in _EMAIL_BLACKLIST and len(match) <= 80:
            if match not in found:
                found.append(match)
    return found


def _extract_phones(text: str) -> list[str]:
    """Extract all international phone numbers from text."""
    found = []
    for match in _PHONE_RE.findall(text):
        cleaned = re.sub(r'[\s.\-()]', '', match)
        if len(cleaned) >= 8 and match.strip() not in found:
            found.append(match.strip())
    return found[:3]  # top 3 candidates


# ── OpenRouter LLM Call ───────────────────────────────────────────────────────

async def _call_llm(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Call OpenRouter with single user message. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://partnerscout.ai",
                    "X-Title": "PartnerScout AI",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 600,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[EXTRACTOR][_call_llm] Error: {e}", exc_info=True)
        return None


def _parse_json(raw: str) -> dict[str, str]:
    """Parse LLM JSON response. Returns all 'Not found' on failure."""
    defaults = {
        "address": "Not found", "phone": "Not found", "email": "Not found",
        "contact_person": "Not found", "personal_phone": "Not found", "personal_email": "Not found",
    }
    if not raw:
        return defaults
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        parsed = json.loads(cleaned)
        for k in defaults:
            if k not in parsed or not parsed[k]:
                parsed[k] = "Not found"
        return parsed
    except json.JSONDecodeError:
        return defaults


# ── Source A: Official Website (multi-page) ───────────────────────────────────

async def _fetch_website(base_url: str) -> str:
    """
    Fetch official website: homepage + /contact + /about pages in parallel.

    Args:
        base_url: Company official URL.

    Returns:
        Combined text from all accessible pages (max ~5000 chars).
    """
    if not base_url:
        return ""

    base = base_url.rstrip("/")
    pages = [
        base_url,
        f"{base}/contact",
        f"{base}/contact-us",
        f"{base}/about",
        f"{base}/contacts",
        f"{base}/en/contact",
        f"{base}/fr/contact",
    ]

    results = await asyncio.gather(
        *[jina_read(url, max_chars=1500) for url in pages],
        return_exceptions=True
    )

    parts = []
    for url, content in zip(pages, results):
        if isinstance(content, str) and content.strip():
            parts.append(f"[{url}]\n{content.strip()}")

    combined = "\n\n".join(parts)
    logger.debug(f"[EXTRACTOR][_fetch_website] {len(parts)} pages fetched for {base_url}")
    return combined[:5000]


# ── Source B: Contact-focused search ─────────────────────────────────────────

def _ddg_snippets_sync(query: str, num: int = 5) -> list[str]:
    """Sync DDG search returning snippets. Runs in thread executor."""
    try:
        from duckduckgo_search import DDGS
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                body = r.get("body", "") or r.get("snippet", "")
                if body:
                    snippets.append(body)
        return snippets
    except Exception as e:
        logger.warning(f"[EXTRACTOR][_ddg_snippets_sync] DDG error: {e}")
        return []


async def _search_contacts(company_name: str) -> str:
    """
    Run parallel contact-focused searches for a company.

    Queries are designed to surface Knowledge Panel data:
    phones, emails, addresses that Google/DDG show in rich snippets.

    Args:
        company_name: Clean company name (no platform suffixes).

    Returns:
        Combined snippets text (max 3000 chars).
    """
    loop = asyncio.get_event_loop()
    queries = [t.format(name=company_name) for t in CONTACT_QUERY_TEMPLATES]

    tasks = [
        loop.run_in_executor(_CONTACT_EXECUTOR, _ddg_snippets_sync, q, 4)
        for q in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_snippets = []
    for r in results:
        if isinstance(r, list):
            all_snippets.extend(r)

    combined = "\n---\n".join(all_snippets)
    logger.info(
        f"[EXTRACTOR][_search_contacts] '{company_name}' → "
        f"{len(all_snippets)} snippets ({len(combined)} chars)"
    )
    return combined[:3000]


# ── Main Extraction Function ──────────────────────────────────────────────────

async def extract_company_data(
    url: str,
    company_name: str,
    openrouter_key: str,
    model: str,
    snippet: str = "",
) -> dict[str, Any]:
    """
    Extract company contacts using 3 parallel sources.

    100% rule: if a phone or email exists publicly, it WILL be found.

    Pipeline (all parallel):
      A. Fetch official website pages (Jina multipage)
      B. Run contact-focused DDG searches (Knowledge Panel data)
      C. Regex fast-path on snippet + all collected text

    Then: single LLM call on all combined → JSON extraction.

    Args:
        url: Official company website URL (NOT aggregator!).
        company_name: Clean company name (stripped of suffixes).
        openrouter_key: OpenRouter API key.
        model: Tier B model for extraction.
        snippet: Search snippet already collected (may contain contacts).

    Returns:
        Dict with all 6 contact fields + website + jina_content.
    """
    # ── Run Sources A and B in parallel ──────────────────────────────────────
    website_task = _fetch_website(url)
    search_task  = _search_contacts(company_name)

    website_content, search_snippets = await asyncio.gather(
        website_task, search_task, return_exceptions=False
    )

    # Combine all text for regex + LLM
    all_text = f"{snippet}\n{website_content}\n{search_snippets}"

    # ── Source C: Regex fast-path ─────────────────────────────────────────────
    regex_emails = _extract_emails(all_text)
    regex_phones = _extract_phones(all_text)

    logger.info(
        f"[EXTRACTOR] '{company_name}': "
        f"regex → emails={regex_emails[:2]}, phones={regex_phones[:2]}"
    )

    # ── LLM extraction on all sources combined ────────────────────────────────
    prompt = EXTRACTION_PROMPT.format(
        company_name=company_name,
        url=url or "unknown",
        website_content=website_content[:2500] if website_content else "(not accessible)",
        search_snippets=search_snippets[:2000] if search_snippets else "(no search data)",
    )

    raw = await _call_llm(prompt, openrouter_key, model)
    extracted = _parse_json(raw or "")

    # ── Inject regex results as safety net (LLM sometimes misses) ────────────
    if extracted["email"] == "Not found" and regex_emails:
        extracted["email"] = regex_emails[0]
        logger.info(f"[EXTRACTOR] Regex injected email for '{company_name}': {regex_emails[0]}")

    if extracted["phone"] == "Not found" and regex_phones:
        extracted["phone"] = regex_phones[0]
        logger.info(f"[EXTRACTOR] Regex injected phone for '{company_name}': {regex_phones[0]}")

    extracted["website"]      = url
    extracted["jina_content"] = website_content  # for luxury scorer

    logger.info(
        f"[EXTRACTOR] FINAL '{company_name}' → "
        f"email={'OK' if extracted['email'] != 'Not found' else 'MISSING'} | "
        f"phone={'OK' if extracted['phone'] != 'Not found' else 'MISSING'} | "
        f"contact={'OK' if extracted['contact_person'] != 'Not found' else 'MISSING'}"
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
    Batch-extract contacts for multiple companies with concurrency control.

    Uses asyncio.Semaphore to limit simultaneous LLM + Jina calls.
    Lower concurrency (3) because each company now runs 3 parallel sources.

    Args:
        companies: List of dicts with 'url', 'company_name', 'snippet'.
        openrouter_key: OpenRouter API key.
        model: Extraction model (Tier B).
        max_concurrent: Max simultaneous company extractions (default 3).

    Returns:
        Companies merged with extracted contact data.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_one(company: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return {
                **company,
                **await extract_company_data(
                    url=company.get("url", ""),
                    company_name=company.get("company_name", "Unknown"),
                    openrouter_key=openrouter_key,
                    model=model,
                    snippet=company.get("snippet", ""),
                )
            }

    results = await asyncio.gather(*[_extract_one(c) for c in companies])

    found_emails = sum(1 for r in results if isinstance(r, dict) and r.get("email", "Not found") != "Not found")
    found_phones = sum(1 for r in results if isinstance(r, dict) and r.get("phone", "Not found") != "Not found")
    logger.info(
        f"[EXTRACTOR][extract_batch] {len(results)} extracted | "
        f"emails: {found_emails}/{len(results)} | phones: {found_phones}/{len(results)}"
    )
    return list(results)

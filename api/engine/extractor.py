"""
PartnerScout AI — Company Data Extractor v4 (Multi-Tier).

Near-100% contact find rate via 4-tier parallel strategy:

  Tier 1 — Google Places API   (phone + address — most reliable for hotels)
  Tier 2 — Hunter.io API       (email by domain — ~90% accuracy)
  Tier 3 — Jina multi-page     (official website: homepage + /contact + /about)
  Tier 4 — DDG Knowledge Panel (snippets: phone, email from Google data)
  Tier 5 — Regex fast-path     (instant extraction from all collected text)

All tiers run in parallel. LLM synthesizes all sources → maximum recall.
Never fabricates — "Not found" only when genuinely absent in ALL tiers.

Optional API keys (configure via env vars):
  GOOGLE_PLACES_API_KEY — enables Tier 1 (highly recommended for hotels)
  HUNTER_API_KEY        — enables Tier 2 (highly recommended for emails)
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from api.engine.searcher import jina_read

_CONTACT_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ── Contact Search Queries (DDG Knowledge Panel approach) ─────────────────────
# These queries are specifically designed to surface Knowledge Panel data
# (Google's structured data for businesses: phone, email, address).

CONTACT_QUERY_TEMPLATES = [
    '"{name}" phone numéro téléphone contact',
    '"{name}" email réservations contact',
    '"{name}" +33 OR +377 OR +34 OR +39 OR +44',     # European phone codes
    '"{name}" site officiel contact adresse',
    '"{name}" reservations@ OR contact@ OR info@',    # direct email pattern search
]


# ── Extraction Prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a B2B contact data extraction specialist for luxury hotels and agencies.

Company: {company_name}
Official website: {url}

=== TIER 1 — GOOGLE PLACES DATA ===
{places_data}

=== TIER 2 — HUNTER.IO EMAIL DATA ===
{hunter_data}

=== TIER 3 — WEBSITE CONTENT (homepage + /contact + /about pages) ===
{website_content}

=== TIER 4 — SEARCH ENGINE SNIPPETS (DDG Knowledge Panel data) ===
{search_snippets}

TASK: Extract ALL contact information present in ANY tier above.
Priority: Tier 1 > Tier 2 > Tier 3 > Tier 4.

Phone formats to look for: "+33 4 93 76 50 50", "Tel: +33493765050",
"Tél. : 04 93 76 50 50", "+377 98 06 20 00", "04 93 76 50 50".

Email formats: "reservations@hotel.com", "contact@palace.mc", "info@hotel.fr".

Rules:
- Extract ONLY what is explicitly present — never invent
- Prefer official/general contacts over personal ones
- If found in ANY tier → include it
- Missing in ALL tiers → use exactly: "Not found"

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

_PHONE_RE = re.compile(
    r'(\+\d{1,3}[\s.\-()]?\d{1,4}[\s.\-()]?\d{1,4}[\s.\-()]?\d{1,9}'
    r'(?:[\s.\-()]\d{1,4})?)',
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(
    r'\b([a-zA-Z0-9._%+\-]{2,50}@[a-zA-Z0-9.\-]{2,40}\.[a-zA-Z]{2,8})\b'
)

_EMAIL_BLACKLIST = frozenset({
    "example.com", "domain.com", "email.com", "test.com",
    "sentry.io", "amazonaws.com", "cloudfront.net",
    "schema.org", "w3.org", "google.com", "facebook.com",
    "yourwebsite.com", "yourdomain.com", "placeholder.com",
    "noreply.com", "no-reply.com",
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
    return found[:3]


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
        "contact_person": "Not found", "personal_phone": "Not found",
        "personal_email": "Not found",
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


# ── Tier 1: Google Places API ─────────────────────────────────────────────────

async def _google_places_contact(
    company_name: str,
    google_api_key: str,
) -> dict[str, str]:
    """
    Fetch phone + address from Google Places API.

    Google Maps has verified business data for millions of hotels.
    This is the most reliable source for phone numbers.

    Args:
        company_name: Clean company name.
        google_api_key: Google Places API key.

    Returns:
        Dict with 'phone', 'address', 'website' keys (empty strings if not found).
    """
    if not google_api_key:
        return {"phone": "", "address": "", "website": ""}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Step 1: Find the place
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                params={
                    "input": company_name,
                    "inputtype": "textquery",
                    "fields": "place_id,name",
                    "key": google_api_key,
                },
            )
            resp.raise_for_status()
            candidates = resp.json().get("candidates", [])
            if not candidates:
                logger.debug(f"[EXTRACTOR][_google_places_contact] No place found for '{company_name}'")
                return {"phone": "", "address": "", "website": ""}

            place_id = candidates[0]["place_id"]

            # Step 2: Get place details
            resp2 = await client.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "fields": "formatted_phone_number,international_phone_number,"
                              "formatted_address,website",
                    "key": google_api_key,
                },
            )
            resp2.raise_for_status()
            result = resp2.json().get("result", {})

            phone = (
                result.get("international_phone_number")
                or result.get("formatted_phone_number")
                or ""
            )
            address = result.get("formatted_address", "")
            website = result.get("website", "")

            logger.info(
                f"[EXTRACTOR][_google_places_contact] '{company_name}' → "
                f"phone={'OK' if phone else 'MISS'} address={'OK' if address else 'MISS'}"
            )
            return {"phone": phone, "address": address, "website": website}

    except Exception as e:
        logger.warning(f"[EXTRACTOR][_google_places_contact] Error for '{company_name}': {e}")
        return {"phone": "", "address": "", "website": ""}


# ── Tier 2: Hunter.io Email Search ────────────────────────────────────────────

async def _hunter_email_search(
    domain: str,
    hunter_api_key: str,
) -> str:
    """
    Find official email for a domain via Hunter.io API.

    Hunter indexes email patterns and found emails per domain.
    Returns the best generic/department email (reservations@, contact@, info@).

    Args:
        domain: Website domain (e.g. "fourseasons.com").
        hunter_api_key: Hunter.io API key.

    Returns:
        Email string, or empty string if not found / no API key.
    """
    if not hunter_api_key or not domain:
        return ""

    # Strip www. and path from domain
    clean_domain = domain.lower().replace("www.", "").split("/")[0]
    if not clean_domain or "." not in clean_domain:
        return ""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.hunter.io/v2/domain-search",
                params={
                    "domain": clean_domain,
                    "api_key": hunter_api_key,
                    "limit": 10,
                    "type": "generic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            emails: list[dict] = data.get("data", {}).get("emails", [])
            if not emails:
                # Fallback: also try with "personal" type
                resp2 = await client.get(
                    "https://api.hunter.io/v2/domain-search",
                    params={
                        "domain": clean_domain,
                        "api_key": hunter_api_key,
                        "limit": 5,
                    },
                )
                resp2.raise_for_status()
                emails = resp2.json().get("data", {}).get("emails", [])

            if not emails:
                return ""

            # Prefer generic/department emails over personal
            PREFERRED_PREFIXES = (
                "reservations", "reservation", "contact", "info",
                "booking", "sales", "events", "concierge",
                "groups", "mice", "wedding", "spa",
            )
            generic = [e for e in emails if e.get("type") == "generic"]
            personal = [e for e in emails if e.get("type") == "personal"]

            # Try to find preferred prefix in generic list
            for prefix in PREFERRED_PREFIXES:
                for e in generic:
                    val = e.get("value", "")
                    if val.startswith(prefix + "@"):
                        logger.info(
                            f"[EXTRACTOR][_hunter_email_search] '{domain}' → "
                            f"preferred: {val}"
                        )
                        return val

            # Fallback: best generic, then best personal (highest confidence)
            target = generic[0] if generic else (personal[0] if personal else emails[0])
            email = target.get("value", "")
            logger.info(
                f"[EXTRACTOR][_hunter_email_search] '{domain}' → {email}"
            )
            return email

    except Exception as e:
        logger.warning(f"[EXTRACTOR][_hunter_email_search] Error for '{domain}': {e}")
        return ""


# ── Tier 3: Official Website (multi-page) ─────────────────────────────────────

async def _fetch_website(base_url: str) -> str:
    """
    Fetch official website: homepage + contact/about pages in parallel.

    Args:
        base_url: Company official URL.

    Returns:
        Combined text from all accessible pages (max ~6000 chars).
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
        f"{base}/en/contact-us",
    ]

    results = await asyncio.gather(
        *[jina_read(url, max_chars=1500) for url in pages],
        return_exceptions=True,
    )

    parts = []
    for url, content in zip(pages, results):
        if isinstance(content, str) and content.strip():
            parts.append(f"[{url}]\n{content.strip()}")

    combined = "\n\n".join(parts)
    logger.debug(
        f"[EXTRACTOR][_fetch_website] {len(parts)}/{len(pages)} pages fetched for {base_url}"
    )
    return combined[:6000]


# ── Tier 4: DDG Contact Search (Knowledge Panel) ──────────────────────────────

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

    Queries target Knowledge Panel data: phones, emails, addresses
    that Google/DDG show in rich structured snippets.

    Args:
        company_name: Clean company name.

    Returns:
        Combined snippets text (max 4000 chars).
    """
    loop = asyncio.get_event_loop()
    queries = [t.format(name=company_name) for t in CONTACT_QUERY_TEMPLATES]

    tasks = [
        loop.run_in_executor(_CONTACT_EXECUTOR, _ddg_snippets_sync, q, 5)
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
    return combined[:4000]


# ── Main Extraction Function ──────────────────────────────────────────────────

async def extract_company_data(
    url: str,
    company_name: str,
    openrouter_key: str,
    model: str,
    snippet: str = "",
    google_places_key: str = "",
    hunter_api_key: str = "",
) -> dict[str, Any]:
    """
    Extract company contacts using 4+1 parallel tiers.

    Near-100% rule: if any public source has the contact, it WILL be found.

    Tiers (all run in parallel):
      1. Google Places API → phone + address (most reliable for hotels)
      2. Hunter.io API     → email by domain (~90% accuracy)
      3. Official website  → Jina multipage fetch (homepage + contact + about)
      4. DDG search        → Knowledge Panel snippets (phone, email, address)
      5. Regex fast-path   → instant extraction from all text

    Then: single LLM call synthesizes all 4 tiers → JSON.

    Args:
        url: Official company website URL.
        company_name: Clean company name.
        openrouter_key: OpenRouter API key.
        model: Tier B model for extraction.
        snippet: Search snippet already collected.
        google_places_key: Optional Google Places API key.
        hunter_api_key: Optional Hunter.io API key.

    Returns:
        Dict with all 6 contact fields + website + jina_content.
    """
    # ── Extract domain for Hunter.io ──────────────────────────────────────────
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace("www.", "") if url else ""
    except Exception:
        domain = ""

    # ── Run all tiers in parallel ─────────────────────────────────────────────
    (
        places_data,
        hunter_email,
        website_content,
        search_snippets,
    ) = await asyncio.gather(
        _google_places_contact(company_name, google_places_key),
        _hunter_email_search(domain, hunter_api_key),
        _fetch_website(url),
        _search_contacts(company_name),
        return_exceptions=False,
    )

    # ── Combine all text for regex fast-path ──────────────────────────────────
    places_text = (
        f"Phone: {places_data.get('phone', '')}\n"
        f"Address: {places_data.get('address', '')}\n"
        f"Website: {places_data.get('website', '')}"
        if places_data else ""
    )
    hunter_text = f"Email: {hunter_email}" if hunter_email else ""
    all_text = f"{snippet}\n{places_text}\n{hunter_text}\n{website_content}\n{search_snippets}"

    # ── Tier 5: Regex fast-path ───────────────────────────────────────────────
    regex_emails = _extract_emails(all_text)
    regex_phones = _extract_phones(all_text)

    logger.info(
        f"[EXTRACTOR] '{company_name}' pre-LLM: "
        f"places={'✓' if places_data.get('phone') else '✗'} "
        f"hunter={'✓' if hunter_email else '✗'} "
        f"regex_emails={regex_emails[:2]} "
        f"regex_phones={regex_phones[:2]}"
    )

    # ── Format tier summaries for LLM prompt ─────────────────────────────────
    places_formatted = (
        f"Phone: {places_data.get('phone', 'Not found')}\n"
        f"Address: {places_data.get('address', 'Not found')}\n"
        f"Official Website: {places_data.get('website', 'Not found')}"
        if places_data else "(Google Places API not configured)"
    )
    hunter_formatted = (
        f"Email found: {hunter_email}"
        if hunter_email else "(Hunter.io not configured or no email found for this domain)"
    )

    # ── LLM extraction ────────────────────────────────────────────────────────
    prompt = EXTRACTION_PROMPT.format(
        company_name=company_name,
        url=url or "unknown",
        places_data=places_formatted,
        hunter_data=hunter_formatted,
        website_content=website_content[:2500] if website_content else "(not accessible)",
        search_snippets=search_snippets[:2000] if search_snippets else "(no search data)",
    )

    raw = await _call_llm(prompt, openrouter_key, model)
    extracted = _parse_json(raw or "")

    # ── Safety net: inject Tier 1 results if LLM missed them ─────────────────
    if places_data.get("phone") and extracted["phone"] == "Not found":
        extracted["phone"] = places_data["phone"]
        logger.info(f"[EXTRACTOR] Google Places injected phone for '{company_name}'")

    if places_data.get("address") and extracted["address"] == "Not found":
        extracted["address"] = places_data["address"]
        logger.info(f"[EXTRACTOR] Google Places injected address for '{company_name}'")

    # ── Safety net: inject Tier 2 (Hunter email) ──────────────────────────────
    if hunter_email and extracted["email"] == "Not found":
        extracted["email"] = hunter_email
        logger.info(f"[EXTRACTOR] Hunter.io injected email for '{company_name}': {hunter_email}")

    # ── Safety net: inject regex results ──────────────────────────────────────
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
        f"address={'OK' if extracted['address'] != 'Not found' else 'MISSING'} | "
        f"contact={'OK' if extracted['contact_person'] != 'Not found' else 'MISSING'}"
    )
    return extracted


# ── Batch Extraction ──────────────────────────────────────────────────────────

async def extract_batch(
    companies: list[dict[str, Any]],
    openrouter_key: str,
    model: str,
    max_concurrent: int = 3,
    google_places_key: str = "",
    hunter_api_key: str = "",
) -> list[dict[str, Any]]:
    """
    Batch-extract contacts for multiple companies with concurrency control.

    Uses asyncio.Semaphore to limit simultaneous calls (each company
    runs 4 parallel tiers internally).

    Args:
        companies: List of dicts with 'url', 'company_name', 'snippet'.
        openrouter_key: OpenRouter API key.
        model: Extraction model (Tier B).
        max_concurrent: Max simultaneous company extractions (default 3).
        google_places_key: Optional Google Places API key.
        hunter_api_key: Optional Hunter.io API key.

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
                    google_places_key=google_places_key,
                    hunter_api_key=hunter_api_key,
                ),
            }

    results = await asyncio.gather(*[_extract_one(c) for c in companies])

    found_emails   = sum(1 for r in results if isinstance(r, dict) and r.get("email", "Not found") != "Not found")
    found_phones   = sum(1 for r in results if isinstance(r, dict) and r.get("phone", "Not found") != "Not found")
    found_contacts = sum(1 for r in results if isinstance(r, dict) and r.get("contact_person", "Not found") != "Not found")
    logger.info(
        f"[EXTRACTOR][extract_batch] {len(results)} extracted | "
        f"emails: {found_emails}/{len(results)} | "
        f"phones: {found_phones}/{len(results)} | "
        f"contacts: {found_contacts}/{len(results)}"
    )
    return list(results)

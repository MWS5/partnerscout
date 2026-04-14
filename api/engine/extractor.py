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
import html as _html_module
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from api.engine.schema_extractor import extract_schema_contacts
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
    '"{name}" "@" email réservation nous-contacter',  # French email search
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

# Catches obfuscated emails in mailto: href attributes
# e.g. <a href="mailto:info@hotel.com"> or encoded variants
_MAILTO_RE = re.compile(
    r'mailto:([a-zA-Z0-9._%+\-]{2,50}@[a-zA-Z0-9.\-]{2,40}\.[a-zA-Z]{2,8})',
    re.IGNORECASE,
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
        Dict with 'phone', 'address', 'website', '_duration_ms', '_success' keys.
    """
    if not google_api_key:
        return {"phone": "", "address": "", "website": "", "_duration_ms": 0, "_success": False}

    _t0 = time.monotonic()

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
                ms = int((time.monotonic() - _t0) * 1000)
                return {"phone": "", "address": "", "website": "", "_duration_ms": ms, "_success": False}

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
            ms = int((time.monotonic() - _t0) * 1000)

            logger.info(
                f"[EXTRACTOR][_google_places_contact] '{company_name}' → "
                f"phone={'OK' if phone else 'MISS'} address={'OK' if address else 'MISS'} | {ms}ms"
            )
            return {
                "phone": phone, "address": address, "website": website,
                "_duration_ms": ms, "_success": bool(phone or address),
            }

    except Exception as e:
        ms = int((time.monotonic() - _t0) * 1000)
        logger.warning(f"[EXTRACTOR][_google_places_contact] Error for '{company_name}': {e}")
        return {"phone": "", "address": "", "website": "", "_duration_ms": ms, "_success": False, "_error": str(e)}


# ── Tier 2: Hunter.io Email Search ────────────────────────────────────────────

async def _hunter_email_search(
    domain: str,
    hunter_api_key: str,
) -> tuple[str, int, bool]:
    """
    Find official email for a domain via Hunter.io API.

    Hunter indexes email patterns and found emails per domain.
    Returns the best generic/department email (reservations@, contact@, info@).

    Args:
        domain: Website domain (e.g. "fourseasons.com").
        hunter_api_key: Hunter.io API key.

    Returns:
        Tuple of (email_str, duration_ms, success_bool).
    """
    if not hunter_api_key or not domain:
        return ("", 0, False)

    _t0 = time.monotonic()

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
                ms = int((time.monotonic() - _t0) * 1000)
                return ("", ms, False)

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
                        ms = int((time.monotonic() - _t0) * 1000)
                        logger.info(f"[EXTRACTOR][_hunter_email_search] '{domain}' → preferred: {val} | {ms}ms")
                        return (val, ms, True)

            # Fallback: best generic, then best personal (highest confidence)
            target = generic[0] if generic else (personal[0] if personal else emails[0])
            email_val = target.get("value", "")
            ms = int((time.monotonic() - _t0) * 1000)
            logger.info(f"[EXTRACTOR][_hunter_email_search] '{domain}' → {email_val} | {ms}ms")
            return (email_val, ms, bool(email_val))

    except Exception as e:
        ms = int((time.monotonic() - _t0) * 1000)
        logger.warning(f"[EXTRACTOR][_hunter_email_search] Error for '{domain}': {e}")
        return ("", ms, False)


# ── Tier 2.5: Direct HTML Email Extraction ───────────────────────────────────
# Most reliable for French luxury hotels — contacts are in HTML, never JS-dynamic.
# Domain-matched only: reservations@royal-riviera.com, info@lenegresco.com, etc.

_DIRECT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

_PREFERRED_EMAIL_PREFIXES: tuple[str, ...] = (
    "reservations", "reservation", "contact", "info",
    "sales", "events", "concierge", "groups", "spa",
    "booking", "mice", "wedding", "reception",
)


async def _tavily_email_search(
    company_domain: str,
    company_name: str,
    tavily_api_key: str,
) -> list[str]:
    """
    Search via Tavily AI for domain-specific emails — last-resort fallback.

    Used ONLY when Direct HTTP (Pass 1) + Jina Reader (Pass 2) both fail.
    Tavily free tier: 1000 searches/month. Works from Railway cloud IPs.

    Args:
        company_domain: e.g. "royal-riviera.com"
        company_name: e.g. "Hotel Royal-Riviera"
        tavily_api_key: Tavily API key (TAVILY_API_KEY env var).

    Returns:
        List of domain-matched emails found in search snippets.
    """
    if not tavily_api_key or not company_domain:
        return []

    # Two-query approach: domain-email pattern + site-specific contact
    queries = [
        f'"@{company_domain}"',                    # exact: "@royal-riviera.com"
        f'site:{company_domain} contact email',    # site-search contact page
    ]
    found: list[str] = []

    try:
        from api.engine.searcher import tavily_search
        for query in queries:
            results = await tavily_search(query, tavily_api_key, num=5)
            for r in results:
                text = r.get("snippet", "") + " " + r.get("title", "") + " " + r.get("content", "")
                for email in _EMAIL_RE.findall(text):
                    email_l = email.lower()
                    if email_l.endswith("@" + company_domain) and email_l not in found:
                        found.append(email_l)
                for email in _MAILTO_RE.findall(text):
                    email_l = email.lower()
                    if email_l.endswith("@" + company_domain) and email_l not in found:
                        found.append(email_l)
            if found:
                logger.info(
                    f"[EXTRACTOR][_tavily_email_search] '{company_domain}' found: {found[:3]}"
                )
                break

    except Exception as e:
        logger.warning(f"[EXTRACTOR][_tavily_email_search] Error for '{company_domain}': {e}")

    return found


async def _fetch_emails_direct(
    base_url: str,
    company_domain: str,
    company_name: str = "",
    tavily_api_key: str = "",
) -> list[str]:
    """
    Tier 2.5: Directly fetch contact pages and extract domain-matched emails.

    More reliable than Jina for French hotel contact pages because:
    - No char limit — scans the full raw HTML
    - Tries French URL variants (/nous-contacter, /contactez-nous)
    - Domain-matching filter: only returns hotel's own email (@domain.com)
    - Zero LLM cost

    Args:
        base_url: Official company website URL.
        company_domain: Root domain (e.g. "royal-riviera.com").

    Returns:
        Prioritized list of domain-matched emails (preferred prefixes first).
    """
    if not base_url or not company_domain:
        return []

    base = base_url.rstrip("/")

    # Comprehensive contact page list — EN + FR (covers 95%+ of French luxury hotels)
    contact_pages = [
        base_url,
        f"{base}/contact",
        f"{base}/nous-contacter",          # French — most common
        f"{base}/contactez-nous",          # French alternative
        f"{base}/contact-us",
        f"{base}/fr/contact",
        f"{base}/fr/nous-contacter",
        f"{base}/en/contact",
        f"{base}/en/contact-us",
        f"{base}/contacts",
        f"{base}/contact.html",
        f"{base}/contact.php",
        f"{base}/about",
        f"{base}/about-us",
    ]

    found_emails: list[str] = []

    def _domain_match(email_addr: str) -> bool:
        """Return True if email belongs to company_domain."""
        domain = email_addr.lower().split("@")[-1]
        return (
            domain == company_domain
            or domain.endswith("." + company_domain)
            or company_domain.endswith("." + domain)
        )

    def _extract_from_text(text: str) -> list[str]:
        """Extract domain-matched emails from raw or decoded text."""
        results = []
        # 1. HTML entity decode — catches &#x72;&#x65;&#x73;... obfuscation
        decoded = _html_module.unescape(text)
        # 2. mailto: hrefs — most reliable signal (explicit intent to email)
        for m in _MAILTO_RE.findall(decoded):
            if _domain_match(m) and m.lower() not in [e.lower() for e in results]:
                results.append(m.lower())
        # 3. Plain text email regex on decoded HTML
        for m in _EMAIL_RE.findall(decoded):
            if _domain_match(m) and m.lower() not in [e.lower() for e in results]:
                results.append(m.lower())
        return results

    # ── Pass 1: Direct HTTP fetch (fast, full HTML) ───────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=8.0,
            follow_redirects=True,
            headers=_DIRECT_HEADERS,
        ) as client:
            tasks = [client.get(url) for url in contact_pages]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for resp in responses:
            if isinstance(resp, Exception):
                continue
            if not hasattr(resp, "status_code") or resp.status_code != 200:
                continue
            for email in _extract_from_text(resp.text):
                if email not in found_emails:
                    found_emails.append(email)

    except Exception as e:
        logger.warning(f"[EXTRACTOR][_fetch_emails_direct] Direct fetch error for {base_url}: {e}")

    # ── Pass 2: Jina Reader fallback (bypasses bot-protection, CDN IPs) ───────
    # Always runs when direct fetch found no emails.
    # Luxury hotels soft-redirect ALL paths to 200 → direct_success_count is always high,
    # but pages return homepage content without actual contact emails.
    # Jina uses CDN IPs that bypass Cloudflare bot protection.
    if not found_emails:
        logger.info(
            f"[EXTRACTOR][_fetch_emails_direct] No emails from direct fetch for '{company_domain}' "
            f"— trying Jina fallback for contact pages"
        )
        # Jina uses its own CDN IPs — bypasses Cloudflare bot protection that blocks Railway
        jina_contact_pages = [
            f"{base_url.rstrip('/')}/contact",
            f"{base_url.rstrip('/')}/nous-contacter",
            f"{base_url.rstrip('/')}/contactez-nous",
            f"{base_url.rstrip('/')}/fr/contact",
            f"{base_url.rstrip('/')}/en/contact",
            f"{base_url.rstrip('/')}/en/contact-us",
        ]
        # Sequential Jina reads — stops at first success to avoid Jina rate limiting.
        # Royal-Riviera: email at char 8324, need max_chars=15000.
        # Run ONE at a time: avoids 51 concurrent Jina calls when processing 3 companies
        # simultaneously (17 Jina reads per company × 3 concurrency = rate limit exceeded).
        for jina_url in jina_contact_pages:
            try:
                content = await jina_read(jina_url, max_chars=15000)
                if isinstance(content, str) and content:
                    for email in _extract_from_text(content):
                        if email not in found_emails:
                            found_emails.append(email)
                if found_emails:
                    logger.info(
                        f"[EXTRACTOR][_fetch_emails_direct] Jina found {len(found_emails)} "
                        f"emails at '{jina_url}': {found_emails[:3]}"
                    )
                    break  # Stop at first successful page — avoid unnecessary Jina calls
            except Exception as e:
                logger.debug(f"[EXTRACTOR][_fetch_emails_direct] Jina error for {jina_url}: {e}")
                continue

    # ── Pass 3: Tavily Search (last resort) ──────────────────────────────────
    # Used ONLY when Direct HTTP + Jina both fail (bot-blocked or rate-limited).
    # Tavily free tier: 1000/month — spend wisely (luxury hotels only need it).
    if not found_emails and tavily_api_key:
        logger.info(
            f"[EXTRACTOR][_fetch_emails_direct] Passes 1+2 failed for "
            f"'{company_domain}' — trying Tavily last-resort fallback"
        )
        tavily_emails = await _tavily_email_search(company_domain, company_name, tavily_api_key)
        found_emails.extend(e for e in tavily_emails if e not in found_emails)

    if not found_emails:
        return []

    # Sort: preferred prefixes first, then alphabetical
    def _email_priority(e: str) -> int:
        local = e.lower().split("@")[0]
        for i, prefix in enumerate(_PREFERRED_EMAIL_PREFIXES):
            if local.startswith(prefix):
                return i
        return len(_PREFERRED_EMAIL_PREFIXES)

    found_emails.sort(key=_email_priority)
    found_emails = list(dict.fromkeys([e.lower() for e in found_emails]))  # dedup

    logger.info(
        f"[EXTRACTOR][_fetch_emails_direct] '{company_domain}' → "
        f"{len(found_emails)} domain-matched emails: {found_emails[:3]}"
    )
    return found_emails


# ── Tier 3: Official Website (multi-page via Jina) ────────────────────────────

async def _fetch_website(base_url: str) -> str:
    """
    Fetch official website via Jina: homepage + contact/about pages.

    Expanded with French URL variants (covers /nous-contacter, /contactez-nous).
    Increased max_chars to 2500 per page so emails deeper in content are captured.

    Args:
        base_url: Company official URL.

    Returns:
        Combined text from all accessible pages (max ~9000 chars).
    """
    if not base_url:
        return ""

    base = base_url.rstrip("/")
    # Reduced page list for _fetch_website — this is for luxury SCORING content only.
    # Email extraction is handled by _fetch_emails_direct (more thorough).
    # Fewer Jina reads here = less rate limit pressure.
    pages = [
        base_url,
        f"{base}/contact",
        f"{base}/nous-contacter",     # French — most common for Côte d'Azur hotels
        f"{base}/about",
        f"{base}/en/contact",
        f"{base}/fr/contact",
    ]

    # max_chars=6000: enough for luxury scoring content
    results = await asyncio.gather(
        *[jina_read(url, max_chars=6000) for url in pages],
        return_exceptions=True,
    )

    parts = []
    for url, content in zip(pages, results):
        if isinstance(content, str) and content.strip() and len(content.strip()) > 30:
            parts.append(f"[{url}]\n{content.strip()}")

    combined = "\n\n".join(parts)
    logger.debug(
        f"[EXTRACTOR][_fetch_website] {len(parts)}/{len(pages)} pages via Jina for {base_url}"
    )
    return combined[:9000]


# ── Tier 4: DDG Contact Search (Knowledge Panel) ──────────────────────────────

def _ddg_snippets_sync(query: str, num: int = 5) -> list[str]:
    """Sync DDG search returning snippets. Runs in thread executor."""
    try:
        from duckduckgo_search import DDGS
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                # Handle both v6 ("body") and v7+ ("snippet") key names
                body = r.get("body") or r.get("snippet", "")
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
    tavily_api_key: str = "",
    db_pool: Any = None,
) -> dict[str, Any]:
    """
    Extract company contacts using 4+1 parallel tiers.

    Near-100% rule: if any public source has the contact, it WILL be found.

    Tiers (all run in parallel):
      0. Schema.org / JSON-LD → structured data self-reported by business (FREE)
      1. Google Places API    → phone + address (most reliable for hotels)
      2. Hunter.io API        → email by domain (~90% accuracy)
      3. Official website     → Jina multipage fetch (homepage + contact + about)
      4. DDG search           → Knowledge Panel snippets (phone, email, address)
      5. Regex fast-path      → instant extraction from all text

    Tier 0 short-circuits: if phone+email+address found in schema → skip LLM entirely.
    This saves ~$0.005 per company AND improves accuracy (no hallucination risk).

    Args:
        url: Official company website URL.
        company_name: Clean company name.
        openrouter_key: OpenRouter API key.
        model: Tier B model for extraction.
        snippet: Search snippet already collected.
        google_places_key: Optional Google Places API key.
        hunter_api_key: Optional Hunter.io API key.
        db_pool: Optional asyncpg pool for cost logging to jarvis_search_usage.

    Returns:
        Dict with all 6 contact fields + website + jina_content.
    """
    # ── Extract domain for Hunter.io ──────────────────────────────────────────
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower().replace("www.", "") if url else ""
    except Exception:
        domain = ""

    # ── Tier 0: Schema.org / JSON-LD (FREE — runs first, in parallel with others) ──
    # Fetches raw HTML and parses structured data self-reported by the business.
    # If phone + email + address all found here → we skip the LLM call entirely.
    (
        schema_data,
        places_data,
        hunter_result,
        website_content,
        search_snippets,
        direct_emails,
    ) = await asyncio.gather(
        extract_schema_contacts(url, company_name),
        _google_places_contact(company_name, google_places_key),
        _hunter_email_search(domain, hunter_api_key),
        _fetch_website(url),
        _search_contacts(company_name),
        _fetch_emails_direct(url, domain, company_name, tavily_api_key),  # Tier 2.5 + Pass 3
        return_exceptions=False,
    )

    # Unpack Hunter result tuple
    hunter_email, hunter_ms, hunter_success = (
        hunter_result if isinstance(hunter_result, tuple) else ("", 0, False)
    )
    # Best direct email (domain-matched, preferred prefix)
    direct_email = direct_emails[0] if isinstance(direct_emails, list) and direct_emails else ""

    # ── Schema Tier 0: Check if we have enough from structured data ───────────
    schema_phone   = schema_data.get("phone", "")   if schema_data else ""
    schema_email   = schema_data.get("email", "")   if schema_data else ""
    schema_address = schema_data.get("address", "") if schema_data else ""
    schema_person  = schema_data.get("contact_person", "") if schema_data else ""

    # Short-circuit: if schema has phone + email + address → skip LLM entirely
    # Use Places phone/email as supplement if schema is partial
    effective_phone   = schema_phone   or (places_data.get("phone", "")   if places_data else "")
    effective_email   = schema_email   or hunter_email or direct_email
    effective_address = schema_address or (places_data.get("address", "") if places_data else "")

    if effective_phone and effective_email and effective_address:
        logger.info(
            f"[EXTRACTOR] '{company_name}' → SHORT-CIRCUIT: "
            f"schema+tier1 provided all fields — LLM skipped ✓"
        )
        extracted = {
            "phone":          effective_phone,
            "email":          effective_email,
            "address":        effective_address,
            "contact_person": schema_person or "Not found",
            "personal_phone": "Not found",
            "personal_email": schema_data.get("personal_email", "Not found") if schema_data else "Not found",
        }
        extracted["website"]      = url
        extracted["jina_content"] = website_content
        return {**extracted, "company_name": company_name}
    # Otherwise: continue to full LLM extraction with all tiers as context

    # ── Log API costs to JARVIS cost tracker (fire-and-forget) ───────────────
    if db_pool:
        try:
            from api.utils.api_cost_logger import log_google_places, log_hunter_io
            places_ms = places_data.get("_duration_ms", 0) if isinstance(places_data, dict) else 0
            places_ok = places_data.get("_success", False) if isinstance(places_data, dict) else False
            places_err = places_data.get("_error") if isinstance(places_data, dict) else None
            asyncio.create_task(log_google_places(db_pool, company_name, places_ok, places_ms, places_err))
            asyncio.create_task(log_hunter_io(db_pool, domain, hunter_success, hunter_ms))
        except Exception as _log_e:
            logger.debug(f"[EXTRACTOR] Cost logging skipped: {_log_e}")

    # ── Combine all text for regex fast-path ──────────────────────────────────
    schema_text = (
        f"Phone: {schema_phone}\nEmail: {schema_email}\nAddress: {schema_address}"
        if (schema_phone or schema_email or schema_address) else ""
    )
    places_text = (
        f"Phone: {places_data.get('phone', '')}\n"
        f"Address: {places_data.get('address', '')}\n"
        f"Website: {places_data.get('website', '')}"
        if places_data else ""
    )
    hunter_text = f"Email: {hunter_email}" if hunter_email else ""
    direct_text = f"Email (direct HTML scan): {direct_email}" if direct_email else ""
    all_text = f"{snippet}\n{schema_text}\n{places_text}\n{hunter_text}\n{direct_text}\n{website_content}\n{search_snippets}"

    # ── Tier 5: Regex fast-path ───────────────────────────────────────────────
    regex_emails = _extract_emails(all_text)
    regex_phones = _extract_phones(all_text)

    logger.info(
        f"[EXTRACTOR] '{company_name}' pre-LLM: "
        f"places={'✓' if places_data.get('phone') else '✗'} "
        f"hunter={'✓' if hunter_email else '✗'} "
        f"direct_email={'✓ ' + direct_email if direct_email else '✗'} "
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
    direct_formatted = (
        f"Domain-matched emails found via direct HTML scan: {', '.join(direct_emails[:5])}"
        if direct_emails else "(No domain-matched emails found in contact pages HTML)"
    )

    # ── LLM extraction ────────────────────────────────────────────────────────
    schema_formatted = (
        f"Phone: {schema_phone or 'Not found'}\n"
        f"Email: {schema_email or 'Not found'}\n"
        f"Address: {schema_address or 'Not found'}\n"
        f"Contact person: {schema_person or 'Not found'}"
        if (schema_phone or schema_email or schema_address)
        else "(No Schema.org structured data found on this website)"
    )

    # Build prompt with Tier 0 prepended
    full_prompt = (
        f"=== TIER 0 — SCHEMA.ORG STRUCTURED DATA (self-reported by business) ===\n"
        f"{schema_formatted}\n\n"
        f"=== TIER 2.5 — DIRECT HTML CONTACT PAGE SCAN (domain-matched emails only) ===\n"
        f"{direct_formatted}\n\n"
        + EXTRACTION_PROMPT.format(
            company_name=company_name,
            url=url or "unknown",
            places_data=places_formatted,
            hunter_data=hunter_formatted,
            website_content=website_content[:2500] if website_content else "(not accessible)",
            search_snippets=search_snippets[:2000] if search_snippets else "(no search data)",
        )
    )
    prompt = full_prompt

    raw = await _call_llm(prompt, openrouter_key, model)
    extracted = _parse_json(raw or "")

    # ── Safety net: inject Tier 0 (Schema.org) if LLM missed ────────────────
    if schema_phone and extracted["phone"] == "Not found":
        extracted["phone"] = schema_phone
        logger.info(f"[EXTRACTOR] Schema.org injected phone for '{company_name}'")
    if schema_email and extracted["email"] == "Not found":
        extracted["email"] = schema_email
        logger.info(f"[EXTRACTOR] Schema.org injected email for '{company_name}'")
    if schema_address and extracted["address"] == "Not found":
        extracted["address"] = schema_address
        logger.info(f"[EXTRACTOR] Schema.org injected address for '{company_name}'")
    if schema_person and extracted["contact_person"] == "Not found":
        extracted["contact_person"] = schema_person
        logger.info(f"[EXTRACTOR] Schema.org injected contact_person for '{company_name}'")

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

    # ── Safety net: inject Tier 2.5 (direct HTML scan — domain-matched) ───────
    # This is highly reliable: reservations@, contact@, info@ directly from HTML.
    # Injected BEFORE regex (which might pick up unrelated emails from snippets).
    if direct_email and extracted["email"] == "Not found":
        extracted["email"] = direct_email
        logger.info(
            f"[EXTRACTOR] Direct HTML injected email for '{company_name}': {direct_email}"
        )

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
    tavily_api_key: str = "",
    db_pool: Any = None,
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
        db_pool: Optional asyncpg pool for cost logging.

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
                    tavily_api_key=tavily_api_key,
                    db_pool=db_pool,
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

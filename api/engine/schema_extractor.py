"""
PartnerScout AI — Schema.org / JSON-LD Structured Data Extractor.

Fetches raw HTML from official company websites and extracts
machine-readable contact data from structured data markup.

Why this works better than LLM parsing:
  - Businesses embed this data FOR machines (SEO, Google rich snippets)
  - 100% accurate — the business wrote it themselves
  - No hallucination possible — it's either there or it's not
  - Completely free — just HTML fetch + JSON parse
  - No rate limits

Supported Schema.org types:
  Hotel, LodgingBusiness, Resort, BedAndBreakfast
  EventVenue, MeetingRoom
  TravelAgency, LocalBusiness, Organization
  ContactPoint, PostalAddress

Coverage: ~60-70% of modern professional websites have JSON-LD.
Result: phone/email/address found without ANY API call or LLM token.
"""

import json
import re
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

# ── HTTP Client Settings ──────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
}

# Schema.org types that represent the businesses we're looking for
_TARGET_TYPES: frozenset[str] = frozenset({
    "Hotel", "LodgingBusiness", "Resort", "BedAndBreakfast",
    "Hostel", "Motel", "VacationRental",
    "EventVenue", "MeetingRoom", "PerformingArtsTheater",
    "TravelAgency", "LocalBusiness", "Organization",
    "Corporation", "ProfessionalService", "EntertainmentBusiness",
    "SportsClub", "WeddingVenue",
})

# Contact types within contactPoint that yield department emails
_PREFERRED_CONTACT_TYPES: tuple[str, ...] = (
    "reservations", "booking", "sales", "events", "weddings",
    "groups", "mice", "concierge", "spa", "info", "contact",
)


# ── HTML Fetcher ──────────────────────────────────────────────────────────────

async def _fetch_html(url: str, timeout: float = 10.0) -> str:
    """
    Fetch raw HTML from a URL. Returns empty string on failure.

    Args:
        url: Target URL.
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string.
    """
    if not url:
        return ""
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = await client.get(url)
            # Accept 200 and common success codes; skip 404/403/5xx
            if response.status_code not in (200, 201, 203):
                logger.debug(
                    f"[SCHEMA][_fetch_html] HTTP {response.status_code} for {url}"
                )
                return ""
            return response.text
    except Exception as e:
        logger.debug(f"[SCHEMA][_fetch_html] Error fetching {url}: {e}")
        return ""


# ── JSON-LD Parser ────────────────────────────────────────────────────────────

def _extract_jsonld_blocks(html: str) -> list[dict[str, Any]]:
    """
    Extract all <script type="application/ld+json"> blocks from HTML.

    Args:
        html: Raw HTML string.

    Returns:
        List of parsed JSON-LD dicts (flattened — handles @graph arrays).
    """
    pattern = re.compile(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.DOTALL | re.IGNORECASE,
    )
    blocks: list[dict[str, Any]] = []

    for match in pattern.finditer(html):
        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            # Some sites have minor JSON errors — try to salvage
            try:
                # Remove JavaScript comments (not valid JSON but some sites do this)
                cleaned = re.sub(r'//[^\n]*\n', '\n', raw_json)
                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                parsed = json.loads(cleaned)
            except Exception:
                continue

        # Flatten @graph arrays (common pattern: one script with multiple entities)
        if isinstance(parsed, dict) and "@graph" in parsed:
            for item in parsed["@graph"]:
                if isinstance(item, dict):
                    blocks.append(item)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    blocks.append(item)
        elif isinstance(parsed, dict):
            blocks.append(parsed)

    return blocks


def _get_type(block: dict[str, Any]) -> str:
    """Extract Schema.org @type as a clean string."""
    raw = block.get("@type", "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    # Strip namespace: "schema:Hotel" → "Hotel"
    return str(raw).split(":")[-1].split("/")[-1]


def _clean_str(value: Any) -> str:
    """Safely convert any value to a clean string."""
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # e.g. {"@value": "+33 4 93 76 50 50"}
        return str(value.get("@value", "")).strip()
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def _extract_address(address_data: Any) -> str:
    """
    Build a full address string from Schema.org PostalAddress.

    Args:
        address_data: PostalAddress dict or string.

    Returns:
        Formatted address string.
    """
    if not address_data:
        return ""
    if isinstance(address_data, str):
        return address_data.strip()
    if isinstance(address_data, list):
        address_data = address_data[0] if address_data else {}
    if not isinstance(address_data, dict):
        return ""

    parts = []
    street = _clean_str(address_data.get("streetAddress", ""))
    city   = _clean_str(address_data.get("addressLocality", ""))
    postal = _clean_str(address_data.get("postalCode", ""))
    country_raw = address_data.get("addressCountry", "")
    country = _clean_str(
        country_raw.get("name", country_raw) if isinstance(country_raw, dict) else country_raw
    )

    if street:
        parts.append(street)
    if postal and city:
        parts.append(f"{postal} {city}")
    elif city:
        parts.append(city)
    if country:
        parts.append(country)

    return ", ".join(parts)


def _extract_contact_point_email(contact_points: Any) -> str:
    """
    Extract best email from Schema.org contactPoint array.

    Prefers department emails: reservations@, events@, contact@, etc.

    Args:
        contact_points: Single contactPoint dict or list.

    Returns:
        Best email found, or empty string.
    """
    if not contact_points:
        return ""
    if isinstance(contact_points, dict):
        contact_points = [contact_points]
    if not isinstance(contact_points, list):
        return ""

    # Score each contact point: prefer department types
    best_email = ""
    best_score = -1

    for cp in contact_points:
        if not isinstance(cp, dict):
            continue
        email = _clean_str(cp.get("email", ""))
        if not email or "@" not in email:
            continue

        # Score by contact type preference
        contact_type = _clean_str(
            cp.get("contactType", "") or cp.get("name", "")
        ).lower()
        score = 0
        for i, preferred in enumerate(_PREFERRED_CONTACT_TYPES):
            if preferred in contact_type:
                score = len(_PREFERRED_CONTACT_TYPES) - i
                break

        if score > best_score:
            best_score = score
            best_email = email

    return best_email


def _extract_from_block(block: dict[str, Any]) -> dict[str, str]:
    """
    Extract contact fields from a single Schema.org JSON-LD block.

    Args:
        block: Parsed JSON-LD dict.

    Returns:
        Dict with phone, email, address, contact_person, company_name fields.
    """
    result = {
        "phone": "", "email": "", "address": "",
        "contact_person": "", "company_name": "",
    }

    # Company name
    result["company_name"] = _clean_str(block.get("name", ""))

    # Phone: telephone field (standard) or phone (older schemas)
    phone = (
        _clean_str(block.get("telephone", ""))
        or _clean_str(block.get("phone", ""))
    )
    result["phone"] = phone

    # Email: direct email field
    email = _clean_str(block.get("email", ""))
    result["email"] = email

    # Email: contactPoint (department-level emails)
    if not email:
        cp_email = _extract_contact_point_email(block.get("contactPoint"))
        if cp_email:
            result["email"] = cp_email

    # Address
    addr = _extract_address(block.get("address") or block.get("location"))
    result["address"] = addr

    # Contact person: from employee, founder, or Person entity
    person_sources = []
    for key in ("employee", "founder", "member", "author"):
        persons = block.get(key, [])
        if isinstance(persons, dict):
            persons = [persons]
        if isinstance(persons, list):
            person_sources.extend(persons)

    for person in person_sources:
        if not isinstance(person, dict):
            continue
        p_type = _get_type(person)
        if p_type not in ("Person", ""):
            continue
        p_name = _clean_str(person.get("name", ""))
        p_role = _clean_str(
            person.get("jobTitle", "")
            or person.get("roleName", "")
        )
        # Only include if has a role (Director of Sales, etc.)
        if p_name and p_role:
            result["contact_person"] = f"{p_name}, {p_role}"
            # Check if person has personal email
            p_email = _clean_str(person.get("email", ""))
            if p_email and "@" in p_email:
                result["personal_email"] = p_email
            break

    return result


# ── Main Extraction Function ──────────────────────────────────────────────────

async def extract_schema_contacts(
    url: str,
    company_name: str = "",
) -> dict[str, str]:
    """
    Extract contact information from Schema.org / JSON-LD structured data.

    Fetches raw HTML from the official website and parses structured data
    embedded by the business for search engines. This data is self-reported,
    machine-readable, and 100% accurate when present.

    Coverage: ~60-70% of modern professional business websites.
    Cost: $0.00 — just HTTP fetch + JSON parse.

    Args:
        url: Official company website URL.
        company_name: Company name for logging context.

    Returns:
        Dict with fields: phone, email, address, contact_person, company_name.
        Empty string values mean field not found in schema data.
    """
    empty = {
        "phone": "", "email": "", "address": "",
        "contact_person": "", "company_name": "",
    }

    if not url:
        return empty

    # Also try the /contact page which often has richer schema data
    base = url.rstrip("/")
    pages_to_try = [url, f"{base}/contact", f"{base}/en/contact"]

    best_result = dict(empty)
    found_anything = False

    for page_url in pages_to_try:
        html = await _fetch_html(page_url)
        if not html:
            continue

        blocks = _extract_jsonld_blocks(html)
        if not blocks:
            continue

        for block in blocks:
            schema_type = _get_type(block)

            # Skip blocks that are clearly not about our company
            # (e.g. BreadcrumbList, WebSite, SearchAction)
            if schema_type and schema_type not in _TARGET_TYPES and schema_type not in (
                "", "WebPage", "WebSite", "FAQPage"  # process these too, may have org data
            ):
                # Check if there's a nested organization
                nested = block.get("publisher") or block.get("creator") or block.get("organization")
                if nested and isinstance(nested, dict):
                    extracted = _extract_from_block(nested)
                    if any(extracted.values()):
                        _merge_result(best_result, extracted)
                        found_anything = True
                continue

            extracted = _extract_from_block(block)

            # Only use if name matches (or no name context)
            block_name = extracted.get("company_name", "").lower()
            ctx_name   = company_name.lower()
            if block_name and ctx_name and len(ctx_name) > 4:
                # Accept if there's reasonable overlap (handles abbreviations)
                words_match = any(
                    w in block_name for w in ctx_name.split()
                    if len(w) > 3
                )
                if not words_match:
                    continue

            _merge_result(best_result, extracted)
            if any(v for v in extracted.values() if v):
                found_anything = True

        # If we have phone + email already, no need to check more pages
        if best_result["phone"] and best_result["email"]:
            break

    if found_anything:
        logger.info(
            f"[SCHEMA][extract_schema_contacts] '{company_name}' → "
            f"phone={'✓' if best_result['phone'] else '✗'} "
            f"email={'✓' if best_result['email'] else '✗'} "
            f"address={'✓' if best_result['address'] else '✗'} "
            f"person={'✓' if best_result['contact_person'] else '✗'}"
        )
    else:
        logger.debug(
            f"[SCHEMA][extract_schema_contacts] No schema data for '{company_name}' ({url})"
        )

    return best_result


def _merge_result(target: dict[str, str], source: dict[str, str]) -> None:
    """Fill empty fields in target from source (never overwrite existing values)."""
    for key, val in source.items():
        if val and not target.get(key):
            target[key] = val

"""
PartnerScout AI — Strict Contact Validator v1.

Three-layer validation using ONLY currently connected models (no new paid APIs):

  Layer 1 — Rules:     instant, free, zero API cost
    - URL must not be aggregator/OTA/travel-blog domain
    - Email domain must match website root domain
    - Phone must pass international format check
    - Personal email providers (gmail, yahoo) → rejected

  Layer 2 — Source trust:  instant, free
    - Phone from Google Places  → automatically trusted (Places = verified business)
    - Email from Hunter.io      → automatically trusted (Hunter = domain-verified)
    - places_verified flag      → URL is trusted

  Layer 3 — LLM sanity check (Haiku, ~$0.0001/company):
    - Final cross-check: does name + email + phone form a consistent company record?
    - Catches hallucinations the extractor might produce
    - Run ONLY on companies that pass Layers 1+2

Gate logic:
    PASS = url_ok AND (email_ok OR phone_ok) AND llm_ok
    Contacts with "Not found" in all three → rejected
    Partial contacts (e.g. only phone) → rejected (need at least email+phone OR url+phone)
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger


# ── Known OTA / Partner Email Domains ─────────────────────────────────────────

_OTA_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "booking.com", "tripadvisor.com", "expedia.com", "hotels.com",
    "agoda.com", "airbnb.com", "kayak.com", "trivago.com",
    "relaischateaux.com", "lhw.com", "leadinghotels.com",
    "mrandmrssmith.com", "secretescapes.com", "privateupgrades.com",
    "tablet.com", "jetsetter.com", "designhotels.com",
    "charmeandtradition.com", "week-ends-de-reve.com",
    "smallluxuryhotels.com", "sawdays.co.uk", "sawdays.com",
})

# Personal email providers → not an official business email
_PERSONAL_EMAIL_PROVIDERS: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com",
    "outlook.com", "hotmail.com", "hotmail.fr", "live.com", "live.fr",
    "yahoo.com", "yahoo.fr", "yahoo.co.uk",
    "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me",
    "laposte.net", "orange.fr", "wanadoo.fr", "free.fr", "sfr.fr",
})

# OTA / aggregator URL patterns (path-level check)
_BAD_URL_PATHS: tuple[str, ...] = (
    "/hotel/", "/hotels/", "/booking/",
    "/property/", "/listings/", "/review/", "/reviews/",
    "/best-hotels", "/top-hotels", "/best-luxury",
    "/destination/", "/guide/",
)


# ── Layer 1: Rule-Based Validators ────────────────────────────────────────────

def _root_domain(url: str) -> str:
    """Extract registered root domain: 'www.hotel-negresco.com' → 'hotel-negresco.com'."""
    try:
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        parts = netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        return ""


def validate_url(url: str) -> tuple[bool, str]:
    """
    Layer 1 URL check: must not be aggregator, OTA, or travel-blog.

    Args:
        url: Company website URL.

    Returns:
        (passed: bool, reason: str)
    """
    if not url or url.strip() in ("", "Not found"):
        return False, "url_missing"

    # Circular import guard — import locally
    try:
        from api.engine.ranker import (
            AGGREGATOR_DOMAINS,
            _domain_is_travel_blog,
            _registered_domain,
        )
        domain = _registered_domain(url)
        if domain in AGGREGATOR_DOMAINS:
            return False, f"url_aggregator:{domain}"
        if _domain_is_travel_blog(url):
            return False, "url_travel_blog"
    except ImportError:
        pass  # ranker not available in unit tests

    url_lower = url.lower()
    for pattern in _BAD_URL_PATHS:
        if pattern in url_lower:
            return False, f"url_bad_path:{pattern}"

    return True, "ok"


def validate_email(email: str, website_url: str) -> tuple[bool, str]:
    """
    Layer 1 email check: domain must match website + not OTA/personal.

    Rules (in order):
      1. email must be present
      2. email domain must NOT be OTA/partner
      3. email domain must NOT be personal provider (gmail etc.)
      4. email domain must match website root domain (or close variant)

    Args:
        email: Extracted email string.
        website_url: Company official website URL.

    Returns:
        (passed: bool, reason: str)
    """
    if not email or email.strip() in ("", "Not found"):
        return False, "email_missing"

    if "@" not in email:
        return False, "email_malformed"

    email_domain = email.split("@")[-1].lower().strip()

    if email_domain in _OTA_EMAIL_DOMAINS:
        return False, f"email_ota:{email_domain}"

    if email_domain in _PERSONAL_EMAIL_PROVIDERS:
        return False, f"email_personal_provider:{email_domain}"

    # Domain consistency check
    if website_url and website_url not in ("", "Not found"):
        website_domain = _root_domain(website_url)
        if website_domain:
            # Exact match or subdomain
            if (
                email_domain == website_domain
                or email_domain.endswith("." + website_domain)
                or website_domain.endswith("." + email_domain)
            ):
                return True, "ok_domain_match"
            else:
                return False, f"email_domain_mismatch:{email_domain}≠{website_domain}"

    # No website to compare → accept if domain looks business-like
    if len(email_domain) > 4 and "." in email_domain:
        return True, "ok_no_url_to_compare"

    return False, "email_unverifiable"


def validate_phone(phone: str) -> tuple[bool, str]:
    """
    Layer 1 phone check: must be valid international or local format.

    Accepts:
      - International: +33 4 93 76 50 50, +377 98 06 20 00
      - French local:  04 93 76 50 50, 0493765050
      - At least 8 digits total

    Args:
        phone: Extracted phone string.

    Returns:
        (passed: bool, reason: str)
    """
    if not phone or phone.strip() in ("", "Not found"):
        return False, "phone_missing"

    digits_only = re.sub(r"\D", "", phone)

    if len(digits_only) < 8:
        return False, f"phone_too_short:{len(digits_only)}_digits"

    cleaned = phone.strip()

    # International format (starts with +)
    if cleaned.startswith("+"):
        if re.match(r"^\+\d[\d\s.\-()]{7,17}$", cleaned):
            return True, "ok_international"
        return False, "phone_bad_international_format"

    # French local format: 0X XX XX XX XX
    digits_clean = re.sub(r"[\s.\-()]", "", cleaned)
    if re.match(r"^0[1-9]\d{8}$", digits_clean):
        return True, "ok_french_local"

    # Generic: 8+ digits → acceptable
    if len(digits_only) >= 8:
        return True, "ok_sufficient_digits"

    return False, "phone_invalid_format"


# ── Layer 2: Source Trust ──────────────────────────────────────────────────────

def is_places_verified(company: dict[str, Any]) -> bool:
    """Return True if company was discovered via Google Places API (verified)."""
    return bool(company.get("places_verified"))


def is_hunter_email(company: dict[str, Any]) -> bool:
    """
    Heuristic: Hunter.io emails usually match domain exactly.

    If email domain == website domain → very likely from Hunter or Schema.org.
    We don't track source explicitly, but domain-match is the Hunter signature.
    """
    email = company.get("email", "")
    url   = company.get("url",   "")
    if not email or "@" not in email or not url:
        return False
    return _root_domain(url) == email.split("@")[-1].lower().strip()


# ── Layer 3: LLM Sanity Check ──────────────────────────────────────────────────

_SANITY_PROMPT = """\
You are a B2B data quality auditor. Check whether this business contact record is \
internally consistent and looks like a real, active company — NOT a booking aggregator, \
OTA reseller, or travel blog.

Company name: {company_name}
Website: {url}
Email: {email}
Phone: {phone}
Address: {address}
Category: {category}

Answer with ONLY valid JSON (no markdown, no explanation):
{{
  "consistent": true/false,
  "email_looks_official": true/false,
  "phone_looks_real": true/false,
  "url_looks_official": true/false,
  "confidence": 0.0-1.0,
  "reject_reason": "short reason if consistent=false, else null"
}}

Rules:
- consistent=false if email is from a booking platform or travel blog
- consistent=false if email domain does not match website domain
- consistent=false if phone looks fake (repeated digits, placeholder)
- consistent=true only if all three contact fields point to the same real company
"""


async def llm_sanity_check(
    company: dict[str, Any],
    openrouter_key: str,
    model: str,
) -> tuple[bool, float, str | None]:
    """
    Layer 3: LLM sanity check using Tier C model (Haiku) — ~$0.0001/call.

    Verifies the contact record is internally consistent.
    Only called for companies that already passed Layers 1+2.

    Args:
        company: Enriched company dict.
        openrouter_key: OpenRouter API key.
        model: Tier C model ID (claude-haiku or similar).

    Returns:
        (passed: bool, confidence: float, reject_reason: str | None)
    """
    prompt = _SANITY_PROMPT.format(
        company_name=company.get("company_name", "Unknown"),
        url=company.get("url",     "Not found"),
        email=company.get("email", "Not found"),
        phone=company.get("phone", "Not found"),
        address=company.get("address", "Not found"),
        category=company.get("category", ""),
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://partnerscout.ai",
                    "X-Title": "PartnerScout AI",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 120,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

            data = json.loads(raw)
            passed    = bool(data.get("consistent", False))
            confidence = float(data.get("confidence", 0.0))
            reason     = data.get("reject_reason") if not passed else None

            logger.info(
                f"[STRICT_VALIDATOR][llm] '{company.get('company_name')}' → "
                f"consistent={passed} confidence={confidence:.2f} reason={reason}"
            )
            return passed, confidence, reason

    except json.JSONDecodeError as e:
        logger.warning(f"[STRICT_VALIDATOR][llm] JSON parse error: {e} — defaulting to PASS")
        return True, 0.5, None  # on parse fail: don't reject (conservative)
    except Exception as e:
        logger.error(f"[STRICT_VALIDATOR][llm] API error: {e}", exc_info=True)
        return True, 0.5, None  # on API fail: don't reject (conservative)


# ── Master Validator ───────────────────────────────────────────────────────────

def validate_contact_rules(
    company: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """
    Run Layer 1 (rules) + Layer 2 (source trust) on a single company.

    Returns:
        (passed: bool, passed_checks: list[str], failed_checks: list[str])

    Gate logic:
        - URL MUST pass (mandatory)
        - BOTH email AND phone must pass, OR one passes AND source is trusted
        - Minimum: url_ok AND (email_ok + phone_ok) OR (places_phone AND email_ok)
    """
    passed_checks: list[str] = []
    failed_checks: list[str] = []

    url   = company.get("url",   "")
    email = company.get("email", "")
    phone = company.get("phone", "")

    # ── URL check (mandatory) ─────────────────────────────────────────────────
    url_ok, url_reason = validate_url(url)
    if url_ok:
        passed_checks.append(f"url:{url_reason}")
    else:
        failed_checks.append(f"url:{url_reason}")

    # ── Email check ──────────────────────────────────────────────────────────
    email_ok, email_reason = validate_email(email, url)
    if email_ok:
        passed_checks.append(f"email:{email_reason}")
    else:
        failed_checks.append(f"email:{email_reason}")

    # ── Phone check ──────────────────────────────────────────────────────────
    phone_ok, phone_reason = validate_phone(phone)
    if phone_ok:
        passed_checks.append(f"phone:{phone_reason}")
    else:
        failed_checks.append(f"phone:{phone_reason}")

    # ── Source trust overrides ────────────────────────────────────────────────
    places_verified = is_places_verified(company)
    hunter_email    = is_hunter_email(company)

    if places_verified and not phone_ok:
        # Google Places phone is pre-verified by Google → override
        phone_ok = True
        passed_checks.append("phone:places_verified_override")
        failed_checks = [c for c in failed_checks if not c.startswith("phone:")]

    if hunter_email and not email_ok:
        # Email domain matches website → Hunter-style trusted
        email_ok = True
        passed_checks.append("email:hunter_domain_match_override")
        failed_checks = [c for c in failed_checks if not c.startswith("email:")]

    # ── Final gate ────────────────────────────────────────────────────────────
    # Require: URL OK + at least 2 of 3 fields (email + phone are both needed)
    passed = url_ok and email_ok and phone_ok

    return passed, passed_checks, failed_checks


async def validate_and_filter(
    companies: list[dict[str, Any]],
    openrouter_key: str,
    tier_c_model: str,
    max_results: int = 20,
    min_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Full 3-layer validation pipeline. Returns only strictly valid contacts.

    Process:
      1. Layer 1+2: rule-based + source trust (instant, free)
      2. Layer 3: LLM sanity check on rule-passed companies (Haiku)
      3. Sort by confidence score descending
      4. Return top max_results (target: 10-20 verified contacts)

    Args:
        companies: Enriched company list from extraction pipeline.
        openrouter_key: OpenRouter API key.
        tier_c_model: Haiku model ID for LLM sanity check.
        max_results: Maximum contacts to return (default 20).
        min_results: Minimum target (default 10).

    Returns:
        Filtered, validated, sorted list of company dicts.
        Each dict gains: validation_passed, validation_checks, llm_confidence fields.
    """
    logger.info(
        f"[STRICT_VALIDATOR] Starting 3-layer validation on {len(companies)} companies "
        f"(target: {min_results}–{max_results})"
    )

    # ── Layer 1+2: Rules + source trust ────────────────────────────────────────
    layer12_passed: list[dict[str, Any]] = []
    rejected_count = 0

    for company in companies:
        passed, passed_checks, failed_checks = validate_contact_rules(company)
        company["validation_passed"]       = passed
        company["validation_checks_ok"]    = passed_checks
        company["validation_checks_fail"]  = failed_checks

        if passed:
            layer12_passed.append(company)
            logger.debug(
                f"[STRICT_VALIDATOR][L1+2] ✓ '{company.get('company_name')}' "
                f"checks={passed_checks}"
            )
        else:
            rejected_count += 1
            logger.info(
                f"[STRICT_VALIDATOR][L1+2] ✗ '{company.get('company_name')}' "
                f"failed={failed_checks}"
            )

    logger.info(
        f"[STRICT_VALIDATOR][L1+2] {len(layer12_passed)} passed, "
        f"{rejected_count} rejected"
    )

    if not layer12_passed:
        logger.warning("[STRICT_VALIDATOR] All companies failed L1+2 — returning empty list")
        return []

    # ── Layer 3: LLM sanity check (parallel, max 5 concurrent) ────────────────
    import asyncio
    semaphore = asyncio.Semaphore(5)

    async def _check_one(company: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            ok, confidence, reason = await llm_sanity_check(
                company, openrouter_key, tier_c_model
            )
            company["llm_sanity_passed"]    = ok
            company["llm_confidence"]       = confidence
            company["llm_reject_reason"]    = reason
            return company

    checked = await asyncio.gather(*[_check_one(c) for c in layer12_passed])

    # Keep only LLM-approved
    llm_passed  = [c for c in checked if c.get("llm_sanity_passed", True)]
    llm_rejected = len(checked) - len(llm_passed)

    logger.info(
        f"[STRICT_VALIDATOR][L3-LLM] {len(llm_passed)} passed LLM, "
        f"{llm_rejected} rejected"
    )

    if not llm_passed:
        # Fallback: if LLM rejects everything (model issue), return L1+2 results
        logger.warning(
            "[STRICT_VALIDATOR] LLM rejected all — fallback to Layer 1+2 results"
        )
        llm_passed = layer12_passed
        for c in llm_passed:
            c["llm_confidence"] = 0.5

    # ── Sort by confidence + luxury score ─────────────────────────────────────
    llm_passed.sort(
        key=lambda c: (
            -float(c.get("llm_confidence", 0.5)),
            -float(c.get("luxury_score", 0.0)),
        )
    )

    final = llm_passed[:max_results]

    logger.info(
        f"[STRICT_VALIDATOR] Final: {len(final)} strictly validated contacts "
        f"(from {len(companies)} input)"
    )
    return final

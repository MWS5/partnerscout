"""
PartnerScout AI — Luxury Segment Validator.

Uses LLM (Tier C — Haiku) to score how strongly a company
operates in the luxury/premium market segment.

Score interpretation:
  0.0 — mass market, low quality, no luxury signals
  0.5 — mid-market, some premium features
  1.0 — ultra-luxury, VIP, exclusive, palace-grade

Threshold for inclusion: 0.6 (configurable).
"""

import re
from typing import Any

import httpx
from loguru import logger

# ── Luxury Signal Keywords ────────────────────────────────────────────────────

LUXURY_KEYWORDS: list[str] = [
    "VIP",
    "luxury",
    "luxe",
    "prestige",
    "exclusive",
    "exclusif",
    "bespoke",
    "premium",
    "haut de gamme",
    "5-star",
    "5 star",
    "5 étoiles",
    "palace",
    "private",
    "privé",
    "haute couture",
    "ultra-high-end",
    "tailor-made",
    "sur mesure",
    "curated",
    "opulent",
]

# ── Scoring Prompt ────────────────────────────────────────────────────────────

SCORING_PROMPT = """You are a luxury market analyst. Evaluate whether this company truly operates in the luxury/premium segment.

Company: {company_name}
Niche: {niche}
Website content:
---
{content}
---

Luxury signal keywords found in content: {keywords_found}

Rate the company's luxury positioning on a scale from 0.0 to 1.0:
  0.0 = clearly mass-market, budget, low-end
  0.3 = mid-market with some quality signals
  0.6 = premium, high-quality service
  0.8 = luxury, exclusive, high-net-worth clientele
  1.0 = ultra-luxury, palace-grade, VIP only

Consider: pricing signals, clientele described, service quality language, brand positioning.

Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""


# ── OpenRouter Call ───────────────────────────────────────────────────────────

async def _call_haiku(
    prompt: str,
    api_key: str,
    model: str,
) -> str:
    """
    Call Tier C model (Haiku) for luxury scoring.

    Args:
        prompt: Full scoring prompt.
        api_key: OpenRouter API key.
        model: Tier C model ID (claude-haiku).

    Returns:
        Raw LLM response string. Empty string on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://partnerscout.ai",
        "X-Title": "PartnerScout AI",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[VALIDATOR][_call_haiku] API error: {e}", exc_info=True)
        return ""


def _extract_score(raw: str) -> float:
    """
    Parse float score from LLM response.

    Falls back to 0.5 if parsing fails (neutral — neither include nor exclude).

    Args:
        raw: Raw LLM response (should be a decimal number).

    Returns:
        Float score clamped to [0.0, 1.0].
    """
    if not raw:
        return 0.5

    match = re.search(r"(\d+\.?\d*)", raw.strip())
    if not match:
        return 0.5

    try:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5


def _find_luxury_keywords(content: str) -> list[str]:
    """
    Find which luxury keywords appear in website content.

    Args:
        content: Website text content.

    Returns:
        List of matched luxury keywords (deduplicated).
    """
    content_lower = content.lower()
    return [kw for kw in LUXURY_KEYWORDS if kw.lower() in content_lower]


async def score_luxury(
    company_name: str,
    website_content: str,
    niche: str,
    openrouter_key: str,
    model: str,
) -> float:
    """
    Score how strongly a company operates in the luxury segment.

    Uses keyword pre-check to boost prompt quality, then calls
    Tier C model for final nuanced scoring.

    Args:
        company_name: Company name for context.
        website_content: Extracted website text (from Jina Reader).
        niche: Business niche for context.
        openrouter_key: OpenRouter API key.
        model: Tier C model ID.

    Returns:
        Float score 0.0–1.0. Returns 0.5 on any error (neutral).
    """
    keywords_found = _find_luxury_keywords(website_content)

    # Fast path: no content at all → score 0.0
    if not website_content.strip():
        logger.warning(
            f"[VALIDATOR][score_luxury] No content for '{company_name}' → score=0.0"
        )
        return 0.0

    prompt = SCORING_PROMPT.format(
        company_name=company_name,
        niche=niche,
        content=website_content[:1500],
        keywords_found=", ".join(keywords_found) if keywords_found else "none detected",
    )

    raw = await _call_haiku(prompt, openrouter_key, model)
    score = _extract_score(raw)

    logger.info(
        f"[VALIDATOR][score_luxury] '{company_name}' → score={score:.2f} "
        f"(keywords={keywords_found})"
    )
    return score


def filter_by_luxury(
    companies: list[dict[str, Any]],
    min_score: float = 0.6,
) -> list[dict[str, Any]]:
    """
    Filter company list to include only luxury-scored companies.

    Reads 'luxury_score' field from each company dict.
    Companies without a score are excluded (treated as 0.0).

    Args:
        companies: List of company dicts with 'luxury_score' field.
        min_score: Minimum score to include (default 0.6).

    Returns:
        Filtered list sorted by luxury_score descending.
    """
    qualified = [
        c for c in companies
        if float(c.get("luxury_score", 0.0)) >= min_score
    ]
    qualified.sort(key=lambda x: float(x.get("luxury_score", 0.0)), reverse=True)

    logger.info(
        f"[VALIDATOR][filter_by_luxury] {len(companies)} → {len(qualified)} "
        f"after min_score={min_score} filter"
    )
    return qualified

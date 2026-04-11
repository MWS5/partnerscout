"""
PartnerScout AI — Company Data Extractor.

Extracts structured contact data from company websites using:
  1. Jina Reader — fetches clean page content
  2. LLM via OpenRouter — extracts structured fields from content

Extraction fields: address, phone, email, contact_person,
personal_phone, personal_email.

Never fabricates data — uses "Not found" for missing fields.
Runs batch extraction with asyncio semaphore for concurrency control.
"""

import asyncio
import json
from typing import Any, Optional

import httpx
from loguru import logger

from api.engine.searcher import jina_read

# ── LLM Extraction Prompt ─────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a B2B data extraction specialist. Extract company contact information from the provided website text.

Company name: {company_name}
Website content:
---
{content}
---

Extract ONLY information explicitly present in the text. Do NOT invent or guess any data.
If a field is not found, use the exact string "Not found".

Return ONLY valid JSON with these exact fields:
{{
  "address": "full postal address or Not found",
  "phone": "main company phone number or Not found",
  "email": "general contact email or Not found",
  "contact_person": "name and title of key contact (e.g. 'Jean Dupont, Director of Sales') or Not found",
  "personal_phone": "direct phone of the contact person or Not found",
  "personal_email": "direct email of the contact person or Not found"
}}

Output only the JSON object. No explanation, no markdown, no code fences."""


# ── OpenRouter LLM Call ───────────────────────────────────────────────────────

async def _call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    """
    Call OpenRouter API with a single user message.

    Args:
        prompt: Full prompt string to send.
        api_key: OpenRouter API key.
        model: OpenRouter model ID.

    Returns:
        LLM response text, or None on failure.
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
        "temperature": 0.1,
        "max_tokens": 500,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[EXTRACTOR][_call_openrouter] API error (model={model}): {e}", exc_info=True)
        return None


def _parse_extraction_response(raw: str) -> dict[str, str]:
    """
    Parse JSON from LLM extraction response.

    Strips markdown fences if present before parsing.

    Args:
        raw: Raw LLM response string.

    Returns:
        Dict with extraction fields, or all "Not found" on parse failure.
    """
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


async def extract_company_data(
    url: str,
    company_name: str,
    openrouter_key: str,
    model: str,
) -> dict[str, Any]:
    """
    Extract structured company contact data from a website.

    Pipeline:
      1. Fetch page content via Jina Reader.
      2. Build extraction prompt with content.
      3. Call LLM for structured JSON extraction.
      4. Parse and validate response.

    Args:
        url: Company website URL.
        company_name: Company name (used in prompt for context).
        openrouter_key: OpenRouter API key.
        model: OpenRouter model ID (use Tier B for cost efficiency).

    Returns:
        Dict with address, phone, email, contact_person,
        personal_phone, personal_email. Never empty — uses "Not found".
    """
    content = await jina_read(url, max_chars=3000)

    if not content:
        logger.warning(f"[EXTRACTOR][extract_company_data] No content from Jina for {url}")
        return {
            "website": url,
            "jina_content": "",
            "address": "Not found",
            "phone": "Not found",
            "email": "Not found",
            "contact_person": "Not found",
            "personal_phone": "Not found",
            "personal_email": "Not found",
        }

    prompt = EXTRACTION_PROMPT.format(
        company_name=company_name,
        content=content[:2500],
    )

    raw_response = await _call_openrouter(prompt, openrouter_key, model)
    extracted = _parse_extraction_response(raw_response or "")
    extracted["website"] = url
    extracted["jina_content"] = content  # pass full content for luxury scoring

    logger.info(
        f"[EXTRACTOR][extract_company_data] Extracted data for '{company_name}' "
        f"(email_found={extracted['email'] != 'Not found'})"
    )
    return extracted


async def extract_batch(
    companies: list[dict[str, Any]],
    openrouter_key: str,
    model: str,
    max_concurrent: int = 5,
) -> list[dict[str, Any]]:
    """
    Batch-extract contact data for multiple companies with concurrency control.

    Uses asyncio.Semaphore to limit simultaneous LLM+Jina calls.
    Each company dict must have 'url' and 'company_name' fields.

    Args:
        companies: List of company dicts with at least 'url' and 'company_name'.
        openrouter_key: OpenRouter API key.
        model: OpenRouter model ID.
        max_concurrent: Max simultaneous extraction tasks (default 5).

    Returns:
        List of company dicts merged with extracted contact fields.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_with_semaphore(company: dict[str, Any]) -> dict[str, Any]:
        """Wrap single extraction with semaphore guard."""
        async with semaphore:
            url = company.get("url", "")
            name = company.get("company_name", "Unknown")
            extracted = await extract_company_data(url, name, openrouter_key, model)
            return {**company, **extracted}

    tasks = [_extract_with_semaphore(c) for c in companies]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    logger.info(
        f"[EXTRACTOR][extract_batch] Extracted {len(results)} companies "
        f"(concurrency={max_concurrent})"
    )
    return list(results)

"""
PartnerScout AI — Multi-Source Search Engine.

Standalone async search layer supporting:
  - DuckDuckGo (via duckduckgo-search, sync → thread executor)
  - Brave Search API v1
  - SearXNG (self-hosted)
  - Jina Reader for full-page content extraction

All functions return a unified result dict:
    {"title": str, "url": str, "snippet": str, "source": str}

Never raises — returns empty list on any failure.
Uses loguru for structured error logging.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Shared thread pool for sync DDG calls
_DDG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Unified result type alias
SearchResultItem = dict[str, str]


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _ddg_search_sync(query: str, num: int) -> list[SearchResultItem]:
    """
    Synchronous DDG search (must run in thread executor).

    Args:
        query: Search query string.
        num: Number of results to fetch.

    Returns:
        List of unified result dicts.
    """
    try:
        from duckduckgo_search import DDGS

        results: list[SearchResultItem] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "ddg",
                })
        return results
    except Exception as e:
        logger.error(f"[SEARCHER][_ddg_search_sync] DDG error for '{query}': {e}")
        return []


async def duckduckgo_search(query: str, num: int = 5) -> list[SearchResultItem]:
    """
    Async DuckDuckGo search — offloads sync library to thread executor.

    Args:
        query: Search query string.
        num: Number of results (default 5).

    Returns:
        List of unified result dicts. Empty list on failure.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            _DDG_EXECUTOR,
            _ddg_search_sync,
            query,
            num,
        )
        logger.debug(f"[SEARCHER][duckduckgo_search] '{query}' → {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"[SEARCHER][duckduckgo_search] Executor error: {e}", exc_info=True)
        return []


# ── Brave Search ──────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def _brave_search_with_retry(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    num: int,
) -> list[SearchResultItem]:
    """
    Internal Brave Search call with tenacity retry.

    Args:
        client: Shared httpx client.
        query: Search query string.
        api_key: Brave Search API key.
        num: Number of results.

    Returns:
        List of unified result dicts.
    """
    response = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
        params={"q": query, "count": num},
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()
    results: list[SearchResultItem] = []
    for r in data.get("web", {}).get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
            "source": "brave",
        })
    return results


async def brave_search(query: str, api_key: str, num: int = 5) -> list[SearchResultItem]:
    """
    Async Brave Search API v1 call.

    Args:
        query: Search query string.
        api_key: Brave Search API key.
        num: Number of results (default 5).

    Returns:
        List of unified result dicts. Empty list on failure.
    """
    try:
        async with httpx.AsyncClient() as client:
            results = await _brave_search_with_retry(client, query, api_key, num)
            logger.debug(f"[SEARCHER][brave_search] '{query}' → {len(results)} results")
            return results
    except Exception as e:
        logger.error(f"[SEARCHER][brave_search] Error for '{query}': {e}", exc_info=True)
        return []


# ── SearXNG ───────────────────────────────────────────────────────────────────

async def searxng_search(
    query: str,
    base_url: str,
    num: int = 5,
) -> list[SearchResultItem]:
    """
    Async SearXNG self-hosted search.

    Args:
        query: Search query string.
        base_url: Base URL of the SearXNG instance (e.g. http://searxng:8080).
        num: Number of results (default 5).

    Returns:
        List of unified result dicts. Empty list on failure.
    """
    if not base_url:
        return []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "pageno": 1},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        results: list[SearchResultItem] = []
        for r in data.get("results", [])[:num]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "source": "searxng",
            })
        logger.debug(f"[SEARCHER][searxng_search] '{query}' → {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"[SEARCHER][searxng_search] Error for '{query}': {e}", exc_info=True)
        return []


# ── Jina Reader ───────────────────────────────────────────────────────────────

async def jina_read(url: str, max_chars: int = 3000) -> str:
    """
    Extract readable text content from a URL via Jina Reader.

    Jina Reader (r.jina.ai) converts any URL to clean markdown text.
    Free tier, no API key required.

    Args:
        url: Target URL to read.
        max_chars: Maximum characters to return (default 3000).

    Returns:
        Extracted text content. Empty string on failure.
    """
    if not url:
        return ""

    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                jina_url,
                headers={"Accept": "text/plain"},
                timeout=15.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            content = response.text[:max_chars]
            logger.debug(f"[SEARCHER][jina_read] {url} → {len(content)} chars")
            return content
    except Exception as e:
        logger.error(f"[SEARCHER][jina_read] Error reading '{url}': {e}", exc_info=True)
        return ""


async def jina_read_batch(
    urls: list[str],
    max_chars_each: int = 2000,
) -> list[str]:
    """
    Parallel batch content extraction via Jina Reader.

    Uses asyncio.gather for concurrent fetching.
    Individual failures return empty string without affecting others.

    Args:
        urls: List of URLs to read.
        max_chars_each: Max chars per URL (default 2000).

    Returns:
        List of content strings, same order as input URLs.
    """
    if not urls:
        return []

    tasks = [jina_read(url, max_chars_each) for url in urls]
    results: list[str] = await asyncio.gather(*tasks, return_exceptions=False)
    logger.info(f"[SEARCHER][jina_read_batch] Read {len(urls)} URLs in parallel")
    return list(results)

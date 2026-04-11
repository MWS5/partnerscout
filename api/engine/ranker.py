"""
PartnerScout AI — BM25 Result Ranker.

Ranks raw search results using BM25 relevance scoring.
Deduplicates by URL before ranking to avoid inflating scores
for repeated pages.

Usage:
    ranked = bm25_rank("luxury hotel Nice", results, top_k=10)
"""

from typing import Any
from urllib.parse import urlparse

from loguru import logger
from rank_bm25 import BM25Okapi


def deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate results by normalized URL.

    Normalization strips trailing slashes and lowercases scheme/host.
    First occurrence of each URL is kept.

    Args:
        results: List of search result dicts with 'url' key.

    Returns:
        Deduplicated list preserving insertion order.
    """
    seen_urls: set[str] = set()
    unique: list[dict[str, Any]] = []

    for result in results:
        url = result.get("url", "")
        if not url:
            continue

        try:
            parsed = urlparse(url)
            normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
        except Exception:
            normalized = url.lower().rstrip("/")

        if normalized not in seen_urls:
            seen_urls.add(normalized)
            unique.append(result)

    logger.debug(f"[RANKER][deduplicate] {len(results)} → {len(unique)} after dedup")
    return unique


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace tokenizer for BM25.

    Lowercases and splits on whitespace.
    Filters empty tokens.

    Args:
        text: Input string.

    Returns:
        List of lowercase word tokens.
    """
    return [tok for tok in text.lower().split() if tok]


def _build_corpus_text(result: dict[str, Any]) -> str:
    """
    Combine title and snippet into a single BM25-indexable text.

    Args:
        result: Search result dict.

    Returns:
        Combined text string.
    """
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    return f"{title} {snippet}"


def bm25_rank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Rank search results by BM25 relevance to the query.

    Adds 'bm25_score' field to each result dict.
    Returns top_k results sorted by score descending.

    Handles edge cases:
    - Empty results list → returns empty list
    - Single result → returns with score 1.0
    - Query not matching any document → returns original order with score 0.0

    Args:
        query: The original search query used to retrieve results.
        results: List of search result dicts (must have title, snippet fields).
        top_k: Maximum number of results to return (default 10).

    Returns:
        Top-k results sorted by BM25 score descending, each with 'bm25_score' field.
    """
    if not results:
        return []

    if len(results) == 1:
        results[0]["bm25_score"] = 1.0
        return results

    corpus_texts = [_build_corpus_text(r) for r in results]
    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    tokenized_query = _tokenize(query)

    if not tokenized_query:
        for r in results:
            r["bm25_score"] = 0.0
        return results[:top_k]

    try:
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
    except Exception as e:
        logger.error(f"[RANKER][bm25_rank] BM25 scoring error: {e}", exc_info=True)
        for r in results:
            r["bm25_score"] = 0.0
        return results[:top_k]

    scored_results = []
    for result, score in zip(results, scores):
        result_copy = dict(result)
        result_copy["bm25_score"] = float(score)
        scored_results.append(result_copy)

    scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
    top_results = scored_results[:top_k]

    logger.info(
        f"[RANKER][bm25_rank] Ranked {len(results)} results, "
        f"returning top {len(top_results)} "
        f"(top score: {top_results[0]['bm25_score']:.3f} if top_results else 0)"
    )
    return top_results

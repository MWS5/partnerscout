"""
PartnerScout AI — BM25 Result Ranker + Official Site Filter.

Two responsibilities:
  1. filter_official_sites()  — removes Wikipedia, aggregators, blogs, OTAs
  2. bm25_rank()              — ranks remaining official results by relevance
  3. clean_company_name()     — strips platform suffixes from search titles
"""

import re
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from rank_bm25 import BM25Okapi


# ── Aggregator / Non-Official Domain Blacklist ────────────────────────────────
# Any URL whose registered domain matches one of these is NOT an official site.
# Rule: if Google shows it, but it's not the hotel's own domain → discard.

AGGREGATOR_DOMAINS: frozenset[str] = frozenset({
    # Encyclopedias
    "wikipedia.org", "wikimedia.org", "wikidata.org",
    # OTA / Booking platforms
    "booking.com", "hotels.com", "expedia.com", "expedia.fr",
    "kayak.com", "trivago.com", "trivago.fr",
    "agoda.com", "orbitz.com", "travelocity.com",
    "airbnb.com", "vrbo.com",
    # Review platforms
    "tripadvisor.com", "tripadvisor.fr", "tripadvisor.co.uk",
    "yelp.com", "yelp.fr", "trustpilot.com",
    "thefork.com", "lafourchette.com", "zomato.com",
    # Travel media / blogs / guides
    "lonelyplanet.com", "timeout.com", "timeout.fr",
    "fodors.com", "frommers.com",
    "cntraveler.com", "travelandleisure.com", "cntraveller.com",
    "theguardian.com", "telegraph.co.uk", "independent.co.uk",
    "nytimes.com", "forbes.com", "vogue.com", "architecturaldigest.com",
    # Regional tourism portals
    "explorenicecotedazur.com", "explorefrance.com", "visitmonaco.com",
    "cotedazur-tourisme.com", "nicetourisme.com", "cannes-destination.fr",
    "monaco-tourisme.com", "saint-tropez-tourisme.fr",
    # Aggregator / concierge resellers
    "privateupgrades.com", "tablet.com", "mrandmrssmith.com",
    "smallluxuryhotels.com", "designhotels.com",
    "week-ends-de-reve.com", "charmeandtradition.com",
    "relaischateaux.com",  # keep if direct hotel page needed, but it's aggregator
    # Business directories
    "pagesjaunes.fr", "annuaire.com", "118000.fr", "kompass.com",
    "societe.com", "infogreffe.fr", "sirene.fr",
    # Social / search
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "pinterest.com",
    "google.com", "google.fr", "bing.com", "maps.google.com",
})

# Title keywords that always indicate non-official pages
NON_OFFICIAL_TITLE_PATTERNS: list[str] = [
    "wikipedia", "tripadvisor", "trip advisor", "booking.com",
    "hotels.com", "expedia", "airbnb", "yelp", "trustpilot",
    "thefork", "lafourchette", "lonelyplanet", "timeout",
    "cntraveler", "travelandleisure", "privateupgrades",
    "week-ends de reve", "week-ends-de-reve",
    "guide", "annuaire", "répertoire", "directory",
    "avis clients", "review", "comparatif",
]

# Title suffixes to strip when extracting company name from search title
TITLE_STRIP_SUFFIXES: list[str] = [
    " - Wikipedia", " — Wikipedia", " | Wikipedia",
    " | Booking.com", " on Booking.com",
    " | TripAdvisor", " - TripAdvisor",
    " | Hotels.com", " | Expedia",
    " - Côte d'Azur CVB", " - Nice Côte d'Azur CVB",
    " - Week-ends de Rêve", " - Private Upgrades",
    " with VIP benefits",
    " | Four Seasons",   # keep company name, strip platform
    " | Marriott",
    " | Hilton",
    " | AccorHotels",
    " | Sofitel",
    " | Novotel",
    " - Luxury Hotel",
    " • 5 Star Luxury Hotel • Excellence Riviera",
    " - Excellence Riviera",
]


# ── Official Site Filter ──────────────────────────────────────────────────────

def _registered_domain(url: str) -> str:
    """
    Extract registered domain from URL (e.g. 'en.wikipedia.org' → 'wikipedia.org').

    Args:
        url: Full URL string.

    Returns:
        Registered domain (last 2 parts), lowercase.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        parts = netloc.replace("www.", "").split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return netloc
    except Exception:
        return ""


def _title_is_aggregator(title: str) -> bool:
    """Check if title contains non-official site patterns."""
    title_lower = title.lower()
    return any(pat in title_lower for pat in NON_OFFICIAL_TITLE_PATTERNS)


def filter_official_sites(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter search results to keep only official company websites.

    Removes: Wikipedia, OTAs (Booking/Expedia/Hotels.com),
    review platforms (TripAdvisor/Yelp), travel media/blogs,
    business directories, and social media.

    Rule: if Google shows a hotel phone on a non-hotel page → we skip
    that page and fetch the hotel's own website instead.

    Args:
        results: Raw search result dicts with 'url' and 'title'.

    Returns:
        Filtered list containing only official company pages.
    """
    official: list[dict[str, Any]] = []
    skipped: list[str] = []

    for r in results:
        url   = r.get("url", "")
        title = r.get("title", "")

        domain = _registered_domain(url)

        if domain in AGGREGATOR_DOMAINS:
            skipped.append(f"{domain} ({title[:40]})")
            continue

        if _title_is_aggregator(title):
            skipped.append(f"title-match ({title[:40]})")
            continue

        official.append(r)

    if skipped:
        logger.info(
            f"[RANKER][filter_official_sites] Removed {len(skipped)} aggregators: "
            f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
        )
    logger.info(f"[RANKER][filter_official_sites] {len(results)} → {len(official)} official sites kept")
    return official


# ── Company Name Cleaner ──────────────────────────────────────────────────────

def clean_company_name(title: str) -> str:
    """
    Extract clean company name from a search result title.

    Strips platform suffixes (Wikipedia, TripAdvisor, CVB, etc.)
    and split markers (|, •, ›).

    Examples:
        "Grand-Hôtel du Cap-Ferrat - Wikipedia"  → "Grand-Hôtel du Cap-Ferrat"
        "Hôtel de Paris Monte-Carlo • 5 Star…"   → "Hôtel de Paris Monte-Carlo"
        "Book Hotel X | Monaco with VIP benefits" → "Hotel X"

    Args:
        title: Raw search result title string.

    Returns:
        Cleaned company name.
    """
    name = title

    # Strip known platform suffixes (case-sensitive, longest first)
    for suffix in sorted(TITLE_STRIP_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        if suffix.lower() in name.lower():
            idx = name.lower().find(suffix.lower())
            if idx > 10:  # only strip if company name part is long enough
                name = name[:idx]

    # Split on | › • — and take first meaningful part (longest to avoid "Book X")
    parts = re.split(r"\s*[|›•]\s*", name)
    # Filter parts that look like company names (>5 chars, not generic phrases)
    generic = {"book", "reserve", "find", "search", "compare", "read", "visit"}
    clean_parts = [p.strip() for p in parts if len(p.strip()) > 5 and p.strip().lower().split()[0] not in generic]
    name = clean_parts[0] if clean_parts else parts[0].strip()

    # Remove leading "Book " / "Reserve " / "Find "
    name = re.sub(r"^(Book|Reserve|Find|Visit|Discover|Welcome to)\s+", "", name, flags=re.IGNORECASE)

    return name.strip()


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate results by normalized URL.

    Also deduplicates by registered domain — keeps only 1 result per domain
    to avoid 5 pages from the same site blocking other companies.

    Args:
        results: List of search result dicts with 'url' key.

    Returns:
        Deduplicated list preserving insertion order.
    """
    seen_urls: set[str]    = set()
    seen_domains: set[str] = set()
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

        domain = _registered_domain(url)

        if normalized in seen_urls:
            continue
        if domain in seen_domains:
            continue  # skip additional pages from same domain

        seen_urls.add(normalized)
        seen_domains.add(domain)
        unique.append(result)

    logger.debug(f"[RANKER][deduplicate] {len(results)} → {len(unique)} (url+domain dedup)")
    return unique


# ── BM25 Ranker ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer."""
    return [tok for tok in text.lower().split() if tok]


def _build_corpus_text(result: dict[str, Any]) -> str:
    """Combine title and snippet for BM25 indexing."""
    return f"{result.get('title', '')} {result.get('snippet', '')}"


def bm25_rank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Rank search results by BM25 relevance to the query.

    Adds 'bm25_score' field to each result dict.
    Returns top_k results sorted by score descending.

    Args:
        query: Search query string.
        results: List of result dicts (must have title, snippet).
        top_k: Max results to return.

    Returns:
        Top-k results with 'bm25_score', sorted descending.
    """
    if not results:
        return []

    if len(results) == 1:
        results[0]["bm25_score"] = 1.0
        return results

    corpus_texts     = [_build_corpus_text(r) for r in results]
    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    tokenized_query  = _tokenize(query)

    if not tokenized_query:
        for r in results:
            r["bm25_score"] = 0.0
        return results[:top_k]

    try:
        bm25   = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
    except Exception as e:
        logger.error(f"[RANKER][bm25_rank] BM25 error: {e}", exc_info=True)
        for r in results:
            r["bm25_score"] = 0.0
        return results[:top_k]

    scored = [
        {**result, "bm25_score": float(score)}
        for result, score in zip(results, scores)
    ]
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    top = scored[:top_k]

    logger.info(
        f"[RANKER][bm25_rank] {len(results)} → top {len(top)} "
        f"(best score: {top[0]['bm25_score']:.3f})"
    )
    return top

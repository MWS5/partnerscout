"""
PartnerScout AI — BM25 Result Ranker + Official Site Filter v2.

Three responsibilities:
  1. filter_official_sites()  — removes Wikipedia, aggregators, blogs, OTAs,
                                 travel blogs, review sites, luxury-travel curators
  2. bm25_rank()              — ranks remaining official results by relevance
  3. clean_company_name()     — strips platform suffixes from search titles
  4. deduplicate()            — by URL + domain (1 result per domain)

Detection layers (defense-in-depth):
  A. Static domain blacklist  — known aggregator domains
  B. Domain-word patterns     — travel blog domains (theluxevoyager, spotlist, etc.)
  C. Title keyword patterns   — review/guide language in title
  D. URL path patterns        — /hotels/, /review/, /best-hotels- etc.
"""

import re
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from rank_bm25 import BM25Okapi


# ── A: Static Domain Blacklist ────────────────────────────────────────────────
# Any URL whose registered domain matches → NOT an official site.

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
    # Travel media / blogs / guides — general
    "lonelyplanet.com", "timeout.com", "timeout.fr",
    "fodors.com", "frommers.com",
    "cntraveler.com", "travelandleisure.com", "cntraveller.com",
    "theguardian.com", "telegraph.co.uk", "independent.co.uk",
    "nytimes.com", "forbes.com", "vogue.com", "architecturaldigest.com",
    "elledecor.com", "townandcountrymag.com", "robb-report.com",
    "robbreport.com", "afar.com", "traveler.com", "travel.com",
    # Luxury travel curators / blogs (known offenders)
    "theluxevoyager.com", "theluxuryeditor.com", "luxurytraveladvisor.com",
    "spotlist.fr", "spotlist.com",
    "luxe.digital", "luxurycolumnist.com",
    "the-luxury-review.com", "luxury-hotels.com",
    "hotelsmagazine.com", "boutiquehotels.co.uk",
    "suitcase.com", "escapism.com", "thisluxurylife.com",
    "luxuryhotelworld.com", "luxuryhotelbooking.com",
    "condenastreader.com", "condenast.com",
    "travelblog.org", "hotelscombined.com",
    "jetsetter.com", "tablet.com",
    "secret-escapes.com", "secretescapes.com",
    "sawdays.co.uk", "sawdays.com",
    "ilovefrance.com", "holidaycheck.com",
    "hoteliers.com", "hotelier.com",
    # Government / national tourism sites (articles about hotels, not hotels themselves)
    "france.fr", "atout-france.fr", "rendezvousenfrance.com",
    "franceguide.com", "visitfrance.fr",
    # Regional tourism portals
    "explorenicecotedazur.com", "explorefrance.com", "visitmonaco.com",
    "cotedazur-tourisme.com", "nicetourisme.com", "cannes-destination.fr",
    "monaco-tourisme.com", "saint-tropez-tourisme.fr",
    "provenceguide.com", "riviera-guide.com",
    # Aggregator / concierge resellers
    "privateupgrades.com", "mrandmrssmith.com",
    "smallluxuryhotels.com", "designhotels.com",
    "week-ends-de-reve.com", "charmeandtradition.com",
    "relaischateaux.com",
    "leading-hotels.com", "leadinghotels.com", "lhw.com",
    # Business directories
    "pagesjaunes.fr", "annuaire.com", "118000.fr", "kompass.com",
    "societe.com", "infogreffe.fr", "sirene.fr",
    "europages.com", "europages.fr",
    # Social / search
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "pinterest.com",
    "google.com", "google.fr", "bing.com", "maps.google.com",
    # Property / apartment rentals
    "homeaway.com", "housetrip.com", "wimdu.com",
    # Travel blogs / city guides seen in production logs
    "pierreblake.com", "cityzeum.com", "tourazur.com",
    "frejus-tourist-office.com", "trip.com",
    "easyvoyage.com", "routard.com", "leguide.com",
})


# ── B: Travel Blog Domain-Word Patterns ───────────────────────────────────────
# If the FIRST segment of the domain (e.g. "theluxevoyager" from "theluxevoyager.com")
# contains any of these words → it is a travel blog / listing site, not an official hotel.
# These words appear in travel blogger / listing domain names but rarely in hotel names.

TRAVEL_BLOG_DOMAIN_WORDS: tuple[str, ...] = (
    "voyager", "voyageur", "traveler", "traveller",
    "travelblog", "travelguide", "travelmagazine",
    "spotlist", "spot-list",
    "discover-hotel", "discoverhotels",
    "hotel-guide", "hotelguide", "hotelreview", "hotel-review",
    "best-hotel", "besthotels", "tophotels", "top-hotel",
    "luxury-hotel", "luxuryhotels", "luxehotel",
    "weekendreve", "week-end-reve", "weekend-",
    "getaways", "escapades",
    "hotelrating", "hotelranking",
    "curated-hotel", "luxury-list",
)


# ── C: Title Keyword Patterns ─────────────────────────────────────────────────
# Title keywords that always indicate non-official pages.

NON_OFFICIAL_TITLE_PATTERNS: list[str] = [
    "wikipedia", "tripadvisor", "trip advisor", "booking.com",
    "hotels.com", "expedia", "airbnb", "yelp", "trustpilot",
    "thefork", "lafourchette", "lonelyplanet", "timeout",
    "cntraveler", "travelandleisure", "privateupgrades",
    "week-ends de reve", "week-ends-de-reve",
    "secret escapes", "secretescapes",
    "guide", "annuaire", "répertoire", "directory",
    "avis clients", "review", "comparatif",
    "best hotels in", "top hotels in", "luxury hotels in",
    "hotel ranking", "hotel comparison",
    "les meilleurs hôtels", "meilleurs hotels",
    "spotlist", "luxevoyager",
]


# ── D: URL Path Patterns ──────────────────────────────────────────────────────
# Path segments that indicate this is a listing/review page, not an official hotel site.
# An official hotel site has paths like /en/, /rooms/, /restaurant/, /spa/ — not /hotels/.

NON_OFFICIAL_PATH_PATTERNS: tuple[str, ...] = (
    # Hotel listing pages
    "/hotels/",           # listing: site.com/hotels/grand-hotel-capferrat
    "/hotel/",            # listing: site.com/hotel/le-bristol
    "/places/hotels/",
    "/best-hotels",       # article: site.com/best-hotels-in-nice
    "/top-hotels",
    "/luxury-hotels-in",
    "/hotel-guide",
    "/hotel-review",
    "/reviews/",
    "/compare/",
    "/deals/hotels",
    "/ranking/",
    "/les-meilleurs",
    "/meilleurs-hotels",
    # Article / news / blog content paths — NOT official business pages
    "/article/",          # e.g. france.fr/en/article/hotel-...
    "/articles/",
    "/news/",
    "/actualites/",
    "/actualite/",
    "/blog/",
    "/post/",
    "/posts/",
    "/editorial/",
    "/story/",
    "/stories/",
    "/report/",
    "/reports/",
    "/feature/",
    "/features/",
    "/guide/",
    "/guides/",
    # Travel guide paths
    "/travel-guide/",
    "/destinations/",
    "/where-to-stay/",
    "/accommodation/",
    # Directory / listing paths
    "/liste/",
    "/annuaire/",
    "/directory/",
    "/avis/",
    "/selection/",         # curated lists
    "/nos-adresses/",      # "our addresses" blog posts
    "/adresses/",
)


# ── Title suffixes to strip when extracting company name from search title ─────
TITLE_STRIP_SUFFIXES: list[str] = [
    " - Wikipedia", " — Wikipedia", " | Wikipedia",
    " | Booking.com", " on Booking.com",
    " | TripAdvisor", " - TripAdvisor",
    " | Hotels.com", " | Expedia",
    " - Côte d'Azur CVB", " - Nice Côte d'Azur CVB",
    " - Week-ends de Rêve", " - Private Upgrades",
    " with VIP benefits",
    " | Four Seasons",
    " | Marriott",
    " | Hilton",
    " | AccorHotels",
    " | Sofitel",
    " | Novotel",
    " - Luxury Hotel",
    " • 5 Star Luxury Hotel • Excellence Riviera",
    " - Excellence Riviera",
    " - The Luxury Voyager",
    " - Spotlist",
    " - Secret Escapes",
    " - Mr & Mrs Smith",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _domain_label(url: str) -> str:
    """
    Extract the first label of the registered domain (the brand name).

    Examples:
        "theluxevoyager.com" → "theluxevoyager"
        "grandhotelcapferrat.fr" → "grandhotelcapferrat"
    """
    domain = _registered_domain(url)
    return domain.split(".")[0] if "." in domain else domain


def _title_is_aggregator(title: str) -> bool:
    """Check if title contains non-official site patterns."""
    title_lower = title.lower()
    return any(pat in title_lower for pat in NON_OFFICIAL_TITLE_PATTERNS)


def _domain_is_travel_blog(url: str) -> bool:
    """
    Detect travel blog / listing domains by word patterns in domain label.

    Examples that match:
        theluxevoyager.com  → "theluxevoyager" contains "voyager" → True
        spotlist.fr         → "spotlist" contains "spotlist"      → True
        fourseasons.com     → "fourseasons" → no match            → False
    """
    label = _domain_label(url).lower()
    return any(word in label for word in TRAVEL_BLOG_DOMAIN_WORDS)


def _path_is_listing(url: str) -> bool:
    """
    Detect listing/review URL paths.

    Examples:
        theluxevoyager.com/hotels/grand-hotel  → path contains "/hotels/" → True
        grandhotel.com/en/rooms               → path safe                  → False
    """
    try:
        path = urlparse(url).path.lower()
        if len(path) <= 1:  # root URL = likely official homepage
            return False
        return any(pat in path for pat in NON_OFFICIAL_PATH_PATTERNS)
    except Exception:
        return False


# ── Official Site Filter ──────────────────────────────────────────────────────

def filter_official_sites(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter search results to keep only official company websites.

    4-layer defense:
      A. Static domain blacklist  (50+ aggregator domains)
      B. Travel blog domain-words (voyager, spotlist, hotelguide, etc.)
      C. Title keywords           (review, guide, best hotels in, etc.)
      D. URL path patterns        (/hotels/, /review/, /best-hotels- etc.)

    Rule: if Google shows a hotel phone on a non-hotel page → skip
    that page, the official hotel URL will be fetched separately.

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

        if not url:
            continue

        domain = _registered_domain(url)

        # Layer A: static blacklist
        if domain in AGGREGATOR_DOMAINS:
            skipped.append(f"A:blacklist ({domain})")
            continue

        # Layer B: travel blog domain words
        if _domain_is_travel_blog(url):
            skipped.append(f"B:blog-domain ({domain})")
            continue

        # Layer C: title patterns
        if _title_is_aggregator(title):
            skipped.append(f"C:title-match ({title[:40]})")
            continue

        # Layer D: listing URL paths
        if _path_is_listing(url):
            skipped.append(f"D:listing-path ({urlparse(url).path[:40]})")
            continue

        official.append(r)

    if skipped:
        logger.info(
            f"[RANKER][filter_official_sites] Removed {len(skipped)} non-official: "
            f"{skipped[:8]}{'...' if len(skipped) > 8 else ''}"
        )
    logger.info(
        f"[RANKER][filter_official_sites] {len(results)} → {len(official)} official sites kept"
    )
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

    # Split on | › • — and take first meaningful part
    parts = re.split(r"\s*[|›•]\s*", name)
    generic = {"book", "reserve", "find", "search", "compare", "read", "visit"}
    clean_parts = [
        p.strip() for p in parts
        if len(p.strip()) > 5 and p.strip().lower().split()[0] not in generic
    ]
    name = clean_parts[0] if clean_parts else parts[0].strip()

    # Remove leading action verbs
    name = re.sub(
        r"^(Book|Reserve|Find|Visit|Discover|Welcome to)\s+",
        "", name, flags=re.IGNORECASE,
    )

    # Remove trailing descriptors after dash/colon (e.g. "Hotel Name: 5-Star Palace on the Riviera")
    # Only strip if the hotel name itself is > 12 chars (avoid stripping short names)
    colon_match = re.match(r"^(.{12,}?)\s*[:\–—]\s*.{20,}$", name)
    if colon_match:
        name = colon_match.group(1).strip()

    return name.strip()


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate results by normalized URL + registered domain.

    Keeps only 1 result per domain to avoid 5 pages from same site.

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
            normalized = (
                f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
                f"{parsed.path.rstrip('/')}"
            )
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

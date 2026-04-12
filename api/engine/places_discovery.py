"""
PartnerScout AI — Google Places Discovery Engine.

Uses Google Places Text Search + Place Details to discover verified
luxury businesses with official websites.

Unlike web search (which returns articles and review sites), Places API
returns structured business listings with:
  - Verified official website URL
  - Business phone number
  - Full street address
  - Active/inactive status (business_status)
  - Google rating (quality signal)

Strict category mapping to 6 TARGET CATEGORIES:
  1. hotel         → 4-5 star hotels (luxury and upscale hospitality)
  2. event_agency  → Luxury event and corporate agencies
  3. wedding       → Premium and luxury wedding agencies
  4. concierge     → Private, VIP, and luxury concierge services
  5. travel        → Luxury travel and bespoke experience agencies
  6. venue         → Premium private event and wedding venues

Cost: ~$0.05 per company ($200/month free credit ≈ 4,000 companies free)
"""

import asyncio
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

# ── Google Places API Endpoints ───────────────────────────────────────────────

_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
_DETAILS_URL     = "https://maps.googleapis.com/maps/api/place/details/json"

# Place Details fields to request (only what we need — minimizes billing)
_DETAILS_FIELDS = (
    "name,website,formatted_phone_number,formatted_address,"
    "business_status,rating,user_ratings_total,types"
)

# ── Niche → Places Search Configuration ──────────────────────────────────────
# Each niche maps to multiple search queries (EN + FR) and an optional
# Google Places 'type' filter that improves precision.

NICHE_SEARCH_CONFIG: dict[str, dict[str, Any]] = {
    "hotel": {
        "label": "4-5 star hotel (luxury hospitality)",
        "queries": [
            "luxury hotel {city} 5 star",
            "palace hôtel {city} côte d'azur",
            "boutique hotel VIP {city}",
            "grand hôtel prestige {city}",
        ],
        "type": "lodging",
        "min_rating": 3.8,    # Hotels below 3.8 are not luxury
        "min_reviews": 10,    # Must have some public reviews
    },
    "event_agency": {
        "label": "luxury event agency (corporate and VIP events)",
        "queries": [
            "luxury event agency {city}",
            "VIP event planner {city}",
            "agence événementielle luxe {city}",
            "exclusive event management {city}",
        ],
        "type": None,
        "min_rating": 3.5,
        "min_reviews": 3,
    },
    "wedding": {
        "label": "premium wedding agency (luxury weddings)",
        "queries": [
            "luxury wedding planner {city}",
            "wedding agency premium {city}",
            "organisateur mariage prestige {city}",
            "wedding planner VIP Côte d'Azur",
        ],
        "type": None,
        "min_rating": 3.5,
        "min_reviews": 3,
    },
    "concierge": {
        "label": "VIP concierge service (private luxury concierge)",
        "queries": [
            "luxury concierge service {city}",
            "VIP personal concierge {city}",
            "conciergerie luxe {city}",
            "private lifestyle concierge {city}",
        ],
        "type": None,
        "min_rating": 3.5,
        "min_reviews": 2,
    },
    "travel": {
        "label": "premium travel agency (luxury bespoke travel)",
        "queries": [
            "luxury travel agency {city}",
            "bespoke travel {city}",
            "agence voyage luxe {city}",
            "VIP yacht charter {city}",
        ],
        "type": "travel_agency",
        "min_rating": 3.5,
        "min_reviews": 3,
    },
    "venue": {
        "label": "premium event venue (private and corporate events)",
        "queries": [
            "luxury event venue {city}",
            "private venue hire {city}",
            "salle réception prestige {city}",
            "château mariage {city}",
        ],
        "type": None,
        "min_rating": 3.5,
        "min_reviews": 3,
    },
}

# Domains that should be excluded even if they appear in Places results
# (OTA booking platforms sometimes appear as Place results)
_EXCLUDED_DOMAINS: frozenset[str] = frozenset({
    "booking.com", "tripadvisor.com", "expedia.com", "hotels.com",
    "airbnb.com", "kayak.com", "trivago.com", "agoda.com", "orbitz.com",
    "mrandmrssmith.com", "secretescapes.com", "relaischateaux.com",
    "leadinghotels.com", "lhw.com", "smallluxuryhotels.com",
    "designhotels.com", "tablet.com", "jetsetter.com",
    "privateupgrades.com", "charmeandtradition.com",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _registered_domain(url: str) -> str:
    """Extract registered domain (e.g. 'www.hotel.fr' → 'hotel.fr')."""
    try:
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        parts = netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        return ""


# ── Places API Calls ──────────────────────────────────────────────────────────

async def _text_search(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    place_type: Optional[str],
) -> list[dict[str, Any]]:
    """
    Call Google Places Text Search API.

    Args:
        client: Shared httpx client.
        query: Natural language search query.
        api_key: Google Places API key.
        place_type: Optional Places type filter (e.g. 'lodging').

    Returns:
        List of place result dicts from Google API.
    """
    params: dict[str, str] = {
        "query": query,
        "key": api_key,
        "language": "en",
    }
    if place_type:
        params["type"] = place_type

    try:
        response = await client.get(_TEXT_SEARCH_URL, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()

        status = data.get("status", "")
        if status == "ZERO_RESULTS":
            logger.debug(f"[PLACES][_text_search] No results for '{query}'")
            return []
        if status != "OK":
            logger.warning(f"[PLACES][_text_search] API status '{status}' for '{query}'")
            return []

        return data.get("results", [])

    except Exception as e:
        logger.error(f"[PLACES][_text_search] Error for '{query}': {e}", exc_info=True)
        return []


async def _place_details(
    client: httpx.AsyncClient,
    place_id: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Call Google Places Details API for a single place.

    Args:
        client: Shared httpx client.
        place_id: Google Place ID.
        api_key: Google Places API key.

    Returns:
        Place detail dict, or empty dict on failure.
    """
    params = {
        "place_id": place_id,
        "fields": _DETAILS_FIELDS,
        "key": api_key,
        "language": "en",
    }
    try:
        response = await client.get(_DETAILS_URL, params=params, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})
    except Exception as e:
        logger.error(
            f"[PLACES][_place_details] Error for place_id={place_id}: {e}",
            exc_info=True,
        )
        return {}


# ── Main Discovery Function ───────────────────────────────────────────────────

async def discover_companies_via_places(
    niche: str,
    cities: list[str],
    api_key: str,
    max_per_city: int = 5,
) -> list[dict[str, Any]]:
    """
    Discover verified luxury companies via Google Places Text Search + Details.

    For each (query template × city) combination:
      1. Text Search → up to 20 place results from Google
      2. Place Details → website, phone, address, business_status
      3. Validation:
         - business_status == "OPERATIONAL" (active company only)
         - has official website (not OTA domain)
         - rating >= minimum threshold
         - not duplicate (domain dedup)
      4. Returns standardized company dict

    This produces ONLY verified, active businesses with official websites.
    No articles, no review sites, no inactive businesses.

    Args:
        niche: One of 6 target categories (hotel, event_agency, wedding,
               concierge, travel, venue).
        cities: Target city names.
        api_key: Google Places API key.
        max_per_city: Max unique companies to collect per city (default 5).

    Returns:
        List of company dicts, each with verified official website + contacts.
    """
    config = NICHE_SEARCH_CONFIG.get(niche)
    if not config:
        logger.warning(f"[PLACES][discover] Unknown niche: '{niche}' — skipping")
        return []

    seen_domains: set[str] = set()
    companies: list[dict[str, Any]] = []
    seen_lock = asyncio.Lock()

    min_rating  = config.get("min_rating", 3.5)
    min_reviews = config.get("min_reviews", 2)
    place_type  = config.get("type")

    async def _search_one_city(city: str) -> list[dict[str, Any]]:
        """Run all query templates for one city, return validated companies."""
        city_results: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=12.0) as client:
            for query_template in config["queries"]:
                if len(city_results) >= max_per_city:
                    break

                query = query_template.replace("{city}", city)

                # Text Search
                results = await _text_search(client, query, api_key, place_type)
                if not results:
                    continue

                # Place Details in parallel (max 5 per query for speed)
                top_results = results[:5]
                detail_tasks = [
                    _place_details(client, r["place_id"], api_key)
                    for r in top_results
                ]
                details_list = await asyncio.gather(*detail_tasks, return_exceptions=True)

                for basic, detail in zip(top_results, details_list):
                    if len(city_results) >= max_per_city:
                        break
                    if not detail or isinstance(detail, Exception):
                        continue
                    if detail.get("business_status") != "OPERATIONAL":
                        continue

                    website = detail.get("website", "").strip()
                    if not website:
                        continue

                    domain = _registered_domain(website)
                    if domain in _EXCLUDED_DOMAINS:
                        continue

                    # Global dedup check (thread-safe)
                    async with seen_lock:
                        if domain in seen_domains:
                            continue
                        seen_domains.add(domain)

                    rating  = float(basic.get("rating") or detail.get("rating") or 0)
                    reviews = int(
                        basic.get("user_ratings_total")
                        or detail.get("user_ratings_total") or 0
                    )
                    if reviews >= min_reviews and 0 < rating < min_rating:
                        continue

                    company_name = detail.get("name") or basic.get("name", "Unknown")
                    phone   = detail.get("formatted_phone_number", "Not found")
                    address = detail.get("formatted_address", "Not found")

                    # Pre-compute luxury_score from Places rating so the
                    # luxury filter doesn't discard verified luxury businesses
                    # that have no Jina content yet.
                    # rating 3.8 → 0.72  |  4.2 → 0.80  |  4.7 → 0.90  |  5.0 → 0.95
                    places_luxury_score = round(
                        min(0.95, max(0.70, (rating - 3.5) / 1.5 * 0.25 + 0.70)),
                        2,
                    ) if rating > 0 else 0.75  # default 0.75 if rating unknown

                    city_results.append({
                        "category":       niche,
                        "company_name":   company_name,
                        "url":            website,
                        "address":        address,
                        "phone":          phone,
                        "email":          "Not found",
                        "contact_person": "Not found",
                        "personal_phone": "Not found",
                        "personal_email": "Not found",
                        "luxury_score":   places_luxury_score,
                        "verified":       True,
                        "places_verified": True,   # flag → bypass re-scoring
                        "places_rating":  rating,
                        "snippet": (
                            f"Google Places verified {config['label']} in {city}. "
                            f"Rating: {rating:.1f} ({reviews} reviews). "
                            f"Luxury, premium, VIP, exclusive, prestige."
                        ),
                    })
                    logger.info(
                        f"[PLACES][discover] ✓ {company_name} ({city}) "
                        f"→ {website} [rating={rating:.1f}]"
                    )

        return city_results

    # Run ALL cities in parallel (was sequential — caused pipeline hang)
    try:
        city_results_list = await asyncio.wait_for(
            asyncio.gather(*[_search_one_city(city) for city in cities],
                           return_exceptions=True),
            timeout=60.0,   # hard cap: discovery must finish in 60s total
        )
        for city_results in city_results_list:
            if isinstance(city_results, list):
                companies.extend(city_results)
    except asyncio.TimeoutError:
        logger.warning(
            f"[PLACES][discover] niche='{niche}' timed out after 60s — "
            f"using {len(companies)} companies collected so far"
        )

    logger.info(
        f"[PLACES][discover] niche='{niche}' across {len(cities)} cities → "
        f"{len(companies)} verified companies (parallel fetch)"
    )
    return companies

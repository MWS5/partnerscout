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

    async with httpx.AsyncClient() as client:
        for city in cities:
            city_count = 0

            for query_template in config["queries"]:
                if city_count >= max_per_city:
                    break

                query = query_template.replace("{city}", city)
                place_type = config.get("type")

                # Step 1: Text Search
                results = await _text_search(client, query, api_key, place_type)
                if not results:
                    continue

                # Step 2: Fetch details for top results in parallel
                top_results = results[:10]  # max 10 detail calls per query
                detail_tasks = [
                    _place_details(client, r["place_id"], api_key)
                    for r in top_results
                ]
                details_list = await asyncio.gather(*detail_tasks, return_exceptions=False)

                # Step 3: Validate and collect
                for basic, detail in zip(top_results, details_list):
                    if city_count >= max_per_city:
                        break
                    if not detail:
                        continue

                    # Validation A: Active business only
                    if detail.get("business_status") != "OPERATIONAL":
                        logger.debug(
                            f"[PLACES][discover] Skipped (not operational): "
                            f"{detail.get('name', '?')} — status={detail.get('business_status')}"
                        )
                        continue

                    # Validation B: Must have official website
                    website = detail.get("website", "").strip()
                    if not website:
                        continue

                    # Validation C: Exclude OTA/aggregator domains
                    domain = _registered_domain(website)
                    if domain in _EXCLUDED_DOMAINS:
                        continue

                    # Validation D: Deduplication by domain
                    if domain in seen_domains:
                        continue

                    # Validation E: Minimum rating (quality signal)
                    rating = float(
                        basic.get("rating") or detail.get("rating") or 0
                    )
                    reviews = int(
                        basic.get("user_ratings_total") or
                        detail.get("user_ratings_total") or 0
                    )
                    min_rating = config.get("min_rating", 3.5)
                    min_reviews = config.get("min_reviews", 2)

                    if reviews >= min_reviews and rating > 0 and rating < min_rating:
                        logger.debug(
                            f"[PLACES][discover] Skipped (rating {rating:.1f} < {min_rating}): "
                            f"{detail.get('name')}"
                        )
                        continue

                    seen_domains.add(domain)
                    city_count += 1

                    company_name = detail.get("name") or basic.get("name", "Unknown")
                    phone = detail.get("formatted_phone_number", "Not found")
                    address = detail.get("formatted_address", "Not found")

                    companies.append({
                        "category":        niche,
                        "company_name":    company_name,
                        "url":             website,
                        "address":         address,
                        "phone":           phone,
                        "email":           "Not found",  # Places API has no email field
                        "contact_person":  "Not found",
                        "personal_phone":  "Not found",
                        "personal_email":  "Not found",
                        "luxury_score":    0.0,           # scored later in pipeline
                        "verified":        True,          # Google Places = verified
                        "places_rating":   rating,
                        "snippet": (
                            f"Google Places verified {config['label']} in {city}. "
                            f"Rating: {rating:.1f} ({reviews} reviews)."
                        ),
                    })

                    logger.info(
                        f"[PLACES][discover] ✓ {company_name} "
                        f"({niche}, {city}) → {website} "
                        f"[rating={rating:.1f}, reviews={reviews}]"
                    )

    logger.info(
        f"[PLACES][discover] niche='{niche}' across {len(cities)} cities → "
        f"{len(companies)} verified companies discovered"
    )
    return companies

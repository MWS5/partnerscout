"""
Query Matrix Generator for PartnerScout AI.

Generates multilingual search queries: category × city × language.
Each niche has EN + FR query templates with {city} placeholder.
Supports luxury qualifier injection for segment-aware queries.

Usage:
    queries = generate_queries(
        niches=["hotel", "event_agency"],
        regions=["Nice", "Monaco"],
        segment="luxury",
    )
"""

import itertools
from typing import Literal

from loguru import logger


# ── Category Query Templates ──────────────────────────────────────────────────

CATEGORIES: dict[str, dict[str, list[str]]] = {
    "hotel": {
        "en": [
            "luxury hotel {city} 5 star",
            "palace hotel {city} Côte d'Azur",
            "boutique hotel {city} VIP",
            "5 star hotel {city} French Riviera contact",
        ],
        "fr": [
            "hôtel luxe {city} 5 étoiles",
            "palace {city} côte d'azur contact",
            "hôtel prestige {city}",
            "hôtel boutique {city} VIP directeur",
        ],
    },
    "event_agency": {
        "en": [
            "luxury event agency {city}",
            "VIP event planner {city} French Riviera",
            "exclusive event management {city} contact",
            "premium event organizer {city}",
        ],
        "fr": [
            "agence événementielle luxe {city}",
            "organisateur événements prestige {city}",
            "agence événements VIP {city} contact",
            "planificateur événements haut de gamme {city}",
        ],
    },
    "wedding": {
        "en": [
            "luxury wedding planner {city}",
            "VIP wedding organizer {city} French Riviera",
            "exclusive wedding venue {city} contact",
            "high-end wedding coordinator {city}",
        ],
        "fr": [
            "organisateur mariage luxe {city}",
            "wedding planner prestige {city}",
            "agence mariage haut de gamme {city} contact",
            "coordinateur mariage VIP {city}",
        ],
    },
    "concierge": {
        "en": [
            "luxury concierge service {city}",
            "VIP lifestyle concierge {city}",
            "personal concierge {city} French Riviera contact",
            "exclusive concierge company {city}",
        ],
        "fr": [
            "service conciergerie luxe {city}",
            "conciergerie VIP {city} contact",
            "conciergerie prestige {city} côte d'azur",
            "conciergerie personnelle {city}",
        ],
    },
    "travel": {
        "en": [
            "luxury travel agency {city}",
            "VIP yacht charter {city} French Riviera",
            "private jet travel {city} contact",
            "exclusive travel planner {city}",
        ],
        "fr": [
            "agence voyage luxe {city}",
            "location yacht VIP {city}",
            "voyage privé prestige {city} contact",
            "agence voyage haut de gamme {city}",
        ],
    },
    "venue": {
        "en": [
            "luxury event venue {city}",
            "exclusive venue hire {city} French Riviera",
            "VIP venue rental {city} contact",
            "premium private venue {city}",
        ],
        "fr": [
            "salle événements luxe {city}",
            "lieu réception prestige {city}",
            "location salle VIP {city} contact",
            "venue privatisation haut de gamme {city}",
        ],
    },
}

# ── Geography ─────────────────────────────────────────────────────────────────

CITIES: list[str] = [
    "Nice",
    "Cannes",
    "Monaco",
    "Antibes",
    "Saint-Tropez",
    "Menton",
    "Cap-Ferrat",
    "Juan-les-Pins",
]

# ── Luxury Qualifier Injection ────────────────────────────────────────────────

LUXURY_QUALIFIERS: list[str] = [
    "luxury",
    "luxe",
    "VIP",
    "prestige",
    "exclusive",
    "exclusif",
    "bespoke",
    "premium",
    "haut de gamme",
    "5 star",
    "5 étoiles",
    "palace",
    "private",
    "privé",
]

SegmentType = Literal["luxury", "premium", "general"]


def _get_templates_for_niche(niche: str) -> list[str]:
    """
    Return all EN + FR templates for a given niche.

    Args:
        niche: One of the 6 supported niche keys.

    Returns:
        Flat list of query template strings.
    """
    cat = CATEGORIES.get(niche)
    if not cat:
        logger.warning(f"[QUERY_MATRIX][_get_templates_for_niche] Unknown niche: {niche}")
        return []
    return cat.get("en", []) + cat.get("fr", [])


def _apply_city(template: str, city: str) -> str:
    """
    Substitute {city} placeholder in a query template.

    Args:
        template: Query string with {city} placeholder.
        city: Target city name.

    Returns:
        Formatted query string.
    """
    return template.replace("{city}", city)


def _add_luxury_qualifier(query: str, segment: SegmentType) -> list[str]:
    """
    Return query as-is for luxury; add qualifier variants for premium/general.

    For luxury segment, templates already contain luxury terms.
    For lower segments, we inject qualifiers to bias toward quality results.

    Args:
        query: Base search query.
        segment: Market segment filter.

    Returns:
        List of query variants (1 for luxury, multiple for others).
    """
    if segment == "luxury":
        return [query]
    qualifier = "luxury" if segment == "premium" else ""
    if qualifier and qualifier not in query.lower():
        return [query, f"{qualifier} {query}"]
    return [query]


def deduplicate_queries(queries: list[str]) -> list[str]:
    """
    Remove duplicate and near-duplicate queries (case-insensitive).

    Args:
        queries: Raw list of query strings.

    Returns:
        Deduplicated list preserving first-occurrence order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        normalized = q.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(q)
    return result


def generate_queries(
    niches: list[str],
    regions: list[str],
    segment: SegmentType = "luxury",
    max_queries: int = 40,
) -> list[str]:
    """
    Generate all search query combinations for given niches and regions.

    Algorithm:
        1. For each niche × city pair, apply all templates.
        2. Optionally inject luxury qualifiers based on segment.
        3. Deduplicate (case-insensitive).
        4. Truncate to max_queries.

    Args:
        niches: List of target niche keys (e.g. ["hotel", "wedding"]).
        regions: List of target city/region names.
        segment: Market segment — affects qualifier injection.
        max_queries: Hard cap on output size (default 40).

    Returns:
        Deduplicated list of search query strings, capped at max_queries.
    """
    raw_queries: list[str] = []

    cities = regions if regions else CITIES

    for niche, city in itertools.product(niches, cities):
        templates = _get_templates_for_niche(niche)
        for template in templates:
            base_query = _apply_city(template, city)
            variants = _add_luxury_qualifier(base_query, segment)
            raw_queries.extend(variants)

    unique = deduplicate_queries(raw_queries)
    final = unique[:max_queries]

    logger.info(
        f"[QUERY_MATRIX][generate_queries] "
        f"Generated {len(final)} queries "
        f"(niches={niches}, regions={cities}, segment={segment})"
    )
    return final

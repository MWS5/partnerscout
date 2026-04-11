"""
PartnerScout AI — Result Exporter.

Converts enriched company records to CSV and JSON formats.
Provides trial blurring logic to show value while protecting
full contact data behind a paywall.

Trial blurring rules:
  - email: show domain, hide local part (m***@domain.com)
  - contact_person: keep job title only, hide name
  - personal_phone: locked message
  - personal_email: locked message
  - Truncated to first 10 companies
"""

import csv
import io
import json
from typing import Any

from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

TRIAL_LIMIT = 10
LOCKED_MESSAGE = "🔒 Unlock with full access"

CSV_FIELDNAMES = [
    "category",
    "company_name",
    "website",
    "address",
    "phone",
    "email",
    "contact_person",
    "personal_phone",
    "personal_email",
    "luxury_score",
    "verified",
]


# ── Email Blurring ────────────────────────────────────────────────────────────

def _blur_email(email: str) -> str:
    """
    Blur email address for trial: show domain, hide local part.

    Examples:
        "jean.dupont@hotel-negresco.com" → "j***@hotel-negresco.com"
        "info@palace.fr" → "i***@palace.fr"
        "Not found" → "Not found"

    Args:
        email: Raw email string.

    Returns:
        Blurred email string.
    """
    if not email or email == "Not found":
        return email

    if "@" not in email:
        return "***@***"

    local, domain = email.rsplit("@", 1)
    if len(local) >= 1:
        blurred_local = local[0] + "***"
    else:
        blurred_local = "***"

    return f"{blurred_local}@{domain}"


def _extract_title_only(contact_person: str) -> str:
    """
    Extract job title from contact_person string, hiding name.

    Assumes format "Name, Title" or "Name Title".
    Falls back to "Senior Manager" if no title detectable.

    Args:
        contact_person: Raw contact person string.

    Returns:
        Title-only string.
    """
    if not contact_person or contact_person == "Not found":
        return "Senior Manager"

    # Try comma-separated "Name, Title"
    if "," in contact_person:
        parts = contact_person.split(",", 1)
        title = parts[1].strip()
        return title if title else "Senior Manager"

    # Try to detect known title keywords
    title_keywords = [
        "Director", "Manager", "CEO", "COO", "CMO", "Partner",
        "Owner", "Founder", "Head", "Chief", "Coordinator",
        "Directeur", "Gérant", "Responsable", "Président",
    ]
    for keyword in title_keywords:
        if keyword.lower() in contact_person.lower():
            return keyword

    return "Senior Manager"


# ── Core Export Functions ─────────────────────────────────────────────────────

def to_csv(companies: list[dict[str, Any]]) -> str:
    """
    Serialize company records to CSV string.

    Args:
        companies: List of company dicts with standard fields.

    Returns:
        CSV-formatted string with header row.
    """
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_FIELDNAMES,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for company in companies:
        row = {field: company.get(field, "") for field in CSV_FIELDNAMES}
        writer.writerow(row)

    csv_str = output.getvalue()
    logger.info(f"[EXPORTER][to_csv] Exported {len(companies)} companies to CSV")
    return csv_str


def to_json(companies: list[dict[str, Any]]) -> str:
    """
    Serialize company records to formatted JSON string.

    Args:
        companies: List of company dicts.

    Returns:
        Pretty-printed JSON string.
    """
    output = json.dumps(companies, indent=2, ensure_ascii=False, default=str)
    logger.info(f"[EXPORTER][to_json] Exported {len(companies)} companies to JSON")
    return output


def blur_for_trial(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply trial-mode blurring to top 10 company records.

    Blurring rules:
      - email → first_char***@domain.com
      - contact_person → title only (name hidden)
      - personal_phone → LOCKED_MESSAGE
      - personal_email → LOCKED_MESSAGE
      - Truncated to first TRIAL_LIMIT (10) companies

    Args:
        companies: Full list of enriched company dicts.

    Returns:
        List of up to 10 blurred company dicts.
    """
    trial_companies = companies[:TRIAL_LIMIT]
    blurred: list[dict[str, Any]] = []

    for company in trial_companies:
        blurred_record = dict(company)

        # Blur general email
        raw_email = company.get("email", "Not found")
        blurred_record["email"] = _blur_email(raw_email)

        # Extract title from contact person, hide name
        raw_contact = company.get("contact_person", "Not found")
        blurred_record["contact_person"] = _extract_title_only(raw_contact)

        # Lock direct contact details
        blurred_record["personal_phone"] = LOCKED_MESSAGE
        blurred_record["personal_email"] = LOCKED_MESSAGE

        blurred.append(blurred_record)

    logger.info(
        f"[EXPORTER][blur_for_trial] Blurred {len(blurred)} records "
        f"(from {len(companies)} total)"
    )
    return blurred

"""
PartnerScout AI — Result Models.

Defines Pydantic models for company records, trial-blurred records,
and the SearchResult wrapper returned to clients.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CompanyRecord(BaseModel):
    """
    Full enriched company record with all contact data.

    Only returned for paid orders. All 9 base fields plus
    luxury_score and verified flag from the validation pipeline.
    """

    category: str = Field(..., description="Business niche/category.")
    company_name: str = Field(..., description="Official company name.")
    website: Optional[str] = Field(default=None, description="Company website URL.")
    address: Optional[str] = Field(default=None, description="Physical address.")
    phone: Optional[str] = Field(default=None, description="Main company phone.")
    email: Optional[str] = Field(default=None, description="General contact email.")
    contact_person: Optional[str] = Field(
        default=None,
        description="Name and title of key contact person.",
    )
    personal_phone: Optional[str] = Field(
        default=None,
        description="Direct phone of contact person.",
    )
    personal_email: Optional[str] = Field(
        default=None,
        description="Direct email of contact person.",
    )
    luxury_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM-validated luxury confidence score (0=mass market, 1=ultra-luxury).",
    )
    verified: bool = Field(
        default=False,
        description="True if data was cross-validated from multiple sources.",
    )


class TrialCompanyRecord(BaseModel):
    """
    Blurred company record for trial/preview mode.

    Contact details are obfuscated to incentivize conversion.
    Email shows domain only; personal contacts are locked.
    """

    category: str = Field(..., description="Business niche/category.")
    company_name: str = Field(..., description="Official company name.")
    website: Optional[str] = Field(default=None, description="Company website URL.")
    address: Optional[str] = Field(default=None, description="Physical address.")
    phone: Optional[str] = Field(default=None, description="Main company phone.")
    email: Optional[str] = Field(
        default=None,
        description="Partially blurred email: m***@domain.com",
    )
    contact_person: Optional[str] = Field(
        default=None,
        description="Role/title only — name hidden in trial.",
    )
    personal_phone: str = Field(
        default="🔒 Unlock with full access",
        description="Locked in trial mode.",
    )
    personal_email: str = Field(
        default="🔒 Unlock with full access",
        description="Locked in trial mode.",
    )
    luxury_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM luxury confidence score.",
    )


class SearchResult(BaseModel):
    """
    Wrapper for a completed lead generation result set.

    Returned by export and status endpoints upon completion.
    Contains either CompanyRecord (paid) or TrialCompanyRecord (trial) list.
    """

    order_id: UUID = Field(..., description="Parent order UUID.")
    companies: list[CompanyRecord | TrialCompanyRecord] = Field(
        default_factory=list,
        description="List of enriched company records.",
    )
    total_found: int = Field(
        default=0,
        description="Total companies found before trial truncation.",
    )
    generated_at: datetime = Field(
        ...,
        description="Timestamp when result set was finalized (UTC).",
    )
    is_trial: bool = Field(
        default=False,
        description="Indicates if this is a blurred trial result.",
    )

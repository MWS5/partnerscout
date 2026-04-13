"""
PartnerScout AI — Order Models.

Defines all Pydantic models for order creation, status tracking,
and full DB record representation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


class NicheEnum(str, Enum):
    """
    Active niches for PartnerScout demo.

    Only two niches are exposed for client presentation:
      - hotel        → 4-5 star hotels & event venues (HOTELS & EVENTS)
      - event_agency → luxury event and corporate agencies (AGENCIES)
    """

    hotel = "hotel"
    event_agency = "event_agency"


# Human-readable display labels for niches
NICHE_LABELS: dict[str, str] = {
    "hotel":        "Hotels & Events",
    "event_agency": "Agencies",
}


class SegmentEnum(str, Enum):
    """Market segment filter for result quality."""

    luxury = "luxury"
    premium = "premium"
    general = "general"


class OrderStatusEnum(str, Enum):
    """Lifecycle states of a PartnerScout order."""

    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class OrderCreate(BaseModel):
    """
    Payload to create a new lead generation order.

    Fields:
        email: Client email for result delivery and notifications.
        niches: One or more target business categories.
        regions: List of city/region names to search within.
        segment: Minimum market segment threshold.
        count_target: Desired number of final leads (ignored for trial).
        is_trial: If True, returns 10 blurred results without payment.
    """

    email: EmailStr = Field(..., description="Client email address.")
    niches: list[NicheEnum] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Target niches. At least one required.",
    )
    regions: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Target cities or regions (e.g. ['Nice', 'Monaco']).",
    )
    segment: SegmentEnum = Field(
        default=SegmentEnum.luxury,
        description="Minimum market segment for filtering.",
    )
    count_target: int = Field(
        default=15,
        ge=10,
        le=20,
        description="Target number of strictly validated leads (10–20). Demo mode.",
    )
    is_trial: bool = Field(
        default=False,
        description="Trial mode: top 10 results with blurred contact details.",
    )

    @field_validator("regions")
    @classmethod
    def strip_regions(cls, v: list[str]) -> list[str]:
        """Strip whitespace from region names."""
        return [r.strip() for r in v if r.strip()]


class OrderStatus(BaseModel):
    """
    Public-facing order status response.

    Returned by GET /api/v1/orders/{order_id}.
    """

    id: UUID = Field(..., description="Order UUID.")
    email: str = Field(..., description="Client email.")
    status: OrderStatusEnum = Field(..., description="Current pipeline status.")
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Pipeline completion percentage (0–100).",
    )
    result_url: Optional[str] = Field(
        default=None,
        description="Download URL when status is 'done'.",
    )
    created_at: datetime = Field(..., description="Order creation timestamp (UTC).")
    error_msg: Optional[str] = Field(
        default=None,
        description="Human-readable error description when status is 'failed'.",
    )


class OrderDB(BaseModel):
    """
    Full database record for a PartnerScout order.

    Maps directly to the ps_orders table schema.
    """

    id: UUID
    email: str
    niches: list[str]
    regions: list[str]
    segment: str
    count_target: int
    is_trial: bool
    status: str
    progress: int
    stripe_payment_id: Optional[str] = None
    result_url: Optional[str] = None
    error_msg: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

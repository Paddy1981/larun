"""
User Schemas

Pydantic models for user profile and subscription validation.

Agent: BETA
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SubscriptionInfo(BaseModel):
    """Subscription information in user profile."""

    plan: Optional[str] = None  # hobbyist_monthly, hobbyist_annual
    status: Optional[str] = None  # active, canceled, past_due
    current_period_end: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """User profile response."""

    id: int
    email: str
    name: Optional[str] = None
    created_at: datetime
    subscription: Optional[SubscriptionInfo] = None

    class Config:
        from_attributes = True


class UsageResponse(BaseModel):
    """Usage statistics response."""

    analyses_this_month: int
    analyses_limit: int
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class UpdateProfileRequest(BaseModel):
    """Request body for profile update."""

    name: Optional[str] = Field(None, max_length=255)


class UpdateProfileResponse(BaseModel):
    """Response for profile update."""

    user: UserProfile
    message: str = "Profile updated successfully"


class CreateCheckoutRequest(BaseModel):
    """Request body for creating Stripe checkout session."""

    plan: str = Field(
        ...,
        pattern=r"^(hobbyist_monthly|hobbyist_annual)$",
        description="Subscription plan",
    )
    success_url: str = Field(..., description="URL to redirect on success")
    cancel_url: str = Field(..., description="URL to redirect on cancel")


class CreateCheckoutResponse(BaseModel):
    """Response for checkout session creation."""

    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    """Response for Stripe customer portal."""

    portal_url: str


class SubscriptionResponse(BaseModel):
    """Full subscription details response."""

    id: int
    plan: str
    status: str
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    analyses_this_period: int
    analyses_remaining: int

    class Config:
        from_attributes = True

"""
Subscription Model

SQLAlchemy model for user subscription plans and billing.

Agent: BETA
"""

from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, DateTime, Integer, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.models.database import Base

if TYPE_CHECKING:
    from src.api.models.user import User


class SubscriptionPlan(str, Enum):
    """Available subscription plans."""

    HOBBYIST_MONTHLY = "hobbyist_monthly"
    HOBBYIST_ANNUAL = "hobbyist_annual"


class SubscriptionStatus(str, Enum):
    """Subscription status states."""

    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


# Plan limits - analyses per month
PLAN_LIMITS = {
    SubscriptionPlan.HOBBYIST_MONTHLY: 25,
    SubscriptionPlan.HOBBYIST_ANNUAL: 25,
}


class Subscription(Base):
    """
    User subscription model.

    Tracks Stripe subscription status and usage limits.
    Managed primarily through Stripe webhooks (DELTA handles).
    """

    __tablename__ = "subscriptions"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign key to user (one-to-one relationship)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # Stripe references
    stripe_subscription_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    stripe_price_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # Subscription details
    plan: Mapped[SubscriptionPlan] = mapped_column(
        String(50),
        nullable=False,
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        String(50),
        default=SubscriptionStatus.ACTIVE,
        nullable=False,
        index=True,
    )

    # Billing period
    current_period_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    current_period_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Usage tracking for current period
    analyses_this_period: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    canceled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subscription")

    def __repr__(self) -> str:
        return f"<Subscription(id={self.id}, plan={self.plan}, status={self.status})>"

    @property
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status in (SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING)

    @property
    def analysis_limit(self) -> int:
        """Get the monthly analysis limit for this plan."""
        return PLAN_LIMITS.get(self.plan, 0)

    @property
    def analyses_remaining(self) -> int:
        """Get remaining analyses for current period."""
        return max(0, self.analysis_limit - self.analyses_this_period)

    @property
    def can_analyze(self) -> bool:
        """Check if user can perform more analyses."""
        return self.is_active and self.analyses_remaining > 0

    def increment_usage(self) -> bool:
        """
        Increment usage counter.

        Returns True if successful, False if limit exceeded.
        """
        if not self.can_analyze:
            return False
        self.analyses_this_period += 1
        return True

    def reset_usage(self) -> None:
        """Reset usage counter for new billing period."""
        self.analyses_this_period = 0

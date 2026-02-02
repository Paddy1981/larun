"""
Subscription Routes

Endpoints for subscription management and Stripe integration.

Agent: BETA
Note: Stripe integration handled by DELTA. These are stub endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import settings
from src.api.models.database import get_db
from src.api.models.user import User
from src.api.models.subscription import Subscription, SubscriptionPlan, SubscriptionStatus
from src.api.schemas.user import (
    CreateCheckoutRequest,
    CreateCheckoutResponse,
    PortalResponse,
    SubscriptionResponse,
)
from src.api.dependencies import get_current_user, get_user_subscription

router = APIRouter()


@router.post(
    "/create-checkout",
    response_model=CreateCheckoutResponse,
    summary="Create checkout session",
    description="Create a Stripe checkout session for subscription.",
)
async def create_checkout(
    request: CreateCheckoutRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CreateCheckoutResponse:
    """
    Create Stripe checkout session.

    - **plan**: Subscription plan (hobbyist_monthly or hobbyist_annual)
    - **success_url**: URL to redirect after successful payment
    - **cancel_url**: URL to redirect if user cancels

    Returns checkout URL to redirect user to Stripe.

    NOTE: Full Stripe integration implemented by DELTA.
    This is a stub that returns a mock checkout URL.
    """
    # Check if user already has active subscription
    if current_user.subscription and current_user.subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "already_subscribed",
                    "message": "User already has an active subscription. Use the customer portal to modify.",
                }
            },
        )

    # Validate plan
    if request.plan not in ["hobbyist_monthly", "hobbyist_annual"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_plan",
                    "message": f"Invalid plan: {request.plan}",
                }
            },
        )

    # TODO: Create actual Stripe checkout session (DELTA handles)
    # For now, return mock response

    if settings.STRIPE_SECRET_KEY:
        # When Stripe is configured, create real checkout session
        # This will be implemented when DELTA provides Stripe integration
        pass

    # Mock response for development
    mock_session_id = f"cs_mock_{current_user.id}_{request.plan}"
    mock_checkout_url = f"https://checkout.stripe.com/mock/{mock_session_id}"

    return CreateCheckoutResponse(
        checkout_url=mock_checkout_url,
        session_id=mock_session_id,
    )


@router.get(
    "/portal",
    response_model=PortalResponse,
    summary="Get customer portal URL",
    description="Get Stripe customer portal URL for subscription management.",
)
async def get_portal(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortalResponse:
    """
    Get Stripe customer portal URL.

    Returns URL to redirect user to manage their subscription
    (update payment method, cancel, view invoices).

    NOTE: Full Stripe integration implemented by DELTA.
    This is a stub that returns a mock portal URL.
    """
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "no_customer",
                    "message": "No Stripe customer found. Subscribe first.",
                }
            },
        )

    # TODO: Create actual Stripe portal session (DELTA handles)
    # For now, return mock response

    if settings.STRIPE_SECRET_KEY:
        # When Stripe is configured, create real portal session
        pass

    # Mock response for development
    mock_portal_url = f"https://billing.stripe.com/mock/{current_user.stripe_customer_id}"

    return PortalResponse(portal_url=mock_portal_url)


@router.get(
    "/status",
    response_model=SubscriptionResponse,
    summary="Get subscription status",
    description="Get the current user's subscription details.",
)
async def get_subscription_status(
    current_user: Annotated[User, Depends(get_current_user)],
    subscription: Annotated[Subscription, Depends(get_user_subscription)],
) -> SubscriptionResponse:
    """
    Get current subscription status.

    Returns subscription details including usage.
    """
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "no_subscription",
                    "message": "No subscription found",
                }
            },
        )

    return SubscriptionResponse(
        id=subscription.id,
        plan=subscription.plan.value,
        status=subscription.status.value,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        analyses_this_period=subscription.analyses_this_period,
        analyses_remaining=subscription.analyses_remaining,
    )


@router.post(
    "/cancel",
    summary="Cancel subscription",
    description="Cancel the current subscription at period end.",
)
async def cancel_subscription(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Cancel current subscription.

    Subscription remains active until the end of the billing period.

    NOTE: Full Stripe integration implemented by DELTA.
    """
    if not current_user.subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "no_subscription",
                    "message": "No subscription to cancel",
                }
            },
        )

    if current_user.subscription.status == SubscriptionStatus.CANCELED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "already_canceled",
                    "message": "Subscription is already canceled",
                }
            },
        )

    # TODO: Cancel via Stripe API (DELTA handles)
    # For now, just update local status
    current_user.subscription.status = SubscriptionStatus.CANCELED

    await db.flush()

    return {
        "message": "Subscription will be canceled at the end of the billing period",
        "period_end": current_user.subscription.current_period_end.isoformat()
        if current_user.subscription.current_period_end
        else None,
    }


# Webhook endpoint (DELTA handles actual Stripe webhooks)
@router.post(
    "/webhook",
    include_in_schema=False,
    summary="Stripe webhook handler",
)
async def stripe_webhook():
    """
    Handle Stripe webhook events.

    NOTE: This is a placeholder. DELTA handles actual Stripe webhooks
    and forwards relevant events to update subscription status.
    """
    # DELTA will call internal API to update subscriptions
    # based on Stripe webhook events
    return {"received": True}

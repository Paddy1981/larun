"""
User Routes

Endpoints for user profile and usage management.

Agent: BETA
"""

from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.database import get_db
from src.api.models.user import User
from src.api.models.analysis import Analysis
from src.api.schemas.user import (
    UserProfile,
    UsageResponse,
    UpdateProfileRequest,
    UpdateProfileResponse,
    SubscriptionInfo,
)
from src.api.dependencies import get_current_user

router = APIRouter()


@router.get(
    "/profile",
    response_model=UserProfile,
    summary="Get user profile",
    description="Get the current user's profile with subscription info.",
)
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserProfile:
    """
    Get current user's profile.

    Returns user info and active subscription details.
    """
    # Build subscription info
    subscription_info = None
    if current_user.subscription:
        subscription_info = SubscriptionInfo(
            plan=current_user.subscription.plan.value,
            status=current_user.subscription.status.value,
            current_period_end=current_user.subscription.current_period_end,
        )

    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
        subscription=subscription_info,
    )


@router.patch(
    "/profile",
    response_model=UpdateProfileResponse,
    summary="Update user profile",
    description="Update the current user's profile.",
)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UpdateProfileResponse:
    """
    Update current user's profile.

    - **name**: New display name
    """
    if request.name is not None:
        current_user.name = request.name

    await db.flush()
    await db.refresh(current_user)

    # Build subscription info
    subscription_info = None
    if current_user.subscription:
        subscription_info = SubscriptionInfo(
            plan=current_user.subscription.plan.value,
            status=current_user.subscription.status.value,
            current_period_end=current_user.subscription.current_period_end,
        )

    return UpdateProfileResponse(
        user=UserProfile(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            created_at=current_user.created_at,
            subscription=subscription_info,
        ),
        message="Profile updated successfully",
    )


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Get usage statistics",
    description="Get the current user's analysis usage for the billing period.",
)
async def get_usage(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UsageResponse:
    """
    Get current usage statistics.

    Returns analyses used this period vs. limit.
    """
    if current_user.subscription is None:
        return UsageResponse(
            analyses_this_month=0,
            analyses_limit=0,
            period_start=None,
            period_end=None,
        )

    sub = current_user.subscription

    return UsageResponse(
        analyses_this_month=sub.analyses_this_period,
        analyses_limit=sub.analysis_limit,
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
    )


@router.get(
    "/stats",
    summary="Get user statistics",
    description="Get aggregate statistics for the user's analyses.",
)
async def get_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Get aggregate statistics.

    Returns counts and averages for user's analyses.
    """
    # Total analyses
    total_result = await db.execute(
        select(func.count(Analysis.id)).where(Analysis.user_id == current_user.id)
    )
    total_analyses = total_result.scalar() or 0

    # Completed analyses with detections
    detections_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.user_id == current_user.id,
            Analysis.detection == True,
        )
    )
    total_detections = detections_result.scalar() or 0

    # Average confidence (for completed analyses)
    avg_confidence_result = await db.execute(
        select(func.avg(Analysis.confidence)).where(
            Analysis.user_id == current_user.id,
            Analysis.confidence.isnot(None),
        )
    )
    avg_confidence = avg_confidence_result.scalar() or 0.0

    return {
        "total_analyses": total_analyses,
        "total_detections": total_detections,
        "detection_rate": total_detections / total_analyses if total_analyses > 0 else 0,
        "average_confidence": round(avg_confidence, 3),
    }

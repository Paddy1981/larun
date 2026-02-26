"""
Leaderboard Routes — Discovery rankings and user statistics.

GET /api/v2/leaderboard               — Top discoverers
GET /api/v2/users/{user_id}/stats     — Individual user stats
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/leaderboard")
async def get_leaderboard(
    period: Literal["all", "month", "week"] = Query("all", description="Time period"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get the discovery leaderboard.

    Ranks users by verified discovery count and total points.
    """
    from larun.discovery.leaderboard import DiscoveryLeaderboard

    lb = DiscoveryLeaderboard()
    return lb.get_leaderboard(period=period, limit=limit, offset=offset)


@router.get("/users/{user_id}/stats")
async def user_stats(user_id: str):
    """Get discovery statistics for a specific user."""
    from larun.discovery.leaderboard import DiscoveryLeaderboard

    lb = DiscoveryLeaderboard()
    stats = lb.get_user_stats(user_id)
    if stats is None:
        return {
            "user_id": user_id,
            "rank": "Stargazer",
            "points": 0,
            "total_discoveries": 0,
            "verified_discoveries": 0,
        }
    return stats

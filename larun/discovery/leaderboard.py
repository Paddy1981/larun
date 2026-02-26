"""
Discovery Leaderboard — Track and credit user discoveries.

Implements gamification:
    - Points for analyses, candidate submissions, verifications, confirmed discoveries
    - Ranks: Stargazer → Observer → Explorer → Discoverer → Master Astronomer
    - Permanent credit for verified discoveries
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Literal

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

RANKS = [
    (0,   "Stargazer"),
    (1,   "Observer"),
    (5,   "Explorer"),
    (20,  "Discoverer"),
    (100, "Master Astronomer"),
]

POINT_VALUES = {
    "analysis_run":          1,
    "candidate_submitted":   5,
    "verification_done":     2,
    "discovery_verified":   50,
    "anomaly_confirmed":   100,
}

# Minimum verifications for a candidate to become "verified"
VERIFICATIONS_REQUIRED = 3

# Database table schema (for reference / SQL generation)
SCHEMA = {
    "discoveries": """
        CREATE TABLE IF NOT EXISTS discoveries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
            ra FLOAT NOT NULL,
            dec FLOAT NOT NULL,
            classification VARCHAR(50),
            confidence FLOAT,
            novelty_score FLOAT,
            status VARCHAR(20) DEFAULT 'candidate',
            verification_count INT DEFAULT 0,
            discovered_at TIMESTAMPTZ DEFAULT NOW(),
            data_source VARCHAR(20),
            models_used JSONB
        );
    """,
    "verifications": """
        CREATE TABLE IF NOT EXISTS verifications (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            discovery_id UUID REFERENCES discoveries(id) ON DELETE CASCADE,
            verifier_id UUID REFERENCES users(id) ON DELETE SET NULL,
            verdict VARCHAR(20) NOT NULL,
            notes TEXT,
            verified_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,
    "user_stats": """
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            total_discoveries INT DEFAULT 0,
            verified_discoveries INT DEFAULT 0,
            total_analyses INT DEFAULT 0,
            total_verifications INT DEFAULT 0,
            rank VARCHAR(30) DEFAULT 'Stargazer',
            points INT DEFAULT 0
        );
    """,
}


class DiscoveryLeaderboard:
    """
    Discovery leaderboard and gamification system.

    In production, backed by Supabase PostgreSQL.
    MVP: in-memory dict store (replace with DB client in production).
    """

    def __init__(self, db_client=None):
        """
        Args:
            db_client: Optional Supabase or SQLAlchemy client.
                       Falls back to in-memory store if None.
        """
        self._db = db_client
        self._discoveries: dict[str, dict] = {}  # id → discovery
        self._user_stats: dict[str, dict] = {}   # user_id → stats

    # -------------------------------------------------------------------------
    # Discovery Submission
    # -------------------------------------------------------------------------

    def submit_discovery(
        self,
        user_id: str,
        ra: float,
        dec: float,
        classification: str,
        confidence: float,
        novelty_score: float,
        data_source: str,
        models_used: dict | None = None,
    ) -> dict:
        """
        Record a new discovery candidate.

        Called when a user submits a candidate from the Discovery Engine.

        Returns:
            dict with discovery_id, points_earned, new_rank
        """
        discovery_id = str(uuid.uuid4())
        discovery = {
            "id": discovery_id,
            "user_id": user_id,
            "ra": ra,
            "dec": dec,
            "classification": classification,
            "confidence": confidence,
            "novelty_score": novelty_score,
            "status": "candidate",
            "verification_count": 0,
            "discovered_at": datetime.now(timezone.utc).isoformat(),
            "data_source": data_source,
            "models_used": models_used or {},
        }

        self._discoveries[discovery_id] = discovery
        points = self._award_points(user_id, "candidate_submitted")

        logger.info(f"Discovery submitted: {discovery_id} by user {user_id} (+{points} pts)")

        return {
            "discovery_id": discovery_id,
            "status": "candidate",
            "points_earned": points,
            "new_rank": self.get_rank(user_id),
        }

    # -------------------------------------------------------------------------
    # Leaderboard Queries
    # -------------------------------------------------------------------------

    def get_leaderboard(
        self,
        period: Literal["all", "month", "week"] = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Get top discoverers sorted by verified discoveries."""
        now = datetime.now(timezone.utc)
        cutoff = None
        if period == "month":
            cutoff = now - timedelta(days=30)
        elif period == "week":
            cutoff = now - timedelta(days=7)

        users = []
        for user_id, stats in self._user_stats.items():
            if cutoff:
                # Filter by period (would be a DB query in production)
                # For now include all
                pass
            users.append({
                "user_id": user_id,
                "rank": stats.get("rank", "Stargazer"),
                "points": stats.get("points", 0),
                "verified_discoveries": stats.get("verified_discoveries", 0),
                "total_analyses": stats.get("total_analyses", 0),
            })

        users.sort(key=lambda u: (-u["verified_discoveries"], -u["points"]))
        page = users[offset: offset + limit]

        return {
            "period": period,
            "total_users": len(users),
            "rankings": [
                {"position": offset + i + 1, **u}
                for i, u in enumerate(page)
            ],
        }

    def get_candidates(
        self,
        status: str = "candidate",
        sort: str = "priority",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Get discovery candidates awaiting verification."""
        candidates = [d for d in self._discoveries.values() if d.get("status") == status]

        if sort == "novelty_score":
            candidates.sort(key=lambda d: -d.get("novelty_score", 0))
        elif sort == "discovered_at":
            candidates.sort(key=lambda d: d.get("discovered_at", ""), reverse=True)
        else:
            candidates.sort(key=lambda d: -d.get("novelty_score", 0))

        page = candidates[offset: offset + limit]
        return {"total": len(candidates), "candidates": page}

    def get_user_stats(self, user_id: str) -> dict | None:
        """Get stats for a specific user."""
        if user_id not in self._user_stats:
            return None
        stats = dict(self._user_stats[user_id])
        stats["rank"] = self.get_rank(user_id)
        return stats

    # -------------------------------------------------------------------------
    # Rank System
    # -------------------------------------------------------------------------

    def get_rank(self, user_id: str) -> str:
        """Get current rank based on verified discoveries."""
        stats = self._user_stats.get(user_id, {})
        verified = stats.get("verified_discoveries", 0)

        current_rank = "Stargazer"
        for threshold, rank_name in RANKS:
            if verified >= threshold:
                current_rank = rank_name

        return current_rank

    def get_rank_progress(self, user_id: str) -> dict:
        """Get progress towards next rank."""
        stats = self._user_stats.get(user_id, {})
        verified = stats.get("verified_discoveries", 0)

        current_rank = "Stargazer"
        next_rank = None
        next_threshold = None

        for i, (threshold, rank_name) in enumerate(RANKS):
            if verified >= threshold:
                current_rank = rank_name
                if i + 1 < len(RANKS):
                    next_threshold, next_rank = RANKS[i + 1]

        return {
            "current_rank": current_rank,
            "next_rank": next_rank,
            "next_threshold": next_threshold,
            "verified_discoveries": verified,
            "progress": verified / next_threshold if next_threshold else 1.0,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _award_points(self, user_id: str, event: str) -> int:
        """Award points for an event and update user stats."""
        points = POINT_VALUES.get(event, 0)
        if not points:
            return 0

        if user_id not in self._user_stats:
            self._user_stats[user_id] = {
                "points": 0,
                "total_discoveries": 0,
                "verified_discoveries": 0,
                "total_analyses": 0,
                "total_verifications": 0,
                "rank": "Stargazer",
            }

        self._user_stats[user_id]["points"] = self._user_stats[user_id].get("points", 0) + points

        if event == "analysis_run":
            self._user_stats[user_id]["total_analyses"] = self._user_stats[user_id].get("total_analyses", 0) + 1
        elif event == "candidate_submitted":
            self._user_stats[user_id]["total_discoveries"] = self._user_stats[user_id].get("total_discoveries", 0) + 1
        elif event == "verification_done":
            self._user_stats[user_id]["total_verifications"] = self._user_stats[user_id].get("total_verifications", 0) + 1
        elif event in ("discovery_verified", "anomaly_confirmed"):
            self._user_stats[user_id]["verified_discoveries"] = self._user_stats[user_id].get("verified_discoveries", 0) + 1

        self._user_stats[user_id]["rank"] = self.get_rank(user_id)
        return points

    def mark_verified(self, discovery_id: str) -> None:
        """Mark a discovery as verified and award points to discoverer."""
        if discovery_id not in self._discoveries:
            return
        discovery = self._discoveries[discovery_id]
        discovery["status"] = "verified"

        user_id = discovery.get("user_id")
        if user_id:
            classification = discovery.get("classification", "")
            event = "anomaly_confirmed" if "ANOMALY" in classification else "discovery_verified"
            self._award_points(user_id, event)
            logger.info(f"Discovery {discovery_id} verified. Credited to user {user_id}")

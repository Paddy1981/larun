"""
Verification System — Peer review for discovery candidates.

Three independent confirmations → candidate becomes 'verified'.
Consensus-based: requires majority agreement among verifiers.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

CONFIRMATIONS_REQUIRED = 3  # confirmations needed to verify
REJECTIONS_TO_DISCARD = 2   # rejections needed to discard


class VerificationSystem:
    """
    Peer verification for discovery candidates.

    Process:
    1. User submits discovery → status = 'candidate'
    2. Community members review and vote: confirm / reject / unsure
    3. 3 confirms → status = 'verified', discoverer gets credit
    4. 2 rejects → status = 'rejected'
    """

    def __init__(self, leaderboard=None):
        self._leaderboard = leaderboard
        self._verifications: dict[str, list[dict]] = {}  # discovery_id → [verification]

    def submit_verification(
        self,
        discovery_id: str,
        verdict: Literal["confirm", "reject", "unsure"],
        notes: str | None = None,
        verifier_id: str = "anonymous",
    ) -> dict:
        """
        Submit a verification verdict for a discovery candidate.

        Args:
            discovery_id: UUID of the discovery candidate
            verdict:      'confirm', 'reject', or 'unsure'
            notes:        Optional reviewer notes
            verifier_id:  User ID of the verifier

        Returns:
            dict with verification_id, new_status, verification_count,
                  confirmations, rejections, needs_more_reviews
        """
        if discovery_id not in self._verifications:
            self._verifications[discovery_id] = []

        # Check for duplicate vote from same verifier
        existing = [v for v in self._verifications[discovery_id] if v["verifier_id"] == verifier_id]
        if existing and verifier_id != "anonymous":
            return {"error": "You have already verified this candidate", "discovery_id": discovery_id}

        verification = {
            "id": str(uuid.uuid4()),
            "discovery_id": discovery_id,
            "verifier_id": verifier_id,
            "verdict": verdict,
            "notes": notes,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

        self._verifications[discovery_id].append(verification)

        # Award points to verifier
        if self._leaderboard and verifier_id != "anonymous":
            self._leaderboard._award_points(verifier_id, "verification_done")

        # Check consensus
        all_verifications = self._verifications[discovery_id]
        confirmations = sum(1 for v in all_verifications if v["verdict"] == "confirm")
        rejections = sum(1 for v in all_verifications if v["verdict"] == "reject")
        total = len(all_verifications)

        # Determine new status
        new_status = "candidate"
        if confirmations >= CONFIRMATIONS_REQUIRED:
            new_status = "verified"
            if self._leaderboard:
                self._leaderboard.mark_verified(discovery_id)
            logger.info(f"Discovery {discovery_id} VERIFIED ({confirmations} confirmations)")

        elif rejections >= REJECTIONS_TO_DISCARD:
            new_status = "rejected"
            logger.info(f"Discovery {discovery_id} REJECTED ({rejections} rejections)")

        return {
            "verification_id": verification["id"],
            "discovery_id": discovery_id,
            "verdict": verdict,
            "new_status": new_status,
            "confirmations": confirmations,
            "rejections": rejections,
            "total_reviews": total,
            "needs_more_reviews": new_status == "candidate",
            "confirmations_needed": max(0, CONFIRMATIONS_REQUIRED - confirmations),
        }

    def get_verifications(self, discovery_id: str) -> list[dict]:
        """Get all verifications for a candidate."""
        return self._verifications.get(discovery_id, [])

    def get_status(self, discovery_id: str) -> dict:
        """Get verification status summary for a candidate."""
        verifications = self.get_verifications(discovery_id)
        confirmations = sum(1 for v in verifications if v["verdict"] == "confirm")
        rejections = sum(1 for v in verifications if v["verdict"] == "reject")

        if confirmations >= CONFIRMATIONS_REQUIRED:
            status = "verified"
        elif rejections >= REJECTIONS_TO_DISCARD:
            status = "rejected"
        else:
            status = "candidate"

        return {
            "discovery_id": discovery_id,
            "status": status,
            "confirmations": confirmations,
            "rejections": rejections,
            "total_reviews": len(verifications),
            "confirmations_needed": max(0, CONFIRMATIONS_REQUIRED - confirmations),
        }

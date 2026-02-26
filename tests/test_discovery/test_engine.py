"""Tests for Citizen Discovery Engine."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestDiscoveryEngine:
    def test_discovery_report_structure(self):
        """DiscoveryReport should have correct structure."""
        from larun.discovery.engine import DiscoveryCandidate, DiscoveryReport

        mock_candidate = DiscoveryCandidate(
            target={"ra": 56.75, "dec": 24.12},
            light_curve_meta={"n_points": 100},
            classifications={"VARDET-001": {"label": "PULSATOR", "confidence": 0.92}},
            catalog_match={"known": False, "novelty_score": 0.9, "matches": []},
            consensus={"consensus_label": "PULSATOR", "is_variable": True, "anomaly_detected": False, "blend_detected": False},
            priority=70,
            source="neowise",
            is_candidate=True,
            novelty_score=0.9,
        )

        report = DiscoveryReport(results=[mock_candidate], meta={
            "ra": 56.75, "dec": 24.12, "radius_deg": 0.5,
            "total_targets": 1, "analyzed": 1, "elapsed_seconds": 1.0,
            "sources": ["neowise"], "models_used": "all",
        })

        assert len(report.candidates) == 1
        assert len(report.known) == 0
        response = report.to_response()
        assert "candidates" in response
        assert "stats" in response

    def test_priority_calculation(self):
        """Priority should increase for unknown anomalous objects."""
        from larun.discovery.engine import CitizenDiscoveryEngine

        engine = CitizenDiscoveryEngine.__new__(CitizenDiscoveryEngine)

        # Unknown + strong anomaly = highest priority
        classifications = {
            "ANOMALY-001": {"label": "STRONG_ANOMALY", "confidence": 0.95},
            "VARDET-001": {"label": "TRANSIENT", "confidence": 0.97},
        }
        catalog_match = {"known": False}
        consensus = {"agreement_count": 4}

        priority = engine._calculate_priority(classifications, catalog_match, consensus)
        assert priority >= 80  # Unknown(50) + STRONG_ANOMALY(30) + high conf(20) - capped at 100

    def test_priority_lower_for_known_object(self):
        from larun.discovery.engine import CitizenDiscoveryEngine

        engine = CitizenDiscoveryEngine.__new__(CitizenDiscoveryEngine)

        classifications = {
            "ANOMALY-001": {"label": "NORMAL", "confidence": 0.8},
            "VARDET-001": {"label": "NON_VARIABLE", "confidence": 0.9},
        }
        catalog_match = {"known": True}
        consensus = {"agreement_count": 2}

        priority = engine._calculate_priority(classifications, catalog_match, consensus)
        assert priority < 50  # Known object, no anomaly


class TestDiscoveryLeaderboard:
    def test_submit_discovery(self):
        from larun.discovery.leaderboard import DiscoveryLeaderboard

        lb = DiscoveryLeaderboard()
        result = lb.submit_discovery(
            user_id="user123",
            ra=56.75,
            dec=24.12,
            classification="PULSATOR",
            confidence=0.92,
            novelty_score=0.9,
            data_source="neowise",
        )

        assert "discovery_id" in result
        assert result["status"] == "candidate"
        assert result["points_earned"] > 0

    def test_rank_progression(self):
        from larun.discovery.leaderboard import DiscoveryLeaderboard

        lb = DiscoveryLeaderboard()
        user_id = "test_user"

        # No discoveries → Stargazer
        assert lb.get_rank(user_id) == "Stargazer"

        # Add verified discoveries manually
        lb._user_stats[user_id] = {"verified_discoveries": 6}
        assert lb.get_rank(user_id) == "Explorer"

        lb._user_stats[user_id] = {"verified_discoveries": 25}
        assert lb.get_rank(user_id) == "Discoverer"

    def test_leaderboard_structure(self):
        from larun.discovery.leaderboard import DiscoveryLeaderboard

        lb = DiscoveryLeaderboard()
        lb._user_stats["u1"] = {"verified_discoveries": 5, "points": 100, "rank": "Explorer"}
        lb._user_stats["u2"] = {"verified_discoveries": 1, "points": 20, "rank": "Observer"}

        leaderboard = lb.get_leaderboard()
        assert "rankings" in leaderboard
        assert leaderboard["total_users"] == 2
        # u1 should rank first (more verified discoveries)
        assert leaderboard["rankings"][0]["user_id"] == "u1"


class TestVerificationSystem:
    def test_single_verification(self):
        from larun.discovery.verification import VerificationSystem

        vs = VerificationSystem()
        result = vs.submit_verification(
            discovery_id="disc-001",
            verdict="confirm",
            verifier_id="user1",
        )

        assert result["verdict"] == "confirm"
        assert result["confirmations"] == 1
        assert result["new_status"] == "candidate"

    def test_three_confirmations_verify(self):
        from larun.discovery.verification import VerificationSystem, CONFIRMATIONS_REQUIRED

        vs = VerificationSystem()
        disc_id = "disc-verify"

        for i in range(CONFIRMATIONS_REQUIRED):
            result = vs.submit_verification(
                discovery_id=disc_id,
                verdict="confirm",
                verifier_id=f"user{i}",
            )

        assert result["new_status"] == "verified"

    def test_rejections_discard(self):
        from larun.discovery.verification import VerificationSystem, REJECTIONS_TO_DISCARD

        vs = VerificationSystem()
        disc_id = "disc-reject"

        for i in range(REJECTIONS_TO_DISCARD):
            result = vs.submit_verification(
                discovery_id=disc_id,
                verdict="reject",
                verifier_id=f"user{i}",
            )

        assert result["new_status"] == "rejected"

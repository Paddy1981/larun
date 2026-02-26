"""
Citizen Discovery Engine — Find unknown space objects.

The killer feature of larun.space. Users can systematically search
NASA archives for objects unknown to science and get permanently credited.

Inspired by VARnet's discovery of 1.5M unknown objects from NEOWISE data.
"""

from larun.discovery.engine import CitizenDiscoveryEngine, DiscoveryReport
from larun.discovery.leaderboard import DiscoveryLeaderboard
from larun.discovery.verification import VerificationSystem

__all__ = ["CitizenDiscoveryEngine", "DiscoveryReport", "DiscoveryLeaderboard", "VerificationSystem"]

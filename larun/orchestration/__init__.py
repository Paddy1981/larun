"""
LARUN Orchestration Layer — Claude API as the intelligence bridge.

Claude API is used for exactly three tasks:
    1. Natural language query parsing → structured pipeline calls
    2. Publication-ready report generation from model results
    3. Anomaly explanation in astrophysical context

Estimated cost: ~$0.003 per user query (claude-sonnet-4-20250514, max 1000 tokens)
Monthly cap: $50 (≈16,000 queries)
"""

from larun.orchestration.claude_router import ClaudeOrchestrator
from larun.orchestration.report_generator import ReportGenerator

__all__ = ["ClaudeOrchestrator", "ReportGenerator"]

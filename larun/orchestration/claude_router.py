"""
Claude Orchestrator — Natural language interface for LARUN.

Uses Claude API (Sonnet) for:
    1. parse_query()    — NL → structured pipeline call
    2. generate_report() — model results → publication-ready text
    3. explain_anomaly() — anomaly data → astrophysical interpretation

Cost target: ~$0.003 per call (1000 max tokens, Sonnet model)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 1000

_AVAILABLE_MODELS = (
    "EXOPLANET-001, VSTAR-001, FLARE-001, MICROLENS-001, "
    "SUPERNOVA-001, SPECTYPE-001, ASTERO-001, GALAXY-001, "
    "VARDET-001, ANOMALY-001, DEBLEND-001, PERIODOGRAM-001"
)

_PARSE_SYSTEM = f"""You are the query parser for larun.space, a TinyML space science platform.
Convert the user's natural language query into a structured JSON action.

Available actions:
  - classify: Run models on uploaded data {{model_id: string, target: string}}
  - discover: Run discovery pipeline {{ra: float, dec: float, radius_deg: float, sources: list}}
  - search_catalog: Search VarWISE/VSX/SIMBAD {{ra: float, dec: float, name: string, var_class: string}}
  - pipeline: Fetch + analyze from archive {{source: "tess"|"kepler"|"neowise", target: string}}
  - cross_match: Check if object is known {{ra: float, dec: float}}

Available models: {_AVAILABLE_MODELS}
Available sources: tess, kepler, neowise

Common astronomical objects and their coordinates (J2000):
  - Pleiades: ra=56.75, dec=24.12
  - Andromeda (M31): ra=10.68, dec=41.27
  - Crab Nebula (M1): ra=83.82, dec=22.01
  - Boyajian's Star (KIC 8462852): ra=301.56, dec=44.46
  - Center of Galaxy: ra=266.40, dec=-28.94

Respond ONLY with valid JSON. No explanations, no markdown."""

_REPORT_SYSTEM = """You are a scientific report writer for larun.space, a TinyML space science platform.
Generate a clear, publication-style summary of discovery results.

Include:
  - Object classification from the TinyML model federation
  - Confidence levels and which models agree/disagree
  - Astrophysical significance of the classification
  - Suggested follow-up observations
  - Whether the object is known or a candidate new discovery

Use professional astronomical terminology. Be concise (max 3 paragraphs).
Focus on the science, not on the AI platform details."""

_ANOMALY_SYSTEM = """You are an astrophysics expert for larun.space.
Given anomaly detection results from a TinyML model federation, explain what this object might be.

Consider:
  - Unusual variability patterns and their physical mechanisms
  - Known rare phenomena: Boyajian's Star (KIC 8462852), fast radio bursts,
    long-period variables, symbiotic stars, cataclysmic variables
  - Whether this could be an instrumental artifact or astrophysical
  - Priority level for follow-up (human visual inspection recommended for STRONG_ANOMALY)

Be specific but note this is a candidate requiring independent confirmation.
Keep response to 2 paragraphs."""


class ClaudeOrchestrator:
    """
    Claude API orchestrator for natural language interactions on larun.space.

    Requires LARUN_CLAUDE_API_KEY (or ANTHROPIC_API_KEY) in environment.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("LARUN_CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                if not self._api_key:
                    raise EnvironmentError(
                        "No Claude API key found. Set LARUN_CLAUDE_API_KEY or ANTHROPIC_API_KEY."
                    )
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError("anthropic package required. pip install anthropic")
        return self._client

    # -------------------------------------------------------------------------
    # 1. Natural Language → Structured Action
    # -------------------------------------------------------------------------

    def parse_query(self, user_text: str) -> dict:
        """
        Convert natural language to a structured pipeline action.

        Example:
            Input:  "Find variable stars near the Pleiades in TESS data"
            Output: {
                "action": "discover",
                "params": {
                    "ra": 56.75, "dec": 24.12, "radius_deg": 1.0,
                    "sources": ["tess"],
                    "focus_models": ["VSTAR-001", "VARDET-001"]
                }
            }
        """
        client = self._get_client()
        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=500,
                system=_PARSE_SYSTEM,
                messages=[{"role": "user", "content": user_text}],
            )
            text = response.content[0].text.strip()
            # Clean up any accidental markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(f"Claude response not valid JSON: {exc}")
            return {"action": "error", "message": "Could not parse query", "raw": text}
        except Exception as exc:
            logger.error(f"Claude parse_query error: {exc}")
            return {"action": "error", "message": str(exc)}

    # -------------------------------------------------------------------------
    # 2. Model Results → Publication Report
    # -------------------------------------------------------------------------

    def generate_report(
        self,
        discovery_results: dict,
        target_info: dict | None = None,
    ) -> str:
        """
        Generate a publication-ready report from model federation results.

        Args:
            discovery_results: output from ModelFederation.run_all() or CitizenDiscoveryEngine
            target_info: optional dict with ra, dec, source, tic_id, etc.

        Returns:
            Multi-paragraph text report as a string
        """
        client = self._get_client()

        payload = {
            "model_results": discovery_results,
            "target_info": target_info or {},
        }

        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=_REPORT_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            )
            return response.content[0].text

        except Exception as exc:
            logger.error(f"Claude generate_report error: {exc}")
            return f"Report generation failed: {exc}"

    # -------------------------------------------------------------------------
    # 3. Anomaly Explanation
    # -------------------------------------------------------------------------

    def explain_anomaly(
        self,
        anomaly_data: dict,
        light_curve_summary: dict | None = None,
    ) -> str:
        """
        Interpret what an anomaly detection might mean astrophysically.

        Args:
            anomaly_data: ANOMALY-001 result dict + other model results
            light_curve_summary: optional stats (amplitude, period, n_points, etc.)

        Returns:
            2-paragraph explanation string
        """
        client = self._get_client()

        payload = {
            "anomaly_result": anomaly_data,
            "light_curve_summary": light_curve_summary or {},
        }

        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=500,
                system=_ANOMALY_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
            )
            return response.content[0].text

        except Exception as exc:
            logger.error(f"Claude explain_anomaly error: {exc}")
            return f"Anomaly explanation failed: {exc}"

    # -------------------------------------------------------------------------
    # Availability check
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Claude API is configured and accessible."""
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

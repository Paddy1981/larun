"""
Report Generator — Structured report builder for LARUN discoveries.

Generates JSON, CSV, and text reports from discovery results.
Optional Claude API integration for natural language summaries.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate structured reports from LARUN discovery results.

    Supports:
        - JSON export (full structured data)
        - CSV export (tabular, for analysis tools)
        - Text summary (human-readable)
        - AI-powered publication report (via Claude API)
    """

    def __init__(self, use_claude: bool = True):
        """
        Args:
            use_claude: if True, use Claude API for text report generation
        """
        self._use_claude = use_claude
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from larun.orchestration.claude_router import ClaudeOrchestrator
            self._orchestrator = ClaudeOrchestrator()
        return self._orchestrator

    def to_json(self, discovery_results: dict, pretty: bool = True) -> str:
        """Full structured JSON export."""
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "platform": "larun.space",
            "version": "2.0",
            "results": discovery_results,
        }
        return json.dumps(report, indent=2 if pretty else None, default=str)

    def to_csv(self, candidates: list[dict]) -> str:
        """CSV export of discovery candidates."""
        if not candidates:
            return ""

        output = io.StringIO()
        fieldnames = [
            "ra", "dec", "source", "novelty_score", "consensus_label",
            "consensus_confidence", "anomaly_detected", "blend_detected",
            "vardet_label", "vardet_confidence", "anomaly_score", "period",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for cand in candidates:
            target = cand.get("target", {})
            classifications = cand.get("classifications", {})
            consensus = cand.get("consensus", {})

            vardet = classifications.get("VARDET-001", {})
            anomaly = classifications.get("ANOMALY-001", {})
            periodogram = classifications.get("PERIODOGRAM-001", {})

            writer.writerow({
                "ra": target.get("ra", ""),
                "dec": target.get("dec", ""),
                "source": cand.get("source", ""),
                "novelty_score": cand.get("novelty_score", ""),
                "consensus_label": consensus.get("consensus_label", ""),
                "consensus_confidence": consensus.get("consensus_confidence", ""),
                "anomaly_detected": consensus.get("anomaly_detected", False),
                "blend_detected": consensus.get("blend_detected", False),
                "vardet_label": vardet.get("label", ""),
                "vardet_confidence": vardet.get("confidence", ""),
                "anomaly_score": anomaly.get("confidence", ""),
                "period": periodogram.get("best_period", ""),
            })

        return output.getvalue()

    def to_text_summary(self, discovery_results: dict) -> str:
        """Human-readable text summary (no Claude API required)."""
        candidates = discovery_results.get("candidates", [])
        known = discovery_results.get("known", [])
        anomalies = discovery_results.get("anomalies", [])
        meta = discovery_results.get("meta", {})

        lines = [
            "=" * 60,
            "LARUN.SPACE DISCOVERY REPORT",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
            "",
        ]

        if meta:
            lines += [
                f"Sky Region:  RA={meta.get('ra', '?'):.4f}°, Dec={meta.get('dec', '?'):.4f}°",
                f"Radius:      {meta.get('radius_deg', '?')}°",
                f"Data Source: {meta.get('sources', '?')}",
                f"Models Run:  {meta.get('models_used', 'all')}",
                "",
            ]

        lines += [
            f"SUMMARY",
            f"  Total targets analyzed: {meta.get('total_targets', len(candidates) + len(known))}",
            f"  Discovery candidates:   {len(candidates)}",
            f"  Known objects:          {len(known)}",
            f"  Anomalies detected:     {len(anomalies)}",
            "",
        ]

        if candidates:
            lines += ["DISCOVERY CANDIDATES (priority order):"]
            for i, cand in enumerate(sorted(candidates, key=lambda c: -c.get("priority", 0))[:10], 1):
                target = cand.get("target", {})
                consensus = cand.get("consensus", {})
                lines.append(
                    f"  {i:2d}. RA={target.get('ra', 0):.4f} Dec={target.get('dec', 0):.4f} "
                    f"| {consensus.get('consensus_label', '?')} "
                    f"({consensus.get('consensus_confidence', 0):.0%}) "
                    f"| novelty={cand.get('novelty_score', 0):.2f} "
                    f"| priority={cand.get('priority', 0)}"
                )

        if anomalies:
            lines += ["", "ANOMALIES (priority review):"]
            for anomaly in anomalies[:5]:
                target = anomaly.get("target", {})
                anom_result = anomaly.get("classifications", {}).get("ANOMALY-001", {})
                lines.append(
                    f"  ⚠️  RA={target.get('ra', 0):.4f} Dec={target.get('dec', 0):.4f} "
                    f"| {anom_result.get('label', '?')} "
                    f"| confidence={anom_result.get('confidence', 0):.0%}"
                )

        lines += ["", "=" * 60]
        return "\n".join(lines)

    def to_ai_report(
        self,
        discovery_results: dict,
        target_info: dict | None = None,
    ) -> str:
        """
        Generate publication-ready AI report using Claude API.
        Falls back to text_summary if Claude unavailable.
        """
        if not self._use_claude:
            return self.to_text_summary(discovery_results)

        orchestrator = self._get_orchestrator()
        if not orchestrator.is_available():
            logger.info("Claude API not available, using text summary")
            return self.to_text_summary(discovery_results)

        return orchestrator.generate_report(discovery_results, target_info)

    def generate_full_report(self, discovery_results: dict) -> dict[str, str]:
        """
        Generate all report formats at once.

        Returns:
            dict with keys: 'json', 'csv', 'text', 'ai' (if Claude available)
        """
        candidates = discovery_results.get("candidates", [])
        return {
            "json": self.to_json(discovery_results),
            "csv": self.to_csv(candidates),
            "text": self.to_text_summary(discovery_results),
            "ai": self.to_ai_report(discovery_results),
        }

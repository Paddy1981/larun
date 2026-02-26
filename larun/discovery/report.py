"""
Discovery Report Generator — Shareable discovery reports.

Generates reports for individual discoveries and full discovery runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_discovery_card(candidate: dict) -> dict:
    """
    Generate a shareable discovery card for a single candidate.

    Returns a dict suitable for rendering in the frontend DiscoveryReport component.
    """
    classifications = candidate.get("classifications", {})
    catalog_match = candidate.get("catalog_match", {})
    consensus = candidate.get("consensus", {})
    target = candidate.get("target", {})
    periodogram = classifications.get("PERIODOGRAM-001", {})

    return {
        "title": _generate_title(consensus, catalog_match),
        "coordinates": {
            "ra": target.get("ra", 0),
            "dec": target.get("dec", 0),
            "ra_formatted": _format_ra(target.get("ra", 0)),
            "dec_formatted": _format_dec(target.get("dec", 0)),
        },
        "classification": {
            "label": consensus.get("consensus_label", "UNKNOWN"),
            "confidence": consensus.get("consensus_confidence", 0),
            "is_variable": consensus.get("is_variable", False),
        },
        "novelty": {
            "score": candidate.get("novelty_score", 0),
            "is_new": not catalog_match.get("known", True),
            "known_names": [m.get("name") for m in catalog_match.get("matches", [])[:3]],
        },
        "period": {
            "days": periodogram.get("best_period"),
            "type": periodogram.get("period_type"),
            "confidence": periodogram.get("confidence"),
        } if periodogram.get("best_period") else None,
        "flags": {
            "anomaly": consensus.get("anomaly_detected", False),
            "blend": consensus.get("blend_detected", False),
        },
        "source": candidate.get("source", "unknown"),
        "priority": candidate.get("priority", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate_title(consensus: dict, catalog_match: dict) -> str:
    """Generate a human-readable title for a discovery card."""
    label = consensus.get("consensus_label", "Unknown Object")

    if not catalog_match.get("known", True):
        return f"New {_label_to_name(label)} Candidate"

    known_name = catalog_match.get("matches", [{}])[0].get("name", "Known Object")
    return f"{known_name} — {_label_to_name(label)}"


def _label_to_name(label: str) -> str:
    """Convert model label to human-readable name."""
    names = {
        "NON_VARIABLE": "Non-Variable Star",
        "TRANSIENT": "Transient Object",
        "PULSATOR": "Pulsating Variable Star",
        "ECLIPSING": "Eclipsing Binary / Transit",
        "STRONG_ANOMALY": "Unusual Object",
        "MILD_ANOMALY": "Slightly Unusual Object",
    }
    return names.get(label, label.replace("_", " ").title())


def _format_ra(ra: float) -> str:
    """Convert RA degrees to HH:MM:SS.s format."""
    ra_h = ra / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = ((ra_h - h) * 60 - m) * 60
    return f"{h:02d}h {m:02d}m {s:04.1f}s"


def _format_dec(dec: float) -> str:
    """Convert Dec degrees to ±DD:MM:SS format."""
    sign = "+" if dec >= 0 else "-"
    dec = abs(dec)
    d = int(dec)
    m = int((dec - d) * 60)
    s = ((dec - d) * 60 - m) * 60
    return f"{sign}{d:02d}° {m:02d}' {s:04.1f}\""

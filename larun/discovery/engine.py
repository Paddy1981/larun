"""
Citizen Discovery Engine — Core orchestrator.

Runs the full discovery pipeline:
    1. Query target catalogs in a sky region
    2. Fetch light curves from TESS/Kepler/NEOWISE
    3. Run all 12 TinyML models on each target
    4. Cross-match against 6+ catalogs
    5. Score and rank discovery candidates
    6. Return DiscoveryReport

Value proposition:
    Any user — student, amateur astronomer, researcher —
    can find objects unknown to science, the same way
    Matteo Paz found 1.5M objects with VARnet.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryCandidate:
    """A single candidate discovery from the pipeline."""
    target: dict
    light_curve_meta: dict
    classifications: dict
    catalog_match: dict
    consensus: dict
    priority: int
    source: str
    is_candidate: bool
    novelty_score: float


class CitizenDiscoveryEngine:
    """
    Orchestrates the full citizen discovery workflow.

    Usage:
        engine = CitizenDiscoveryEngine()
        report = engine.discover(ra=56.75, dec=24.12, radius_deg=0.5)
        print(report.summary())
    """

    def __init__(self):
        from larun.pipelines.tess import TESSPipeline
        from larun.pipelines.kepler import KeplerPipeline
        from larun.pipelines.neowise import NEOWISEPipeline
        from larun.pipelines.cross_match import CrossMatchPipeline
        from larun.models.federation import ModelFederation

        self.tess = TESSPipeline()
        self.kepler = KeplerPipeline()
        self.neowise = NEOWISEPipeline()
        self.cross_match = CrossMatchPipeline()
        self.models = ModelFederation()

    def discover(
        self,
        ra: float,
        dec: float,
        radius_deg: float = 0.5,
        sources: list[str] | str = "all",
        max_targets: int = 50,
    ) -> "DiscoveryReport":
        """
        Full discovery pipeline for a sky region.

        Args:
            ra, dec:       Center coordinates (degrees, J2000)
            radius_deg:    Search radius (degrees)
            sources:       'all' or list of ['tess', 'kepler', 'neowise']
            max_targets:   Max number of targets to analyze (quota control)

        Returns:
            DiscoveryReport with candidates, known objects, anomalies
        """
        t_start = time.time()
        logger.info(f"Discovery run: ra={ra:.4f}, dec={dec:.4f}, r={radius_deg}°, sources={sources}")

        if isinstance(sources, str) and sources == "all":
            sources = ["tess", "neowise", "kepler"]

        # 1. Collect targets in the sky region
        targets = self._get_targets_in_region(ra, dec, radius_deg, sources, max_targets)
        logger.info(f"Found {len(targets)} targets in region")

        results = []
        for i, target in enumerate(targets):
            logger.debug(f"Processing target {i + 1}/{len(targets)}: {target}")

            # 2. Fetch light curve
            lc = self._fetch_light_curve(target)
            if lc is None:
                continue

            # 3. Run all TinyML models
            classifications = self.models.run_layer2_parallel(lc)
            consensus = self.models.consensus(classifications)

            # 4. Cross-match catalogs
            catalog_match = self.cross_match.cross_match(
                target.get("ra", ra),
                target.get("dec", dec),
                radius_arcsec=3.0,
            )

            # 5. Determine if this is a candidate
            is_candidate = (
                not catalog_match["known"]
                and classifications.get("VARDET-001", {}).get("label", "NON_VARIABLE") != "NON_VARIABLE"
            )

            priority = self._calculate_priority(classifications, catalog_match, consensus)
            novelty = catalog_match.get("novelty_score", 0.5 if not catalog_match["known"] else 0.0)

            candidate = DiscoveryCandidate(
                target=target,
                light_curve_meta=lc.get("meta", {}),
                classifications=classifications,
                catalog_match=catalog_match,
                consensus=consensus,
                priority=priority,
                source=target.get("source", "unknown"),
                is_candidate=is_candidate,
                novelty_score=novelty,
            )
            results.append(candidate)

        elapsed = round(time.time() - t_start, 1)
        logger.info(
            f"Discovery complete: {len(results)} analyzed, "
            f"{sum(1 for r in results if r.is_candidate)} candidates in {elapsed}s"
        )

        return DiscoveryReport(
            results=results,
            meta={
                "ra": ra,
                "dec": dec,
                "radius_deg": radius_deg,
                "sources": sources,
                "total_targets": len(targets),
                "analyzed": len(results),
                "elapsed_seconds": elapsed,
                "models_used": "all",
            },
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_targets_in_region(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
        sources: list[str],
        max_targets: int,
    ) -> list[dict]:
        """
        Build target list from catalog queries in the sky region.
        Uses NEOWISE cone search + TIC catalog via MAST.
        """
        targets = []
        radius_arcmin = radius_deg * 60

        if "neowise" in sources:
            neowise_targets = self.neowise.query_cone(ra, dec, radius_arcmin, max_sources=max_targets)
            for t in neowise_targets:
                targets.append({**t, "source": "neowise"})

        if ("tess" in sources or "kepler" in sources) and len(targets) < max_targets:
            # Simple grid search within radius for TESS/Kepler targets
            n_grid = min(10, max_targets - len(targets))
            grid_targets = self._grid_targets(ra, dec, radius_deg, n_grid)
            for t in grid_targets:
                if "tess" in sources:
                    targets.append({**t, "source": "tess"})
                elif "kepler" in sources:
                    targets.append({**t, "source": "kepler"})

        return targets[:max_targets]

    def _grid_targets(self, ra: float, dec: float, radius_deg: float, n: int) -> list[dict]:
        """Generate a simple grid of RA/Dec positions within the radius."""
        targets = []
        step = radius_deg / max(int(n ** 0.5), 1)
        for i in range(int(n ** 0.5)):
            for j in range(int(n ** 0.5)):
                t_ra = ra + (i - n ** 0.5 / 2) * step
                t_dec = dec + (j - n ** 0.5 / 2) * step
                targets.append({"ra": float(t_ra), "dec": float(t_dec)})
        return targets[:n]

    def _fetch_light_curve(self, target: dict) -> dict | None:
        """Fetch light curve for a target, using the appropriate pipeline."""
        source = target.get("source", "neowise")
        ra = target.get("ra")
        dec = target.get("dec")

        try:
            if source == "neowise":
                return self.neowise.fetch_light_curve(ra, dec)
            elif source == "tess":
                return self.tess.fetch_light_curve(ra=ra, dec=dec)
            elif source == "kepler":
                return self.kepler.fetch_light_curve(ra=ra, dec=dec)
        except Exception as exc:
            logger.debug(f"Light curve fetch failed for {target}: {exc}")

        return None

    def _calculate_priority(
        self,
        classifications: dict,
        catalog_match: dict,
        consensus: dict,
    ) -> int:
        """
        Calculate discovery priority score (0–100).

        - Unknown object:              +50 points
        - STRONG_ANOMALY detected:     +30 points
        - High VARDET confidence:      +20 points
        - Ensemble agreement:          +10 points bonus
        """
        score = 0

        if not catalog_match.get("known", True):
            score += 50

        anomaly = classifications.get("ANOMALY-001", {})
        if anomaly.get("label") == "STRONG_ANOMALY":
            score += 30
        elif anomaly.get("label") == "MILD_ANOMALY":
            score += 15

        vardet = classifications.get("VARDET-001", {})
        if vardet.get("confidence", 0) > 0.95:
            score += 20
        elif vardet.get("confidence", 0) > 0.80:
            score += 10

        if consensus.get("agreement_count", 0) >= 3:
            score += 10

        return min(score, 100)


# -------------------------------------------------------------------------
# Discovery Report
# -------------------------------------------------------------------------

class DiscoveryReport:
    """
    Structured result from a CitizenDiscoveryEngine.discover() run.

    Properties:
        candidates  — list of DiscoveryCandidate with is_candidate=True
        known       — list of known objects
        anomalies   — list of candidates with STRONG/MILD_ANOMALY
        meta        — run metadata
    """

    def __init__(self, results: list[DiscoveryCandidate], meta: dict):
        self.results = results
        self.meta = meta

        self.candidates = sorted(
            [r for r in results if r.is_candidate],
            key=lambda c: -c.priority,
        )
        self.known = [r for r in results if not r.is_candidate]
        self.anomalies = [
            r for r in results
            if r.classifications.get("ANOMALY-001", {}).get("label") in (
                "MILD_ANOMALY", "STRONG_ANOMALY"
            )
        ]

    def summary(self) -> str:
        """Quick text summary."""
        lines = [
            f"Discovery Report — RA={self.meta['ra']:.4f}, Dec={self.meta['dec']:.4f}",
            f"  Analyzed: {self.meta['analyzed']} / {self.meta['total_targets']} targets",
            f"  Candidates: {len(self.candidates)}",
            f"  Known: {len(self.known)}",
            f"  Anomalies: {len(self.anomalies)}",
        ]
        if self.candidates:
            lines.append("  Top candidates:")
            for c in self.candidates[:5]:
                lines.append(
                    f"    RA={c.target.get('ra', 0):.4f} Dec={c.target.get('dec', 0):.4f} "
                    f"priority={c.priority} novelty={c.novelty_score:.2f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full dict representation for serialization."""
        return {
            "meta": self.meta,
            "candidates": [self._candidate_to_dict(c) for c in self.candidates],
            "known": [self._candidate_to_dict(c) for c in self.known],
            "anomalies": [self._candidate_to_dict(c) for c in self.anomalies],
            "stats": {
                "total_candidates": len(self.candidates),
                "total_known": len(self.known),
                "total_anomalies": len(self.anomalies),
            },
        }

    def to_response(self) -> dict:
        """API response format (trimmed for bandwidth)."""
        return {
            "meta": self.meta,
            "candidates": [
                {
                    "ra": c.target.get("ra"),
                    "dec": c.target.get("dec"),
                    "source": c.source,
                    "priority": c.priority,
                    "novelty_score": c.novelty_score,
                    "consensus": c.consensus,
                    "anomaly_detected": c.consensus.get("anomaly_detected"),
                    "catalog_matches": len(c.catalog_match.get("matches", [])),
                }
                for c in self.candidates
            ],
            "stats": {
                "total_analyzed": self.meta["analyzed"],
                "total_candidates": len(self.candidates),
                "total_anomalies": len(self.anomalies),
                "elapsed_seconds": self.meta.get("elapsed_seconds"),
            },
        }

    @staticmethod
    def _candidate_to_dict(c: DiscoveryCandidate) -> dict:
        return {
            "target": c.target,
            "light_curve_meta": c.light_curve_meta,
            "classifications": c.classifications,
            "catalog_match": c.catalog_match,
            "consensus": c.consensus,
            "priority": c.priority,
            "source": c.source,
            "is_candidate": c.is_candidate,
            "novelty_score": c.novelty_score,
        }

"""
Discovery Routes — Citizen Discovery Engine API.

POST /api/v2/discover                     — Run full discovery pipeline
GET  /api/v2/discover/candidates          — List pending candidates
POST /api/v2/discover/verify/{id}         — Submit peer verification
POST /api/v2/discover/nl                  — Natural language discovery query
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# -------------------------------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------------------------------

class DiscoverRequest(BaseModel):
    ra: float = Field(..., description="Center RA (degrees, J2000)")
    dec: float = Field(..., description="Center Dec (degrees, J2000)")
    radius_deg: float = Field(0.5, ge=0.01, le=5.0, description="Search radius (degrees)")
    sources: list[Literal["tess", "kepler", "neowise"]] | Literal["all"] = Field(
        "all", description="Data sources to query"
    )
    max_targets: int = Field(50, ge=1, le=500, description="Max targets to analyze")
    generate_report: bool = Field(False, description="Generate AI-powered report (requires Claude API key)")


class VerifyRequest(BaseModel):
    verdict: Literal["confirm", "reject", "unsure"]
    notes: str | None = Field(None, max_length=500)


class NLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query e.g. 'find variable stars near Pleiades'")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post("/discover")
async def run_discovery(request: DiscoverRequest):
    """
    Run the full Citizen Discovery Engine pipeline on a sky region.

    Steps:
    1. Query target catalogs in the specified sky region
    2. Fetch light curves from requested data sources
    3. Run all TinyML models on each target
    4. Cross-match against 6+ catalogs to identify known objects
    5. Score and rank discovery candidates
    6. Optionally generate AI-powered report

    Returns candidates sorted by discovery priority.
    """
    from larun.discovery.engine import CitizenDiscoveryEngine

    engine = CitizenDiscoveryEngine()

    sources = request.sources if isinstance(request.sources, list) else ["tess", "kepler", "neowise"]

    report = engine.discover(
        ra=request.ra,
        dec=request.dec,
        radius_deg=request.radius_deg,
        sources=sources,
        max_targets=request.max_targets,
    )

    response = report.to_response()

    if request.generate_report:
        try:
            from larun.orchestration.report_generator import ReportGenerator
            gen = ReportGenerator(use_claude=True)
            response["ai_report"] = gen.to_ai_report(report.to_dict())
        except Exception as exc:
            logger.warning(f"AI report generation failed: {exc}")

    return response


@router.get("/discover/candidates")
async def list_candidates(
    status: str = Query("candidate", description="Filter by status: candidate, verified, rejected"),
    sort: str = Query("priority", description="Sort by: priority, novelty_score, discovered_at"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get discovery candidates awaiting community verification.

    Returns paginated list sorted by discovery priority.
    """
    from larun.discovery.leaderboard import DiscoveryLeaderboard

    lb = DiscoveryLeaderboard()
    return lb.get_candidates(status=status, sort=sort, limit=limit, offset=offset)


@router.post("/discover/verify/{candidate_id}")
async def verify_candidate(candidate_id: str, request: VerifyRequest):
    """
    Submit a peer verification for a discovery candidate.

    Three independent confirmations → candidate becomes 'verified'.
    One rejection resets the count.
    """
    from larun.discovery.verification import VerificationSystem

    vs = VerificationSystem()

    # Validate UUID format
    try:
        uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id format")

    result = vs.submit_verification(
        discovery_id=candidate_id,
        verdict=request.verdict,
        notes=request.notes,
        verifier_id="anonymous",  # Replace with auth user ID in production
    )

    return result


@router.post("/discover/nl")
async def natural_language_discover(request: NLQueryRequest):
    """
    Parse a natural language query and run the appropriate discovery pipeline.

    Example queries:
    - "Find variable stars near the Pleiades in TESS data"
    - "Check if anything near RA 301.56 Dec 44.46 is anomalous"
    - "Search for transients in NEOWISE data near the galactic center"
    """
    from larun.orchestration.claude_router import ClaudeOrchestrator

    orchestrator = ClaudeOrchestrator()

    if not orchestrator.is_available():
        raise HTTPException(
            status_code=503,
            detail="Natural language parsing requires LARUN_CLAUDE_API_KEY to be set",
        )

    # Parse query to structured action
    action = orchestrator.parse_query(request.query)

    if action.get("action") == "error":
        raise HTTPException(status_code=422, detail=action.get("message", "Parse error"))

    # Execute the parsed action
    result = {"parsed_action": action, "query": request.query}

    if action.get("action") == "discover":
        params = action.get("params", {})
        discover_req = DiscoverRequest(
            ra=params.get("ra", 0),
            dec=params.get("dec", 0),
            radius_deg=params.get("radius_deg", 0.5),
            sources=params.get("sources", "all"),
        )
        result["discovery"] = await run_discovery(discover_req)

    elif action.get("action") == "pipeline":
        params = action.get("params", {})
        result["pipeline_params"] = params
        result["message"] = "Use /api/v2/pipeline/{source} endpoint with these params"

    return result

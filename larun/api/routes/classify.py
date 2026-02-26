"""
Classification Routes — Run TinyML model federation on uploaded light curves.

POST /api/v2/classify      — Run single model
POST /api/v2/federation    — Run all Layer 2 models
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-loaded federation instance
_federation = None


def _get_federation():
    global _federation
    if _federation is None:
        from larun.models.federation import ModelFederation
        _federation = ModelFederation()
    return _federation


# -------------------------------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------------------------------

class LightCurveInput(BaseModel):
    times: list[float] = Field(..., description="Time stamps (MJD or any consistent unit)")
    flux: list[float] = Field(..., description="Flux or magnitude values")
    flux_err: list[float] | None = Field(None, description="Flux uncertainties")
    crowdsap: float | None = Field(None, description="TESS crowding metric (0–1)")
    flfrcsap: float | None = Field(None, description="TESS flux fraction metric (0–1)")
    layer1_results: dict | None = Field(None, description="Pre-computed Layer 1 results from browser")


class ClassifyRequest(BaseModel):
    light_curve: LightCurveInput
    model_id: str = Field(..., description="Model ID e.g. 'VARDET-001'")


class FederationRequest(BaseModel):
    light_curve: LightCurveInput
    models: list[str] | str = Field("all", description="Model IDs or 'all'")
    parallel: bool = Field(True, description="Run models in parallel")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post("/classify")
async def classify_single(request: ClassifyRequest):
    """Run a single Layer 2 model on a light curve."""
    valid_models = ["VARDET-001", "ANOMALY-001", "DEBLEND-001", "PERIODOGRAM-001"]
    if request.model_id not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_id}. Layer 2 models: {valid_models}",
        )

    lc = request.light_curve.model_dump()
    fed = _get_federation()
    results = fed.run_layer2(lc, models=[request.model_id])
    return results.get(request.model_id, {"error": "No result"})


@router.post("/federation")
async def run_federation(request: FederationRequest):
    """
    Run all Layer 2 TinyML models on a light curve.

    Optionally include pre-computed Layer 1 results from the browser.
    """
    lc = request.light_curve.model_dump()
    fed = _get_federation()

    if request.parallel:
        results = fed.run_layer2_parallel(lc, models=request.models)
    else:
        results = fed.run_layer2(lc, models=request.models)

    # Merge layer1 if provided
    if lc.get("layer1_results"):
        results.update(lc["layer1_results"])

    consensus = fed.consensus(results)

    return {
        "model_results": results,
        "consensus": consensus,
    }


@router.post("/federation/report")
async def federation_with_report(request: FederationRequest):
    """
    Run model federation and generate an AI-powered report.
    Requires LARUN_CLAUDE_API_KEY to be set.
    """
    lc = request.light_curve.model_dump()
    fed = _get_federation()
    results = fed.run_layer2_parallel(lc, models=request.models)
    consensus = fed.consensus(results)

    # Generate AI report
    try:
        from larun.orchestration.claude_router import ClaudeOrchestrator
        orchestrator = ClaudeOrchestrator()
        if orchestrator.is_available():
            report = orchestrator.generate_report({"model_results": results, "consensus": consensus})
        else:
            from larun.models.federation import ModelFederation
            report = ModelFederation().summary(results)
    except Exception as exc:
        logger.warning(f"Report generation failed: {exc}")
        report = None

    return {
        "model_results": results,
        "consensus": consensus,
        "report": report,
    }

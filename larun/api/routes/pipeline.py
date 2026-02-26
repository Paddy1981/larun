"""
Pipeline Routes — Fetch and analyze data from NASA archives.

POST /api/v2/pipeline/tess    — Fetch TESS light curve + analyze
POST /api/v2/pipeline/kepler  — Fetch Kepler light curve + analyze
POST /api/v2/pipeline/neowise — Fetch NEOWISE infrared photometry + analyze
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# -------------------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------------------

class TESSRequest(BaseModel):
    tic_id: int | None = Field(None, description="TESS Input Catalog ID")
    ra: float | None = Field(None, description="Right ascension (degrees)")
    dec: float | None = Field(None, description="Declination (degrees)")
    sector: int | None = Field(None, description="TESS sector number (None = all)")
    run_models: bool = Field(True, description="Run TinyML models on fetched data")


class KeplerRequest(BaseModel):
    kic_id: int | None = Field(None, description="Kepler Input Catalog ID")
    ra: float | None = Field(None, description="Right ascension (degrees)")
    dec: float | None = Field(None, description="Declination (degrees)")
    quarter: int | None = Field(None, description="Kepler quarter (None = all)")
    cadence: str = Field("long", description="'long' (30 min) or 'short' (1 min)")
    run_models: bool = Field(True, description="Run TinyML models on fetched data")


class NEOWISERequest(BaseModel):
    ra: float = Field(..., description="Right ascension (degrees)")
    dec: float = Field(..., description="Declination (degrees)")
    radius_arcsec: float = Field(10.0, description="Search radius (arcsec)")
    band: str = Field("w1", description="'w1', 'w2', or 'both'")
    run_models: bool = Field(True, description="Run TinyML models on fetched data")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _run_analysis(lc: dict) -> dict:
    """Run Layer 2 model federation on a fetched light curve."""
    from larun.models.federation import ModelFederation
    fed = ModelFederation()
    results = fed.run_layer2_parallel(lc)
    consensus = fed.consensus(results)
    return {"model_results": results, "consensus": consensus}


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post("/pipeline/tess")
async def tess_pipeline(request: TESSRequest):
    """
    Fetch a TESS light curve and optionally run TinyML models.

    Requires lightkurve to be installed.
    """
    from larun.pipelines.tess import TESSPipeline

    pipeline = TESSPipeline()
    if not pipeline.is_available():
        raise HTTPException(status_code=503, detail="lightkurve not installed on server")

    if request.tic_id is None and (request.ra is None or request.dec is None):
        raise HTTPException(status_code=400, detail="Provide either tic_id or ra+dec")

    lc = pipeline.fetch_light_curve(
        tic_id=request.tic_id,
        ra=request.ra,
        dec=request.dec,
        sector=request.sector,
    )

    if lc is None:
        raise HTTPException(status_code=404, detail="No TESS data found for this target")

    response = {
        "light_curve": {
            "n_points": lc["meta"]["n_points"],
            "time_span_days": lc["meta"]["time_span_days"],
            "meta": lc["meta"],
        }
    }

    if request.run_models:
        response["analysis"] = _run_analysis(lc)

    return response


@router.post("/pipeline/kepler")
async def kepler_pipeline(request: KeplerRequest):
    """Fetch a Kepler light curve and optionally run TinyML models."""
    from larun.pipelines.kepler import KeplerPipeline

    pipeline = KeplerPipeline()
    if not pipeline.is_available():
        raise HTTPException(status_code=503, detail="lightkurve not installed on server")

    if request.kic_id is None and (request.ra is None or request.dec is None):
        raise HTTPException(status_code=400, detail="Provide either kic_id or ra+dec")

    lc = pipeline.fetch_light_curve(
        kic_id=request.kic_id,
        ra=request.ra,
        dec=request.dec,
        quarter=request.quarter,
        cadence=request.cadence,
    )

    if lc is None:
        raise HTTPException(status_code=404, detail="No Kepler data found for this target")

    response = {
        "light_curve": {"n_points": lc["meta"]["n_points"], "meta": lc["meta"]}
    }

    if request.run_models:
        response["analysis"] = _run_analysis(lc)

    return response


@router.post("/pipeline/neowise")
async def neowise_pipeline(request: NEOWISERequest):
    """
    Fetch a NEOWISE infrared light curve and optionally run TinyML models.

    This is the core data source for VARnet-inspired discovery.
    W1 (3.4μm) and W2 (4.6μm) infrared bands, ~13 epochs/year, 10-year baseline.
    """
    from larun.pipelines.neowise import NEOWISEPipeline

    pipeline = NEOWISEPipeline()

    lc = pipeline.fetch_light_curve(
        ra=request.ra,
        dec=request.dec,
        radius_arcsec=request.radius_arcsec,
        band=request.band,
    )

    if lc is None:
        raise HTTPException(
            status_code=404,
            detail=f"No NEOWISE data found at ra={request.ra:.4f}, dec={request.dec:.4f}",
        )

    response = {
        "light_curve": {
            "n_points": lc["meta"]["n_points"],
            "time_span_days": lc["meta"]["time_span_days"],
            "meta": lc["meta"],
        }
    }

    if request.run_models:
        response["analysis"] = _run_analysis(lc)

    return response

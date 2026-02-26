"""
Catalog Routes — Browse and search astronomical catalogs.

GET  /api/v2/catalog/cross-match        — Cross-match position against all catalogs
GET  /api/v2/catalog/varwise/search     — Search VarWISE catalog
GET  /api/v2/catalog/varwise/{id}       — Get VarWISE object details + LARUN analysis
GET  /api/v2/catalog/varwise/stats      — VarWISE catalog statistics
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/catalog/cross-match")
async def cross_match(
    ra: float = Query(..., description="Right ascension (degrees)"),
    dec: float = Query(..., description="Declination (degrees)"),
    radius_arcsec: float = Query(3.0, description="Match radius (arcsec)"),
    catalogs: str = Query("all", description="Comma-separated catalog IDs or 'all'"),
):
    """
    Cross-match coordinates against 6+ astronomical catalogs.

    Returns known matches and a novelty score (0=known, 1=completely new).
    Used by the Citizen Discovery Engine to determine if a detection is new.
    """
    from larun.pipelines.cross_match import CrossMatchPipeline

    pipeline = CrossMatchPipeline()
    cat_list = catalogs.split(",") if catalogs != "all" else "all"

    result = pipeline.cross_match(ra, dec, radius_arcsec, catalogs=cat_list)
    return result


@router.get("/catalog/varwise/stats")
async def varwise_stats():
    """Get VarWISE catalog statistics (if loaded)."""
    from larun.pipelines.varwise import VarWISEBrowser

    browser = VarWISEBrowser.instance()
    if browser is None:
        return {
            "loaded": False,
            "message": "VarWISE catalog not loaded. Contact larun.space admin.",
        }
    return browser.stats()


@router.get("/catalog/varwise/search")
async def varwise_search(
    ra: float | None = Query(None, description="RA center (degrees)"),
    dec: float | None = Query(None, description="Dec center (degrees)"),
    radius_arcmin: float = Query(5.0, description="Search radius (arcmin)"),
    var_class: str | None = Query(None, description="Filter by class: TRANSIENT, PULSATOR, ECLIPSING"),
    min_amplitude: float | None = Query(None, description="Minimum variability amplitude (mag)"),
    limit: int = Query(50, ge=1, le=500),
):
    """
    Search the VarWISE catalog of 1.5M variable sources (Paz, 2025).

    Requires VarWISE catalog to be loaded on the server.
    """
    from larun.pipelines.varwise import VarWISEBrowser

    browser = VarWISEBrowser.instance()
    if browser is None:
        raise HTTPException(
            status_code=503,
            detail="VarWISE catalog not loaded. Check /api/v2/catalog/varwise/stats",
        )

    if ra is not None and dec is not None:
        results = browser.search_by_position(ra, dec, radius_arcmin)
    elif var_class is not None:
        results = browser.search_by_class(var_class, limit=limit)
    elif min_amplitude is not None:
        results = browser.search_by_properties(min_amplitude=min_amplitude, var_class=var_class, limit=limit)
    else:
        raise HTTPException(status_code=400, detail="Provide ra+dec or var_class or min_amplitude")

    return {"count": len(results), "objects": results[:limit]}


@router.get("/catalog/varwise/{varwise_id}")
async def varwise_object(
    varwise_id: int,
    run_models: bool = Query(False, description="Run LARUN model federation on this object"),
):
    """
    Get details for a specific VarWISE object.

    Optionally run all 12 LARUN models for deeper characterization beyond VARnet's 4 classes.
    """
    from larun.pipelines.varwise import VarWISEBrowser

    browser = VarWISEBrowser.instance()
    if browser is None:
        raise HTTPException(status_code=503, detail="VarWISE catalog not loaded")

    obj = browser._row_to_dict(varwise_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"VarWISE object {varwise_id} not found")

    response = {"object": obj}

    if run_models:
        larun_results = browser.run_larun_models(varwise_id)
        response["larun_analysis"] = larun_results
        response["comparison"] = {
            "varnet_class": obj.get("varnet_class"),
            "larun_consensus": larun_results.get("consensus", {}),
            "note": "VARnet uses 4 classes; LARUN adds 8+ specialized models for deeper characterization",
        }

    return response

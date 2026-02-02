"""
Analysis Schemas

Pydantic models for analysis request/response validation.

Agent: BETA
"""

from datetime import datetime
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field


class VettingTestSchema(BaseModel):
    """Individual vetting test result."""

    test_name: str
    flag: str  # PASS, WARNING, FAIL
    confidence: float = Field(..., ge=0.0, le=1.0)
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: str


class VettingResultSchema(BaseModel):
    """Combined vetting results."""

    disposition: str  # PLANET_CANDIDATE, LIKELY_FALSE_POSITIVE, INCONCLUSIVE
    confidence: float = Field(..., ge=0.0, le=1.0)
    tests_passed: int
    tests_failed: int
    tests_warning: Optional[int] = 0
    odd_even: Optional[VettingTestSchema] = None
    v_shape: Optional[VettingTestSchema] = None
    secondary_eclipse: Optional[VettingTestSchema] = None
    recommendation: str


class PeriodogramSchema(BaseModel):
    """Periodogram data for visualization."""

    periods: List[float]
    powers: List[float]
    best_period: float
    best_power: Optional[float] = None
    top_periods: Optional[List[float]] = None
    top_powers: Optional[List[float]] = None


class PhaseFoldedSchema(BaseModel):
    """Phase-folded light curve data."""

    phase: List[float]
    flux: List[float]
    flux_err: Optional[List[float]] = None
    binned_phase: List[float]
    binned_flux: List[float]
    binned_flux_err: Optional[List[float]] = None


class LightCurveSchema(BaseModel):
    """Raw light curve data."""

    time: List[float]
    flux: List[float]
    flux_err: Optional[List[float]] = None
    quality: Optional[List[int]] = None


class AnalyzeRequest(BaseModel):
    """Request body for analysis submission."""

    tic_id: str = Field(
        ...,
        description="TESS Input Catalog ID (e.g., 'TIC 12345678' or '12345678')",
        pattern=r"^(TIC\s*)?\d+$",
    )


class AnalyzeResponse(BaseModel):
    """Response for analysis submission."""

    analysis_id: int
    status: str = "pending"
    message: str = "Analysis queued"


class AnalysisResultDetail(BaseModel):
    """Detailed analysis result when completed."""

    detection: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    period_days: Optional[float] = None
    depth_ppm: Optional[float] = None
    duration_hours: Optional[float] = None
    epoch_btjd: Optional[float] = None
    snr: Optional[float] = None

    vetting: Optional[VettingResultSchema] = None
    periodogram: Optional[PeriodogramSchema] = None
    phase_folded: Optional[PhaseFoldedSchema] = None
    raw_lightcurve: Optional[LightCurveSchema] = None

    sectors_used: List[int] = []
    processing_time_seconds: Optional[float] = None


class AnalysisResult(BaseModel):
    """Complete analysis response."""

    id: int
    tic_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None

    # Only present when completed
    result: Optional[AnalysisResultDetail] = None

    # Only present when failed
    error: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysesListResponse(BaseModel):
    """Response for analysis list endpoint."""

    analyses: List[AnalysisResult]
    total: int
    page: int
    per_page: int


class DeleteAnalysisResponse(BaseModel):
    """Response for analysis deletion."""

    message: str = "Analysis deleted successfully"


class AnalysisStatusUpdate(BaseModel):
    """Internal model for analysis status updates."""

    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

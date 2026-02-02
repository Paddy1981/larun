"""
Analysis Routes

Endpoints for submitting and retrieving exoplanet detection analyses.

Agent: BETA
"""

import re
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import settings
from src.api.models.database import get_db
from src.api.models.user import User
from src.api.models.analysis import Analysis, AnalysisStatus
from src.api.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    AnalysisResultDetail,
    AnalysesListResponse,
    DeleteAnalysisResponse,
    VettingResultSchema,
)
from src.api.dependencies import (
    get_current_user,
    check_usage_limit,
    get_detection_service,
    MockDetectionService,
)

router = APIRouter()


def normalize_tic_id(tic_id: str) -> str:
    """Normalize TIC ID to standard format (digits only)."""
    return re.sub(r"[^0-9]", "", tic_id)


async def run_analysis(
    analysis_id: int,
    tic_id: str,
    detection_service: MockDetectionService,
) -> None:
    """
    Background task to run analysis.

    This will be replaced with proper job queue (Redis/Celery)
    when integrating with ALPHA's DetectionService.
    """
    from src.api.models.database import get_db_context

    async with get_db_context() as db:
        # Get the analysis
        result = await db.execute(
            select(Analysis).where(Analysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()

        if analysis is None:
            return

        try:
            # Update status to processing
            analysis.status = AnalysisStatus.PROCESSING
            analysis.started_at = datetime.utcnow()
            await db.flush()

            # Run detection (mock for now)
            detection_result = await detection_service.analyze(tic_id)

            # Update analysis with results
            analysis.set_result(
                detection=detection_result.get("detection", False),
                confidence=detection_result.get("confidence", 0.0),
                period_days=detection_result.get("period_days"),
                depth_ppm=detection_result.get("depth_ppm"),
                duration_hours=detection_result.get("duration_hours"),
                epoch_btjd=detection_result.get("epoch_btjd"),
                snr=detection_result.get("snr"),
                result_json=detection_result,
                processing_time=detection_result.get("processing_time_seconds"),
                sectors=detection_result.get("sectors_used"),
            )

        except Exception as e:
            analysis.set_error(str(e))

        await db.commit()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit analysis request",
    description="Submit a TIC ID for exoplanet transit detection analysis.",
)
async def submit_analysis(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    subscription: Annotated[None, Depends(check_usage_limit)],
    detection_service: Annotated[MockDetectionService, Depends(get_detection_service)],
) -> AnalyzeResponse:
    """
    Submit a TIC ID for analysis.

    - **tic_id**: TESS Input Catalog ID (e.g., "TIC 470710327" or "470710327")

    Returns an analysis ID to poll for results.
    Requires active subscription with available usage quota.
    """
    # Normalize TIC ID
    normalized_tic = normalize_tic_id(request.tic_id)

    if not normalized_tic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_tic_id",
                    "message": "Invalid TIC ID format",
                }
            },
        )

    # Create analysis record
    analysis = Analysis(
        user_id=current_user.id,
        tic_id=normalized_tic,
        status=AnalysisStatus.PENDING,
    )

    db.add(analysis)
    await db.flush()
    await db.refresh(analysis)

    # Increment usage (subscription already validated by check_usage_limit)
    if current_user.subscription:
        current_user.subscription.increment_usage()
        await db.flush()

    # Queue background analysis
    background_tasks.add_task(
        run_analysis,
        analysis.id,
        normalized_tic,
        detection_service,
    )

    return AnalyzeResponse(
        analysis_id=analysis.id,
        status="pending",
        message="Analysis queued. Poll GET /analyze/{id} for results.",
    )


@router.get(
    "/analyze/{analysis_id}",
    response_model=AnalysisResult,
    summary="Get analysis result",
    description="Get the status and results of a submitted analysis.",
)
async def get_analysis(
    analysis_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AnalysisResult:
    """
    Get analysis status and results.

    - **analysis_id**: ID returned from POST /analyze

    Returns current status and results when completed.
    """
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_found",
                    "message": "Analysis not found",
                }
            },
        )

    # Build response
    response = AnalysisResult(
        id=analysis.id,
        tic_id=analysis.tic_id,
        status=analysis.status.value,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
    )

    # Add result details if completed
    if analysis.status == AnalysisStatus.COMPLETED:
        sectors = []
        if analysis.sectors_used:
            sectors = [int(s) for s in analysis.sectors_used.split(",")]

        # Build vetting result if available
        vetting = None
        if analysis.result_json and "vetting" in analysis.result_json:
            v = analysis.result_json["vetting"]
            vetting = VettingResultSchema(
                disposition=v.get("disposition", "INCONCLUSIVE"),
                confidence=v.get("confidence", 0.0),
                tests_passed=v.get("tests_passed", 0),
                tests_failed=v.get("tests_failed", 0),
                recommendation=v.get("recommendation", ""),
            )

        response.result = AnalysisResultDetail(
            detection=analysis.detection or False,
            confidence=analysis.confidence or 0.0,
            period_days=analysis.period_days,
            depth_ppm=analysis.depth_ppm,
            duration_hours=analysis.duration_hours,
            epoch_btjd=analysis.epoch_btjd,
            snr=analysis.snr,
            vetting=vetting,
            sectors_used=sectors,
            processing_time_seconds=analysis.processing_time_seconds,
        )

    elif analysis.status == AnalysisStatus.FAILED:
        response.error = analysis.error_message

    return response


@router.get(
    "/analyses",
    response_model=AnalysesListResponse,
    summary="List user analyses",
    description="Get paginated list of user's analysis history.",
)
async def list_analyses(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status (pending, processing, completed, failed)",
    ),
) -> AnalysesListResponse:
    """
    List user's analyses with pagination.

    - **page**: Page number (default 1)
    - **per_page**: Items per page (default 10, max 100)
    - **status**: Optional status filter
    """
    # Build query
    query = select(Analysis).where(Analysis.user_id == current_user.id)

    if status_filter:
        try:
            status_enum = AnalysisStatus(status_filter)
            query = query.where(Analysis.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "invalid_status",
                        "message": f"Invalid status: {status_filter}",
                    }
                },
            )

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    offset = (page - 1) * per_page
    query = query.order_by(Analysis.created_at.desc()).offset(offset).limit(per_page)
    result = await db.execute(query)
    analyses = result.scalars().all()

    # Build response
    analysis_results = []
    for analysis in analyses:
        ar = AnalysisResult(
            id=analysis.id,
            tic_id=analysis.tic_id,
            status=analysis.status.value,
            created_at=analysis.created_at,
            completed_at=analysis.completed_at,
        )

        if analysis.status == AnalysisStatus.COMPLETED:
            sectors = []
            if analysis.sectors_used:
                sectors = [int(s) for s in analysis.sectors_used.split(",")]

            ar.result = AnalysisResultDetail(
                detection=analysis.detection or False,
                confidence=analysis.confidence or 0.0,
                period_days=analysis.period_days,
                depth_ppm=analysis.depth_ppm,
                duration_hours=analysis.duration_hours,
                snr=analysis.snr,
                sectors_used=sectors,
                processing_time_seconds=analysis.processing_time_seconds,
            )
        elif analysis.status == AnalysisStatus.FAILED:
            ar.error = analysis.error_message

        analysis_results.append(ar)

    return AnalysesListResponse(
        analyses=analysis_results,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.delete(
    "/analyses/{analysis_id}",
    response_model=DeleteAnalysisResponse,
    summary="Delete analysis",
    description="Delete an analysis record.",
)
async def delete_analysis(
    analysis_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DeleteAnalysisResponse:
    """
    Delete an analysis.

    - **analysis_id**: ID of analysis to delete

    Only the owner can delete their analyses.
    """
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
    )
    analysis = result.scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_found",
                    "message": "Analysis not found",
                }
            },
        )

    await db.delete(analysis)
    await db.flush()

    return DeleteAnalysisResponse(message="Analysis deleted successfully")

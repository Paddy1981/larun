"""
Analysis Model

SQLAlchemy model for exoplanet detection analysis jobs and results.

Agent: BETA
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any, Dict, TYPE_CHECKING

from sqlalchemy import String, DateTime, Float, Integer, Text, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.models.database import Base

if TYPE_CHECKING:
    from src.api.models.user import User


class AnalysisStatus(str, Enum):
    """Status of an analysis job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Analysis(Base):
    """
    Analysis job model.

    Stores analysis requests and results from the detection engine.
    Each analysis is associated with a user and tracks a specific TIC ID.
    """

    __tablename__ = "analyses"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign key to user
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Target identification
    tic_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # Job status
    status: Mapped[AnalysisStatus] = mapped_column(
        String(20),
        default=AnalysisStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Detection results (populated when completed)
    detection: Mapped[Optional[bool]] = mapped_column(nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Transit parameters
    period_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    depth_ppm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    epoch_btjd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Full result JSON (from DetectionService)
    result_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Error message if failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Processing metadata
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    sectors_used: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # Comma-separated list of sectors

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")

    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, tic_id={self.tic_id}, status={self.status})>"

    @property
    def is_complete(self) -> bool:
        """Check if analysis has completed (success or failure)."""
        return self.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)

    @property
    def is_successful(self) -> bool:
        """Check if analysis completed successfully."""
        return self.status == AnalysisStatus.COMPLETED

    def set_result(
        self,
        detection: bool,
        confidence: float,
        period_days: Optional[float] = None,
        depth_ppm: Optional[float] = None,
        duration_hours: Optional[float] = None,
        epoch_btjd: Optional[float] = None,
        snr: Optional[float] = None,
        result_json: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None,
        sectors: Optional[list[int]] = None,
    ) -> None:
        """Set analysis results after successful detection."""
        self.status = AnalysisStatus.COMPLETED
        self.detection = detection
        self.confidence = confidence
        self.period_days = period_days
        self.depth_ppm = depth_ppm
        self.duration_hours = duration_hours
        self.epoch_btjd = epoch_btjd
        self.snr = snr
        self.result_json = result_json
        self.processing_time_seconds = processing_time
        if sectors:
            self.sectors_used = ",".join(str(s) for s in sectors)
        self.completed_at = datetime.utcnow()

    def set_error(self, error_message: str) -> None:
        """Set analysis as failed with error message."""
        self.status = AnalysisStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()

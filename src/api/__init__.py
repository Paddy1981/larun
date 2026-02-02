"""
LARUN API Module

FastAPI-based REST API for the LARUN exoplanet detection platform.
Provides endpoints for authentication, analysis, and user management.

Agent: BETA
Branch: claude/mvp-beta-backend
"""

from src.api.main import app

__all__ = ["app"]
__version__ = "0.1.0"

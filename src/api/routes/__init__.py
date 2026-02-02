"""
API Routes Module

Contains all FastAPI route handlers organized by feature.
"""

from src.api.routes import auth, analysis, user, subscription

__all__ = ["auth", "analysis", "user", "subscription"]

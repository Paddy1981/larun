"""
LARUN API v2 — FastAPI application for model federation and discovery.

Mounts onto the existing FastAPI app or runs as a standalone service.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from larun.api.routes import classify, pipeline, discover, catalog, leaderboard

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the LARUN v2 FastAPI application."""

    app = FastAPI(
        title="LARUN.SPACE API v2",
        description="Federation of TinyML for Space Science",
        version="2.0.0",
        docs_url="/api/v2/docs",
        redoc_url="/api/v2/redoc",
        openapi_url="/api/v2/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(classify.router, prefix="/api/v2", tags=["Classification"])
    app.include_router(pipeline.router, prefix="/api/v2", tags=["Data Pipelines"])
    app.include_router(discover.router, prefix="/api/v2", tags=["Discovery"])
    app.include_router(catalog.router, prefix="/api/v2", tags=["Catalog"])
    app.include_router(leaderboard.router, prefix="/api/v2", tags=["Leaderboard"])

    @app.get("/api/v2/health")
    async def health():
        return {"status": "ok", "version": "2.0.0", "platform": "larun.space"}

    @app.get("/api/v2/models")
    async def list_models():
        """List available TinyML models."""
        import json
        from pathlib import Path
        registry_path = Path(__file__).parent.parent.parent / "models" / "MODEL_REGISTRY.json"
        if registry_path.exists():
            with open(registry_path) as f:
                return json.load(f)
        return {"models": []}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

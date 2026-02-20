#!/usr/bin/env python3
"""
LARUN TinyML - FastAPI REST Service (Minimal for Railway)
=========================================================
Lightweight API for TinyML model inference only.

Endpoints:
    GET  /                    - API info
    GET  /health              - Health check
    GET  /api/models          - List TinyML models
    POST /api/analyze/tinyml  - TinyML inference
    GET  /api/quota/{user_id} - Check quota
    GET  /api/user/{user_id}/history - User history
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import os
from datetime import datetime
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="LARUN TinyML API",
    description="REST API for TinyML astronomical model inference",
    version="2.0.0",
    contact={
        "name": "Padmanaban Veeraragavalu",
        "email": "larun@example.com"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Cloud SaaS Integration
# ============================================================================
try:
    from cloud_endpoints import setup_cloud_endpoints
    setup_cloud_endpoints(app)
    logger.info("✅ Cloud endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Cloud endpoints not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to load cloud endpoints: {e}")

# ============================================================================
# Basic Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "LARUN TinyML API",
        "version": "2.0.0",
        "description": "Astronomical TinyML model inference",
        "endpoints": {
            "/health": "Health check",
            "/api/models": "List available TinyML models",
            "/api/analyze/tinyml": "TinyML inference (POST with file)",
            "/api/quota/{user_id}": "Check user quota",
            "/api/user/{user_id}/history": "User analysis history"
        },
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    # Check if models directory exists
    models_available = []
    try:
        from pathlib import Path
        nodes_dir = Path(__file__).parent / "nodes"
        if nodes_dir.exists():
            for node_dir in nodes_dir.iterdir():
                if node_dir.is_dir():
                    model_dir = node_dir / "model"
                    if model_dir.exists():
                        tflite_files = list(model_dir.glob("*.tflite"))
                        if tflite_files:
                            models_available.append(node_dir.name)
    except Exception as e:
        logger.error(f"Error checking models: {e}")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models_available),
        "models": models_available,
        "environment": {
            "python": "3.11",
            "platform": "Railway",
            "region": os.getenv("RAILWAY_REGION", "unknown")
        }
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

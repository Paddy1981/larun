"""
LARUN.SPACE Cloud - Backend Integration
========================================
Add these endpoints to api.py for Cloud SaaS functionality.

Author: LARUN Engineering
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import yaml
import numpy as np
from pathlib import Path

load_dotenv()

# ============================================================================
# Supabase Client
# ============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# Models Registry
# ============================================================================

def load_models_registry():
    """Load TinyML models from registry.yaml"""
    registry_path = Path(__file__).parent / "nodes" / "registry.yaml"
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    models = []
    for node_id, node_data in registry.get('nodes', {}).items():
        if node_data.get('status') in ['enabled', 'installed'] and node_data.get('has_model', True):
            models.append({
                "id": node_id,
                "name": node_data['name'],
                "category": node_data['category'],
                "description": node_data['description'],
                "model_size_kb": node_data['model_size_kb'],
                "version": node_data['version']
            })

    return models

# ============================================================================
# Quota Management
# ============================================================================

QUOTA_LIMITS = {
    "free": 5,
    "monthly": 50,
    "annual": None  # Unlimited
}

async def check_quota(user_id: str) -> Dict[str, Any]:
    """Check if user has quota remaining"""
    # Get current month
    current_month = datetime.now().strftime("%Y-%m")

    # Get user's subscription tier
    user_result = supabase.table("users").select("subscription_tier").eq("id", user_id).execute()

    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found")

    tier = user_result.data[0]["subscription_tier"]
    limit = QUOTA_LIMITS.get(tier, 5)

    # Get usage for current month
    usage_result = supabase.table("usage_quotas").select("*").eq("user_id", user_id).eq("month", current_month).execute()

    if usage_result.data:
        usage = usage_result.data[0]
        count = usage["analyses_count"]
    else:
        count = 0

    # Check if quota exceeded
    if limit is not None and count >= limit:
        return {
            "allowed": False,
            "tier": tier,
            "used": count,
            "limit": limit,
            "message": f"Monthly quota exceeded ({count}/{limit}). Upgrade to continue."
        }

    return {
        "allowed": True,
        "tier": tier,
        "used": count,
        "limit": limit,
        "remaining": None if limit is None else (limit - count)
    }

async def increment_usage(user_id: str):
    """Increment user's analysis count"""
    current_month = datetime.now().strftime("%Y-%m")

    # Call Supabase function
    supabase.rpc("increment_usage", {
        "p_user_id": user_id,
        "p_month": current_month
    }).execute()

# ============================================================================
# Request/Response Models
# ============================================================================

class TinyMLRequest(BaseModel):
    """TinyML analysis request"""
    model_id: str
    user_id: str
    data: List[float]  # Light curve flux values

class TinyMLResponse(BaseModel):
    """TinyML analysis response"""
    model_id: str
    classification: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    timestamp: str

# ============================================================================
# Cloud API Endpoints (ADD THESE TO api.py)
# ============================================================================

def setup_cloud_endpoints(app: FastAPI):
    """Add Cloud SaaS endpoints to the FastAPI app"""

    @app.get("/api/models")
    async def list_models():
        """List available TinyML models"""
        try:
            models = load_models_registry()
            return {
                "status": "success",
                "count": len(models),
                "models": models
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/api/quota/{user_id}")
    async def get_quota(user_id: str):
        """Get user's quota information"""
        try:
            quota_info = await check_quota(user_id)
            return quota_info
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/api/analyze/tinyml")
    async def analyze_tinyml(
        file: UploadFile = File(...),
        model_id: str = Header(...),
        user_id: str = Header(...)
    ):
        """
        Analyze FITS file with TinyML model

        Headers:
          - model-id: Model to use (e.g., EXOPLANET-001)
          - user-id: User UUID from Supabase auth
        """
        start_time = datetime.now()

        try:
            # 1. Check quota
            quota = await check_quota(user_id)
            if not quota["allowed"]:
                raise HTTPException(status_code=429, detail=quota["message"])

            # 2. Load model
            from src.nodes.base import TFLiteNode

            model_path_map = {
                "EXOPLANET-001": "nodes/exoplanet",
                "VSTAR-001": "nodes/variable_star",
                "FLARE-001": "nodes/flare",
                "ASTERO-001": "nodes/asteroseismo",
                "SUPERNOVA-001": "nodes/supernova",
                "GALAXY-001": "nodes/galaxy",
                "SPECTYPE-001": "nodes/spectral_type",
                "MICROLENS-001": "nodes/microlensing"
            }

            if model_id not in model_path_map:
                raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

            model_path = Path(__file__).parent / model_path_map[model_id]

            # 3. Load detector for the model
            detector_file = model_path / "src" / "detector.py"
            if not detector_file.exists():
                raise HTTPException(status_code=500, detail=f"Model detector not found: {model_id}")

            # Import detector dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location("detector", detector_file)
            detector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detector_module)

            # Get detector class (assumes it's the first class in the module)
            detector_class = None
            for name in dir(detector_module):
                obj = getattr(detector_module, name)
                if isinstance(obj, type) and hasattr(obj, 'detect'):
                    detector_class = obj
                    break

            if not detector_class:
                raise HTTPException(status_code=500, detail=f"Detector class not found for {model_id}")

            detector = detector_class()

            # 4. Read FITS file
            file_bytes = await file.read()

            # Parse FITS (simplified - you may need astropy.io.fits for real FITS)
            # For now, assume file contains flux array as text or binary
            try:
                flux_data = np.frombuffer(file_bytes, dtype=np.float32)
            except:
                # If not binary, try parsing as text
                flux_data = np.array([float(x) for x in file_bytes.decode().strip().split()])

            # 5. Run inference
            result = detector.detect(flux_data)

            # 6. Save to database
            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            analysis_data = {
                "user_id": user_id,
                "model_id": model_id,
                "result": result.__dict__ if hasattr(result, '__dict__') else {},
                "classification": result.classification if hasattr(result, 'classification') else "unknown",
                "confidence": float(result.confidence) if hasattr(result, 'confidence') else 0.0,
                "inference_time_ms": inference_time,
                "created_at": datetime.now().isoformat()
            }

            supabase.table("analyses").insert(analysis_data).execute()

            # 7. Increment usage
            await increment_usage(user_id)

            # 8. Return result
            return {
                "status": "success",
                "model_id": model_id,
                "classification": analysis_data["classification"],
                "confidence": analysis_data["confidence"],
                "probabilities": result.probabilities if hasattr(result, 'probabilities') else {},
                "inference_time_ms": inference_time,
                "quota": await check_quota(user_id)
            }

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


    @app.get("/api/user/{user_id}/history")
    async def get_user_history(user_id: str, limit: int = 10):
        """Get user's analysis history"""
        try:
            result = supabase.table("analyses") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()

            return {
                "status": "success",
                "count": len(result.data),
                "analyses": result.data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

# ============================================================================
# Usage: Add to api.py
# ============================================================================
"""
To integrate, add this to your api.py:

from cloud_endpoints import setup_cloud_endpoints

# After creating the FastAPI app:
app = FastAPI(...)

# Add cloud endpoints:
setup_cloud_endpoints(app)
"""

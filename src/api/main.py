"""
LARUN API Main Application

FastAPI application entry point with router configuration,
middleware setup, and lifecycle management.

Agent: BETA
Branch: claude/mvp-beta-backend
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.config import settings
from src.api.models.database import init_db
from src.api.routes import auth, analysis, user, subscription


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan management.

    Handles startup and shutdown tasks:
    - Initialize database tables on startup
    - Clean up resources on shutdown
    """
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    print("Database initialized")

    yield

    # Shutdown
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    LARUN Exoplanet Detection API

    A REST API for the LARUN platform that enables users to:
    - Register and authenticate
    - Submit exoplanet detection analysis requests
    - Retrieve analysis results
    - Manage subscriptions

    ## Authentication
    Most endpoints require JWT bearer token authentication.
    Obtain a token via the /api/v1/auth/login endpoint.

    ## Rate Limiting
    API requests are rate-limited based on subscription tier.
    """,
    version=settings.APP_VERSION,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors with structured response."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "validation_error",
                "message": "Invalid request data",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected errors gracefully."""
    # In debug mode, include error details
    if settings.DEBUG:
        message = str(exc)
    else:
        message = "An internal error occurred"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "internal_error",
                "message": message,
            }
        },
    )


# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"],
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["Analysis"],
)

app.include_router(
    user.router,
    prefix=f"{settings.API_V1_PREFIX}/user",
    tags=["User"],
)

app.include_router(
    subscription.router,
    prefix=f"{settings.API_V1_PREFIX}/subscription",
    tags=["Subscription"],
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns service status for load balancers and monitoring.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/api/docs" if settings.DEBUG else "disabled",
        "api_prefix": settings.API_V1_PREFIX,
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )

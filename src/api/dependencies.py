"""
API Dependencies

FastAPI dependency injection functions for authentication,
database sessions, and service access.

Agent: BETA
"""

from datetime import datetime, timedelta
from typing import Optional, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import settings
from src.api.models.database import get_db
from src.api.models.user import User
from src.api.models.subscription import Subscription
from src.api.schemas.auth import TokenData


# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


# JWT utilities
def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: Optional[int] = payload.get("sub")
        email: Optional[str] = payload.get("email")
        if user_id is None:
            return None
        return TokenData(user_id=user_id, email=email)
    except JWTError:
        return None


# Authentication dependencies
async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Get the current authenticated user from JWT token.

    Raises HTTPException if token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": {
                "code": "unauthorized",
                "message": "Could not validate credentials",
            }
        },
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_access_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "forbidden",
                    "message": "User account is deactivated",
                }
            },
        )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current user, ensuring they are active."""
    return current_user


# Subscription dependencies
async def get_user_subscription(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Optional[Subscription]:
    """Get the current user's subscription if any."""
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == current_user.id)
    )
    return result.scalar_one_or_none()


async def require_active_subscription(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Subscription:
    """
    Require user to have an active subscription.

    Raises HTTPException if no active subscription.
    """
    subscription = await get_user_subscription(current_user, db)

    if subscription is None or not subscription.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "forbidden",
                    "message": "Active subscription required",
                }
            },
        )

    return subscription


async def check_usage_limit(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Subscription:
    """
    Check if user has remaining analyses in their quota.

    Raises HTTPException if usage limit exceeded.
    """
    subscription = await require_active_subscription(current_user, db)

    if not subscription.can_analyze:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "usage_limit_exceeded",
                    "message": f"Monthly analysis limit ({subscription.analysis_limit}) reached",
                    "details": {
                        "limit": subscription.analysis_limit,
                        "used": subscription.analyses_this_period,
                        "period_end": subscription.current_period_end.isoformat()
                        if subscription.current_period_end
                        else None,
                    },
                }
            },
        )

    return subscription


# Mock Detection Service (until ALPHA delivers)
class MockDetectionService:
    """
    Mock detection service for development.

    Will be replaced with real DetectionService from ALPHA.
    """

    async def analyze(self, tic_id: str) -> dict:
        """Mock analysis that returns placeholder data."""
        import random

        return {
            "tic_id": tic_id,
            "detection": random.choice([True, False]),
            "confidence": random.uniform(0.5, 0.99),
            "period_days": random.uniform(1.0, 50.0) if random.random() > 0.3 else None,
            "depth_ppm": random.uniform(100, 5000) if random.random() > 0.3 else None,
            "duration_hours": random.uniform(1.0, 10.0) if random.random() > 0.3 else None,
            "snr": random.uniform(5.0, 50.0),
            "sectors_used": [1, 2, 3],
            "processing_time_seconds": random.uniform(10.0, 30.0),
            "vetting": {
                "disposition": random.choice(
                    ["PLANET_CANDIDATE", "LIKELY_FALSE_POSITIVE", "INCONCLUSIVE"]
                ),
                "confidence": random.uniform(0.5, 0.99),
                "tests_passed": random.randint(1, 3),
                "tests_failed": random.randint(0, 2),
                "recommendation": "Recommend follow-up observations",
            },
        }

    async def get_status(self) -> dict:
        """Get mock service status."""
        return {
            "status": "healthy",
            "version": "mock-0.1.0",
        }


# Detection service dependency
_detection_service: Optional[MockDetectionService] = None


def get_detection_service() -> MockDetectionService:
    """
    Get the detection service instance.

    Currently returns mock service.
    Will be replaced with real DetectionService when ALPHA delivers.
    """
    global _detection_service
    if _detection_service is None:
        _detection_service = MockDetectionService()
    return _detection_service

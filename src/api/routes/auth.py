"""
Authentication Routes

Endpoints for user registration, login, and password management.

Agent: BETA
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import settings
from src.api.models.database import get_db
from src.api.models.user import User
from src.api.schemas.auth import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    UserResponse,
)
from src.api.dependencies import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter()


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password.",
)
async def register(
    request: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegisterResponse:
    """
    Register a new user.

    - **email**: Valid email address (must be unique)
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit)
    - **name**: Optional display name
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": {
                    "code": "email_exists",
                    "message": "An account with this email already exists",
                }
            },
        )

    # Create new user
    user = User(
        email=request.email,
        password_hash=get_password_hash(request.password),
        name=request.name,
        is_active=True,
        is_verified=False,  # Email verification not implemented yet
    )

    db.add(user)
    await db.flush()
    await db.refresh(user)

    return RegisterResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
        ),
        message="Registration successful. Please check your email for verification.",
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login user",
    description="Authenticate with email and password to receive JWT token.",
)
async def login(
    request: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LoginResponse:
    """
    Login and receive access token.

    - **email**: Registered email address
    - **password**: Account password

    Returns JWT bearer token for API authentication.
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "invalid_credentials",
                    "message": "Invalid email or password",
                }
            },
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "code": "account_disabled",
                    "message": "Account has been deactivated",
                }
            },
        )

    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.flush()

    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
        }
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
        ),
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout user",
    description="Logout current user (token invalidation).",
)
async def logout(
    current_user: Annotated[User, Depends(get_current_user)],
) -> LogoutResponse:
    """
    Logout current user.

    Note: JWT tokens are stateless. For full logout, client should
    discard the token. Token blacklisting can be implemented later
    with Redis.
    """
    # In a production system, we would add the token to a blacklist
    # For now, just return success
    return LogoutResponse(message="Logout successful")


@router.post(
    "/reset-password",
    response_model=ResetPasswordResponse,
    summary="Request password reset",
    description="Send password reset email to registered address.",
)
async def reset_password(
    request: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ResetPasswordResponse:
    """
    Request password reset.

    - **email**: Registered email address

    If the email exists, a reset link will be sent.
    Response is the same whether email exists or not (security).
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user:
        # TODO: Generate reset token and send email
        # For now, just log (placeholder for email service)
        print(f"Password reset requested for: {request.email}")

    # Always return same response (security - don't reveal if email exists)
    return ResetPasswordResponse(
        message="If the email exists, a reset link has been sent"
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get the currently authenticated user's info.",
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """
    Get current authenticated user.

    Requires valid JWT token in Authorization header.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
    )

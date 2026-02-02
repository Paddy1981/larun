"""
Authentication Schemas

Pydantic models for authentication request/response validation.

Agent: BETA
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (minimum 8 characters)",
    )
    name: Optional[str] = Field(
        None,
        max_length=255,
        description="Display name",
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(BaseModel):
    """User data in responses."""

    id: int
    email: str
    name: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    """Response for successful registration."""

    user: UserResponse
    message: str = "Registration successful"


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Response for successful login."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse


class LogoutResponse(BaseModel):
    """Response for successful logout."""

    message: str = "Logout successful"


class ResetPasswordRequest(BaseModel):
    """Request body for password reset."""

    email: EmailStr = Field(..., description="User email address")


class ResetPasswordResponse(BaseModel):
    """Response for password reset request."""

    message: str = "If the email exists, a reset link has been sent"


class TokenData(BaseModel):
    """JWT token payload data."""

    user_id: int
    email: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """Request body for token refresh."""

    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Response for token refresh."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int

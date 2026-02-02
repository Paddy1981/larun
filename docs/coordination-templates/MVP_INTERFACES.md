# MVP Interface Contracts
**Version:** 1.0
**Last Updated:** 2026-02-02

This document defines the API contracts between agents. All agents MUST adhere to these interfaces.

---

## 1. Detection Service Interface (ALPHA → BETA)

**Provider:** Agent ALPHA
**Consumer:** Agent BETA
**Status:** 🟡 Draft

### Python Types

```python
# File: shared/types/detection.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


class Disposition(str, Enum):
    PLANET_CANDIDATE = "PLANET_CANDIDATE"
    LIKELY_FALSE_POSITIVE = "LIKELY_FALSE_POSITIVE"
    INCONCLUSIVE = "INCONCLUSIVE"


class TestFlag(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class TestResult:
    """Result of a single vetting test."""
    test_name: str
    flag: TestFlag
    confidence: float  # 0.0 - 1.0
    value: float  # The measured value
    threshold: float  # The threshold for passing
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VettingResult:
    """Combined vetting results."""
    disposition: Disposition
    confidence: float  # 0.0 - 1.0
    tests_passed: int
    tests_failed: int
    tests_warning: int
    odd_even: TestResult
    v_shape: TestResult
    secondary_eclipse: TestResult
    recommendation: str


@dataclass
class LightCurveData:
    """Light curve data for visualization."""
    time: List[float]  # BJD or BTJD
    flux: List[float]  # Normalized flux
    flux_err: List[float]  # Flux errors
    quality: List[int]  # Quality flags


@dataclass
class PhaseFoldedData:
    """Phase-folded light curve."""
    phase: List[float]  # -0.5 to 0.5
    flux: List[float]
    flux_err: List[float]
    binned_phase: List[float]  # Binned for visualization
    binned_flux: List[float]
    binned_flux_err: List[float]


@dataclass
class PeriodogramData:
    """BLS periodogram results."""
    periods: List[float]  # Days
    powers: List[float]  # BLS power
    best_period: float
    best_power: float
    top_periods: List[float]  # Top 3 candidates
    top_powers: List[float]


@dataclass
class DetectionResult:
    """Complete detection analysis result."""
    # Target identification
    tic_id: str
    ra: Optional[float] = None
    dec: Optional[float] = None

    # Detection result
    detection: bool = False
    confidence: float = 0.0  # 0.0 - 1.0

    # Transit parameters (if detected)
    period_days: Optional[float] = None
    depth_ppm: Optional[float] = None
    duration_hours: Optional[float] = None
    epoch_btjd: Optional[float] = None
    snr: Optional[float] = None

    # Detailed results
    vetting: Optional[VettingResult] = None
    periodogram: Optional[PeriodogramData] = None
    phase_folded: Optional[PhaseFoldedData] = None
    raw_lightcurve: Optional[LightCurveData] = None

    # Metadata
    sectors_used: List[int] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Implementation here
        pass
```

### Service Interface

```python
# File: src/detection/service.py

from abc import ABC, abstractmethod
from shared.types.detection import DetectionResult
import numpy as np


class IDetectionService(ABC):
    """Interface for detection service."""

    @abstractmethod
    async def analyze(self, tic_id: str) -> DetectionResult:
        """
        Analyze a target by TIC ID.

        Args:
            tic_id: TESS Input Catalog ID (e.g., "TIC 12345678" or "12345678")

        Returns:
            DetectionResult with all analysis data

        Raises:
            TargetNotFoundError: If TIC ID not found in MAST
            DataUnavailableError: If no light curve data available
            AnalysisError: If analysis fails
        """
        pass

    @abstractmethod
    async def analyze_lightcurve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        tic_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Analyze provided light curve data directly.

        Args:
            time: Time array (BJD or BTJD)
            flux: Normalized flux array
            flux_err: Optional flux error array
            tic_id: Optional TIC ID for metadata

        Returns:
            DetectionResult with all analysis data
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get service health status."""
        pass
```

### Usage Example

```python
from src.detection import DetectionService

# In BETA's analysis endpoint
service = DetectionService()

# Analyze by TIC ID
result = await service.analyze("TIC 470710327")

if result.detection:
    print(f"Planet candidate found!")
    print(f"Period: {result.period_days:.4f} days")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Disposition: {result.vetting.disposition}")
else:
    print("No transit detected")
```

---

## 2. REST API Interface (BETA → GAMMA)

**Provider:** Agent BETA
**Consumer:** Agent GAMMA
**Status:** 🟡 Draft

### Base URL
```
Development: http://localhost:8000/api/v1
Production: https://api.larun.space/api/v1
```

### Authentication Endpoints

```typescript
// POST /api/v1/auth/register
interface RegisterRequest {
  email: string;
  password: string;
  name?: string;
}

interface RegisterResponse {
  user: {
    id: string;
    email: string;
    name: string;
    created_at: string;
  };
  message: string;
}

// POST /api/v1/auth/login
interface LoginRequest {
  email: string;
  password: string;
}

interface LoginResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
}

// POST /api/v1/auth/logout
// Requires: Authorization header
interface LogoutResponse {
  message: string;
}

// POST /api/v1/auth/reset-password
interface ResetPasswordRequest {
  email: string;
}

interface ResetPasswordResponse {
  message: string;
}
```

### Analysis Endpoints

```typescript
// POST /api/v1/analyze
// Requires: Authorization header
interface AnalyzeRequest {
  tic_id: string;
}

interface AnalyzeResponse {
  analysis_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  message: string;
}

// GET /api/v1/analyze/:id
// Requires: Authorization header
interface AnalysisResult {
  id: string;
  tic_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  created_at: string;
  completed_at: string | null;

  // Only present when status === "completed"
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    depth_ppm: number | null;
    duration_hours: number | null;
    epoch_btjd: number | null;
    snr: number | null;

    vetting: {
      disposition: "PLANET_CANDIDATE" | "LIKELY_FALSE_POSITIVE" | "INCONCLUSIVE";
      confidence: number;
      tests_passed: number;
      tests_failed: number;
      odd_even: TestResult;
      v_shape: TestResult;
      secondary_eclipse: TestResult;
      recommendation: string;
    };

    periodogram: {
      periods: number[];
      powers: number[];
      best_period: number;
    };

    phase_folded: {
      phase: number[];
      flux: number[];
      binned_phase: number[];
      binned_flux: number[];
    };

    raw_lightcurve: {
      time: number[];
      flux: number[];
    };

    sectors_used: number[];
    processing_time_seconds: number;
  };

  // Only present when status === "failed"
  error?: string;
}

// GET /api/v1/analyses
// Requires: Authorization header
interface AnalysesListResponse {
  analyses: AnalysisResult[];
  total: number;
  page: number;
  per_page: number;
}

// DELETE /api/v1/analyses/:id
// Requires: Authorization header
interface DeleteResponse {
  message: string;
}
```

### User Endpoints

```typescript
// GET /api/v1/user/profile
// Requires: Authorization header
interface UserProfile {
  id: string;
  email: string;
  name: string;
  created_at: string;
  subscription: {
    plan: "hobbyist_monthly" | "hobbyist_annual" | null;
    status: "active" | "canceled" | "past_due" | null;
    current_period_end: string | null;
  };
}

// GET /api/v1/user/usage
// Requires: Authorization header
interface UsageResponse {
  analyses_this_month: number;
  analyses_limit: number;
  period_start: string;
  period_end: string;
}
```

### Subscription Endpoints

```typescript
// POST /api/v1/subscription/create-checkout
// Requires: Authorization header
interface CreateCheckoutRequest {
  plan: "hobbyist_monthly" | "hobbyist_annual";
  success_url: string;
  cancel_url: string;
}

interface CreateCheckoutResponse {
  checkout_url: string;
  session_id: string;
}

// GET /api/v1/subscription/portal
// Requires: Authorization header
interface PortalResponse {
  portal_url: string;
}
```

### Error Response Format

```typescript
interface APIError {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
}

// Common error codes:
// - "unauthorized": Missing or invalid auth token
// - "forbidden": Insufficient permissions
// - "not_found": Resource not found
// - "validation_error": Invalid request data
// - "rate_limited": Too many requests
// - "usage_limit_exceeded": Monthly analysis limit reached
// - "internal_error": Server error
```

---

## 3. Auth Configuration Interface (DELTA → GAMMA)

**Provider:** Agent DELTA
**Consumer:** Agent GAMMA
**Status:** 🟡 Draft

### NextAuth Configuration

```typescript
// File: web/src/lib/auth.ts (DELTA provides)

import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // Calls BETA's /api/v1/auth/login
        // Returns user object or null
      }
    })
  ],
  callbacks: {
    async jwt({ token, user }) { /* ... */ },
    async session({ session, token }) { /* ... */ }
  },
  pages: {
    signIn: "/auth/login",
    signOut: "/auth/logout",
    error: "/auth/error",
  }
};

// Session type for GAMMA to use
declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      email: string;
      name: string;
      accessToken: string;
    };
  }
}
```

### Usage in GAMMA

```typescript
// In GAMMA's components
import { useSession, signIn, signOut } from "next-auth/react";

function AnalyzeButton() {
  const { data: session } = useSession();

  if (!session) {
    return <button onClick={() => signIn()}>Sign in to analyze</button>;
  }

  return <button onClick={handleAnalyze}>Analyze</button>;
}
```

---

## 4. Stripe Configuration Interface (DELTA → BETA/GAMMA)

**Provider:** Agent DELTA
**Consumer:** Agents BETA, GAMMA
**Status:** 🟡 Draft

### Stripe Product IDs

```typescript
// File: shared/constants/stripe.ts (DELTA provides)

export const STRIPE_PRODUCTS = {
  HOBBYIST_MONTHLY: {
    priceId: process.env.STRIPE_PRICE_HOBBYIST_MONTHLY!,
    name: "Hobbyist Monthly",
    price: 9,
    currency: "usd",
    interval: "month",
    analysisLimit: 25,
  },
  HOBBYIST_ANNUAL: {
    priceId: process.env.STRIPE_PRICE_HOBBYIST_ANNUAL!,
    name: "Hobbyist Annual",
    price: 89,
    currency: "usd",
    interval: "year",
    analysisLimit: 25,  // per month
  },
} as const;
```

### Webhook Events (DELTA handles, BETA receives)

```typescript
// Events DELTA forwards to BETA's internal API

interface SubscriptionActivated {
  event: "subscription.activated";
  user_id: string;
  stripe_customer_id: string;
  stripe_subscription_id: string;
  plan: "hobbyist_monthly" | "hobbyist_annual";
  current_period_end: string;
}

interface SubscriptionCanceled {
  event: "subscription.canceled";
  user_id: string;
  stripe_subscription_id: string;
}

interface SubscriptionUpdated {
  event: "subscription.updated";
  user_id: string;
  stripe_subscription_id: string;
  plan: "hobbyist_monthly" | "hobbyist_annual";
  status: "active" | "past_due" | "canceled";
  current_period_end: string;
}
```

---

## Interface Status Summary

| Interface | Provider | Consumer | Status | Version |
|-----------|----------|----------|--------|---------|
| DetectionService | ALPHA | BETA | 🟡 Draft | 1.0 |
| REST API | BETA | GAMMA | 🟡 Draft | 1.0 |
| Auth Config | DELTA | GAMMA | 🟡 Draft | 1.0 |
| Stripe Config | DELTA | BETA, GAMMA | 🟡 Draft | 1.0 |

**Status Legend:**
- 🟡 Draft - Interface defined, not yet implemented
- 🟢 Approved - Interface reviewed and approved
- 🔵 Implemented - Interface implemented by provider
- ✅ Verified - Interface tested by consumer

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-02-02 | System | Initial interface definitions |

---

*All agents must notify others before changing interfaces.*
*Use HANDOFF_NOTES.md to communicate interface changes.*

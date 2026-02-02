# LARUN.SPACE MVP - Parallel Implementation Plan
## 4-Agent Concurrent Development Strategy

**Document:** PARALLEL-MVP-2026-001
**Version:** 1.0
**Date:** February 2, 2026
**Target:** 4-week accelerated MVP delivery

---

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    4-AGENT PARALLEL DEVELOPMENT                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ║
║   │   ALPHA     │  │    BETA     │  │   GAMMA     │  │   DELTA     │        ║
║   │  Detection  │  │   Backend   │  │  Frontend   │  │  Platform   │        ║
║   │   Engine    │  │     API     │  │     UI      │  │   DevOps    │        ║
║   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        ║
║          │                │                │                │                ║
║          └────────────────┴────────────────┴────────────────┘                ║
║                                   │                                          ║
║                          ┌───────┴───────┐                                   ║
║                          │ COORDINATION  │                                   ║
║                          │    LAYER      │                                   ║
║                          └───────────────┘                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 1. AGENT ASSIGNMENTS

### Agent ALPHA - Detection Engine
```
Focus: Core astronomical analysis pipeline
Owner: Claude Agent 1
Branch: claude/mvp-alpha-detection

Responsibilities:
├── Transit detection model optimization
├── BLS periodogram enhancements
├── Phase folding accuracy
├── Vetting suite refinement
└── Detection API service layer

Files Owned (exclusive write access):
├── src/skills/periodogram.py
├── src/skills/vetting.py
├── src/skills/transit_fitting.py (new)
├── src/detection/
│   ├── __init__.py
│   ├── detector.py
│   ├── bls_engine.py
│   └── phase_folder.py
├── src/services/detection_service.py (new)
└── tests/test_detection/

Interface Contract:
- Exposes: DetectionService class with analyze() method
- Input: TIC ID or light curve data
- Output: DetectionResult dataclass
```

### Agent BETA - Backend API
```
Focus: REST API, database, job processing
Owner: Claude Agent 2
Branch: claude/mvp-beta-backend

Responsibilities:
├── FastAPI endpoint implementation
├── Database schema and migrations
├── Background job queue
├── User management APIs
└── Analysis result storage

Files Owned (exclusive write access):
├── src/api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── auth.py
│   │   ├── analysis.py
│   │   ├── user.py
│   │   └── subscription.py
│   ├── models/
│   │   ├── database.py
│   │   ├── user.py
│   │   ├── analysis.py
│   │   └── subscription.py
│   └── services/
│       ├── job_queue.py
│       └── email_service.py
├── alembic/ (migrations)
└── tests/test_api/

Interface Contract:
- Exposes: REST API at /api/v1/*
- Consumes: DetectionService from ALPHA
- Database: PostgreSQL with defined schema
```

### Agent GAMMA - Frontend UI
```
Focus: Web interface, visualizations, UX
Owner: Claude Agent 3
Branch: claude/mvp-gamma-frontend

Responsibilities:
├── Next.js application setup
├── Landing page
├── Analysis interface
├── Results visualization (Plotly)
├── User dashboard
└── Responsive design

Files Owned (exclusive write access):
├── web/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx (landing)
│   │   │   ├── analyze/
│   │   │   ├── dashboard/
│   │   │   ├── auth/
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── LightCurvePlot.tsx
│   │   │   ├── PeriodogramPlot.tsx
│   │   │   ├── VettingResults.tsx
│   │   │   └── AnalysisForm.tsx
│   │   ├── lib/
│   │   │   └── api-client.ts
│   │   └── styles/
│   └── public/
└── tests/test_frontend/

Interface Contract:
- Consumes: REST API from BETA
- API Client: Typed fetch wrapper
- State: React Query for server state
```

### Agent DELTA - Platform & DevOps
```
Focus: Authentication, payments, infrastructure
Owner: Claude Agent 4
Branch: claude/mvp-delta-platform

Responsibilities:
├── Stripe integration
├── NextAuth.js setup
├── Docker configuration
├── CI/CD pipelines
├── Environment configuration
└── Deployment scripts

Files Owned (exclusive write access):
├── web/src/app/api/auth/ (NextAuth)
├── web/src/app/api/stripe/
├── web/src/lib/
│   ├── auth.ts
│   ├── stripe.ts
│   └── config.ts
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.web
│   └── docker-compose.yml
├── .github/workflows/
│   ├── deploy.yml
│   └── test.yml
├── infrastructure/
│   ├── terraform/ (optional)
│   └── scripts/
└── .env.example

Interface Contract:
- Provides: Auth middleware for API
- Provides: Stripe webhook handlers
- Provides: Deployment configuration
```

---

## 2. COORDINATION SYSTEM

### 2.1 Directory Structure
```
.coordination/
├── STATUS.md              # Real-time status of all agents
├── INTERFACES.md          # API contracts between components
├── FILE_LOCKS.md          # Current file ownership
├── HANDOFF_QUEUE.md       # Tasks waiting for dependencies
├── INTEGRATION_LOG.md     # Integration test results
└── agents/
    ├── ALPHA_WORKLOG.md   # Agent Alpha's progress
    ├── BETA_WORKLOG.md    # Agent Beta's progress
    ├── GAMMA_WORKLOG.md   # Agent Gamma's progress
    └── DELTA_WORKLOG.md   # Agent Delta's progress
```

### 2.2 Status Board Format
```markdown
# AGENT STATUS BOARD
Last Updated: [timestamp]

## Current Sprint: Week 1 - Foundation

| Agent | Status | Current Task | Blocked By | ETA |
|-------|--------|--------------|------------|-----|
| ALPHA | 🟢 Active | BLS optimization | - | 2h |
| BETA  | 🟡 Waiting | DB schema | DELTA (env) | 4h |
| GAMMA | 🟢 Active | Landing page | - | 3h |
| DELTA | 🟢 Active | Docker setup | - | 1h |

## Blockers
- [ ] BETA waiting for DATABASE_URL from DELTA

## Today's Integration Points
- [ ] 14:00 - ALPHA provides DetectionService interface
- [ ] 16:00 - DELTA provides auth middleware
- [ ] 18:00 - Integration test: BETA + ALPHA
```

### 2.3 Interface Contract Template
```markdown
# Interface: DetectionService

Provider: ALPHA
Consumers: BETA

## Python Interface
```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class DetectionResult:
    tic_id: str
    detection: bool
    confidence: float  # 0.0 - 1.0
    period_days: Optional[float]
    depth_ppm: Optional[float]
    duration_hours: Optional[float]
    epoch_btjd: Optional[float]
    snr: Optional[float]
    vetting: VettingResult
    phase_folded: PhaseFoldedData
    raw_lightcurve: LightCurveData

@dataclass
class VettingResult:
    disposition: str  # "PLANET_CANDIDATE" | "LIKELY_FALSE_POSITIVE" | "INCONCLUSIVE"
    confidence: float
    odd_even: TestResult
    v_shape: TestResult
    secondary: TestResult

class DetectionService:
    async def analyze(self, tic_id: str) -> DetectionResult:
        """Main entry point for transit analysis."""
        pass

    async def analyze_lightcurve(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """Analyze provided light curve data."""
        pass
```

Status: 🟡 Draft | 🟢 Approved | 🔵 Implemented
Current: 🟡 Draft
```

---

## 3. WEEKLY SPRINT PLAN

### Week 1: Foundation (Days 1-7)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ WEEK 1: PARALLEL FOUNDATION WORK                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ALPHA (Detection)         BETA (Backend)                                  │
│  ┌──────────────────┐      ┌──────────────────┐                           │
│  │ Day 1-2:         │      │ Day 1-2:         │                           │
│  │ - Refactor BLS   │      │ - DB schema      │                           │
│  │ - DetectionSvc   │      │ - Alembic setup  │                           │
│  │   interface      │      │ - Base models    │                           │
│  ├──────────────────┤      ├──────────────────┤                           │
│  │ Day 3-4:         │      │ Day 3-4:         │                           │
│  │ - Phase folding  │      │ - Auth endpoints │                           │
│  │ - Vetting tests  │      │ - User CRUD      │                           │
│  ├──────────────────┤      ├──────────────────┤                           │
│  │ Day 5-7:         │      │ Day 5-7:         │                           │
│  │ - Unit tests     │      │ - Analysis API   │                           │
│  │ - Integration    │  ──► │ - Job queue      │                           │
│  └──────────────────┘      └──────────────────┘                           │
│                                                                            │
│  GAMMA (Frontend)          DELTA (Platform)                                │
│  ┌──────────────────┐      ┌──────────────────┐                           │
│  │ Day 1-2:         │      │ Day 1-2:         │                           │
│  │ - Next.js setup  │      │ - Docker configs │                           │
│  │ - Tailwind       │      │ - .env setup     │                           │
│  │ - Component lib  │      │ - CI pipeline    │                           │
│  ├──────────────────┤      ├──────────────────┤                           │
│  │ Day 3-4:         │      │ Day 3-4:         │                           │
│  │ - Landing page   │      │ - NextAuth.js    │                           │
│  │ - Auth UI        │  ◄── │ - Auth config    │                           │
│  ├──────────────────┤      ├──────────────────┤                           │
│  │ Day 5-7:         │      │ Day 5-7:         │                           │
│  │ - Analysis form  │      │ - Stripe setup   │                           │
│  │ - Mock results   │      │ - Webhooks       │                           │
│  └──────────────────┘      └──────────────────┘                           │
│                                                                            │
│  INTEGRATION CHECKPOINT: Day 7                                             │
│  - ALPHA DetectionService callable from BETA                               │
│  - DELTA auth working with GAMMA                                           │
│  - All agents can run locally with docker-compose                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### Week 1 Deliverables by Agent

**ALPHA - Detection Engine**
```
□ Day 1: Create src/detection/ module structure
□ Day 1: Define DetectionResult, VettingResult dataclasses
□ Day 2: Refactor BLS from skills to detection module
□ Day 2: Create DetectionService class with interface
□ Day 3: Implement phase_fold() with sub-second accuracy
□ Day 3: Refactor vetting tests to new structure
□ Day 4: Add comprehensive logging
□ Day 4: Create detection CLI for testing
□ Day 5: Write unit tests (>80% coverage)
□ Day 6: Integration tests with sample TIC IDs
□ Day 7: Documentation and interface finalization
```

**BETA - Backend API**
```
□ Day 1: Create src/api/ module structure
□ Day 1: Set up SQLAlchemy + Alembic
□ Day 2: Define database models (User, Analysis, Subscription)
□ Day 2: Create initial migration
□ Day 3: Implement /api/auth/* endpoints
□ Day 3: Implement /api/user/* endpoints
□ Day 4: Implement /api/analyze endpoint (stub)
□ Day 4: Set up Redis + job queue
□ Day 5: Connect to ALPHA's DetectionService
□ Day 6: Implement job status polling
□ Day 7: API documentation (OpenAPI/Swagger)
```

**GAMMA - Frontend**
```
□ Day 1: npx create-next-app with TypeScript + Tailwind
□ Day 1: Set up project structure
□ Day 2: Create component library (Button, Card, Input, etc.)
□ Day 2: Set up API client with types
□ Day 3: Build landing page (hero, features, pricing)
□ Day 3: Build auth pages (login, register, forgot-password)
□ Day 4: Connect auth UI to DELTA's NextAuth
□ Day 4: Build analysis form component
□ Day 5: Build results display (mock data)
□ Day 6: Build light curve visualization (Plotly)
□ Day 7: Responsive testing + polish
```

**DELTA - Platform**
```
□ Day 1: Create docker/ directory structure
□ Day 1: Dockerfile for API (Python)
□ Day 1: docker-compose.yml with all services
□ Day 2: Set up .env.example with all variables
□ Day 2: Create GitHub Actions test workflow
□ Day 3: Implement NextAuth.js configuration
□ Day 3: Set up auth providers (email, optional OAuth)
□ Day 4: Create Stripe products and prices
□ Day 4: Implement checkout session creation
□ Day 5: Implement Stripe webhooks
□ Day 5: Usage limit enforcement logic
□ Day 6: Deployment scripts (Vercel + Railway)
□ Day 7: Production environment setup
```

---

### Week 2: Integration (Days 8-14)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ WEEK 2: INTEGRATION & FEATURE COMPLETION                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ALPHA ──────────► BETA ──────────► GAMMA                                  │
│  Detection         API              UI                                     │
│  Service           Endpoints        Components                             │
│                                                                            │
│  Day 8-10: Full Pipeline Integration                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ User enters TIC ID → API queues job → Detection runs → Results show │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Day 11-12: Payment Integration                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ User subscribes → Stripe checkout → Webhook → Account activated     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  Day 13-14: End-to-End Testing                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Complete user journey from signup to analysis to results            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### Week 2 Deliverables by Agent

**ALPHA - Detection Engine**
```
□ Day 8: Optimize BLS for <30s execution
□ Day 9: Add caching for repeated TIC queries
□ Day 10: Performance benchmarking (20 targets)
□ Day 11: Error handling improvements
□ Day 12: Edge case handling (no data, partial data)
□ Day 13: Load testing support
□ Day 14: Final accuracy validation
```

**BETA - Backend API**
```
□ Day 8: Full integration with DetectionService
□ Day 9: Analysis history endpoints
□ Day 10: Usage tracking implementation
□ Day 11: Subscription status in responses
□ Day 12: Rate limiting implementation
□ Day 13: Error response standardization
□ Day 14: API load testing
```

**GAMMA - Frontend**
```
□ Day 8: Connect analysis form to real API
□ Day 9: Real-time job status polling
□ Day 10: Results page with real data
□ Day 11: Dashboard with analysis history
□ Day 12: Subscription management UI
□ Day 13: Error states and edge cases
□ Day 14: Mobile responsiveness final pass
```

**DELTA - Platform**
```
□ Day 8: Full auth flow testing
□ Day 9: Stripe subscription flow testing
□ Day 10: Usage limit enforcement testing
□ Day 11: Production deployment (staging)
□ Day 12: SSL and domain configuration
□ Day 13: Monitoring setup (Sentry, analytics)
□ Day 14: Security audit checklist
```

---

### Week 3: Polish & Testing (Days 15-21)

```
□ All agents: Bug fixes from integration
□ All agents: Performance optimization
□ ALPHA: Accuracy improvements if needed
□ BETA: Database optimization
□ GAMMA: UI/UX polish
□ DELTA: Security hardening
□ Integration: Full E2E test suite
□ Integration: Load testing (50 concurrent users)
```

---

### Week 4: Launch Prep (Days 22-28)

```
□ Beta testing with 10 users
□ Bug fixes from beta feedback
□ Documentation completion
□ Marketing page content
□ Support email setup
□ Soft launch
□ Monitor and hotfix
```

---

## 4. COMMUNICATION PROTOCOL

### 4.1 Handoff Messages

When an agent completes work that another agent depends on:

```markdown
## HANDOFF: ALPHA → BETA
Date: 2026-02-03 14:00 UTC
From: Agent ALPHA
To: Agent BETA

### Completed
- DetectionService class implemented
- All tests passing (47/47)
- Interface matches INTERFACES.md spec

### Files Changed
- src/detection/service.py (new)
- src/detection/models.py (new)
- tests/test_detection/test_service.py (new)

### How to Use
```python
from src.detection import DetectionService

service = DetectionService()
result = await service.analyze("TIC 12345678")
print(result.detection)  # True/False
print(result.confidence)  # 0.87
```

### Known Issues
- None

### Next Steps for BETA
- Import DetectionService in analysis endpoint
- Call service.analyze() in job worker

### Branch
claude/mvp-alpha-detection @ commit abc123
```

### 4.2 Blocking Notifications

When an agent is blocked:

```markdown
## BLOCKER: BETA blocked by DELTA
Date: 2026-02-03 10:00 UTC
From: Agent BETA
Blocking Agent: DELTA

### What I Need
DATABASE_URL environment variable and database credentials

### Why I'm Blocked
Cannot run migrations or test database models

### Impact
- 4 tasks blocked
- Estimated delay: 2 hours after resolution

### Workaround Attempted
- Using SQLite locally (partial success)
- Need PostgreSQL for full compatibility

### Priority
HIGH - Blocking critical path
```

### 4.3 Daily Sync Format

Each agent updates STATUS.md at start of day:

```markdown
## ALPHA Daily Update - 2026-02-03

### Yesterday
- ✅ Refactored BLS to detection module
- ✅ Created DetectionService interface
- ⚠️ Phase folding 90% complete (edge case found)

### Today
- [ ] Fix phase folding edge case
- [ ] Complete vetting test refactor
- [ ] Write unit tests

### Blockers
- None

### Need from Others
- BETA: Confirmation on DetectionResult schema
- DELTA: None

### Integration Ready
- DetectionService.analyze() ready for BETA integration
```

---

## 5. GIT WORKFLOW

### 5.1 Branch Strategy

```
main
├── develop
│   ├── claude/mvp-alpha-detection  (Agent ALPHA)
│   ├── claude/mvp-beta-backend     (Agent BETA)
│   ├── claude/mvp-gamma-frontend   (Agent GAMMA)
│   └── claude/mvp-delta-platform   (Agent DELTA)
│
└── Integration branches (created as needed)
    ├── integrate/alpha-beta
    ├── integrate/gamma-delta
    └── integrate/full-stack
```

### 5.2 Merge Rules

1. **Daily**: Agents push to their own branches
2. **Integration Points**: Create integration branches
3. **End of Week**: Merge integration branches to develop
4. **Launch**: Merge develop to main

### 5.3 Conflict Resolution

If two agents need the same file:
1. First agent to claim in FILE_LOCKS.md owns it
2. Second agent creates interface request
3. Owning agent exposes interface
4. Never edit files you don't own

---

## 6. SHARED RESOURCES

### 6.1 Shared Types (all agents can read)

```
shared/
├── types/
│   ├── detection.py    # DetectionResult, VettingResult
│   ├── user.py         # UserProfile, Subscription
│   └── api.py          # APIResponse, APIError
└── constants/
    ├── config.py       # Shared configuration
    └── enums.py        # Status enums, etc.
```

### 6.2 Shared Dependencies

```
# requirements.txt (DELTA maintains)
# All agents use same versions

numpy==1.24.0
pandas==2.0.0
astropy==5.3.0
lightkurve==2.4.0
fastapi==0.100.0
sqlalchemy==2.0.0
pydantic==2.0.0
```

---

## 7. AGENT STARTUP INSTRUCTIONS

### For Agent ALPHA (Detection)
```markdown
You are Agent ALPHA, responsible for the Detection Engine.

Your branch: claude/mvp-alpha-detection
Your files: src/detection/*, src/skills/*, tests/test_detection/*

DO NOT modify files owned by other agents.

Your first task:
1. Read .coordination/STATUS.md
2. Read .coordination/INTERFACES.md
3. Create your branch from develop
4. Start with Week 1, Day 1 tasks
5. Update .coordination/agents/ALPHA_WORKLOG.md daily

Interface you must provide:
- DetectionService class with analyze(tic_id) method
- Must return DetectionResult dataclass
- See INTERFACES.md for exact specification

When complete, create HANDOFF message for BETA.
```

### For Agent BETA (Backend)
```markdown
You are Agent BETA, responsible for the Backend API.

Your branch: claude/mvp-beta-backend
Your files: src/api/*, alembic/*, tests/test_api/*

DO NOT modify files owned by other agents.

Your first task:
1. Read .coordination/STATUS.md
2. Read .coordination/INTERFACES.md
3. Create your branch from develop
4. Start with Week 1, Day 1 tasks
5. Update .coordination/agents/BETA_WORKLOG.md daily

You will consume:
- DetectionService from ALPHA (wait for HANDOFF)
- Auth middleware from DELTA (wait for HANDOFF)

You will provide:
- REST API endpoints for GAMMA
- See INTERFACES.md for API specification
```

### For Agent GAMMA (Frontend)
```markdown
You are Agent GAMMA, responsible for the Frontend UI.

Your branch: claude/mvp-gamma-frontend
Your files: web/*, tests/test_frontend/*

DO NOT modify files owned by other agents.

Your first task:
1. Read .coordination/STATUS.md
2. Read .coordination/INTERFACES.md
3. Create your branch from develop
4. Start with Week 1, Day 1 tasks
5. Update .coordination/agents/GAMMA_WORKLOG.md daily

You will consume:
- REST API from BETA
- Auth config from DELTA

Start with mock data, replace with real API when BETA ready.
```

### For Agent DELTA (Platform)
```markdown
You are Agent DELTA, responsible for Platform & DevOps.

Your branch: claude/mvp-delta-platform
Your files: docker/*, infrastructure/*, web/src/lib/auth.ts,
           web/src/lib/stripe.ts, .github/workflows/*

DO NOT modify files owned by other agents.

Your first task:
1. Read .coordination/STATUS.md
2. Read .coordination/INTERFACES.md
3. Create your branch from develop
4. Start with Week 1, Day 1 tasks
5. Update .coordination/agents/DELTA_WORKLOG.md daily

You provide to all agents:
- Docker configuration
- Environment variables
- Auth middleware
- Stripe integration
- CI/CD pipelines

Unblock other agents ASAP - they depend on your infrastructure.
```

---

## 8. QUICK REFERENCE

### File Ownership Matrix

| Directory/File | ALPHA | BETA | GAMMA | DELTA |
|----------------|-------|------|-------|-------|
| src/detection/ | ✅ Write | Read | - | - |
| src/skills/ | ✅ Write | Read | - | - |
| src/api/ | Read | ✅ Write | - | - |
| alembic/ | - | ✅ Write | - | - |
| web/src/app/ | - | - | ✅ Write | - |
| web/src/components/ | - | - | ✅ Write | - |
| web/src/lib/auth.ts | - | - | Read | ✅ Write |
| web/src/lib/stripe.ts | - | - | Read | ✅ Write |
| docker/ | Read | Read | Read | ✅ Write |
| .github/workflows/ | - | - | - | ✅ Write |
| shared/ | Read | Read | Read | ✅ Write |
| .coordination/ | ✅ Write | ✅ Write | ✅ Write | ✅ Write |

### Integration Checkpoints

| Day | Checkpoint | Agents | Verification |
|-----|------------|--------|--------------|
| 7 | Detection callable | ALPHA + BETA | Unit test passes |
| 7 | Auth working | DELTA + GAMMA | Login flow works |
| 10 | Full analysis | ALL | TIC → Results |
| 14 | Payment flow | DELTA + GAMMA + BETA | Subscribe works |
| 21 | E2E complete | ALL | Full user journey |

---

## 9. EMERGENCY PROCEDURES

### If Agent Goes Offline
1. Other agents continue on non-blocked tasks
2. Mark blocked tasks in HANDOFF_QUEUE.md
3. New agent can pick up from worklog

### If Integration Fails
1. Identify which interface contract broken
2. Both agents review INTERFACES.md
3. Resolve in integration branch
4. Update contract if needed

### If Behind Schedule
1. Identify critical path items
2. Defer non-essential features
3. Focus all agents on blockers
4. Consider scope reduction

---

*Document: PARALLEL-MVP-2026-001*
*Version: 1.0*
*For: Multi-Agent Claude Development*

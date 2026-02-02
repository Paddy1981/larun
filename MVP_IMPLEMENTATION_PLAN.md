# LARUN.SPACE - Hobbyist MVP Implementation Plan
## Priority 1: Minimum Viable Product for Market Launch

**Document:** MVP-LARUN-2026-001
**Target Launch:** 6-8 weeks from start
**Tier:** Hobbyist ($9/month, $89/year)
**Goal:** 50-100 paying users in first 90 days

---

## EXECUTIVE SUMMARY

This document defines the ABSOLUTE MINIMUM requirements to launch LARUN.SPACE Hobbyist tier. Everything not listed here is OUT OF SCOPE for MVP and deferred to Phase 2+.

### What Hobbyist MVP IS:
- Basic transit detection with decent accuracy (≥85% AUC acceptable for MVP)
- Simple vetting diagnostics (odd-even, V-shape, secondary eclipse)
- Web interface where users input TIC ID → get results
- User accounts with usage limits (25 targets/month)
- Payment processing ($9/month subscription)

### What Hobbyist MVP is NOT:
- FPP/NFPP calculation (Professional tier)
- MCMC posterior distributions (Professional tier)
- Gaia DR3 integration (Education+ tier)
- Peer review system (Professional tier)
- ExoFOP submission (Professional tier)
- DOI assignment (Professional tier)
- Multi-user/organization accounts (Education tier)
- API access (Professional tier)

---

## PHASE 1: MVP TECHNICAL REQUIREMENTS

### 1. DETECTION ENGINE (CRITICAL)

#### 1.1 Transit Detection Model
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
- Accept TIC ID or coordinates as input
- Retrieve TESS light curve from MAST
- Run transit detection algorithm
- Return: detected (yes/no), confidence score, period estimate

Minimum Accuracy Target: 85% AUC (upgrade to 95% in Phase 2)

Implementation Options (choose one):
□ Option A: TinyML model (if already built)
□ Option B: BLS + threshold-based detection
□ Option C: Simplified CNN on phase-folded light curves

Output Format:
{
  "tic_id": "string",
  "detection": true/false,
  "confidence": 0.0-1.0,
  "period_days": float,
  "depth_ppm": float,
  "duration_hours": float,
  "epoch_btjd": float,
  "snr": float
}
```

#### 1.2 BLS Periodogram
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
- Period search range: 0.5 - 30 days (MVP scope)
- Return top 3 period candidates with power values
- Processing time: < 60 seconds per target

Libraries to Use:
- astropy.timeseries.BoxLeastSquares
- OR transitleastsquares (TLS) if already integrated

Output Format:
{
  "periods": [float, float, float],
  "powers": [float, float, float],
  "best_period": float,
  "best_power": float
}
```

#### 1.3 Phase Folding
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
- Fold light curve at detected period
- Bin data for visualization (50-100 bins)
- Center transit at phase 0.0

Output Format:
{
  "phase": [float array],
  "flux": [float array],
  "flux_err": [float array],
  "binned_phase": [float array],
  "binned_flux": [float array]
}
```

---

### 2. VETTING SUITE (CRITICAL)

#### 2.1 Odd-Even Transit Test
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Purpose: Detect eclipsing binaries with different primary/secondary depths

Implementation:
1. Separate transits into odd (1,3,5...) and even (2,4,6...) groups
2. Calculate mean depth for each group
3. Compute significance of difference: |depth_odd - depth_even| / sqrt(err_odd² + err_even²)
4. Flag if significance > 3σ

Output Format:
{
  "odd_depth_ppm": float,
  "even_depth_ppm": float,
  "depth_difference_sigma": float,
  "flag": "PASS" / "WARNING" / "FAIL",
  "interpretation": "string"
}

Pass Criteria: depth_difference_sigma < 3.0
```

#### 2.2 V-Shape Detection
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Purpose: Distinguish planetary transits (flat bottom) from grazing binaries (V-shape)

Implementation:
1. Fit trapezoid model to phase-folded transit
2. Calculate V-shape parameter: V = 1 - (flat_duration / total_duration)
3. V near 0 = flat bottom (planet), V near 1 = V-shape (binary)

Output Format:
{
  "v_shape_parameter": float,  // 0.0 to 1.0
  "flag": "PASS" / "WARNING" / "FAIL",
  "interpretation": "string"
}

Pass Criteria: v_shape_parameter < 0.5
Warning: 0.5 - 0.7
Fail: > 0.7
```

#### 2.3 Secondary Eclipse Check
```
Priority: P1 (Important)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Purpose: Detect eclipsing binaries by finding eclipse at phase 0.5

Implementation:
1. Extract flux around phase 0.5 (±0.05)
2. Compare to out-of-eclipse baseline
3. Calculate depth and significance of any dip

Output Format:
{
  "secondary_depth_ppm": float,
  "secondary_significance": float,
  "secondary_detected": true/false,
  "flag": "PASS" / "WARNING" / "FAIL",
  "interpretation": "string"
}

Pass Criteria: No significant secondary (significance < 3σ) OR secondary consistent with planetary thermal emission (depth < 500 ppm for hot Jupiters)
```

#### 2.4 Vetting Summary Generator
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Combine all vetting results into user-friendly summary:

Output Format:
{
  "overall_disposition": "PLANET_CANDIDATE" / "LIKELY_FALSE_POSITIVE" / "INCONCLUSIVE",
  "confidence_score": 0.0-1.0,
  "tests_passed": int,
  "tests_failed": int,
  "tests_warning": int,
  "recommendation": "string",
  "details": {
    "odd_even": {...},
    "v_shape": {...},
    "secondary": {...}
  }
}
```

---

### 3. DATA ACCESS LAYER (CRITICAL)

#### 3.1 MAST/TESS Integration
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
- Query by TIC ID
- Retrieve 2-minute cadence light curves (preferred)
- Fallback to FFI if 2-min not available
- Handle multi-sector data (stitch up to 3 sectors for MVP)
- Cache downloaded data to reduce API calls

Libraries:
- lightkurve (recommended)
- astroquery.mast

Key Functions Needed:
□ get_light_curve(tic_id) -> LightCurve object
□ get_available_sectors(tic_id) -> list of sectors
□ stitch_sectors(light_curves) -> combined LightCurve

Error Handling:
- No data available: Return clear error message
- Partial data: Process what's available with warning
- API timeout: Retry with exponential backoff (3 attempts)
```

#### 3.2 TIC Lookup (Basic)
```
Priority: P1 (Important)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
- Validate TIC ID exists
- Retrieve basic stellar parameters (Teff, radius, magnitude)
- Get known planet status (is this already a confirmed planet?)

Output Format:
{
  "tic_id": "string",
  "ra": float,
  "dec": float,
  "tmag": float,
  "teff": float,
  "radius_solar": float,
  "known_planet_host": true/false,
  "toi_designation": "string" or null
}

Note: Full Gaia integration deferred to Phase 2
```

---

### 4. WEB INTERFACE (CRITICAL)

#### 4.1 Landing Page
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Elements Required:
□ Hero section with value proposition
□ "How it works" 3-step explanation
□ Pricing display ($9/month, $89/year)
□ Sign up / Login buttons
□ Sample result screenshot
□ Footer with terms, privacy, contact

Tech Stack Recommendation:
- Next.js or React
- Tailwind CSS for styling
- Vercel for hosting
```

#### 4.2 User Authentication
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
□ Email/password registration
□ Email verification
□ Login / Logout
□ Password reset
□ Session management

Implementation Options:
□ Option A: NextAuth.js (recommended for Next.js)
□ Option B: Firebase Auth
□ Option C: Supabase Auth
□ Option D: Auth0

Security Requirements:
- Passwords hashed with bcrypt (min 10 rounds)
- HTTPS only
- Secure session cookies (httpOnly, secure, sameSite)
```

#### 4.3 Analysis Interface
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

User Flow:
1. User enters TIC ID in search box
2. System validates TIC ID exists
3. User clicks "Analyze"
4. Loading state while processing (show progress)
5. Results displayed with visualizations

Required UI Components:
□ TIC ID input field with validation
□ "Analyze" button
□ Progress indicator (processing can take 30-60 seconds)
□ Results panel with:
  □ Detection verdict (Candidate / Not Detected / False Positive)
  □ Confidence score visualization
  □ Period and depth display
  □ Phase-folded light curve plot (interactive)
  □ Raw light curve plot
  □ Vetting results breakdown
  □ "Download Report" button (PDF)
□ Error states for failed analyses

Visualization Library:
- Plotly.js (recommended for interactivity)
- OR Chart.js
- OR Recharts (if React)
```

#### 4.4 User Dashboard
```
Priority: P1 (Important)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Required Elements:
□ Analysis history (last 20 analyses)
□ Usage counter (X / 25 targets this month)
□ Subscription status
□ Saved favorites (bookmark interesting targets)
□ Account settings

Analysis History Table Columns:
- Date
- TIC ID
- Result (Candidate/FP/None)
- Confidence
- Actions (View, Delete)
```

#### 4.5 Subscription Management
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Requirements:
□ Display current plan
□ Upgrade/downgrade options
□ Payment method management
□ Billing history
□ Cancel subscription

Stripe Integration:
□ Stripe Checkout for initial subscription
□ Stripe Customer Portal for management
□ Webhook handling for subscription events
```

---

### 5. PAYMENT SYSTEM (CRITICAL)

#### 5.1 Stripe Integration
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Products to Create in Stripe:
□ Hobbyist Monthly: $9/month (price_xxxxx)
□ Hobbyist Annual: $89/year (price_xxxxx)

Webhook Events to Handle:
□ checkout.session.completed -> Activate subscription
□ customer.subscription.updated -> Update plan
□ customer.subscription.deleted -> Deactivate account
□ invoice.payment_failed -> Send warning email

Implementation Checklist:
□ Create Stripe account (if not exists)
□ Set up products and prices
□ Implement Checkout Session creation
□ Set up webhook endpoint
□ Test with Stripe test mode
□ Go live with production keys
```

#### 5.2 Usage Limits Enforcement
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Hobbyist Limit: 25 targets per calendar month

Implementation:
1. Track analyses in database with timestamp
2. Before each analysis, check count this month
3. If limit reached, show upgrade prompt
4. Reset counter on 1st of each month

Database Schema:
analyses {
  id: uuid
  user_id: uuid
  tic_id: string
  created_at: timestamp
  result: jsonb
}

Query for limit check:
SELECT COUNT(*) FROM analyses
WHERE user_id = ?
AND created_at >= date_trunc('month', now())
```

---

### 6. BACKEND API (CRITICAL)

#### 6.1 API Endpoints
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Required Endpoints:

POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/reset-password

GET  /api/user/profile
PUT  /api/user/profile
GET  /api/user/usage

POST /api/analyze
  Input: { tic_id: string }
  Output: { analysis_id: string, status: "processing" }

GET  /api/analyze/:id
  Output: Full analysis results

GET  /api/analyses
  Output: List of user's analyses (paginated)

DELETE /api/analyses/:id

POST /api/subscription/create-checkout
POST /api/subscription/webhook (Stripe)
GET  /api/subscription/portal
```

#### 6.2 Background Job Processing
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Analysis jobs should run async (30-60 second processing time)

Options:
□ Option A: Serverless function with longer timeout (Vercel Pro, AWS Lambda)
□ Option B: Dedicated worker process (Railway, Render)
□ Option C: Queue system (Redis + Bull, AWS SQS)

Recommended for MVP:
- Vercel Pro with 60-second function timeout
- OR Railway with background worker

Job Flow:
1. POST /api/analyze creates job, returns job_id immediately
2. Frontend polls GET /api/analyze/:id every 5 seconds
3. Worker processes job, updates database
4. Poll returns completed results
```

---

### 7. DATABASE (CRITICAL)

#### 7.1 Schema Design
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Recommended: PostgreSQL (Supabase, Railway, or Neon)

Tables:

users {
  id: uuid PRIMARY KEY
  email: string UNIQUE
  password_hash: string
  created_at: timestamp
  email_verified: boolean
  stripe_customer_id: string
}

subscriptions {
  id: uuid PRIMARY KEY
  user_id: uuid REFERENCES users
  stripe_subscription_id: string
  status: enum ('active', 'canceled', 'past_due')
  plan: enum ('hobbyist_monthly', 'hobbyist_annual')
  current_period_end: timestamp
}

analyses {
  id: uuid PRIMARY KEY
  user_id: uuid REFERENCES users
  tic_id: string
  status: enum ('pending', 'processing', 'completed', 'failed')
  created_at: timestamp
  completed_at: timestamp
  result: jsonb
  error: string
}

favorites {
  id: uuid PRIMARY KEY
  user_id: uuid REFERENCES users
  tic_id: string
  analysis_id: uuid REFERENCES analyses
  created_at: timestamp
}

Indexes:
- analyses(user_id, created_at) for usage counting
- analyses(status) for job processing
- users(email) for login
- users(stripe_customer_id) for webhook handling
```

---

### 8. INFRASTRUCTURE (CRITICAL)

#### 8.1 Hosting Setup
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Recommended Stack for MVP:

Frontend + API:
□ Vercel (Next.js) - Free tier to start, Pro for longer functions

Database:
□ Supabase (PostgreSQL) - Free tier: 500MB, 2 projects
□ OR Neon (PostgreSQL) - Free tier: 3GB

Background Processing:
□ Vercel Pro Functions (60s timeout)
□ OR Railway worker ($5/month)

File Storage (for cached light curves):
□ Vercel Blob
□ OR Supabase Storage
□ OR AWS S3

Domain:
□ larun.space (if owned)
□ Configure DNS
□ SSL automatic via Vercel
```

#### 8.2 Environment Variables
```
Priority: P0 (Blocker)
Status: [ ] Not Started / [ ] In Progress / [ ] Complete / [ ] Tested

Required Environment Variables:

# Database
DATABASE_URL=postgresql://...

# Auth
JWT_SECRET=random-32-char-string
NEXTAUTH_SECRET=random-32-char-string
NEXTAUTH_URL=https://larun.space

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_live_...

# Email (for verification, password reset)
SMTP_HOST=smtp.sendgrid.net
SMTP_USER=apikey
SMTP_PASS=SG.xxxxx

# Optional: Error tracking
SENTRY_DSN=https://...
```

---

## PHASE 1 IMPLEMENTATION CHECKLIST

### Week 1-2: Core Detection Pipeline
```
□ Set up project repository
□ Configure development environment
□ Implement MAST data access
□ Implement BLS periodogram
□ Implement phase folding
□ Implement transit detection (model or threshold-based)
□ Write unit tests for detection pipeline
□ Test on 10 known planets, 10 known false positives
```

### Week 3-4: Vetting & API
```
□ Implement odd-even test
□ Implement V-shape detection
□ Implement secondary eclipse check
□ Create vetting summary generator
□ Set up database (Supabase/Neon)
□ Implement API endpoints
□ Set up background job processing
□ Test full pipeline end-to-end
```

### Week 5-6: Frontend & Auth
```
□ Create landing page
□ Implement user authentication
□ Build analysis interface
□ Build results display with visualizations
□ Build user dashboard
□ Implement usage tracking
□ Mobile responsive testing
```

### Week 7-8: Payments & Launch
```
□ Set up Stripe products
□ Implement checkout flow
□ Implement webhook handling
□ Set up subscription management
□ Production deployment
□ Domain configuration
□ Final testing (10 test users)
□ Soft launch to early adopters
□ Monitor and fix critical bugs
```

---

## MVP SUCCESS CRITERIA

### Launch Readiness Checklist
```
□ User can register and verify email
□ User can subscribe via Stripe ($9/month or $89/year)
□ User can input TIC ID and receive analysis within 2 minutes
□ Analysis includes detection result + 3 vetting tests
□ User can view analysis history
□ Usage limit (25/month) is enforced
□ User can cancel subscription
□ System handles errors gracefully with user-friendly messages
□ Site loads in < 3 seconds
□ Works on mobile browsers
```

### Quality Thresholds
```
□ Detection accuracy: ≥85% on test set of 50 known planets
□ False positive rate: <20% (can improve post-launch)
□ API uptime: >99% (use Vercel/Railway reliability)
□ Page load time: <3 seconds
□ Analysis completion: <2 minutes for 95% of targets
```

### Post-Launch Week 1 Targets
```
□ 10+ registered users
□ 3+ paying subscribers
□ <5 critical bugs reported
□ Zero security incidents
□ Collect feedback from 5 users
```

---

## WHAT'S EXPLICITLY OUT OF SCOPE FOR MVP

Do NOT build these features for initial launch:

```
❌ FPP/NFPP Bayesian calculation (Phase 2)
❌ Gaia DR3 integration (Phase 2)
❌ MCMC posterior distributions (Phase 2)
❌ Multi-sector stitching beyond 3 sectors (Phase 2)
❌ TLS algorithm (BLS is sufficient for MVP)
❌ Peer review system (Phase 3)
❌ ExoFOP submission (Phase 3)
❌ DOI assignment (Phase 3)
❌ API access for users (Phase 3)
❌ Education tier features (Phase 2)
❌ Professional tier features (Phase 3)
❌ Mobile app (Web responsive is sufficient)
❌ Advanced stellar characterization
❌ Centroid analysis
❌ Real-time TESS data monitoring
❌ Batch processing
❌ Custom aperture photometry
```

---

## RISK MITIGATION

### Technical Risks
```
Risk: Detection accuracy too low
Mitigation: Start with conservative thresholds, display confidence prominently, iterate based on user feedback

Risk: MAST API rate limiting
Mitigation: Implement caching, queue requests, show estimated wait times

Risk: Processing takes too long
Mitigation: Show progress indicator, process in background, email when complete

Risk: Stripe integration issues
Mitigation: Test thoroughly in test mode, start with simple checkout flow
```

### Business Risks
```
Risk: No users sign up
Mitigation: Soft launch to astronomy communities (Reddit r/exoplanets, Twitter astronomy), offer free trial week

Risk: Users churn after first month
Mitigation: Collect feedback aggressively, iterate fast, add value quickly

Risk: Competitors copy features
Mitigation: Move fast, build community, iterate toward professional features
```

---

## SUPPORT & DOCUMENTATION

### Minimum Documentation for Launch
```
□ FAQ page (10 common questions)
□ "How to use" tutorial (1 page with screenshots)
□ Terms of Service
□ Privacy Policy
□ Contact email configured
```

### Support Channels
```
□ Email: support@larun.space
□ Response time target: <24 hours
□ Consider: Discord community (free, builds engagement)
```

---

## METRICS TO TRACK FROM DAY 1

```
□ Total registered users
□ Conversion rate (registered → paid)
□ Monthly Active Users (MAU)
□ Analyses per user per month
□ Churn rate (monthly)
□ Revenue (MRR)
□ Error rate (failed analyses / total)
□ Average processing time
□ User feedback (NPS or simple rating)
```

Use: Vercel Analytics (free) + Stripe Dashboard + Simple database queries

---

## PHASE 2 PREVIEW (After MVP Success)

Once MVP is stable with 50+ paying users:

```
Phase 2A: Accuracy Improvements (Month 3)
- Upgrade detection model to 95% AUC
- Add TLS algorithm option
- Improve vetting accuracy

Phase 2B: Education Tier (Month 4)
- Simplified FPP calculation
- Basic Gaia integration
- Multi-user accounts
- Admin dashboard
- Curriculum materials

Phase 2C: Professional Foundation (Month 5-6)
- Full Bayesian FPP
- MCMC posteriors
- Enhanced Gaia integration
- Peer review system (basic)
```

---

## QUICK START COMMANDS

```bash
# Clone and setup (example for Next.js)
npx create-next-app@latest larun-space --typescript --tailwind --app
cd larun-space

# Install key dependencies
npm install @stripe/stripe-js stripe
npm install next-auth
npm install @supabase/supabase-js
npm install plotly.js-dist-min react-plotly.js
npm install lightkurve  # If using Python backend

# Database setup (Supabase CLI)
npx supabase init
npx supabase db push

# Run development
npm run dev
```

---

## EXISTING LARUN COMPONENTS TO REUSE

The following components from the current LARUN codebase can be leveraged for MVP:

### Already Implemented (Ready to Use)
| Component | Location | Status |
|-----------|----------|--------|
| BLS Periodogram | `src/skills/periodogram.py` | ✅ Ready |
| Phase Folding | `src/skills/periodogram.py` | ✅ Ready |
| Odd-Even Test | `src/skills/vetting.py` | ✅ Ready |
| V-Shape Detection | `src/skills/vetting.py` | ✅ Ready |
| Secondary Eclipse | `src/skills/vetting.py` | ✅ Ready |
| MAST Data Access | `src/pipeline/nasa_pipeline.py` | ✅ Ready |
| Transit Detection | `src/model/spectral_cnn.py` | ⚠️ 81.8% AUC |
| FastAPI Backend | `api.py` | ✅ Ready |

### Needs Adaptation for Web
| Component | Current | Needed for MVP |
|-----------|---------|----------------|
| CLI Interface | `larun.py` | → Web API endpoints |
| Report Generator | `src/reporter/` | → PDF download |
| Figure Generator | `src/skills/figures.py` | → Plotly.js |

---

## CONTACT & ESCALATION

For MVP development questions:
- Technical blockers: Document and escalate immediately
- Scope creep: Refer to "OUT OF SCOPE" section
- Timeline risk: Cut features, not quality

---

*This document is the single source of truth for MVP scope. Any feature not listed here requires explicit approval before implementation.*

**START DATE:** _______________
**TARGET LAUNCH:** _______________
**OWNER:** Padmanaban

---

## DAILY STANDUP TEMPLATE

```
Yesterday:
- [ ] What I completed

Today:
- [ ] What I'm working on

Blockers:
- [ ] Any issues preventing progress

MVP Checklist Progress: X / 47 items complete
```

---

*Document Version: 1.0*
*Last Updated: February 2, 2026*

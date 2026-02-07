# Astrodata STEM Learning Platform - Development Plan

## Vision

A comprehensive STEM learning platform for astronomy and space science education, featuring interactive simulations, real NASA data analysis, AI-powered tools, and professional certifications.

---

## Phase 1: Foundation (Weeks 1-4)

### 1.1 Core Infrastructure

| Component | Technology | Priority |
|-----------|------------|----------|
| Frontend | Next.js 14 (App Router) | High |
| Database | Supabase (PostgreSQL) | High |
| Auth | NextAuth.js + Supabase Auth | High |
| Payments | LemonSqueezy / Stripe | Medium |
| Hosting | Vercel | High |
| CDN | Vercel Edge / Cloudflare | Medium |

### 1.2 Database Schema

```sql
-- Users and Profiles
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  role TEXT DEFAULT 'student', -- student, educator, institution_admin
  institution_id UUID REFERENCES institutions(id),
  subscription_tier TEXT DEFAULT 'free',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Institutions (Schools, Universities)
CREATE TABLE institutions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  type TEXT NOT NULL, -- k12, university, research, other
  domain TEXT, -- for email domain verification
  max_seats INTEGER DEFAULT 30,
  subscription_tier TEXT DEFAULT 'classroom',
  admin_user_id UUID REFERENCES auth.users(id),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Courses
CREATE TABLE courses (
  id TEXT PRIMARY KEY, -- e.g., 'exoplanet-101'
  title TEXT NOT NULL,
  description TEXT,
  difficulty TEXT NOT NULL, -- beginner, intermediate, advanced
  duration_hours INTEGER,
  thumbnail_url TEXT,
  is_public BOOLEAN DEFAULT true,
  required_tier TEXT DEFAULT 'free',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Lessons within Courses
CREATE TABLE lessons (
  id TEXT PRIMARY KEY,
  course_id TEXT REFERENCES courses(id),
  title TEXT NOT NULL,
  content_type TEXT NOT NULL, -- video, text, interactive, quiz
  content JSONB,
  order_index INTEGER,
  duration_minutes INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- User Progress
CREATE TABLE user_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  course_id TEXT REFERENCES courses(id),
  lesson_id TEXT REFERENCES lessons(id),
  status TEXT DEFAULT 'not_started', -- not_started, in_progress, completed
  score INTEGER,
  time_spent_seconds INTEGER DEFAULT 0,
  completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, lesson_id)
);

-- Certificates
CREATE TABLE certificates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  course_id TEXT REFERENCES courses(id),
  certificate_number TEXT UNIQUE,
  issued_at TIMESTAMP DEFAULT NOW(),
  pdf_url TEXT,
  verification_url TEXT
);

-- Interactive Labs
CREATE TABLE labs (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  type TEXT NOT NULL, -- simulation, data_analysis, model_training
  config JSONB,
  required_tier TEXT DEFAULT 'free',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Lab Sessions (user attempts)
CREATE TABLE lab_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  lab_id TEXT REFERENCES labs(id),
  started_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP,
  results JSONB,
  score INTEGER
);

-- Quizzes
CREATE TABLE quizzes (
  id TEXT PRIMARY KEY,
  lesson_id TEXT REFERENCES lessons(id),
  questions JSONB NOT NULL,
  passing_score INTEGER DEFAULT 70,
  max_attempts INTEGER DEFAULT 3
);

-- Quiz Attempts
CREATE TABLE quiz_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  quiz_id TEXT REFERENCES quizzes(id),
  answers JSONB,
  score INTEGER,
  passed BOOLEAN,
  attempted_at TIMESTAMP DEFAULT NOW()
);

-- Achievements/Badges
CREATE TABLE achievements (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  icon_url TEXT,
  criteria JSONB -- conditions to earn
);

CREATE TABLE user_achievements (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  achievement_id TEXT REFERENCES achievements(id),
  earned_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, achievement_id)
);

-- API Usage Tracking
CREATE TABLE api_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  endpoint TEXT NOT NULL,
  method TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW(),
  response_time_ms INTEGER
);
```

### 1.3 Authentication Flow

```
Public Access:
  /                    -> Landing page
  /explore             -> Interactive demos
  /learn               -> Course catalog
  /learn/[course]      -> Course overview
  /playground          -> Model playground

Auth Required:
  /dashboard           -> User dashboard
  /learn/[course]/[lesson] -> Lesson content (progress saved)
  /certificates        -> View certificates
  /settings            -> Account settings

Institution Admin:
  /admin/students      -> Manage students
  /admin/progress      -> View class progress
  /admin/assignments   -> Create assignments
```

---

## Phase 2: Learning Content (Weeks 5-8)

### 2.1 Course Structure

```
Learning Paths:
├── Beginner Track
│   ├── Introduction to Astronomy
│   ├── Understanding Light Curves
│   └── Your First Exoplanet Detection
│
├── Intermediate Track
│   ├── Transit Detection Methods
│   ├── Data Analysis with Python
│   └── Machine Learning Basics
│
└── Advanced Track
    ├── TinyML for Space Applications
    ├── Building Detection Pipelines
    └── Research Project
```

### 2.2 Course Content Types

| Type | Description | Implementation |
|------|-------------|----------------|
| Video | Pre-recorded lessons | Mux / YouTube embed |
| Text | Written tutorials | MDX with code blocks |
| Interactive | Simulations | React components |
| Quiz | Knowledge checks | Custom quiz engine |
| Lab | Hands-on exercises | Jupyter-lite / Custom |
| Project | Capstone work | Submission system |

### 2.3 Initial Courses

**Course 1: Exoplanet Detection 101**
```
Lessons:
1. What is an Exoplanet?
2. The Transit Method Explained
3. Understanding Light Curves
4. Reading NASA Data
5. Your First Detection (Lab)
6. Quiz: Fundamentals
Duration: 2 hours
Certificate: Exoplanet Explorer
```

**Course 2: Light Curve Analysis**
```
Lessons:
1. Anatomy of a Light Curve
2. Signal vs Noise
3. Transit Depth and Duration
4. Identifying False Positives
5. Hands-on Analysis (Lab)
6. Final Assessment
Duration: 3 hours
Certificate: Light Curve Analyst
```

**Course 3: Machine Learning for Astronomy**
```
Lessons:
1. ML in Astronomy Overview
2. Feature Engineering
3. Training a Classifier
4. Model Evaluation
5. Build Your Own Model (Lab)
6. Project Submission
Duration: 5 hours
Certificate: Astro ML Specialist
```

---

## Phase 3: Interactive Features (Weeks 9-12)

### 3.1 Interactive Labs

| Lab | Description | Technology |
|-----|-------------|------------|
| Transit Simulator | Visualize planet transits | Three.js / D3 |
| Light Curve Explorer | Analyze real Kepler data | Plotly / Chart.js |
| Model Playground | Train/test models | TensorFlow.js |
| Signal Hunter | Find transits in data | Custom game engine |
| Orbit Designer | Create planetary systems | Physics simulation |

### 3.2 Gamification

```
Achievement System:
- First Login
- Complete First Lesson
- Finish First Course
- Perfect Quiz Score
- 7-Day Streak
- Discover a Transit
- Train a Model
- Earn Certificate

Leaderboards:
- Weekly Top Learners
- Most Discoveries
- Quiz Champions
- Institution Rankings
```

### 3.3 Real Data Integration

```typescript
// NASA Data APIs
const dataSources = {
  tess: 'https://mast.stsci.edu/api/v0.1/tess',
  kepler: 'https://exoplanetarchive.ipac.caltech.edu/TAP',
  exoplanets: 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI'
};

// Features:
// - Browse confirmed exoplanets
// - View real light curves
// - Analyze TOI candidates
// - Compare with ML predictions
```

---

## Phase 4: Monetization (Weeks 13-16)

### 4.1 Pricing Tiers

**Individual Plans**

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | 2 courses, basic labs, community |
| Student | $9/mo | All courses, certificates, API (1K/mo) |
| Pro | $29/mo | Priority support, API (10K/mo), downloads |

**Institution Plans**

| Tier | Price | Features |
|------|-------|----------|
| Classroom | $199/yr | 30 seats, teacher dashboard |
| Department | $999/yr | 150 seats, LMS integration, analytics |
| Campus | $4,999/yr | Unlimited, SSO, dedicated support |

### 4.2 Payment Integration

```typescript
// LemonSqueezy Integration
const variants = {
  student_monthly: 'variant_xxxxx',
  student_annual: 'variant_xxxxx',
  pro_monthly: 'variant_xxxxx',
  pro_annual: 'variant_xxxxx',
  classroom: 'variant_xxxxx',
  department: 'variant_xxxxx',
  campus: 'variant_xxxxx'
};
```

### 4.3 Usage Limits

| Feature | Free | Student | Pro | Institution |
|---------|------|---------|-----|-------------|
| Courses | 2 | All | All | All |
| Labs | 5/month | Unlimited | Unlimited | Unlimited |
| API Calls | 100/month | 1,000/month | 10,000/month | Custom |
| Certificates | No | Yes | Yes | Yes |
| Support | Community | Email | Priority | Dedicated |

---

## Phase 5: Institution Features (Weeks 17-20)

### 5.1 Teacher Dashboard

```
Features:
- Class roster management
- Assignment creation
- Progress monitoring
- Grade export
- Bulk enrollment
- Custom learning paths
```

### 5.2 LMS Integration

```
Supported Platforms:
- Canvas (LTI 1.3)
- Blackboard
- Moodle
- Google Classroom

Features:
- Single Sign-On
- Grade passback
- Assignment sync
- Roster sync
```

### 5.3 Analytics

```
Metrics:
- Course completion rates
- Average quiz scores
- Time on platform
- Most popular content
- Drop-off points
- Engagement trends
```

---

## Phase 6: Advanced Features (Weeks 21-24)

### 6.1 AI Tutor

```
Features:
- Answer questions about content
- Explain concepts
- Provide hints in labs
- Review submissions
- Personalized recommendations

Technology:
- Claude API / OpenAI
- RAG with course content
- Context-aware responses
```

### 6.2 Research Mode

```
Features:
- Access to full NASA catalogs
- Batch processing
- Custom model training
- Publication-ready exports
- Collaboration tools
```

### 6.3 Mobile App

```
Features:
- Offline course viewing
- Push notifications
- Progress sync
- Quick quizzes

Technology:
- React Native / Expo
- Shared component library
```

---

## Technical Architecture

### Frontend Structure

```
astrodata/
├── src/
│   ├── app/
│   │   ├── (public)/           # No auth required
│   │   │   ├── page.tsx        # Landing
│   │   │   ├── explore/
│   │   │   ├── learn/
│   │   │   └── playground/
│   │   │
│   │   ├── (auth)/             # Auth required
│   │   │   ├── dashboard/
│   │   │   ├── certificates/
│   │   │   └── settings/
│   │   │
│   │   ├── (admin)/            # Institution admin
│   │   │   ├── students/
│   │   │   ├── progress/
│   │   │   └── assignments/
│   │   │
│   │   └── api/
│   │       ├── auth/
│   │       ├── courses/
│   │       ├── progress/
│   │       └── webhooks/
│   │
│   ├── components/
│   │   ├── ui/                 # Base components
│   │   ├── course/             # Course components
│   │   ├── lab/                # Lab components
│   │   └── charts/             # Data viz
│   │
│   ├── lib/
│   │   ├── supabase.ts
│   │   ├── auth.ts
│   │   ├── nasa-api.ts
│   │   └── ml-inference.ts
│   │
│   └── content/
│       ├── courses/            # MDX course content
│       └── labs/               # Lab configurations
```

### API Routes

```
/api/v1/
├── auth/
│   ├── [...nextauth]
│   └── session
├── courses/
│   ├── GET /                   # List courses
│   ├── GET /[id]               # Course details
│   └── GET /[id]/lessons       # Course lessons
├── progress/
│   ├── GET /                   # User progress
│   ├── POST /[lessonId]        # Update progress
│   └── GET /certificates       # User certificates
├── labs/
│   ├── GET /                   # List labs
│   ├── POST /[id]/start        # Start session
│   └── POST /[id]/submit       # Submit results
├── quizzes/
│   ├── GET /[id]               # Get quiz
│   └── POST /[id]/submit       # Submit attempt
└── admin/
    ├── GET /students           # List students
    ├── POST /invite            # Invite student
    └── GET /analytics          # Class analytics
```

---

## Development Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Foundation | 4 weeks | Auth, DB, basic UI |
| 2. Content | 4 weeks | 3 courses, quiz system |
| 3. Interactive | 4 weeks | Labs, gamification |
| 4. Monetization | 4 weeks | Payments, tiers |
| 5. Institutions | 4 weeks | Admin tools, LMS |
| 6. Advanced | 4 weeks | AI tutor, research mode |

**Total: 24 weeks (6 months)**

---

## MVP Scope (8 weeks)

For a minimum viable product, focus on:

1. **Authentication** (public + login)
2. **2-3 courses** with text/quiz content
3. **1 interactive lab** (Transit Simulator)
4. **Progress tracking** for logged-in users
5. **Basic certificates** (PDF generation)
6. **Simple pricing** (Free + Paid tier)

This allows launch and validation before building advanced features.

---

## Success Metrics

| Metric | Target (6 months) |
|--------|-------------------|
| Registered users | 5,000 |
| Paid subscribers | 200 |
| Course completions | 1,000 |
| Certificates issued | 500 |
| Institution accounts | 10 |
| MRR | $3,000 |

---

## Next Steps

1. Set up Supabase project for astrodata
2. Initialize Next.js project structure
3. Implement authentication flow
4. Create first course content
5. Build progress tracking system
6. Deploy MVP to Vercel

---

*Document Version: 1.0*
*Last Updated: 2026-02-07*

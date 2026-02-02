# LARUN.SPACE Implementation Plan

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  LARUN.SPACE - Professional-Grade Exoplanet Detection Platform              ║
║  Implementation Plan based on TRD-LARUN-2026-001                            ║
║  Created: February 2, 2026                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Executive Summary

This document provides a detailed implementation plan for transforming LARUN from a research prototype (81.8% AUC) to a professional-grade exoplanet detection platform meeting all 39 requirements in the Technical Requirements Document (TRD).

**Key Metrics:**
- Current Detection AUC: 81.8% → Target: ≥95%
- Requirements Complete: 14/39 (36%)
- Estimated Development: 8 Phases, ~20 weeks

---

## Table of Contents

1. [Gap Analysis Summary](#1-gap-analysis-summary)
2. [Implementation Phases](#2-implementation-phases)
3. [Phase 1: Core Detection Enhancement](#3-phase-1-core-detection-enhancement)
4. [Phase 2: Validation Engine](#4-phase-2-validation-engine)
5. [Phase 3: Uncertainty Quantification](#5-phase-3-uncertainty-quantification)
6. [Phase 4: External Integrations](#6-phase-4-external-integrations)
7. [Phase 5: Publication Pipeline](#7-phase-5-publication-pipeline)
8. [Phase 6: Web Platform & UI](#8-phase-6-web-platform--ui)
9. [Phase 7: Peer Review System](#9-phase-7-peer-review-system)
10. [Phase 8: Security & Performance](#10-phase-8-security--performance)
11. [Risk Assessment](#11-risk-assessment)
12. [Resource Requirements](#12-resource-requirements)

---

## 1. Gap Analysis Summary

### 1.1 Requirements Status Overview

| Category | Total | Complete | Partial | Not Started |
|----------|-------|----------|---------|-------------|
| Detection (DET) | 7 | 3 | 1 | 3 |
| Validation (VAL) | 6 | 0 | 2 | 4 |
| Vetting (VET) | 5 | 3 | 0 | 2 |
| Uncertainty (UNC) | 3 | 0 | 1 | 2 |
| External (EXT) | 4 | 2 | 1 | 1 |
| Publication (PUB) | 3 | 0 | 0 | 3 |
| Review (REV) | 3 | 0 | 0 | 3 |
| Performance (PER) | 3 | 1 | 1 | 1 |
| UI | 3 | 0 | 1 | 2 |
| Security (SEC) | 3 | 0 | 1 | 2 |
| **TOTAL** | **39** | **9** | **8** | **22** |

### 1.2 Current Implementation Mapping

#### Detection Requirements

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| DET-001 | AUC ≥95% | ⚠️ GAP | 81.8% AUC | Need +13.2% improvement |
| DET-002 | SDE 7.1σ | ✅ DONE | `min_snr=7.0` in periodogram.py | Minor calibration |
| DET-003 | BLS Periodogram | ✅ DONE | `src/skills/periodogram.py` | Benchmarking needed |
| DET-004 | TLS Support | ❌ MISSING | None | Full implementation |
| DET-005 | Phase Folding | ✅ DONE | `phase_fold()` in periodogram.py | Sub-second accuracy TBD |
| DET-006 | Multi-sector Stitching | ⚠️ PARTIAL | Basic support in pipeline | 6+ sector support |
| DET-007 | Real-time Processing | ❌ MISSING | None | Full implementation |

#### Validation Requirements

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| VAL-001 | FPP Engine | ⚠️ PARTIAL | `src/skills/fpp.py` (simplified) | Full Bayesian engine |
| VAL-002 | NFPP Calculation | ❌ MISSING | None | Gaia-based NFPP |
| VAL-003 | 8 FP Scenarios | ⚠️ PARTIAL | 5 scenarios implemented | +3 scenarios needed |
| VAL-004 | TTV-Robust | ❌ MISSING | None | TTV handling up to 10% |
| VAL-005 | Ensemble Validation | ❌ MISSING | None | ML + Bayesian ensemble |
| VAL-006 | Centroid Analysis | ❌ MISSING | None | TPF centroid motion |

#### Vetting Requirements

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| VET-001 | Odd-Even Test | ✅ DONE | `vetting.py:odd_even_test()` | None |
| VET-002 | V-Shape Detection | ✅ DONE | `vetting.py:v_shape_test()` | Calibration needed |
| VET-003 | Secondary Eclipse | ✅ DONE | `vetting.py:secondary_eclipse_test()` | None |
| VET-004 | Flux Contamination | ❌ MISSING | None | Gaia-based contamination |
| VET-005 | Stellar Variability | ❌ MISSING | None | Classification system |

#### Uncertainty Quantification

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| UNC-001 | MCMC Posteriors | ❌ MISSING | scipy.optimize only | emcee/PyMC integration |
| UNC-002 | Uncertainty Propagation | ⚠️ PARTIAL | Basic error estimates | Full propagation chain |
| UNC-003 | Confidence Intervals | ❌ MISSING | None | Asymmetric intervals |

#### External Integrations

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| EXT-001 | MAST/TESS Access | ✅ DONE | `nasa_pipeline.py` | TPF/FFI support |
| EXT-002 | Gaia DR3 | ✅ DONE | `src/skills/gaia.py` | Validation needed |
| EXT-003 | TIC Cross-matching | ⚠️ PARTIAL | Basic support | Full sector info |
| EXT-004 | ExoFOP Submission | ❌ MISSING | None | Full implementation |

#### Publication Pipeline

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| PUB-001 | DOI Assignment | ❌ MISSING | None | Zenodo integration |
| PUB-002 | Journal Templates | ❌ MISSING | None | AASTeX, MNRAS templates |
| PUB-003 | CTOI Submission | ❌ MISSING | None | Format generation |

#### Peer Review System

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| REV-001 | Peer Review Workflow | ❌ MISSING | None | Full workflow |
| REV-002 | Expert Assignment | ❌ MISSING | None | Matching algorithm |
| REV-003 | Review Tracking | ❌ MISSING | None | Audit system |

#### Performance & UI

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| PER-001 | <5min Processing | ✅ LIKELY | Current pipeline | Benchmark validation |
| PER-002 | Batch Processing | ⚠️ PARTIAL | Basic async support | 100-target queue |
| PER-003 | 100 Concurrent Users | ❌ MISSING | None | Web infrastructure |
| UI-001 | No-Code Interface | ❌ MISSING | CLI only | Full web UI |
| UI-002 | Interactive Viz | ⚠️ PARTIAL | `figures.py` | Plotly/D3.js interactive |
| UI-003 | Mobile Responsive | ❌ MISSING | None | Responsive design |

#### Security

| Req ID | Requirement | Status | Current Implementation | Gap |
|--------|-------------|--------|----------------------|-----|
| SEC-001 | Authentication | ❌ MISSING | None | OAuth 2.0 + MFA |
| SEC-002 | Encryption | ❌ MISSING | None | TLS 1.3 + AES-256 |
| SEC-003 | API Security | ⚠️ PARTIAL | Basic FastAPI | Rate limiting, tokens |

---

## 2. Implementation Phases

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Core Detection Enhancement                    [Weeks 1-3]        │
│  ├─ DET-001: Improve AUC from 81.8% to 95%                                │
│  ├─ DET-004: Implement TLS algorithm                                       │
│  └─ DET-006: Multi-sector stitching (6+ sectors)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: Validation Engine                             [Weeks 3-5]        │
│  ├─ VAL-001: Full Bayesian FPP engine                                      │
│  ├─ VAL-002: NFPP with Gaia                                               │
│  ├─ VAL-003: Complete 8 FP scenarios                                       │
│  ├─ VAL-004: TTV-robust validation                                        │
│  └─ VAL-005: Ensemble validation (ML + Bayesian)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Uncertainty Quantification                    [Weeks 5-7]        │
│  ├─ UNC-001: MCMC with emcee/PyMC                                         │
│  ├─ UNC-002: Full uncertainty propagation                                  │
│  ├─ UNC-003: Asymmetric confidence intervals                              │
│  └─ VAL-006: Centroid analysis                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 4: External Integrations                         [Weeks 7-9]        │
│  ├─ EXT-001: TPF/FFI support                                              │
│  ├─ EXT-003: Full TIC cross-matching                                      │
│  ├─ EXT-004: ExoFOP submission                                            │
│  └─ VET-004/005: Contamination & variability                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 5: Publication Pipeline                          [Weeks 9-11]       │
│  ├─ PUB-001: Zenodo DOI integration                                       │
│  ├─ PUB-002: Journal templates (AJ, ApJ, MNRAS, A&A)                      │
│  └─ PUB-003: CTOI submission format                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 6: Web Platform & UI                             [Weeks 11-15]      │
│  ├─ UI-001: No-code web interface                                         │
│  ├─ UI-002: Interactive visualizations                                     │
│  ├─ UI-003: Mobile responsive design                                       │
│  └─ DET-007: Real-time processing                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 7: Peer Review System                            [Weeks 15-17]      │
│  ├─ REV-001: Review workflow                                              │
│  ├─ REV-002: Expert assignment                                            │
│  └─ REV-003: Audit tracking                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 8: Security & Performance                        [Weeks 17-20]      │
│  ├─ SEC-001: OAuth 2.0 + MFA                                              │
│  ├─ SEC-002: TLS 1.3 + AES-256                                            │
│  ├─ SEC-003: API security hardening                                        │
│  ├─ PER-002: Batch processing (100 targets)                               │
│  └─ PER-003: 100 concurrent users                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Priority Matrix

```
                         IMPACT
                         High │ ★ DET-001 (AUC)      ★ VAL-001 (FPP)
                              │ ★ UNC-001 (MCMC)     ★ UI-001 (Web)
                              │ ★ SEC-001 (Auth)
                              │
                         Med  │ ○ DET-004 (TLS)      ○ VAL-005 (Ensemble)
                              │ ○ PUB-001 (DOI)      ○ REV-001 (Review)
                              │
                         Low  │ · UI-003 (Mobile)    · PUB-002 (Templates)
                              │ · REV-002 (Expert)
                              │
                              └──────────────────────────────────────────
                                   Low              Med              High
                                               COMPLEXITY
```

---

## 3. Phase 1: Core Detection Enhancement

**Duration:** Weeks 1-3
**Priority:** CRITICAL
**Dependencies:** None

### 3.1 DET-001: Detection Model AUC ≥95%

**Current State:** 81.8% AUC with basic 1D CNN
**Target:** ≥95% AUC on Kepler/TESS test dataset

#### Implementation Tasks

```python
# File: src/model/enhanced_cnn.py

class EnhancedTransitCNN:
    """
    Enhanced CNN architecture for 95%+ AUC detection.

    Improvements over baseline:
    1. Deeper architecture with residual connections
    2. Attention mechanism for transit localization
    3. Multi-scale feature extraction
    4. Data augmentation pipeline
    5. Knowledge distillation from larger model
    """
```

**Task Breakdown:**

| Task | Description | File(s) | Est. Hours |
|------|-------------|---------|------------|
| 1.1 | Expand training dataset to 500+ per class | `scripts/fetch_training_data.py` | 8h |
| 1.2 | Implement data augmentation pipeline | `src/augmentation.py` | 6h |
| 1.3 | Design enhanced CNN with residual blocks | `src/model/enhanced_cnn.py` | 12h |
| 1.4 | Add attention mechanism | `src/model/attention.py` | 8h |
| 1.5 | Implement knowledge distillation | `src/model/distillation.py` | 10h |
| 1.6 | Training with K-fold cross-validation | `scripts/train_enhanced.py` | 8h |
| 1.7 | Quantization-aware training | `src/model/quantize.py` | 6h |
| 1.8 | Benchmark against Kepler DR25 | `tests/test_det_001.py` | 4h |

**Architecture:**

```
Input (1024 bins)
    │
    ├─── Conv1D(32, 7) + BatchNorm + ReLU
    │         │
    │    ResidualBlock(32)
    │         │
    │    ResidualBlock(64)
    │         │
    │    AttentionLayer(64)
    │         │
    │    ResidualBlock(128)
    │         │
    │    GlobalAvgPool
    │         │
    │    Dense(64) + Dropout(0.3)
    │         │
    │    Dense(6, softmax)  # 6-class output
    │
Output: [Planet, EB, BEB, Variable, Artifact, NoSignal]
```

**Acceptance Criteria:**
- [ ] AUC ≥ 0.95 on held-out test set
- [ ] No single fold below 0.93 in 10-fold CV
- [ ] Model size < 100KB (TFLite)
- [ ] Inference time < 10ms on Cortex-M4

### 3.2 DET-004: Transit Least Squares (TLS)

**Current State:** Not implemented
**Target:** TLS with 15% better sensitivity for small planets

#### Implementation

```python
# File: src/skills/tls.py

"""
Transit Least Squares Implementation
====================================
Based on: Hippke & Heller (2019) - A&A 623, A39

TLS improves upon BLS by:
1. Using realistic limb-darkened transit shapes
2. Proper handling of transit ingress/egress
3. Better SNR estimation for small planets
"""

class TLSPeriodogram:
    """
    Transit Least Squares for improved small planet detection.

    Features:
    - Realistic Mandel-Agol transit shapes
    - Configurable limb darkening
    - Period/duration grid optimization
    - FAP calculation using bootstrap
    """

    def __init__(
        self,
        min_period: float = 0.5,
        max_period: float = 100.0,
        limb_dark: str = "quadratic",
        stellar_params: Optional[StellarParams] = None
    ):
        ...

    def compute(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None
    ) -> TLSResult:
        """
        Compute TLS periodogram.

        Returns TLSResult with:
        - periods, power arrays
        - best_period, best_power
        - transit parameters (depth, duration, impact)
        - SDE (Signal Detection Efficiency)
        - FAP (False Alarm Probability)
        """
        ...
```

**Task Breakdown:**

| Task | Description | File(s) | Est. Hours |
|------|-------------|---------|------------|
| 1.9 | Core TLS algorithm | `src/skills/tls.py` | 16h |
| 1.10 | Transit model templates | `src/skills/tls_models.py` | 8h |
| 1.11 | Grid optimization | `src/skills/tls.py` | 6h |
| 1.12 | FAP via bootstrap | `src/skills/tls.py` | 4h |
| 1.13 | Compare TLS vs BLS | `tests/test_det_004.py` | 4h |
| 1.14 | CLI integration | `larun.py` | 2h |

**Acceptance Criteria:**
- [ ] TLS detects ≥15% more Earth-sized planets than BLS
- [ ] Processing time within 2x of BLS
- [ ] Integrated with CLI: `/tls` command

### 3.3 DET-006: Multi-sector Stitching

**Current State:** Basic single-sector support
**Target:** Seamless 6+ sector stitching

#### Implementation

```python
# File: src/pipeline/sector_stitcher.py

class SectorStitcher:
    """
    Multi-sector light curve stitching for TESS data.

    Handles:
    - Inter-sector gaps (momentum dumps, data gaps)
    - Different cadences (2-min, 20-sec, FFI)
    - Systematic offset normalization
    - Quality flag propagation
    """

    def stitch(
        self,
        sector_data: List[SectorLightCurve],
        normalize_method: str = "median",
        gap_fill: str = "none"
    ) -> StitchedLightCurve:
        """
        Stitch multiple sectors into continuous light curve.

        Args:
            sector_data: List of per-sector light curves
            normalize_method: "median", "mean", or "running"
            gap_fill: "none", "interpolate", or "model"

        Returns:
            StitchedLightCurve with continuous time/flux arrays
        """
        ...
```

**Task Breakdown:**

| Task | Description | File(s) | Est. Hours |
|------|-------------|---------|------------|
| 1.15 | Sector stitcher class | `src/pipeline/sector_stitcher.py` | 10h |
| 1.16 | Cadence harmonization | `src/pipeline/sector_stitcher.py` | 6h |
| 1.17 | Quality flag handling | `src/pipeline/sector_stitcher.py` | 4h |
| 1.18 | Integration tests | `tests/test_det_006.py` | 4h |

**Acceptance Criteria:**
- [ ] Successful stitching of 6+ sectors
- [ ] Flux offset < 0.1% between sectors
- [ ] No visible artifacts at sector boundaries

---

## 4. Phase 2: Validation Engine

**Duration:** Weeks 3-5
**Priority:** CRITICAL
**Dependencies:** Phase 1 (detection improvements)

### 4.1 VAL-001: Full Bayesian FPP Engine

**Current State:** Simplified FPP in `fpp.py`
**Target:** TRICERATOPS-equivalent Bayesian engine

#### Architecture

```python
# File: src/validation/fpp_engine.py

class BayesianFPPEngine:
    """
    Full Bayesian False Positive Probability engine.

    Implements methodology from:
    - Morton (2012) - VESPA
    - Giacalone & Dressing (2020) - TRICERATOPS

    Evaluates posterior probabilities for all FP scenarios
    using transit photometry, stellar properties, and
    nearby source information.
    """

    def __init__(
        self,
        stellar_params: StellarParams,
        gaia_sources: List[GaiaSource],
        transit_params: TransitParams
    ):
        self.stellar = stellar_params
        self.nearby_sources = gaia_sources
        self.transit = transit_params

        # Initialize scenario priors
        self.priors = self._calculate_priors()

    def calculate_fpp(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        n_samples: int = 10000
    ) -> FPPResult:
        """
        Calculate FPP with full scenario breakdown.

        Returns:
            FPPResult with:
            - fpp: Overall false positive probability
            - nfpp: Nearby false positive probability
            - scenarios: Dict[str, ScenarioProbability]
            - posteriors: Full posterior samples
        """
        ...
```

**Scenario Coverage (VAL-003):**

```python
SCENARIOS = {
    'planet': PlanetScenario(),           # Genuine planet
    'eb': EclipsingBinaryScenario(),      # Target is EB
    'beb': BackgroundEBScenario(),        # Background EB
    'heb': HierarchicalEBScenario(),      # Triple system
    'neb': NearbyEBScenario(),            # Blended nearby EB
    'btp': BoundCompanionPlanet(),        # Planet on companion
    'artifact': InstrumentalArtifact(),   # Systematic
    'variability': StellarVariability(),  # Spots/pulsations
}
```

**Task Breakdown:**

| Task | Description | File(s) | Est. Hours |
|------|-------------|---------|------------|
| 2.1 | FPP engine architecture | `src/validation/fpp_engine.py` | 16h |
| 2.2 | Planet scenario model | `src/validation/scenarios/planet.py` | 8h |
| 2.3 | EB scenario models (3 types) | `src/validation/scenarios/eb.py` | 12h |
| 2.4 | Blend/artifact scenarios | `src/validation/scenarios/blend.py` | 8h |
| 2.5 | Prior probability calibration | `src/validation/priors.py` | 6h |
| 2.6 | NFPP calculation (VAL-002) | `src/validation/nfpp.py` | 10h |
| 2.7 | Validation against TRICERATOPS | `tests/test_val_001.py` | 8h |

### 4.2 VAL-004: TTV-Robust Validation

**Current State:** Not implemented
**Target:** Valid FPP for TTV amplitudes up to 10%

```python
# File: src/validation/ttv_handler.py

class TTVRobustValidator:
    """
    Handle Transit Timing Variations in FPP calculation.

    Problem: TRICERATOPS fails for TTV > 5.38% amplitude
    Solution: Use TTV-aware transit model with variable epochs

    Reference: TRD Section 4 (VAL-004)
    """

    def validate_with_ttv(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        transit_times: np.ndarray,
        max_ttv_fraction: float = 0.10
    ) -> TTVValidationResult:
        """
        Perform FPP calculation accounting for TTVs.

        1. Fit individual transit times
        2. Calculate TTV amplitude
        3. If TTV < threshold: use standard FPP
        4. If TTV > threshold: use TTV-aware model
        5. Flag for additional scrutiny if TTV > 10%
        """
        ...
```

### 4.3 VAL-005: Ensemble Validation

**Current State:** Single method only
**Target:** ML + Bayesian ensemble with agreement reporting

```python
# File: src/validation/ensemble.py

class EnsembleValidator:
    """
    Ensemble validation combining multiple methods.

    Methods:
    1. Bayesian FPP (TRICERATOPS-style)
    2. ML classification (ExoMiner-style)
    3. Statistical vetting tests

    Reports agreement/disagreement for each candidate.
    """

    def validate(
        self,
        candidate: TransitCandidate
    ) -> EnsembleResult:
        # Run all validators
        fpp_result = self.bayesian_fpp.calculate(candidate)
        ml_result = self.ml_classifier.predict(candidate)
        vet_result = self.vetting_suite.run_all(candidate)

        # Calculate agreement
        agreement = self._calculate_agreement(
            fpp_result, ml_result, vet_result
        )

        return EnsembleResult(
            fpp=fpp_result,
            ml_prediction=ml_result,
            vetting=vet_result,
            agreement_score=agreement,
            recommendation=self._get_recommendation(agreement)
        )
```

---

## 5. Phase 3: Uncertainty Quantification

**Duration:** Weeks 5-7
**Priority:** CRITICAL
**Dependencies:** Phase 2 (validation engine)

### 5.1 UNC-001: MCMC Posterior Distributions

**Current State:** scipy.optimize only
**Target:** Full MCMC with emcee

```python
# File: src/fitting/mcmc_fitter.py

class MCMCTransitFitter:
    """
    MCMC-based transit model fitting using emcee.

    Generates full posterior distributions for:
    - Period (P)
    - Epoch (T0)
    - Transit depth (Rp/Rs)^2
    - Impact parameter (b)
    - Transit duration (T14)
    - Limb darkening coefficients (u1, u2)

    Based on: Foreman-Mackey et al. (2013) - emcee
    """

    def __init__(
        self,
        n_walkers: int = 32,
        n_steps: int = 5000,
        n_burn: int = 1000
    ):
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burn = n_burn

    def fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        initial_params: Dict[str, float]
    ) -> MCMCResult:
        """
        Run MCMC fit and return posterior samples.

        Returns:
            MCMCResult with:
            - samples: (n_samples, n_params) array
            - medians: Median of each parameter
            - uncertainties: 16th/84th percentiles
            - gelman_rubin: Convergence diagnostic
            - effective_n: Effective sample size
            - corner_plot: Path to corner plot
        """
        import emcee

        # Set up sampler
        ndim = len(initial_params)
        sampler = emcee.EnsembleSampler(
            self.n_walkers, ndim, self._log_posterior,
            args=(time, flux, flux_err)
        )

        # Initialize walkers
        p0 = self._initialize_walkers(initial_params)

        # Run MCMC
        sampler.run_mcmc(p0, self.n_steps, progress=True)

        # Extract results
        samples = sampler.get_chain(discard=self.n_burn, flat=True)

        return self._build_result(samples)
```

**Task Breakdown:**

| Task | Description | File(s) | Est. Hours |
|------|-------------|---------|------------|
| 3.1 | MCMC fitter with emcee | `src/fitting/mcmc_fitter.py` | 16h |
| 3.2 | Convergence diagnostics | `src/fitting/diagnostics.py` | 6h |
| 3.3 | Corner plot generation | `src/fitting/plotting.py` | 4h |
| 3.4 | Uncertainty propagation (UNC-002) | `src/fitting/propagation.py` | 10h |
| 3.5 | Asymmetric intervals (UNC-003) | `src/fitting/intervals.py` | 4h |
| 3.6 | Integration tests | `tests/test_unc.py` | 6h |

### 5.2 VAL-006: Centroid Analysis

```python
# File: src/validation/centroid.py

class CentroidAnalyzer:
    """
    Centroid motion analysis during transit.

    If transit source is off-target, flux-weighted centroid
    will shift during transit. Detects:
    - Background EBs
    - Blended nearby sources
    - Scattered light contamination

    Requires Target Pixel File (TPF) data.
    """

    def analyze(
        self,
        tpf: TargetPixelFile,
        transit_mask: np.ndarray
    ) -> CentroidResult:
        """
        Measure centroid shift during transit.

        Returns:
            CentroidResult with:
            - in_transit_centroid: (x, y) during transit
            - out_transit_centroid: (x, y) outside transit
            - offset: Distance between centroids (pixels)
            - offset_arcsec: Offset in arcseconds
            - significance: Offset / uncertainty
            - is_on_target: bool
        """
        ...
```

---

## 6. Phase 4: External Integrations

**Duration:** Weeks 7-9
**Priority:** HIGH
**Dependencies:** Phase 3 (MCMC fitting)

### 6.1 EXT-004: ExoFOP Submission

```python
# File: src/integrations/exofop.py

class ExoFOPSubmitter:
    """
    Generate and submit data packages to ExoFOP-TESS.

    ExoFOP submission format includes:
    - Target information (TIC ID, coordinates)
    - Transit parameters (period, epoch, depth, duration)
    - FPP results and vetting status
    - Light curve data and plots
    - Observer information

    Reference: https://exofop.ipac.caltech.edu/tess/
    """

    def generate_package(
        self,
        candidate: ValidatedCandidate
    ) -> ExoFOPPackage:
        """Generate ExoFOP-compatible data package."""
        ...

    def submit(
        self,
        package: ExoFOPPackage,
        api_key: str
    ) -> SubmissionResult:
        """Submit package to ExoFOP."""
        ...
```

### 6.2 VET-004: Flux Contamination Assessment

```python
# File: src/vetting/contamination.py

class ContaminationAssessor:
    """
    Calculate flux contamination from nearby Gaia sources.

    Uses:
    - Gaia DR3 magnitudes (G, BP, RP)
    - TESS bandpass transmission
    - Aperture geometry (pixel mask)

    Outputs:
    - Contamination ratio (0-1)
    - Source-by-source breakdown
    - Corrected transit depth
    """

    def calculate(
        self,
        target_tmag: float,
        gaia_sources: List[GaiaSource],
        aperture_mask: np.ndarray,
        pixel_scale: float = 21.0  # arcsec/pixel
    ) -> ContaminationResult:
        ...
```

### 6.3 VET-005: Stellar Variability Classification

```python
# File: src/vetting/variability.py

class VariabilityClassifier:
    """
    Classify stellar variability patterns.

    Types:
    - Rotational modulation (starspots)
    - Pulsations (delta Scuti, gamma Dor, RR Lyrae)
    - Flares (M-dwarf flares)
    - Eclipsing binaries (detached, contact)
    - Systematic trends

    Uses ML classifier trained on TESS TASOC labels.
    """

    def classify(
        self,
        time: np.ndarray,
        flux: np.ndarray
    ) -> VariabilityResult:
        # Extract variability features
        features = self._extract_features(time, flux)

        # Classify
        probs = self.classifier.predict_proba(features)

        return VariabilityResult(
            primary_type=self._get_primary_type(probs),
            probabilities=probs,
            rotation_period=self._estimate_rotation(time, flux),
            recommendations=self._get_recommendations(probs)
        )
```

---

## 7. Phase 5: Publication Pipeline

**Duration:** Weeks 9-11
**Priority:** HIGH
**Dependencies:** Phase 4

### 7.1 PUB-001: Zenodo DOI Integration

```python
# File: src/publication/doi.py

class DOIManager:
    """
    Assign DOIs to validated discoveries via Zenodo.

    Workflow:
    1. Create Zenodo deposition
    2. Upload discovery data package
    3. Add metadata (title, authors, description)
    4. Publish and receive DOI
    5. Store DOI in discovery record
    """

    def create_doi(
        self,
        discovery: ValidatedDiscovery,
        authors: List[Author]
    ) -> str:
        """
        Create Zenodo DOI for discovery.

        Returns:
            DOI string (e.g., "10.5281/zenodo.1234567")
        """
        ...
```

### 7.2 PUB-002: Journal Templates

```python
# File: src/publication/templates.py

class JournalTemplateGenerator:
    """
    Generate publication-ready templates for major journals.

    Supported formats:
    - AASTeX (AJ, ApJ, ApJL)
    - MNRAS
    - A&A
    - arXiv preprint
    """

    def generate(
        self,
        discovery: ValidatedDiscovery,
        format: str = "aastex"
    ) -> str:
        """Generate LaTeX template with all figures and tables."""
        template = self._load_template(format)

        # Fill in discovery data
        template = template.replace("{{PLANET_NAME}}", discovery.name)
        template = template.replace("{{PERIOD}}", f"{discovery.period:.6f}")
        # ... etc

        return template
```

### 7.3 PUB-003: CTOI Submission

```python
# File: src/publication/ctoi.py

class CTOIGenerator:
    """
    Generate Community TESS Object of Interest submissions.

    CTOI format requirements:
    - TIC ID
    - Period, epoch, depth, duration
    - FPP value and method
    - Light curve and phase fold plots
    - Contact information
    """

    def generate(
        self,
        candidate: ValidatedCandidate
    ) -> CTOISubmission:
        """Generate CTOI submission package."""
        ...
```

---

## 8. Phase 6: Web Platform & UI

**Duration:** Weeks 11-15
**Priority:** CRITICAL
**Dependencies:** Phases 1-5 (core functionality)

### 8.1 UI-001: No-Code Web Interface

**Technology Stack:**
- Frontend: React 18 + TypeScript + Tailwind CSS
- Backend: FastAPI (existing) + PostgreSQL
- Visualization: Plotly.js + D3.js
- Deployment: Docker + Kubernetes

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Frontend (React)                      │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard │ Analysis │ Results │ Review │ Publication │ Profile │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                       │
├─────────────────────────────────────────────────────────────────┤
│  /api/v1/analyze │ /api/v1/validate │ /api/v1/submit │ /api/v1/review │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Detection      │ │  Validation     │ │  Publication    │
│  Service        │ │  Service        │ │  Service        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PostgreSQL + Redis Cache                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Pages:**

1. **Dashboard** - Overview, recent analyses, notifications
2. **Analysis Wizard** - Step-by-step no-code analysis
3. **Results Viewer** - Interactive light curves, periodograms
4. **Validation Report** - FPP breakdown, vetting results
5. **Publication Hub** - DOI, templates, CTOI submission
6. **Peer Review** - Review queue, comments, endorsements

**Task Breakdown:**

| Task | Description | Est. Hours |
|------|-------------|------------|
| 6.1 | React project setup | 8h |
| 6.2 | Authentication UI | 12h |
| 6.3 | Dashboard page | 16h |
| 6.4 | Analysis wizard | 24h |
| 6.5 | Results viewer (UI-002) | 20h |
| 6.6 | Interactive visualizations | 16h |
| 6.7 | Mobile responsive (UI-003) | 12h |
| 6.8 | API integration | 16h |
| 6.9 | Real-time updates (DET-007) | 12h |

### 8.2 DET-007: Real-time Processing

```python
# File: src/realtime/processor.py

class RealtimeProcessor:
    """
    Process incoming TESS data within 24 hours of release.

    Workflow:
    1. Monitor MAST for new data releases
    2. Download priority targets automatically
    3. Run detection pipeline
    4. Send notifications for candidates
    5. Queue for validation
    """

    async def monitor_releases(self):
        """Poll MAST for new sector releases."""
        while True:
            new_data = await self._check_mast()
            if new_data:
                await self._process_new_data(new_data)
            await asyncio.sleep(3600)  # Check hourly
```

---

## 9. Phase 7: Peer Review System

**Duration:** Weeks 15-17
**Priority:** HIGH
**Dependencies:** Phase 6 (web platform)

### 9.1 REV-001: Review Workflow

```python
# File: src/review/workflow.py

class ReviewWorkflow:
    """
    Structured peer review workflow for candidates.

    States:
    1. SUBMITTED - Candidate submitted for review
    2. ASSIGNED - Reviewers assigned
    3. IN_REVIEW - Active review in progress
    4. REVIEWED - Reviews complete
    5. ENDORSED - Community endorsed
    6. VALIDATED - Final validation
    """

    def submit_for_review(
        self,
        candidate: ValidatedCandidate
    ) -> ReviewSession:
        """Create new review session."""
        ...

    def add_review(
        self,
        session_id: str,
        reviewer_id: str,
        review: ReviewContent
    ) -> Review:
        """Add reviewer feedback."""
        ...

    def endorse(
        self,
        session_id: str,
        endorser_id: str
    ) -> Endorsement:
        """Add community endorsement."""
        ...
```

### 9.2 REV-002: Expert Assignment

```python
# File: src/review/assignment.py

class ExpertMatcher:
    """
    Match candidates with expert reviewers.

    Factors:
    - Stellar type expertise (M-dwarf specialist, etc.)
    - Planet type experience (hot Jupiters, temperate, etc.)
    - Validation method familiarity
    - Availability and response time
    """

    def assign_reviewers(
        self,
        candidate: Candidate,
        n_reviewers: int = 2
    ) -> List[Reviewer]:
        """Find best matching reviewers."""
        ...
```

---

## 10. Phase 8: Security & Performance

**Duration:** Weeks 17-20
**Priority:** CRITICAL
**Dependencies:** Phase 7

### 10.1 SEC-001: OAuth 2.0 + MFA

```python
# File: src/auth/oauth.py

class AuthenticationService:
    """
    OAuth 2.0 authentication with MFA support.

    Providers:
    - ORCID (academic identity)
    - Google
    - GitHub

    MFA Methods:
    - TOTP (Google Authenticator, etc.)
    - Email verification
    """
```

### 10.2 SEC-002: Encryption

```python
# File: src/security/encryption.py

class EncryptionService:
    """
    Data encryption for transit and at-rest.

    - TLS 1.3 for all API endpoints
    - AES-256-GCM for sensitive data at rest
    - Key rotation every 90 days
    """
```

### 10.3 PER-002/003: Batch Processing & Scaling

```python
# File: src/processing/batch.py

class BatchProcessor:
    """
    Process up to 100 targets in parallel.

    Features:
    - Job queue with priority
    - Progress tracking
    - Automatic retry on failure
    - Resource throttling
    """

    async def submit_batch(
        self,
        targets: List[str],
        user_id: str
    ) -> BatchJob:
        """Submit batch job."""
        ...
```

---

## 11. Risk Assessment

### High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| AUC improvement falls short | Cannot launch | Early benchmarking, iterative improvement |
| MCMC performance too slow | User experience | Optimize chains, use JAX/NumPyro |
| Web platform complexity | Delayed launch | MVP approach, phased feature release |
| Security vulnerabilities | Reputation damage | External audit, penetration testing |

### Medium Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| ExoFOP API changes | Submission fails | Abstract integration layer |
| Gaia DR3 data gaps | NFPP inaccurate | Fallback to TIC contamination |
| Concurrent user scaling | Poor performance | Load testing, horizontal scaling |

---

## 12. Resource Requirements

### Development Team

| Role | Count | Phase Focus |
|------|-------|-------------|
| ML Engineer | 1 | Phase 1 (detection) |
| Backend Developer | 1 | Phases 2-5 (validation, pipeline) |
| Frontend Developer | 1 | Phase 6 (web UI) |
| DevOps Engineer | 0.5 | Phases 6, 8 (deployment, security) |
| Astronomer Advisor | 0.25 | All phases (domain validation) |

### Infrastructure

| Resource | Specification | Purpose |
|----------|--------------|---------|
| GPU Server | 1x A100 or 4x T4 | Model training |
| API Server | 4 vCPU, 16GB RAM | Production API |
| Database | PostgreSQL 15, 100GB | User data, results |
| Cache | Redis, 8GB | Session, job queue |
| Storage | S3-compatible, 1TB | Light curves, models |

### External Services

| Service | Purpose | Est. Cost/Month |
|---------|---------|-----------------|
| Zenodo | DOI minting | Free |
| Auth0/ORCID | Authentication | ~$50 |
| AWS/GCP | Infrastructure | ~$500 |
| Monitoring | Datadog/Sentry | ~$100 |

---

## Summary: Implementation Roadmap

```
Week 1-3:   Phase 1 - Core Detection (AUC 95%, TLS, Multi-sector)
Week 3-5:   Phase 2 - Validation Engine (FPP, NFPP, Ensemble)
Week 5-7:   Phase 3 - Uncertainty (MCMC, Propagation, Centroid)
Week 7-9:   Phase 4 - Integrations (ExoFOP, Contamination, Variability)
Week 9-11:  Phase 5 - Publication (DOI, Templates, CTOI)
Week 11-15: Phase 6 - Web Platform (No-code UI, Real-time)
Week 15-17: Phase 7 - Peer Review (Workflow, Assignment, Tracking)
Week 17-20: Phase 8 - Security & Performance (Auth, Encryption, Scaling)
```

**Milestones:**
- Week 5: Detection AUC ≥95% achieved
- Week 9: Full validation pipeline complete
- Week 15: Web platform beta launch
- Week 20: Production launch ready

---

*Document: IMPL-LARUN-2026-001*
*Version: 1.0*
*Date: February 2, 2026*
*Author: LARUN Development Team*

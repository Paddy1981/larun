# LARUN Development Plan

```
╔══════════════════════════════════════════════════════════════════════════╗
║  LARUN TinyML - Development Roadmap                                      ║
║  Created by: Padmanaban Veeraragavalu (Larun Engineering)               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Current Status (v2.0.0)

### Completed
- [x] Core CLI with skills system
- [x] Chat interface (larun_chat.py)
- [x] NASA data fetching (TESS/Kepler via lightkurve)
- [x] Basic transit detection model (81.8% accuracy)
- [x] TFLite export for edge deployment
- [x] Distributed training system
- [x] Google Colab notebook for GPU training
- [x] Code generation addon

### Models Available
| Model | Size | Accuracy | Location |
|-------|------|----------|----------|
| Transit Detector | 50 KB | 81.8% | `models/real/astro_tinyml_real.tflite` |
| Quantized INT8 | ~25 KB | ~80% | `models/real/astro_tinyml_real_int8.tflite` |

---

## Phase 1: Core Algorithm Improvements (Priority: HIGH)

### 1.1 BLS Periodogram Implementation
**Reference**: `docs/research/EXOPLANET_DETECTION.md`

```
Status: COMPLETE
Complexity: Medium
Timeline: Sprint 1
```

**Tasks:**
- [x] Implement Box Least Squares algorithm in `src/skills/periodogram.py`
- [x] Add period grid optimization for efficiency
- [x] Integrate with astropy.timeseries.BoxLeastSquares
- [x] Create CLI command: `larun analyze bls` (also `/bls`)
- [x] Add unit tests with synthetic transit data
- [ ] Benchmark against published Kepler results

**Expected Outcome:**
- Detect periodic transits with SNR > 7
- Support period range 0.5 - 100 days
- False alarm probability calculation

### 1.2 Phase Folding & Transit Fitting
**Reference**: `docs/research/EXOPLANET_DETECTION.md`

```
Status: COMPLETE
Complexity: Medium
Timeline: Sprint 1-2
```

**Tasks:**
- [x] Implement phase folding algorithm (src/skills/periodogram.py)
- [x] Add binning for visualization
- [x] Integrate batman for transit model fitting (src/skills/transit_fit.py)
- [x] Fit: Rp/Rs, a/Rs, inclination, limb darkening
- [x] Create CLI command: `/fit`, `/phase`

### 1.3 False Positive Identification
**Reference**: `docs/research/EXOPLANET_DETECTION.md`

```
Status: COMPLETE
Complexity: High
Timeline: Sprint 2
```

**Tasks:**
- [x] Odd-even transit depth test (src/skills/vetting.py)
- [x] Secondary eclipse search
- [x] Grazing binary detection (V-shaped)
- [x] Duration test
- [x] Create CLI command: `/vet`
- [ ] Centroid shift analysis (requires TPF data)
- [ ] Create FPP (False Positive Probability) calculator

---

## Phase 2: Model Training Improvements (Priority: HIGH)

### 2.1 Increase Training Data
```
Current: ~150 samples (mixed real + synthetic)
Target: 500+ samples each class
Next retraining: After Phase 1 complete
```

**Tasks:**
- [ ] Fetch 500+ confirmed exoplanets from NASA Archive
- [x] Implement data augmentation (src/augmentation.py)
- [ ] Add K-fold cross-validation
- [ ] Target: 90%+ validation accuracy

### 2.2 Model Architecture Optimization
**Reference**: `docs/research/TINYML_OPTIMIZATION.md`

**Tasks:**
- [ ] Experiment with different architectures:
  - [ ] Deeper CNN (more layers, fewer filters)
  - [ ] 1D ResNet blocks
  - [ ] Attention mechanisms
- [ ] Knowledge distillation from larger model
- [ ] Pruning and quantization-aware training
- [ ] Target: <50KB with 90% accuracy

### 2.3 Multi-class Classification
```
Current: Binary (planet/no-planet)
Target: 6-class classification
```

**Classes:**
1. Confirmed Planet
2. Eclipsing Binary
3. Variable Star
4. Stellar Activity
5. Instrumental Artifact
6. No Signal

---

## Phase 3: New Data Sources (Priority: MEDIUM)

### 3.1 Gaia DR3 Integration
**Reference**: `docs/integrations/GAIA_INTEGRATION.md` (to create)

**Tasks:**
- [ ] Add astroquery.gaia support
- [ ] Fetch stellar parameters (Teff, logg, metallicity)
- [ ] Cross-match with TIC catalog
- [ ] Calculate stellar radii for planet radius estimation
- [ ] Create skill: `larun data gaia`

### 3.2 JWST Data Access
**Reference**: `docs/integrations/JWST_INTEGRATION.md` (to create)

**Tasks:**
- [ ] Access JWST spectra via MAST
- [ ] Atmospheric transmission spectra parsing
- [ ] Integration with exoplanet characterization
- [ ] Create skill: `larun data jwst`

### 3.3 Ground-based Survey Integration
**Tasks:**
- [ ] TESS Follow-up Observing Program (TFOP) data
- [ ] ExoFOP integration
- [ ] Radial velocity data access

---

## Phase 4: Galaxy Classification (Priority: MEDIUM)

### 4.1 Galaxy Morphology CNN
**Reference**: `docs/research/GALAXY_CLASSIFICATION.md`

**Tasks:**
- [ ] Download Galaxy Zoo dataset
- [ ] Create image preprocessing pipeline
- [ ] Train CNN for morphology classification:
  - Spiral
  - Elliptical
  - Irregular
  - Merger
- [ ] Convert to TFLite (<100KB)
- [ ] Create skill: `larun classify galaxy`

### 4.2 Galaxy Redshift Estimation
**Tasks:**
- [ ] Photometric redshift from colors
- [ ] Integration with SDSS data

---

## Phase 5: Advanced Features (Priority: LOWER)

### 5.1 Multi-planet Detection
**Reference**: `docs/research/EXOPLANET_DETECTION.md`

**Tasks:**
- [ ] Iterative transit removal
- [ ] Transit Timing Variations (TTV) detection
- [ ] Search for additional planets in residuals

### 5.2 Exomoon Search
**Tasks:**
- [ ] Transit timing/duration variations
- [ ] Photodynamic modeling
- [ ] (Research-level complexity)

### 5.3 Spectral Analysis
**Reference**: `docs/research/SPECTROSCOPY.md`

**Tasks:**
- [ ] Spectral line identification
- [ ] Radial velocity extraction
- [ ] Stellar classification from spectra

---

## Phase 6: Deployment & Integration (Ongoing)

### 6.1 Edge Deployment
**Reference**: `docs/research/TINYML_OPTIMIZATION.md`

**Targets:**
- [ ] Raspberry Pi deployment guide
- [ ] ESP32 deployment (TensorFlow Lite Micro)
- [ ] Mobile app (TFLite Android/iOS)

### 6.2 Web Dashboard
**Tasks:**
- [ ] Interactive dashboard enhancement
- [ ] Real-time data visualization
- [ ] Cloud deployment option

### 6.3 API Service
**Tasks:**
- [ ] FastAPI REST service
- [ ] Docker containerization
- [ ] Cloud deployment (GCP/AWS)

---

## Sprint Schedule

| Sprint | Focus | Duration |
|--------|-------|----------|
| Sprint 1 | BLS + Phase Folding | 2 weeks |
| Sprint 2 | False Positive + Model Training | 2 weeks |
| Sprint 3 | Gaia Integration + Multi-class | 2 weeks |
| Sprint 4 | Galaxy Classification | 2 weeks |
| Sprint 5 | Edge Deployment + API | 2 weeks |

---

## Skill Implementation Checklist

For each new skill, follow `docs/skills/SKILL_DEVELOPMENT.md`:

- [ ] YAML definition in `skills/`
- [ ] Python implementation in `src/skills/`
- [ ] Unit tests in `tests/`
- [ ] CLI integration in `larun.py`
- [ ] Documentation update
- [ ] TinyML optimization check

---

## Research Documentation Status

| Document | Status | Priority |
|----------|--------|----------|
| EXOPLANET_DETECTION.md | Complete | Reference |
| NASA_DATA_SOURCES.md | Complete | Reference |
| TINYML_OPTIMIZATION.md | Complete | Reference |
| SKILL_DEVELOPMENT.md | Complete | Reference |
| MAST_INTEGRATION.md | Complete | Reference |
| GALAXY_CLASSIFICATION.md | Complete | Reference |
| IMAGE_PROCESSING.md | Complete | Reference |
| STELLAR_PHYSICS.md | Complete | Reference |
| SPECTROSCOPY.md | Complete | Reference |
| GAIA_INTEGRATION.md | To Create | Sprint 3 |
| JWST_INTEGRATION.md | To Create | Sprint 3 |

---

## Next Immediate Actions

1. **Test current model** - Run `/detect` with real target
2. **Start BLS implementation** - High priority algorithm
3. **Improve training data** - Use Colab for larger dataset
4. **Create Gaia integration docs** - Needed for stellar parameters

---

*Created by: Padmanaban Veeraragavalu (Larun Engineering)*
*With AI assistance from Claude (Anthropic)*
*Last Updated: January 2026*

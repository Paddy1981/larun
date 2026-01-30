# LARUN TinyML - Skills Roadmap
## Specialized AI for Astronomical Data Analysis

```
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                          ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                          ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                          ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                          ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                          ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                          ║
║                                                                          ║
║     TinyML for Space Science                                             ║
║     Larun. × Astrodata                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 Vision

LARUN TinyML is a **terminal-based AI assistant** specialized for astronomical research. 
Like Claude Code but focused exclusively on space science - helping researchers analyze 
real NASA data, detect anomalies, and contribute to new discoveries.

---

## 📋 Skill Categories

### 🔴 TIER 1: Core Skills (Foundation)
*Essential capabilities for basic astronomical analysis*

| Skill ID | Skill Name | Description | Status |
|----------|------------|-------------|--------|
| `DATA-001` | **NASA Data Ingestion** | Fetch data from MAST, TESS, Kepler, Exoplanet Archive | ✅ Done |
| `DATA-002` | **FITS File Parser** | Read/write FITS files (standard astronomy format) | 🔄 Partial |
| `DATA-003` | **Light Curve Processing** | Normalize, detrend, clean light curves | ✅ Done |
| `MODEL-001` | **Spectral CNN** | 1D CNN for spectral classification | ✅ Done |
| `MODEL-002` | **TFLite Export** | Export models for edge deployment | ✅ Done |
| `DETECT-001` | **Transit Detection** | Identify planetary transit signals | ✅ Done |
| `DETECT-002` | **Anomaly Detection** | Flag unusual patterns in data | ✅ Done |
| `REPORT-001` | **NASA Report Generator** | Create submission-ready reports | ✅ Done |

---

### 🟡 TIER 2: Advanced Analysis Skills
*Deeper analysis capabilities for serious research*

| Skill ID | Skill Name | Description | Priority |
|----------|------------|-------------|----------|
| `ANAL-001` | **BLS Periodogram** | Box Least Squares for periodic transit search | High |
| `ANAL-002` | **Phase Folding** | Fold light curves on orbital period | High |
| `ANAL-003` | **Transit Fitting** | Fit transit models (Mandel-Agol) | High |
| `ANAL-004` | **Limb Darkening** | Calculate limb darkening coefficients | Medium |
| `ANAL-005` | **SNR Calculator** | Signal-to-noise ratio analysis | ✅ Done |
| `ANAL-006` | **False Positive Probability** | Estimate FPP for candidates | High |
| `ANAL-007` | **Centroid Analysis** | Check for background eclipsing binaries | Medium |
| `ANAL-008` | **Odd-Even Depth Test** | Verify transit consistency | Medium |

---

### 🟢 TIER 3: Stellar Characterization Skills
*Understand the host stars*

| Skill ID | Skill Name | Description | Priority |
|----------|------------|-------------|----------|
| `STAR-001` | **Stellar Classification** | Classify star types (OBAFGKM) | High |
| `STAR-002` | **Effective Temperature** | Estimate Teff from photometry | High |
| `STAR-003` | **Stellar Radius** | Calculate R★ from luminosity | High |
| `STAR-004` | **Stellar Mass** | Estimate M★ from spectral type | Medium |
| `STAR-005` | **Metallicity Estimation** | [Fe/H] from spectral features | Medium |
| `STAR-006` | **Age Estimation** | Gyrochronology, isochrone fitting | Low |
| `STAR-007` | **Activity Indicators** | Flare detection, rotation period | Medium |
| `STAR-008` | **Binary Detection** | Identify binary star systems | High |

---

### 🔵 TIER 4: Exoplanet Characterization Skills
*Characterize discovered planets*

| Skill ID | Skill Name | Description | Priority |
|----------|------------|-------------|----------|
| `PLANET-001` | **Radius Estimation** | Calculate Rp from transit depth | High |
| `PLANET-002` | **Orbital Period** | Determine precise orbital period | High |
| `PLANET-003` | **Semi-major Axis** | Calculate orbital distance (Kepler's 3rd) | High |
| `PLANET-004` | **Equilibrium Temperature** | Estimate Teq assuming albedo | Medium |
| `PLANET-005` | **Habitability Assessment** | Check if in habitable zone | High |
| `PLANET-006` | **Mass Estimation** | Mass-radius relationships | Medium |
| `PLANET-007` | **Density Calculation** | Bulk density from M and R | Medium |
| `PLANET-008` | **Composition Inference** | Rocky, gaseous, water world | Medium |
| `PLANET-009` | **TTV Analysis** | Transit Timing Variations for multi-planet | Low |
| `PLANET-010` | **Atmospheric Indicators** | Basic atmospheric assessment | Low |

---

### 🟣 TIER 5: Multi-Mission Integration Skills
*Work with multiple data sources*

| Skill ID | Skill Name | Description | Data Source |
|----------|------------|-------------|-------------|
| `MISSION-001` | **TESS Integration** | Full TESS data pipeline | MAST |
| `MISSION-002` | **Kepler Integration** | Kepler/K2 data pipeline | MAST |
| `MISSION-003` | **Gaia Integration** | Stellar parameters from Gaia | ESA |
| `MISSION-004` | **2MASS Integration** | Infrared photometry | IPAC |
| `MISSION-005` | **WISE Integration** | Mid-IR data for dust/disks | IPAC |
| `MISSION-006` | **Hubble Integration** | HST imaging and spectra | MAST |
| `MISSION-007` | **JWST Integration** | James Webb data access | MAST |
| `MISSION-008` | **Ground-Based Catalogs** | SDSS, Pan-STARRS, etc. | Various |
| `MISSION-009` | **Exoplanet Archive Sync** | Cross-reference discoveries | NASA |
| `MISSION-010` | **SIMBAD/VizieR Query** | Astronomical database queries | CDS |

---

### ⚫ TIER 6: Advanced Discovery Skills
*Cutting-edge capabilities for new discoveries*

| Skill ID | Skill Name | Description | Complexity |
|----------|------------|-------------|------------|
| `DISC-001` | **Multi-Planet Detection** | Find systems with multiple planets | High |
| `DISC-002` | **Circumbinary Detection** | Planets orbiting binary stars | High |
| `DISC-003` | **Ultra-Short Period** | Detect USP planets (P < 1 day) | Medium |
| `DISC-004` | **Long Period Detection** | Single-transit event analysis | High |
| `DISC-005` | **Exomoon Search** | Hunt for exomoons in transits | Very High |
| `DISC-006` | **Exoring Detection** | Ring systems around exoplanets | Very High |
| `DISC-007` | **Trojan Detection** | Co-orbital companions | Very High |
| `DISC-008` | **Disintegrating Planets** | Dust tails, evaporating worlds | High |
| `DISC-009` | **Young Planet Search** | Planets in star-forming regions | High |
| `DISC-010` | **Free-Floating Planets** | Rogue planets via microlensing | Very High |

---

### 🔶 TIER 7: Specialized Domain Skills
*Niche astronomical applications*

| Skill ID | Skill Name | Description | Domain |
|----------|------------|-------------|--------|
| `SPEC-001` | **Variable Star Classification** | Cepheids, RR Lyrae, eclipsing binaries | Stellar |
| `SPEC-002` | **Supernova Detection** | Early supernova identification | Transient |
| `SPEC-003` | **Asteroid Detection** | Moving object detection | Solar System |
| `SPEC-004` | **Comet Analysis** | Light curve analysis for comets | Solar System |
| `SPEC-005` | **AGN Variability** | Active galactic nuclei monitoring | Extragalactic |
| `SPEC-006` | **Gravitational Lensing** | Microlensing event detection | Cosmology |
| `SPEC-007` | **Stellar Flare Analysis** | Characterize stellar flares | Stellar |
| `SPEC-008` | **Pulsar Timing** | Pulsar period analysis | Compact Objects |
| `SPEC-009` | **Brown Dwarf Detection** | Cool substellar objects | Stellar |
| `SPEC-010` | **Debris Disk Analysis** | Circumstellar disk detection | Planetary |

---

### 🔷 TIER 8: Research & Publication Skills
*Help with the research process*

| Skill ID | Skill Name | Description | Output |
|----------|------------|-------------|--------|
| `RES-001` | **Literature Search** | Find relevant papers (ADS, arXiv) | References |
| `RES-002` | **Citation Generator** | Generate BibTeX citations | BibTeX |
| `RES-003` | **Figure Generator** | Publication-quality plots | PNG/PDF |
| `RES-004` | **Table Formatter** | LaTeX/ASCII tables | LaTeX |
| `RES-005` | **Abstract Writer** | Draft paper abstracts | Text |
| `RES-006` | **Method Description** | Write methods sections | Text |
| `RES-007` | **Comparison Analysis** | Compare with known objects | Report |
| `RES-008` | **Statistical Summary** | Generate statistics | JSON/CSV |
| `RES-009` | **Observation Planning** | Plan follow-up observations | Schedule |
| `RES-010` | **Proposal Assistant** | Help write telescope proposals | Text |

---

### 🔸 TIER 9: Collaboration & Community Skills
*Connect with the astronomy community*

| Skill ID | Skill Name | Description | Platform |
|----------|------------|-------------|----------|
| `COMM-001` | **ExoFOP Integration** | Submit/query candidate info | ExoFOP |
| `COMM-002` | **TESS Alert System** | Real-time TESS alerts | MAST |
| `COMM-003` | **TNS Reporter** | Transient Name Server submission | TNS |
| `COMM-004` | **Astronomer's Telegram** | ATel draft preparation | ATel |
| `COMM-005` | **Citizen Science Link** | Planet Hunters TESS integration | Zooniverse |
| `COMM-006` | **Observatory Scheduler** | Request telescope time | Various |
| `COMM-007` | **Collaboration Finder** | Match researchers by interest | Network |
| `COMM-008` | **Data Sharing** | Share datasets securely | Cloud |

---

## 🛠️ Implementation Priority Matrix

```
                    IMPACT
                    High │  ★ ANAL-001 (BLS)     ★ PLANET-005 (Habitability)
                         │  ★ STAR-001 (Class)   ★ DISC-001 (Multi-planet)
                         │  ★ MISSION-003 (Gaia) ★ RES-003 (Figures)
                         │
                    Med  │  ○ ANAL-006 (FPP)     ○ SPEC-001 (Variables)
                         │  ○ STAR-007 (Activity) ○ COMM-001 (ExoFOP)
                         │
                    Low  │  · DISC-005 (Exomoon) · PLANET-010 (Atmosphere)
                         │  · SPEC-008 (Pulsar)  · DISC-010 (FFP)
                         │
                         └──────────────────────────────────────────────
                              Low              Med              High
                                          FEASIBILITY
```

---

## 🚀 Suggested Implementation Phases

### Phase 1: Foundation (Current)
- [x] NASA data ingestion
- [x] Light curve processing
- [x] Basic transit detection
- [x] TinyML model training
- [x] Report generation

### Phase 2: Analysis Enhancement (Next)
- [ ] BLS periodogram implementation
- [ ] Phase folding capability
- [ ] Transit model fitting
- [ ] False positive probability
- [ ] Gaia cross-matching

### Phase 3: Stellar & Planetary
- [ ] Stellar classification
- [ ] Planet radius/period estimation
- [ ] Habitability zone calculation
- [ ] Multi-planet detection
- [ ] TTV analysis

### Phase 4: Multi-Mission
- [ ] Full TESS pipeline
- [ ] Kepler/K2 integration
- [ ] JWST data access
- [ ] Cross-catalog matching
- [ ] SIMBAD/VizieR queries

### Phase 5: Discovery Engine
- [ ] Automated candidate ranking
- [ ] Novel detection algorithms
- [ ] ML-based false positive rejection
- [ ] Real-time alert processing
- [ ] Publication assistance

---

## 💻 CLI Interface Design

```bash
# LARUN TinyML CLI Structure

larun --help

Commands:
  larun fetch      # Data fetching skills
  larun analyze    # Analysis skills
  larun detect     # Detection skills
  larun classify   # Classification skills
  larun report     # Reporting skills
  larun research   # Research assistance
  larun train      # Model training
  larun export     # Export results

Examples:
  larun fetch --source tess --target "TIC 12345678"
  larun analyze --type bls --input lightcurve.fits
  larun detect --mode transit --confidence 0.9
  larun classify --star --input spectrum.fits
  larun report --format nasa --output candidate_report.pdf
  larun research --search "hot jupiter" --limit 50
```

---

## 📊 Skill Dependency Graph

```
                    ┌─────────────────┐
                    │   DATA-001      │
                    │ NASA Ingestion  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │DATA-002 │   │DATA-003 │   │MISSION-*│
        │  FITS   │   │Light Crv│   │ Multi   │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             └──────┬──────┴──────┬──────┘
                    │             │
                    ▼             ▼
              ┌─────────┐   ┌─────────┐
              │DETECT-* │   │ STAR-*  │
              │Detection│   │ Stellar │
              └────┬────┘   └────┬────┘
                   │             │
                   └──────┬──────┘
                          │
                          ▼
                    ┌─────────┐
                    │PLANET-* │
                    │ Planets │
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
        ┌─────────┐┌─────────┐┌─────────┐
        │ DISC-*  ││ RES-*   ││ COMM-*  │
        │Discovery││Research ││Community│
        └─────────┘└─────────┘└─────────┘
```

---

## 🎓 Contribution to Space Community

### What LARUN TinyML Can Enable:

1. **Democratize Discovery**
   - Anyone with a laptop can analyze real NASA data
   - Citizen scientists can contribute to exoplanet discovery
   - Educational tool for astronomy students

2. **Accelerate Research**
   - Automate tedious data processing
   - Quick screening of thousands of candidates
   - Generate publication-ready outputs

3. **Edge Deployment**
   - Run analysis on small devices (Raspberry Pi, etc.)
   - Enable remote observatory automation
   - Reduce cloud computing costs

4. **New Discoveries**
   - Find planets missed by standard pipelines
   - Detect rare phenomena (exomoons, rings)
   - Cross-mission candidate validation

---

## 📝 License & Attribution

```
LARUN TinyML - Open Source Astronomy AI
Larun. × Astrodata

MIT License - Free for scientific research and education

Data Sources:
- NASA Exoplanet Archive
- MAST (Mikulski Archive for Space Telescopes)
- ESA Gaia Archive
- Various ground-based surveys

If you use LARUN TinyML in your research, please cite:
"LARUN TinyML: A TinyML Framework for Astronomical Data Analysis"
```

---

*Last Updated: {date}*
*Version: 1.0*
*Larun. × Astrodata*

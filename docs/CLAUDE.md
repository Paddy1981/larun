# LARUN - Claude Code Integration Guide

```
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                          ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                          ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                          ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                          ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                          ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                          ║
║                                                                          ║
║     TinyML for Space Science - Claude Code Integration                   ║
║     Larun. × Astrodata                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## Overview

LARUN is a specialized AI assistant for astronomical data analysis, designed to work like Claude Code but focused exclusively on space science. This document guides Claude Code on how to extend, integrate, and develop LARUN capabilities.

## Project Structure

```
larun/
├── CLAUDE.md                    # This file - Claude Code instructions
├── research/                    # Research documentation
│   ├── NASA_DATA_SOURCES.md     # NASA APIs and data access
│   ├── EXOPLANET_DETECTION.md   # Transit detection methods
│   ├── GALAXY_CLASSIFICATION.md # Galaxy morphology ML
│   ├── TINYML_OPTIMIZATION.md   # Edge deployment strategies
│   ├── IMAGE_PROCESSING.md      # Astronomical image analysis
│   └── STELLAR_PHYSICS.md       # Stellar classification science
├── skills/                      # Skill definitions
│   ├── skills.yaml              # Core skill definitions
│   ├── image_skills.yaml        # Image analysis skills
│   └── SKILL_DEVELOPMENT.md     # How to create new skills
├── integrations/                # Integration guides
│   ├── MAST_INTEGRATION.md      # MAST archive integration
│   ├── GAIA_INTEGRATION.md      # Gaia DR3 integration
│   └── JWST_INTEGRATION.md      # JWST data integration
├── src/                         # Source code
├── models/                      # Trained models
├── data/                        # Data cache
└── output/                      # Generated outputs
```

## Claude Code Instructions

### When Working on LARUN:

1. **Always check research docs first** - Before implementing any astronomy feature, read the relevant research .md file
2. **Follow skill patterns** - New capabilities should follow the skill definition format in skills/
3. **Use NASA APIs correctly** - Refer to NASA_DATA_SOURCES.md for proper API usage
4. **Maintain TinyML focus** - Keep models small (<100KB) for edge deployment
5. **Test with real data** - Always validate against actual NASA data

### Priority Development Areas:

| Priority | Area | Research Doc |
|----------|------|--------------|
| 🔴 High | BLS Periodogram | EXOPLANET_DETECTION.md |
| 🔴 High | Galaxy CNN Training | GALAXY_CLASSIFICATION.md |
| 🔴 High | Gaia Integration | GAIA_INTEGRATION.md |
| 🟡 Medium | JWST Data Access | JWST_INTEGRATION.md |
| 🟡 Medium | Multi-planet Detection | EXOPLANET_DETECTION.md |
| 🟢 Lower | Exomoon Search | EXOPLANET_DETECTION.md |

### Code Style Guidelines:

```python
# LARUN Code Style
# - Use type hints
# - Include docstrings with examples
# - Follow astronomy naming conventions
# - Keep functions focused and small
# - Cache expensive computations

def detect_transit(
    flux: np.ndarray,
    time: np.ndarray,
    min_depth: float = 0.0001,
    min_snr: float = 7.0
) -> List[TransitCandidate]:
    """
    Detect planetary transits in light curve data.
    
    Args:
        flux: Normalized flux values
        time: Time array (BJD)
        min_depth: Minimum transit depth (default: 100 ppm)
        min_snr: Minimum signal-to-noise ratio
        
    Returns:
        List of TransitCandidate objects
        
    Example:
        >>> candidates = detect_transit(flux, time, min_depth=0.001)
        >>> for c in candidates:
        ...     print(f"Period: {c.period:.2f} days, Depth: {c.depth:.4f}")
    """
    pass
```

### Testing Requirements:

1. **Unit tests** for all new functions
2. **Integration tests** with real NASA data samples
3. **Model accuracy benchmarks** against published results
4. **Edge deployment tests** on resource-constrained environments

## Key APIs and Libraries

### Python Dependencies:
```
lightkurve>=2.0       # TESS/Kepler data access
astroquery>=0.4       # NASA archive queries
astropy>=5.0          # Astronomical computations
tensorflow>=2.10      # ML models
numpy>=1.21           # Numerical operations
scipy>=1.9            # Signal processing
photutils>=1.5        # Photometry
```

### NASA APIs:
- **MAST**: https://mast.stsci.edu/api/v0/
- **Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/TAP/
- **Gaia**: https://gea.esac.esa.int/archive/

## Skill Development Workflow

```
1. Research → Read relevant .md docs
2. Design → Define skill in YAML format
3. Implement → Write Python code
4. Test → Validate with real data
5. Optimize → Reduce model size for TinyML
6. Document → Update research docs
7. Integrate → Add to CLI and dashboard
```

## Model Constraints (TinyML)

| Constraint | Value | Reason |
|------------|-------|--------|
| Max Model Size | 100 KB | Microcontroller deployment |
| Max Parameters | 100,000 | Memory limits |
| Quantization | INT8 | Speed + size |
| Input Size | 1024 pts | Fixed processing |
| Inference Time | <10 ms | Real-time analysis |

## Research Documentation Index

| Document | Purpose |
|----------|---------|
| [NASA_DATA_SOURCES.md](research/NASA_DATA_SOURCES.md) | How to access NASA data |
| [EXOPLANET_DETECTION.md](research/EXOPLANET_DETECTION.md) | Transit detection science |
| [GALAXY_CLASSIFICATION.md](research/GALAXY_CLASSIFICATION.md) | Galaxy morphology ML |
| [TINYML_OPTIMIZATION.md](research/TINYML_OPTIMIZATION.md) | Edge deployment |
| [IMAGE_PROCESSING.md](research/IMAGE_PROCESSING.md) | Astronomical imaging |
| [STELLAR_PHYSICS.md](research/STELLAR_PHYSICS.md) | Star classification |

## Contact & Attribution

**Project**: LARUN - TinyML for Space Science
**Brand**: Larun. × Astrodata
**License**: MIT
**Repository**: https://github.com/Paddy1981/larun

---

*This document is designed to be read by Claude Code to understand how to develop and extend LARUN capabilities.*

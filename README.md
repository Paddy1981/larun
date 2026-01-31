# LARUN - Democratizing Space Discovery

<p align="center">
  <strong>LARUN</strong> × <strong>Astrodata</strong>
</p>

<p align="center">
  <em>"Making the Universe Accessible, Fun, and Full of Opportunity"</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-BSL--1.1-green?style=flat-square" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/accuracy-81.8%25-brightgreen?style=flat-square" alt="Accuracy"></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/NASA-MAST-red?style=flat-square&logo=nasa&logoColor=white" alt="NASA MAST"></a>
  <a href="#"><img src="https://img.shields.io/badge/ESA-Gaia_DR3-blue?style=flat-square" alt="Gaia DR3"></a>
  <a href="#"><img src="https://img.shields.io/badge/TESS-Kepler-purple?style=flat-square" alt="TESS/Kepler"></a>
</p>

```
╔══════════════════════════════════════════════════════════════════════════╗
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                          ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                          ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                          ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                          ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║                          ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                          ║
║                                                                          ║
║     TinyML for Space Science • Source Available • For Everyone           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Our Mission

> **To democratize space science by making professional-grade astronomical tools accessible to everyone — from world-class researchers to curious students, passionate amateur astronomers, and everyday people who look up at the stars and wonder.**

We believe:
- **Space discovery shouldn't be locked behind institutional walls**
- **Anyone with curiosity should be able to explore the cosmos**
- **Technology should create opportunities, not barriers**
- **Learning about space should be fun, not intimidating**
- **Real discoveries can come from anywhere**

---

## Who Is This For?

| Audience | What You Can Do |
|----------|-----------------|
| **Researchers** | Publication-ready algorithms, reproducible pipelines, direct NASA data integration |
| **Students** | Learn with real data, earn certifications, build your portfolio |
| **Amateur Astronomers** | Discover real exoplanets, join community campaigns, contribute to science |
| **Everyone** | "My First Exoplanet" experience, no expertise required |

---

## Features

| Category | Capabilities |
|----------|--------------|
| **Interactive CLI** | Claude-like terminal interface for astronomical analysis |
| **Chat Interface** | Natural language conversational AI for astronomy |
| **TinyML Model** | Lightweight neural network (<100KB) runs on Raspberry Pi |
| **NASA Data Pipeline** | Direct access to MAST, TESS, Kepler archives |
| **Transit Detection** | Automated exoplanet transit signal identification (81.8% accuracy) |
| **Vetting Suite** | False positive identification (odd-even, secondary eclipse, V-shape) |
| **Analysis Tools** | BLS periodogram, phase folding, transit fitting, TTV analysis |
| **Reporting** | NASA-compatible reports in standard formats |
| **Extensible** | 24+ skills, developer addons, code generation |

---

## Quick Start

### One-Command Install

**Windows PowerShell:**
```powershell
irm https://raw.githubusercontent.com/Paddy1981/larun/main/install.ps1 | iex
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Paddy1981/larun/main/install.sh | bash
```

**Using pip:**
```bash
pip install larun
```

### Start Exploring

```bash
# Interactive CLI Mode
larun

# Or Chat Mode (natural language)
larun-chat

# Or from source
python larun.py
python larun_chat.py
```

### Essential Commands

```
/help           - Show help
/skills         - List all 24+ available skills
/detect         - Run exoplanet detection on data
/fetch <target> - Fetch NASA data (e.g., /fetch TIC 307210830)
/fit            - Fit transit parameters
/vet            - Check for false positives
/phase          - Phase fold light curve
/ttv            - Transit Timing Variations analysis
```

---

## Documentation

### Core Documents

| Document | Description |
|----------|-------------|
| [**VISION.md**](VISION.md) | Our mission, ecosystem roadmap, and how we're democratizing space |
| [**DEVELOPMENT_PLAN.md**](DEVELOPMENT_PLAN.md) | Technical roadmap, sprint planning, feature status |
| [**INSTALL.md**](INSTALL.md) | Detailed installation guide for all platforms |
| [**SKILLS_ROADMAP.md**](SKILLS_ROADMAP.md) | Complete skills system documentation |
| [**COLAB_TRAINING.md**](COLAB_TRAINING.md) | Train models with Google Colab free GPU |

### Research & Reference

| Document | Description |
|----------|-------------|
| [docs/research/EXOPLANET_DETECTION.md](docs/research/EXOPLANET_DETECTION.md) | BLS, transit fitting, vetting algorithms |
| [docs/research/TINYML_OPTIMIZATION.md](docs/research/TINYML_OPTIMIZATION.md) | Model optimization for edge devices |
| [docs/research/NASA_DATA_SOURCES.md](docs/research/NASA_DATA_SOURCES.md) | MAST, TESS, Kepler data access |
| [docs/research/GALAXY_CLASSIFICATION.md](docs/research/GALAXY_CLASSIFICATION.md) | Galaxy morphology classification |
| [docs/research/STELLAR_PHYSICS.md](docs/research/STELLAR_PHYSICS.md) | Stellar parameters and physics |
| [docs/research/SPECTROSCOPY.md](docs/research/SPECTROSCOPY.md) | Spectral analysis methods |
| [docs/skills/SKILL_DEVELOPMENT.md](docs/skills/SKILL_DEVELOPMENT.md) | How to create new skills |

### Architecture

| Document | Description |
|----------|-------------|
| [docs/FEDERATED_ARCHITECTURE.md](docs/FEDERATED_ARCHITECTURE.md) | Distributed/federated learning design |
| [docs/CLAUDE.md](docs/CLAUDE.md) | AI assistant integration documentation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LARUN TinyML SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
│  │  NASA Data    │────►│ Preprocessing │────►│   TinyML      │             │
│  │  Pipeline     │     │ & Calibration │     │   Detector    │             │
│  └───────────────┘     └───────────────┘     └───────────────┘             │
│         │                     │                      │                      │
│         ▼                     ▼                      ▼                      │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
│  │ MAST/TESS/    │     │  Calibration  │     │  Detection    │             │
│  │ Kepler/Gaia   │     │  Database     │     │  Results      │             │
│  └───────────────┘     └───────────────┘     └───────────────┘             │
│                                                      │                      │
│                              ┌───────────────────────┼───────────────────┐  │
│                              ▼                       ▼                   ▼  │
│                       ┌──────────┐           ┌──────────┐        ┌────────┐│
│                       │ Vetting  │           │  Transit │        │ Report ││
│                       │ Suite    │           │  Fitting │        │ Gen    ││
│                       └──────────┘           └──────────┘        └────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
larun/
├── larun.py              # Main interactive CLI
├── larun_chat.py         # Chat interface (natural language)
├── run_pipeline.py       # Automated pipeline
├── train_real_data.py    # NASA data training
│
├── src/
│   ├── pipeline/         # NASA data ingestion
│   ├── model/            # TinyML model definition
│   ├── calibration/      # Auto-calibration system
│   ├── detector/         # Transit/anomaly detection
│   ├── reporter/         # NASA report generation
│   └── skills/           # Analysis algorithms (vetting, fitting, TTV)
│
├── skills/               # YAML skill definitions
├── addons/               # Developer addons (codegen)
├── models/               # Saved models (H5, TFLite)
├── data/                 # Training and cached data
├── docs/                 # Research documentation
├── notebooks/            # Jupyter/Colab notebooks
└── tests/                # Test suite
```

---

## Model Specifications

| Specification | Value |
|---------------|-------|
| **Input** | 1D light curve (128-2048 points) |
| **Model Size** | ~50KB (TFLite), ~25KB (INT8 quantized) |
| **Accuracy** | 81.8% (binary transit detection) |
| **Inference Time** | <10ms on edge devices |
| **Target Platform** | Raspberry Pi, ESP32, mobile |

---

## Skills System

LARUN includes 24+ skills organized by capability:

| Tier | Category | Examples |
|------|----------|----------|
| 1 | **Core** | Data ingestion, light curve processing, CNN detection |
| 2 | **Analysis** | BLS periodogram, phase folding, transit fitting |
| 3 | **Vetting** | Odd-even test, secondary eclipse, V-shape detection |
| 4 | **Advanced** | TTV analysis, multi-planet search, exomoon detection |
| 5 | **Stellar** | Classification, temperature, radius estimation |
| 6 | **Multi-Mission** | TESS, Kepler, Gaia, JWST integration |
| 7 | **Research** | Literature search, figure generation, reporting |

---

## Data Sources

| Source | Data Type | Integration |
|--------|-----------|-------------|
| NASA Exoplanet Archive | Confirmed exoplanets | TAP/REST API |
| MAST (STScI) | TESS light curves | `lightkurve` |
| MAST (STScI) | Kepler light curves | `lightkurve` |
| Gaia DR3 | Stellar parameters | `astroquery` |
| JWST | Spectral data | Planned |

---

## Join the Ecosystem

### For Contributors
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### For Educators
Use Larun in your classroom! See our curriculum in [VISION.md](VISION.md#education--learning).

### For Citizen Scientists
Join discovery campaigns and earn recognition! Community features coming soon.

---

## Current Status

| Component | Status |
|-----------|--------|
| Core CLI | Complete |
| Chat Interface | Complete |
| Transit Detection | 81.8% accuracy |
| Vetting Suite | Complete |
| Transit Fitting | Complete |
| TTV Analysis | Complete |
| BLS Periodogram | In Progress |
| Multi-class Model | Planned |
| Web Dashboard | Planned |
| Mobile App | Planned |

---

## License & Intellectual Property

### Copyright

**© 2024-2026 Padmanaban Veeraragavalu (Larun Engineering). All Rights Reserved.**

### Software License

The LARUN software is released under the **Business Source License 1.1 (BSL-1.1)**:

| Use Case | License |
|----------|---------|
| Academic Research | Free |
| Personal/Educational | Free |
| Non-Commercial | Free |
| Commercial/Production | Paid License Required |

After 4 years, each version converts to Apache 2.0.

### Research & Publications

If you use LARUN in your research, please cite:
```
Veeraragavalu, P. (2026). LARUN: TinyML-Powered Astronomical Data Analysis
for Democratized Exoplanet Discovery. Larun Engineering.
https://github.com/Paddy1981/larun
```

### Commercial Licensing

For commercial use, enterprise licensing, white-labeling, or partnership inquiries, contact **Larun Engineering**.

### Core Tools Promise

Core detection algorithms and CLI tools will **always remain free for research and education**.

---

## Acknowledgments

- **NASA** - For open data policies that make citizen science possible
- **lightkurve** team - For excellent data access library
- **TensorFlow Lite** team - For edge ML tools
- **Anthropic** - AI assistance in development (Claude)
- **Google DeepMind** - AI assistance in development (Gemini)

---

<p align="center">
  <strong>LARUN TinyML</strong> - <em>Democratizing astronomical discovery through edge AI</em>
</p>

<p align="center">
  Created by <strong>Padmanaban Veeraragavalu</strong> (Larun Engineering)<br/>
  With AI assistance from Claude (Anthropic) and Gemini (Google DeepMind)
</p>

<p align="center">
  <em>"The stars belong to everyone. Now, so does the science."</em>
</p>

# LARUN TinyML - Astronomical Data Analysis System

<p align="center">
  <strong>LARUN</strong> × <strong>Astrodata</strong>
</p>

<p align="center">
  <em>TinyML-powered astronomical data processing for exoplanet discovery</em>
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
║     TinyML for Space Science                                             ║
║     LARUN × Astrodata                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Created By

**Padmanaban Veeraragavalu** (Larun Engineering)

*With AI assistance from Claude (Anthropic)*

---

## Features

- **Interactive CLI**: Claude-like terminal interface for astronomical analysis
- **Chat Interface**: Natural language conversational AI for astronomy
- **Skills System**: 24+ specialized skills for data analysis, detection, and reporting
- **TinyML Model**: Lightweight neural network (<100KB) for edge deployment
- **NASA Data Pipeline**: Direct ingestion from MAST, TESS, Kepler archives
- **Transit Detection**: Automated exoplanet transit signal identification
- **Report Generator**: NASA-compatible reports in standard formats
- **Developer Addons**: Extensible code generation tools for researchers

## Quick Start

### Interactive Mode (Recommended)
```bash
# Start the LARUN CLI
python larun.py

# Or start Chat Mode (natural language)
python larun_chat.py
```

### Available Commands
```
/help           - Show help
/skills         - List available skills
/skill <ID>     - Show skill details
/run <ID>       - Execute a skill
/train          - Train the TinyML model
/detect         - Run detection on data
/fetch <target> - Fetch NASA data for a target
/addon          - Load developer addons
/generate       - Generate Python scripts (requires codegen addon)
```

### Train with Real NASA Data
```bash
python train_real_data.py --planets 100 --non-planets 100 --epochs 100
```

### Run Complete Pipeline
```bash
python run_pipeline.py --num-stars 50 --epochs 100
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LARUN TinyML System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   NASA Data  │───►│  Preprocessor│───►│   TinyML     │      │
│  │   Pipeline   │    │  & Calibrator│    │   Detector   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ MAST/TESS/   │    │ Calibration  │    │  Detection   │      │
│  │ Kepler Data  │    │ Database     │    │  Results     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │               │
│                                                 ▼               │
│                                          ┌──────────────┐      │
│                                          │ NASA Report  │      │
│                                          │ Generator    │      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
larun/
├── larun.py              # Main interactive CLI
├── larun_chat.py         # Chat interface (natural language)
├── run_pipeline.py       # Automated pipeline
├── train_real_data.py    # NASA data training
├── src/
│   ├── pipeline/         # NASA data ingestion
│   ├── model/            # TinyML model definition
│   ├── calibration/      # Auto-calibration system
│   ├── detector/         # Transit/anomaly detection
│   └── reporter/         # NASA report generation
├── skills/
│   ├── skills.yaml       # Core skills definition
│   └── addons/           # Developer addon skills
├── addons/
│   └── codegen.py        # Code generation addon
├── data/                 # Training and processed data
├── models/               # Saved models (H5, TFLite)
└── output/               # Generated reports
```

## Skills System

LARUN includes 24+ skills organized in tiers:

| Tier | Category | Skills |
|------|----------|--------|
| 1 | Core | Data Ingestion, Light Curve Processing, CNN Model, Transit Detection |
| 2 | Analysis | BLS Periodogram, Phase Folding, Transit Fitting |
| 3 | Stellar | Classification, Temperature, Radius estimation |
| 4 | Planet | Radius, Period, Habitability assessment |
| 5 | Multi-Mission | TESS, Kepler, Gaia integration |
| 6 | Discovery | Multi-planet, Exomoon search |
| 7 | Research | Literature search, Figure generation |

## Developer Addons

Load developer addons for advanced code generation:

```bash
# In larun CLI:
/addon codegen
/generate script transit_search
/generate model cnn_1d
/generate pipeline tess
```

## Data Sources

| Source | Data Type | API |
|--------|-----------|-----|
| NASA Exoplanet Archive | Confirmed exoplanets | TAP/REST |
| MAST (STScI) | TESS light curves | lightkurve |
| MAST (STScI) | Kepler light curves | lightkurve |
| Gaia DR3 | Stellar parameters | astroquery |

## Model Specifications

- **Input**: 1D light curve data (128-2048 points)
- **Model Size**: < 100KB (TinyML optimized)
- **Inference Time**: < 10ms on edge devices
- **Output Classes**: noise, stellar, transit, binary, artifact, unknown

## Installation

```bash
# Clone the repository
git clone https://github.com/Paddy1981/larun.git
cd larun

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start LARUN
python larun.py
```

## License

MIT License - Open for scientific research and education

---

**LARUN TinyML** - *Democratizing astronomical discovery through edge AI*

Created by **Padmanaban Veeraragavalu** (Larun Engineering)

With AI assistance from Claude (Anthropic)

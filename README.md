# 🔭 AstroTinyML - Spectral Data Analysis System

<p align="center">
  <strong>Larun.</strong> × <strong>Astrodata</strong>
</p>

<p align="center">
  <em>TinyML-powered astronomical data processing for NASA-compatible spectral analysis</em>
</p>

---

A TinyML-based spectral data analysis system designed for astronomical data processing, compatible with NASA data formats and reporting standards.

**Developed by Larun. in collaboration with Astrodata.**

## ✨ Features

- **TinyML Model**: Lightweight neural network optimized for edge deployment (<100KB)
- **NASA Data Pipeline**: Ingest FITS files and spectral data from NASA archives (MAST, TESS, Kepler)
- **Auto-Calibration**: Self-calibrating based on confirmed exoplanet discoveries
- **Report Generator**: NASA-compatible reports in standard formats (PDF, JSON, FITS, CSV)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AstroTinyML System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   NASA Data  │───►│  Preprocessor│───►│   TinyML     │      │
│  │   Pipeline   │    │  & Calibrator│    │   Detector   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ FITS/CSV     │    │ Calibration  │    │  Detection   │      │
│  │ Ingestion    │    │ Database     │    │  Results     │      │
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
astro-tinyml/
├── src/
│   ├── pipeline/           # NASA data ingestion
│   ├── model/              # TinyML model definition
│   ├── calibration/        # Auto-calibration system
│   ├── detector/           # Spectral anomaly detection
│   └── reporter/           # NASA report generation
├── data/
│   ├── raw/                # Raw NASA data
│   ├── processed/          # Preprocessed spectral data
│   └── calibration/        # Calibration reference data
├── models/
│   ├── tflite/             # TensorFlow Lite models
│   └── checkpoints/        # Training checkpoints
├── reports/                # Generated NASA reports
├── tests/                  # Unit tests
└── config/                 # Configuration files
```

## Quick Start

### Option 1: Demo with Synthetic Data (Quick Test)
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no internet needed)
python standalone_demo.py
```

### Option 2: Train with REAL NASA Data ⭐
```bash
# Install dependencies
pip install -r requirements.txt

# Train on real NASA data (fetches from MAST & Exoplanet Archive)
python train_real_data.py --planets 100 --non-planets 100 --epochs 100
```

This will:
1. Fetch confirmed exoplanet host stars from NASA Exoplanet Archive
2. Download their light curves from TESS/Kepler via MAST
3. Fetch non-planet stars as negative examples
4. Train the TinyML model on real astronomical data
5. Export models in multiple formats (Keras, TFLite, INT8 quantized)

### Option 3: Full Pipeline
```bash
# Run calibration with known discoveries
python main.py --mode calibrate

# Process new data and detect anomalies
python main.py --mode detect --input data/raw/ --output reports/

# Generate NASA report
python main.py --mode report --format pdf --submit-ready
```

## 🛰️ Real Data Sources

The `train_real_data.py` script fetches from:

| Source | Data Type | API |
|--------|-----------|-----|
| NASA Exoplanet Archive | Confirmed exoplanets (labels) | TAP/REST API |
| MAST (STScI) | TESS light curves | lightkurve |
| MAST (STScI) | Kepler light curves | lightkurve |

### Training Data Composition
- **Positive samples**: Light curves from confirmed exoplanet host stars
- **Negative samples**: Light curves from stars without known planets
- **Augmented**: Simulated eclipsing binaries and instrument artifacts

## Supported NASA Data Sources

- **MAST Archive**: Hubble, Kepler, TESS data
- **Exoplanet Archive**: Confirmed exoplanet data for calibration
- **IRSA**: Infrared spectral data
- **HEASARC**: High-energy astrophysics data

## Model Specifications

- **Input**: 1D spectral data (128-2048 wavelength bins)
- **Model Size**: < 100KB (TinyML optimized)
- **Inference Time**: < 10ms on Cortex-M4
- **Accuracy**: 94.2% on validation set

## License

MIT License - Open for scientific research and education

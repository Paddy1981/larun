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

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample NASA data
python -m src.pipeline.downloader --source mast --target kepler

# Run calibration with known discoveries
python -m src.calibration.calibrator --reference data/calibration/known_exoplanets.csv

# Process new data and detect anomalies
python -m src.detector.run --input data/raw/ --output reports/

# Generate NASA report
python -m src.reporter.generate --format pdf --submit-ready
```

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

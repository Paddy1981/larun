# GitHub Copilot Instructions for LARUN

## Project Overview

LARUN (larun.space) is a TinyML-powered astronomical analysis platform that uses federated machine learning models for exoplanet detection, stellar classification, and other astronomical phenomena analysis.

### Key Architecture: Vinveli-Vinoli-Vidhai Framework

- **Vinveli** (The Cosmos): Data sources - NASA archives, MAST, Kepler, TESS
- **Vinoli** (Speed of Light): Processing layer - TinyML models, feature extraction
- **Vidhai** (Seeds): Output - Classifications, detections, analysis results

---

## Project Structure

```
larun/
├── src/                    # Core source code
│   ├── model/              # TinyML models and training
│   │   ├── specialized_models.py   # 8 specialized classifiers
│   │   ├── data_generators.py      # Synthetic data generation
│   │   └── pipeline_framework.py   # Model orchestration
│   └── ...
├── models/trained/         # Trained model weights (.npz files)
├── web/                    # Next.js web application
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── data/                   # Training data and caches
```

---

## TinyML Models

The platform includes 9 specialized models, all <100KB for edge deployment:

| Model | Purpose | Accuracy |
|-------|---------|----------|
| EXOPLANET-001 | Transit detection | 100% |
| VSTAR-001 | Variable star classification | 99.8% |
| FLARE-001 | Stellar flare detection | 96.7% |
| MICROLENS-001 | Microlensing events | 99.4% |
| SUPERNOVA-001 | Transient detection | 100% |
| GALAXY-001 | Galaxy morphology | 99.9% |
| SPECTYPE-001 | Spectral classification | 95.0% |
| ASTERO-001 | Asteroseismology | 99.8% |
| MULTIVIEW-EXOPLANET | Multi-view detection | 74.2% AUC |

---

## Code Conventions

### Python
- Pure NumPy implementations (no TensorFlow dependency for inference)
- Type hints for all function signatures
- Docstrings for public methods
- Feature-based classifiers with normalization parameters saved in weights

### Model Files
- Weights saved as `.npz` files with normalization stats
- Include `_norm_mean`, `_norm_std`, `_class_labels` in saved models
- Use `SimpleClassifier.load()` for self-contained model loading

### Training Scripts
- `train_simplified.py` - Synthetic data training
- `train_with_nasa_params.py` - Real NASA parameters
- `train_real_exoplanets.py` - Real light curve training

---

## Key Documentation

For detailed information, refer to these documents:

### Architecture & Framework
- [Framework Architecture](../docs/FRAMEWORK_ARCHITECTURE.md) - Vinveli-Vinoli-Vidhai design
- [Federated Architecture](../docs/FEDERATED_ARCHITECTURE.md) - Model federation approach

### Research & Science
- [Exoplanet Detection](../docs/research/EXOPLANET_DETECTION.md) - Transit detection methods
- [NASA Data Sources](../docs/research/NASA_DATA_SOURCES.md) - Data acquisition
- [Galaxy Classification](../docs/research/GALAXY_CLASSIFICATION.md) - Morphology classification
- [TinyML Optimization](../docs/research/TINYML_OPTIMIZATION.md) - Edge deployment

### Development
- [Skill Development](../docs/skills/SKILL_DEVELOPMENT.md) - Adding new capabilities
- [Claude Integration](../docs/CLAUDE.md) - AI assistant configuration
- [Training Guide](../docs/TRAINING_GUIDE.md) - Model training procedures
- [Quickstart](../docs/QUICKSTART.md) - Getting started

### Integrations
- [MAST Integration](../docs/integrations/MAST_INTEGRATION.md) - Mikulski Archive
- [Gaia Integration](../docs/integrations/GAIA_INTEGRATION.md) - ESA Gaia data

---

## Common Tasks

### Training a Model
```python
from train_simplified import SimpleClassifier, train_classifier

model = SimpleClassifier(n_features=45, n_classes=3)
# ... train ...
model.set_normalization(mean, std)
model.set_class_labels(["class_a", "class_b", "class_c"])
model.save("models/trained/MODEL_weights.npz")
```

### Loading and Using a Model
```python
from train_simplified import SimpleClassifier, extract_lightcurve_features

model = SimpleClassifier.load("models/trained/EXOPLANET-001_weights.npz")
features = extract_lightcurve_features(lightcurve).reshape(1, -1)
pred, conf, probs = model.predict_from_raw(features)
print(f"Prediction: {model.class_labels[pred[0]]}")
```

### Feature Extraction
Light curves are converted to 45 statistical features:
- Global statistics (mean, std, min, max, range, median, percentiles)
- Peak characteristics (location, amplitude)
- Derivatives (1st and 2nd order statistics)
- Segment statistics (4 segments × 3 stats)
- Autocorrelation (lags 1, 5, 10)
- Shape statistics (skewness, kurtosis, smoothness)

---

## Testing

```bash
# Run model tests
python -m pytest tests/

# Validate trained models
python scripts/validate_models.py

# Test inference
python -c "from train_simplified import SimpleClassifier; m = SimpleClassifier.load('models/trained/EXOPLANET-001_weights.npz'); print(m.class_labels)"
```

---

## Dependencies

### Core (Required)
- numpy

### Training (Optional)
- pandas
- astroquery (for NASA Exoplanet Archive)
- astropy

### Real Data (Optional)
- lightkurve (for MAST/Kepler/TESS downloads)

---

## Notes for Copilot

1. **Model Architecture**: All models use feature-based MLP classifiers, not CNNs
2. **Normalization**: Always include normalization parameters when saving models
3. **Edge Deployment**: Keep models <100KB, use pure NumPy
4. **Data Sources**: Prefer NASA Exoplanet Archive for real parameters
5. **Testing**: Verify per-class accuracy, not just overall accuracy

# LARUN Model Training Guide

This guide covers how to train the LARUN TinyML model for exoplanet transit detection.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | Optional | NVIDIA with CUDA |
| Storage | 5 GB | 10 GB |

### Software Requirements

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt

# For GPU training (optional)
pip install tensorflow[cuda]
```

---

## Quick Start

### Train with Default Settings

```bash
python train_real_data.py --planets 100 --non-planets 100 --epochs 50
```

### Train with Augmentation

```bash
python train_real_data.py --planets 200 --non-planets 200 --epochs 100 --augment
```

---

## Training Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `train_real_data.py` | Train on NASA exoplanet data | Primary training |
| `train_specialized.py` | K-fold cross-validation | Model validation |
| `train_with_augmentation.py` | Heavy augmentation | Limited data |
| `train_with_class_weights.py` | Handle class imbalance | Imbalanced datasets |

---

## Data Pipeline

### Automatic Data Fetching

The training script automatically fetches data from NASA archives:

1. **NASA Exoplanet Archive** - Confirmed planet parameters
2. **MAST/TESS** - Light curve data via `lightkurve`
3. **MAST/Kepler** - Kepler mission light curves

### Data Structure

```
data/
├── expanded/
│   ├── all_exoplanets.csv    # Planet catalog
│   └── cache/                 # Cached light curves
│       ├── Kepler-11_TESS.npy
│       ├── TOI-700_TESS.npy
│       └── ...
├── raw/                       # Raw downloads
└── processed/                 # Processed training data
```

### Sample Requirements

| Dataset | Minimum | Target | Notes |
|---------|---------|--------|-------|
| Transit (positive) | 50 | 250+ | Confirmed exoplanets |
| Non-transit (negative) | 50 | 250+ | Variable stars, noise |
| Validation | 20% | 20% | Automatic split |

---

## Training Configuration

### Command Line Options

```bash
python train_real_data.py [OPTIONS]

Options:
  --planets N          Number of planet samples (default: 100)
  --non-planets N      Number of non-planet samples (default: 100)
  --epochs N           Training epochs (default: 50)
  --batch-size N       Batch size (default: 32)
  --learning-rate F    Learning rate (default: 0.001)
  --augment            Enable data augmentation
  --kfold N            Use K-fold cross-validation
  --output DIR         Output directory (default: models/real/)
```

### Configuration File

Edit `config/config.yaml`:

```yaml
model:
  name: "astro_tinyml"
  version: "2.0.0"
  input_size: 1024
  num_classes: 6

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping: true
  patience: 10

augmentation:
  enabled: true
  noise_level: 0.01
  time_shift_max: 0.1
  flux_scale_range: [0.9, 1.1]
```

---

## Model Architecture

### SpectralCNN

```
Input (1024,) -> Conv1D -> BatchNorm -> MaxPool ->
Conv1D -> BatchNorm -> MaxPool ->
Conv1D -> BatchNorm -> GlobalAvgPool ->
Dense -> Dropout -> Dense(6) -> Softmax
```

### Output Classes

| Class | ID | Description |
|-------|:--:|-------------|
| noise | 0 | Instrumental noise |
| stellar_signal | 1 | Stellar variability |
| planetary_transit | 2 | Planet transit signal |
| eclipsing_binary | 3 | Eclipsing binary star |
| instrument_artifact | 4 | Instrument artifacts |
| unknown_anomaly | 5 | Unknown signals |

---

## Training Procedure

### Step 1: Fetch Training Data

```bash
# Fetch 500+ samples (recommended)
python train_real_data.py --planets 250 --non-planets 250 --epochs 0
```

This downloads and caches light curves without training.

### Step 2: Train Model

```bash
# Full training run
python train_real_data.py \
    --planets 250 \
    --non-planets 250 \
    --epochs 100 \
    --augment \
    --output models/real/
```

### Step 3: Validate Model

```bash
# K-fold cross-validation
python train_specialized.py --kfold 5
```

### Step 4: Export for Deployment

```bash
# Model is automatically exported to:
# - models/real/astro_tinyml_real.h5       (Keras format)
# - models/real/astro_tinyml_real.tflite   (TensorFlow Lite)
# - models/real/astro_tinyml_real_int8.tflite  (INT8 quantized)
```

---

## Augmentation Techniques

| Technique | Purpose | Config |
|-----------|---------|--------|
| Gaussian Noise | Simulate instrumental noise | `noise_level: 0.01` |
| Time Shift | Transit timing variations | `time_shift_max: 0.1` |
| Flux Scaling | Depth variations | `scale_range: [0.9, 1.1]` |
| Random Dropout | Missing data points | `dropout_rate: 0.05` |
| Transit Depth | Vary planet sizes | `depth_factor: [0.8, 1.2]` |
| Mixup | Blend samples | `alpha: 0.2` |

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/

# Open browser to http://localhost:6006
```

### Training Output

```
Epoch 1/100
250/250 [==============================] - 5s 20ms/step
  loss: 0.6932 - accuracy: 0.5234 - val_loss: 0.6821 - val_accuracy: 0.5567

Epoch 50/100
250/250 [==============================] - 4s 16ms/step
  loss: 0.2145 - accuracy: 0.8734 - val_loss: 0.2456 - val_accuracy: 0.8512

Training complete!
  Final accuracy: 87.34%
  Validation accuracy: 85.12%
  Model saved to: models/real/astro_tinyml_real.h5
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `lightkurve` timeout | Network issues | Retry or use cached data |
| GPU out of memory | Batch too large | Reduce `--batch-size` |
| Low accuracy | Insufficient data | Increase sample count |
| Overfitting | Model too complex | Enable dropout, reduce epochs |

### Data Fetching Errors

```bash
# If NASA API is slow, use cached data:
python train_real_data.py --use-cache

# Check data cache:
ls data/expanded/cache/
```

---

## Performance Targets

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Accuracy | 80% | 90% | 95%+ |
| Precision | 75% | 85% | 90%+ |
| Recall | 75% | 85% | 90%+ |
| F1 Score | 75% | 85% | 90%+ |
| Model Size | <100KB | <50KB | <25KB |

---

## Google Colab Training

For free GPU training, use our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Paddy1981/larun/blob/main/notebooks/train_colab.ipynb)

See [COLAB_TRAINING.md](../COLAB_TRAINING.md) for details.

---

## Next Steps

After training:

1. **Evaluate** - Run `python -m pytest tests/` to verify
2. **Benchmark** - Test on held-out data
3. **Deploy** - See [DEPLOYMENT.md](DEPLOYMENT.md) for edge deployment
4. **Use** - See [QUICKSTART.md](QUICKSTART.md) for inference

---

## References

- [TinyML Optimization](research/TINYML_OPTIMIZATION.md)
- [Exoplanet Detection Methods](research/EXOPLANET_DETECTION.md)
- [NASA Data Sources](research/NASA_DATA_SOURCES.md)

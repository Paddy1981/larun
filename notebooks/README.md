# LARUN Cloud Training Notebooks

Train the LARUN exoplanet detection model using **FREE GPU** resources.

## Available Platforms

| Platform | Notebook | Free GPU | Hours/Week | Best For |
|----------|----------|----------|------------|----------|
| **Kaggle** | `larun_kaggle_training.ipynb` | T4 x2 | 30 hrs | Most free time |
| **Colab** | `larun_colab_training.ipynb` | T4 | ~12 hrs | Easy access |
| **Lightning.ai** | `larun_lightning_training.ipynb` | T4 | 22 hrs/mo | Persistent storage |
| **Paperspace** | `larun_paperspace_training.ipynb` | M4000 | 6 hr sessions | Quick experiments |

## Quick Start

### Option 1: Kaggle (Recommended - 30 hrs/week FREE)

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. **New Notebook** → **File** → **Import Notebook**
3. Upload `larun_kaggle_training.ipynb`
4. **Settings** (right sidebar) → **Accelerator** → **GPU T4 x2**
5. **Run All**

### Option 2: Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File** → **Upload notebook**
3. Upload `larun_colab_training.ipynb`
4. **Runtime** → **Change runtime type** → **T4 GPU**
5. **Runtime** → **Run all**

### Option 3: Lightning.ai (22 hrs/month)

1. Go to [lightning.ai](https://lightning.ai)
2. **Studios** → **New Studio** → Select **GPU**
3. Upload `larun_lightning_training.ipynb`
4. Run cells

### Option 4: Paperspace Gradient

1. Go to [console.paperspace.com/gradient](https://console.paperspace.com/gradient)
2. **Notebooks** → **Create** → **Free GPU**
3. Upload `larun_paperspace_training.ipynb`
4. Run cells

## Training Configuration

Each notebook has configurable parameters at the top:

```python
NUM_PLANETS = 150        # More = better accuracy (but slower)
NUM_NON_PLANETS = 150    # Balance with planets
EPOCHS = 100             # Training iterations
BATCH_SIZE = 64          # Increase with GPU memory
```

**Recommended settings by platform:**

| Platform | Planets | Non-Planets | Batch Size |
|----------|---------|-------------|------------|
| Kaggle (T4 x2) | 200 | 200 | 128 |
| Colab (T4) | 150 | 150 | 64 |
| Lightning (T4) | 150 | 150 | 64 |
| Paperspace (M4000) | 100 | 100 | 32 |

## Output Files

After training completes, you'll get:

```
larun_trained.zip
├── larun_model.h5           # Keras model
├── larun_model.tflite       # TFLite for mobile
├── larun_model_int8.tflite  # Quantized for edge
├── training_data.npz        # Training data
├── training_history.png     # Accuracy/loss plots
└── training_metadata.json   # Training config & results
```

## Using Trained Model

1. Download `larun_trained.zip` from the platform
2. Extract to your LARUN installation:
   ```bash
   unzip larun_trained.zip -d models/cloud/
   ```
3. Test with LARUN CLI:
   ```bash
   python larun.py
   > /classify TOI-700 --model models/cloud/larun_model.h5
   ```

## Troubleshooting

### "No GPU available"
- Check platform settings for GPU acceleration
- Kaggle: Settings → Accelerator → GPU
- Colab: Runtime → Change runtime type → GPU

### "Out of memory"
- Reduce `BATCH_SIZE`
- Reduce `NUM_PLANETS` and `NUM_NON_PLANETS`

### "NASA archive timeout"
- Data fetching can be slow; be patient
- Cached data persists on Kaggle/Lightning/Paperspace

---
*Larun Engineering - Astrodata*

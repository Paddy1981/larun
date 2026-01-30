# LARUN TinyML - Google Colab Training Guide

## Quick Start (One-Click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Paddy1981/larun/blob/main/notebooks/larun_colab_training.ipynb)

Click the badge above to open the training notebook directly in Google Colab!

---

## Manual Setup

### Step 1: Open Google Colab
Go to: https://colab.research.google.com

### Step 2: Open from GitHub
1. Click `File` → `Open notebook`
2. Select `GitHub` tab
3. Enter: `Paddy1981/larun`
4. Select: `notebooks/larun_colab_training.ipynb`

### Step 3: Enable GPU
1. Click `Runtime` → `Change runtime type`
2. Select `T4 GPU` (free tier)
3. Click `Save`

### Step 4: Run Training
1. Click `Runtime` → `Run all`
2. Wait for training to complete (~10-15 minutes)
3. Download the trained model when prompted

---

## What the Notebook Does

```
Phase 1: Install Dependencies        (~1 min)
Phase 2: Fetch Planet Host Data      (~3 min) - 8 parallel workers
Phase 3: Fetch Non-Planet Data       (~2 min) - 8 parallel workers
Phase 4: Prepare Training Data       (~30 sec)
Phase 5: Build & Train Model         (~5 min) - GPU accelerated
Phase 6: Export Models               (~30 sec)
         ↓
         larun_trained.zip (download)
```

---

## Configuration Options

Edit these variables in the notebook to customize training:

```python
NUM_PLANETS = 100        # More = better accuracy, slower
NUM_NON_PLANETS = 100    # Should match NUM_PLANETS
EPOCHS = 100             # More = better fit, risk of overfitting
BATCH_SIZE = 32          # Larger = faster on GPU
INPUT_SIZE = 1024        # Light curve length
```

---

## Expected Results

| Metric | Typical Value |
|--------|---------------|
| Training Time | 10-15 minutes |
| Validation Accuracy | 85-95% |
| Model Size (H5) | ~500 KB |
| Model Size (TFLite) | ~150 KB |
| Model Size (Quantized) | ~80 KB |

---

## Using the Trained Model

After downloading `larun_trained.zip`, extract and copy to your LARUN installation:

```bash
# Extract
unzip larun_trained.zip

# Copy to LARUN
cp larun_trained/larun_model.h5 ~/larun/models/real/astro_tinyml.h5
cp larun_trained/training_data.npz ~/larun/data/real/

# Test with LARUN CLI
cd ~/larun
python larun.py
# Then type: /status
```

---

## Troubleshooting

### "GPU not available"
- Go to `Runtime` → `Change runtime type` → Select `T4 GPU`
- If T4 unavailable, try again later (free tier limits)

### "lightkurve installation failed"
- Run this cell first:
  ```python
  !pip install --upgrade pip
  !pip install lightkurve astroquery
  ```

### "Connection timeout fetching data"
- NASA servers may be slow; re-run the cell
- Consider reducing `NUM_PLANETS` to 50

### "Out of memory"
- Reduce `BATCH_SIZE` to 16
- Reduce `NUM_PLANETS` to 50

---

## Alternative: Kaggle Notebooks

Kaggle offers 30 hours/week of free GPU time:

1. Go to: https://www.kaggle.com/code
2. Create new notebook
3. Upload `larun_colab_training.ipynb`
4. Settings → Accelerator → `GPU P100`
5. Run all cells

---

**Created by: Padmanaban Veeraragavalu (Larun Engineering)**

*With AI assistance from Claude (Anthropic)*

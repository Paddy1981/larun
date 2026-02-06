# EXOPLANET-001 Professional Validation Guide

This document describes how to validate EXOPLANET-001 against the Kepler DR25 benchmark
and document the results for professional/research use.

## Overview

Professional-grade model validation requires:

1. **Large-scale testing** on thousands of real samples
2. **Gold-standard dataset** (Kepler DR25 is the industry standard)
3. **Multiple metrics** (accuracy, precision, recall, F1)
4. **Comparison to baselines** (BLS, other published algorithms)
5. **Reproducible methodology** (documented and scriptable)

## Quick Start

```bash
# Basic validation (500 planets + 500 false positives)
python validate_kepler_dr25.py

# Full validation (all available samples)
python validate_kepler_dr25.py --n-planets 4000 --n-fps 8000

# With light curve download (slower but more accurate)
python validate_kepler_dr25.py --download-lc --n-planets 200 --n-fps 200

# Compare against BLS baseline
python validate_kepler_dr25.py --download-lc --compare-bls
```

## The Kepler DR25 Benchmark

### What is DR25?

Data Release 25 (DR25) is the final Kepler mission data release, containing:

- **4,034 planet candidates** (KOIs with disposition CANDIDATE/CONFIRMED)
- **8,054 false positives** (KOIs with disposition FALSE POSITIVE)
- **197,096 target stars** observed over 4 years

This is the gold standard for exoplanet detection benchmarking.

### Why DR25?

| Reason | Description |
|--------|-------------|
| Completeness | Final, most thoroughly vetted Kepler catalog |
| Ground truth | Dispositions verified by multiple methods |
| Published baselines | Many papers report DR25 results for comparison |
| Real data | Actual space telescope observations, not simulations |

## Validation Methodology

### Step 1: Download the DR25 Catalog

```python
from validate_kepler_dr25 import download_kepler_dr25_catalog

catalog = download_kepler_dr25_catalog()
print(f"Planets: {len(catalog['planets'])}")
print(f"False positives: {len(catalog['false_positives'])}")
```

The catalog is downloaded from the NASA Exoplanet Archive and cached locally.

### Step 2: Feature Extraction

For each KOI (Kepler Object of Interest), we extract 45 features:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Light curve statistics | 14 | std, mean, percentiles, autocorrelation |
| Transit parameters | 5 | depth, period, duration, ratios |
| Stellar parameters | 5 | Teff, radius, luminosity proxies |
| Derived features | 16 | log transforms, combinations |
| Padding | 5 | Reserved for future use |

### Step 3: Model Inference

```python
from train_simplified import SimpleClassifier

model = SimpleClassifier.load("models/trained/EXOPLANET-001_real_weights.npz")

# Predict with automatic normalization
pred_class, confidence = model.predict_from_raw(features)
```

### Step 4: Metric Calculation

We calculate standard classification metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: Fraction of transit predictions that are correct
- **Recall**: Fraction of actual transits that are detected
- **F1 Score**: Harmonic mean of precision and recall

## Output Files

After running validation, you'll find:

```
data/kepler_dr25/
├── dr25_catalog.json          # Cached DR25 catalog
├── validation_results.json    # Detailed metrics and predictions
├── validation_report.md       # Human-readable report
└── lightcurves/               # Cached light curves (if --download-lc)
```

## Interpreting Results

### Accuracy Thresholds

| Accuracy | Assessment |
|----------|------------|
| ≥98% | Exceptional - exceeds most published methods |
| ≥95% | Production-ready - suitable for automated pipelines |
| ≥90% | Strong - good for research with human review |
| ≥85% | Moderate - needs improvement for production |
| <85% | Weak - significant retraining required |

### Comparison to Published Methods

| Algorithm | Accuracy | Reference |
|-----------|----------|-----------|
| Kepler Robovetter | ~95% | Thompson et al. 2018 |
| astronet | ~96% | Shallue & Vanderburg 2018 |
| exoplanet-ml | ~98% | Yu et al. 2019 |
| AstroNet-Triage | ~97% | Ansdell et al. 2018 |
| Random Forest | ~93% | McCauliff et al. 2015 |

To claim "professional-level" performance, EXOPLANET-001 should achieve ≥95% accuracy
on the DR25 benchmark, comparable to the Kepler Robovetter.

## Documenting Results

### For Papers/Publications

Include in your methods section:

```
We validated EXOPLANET-001 against the Kepler DR25 cumulative KOI table
(Thompson et al. 2018), using [N] confirmed planet candidates and [M]
vetted false positives. The model achieved [X]% accuracy, [Y]% precision,
and [Z]% recall, comparable to the Kepler Robovetter (Thompson et al. 2018)
and astronet (Shallue & Vanderburg 2018).
```

### For Website/Marketing

Use factual claims:

✅ Good: "98% accuracy on Kepler DR25 benchmark (N=1,000 samples)"
✅ Good: "Validated on real NASA Kepler mission data"
✅ Good: "Performance comparable to published methods"

❌ Avoid: "Best exoplanet detector ever" (unless you have comparative evidence)
❌ Avoid: "100% accurate" (no model is perfect)

### For Code Documentation

```python
"""
EXOPLANET-001: TinyML Exoplanet Transit Classifier

Performance (Kepler DR25 Benchmark):
- Accuracy: XX.X%
- Precision: XX.X%
- Recall: XX.X%
- F1 Score: XX.X%
- Validated on: YYYY-MM-DD
- Dataset: N confirmed planets, M false positives

Comparison:
- Kepler Robovetter: ~95% accuracy
- astronet: ~96% accuracy
"""
```

## Continuous Validation

For production use, set up automated validation:

```yaml
# .github/workflows/validate.yml
name: Model Validation

on:
  push:
    paths:
      - 'models/trained/*.npz'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install numpy lightkurve
      - name: Run validation
        run: python validate_kepler_dr25.py --n-planets 500 --n-fps 500
      - name: Check accuracy threshold
        run: |
          accuracy=$(python -c "import json; print(json.load(open('data/kepler_dr25/validation_results.json'))['metrics']['accuracy'])")
          if (( $(echo "$accuracy < 90" | bc -l) )); then
            echo "Accuracy below threshold: $accuracy%"
            exit 1
          fi
```

## Advanced Validation

### Cross-Validation

For more robust estimates, use k-fold cross-validation:

```bash
python validate_kepler_dr25.py --cross-validate --k-folds 5
```

### Stratified Sampling

Ensure balanced sampling across planet types:

```bash
python validate_kepler_dr25.py --stratified --by-period
```

### Bootstrap Confidence Intervals

Calculate confidence intervals on metrics:

```bash
python validate_kepler_dr25.py --bootstrap --n-iterations 1000
```

## Troubleshooting

### "lightkurve not installed"

```bash
pip install lightkurve
```

### "NASA Exoplanet Archive timeout"

The archive may be slow. Increase timeout or use cached catalog:

```python
catalog = download_kepler_dr25_catalog(timeout=120)
```

### "Model not found"

Ensure you've trained the model first:

```bash
python train_real_exoplanets.py --n-planets 200 --n-fps 300
```

## References

1. Thompson, S. E., et al. (2018). "Planetary Candidates Observed by Kepler. VIII."
   ApJS, 235, 38.

2. Shallue, C. J., & Vanderburg, A. (2018). "Identifying Exoplanets with Deep Learning."
   AJ, 155, 94.

3. Yu, L., et al. (2019). "Identifying Exoplanets with Deep Learning. III."
   AJ, 158, 25.

4. Ansdell, M., et al. (2018). "Scientific Domain Knowledge Improves Exoplanet Transit
   Classification with Deep Learning." ApJL, 869, L7.

---

*LARUN AstroTinyML - Professional Exoplanet Detection*

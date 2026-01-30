# Galaxy Classification - Research Documentation

## Overview

This document covers machine learning approaches for galaxy morphology classification, with focus on CNN-based methods suitable for LARUN's TinyML implementation.

---

## 1. Galaxy Morphology Fundamentals

### Hubble Sequence (Tuning Fork Diagram)

```
                    Ellipticals
                E0 → E3 → E5 → E7
                         ↓
                        S0 (Lenticular)
                      ↙     ↘
                   Sa        SBa
                   ↓          ↓
                   Sb        SBb
                   ↓          ↓
                   Sc        SBc
                   ↓          ↓
                   Sd        SBd
                  ↓            ↓
              Irregulars    Irregulars
```

### Morphological Classes

| Class | Type | Features | Fraction |
|-------|------|----------|----------|
| E | Elliptical | Smooth, no structure | ~15% |
| S0 | Lenticular | Disk, no arms | ~20% |
| Sa-Sd | Spiral | Arms, varying tightness | ~50% |
| SBa-SBd | Barred Spiral | Central bar + arms | ~25% of spirals |
| Irr | Irregular | No regular structure | ~15% |

### Key Morphological Features

**Ellipticals (E0-E7):**
- Smooth light distribution
- Old stellar population
- No gas/dust
- E-number = 10 × (1 - b/a) where b/a is axis ratio

**Spirals (Sa-Sd):**
- Central bulge + disk + arms
- Sa: tight arms, large bulge
- Sd: loose arms, small bulge
- Blue star-forming regions

**Lenticulars (S0):**
- Disk without spiral arms
- Intermediate between E and S

**Barred (SB):**
- Linear bar through center
- ~60% of disk galaxies have bars

**Irregulars:**
- No regular structure
- Often interacting/disturbed

---

## 2. Galaxy Zoo Classification System

### Primary Decision Tree

```
Q1: Is the galaxy smooth and rounded?
    ├── Smooth → Q7 (How rounded?)
    ├── Features or disk → Q2
    └── Star or artifact → End

Q2: Could this be an edge-on disk?
    ├── Yes → Q9 (Edge-on properties)
    └── No → Q3

Q3: Is there a bar?
    ├── Bar → Q4
    └── No bar → Q4

Q4: Is there a spiral pattern?
    ├── Spiral → Q10 (Arm tightness)
    └── No spiral → Q5

Q5: How prominent is the bulge?
    ├── Dominant / Obvious / Just noticeable / No bulge

Q6: Is there anything odd?
    ├── Ring / Lens / Disturbed / Irregular / Other / Merger
```

### Galaxy Zoo Vote Fractions

For machine learning, we can use vote fractions as soft labels:

```python
# Example Galaxy Zoo 2 vote structure
vote_fractions = {
    'smooth': 0.15,
    'features': 0.80,
    'artifact': 0.05,
    
    # If features:
    'edge_on': 0.10,
    'not_edge_on': 0.90,
    
    # If not edge-on:
    'bar': 0.35,
    'no_bar': 0.65,
    
    # Spiral:
    'spiral': 0.70,
    'no_spiral': 0.30,
    
    # Arm tightness:
    'tight': 0.20,
    'medium': 0.50,
    'loose': 0.30,
    
    # Bulge:
    'dominant': 0.10,
    'obvious': 0.40,
    'just_noticeable': 0.35,
    'no_bulge': 0.15
}
```

---

## 3. CNN Architecture for Galaxy Classification

### Standard Architecture (Reference)
```python
import tensorflow as tf

def create_galaxy_classifier(input_shape=(128, 128, 3), num_classes=6):
    """
    Standard galaxy morphology classifier.
    
    Classes: elliptical, lenticular, spiral, barred_spiral, irregular, merger
    """
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### TinyML Architecture (LARUN Target)
```python
def create_galaxy_classifier_tiny(input_shape=(64, 64, 3), num_classes=6):
    """
    Lightweight galaxy classifier for edge deployment.
    Target: <100KB model size, <10ms inference
    """
    model = tf.keras.Sequential([
        # Depthwise separable convolutions for efficiency
        tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        
        tf.keras.layers.SeparableConv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        
        tf.keras.layers.SeparableConv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### Model Comparison

| Architecture | Parameters | Size (FP32) | Size (INT8) | Accuracy |
|--------------|------------|-------------|-------------|----------|
| Full CNN | ~2M | 8 MB | 2 MB | 95% |
| MobileNet | ~3.4M | 14 MB | 3.5 MB | 93% |
| TinyML | ~50K | 200 KB | 50 KB | 88% |
| Minimal | ~10K | 40 KB | 10 KB | 82% |

---

## 4. Data Augmentation for Galaxy Images

### Rotation Equivariance
Galaxies have no preferred orientation, so rotation augmentation is essential:

```python
import tensorflow as tf
import numpy as np

class GalaxyAugmentation:
    """Augmentation pipeline for galaxy images."""
    
    def __init__(self):
        self.augmentations = [
            self.random_rotation,
            self.random_flip,
            self.random_zoom,
            self.random_brightness,
            self.add_noise
        ]
    
    def random_rotation(self, image):
        """Random rotation (any angle - galaxies are rotationally invariant)."""
        angle = tf.random.uniform([], 0, 360)
        return tfa.image.rotate(image, angle * np.pi / 180)
    
    def random_flip(self, image):
        """Random horizontal and vertical flip."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image
    
    def random_zoom(self, image, zoom_range=(0.8, 1.2)):
        """Random zoom to handle varying galaxy sizes."""
        scale = tf.random.uniform([], zoom_range[0], zoom_range[1])
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        return tf.image.resize_with_crop_or_pad(image, height, width)
    
    def random_brightness(self, image, max_delta=0.2):
        """Random brightness adjustment."""
        return tf.image.random_brightness(image, max_delta)
    
    def add_noise(self, image, noise_level=0.05):
        """Add Gaussian noise to simulate varying image quality."""
        noise = tf.random.normal(tf.shape(image), stddev=noise_level)
        return image + noise
    
    def augment(self, image):
        """Apply random augmentations."""
        for aug in self.augmentations:
            if tf.random.uniform([]) > 0.5:
                image = aug(image)
        return image
```

### Special Considerations

```python
def preprocess_galaxy_image(image, target_size=(128, 128)):
    """
    Preprocess galaxy image for classification.
    
    Steps:
    1. Background subtraction
    2. Centering on galaxy
    3. Normalization
    4. Resize
    """
    # 1. Estimate and subtract background
    background = np.median(image)
    image = image - background
    
    # 2. Find galaxy center (brightest region)
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter(image, sigma=5)
    cy, cx = ndimage.center_of_mass(smoothed)
    
    # 3. Crop around center
    half_size = min(image.shape) // 4
    y1 = max(0, int(cy - half_size))
    y2 = min(image.shape[0], int(cy + half_size))
    x1 = max(0, int(cx - half_size))
    x2 = min(image.shape[1], int(cx + half_size))
    cropped = image[y1:y2, x1:x2]
    
    # 4. Resize
    from PIL import Image
    resized = np.array(Image.fromarray(cropped).resize(target_size))
    
    # 5. Normalize to [0, 1]
    normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    
    return normalized
```

---

## 5. Morphological Parameters

### Non-Parametric Measurements

```python
def calculate_morphology_params(image, segmap):
    """
    Calculate CAS (Concentration-Asymmetry-Smoothness) parameters.
    
    These are classic non-parametric morphology measures.
    """
    # Mask galaxy
    galaxy = image * (segmap > 0)
    
    # Petrosian radius
    r_petro = calculate_petrosian_radius(image, segmap)
    
    # --- CONCENTRATION ---
    # C = 5 * log10(r_80 / r_20)
    r_20 = calculate_radius_fraction(image, segmap, 0.2)
    r_80 = calculate_radius_fraction(image, segmap, 0.8)
    concentration = 5 * np.log10(r_80 / r_20)
    
    # --- ASYMMETRY ---
    # A = sum(|I - I_180|) / sum(|I|) - A_background
    rotated = np.rot90(np.rot90(galaxy))
    residual = np.abs(galaxy - rotated)
    asymmetry = np.sum(residual) / (2 * np.sum(np.abs(galaxy)))
    
    # --- SMOOTHNESS (CLUMPINESS) ---
    # S = sum(|I - I_smooth|) / sum(|I|)
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter(galaxy, sigma=r_petro / 4)
    residual = np.abs(galaxy - smoothed)
    smoothness = np.sum(residual) / np.sum(np.abs(galaxy))
    
    return {
        'concentration': concentration,
        'asymmetry': asymmetry,
        'smoothness': smoothness
    }


def calculate_gini_m20(image, segmap):
    """
    Calculate Gini and M20 parameters.
    
    Gini: inequality of pixel flux distribution
    M20: second-order moment of brightest 20%
    """
    # Flatten galaxy pixels
    galaxy_pixels = image[segmap > 0].flatten()
    galaxy_pixels = np.sort(galaxy_pixels)
    n = len(galaxy_pixels)
    
    # --- GINI ---
    # G = (1/(mean*n*(n-1))) * sum_i((2i - n - 1) * x_i)
    indices = np.arange(1, n + 1)
    gini = np.sum((2 * indices - n - 1) * galaxy_pixels) / (np.mean(galaxy_pixels) * n * (n - 1))
    
    # --- M20 ---
    # M20 = log10(sum_i(M_i) / M_total) for brightest 20%
    # where M_i = f_i * ((x_i - x_c)^2 + (y_i - y_c)^2)
    y_coords, x_coords = np.where(segmap > 0)
    flux = image[segmap > 0]
    
    # Centroid
    total_flux = np.sum(flux)
    x_c = np.sum(x_coords * flux) / total_flux
    y_c = np.sum(y_coords * flux) / total_flux
    
    # Second moment
    M = flux * ((x_coords - x_c)**2 + (y_coords - y_c)**2)
    M_total = np.sum(M)
    
    # Sort by flux and get brightest 20%
    sort_idx = np.argsort(flux)[::-1]
    cumsum_flux = np.cumsum(flux[sort_idx])
    brightest_20_idx = sort_idx[cumsum_flux < 0.2 * total_flux]
    
    M_20 = np.log10(np.sum(M[brightest_20_idx]) / M_total)
    
    return {
        'gini': gini,
        'm20': M_20
    }
```

### Morphology Classification from Parameters

```python
def classify_from_parameters(concentration, asymmetry, gini, m20):
    """
    Simple classification based on morphological parameters.
    """
    # Decision boundaries (approximate)
    if concentration > 4.0 and asymmetry < 0.1:
        return "elliptical"
    elif concentration > 3.5 and asymmetry < 0.2:
        return "lenticular"
    elif asymmetry > 0.35:
        return "merger"
    elif gini > 0.55 and m20 < -2.0:
        return "spiral"
    elif gini < 0.4:
        return "irregular"
    else:
        return "spiral"  # Default
```

---

## 6. Sersic Profile Fitting

### Theory
The Sérsic profile describes galaxy surface brightness:

```
I(r) = I_e × exp(-b_n × [(r/r_e)^(1/n) - 1])

Where:
- I_e = intensity at effective radius
- r_e = effective (half-light) radius
- n = Sérsic index
- b_n ≈ 2n - 1/3 (for n > 0.5)
```

### Sérsic Index Interpretation

| n | Profile Type | Galaxy Type |
|---|--------------|-------------|
| 0.5-1 | Gaussian | Dwarf |
| 1 | Exponential | Disk |
| 2-3 | Intermediate | Bulge + Disk |
| 4 | de Vaucouleurs | Elliptical |
| >4 | Steep core | cD galaxy |

### Implementation
```python
from scipy.optimize import curve_fit

def sersic_profile(r, I_e, r_e, n):
    """1D Sersic profile."""
    b_n = 2 * n - 1/3 + 4/(405*n)  # Approximation
    return I_e * np.exp(-b_n * ((r / r_e)**(1/n) - 1))

def fit_sersic(image, center):
    """
    Fit 1D Sersic profile to galaxy.
    """
    # Create radial profile
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Bin by radius
    max_r = min(center[0], center[1], image.shape[0]-center[1], image.shape[1]-center[0])
    radii = np.arange(1, max_r)
    profile = np.array([np.mean(image[(r >= rad-0.5) & (r < rad+0.5)]) for rad in radii])
    
    # Fit
    try:
        popt, pcov = curve_fit(
            sersic_profile, 
            radii, 
            profile,
            p0=[profile[0], max_r/4, 2],
            bounds=([0, 1, 0.5], [np.inf, max_r, 10])
        )
        return {
            'I_e': popt[0],
            'r_e': popt[1],
            'n': popt[2],
            'success': True
        }
    except:
        return {'success': False}
```

---

## 7. Transfer Learning for Galaxy Classification

### Using Pre-trained Models
```python
def create_galaxy_classifier_transfer(num_classes=6):
    """
    Galaxy classifier using transfer learning from ImageNet.
    """
    # Load pre-trained base
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Freeze base layers
    base_model.trainable = False
    
    # Add classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def fine_tune_galaxy_classifier(model, train_data, val_data):
    """
    Two-stage training: head then full model.
    """
    # Stage 1: Train head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(train_data, validation_data=val_data, epochs=10)
    
    # Stage 2: Fine-tune entire model
    model.layers[0].trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(train_data, validation_data=val_data, epochs=20)
    
    return model
```

---

## 8. Training Data Sources

### Galaxy Zoo

```python
def load_galaxy_zoo_data():
    """
    Load Galaxy Zoo 2 data.
    
    Data: https://data.galaxyzoo.org/
    """
    import pandas as pd
    
    # Load labels
    labels = pd.read_csv('GalaxyZoo2_classifications.csv')
    
    # Key columns:
    # - 't01_smooth_or_features_a01_smooth_fraction'
    # - 't01_smooth_or_features_a02_features_or_disk_fraction'
    # - 't02_edgeon_a04_yes_fraction'
    # - 't03_bar_a06_bar_fraction'
    # - 't04_spiral_a08_spiral_fraction'
    
    return labels
```

### SDSS Cutouts

```python
def download_sdss_cutout(ra, dec, size=128, scale=0.396):
    """
    Download SDSS image cutout.
    
    Args:
        ra, dec: Coordinates (degrees)
        size: Image size (pixels)
        scale: Pixel scale (arcsec/pixel)
    """
    import urllib.request
    
    url = f"https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg?" \
          f"ra={ra}&dec={dec}&width={size}&height={size}&scale={scale}"
    
    urllib.request.urlretrieve(url, f"galaxy_{ra}_{dec}.jpg")
```

### DECaLS

```python
def download_decals_cutout(ra, dec, size=256, pixscale=0.262):
    """
    Download DECaLS image cutout.
    """
    import urllib.request
    
    url = f"https://www.legacysurvey.org/viewer/cutout.jpg?" \
          f"ra={ra}&dec={dec}&layer=ls-dr9&pixscale={pixscale}&size={size}"
    
    urllib.request.urlretrieve(url, f"decals_{ra}_{dec}.jpg")
```

---

## 9. Evaluation Metrics

```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_galaxy_classifier(model, test_data, class_names):
    """
    Comprehensive evaluation of galaxy classifier.
    """
    # Predictions
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_data.labels  # Adjust based on your data format
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Galaxy Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('galaxy_confusion_matrix.png')
    
    # Classification report
    report = classification_report(y_true, y_pred_classes, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score']
    }
```

---

## 10. TinyML Optimization

### Quantization
```python
def quantize_galaxy_model(model, representative_data):
    """
    Quantize model to INT8 for edge deployment.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for calibration
    def representative_dataset():
        for image in representative_data:
            yield [image[np.newaxis, ...].astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    quantized_model = converter.convert()
    
    return quantized_model
```

### Model Pruning
```python
import tensorflow_model_optimization as tfmot

def prune_galaxy_model(model, target_sparsity=0.5):
    """
    Prune model to reduce size.
    """
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=target_sparsity,
        begin_step=0,
        end_step=1000
    )
    
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )
    
    return pruned_model
```

---

## References

1. Dieleman, S., et al. (2015). "Rotation-invariant convolutional neural networks for galaxy morphology prediction." MNRAS, 450, 1441.
2. Willett, K. W., et al. (2013). "Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey." MNRAS, 435, 2835.
3. Conselice, C. J. (2014). "The Evolution of Galaxy Structure Over Cosmic Time." ARA&A, 52, 291.
4. Lotz, J. M., et al. (2004). "A New Nonparametric Approach to Galaxy Morphological Classification." AJ, 128, 163.
5. Peng, C. Y., et al. (2010). "Detailed Decomposition of Galaxy Images. II. Beyond Axisymmetric Models." AJ, 139, 2097.

---

*Last Updated: 2024*
*LARUN - Larun. × Astrodata*

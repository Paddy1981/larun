#!/usr/bin/env python3
"""
AstroTinyML - Real NASA Data Training Pipeline
===============================================
Train the spectral classifier using real data from:
- NASA Exoplanet Archive (confirmed exoplanets for labels)
- MAST Archive (TESS/Kepler light curves)

Larun. × Astrodata
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required packages
def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import lightkurve
    except ImportError:
        missing.append('lightkurve')
    
    try:
        import astroquery
    except ImportError:
        missing.append('astroquery')
    
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    
    if missing:
        print("\n" + "="*60)
        print("MISSING DEPENDENCIES")
        print("="*60)
        print(f"\nPlease install the following packages:\n")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        print("="*60 + "\n")
        sys.exit(1)

check_dependencies()

import pandas as pd
import lightkurve as lk
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from src.model.spectral_cnn import SpectralCNN
from src.augmentation import LightCurveAugmenter


class RealDataFetcher:
    """Fetch real astronomical data from NASA archives."""
    
    def __init__(self, data_dir: str = "data/real"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.data_dir / "exoplanet_cache.json"
        
    def fetch_confirmed_exoplanets(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch confirmed exoplanets from NASA Exoplanet Archive.
        These will be our POSITIVE examples (planetary transits).
        """
        logger.info(f"Fetching confirmed exoplanets from NASA Archive (limit: {limit})...")
        
        try:
            # Query confirmed planets with TESS or Kepler data
            planets = NasaExoplanetArchive.query_criteria(
                table="pscomppars",  # Planetary Systems Composite Parameters
                select="pl_name,hostname,disc_facility,pl_orbper,pl_rade,pl_bmasse,sy_vmag,ra,dec",
                where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'",
                order="pl_name"
            )
            
            df = planets.to_pandas()
            logger.info(f"Found {len(df)} confirmed exoplanets")
            
            # Filter to reasonable limits and save
            df = df.head(limit)
            df.to_csv(self.data_dir / "confirmed_exoplanets.csv", index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching exoplanets: {e}")
            logger.info("Trying alternative method...")
            return self._fetch_exoplanets_fallback(limit)
    
    def _fetch_exoplanets_fallback(self, limit: int) -> pd.DataFrame:
        """Fallback method using direct API."""
        import urllib.request
        import json
        
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+TOP+{limit}+pl_name,hostname,disc_facility,pl_orbper+FROM+pscomppars+WHERE+disc_facility+LIKE+'%25TESS%25'+OR+disc_facility+LIKE+'%25Kepler%25'&format=json".format(limit=limit)
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
                df = pd.DataFrame(data)
                df.to_csv(self.data_dir / "confirmed_exoplanets.csv", index=False)
                return df
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return pd.DataFrame()
    
    def fetch_light_curve(self, target: str, mission: str = "TESS") -> Optional[np.ndarray]:
        """
        Fetch light curve for a specific target with disk caching.
        """
        import hashlib
        
        # Create cache directory
        cache_dir = self.data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename based on target and mission
        cache_key = f"{target}_{mission}".replace(" ", "_").replace("+", "p").replace("-", "m")
        cache_path = cache_dir / f"{cache_key}.npy"
        
        # Check cache first
        if cache_path.exists():
            try:
                # logger.info(f"Loading cached {target}...")
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {target}: {e}")
        
        try:
            logger.info(f"Fetching light curve for {target} ({mission})...")
            
            # Search for light curves
            search_result = lk.search_lightcurve(target, mission=mission)
            
            if len(search_result) == 0:
                logger.warning(f"No light curves found for {target}")
                return None
            
            # Download first available light curve
            lc = search_result[0].download()
            
            if lc is None:
                return None
            
            # Clean and normalize
            lc = lc.remove_nans().normalize()
            
            # Get flux values
            flux = lc.flux.value
            
            # Resample to fixed size (1024 points)
            flux_resampled = self._resample_flux(flux, target_length=1024)
            
            # Save to cache
            np.save(cache_path, flux_resampled)
            
            return flux_resampled
            
        except Exception as e:
            logger.warning(f"Error fetching {target}: {e}")
            return None
    
    def _resample_flux(self, flux: np.ndarray, target_length: int = 1024) -> np.ndarray:
        """Resample flux to fixed length."""
        if len(flux) == target_length:
            return flux
        
        # Linear interpolation to target length
        x_old = np.linspace(0, 1, len(flux))
        x_new = np.linspace(0, 1, target_length)
        flux_resampled = np.interp(x_new, x_old, flux)
        
        return flux_resampled
    
    def fetch_non_planet_stars(self, count: int = 100, mission: str = "TESS") -> List[np.ndarray]:
        """
        Fetch light curves from stars WITHOUT confirmed planets.
        These will be our NEGATIVE examples.
        """
        logger.info(f"Fetching non-planet stars ({count} samples)...")
        
        non_planet_curves = []
        
        try:
            # Optimize: Search for any available light curves in a specific sector
            # This avoids "guessing" TIC IDs that don't exist
            logger.info("  Searching for available TESS targets in Sector 1...")
            # We request more than needed because some might fail to download or be too short
            search_results = lk.search_lightcurve(
                mission=mission, 
                sector=1, 
                limit=count * 3
            )
            
            if len(search_results) == 0:
                logger.warning("  No targets found in Sector 1 search.")
                return []

            logger.info(f"  Found {len(search_results)} candidate targets. Downloading...")

            # Select random subset to avoid bias (though we limited search)
            # But search_result is already a collection, let's iterate
            import random
            indices = list(range(len(search_results)))
            # Shuffle to get random selection if we found many
            random.shuffle(indices)

            for idx in indices:
                if len(non_planet_curves) >= count:
                    break

                try:
                    sr = search_results[idx]
                    target_name = sr.target_name
                    
                    # Basic check: skip likely planet hosts (if we had a list, we'd check against it)
                    # For this demo, we assume random field stars are mostly non-planets
                    
                    lc = sr.download()
                    if lc is None:
                        continue
                        
                    lc = lc.remove_nans().normalize()
                    flux = lc.flux.value
                    flux_resampled = self._resample_flux(flux, target_length=1024)
                    
                    non_planet_curves.append(flux_resampled)
                    logger.info(f"  ✓ Collected {target_name} ({len(non_planet_curves)}/{count})")
                    
                except Exception as e:
                    # logger.warning(f"Failed to process target: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching non-planet stars: {e}")
        
        return non_planet_curves


class RealDataTrainer:
    """Train the TinyML model on real NASA data."""
    
    def __init__(self, output_dir: str = "models/real", augment_factor: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = RealDataFetcher()
        self.augment_factor = augment_factor
        self.augmenter = LightCurveAugmenter()
        
        # Model parameters
        self.input_size = 1024
        self.num_classes = 6
        self.class_names = [
            "noise",
            "stellar_signal",
            "planetary_transit",
            "eclipsing_binary",
            "instrument_artifact",
            "unknown_anomaly"
        ]
    
    def prepare_training_data(
        self, 
        num_planet_samples: int = 50,
        num_non_planet_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from real sources.
        
        Args:
            num_planet_samples: Number of confirmed planet host light curves
            num_non_planet_samples: Number of non-planet light curves
            
        Returns:
            X (features), y (labels)
        """
        print("\n" + "="*70)
        print("   FETCHING REAL NASA DATA")
        print("="*70 + "\n")
        
        X_list = []
        y_list = []
        
        # 1. Fetch confirmed exoplanets (POSITIVE - class 2: planetary_transit)
        print("Step 1: Fetching confirmed exoplanet host stars...")
        print("-" * 50)
        
        exoplanets = self.fetcher.fetch_confirmed_exoplanets(limit=num_planet_samples * 2)
        
        if len(exoplanets) > 0:
            planet_count = 0
            for _, row in exoplanets.iterrows():
                if planet_count >= num_planet_samples:
                    break
                    
                hostname = row.get('hostname', row.get('pl_name', ''))
                if not hostname:
                    continue
                
                flux = self.fetcher.fetch_light_curve(hostname, mission="TESS")
                
                if flux is None:
                    # Try Kepler
                    flux = self.fetcher.fetch_light_curve(hostname, mission="Kepler")
                
                if flux is not None:
                    X_list.append(flux)
                    y_list.append(2)  # planetary_transit class
                    planet_count += 1
                    print(f"  ✓ {hostname}: planetary transit (class 2)")
        
        print(f"\nCollected {sum(1 for y in y_list if y == 2)} planetary transit samples")
        
        # 2. Fetch non-planet stars (class 0: noise or class 1: stellar_signal)
        print("\nStep 2: Fetching non-planet stars for negative samples...")
        print("-" * 50)
        
        non_planet_curves = self.fetcher.fetch_non_planet_stars(
            count=num_non_planet_samples,
            mission="TESS"
        )
        
        for flux in non_planet_curves:
            X_list.append(flux)
            # Randomly assign as noise or stellar signal
            y_list.append(np.random.choice([0, 1]))
        
        print(f"\nCollected {len(non_planet_curves)} non-planet samples")
        
        # 3. Create synthetic examples for rare classes
        print("\nStep 3: Adding synthetic examples for rare classes...")
        print("-" * 50)
        
        # Add some eclipsing binary simulations (class 3)
        for i in range(20):
            flux = self._simulate_eclipsing_binary()
            X_list.append(flux)
            y_list.append(3)
        print("  ✓ Added 20 simulated eclipsing binary samples")
        
        # Add instrument artifact simulations (class 4)
        for i in range(20):
            flux = self._simulate_artifact()
            X_list.append(flux)
            y_list.append(4)
        print("  ✓ Added 20 simulated instrument artifact samples")
        
        # Convert to numpy arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # Normalize
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        
        print(f"\n{'='*70}")
        print(f"TRAINING DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"Feature shape: {X.shape}")
        for i, name in enumerate(self.class_names):
            count = sum(1 for label in y if label == i)
            print(f"  Class {i} ({name}): {count} samples")
        
        # Save prepared data
        np.savez(
            self.output_dir / "real_training_data.npz",
            X=X, y=y, class_names=self.class_names
        )
        print(f"\nData saved to: {self.output_dir / 'real_training_data.npz'}")
        
        return X, y
    
    def _simulate_eclipsing_binary(self) -> np.ndarray:
        """Simulate eclipsing binary light curve."""
        t = np.linspace(0, 10, self.input_size)
        period = np.random.uniform(0.5, 3.0)
        
        # Primary and secondary eclipses
        flux = np.ones(self.input_size)
        phase = (t % period) / period
        
        # Primary eclipse (deeper)
        primary_mask = np.abs(phase - 0.0) < 0.05
        flux[primary_mask] -= np.random.uniform(0.1, 0.4)
        
        # Secondary eclipse (shallower)
        secondary_mask = np.abs(phase - 0.5) < 0.05
        flux[secondary_mask] -= np.random.uniform(0.02, 0.15)
        
        # Add noise
        flux += np.random.normal(0, 0.01, self.input_size)
        
        return flux.astype(np.float32)
    
    def _simulate_artifact(self) -> np.ndarray:
        """Simulate instrument artifact."""
        flux = np.ones(self.input_size)
        
        # Random discontinuities
        num_jumps = np.random.randint(1, 5)
        for _ in range(num_jumps):
            pos = np.random.randint(0, self.input_size)
            flux[pos:] += np.random.uniform(-0.1, 0.1)
        
        # Random spikes
        num_spikes = np.random.randint(5, 20)
        spike_pos = np.random.randint(0, self.input_size, num_spikes)
        flux[spike_pos] += np.random.uniform(-0.5, 0.5, num_spikes)
        
        # Add noise
        flux += np.random.normal(0, 0.02, self.input_size)
        
        return flux.astype(np.float32)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the TinyML model on prepared data."""
        print(f"\n{'='*70}")
        print("   TRAINING MODEL ON REAL DATA")
        print(f"{'='*70}\n")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples (original): {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Augment training data
        if self.augment_factor > 0:
            print(f"Augmenting training data (factor={self.augment_factor})...")
            X_train, y_train = self.augmenter.augment_batch(
                X_train, y_train, factor=self.augment_factor
            )
            print(f"Training samples (augmented): {len(X_train)}")
        
        # Try to use TensorFlow if available
        try:
            import tensorflow as tf
            return self._train_with_tensorflow(X_train, y_train, X_val, y_val, epochs)
        except ImportError:
            logger.warning("TensorFlow not available, using NumPy-based training")
            return self._train_with_numpy(X_train, y_train, X_val, y_val, epochs)
    
    def _train_with_tensorflow(self, X_train, y_train, X_val, y_val, epochs):
        """Train using TensorFlow/Keras."""
        import tensorflow as tf
        
        # Check GPU Availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✅ GPU DETECTED: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"  Note: {e}")
        else:
            print("\n⚠️  NO GPU DETECTED. Training will proceed on CPU (which may be slower).")
            print("To enable GPU, ensure CUDA and cuDNN are installed correctly.")
        
        # Reshape for 1D CNN
        X_train = X_train.reshape(-1, self.input_size, 1)
        X_val = X_val.reshape(-1, self.input_size, 1)
        
        # Build model
        # Build model using unified SpectralCNN
        print("Building model via SpectralCNN...")
        spectral_model = SpectralCNN(
            input_shape=(self.input_size, 1),
            num_classes=self.num_classes,
            use_lstm=True
        )
        model = spectral_model.build_model()
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal Validation Accuracy: {val_acc*100:.1f}%")
        
        # Save model
        model.save(self.output_dir / "astro_tinyml_real.h5")
        print(f"Model saved to: {self.output_dir / 'astro_tinyml_real.h5'}")
        
        # Convert to TFLite
        self._convert_to_tflite(model)
        
        return model, history
    
    def _train_with_numpy(self, X_train, y_train, X_val, y_val, epochs):
        """Fallback NumPy-based training."""
        # Import the numpy model from our codebase
        # Import the numpy model from our codebase
        from src.model.numpy_cnn import NumpyCNN
        
        model = NumpyCNN(input_shape=(self.input_size, 1), num_classes=self.num_classes)
        
        print("\nTraining with NumPy backend...")
        # NumpyCNN uses .fit(), not .train(), and handles validation internally via validation_split
        history = model.fit(X_train, y_train, epochs=epochs)
        
        # Save weights
        model.save(str(self.output_dir / "model_weights_real.json"))
        model.export_to_c_header(str(self.output_dir / "astro_tinyml_real.h"))
        
        return model, history
    
    def _convert_to_tflite(self, model):
        """Convert Keras model to TFLite for edge deployment."""
        import tensorflow as tf
        
        # Standard conversion (Float32)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable Select TF Ops for LSTM support
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        try:
            tflite_model = converter.convert()
            tflite_path = self.output_dir / "astro_tinyml_real.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")
        except Exception as e:
            logger.warning(f"Standard TFLite conversion failed: {e}")
        
        # Quantized conversion (INT8)
        converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_q.target_spec.supported_types = [tf.int8]
        
        # Also need ops support for quantized version
        converter_q.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter_q._experimental_lower_tensor_list_ops = False
        
        try:
            quantized_model = converter_q.convert()
            quantized_path = self.output_dir / "astro_tinyml_real_int8.tflite"
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            print(f"Quantized model saved: {quantized_path} ({len(quantized_model)/1024:.1f} KB)")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")


def main():
    """Main training pipeline."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██╗      █████╗ ██████╗ ██╗   ██╗███╗   ██╗                      ║
║     ██║     ██╔══██╗██╔══██╗██║   ██║████╗  ██║                      ║
║     ██║     ███████║██████╔╝██║   ██║██╔██╗ ██║                      ║
║     ██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║                      ║
║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║██╗                   ║
║     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝                   ║
║                                                                      ║
║     AstroTinyML - Real Data Training Pipeline                        ║
║     Larun. × Astrodata                                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    parser = argparse.ArgumentParser(description="Train AstroTinyML on real NASA data")
    parser.add_argument("--planets", type=int, default=50, 
                        help="Number of confirmed planet hosts to fetch")
    parser.add_argument("--non-planets", type=int, default=50,
                        help="Number of non-planet stars to fetch")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--output", type=str, default="models/real",
                        help="Output directory")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="Data augmentation factor")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RealDataTrainer(output_dir=args.output, augment_factor=args.augment_factor)
    
    # Prepare data
    X, y = trainer.prepare_training_data(
        num_planet_samples=args.planets,
        num_non_planet_samples=args.non_planets
    )
    
    if len(X) < 10:
        print("\n⚠️  Not enough data collected. Check your internet connection.")
        print("    The NASA archives may be temporarily unavailable.")
        sys.exit(1)
    
    # Train model
    model, history = trainer.train_model(X, y, epochs=args.epochs)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     TRAINING COMPLETE                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Model trained on REAL NASA data!                                    ║
║                                                                      ║
║  Output files:                                                       ║
║    • models/real/astro_tinyml_real.h5      (Keras model)            ║
║    • models/real/astro_tinyml_real.tflite  (TFLite for mobile)      ║
║    • models/real/astro_tinyml_real_int8.tflite (Quantized)          ║
║    • models/real/real_training_data.npz    (Training data)          ║
║                                                                      ║
║  Larun. × Astrodata                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()

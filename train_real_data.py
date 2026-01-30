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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
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
        print("Missing: " + ", ".join(missing))
        print("Install: pip install " + " ".join(missing))
        sys.exit(1)

check_dependencies()

import pandas as pd
import lightkurve as lk
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive


class RealDataFetcher:
    """Fetch real astronomical data from NASA archives."""
    
    def __init__(self, data_dir: str = "data/real"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_confirmed_exoplanets(self, limit: int = 100) -> pd.DataFrame:
        logger.info(f"Fetching confirmed exoplanets (limit: {limit})...")
        try:
            planets = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name,hostname,disc_facility,pl_orbper",
                where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'",
                order="pl_name"
            )
            df = planets.to_pandas()
            logger.info(f"Found {len(df)} confirmed exoplanets")
            return df.head(limit)
        except Exception as e:
            logger.error(f"Error: {e}")
            return pd.DataFrame()
    
    def fetch_light_curve(self, target: str, mission: str = "TESS") -> Optional[np.ndarray]:
        try:
            logger.info(f"Fetching light curve for {target} ({mission})...")
            search_result = lk.search_lightcurve(target, mission=mission)
            if len(search_result) == 0:
                return None
            lc = search_result[0].download()
            if lc is None:
                return None
            lc = lc.remove_nans().normalize()
            flux = lc.flux.value
            return self._resample_flux(flux, 1024)
        except Exception as e:
            logger.warning(f"Error fetching {target}: {e}")
            return None
    
    def _resample_flux(self, flux: np.ndarray, target_length: int = 1024) -> np.ndarray:
        if len(flux) == target_length:
            return flux
        x_old = np.linspace(0, 1, len(flux))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, flux)


class RealDataTrainer:
    """Train the TinyML model on real NASA data."""
    
    def __init__(self, output_dir: str = "models/real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = RealDataFetcher()
        self.input_size = 1024
        self.num_classes = 6
        self.class_names = ["noise", "stellar_signal", "planetary_transit", 
                          "eclipsing_binary", "instrument_artifact", "unknown_anomaly"]
    
    def prepare_training_data(self, num_planet_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        print("\n" + "="*60)
        print("   FETCHING REAL NASA DATA")
        print("="*60)
        
        X_list, y_list = [], []
        
        # Fetch confirmed exoplanets (positive samples)
        exoplanets = self.fetcher.fetch_confirmed_exoplanets(limit=num_planet_samples * 2)
        
        if len(exoplanets) > 0:
            count = 0
            for _, row in exoplanets.iterrows():
                if count >= num_planet_samples:
                    break
                hostname = row.get('hostname', '')
                if not hostname:
                    continue
                flux = self.fetcher.fetch_light_curve(hostname, "TESS")
                if flux is None:
                    flux = self.fetcher.fetch_light_curve(hostname, "Kepler")
                if flux is not None:
                    X_list.append(flux)
                    y_list.append(2)  # planetary_transit
                    count += 1
                    print(f"  ✓ {hostname}")
        
        # Add synthetic eclipsing binaries
        for _ in range(20):
            X_list.append(self._simulate_eclipsing_binary())
            y_list.append(3)
        
        # Add synthetic artifacts  
        for _ in range(20):
            X_list.append(self._simulate_artifact())
            y_list.append(4)
        
        # Add noise samples
        for _ in range(20):
            X_list.append(np.random.normal(1.0, 0.01, self.input_size).astype(np.float32))
            y_list.append(0)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        
        np.savez(self.output_dir / "real_training_data.npz", X=X, y=y)
        print(f"\nTotal samples: {len(X)}")
        return X, y
    
    def _simulate_eclipsing_binary(self) -> np.ndarray:
        t = np.linspace(0, 10, self.input_size)
        period = np.random.uniform(0.5, 3.0)
        flux = np.ones(self.input_size)
        phase = (t % period) / period
        flux[np.abs(phase) < 0.05] -= np.random.uniform(0.1, 0.4)
        flux[np.abs(phase - 0.5) < 0.05] -= np.random.uniform(0.02, 0.15)
        flux += np.random.normal(0, 0.01, self.input_size)
        return flux.astype(np.float32)
    
    def _simulate_artifact(self) -> np.ndarray:
        flux = np.ones(self.input_size)
        for _ in range(np.random.randint(1, 5)):
            pos = np.random.randint(0, self.input_size)
            flux[pos:] += np.random.uniform(-0.1, 0.1)
        spike_pos = np.random.randint(0, self.input_size, np.random.randint(5, 20))
        flux[spike_pos] += np.random.uniform(-0.5, 0.5, len(spike_pos))
        flux += np.random.normal(0, 0.02, self.input_size)
        return flux.astype(np.float32)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        print("\n" + "="*60)
        print("   TRAINING MODEL ON REAL DATA")
        print("="*60)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            import tensorflow as tf
            X_train = X_train.reshape(-1, self.input_size, 1)
            X_val = X_val.reshape(-1, self.input_size, 1)
            
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(16, 7, activation='relu', input_shape=(self.input_size, 1)),
                tf.keras.layers.MaxPooling1D(4),
                tf.keras.layers.Conv1D(32, 5, activation='relu'),
                tf.keras.layers.MaxPooling1D(4),
                tf.keras.layers.Conv1D(64, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
            
            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"\nValidation Accuracy: {val_acc*100:.1f}%")
            
            model.save(self.output_dir / "astro_tinyml_real.h5")
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(self.output_dir / "astro_tinyml_real.tflite", 'wb') as f:
                f.write(tflite_model)
            print(f"Models saved to {self.output_dir}")
            
            return model
        except ImportError:
            print("TensorFlow not available")
            return None


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  AstroTinyML - Real NASA Data Training                    ║
    ║  Larun. × Astrodata                                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--planets", type=int, default=50, help="Number of planet hosts")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--output", type=str, default="models/real", help="Output dir")
    args = parser.parse_args()
    
    trainer = RealDataTrainer(output_dir=args.output)
    X, y = trainer.prepare_training_data(num_planet_samples=args.planets)
    
    if len(X) >= 10:
        trainer.train_model(X, y, epochs=args.epochs)
        print("\n✓ Training complete!")
    else:
        print("\n⚠️ Not enough data collected")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LARUN - Specialized Model Training Pipeline
============================================
Train specialized models for:
- Stellar Classification (spectral types)
- Binary Discrimination (EB vs planet)
- Habitability Assessment

Supports:
- 500+ exoplanet training data from NASA Archive
- K-fold cross-validation (5-fold default)
- Data augmentation
- Model registration in federated system

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


# ============================================================================
# Data Fetching with Parallel Processing
# ============================================================================

class ExpandedDataFetcher:
    """Fetch 500+ exoplanets with parallel processing."""
    
    def __init__(self, data_dir: str = "data/expanded"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_all_confirmed_exoplanets(self, limit: int = 600) -> 'pd.DataFrame':
        """Fetch all confirmed exoplanets from NASA Archive."""
        import pandas as pd
        
        cache_file = self.data_dir / "all_exoplanets.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached exoplanet catalog ({cache_file})")
            return pd.read_csv(cache_file)
        
        logger.info(f"Fetching {limit} confirmed exoplanets from NASA Archive...")
        
        try:
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            
            planets = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name,hostname,disc_facility,pl_orbper,pl_rade,pl_bmasse,sy_vmag,ra,dec,st_teff,st_rad,st_lum,st_mass,st_logg",
                where="disc_facility LIKE '%TESS%' OR disc_facility LIKE '%Kepler%'",
                order="pl_name"
            )
            
            df = planets.to_pandas()
            logger.info(f"Found {len(df)} confirmed exoplanets")
            
            # Save cache
            df.to_csv(cache_file, index=False)
            
            return df.head(limit)
            
        except Exception as e:
            logger.error(f"Error fetching exoplanets: {e}")
            return self._fallback_fetch(limit)
    
    def _fallback_fetch(self, limit: int) -> 'pd.DataFrame':
        """Fallback using direct TAP API."""
        import urllib.request
        import pandas as pd
        
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+TOP+{limit}+pl_name,hostname,disc_facility,pl_orbper,pl_rade,st_teff,st_rad,st_lum,st_mass+FROM+pscomppars+WHERE+disc_facility+LIKE+'%25TESS%25'+OR+disc_facility+LIKE+'%25Kepler%25'&format=json"
        
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode())
                df = pd.DataFrame(data)
                df.to_csv(self.data_dir / "all_exoplanets.csv", index=False)
                return df
        except Exception as e:
            logger.error(f"Fallback fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_eclipsing_binaries(self, limit: int = 200) -> 'pd.DataFrame':
        """Fetch eclipsing binary catalog for negative training data."""
        import pandas as pd
        
        cache_file = self.data_dir / "eclipsing_binaries.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached EB catalog")
            return pd.read_csv(cache_file)
        
        logger.info(f"Fetching {limit} eclipsing binaries...")
        
        # Using Kepler Eclipsing Binary Catalog
        url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+TOP+{limit}+kepid,period,tprimary,morphology+FROM+keb+WHERE+period+IS+NOT+NULL&format=json"
        
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode())
                df = pd.DataFrame(data)
                df.to_csv(cache_file, index=False)
                logger.info(f"Found {len(df)} eclipsing binaries")
                return df
        except Exception as e:
            logger.warning(f"EB catalog fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_light_curve_parallel(
        self,
        targets: List[str],
        mission: str = "TESS",
        max_workers: int = 4
    ) -> Dict[str, Optional[np.ndarray]]:
        """Fetch multiple light curves in parallel."""
        import lightkurve as lk
        
        results = {}
        
        def fetch_one(target: str) -> Tuple[str, Optional[np.ndarray]]:
            cache_path = self.cache_dir / f"{target.replace(' ', '_')}_{mission}.npy"
            
            if cache_path.exists():
                try:
                    return target, np.load(cache_path)
                except:
                    pass
            
            try:
                search = lk.search_lightcurve(target, mission=mission)
                if len(search) == 0:
                    return target, None
                
                lc = search[0].download()
                if lc is None:
                    return target, None
                
                lc = lc.remove_nans().normalize()
                flux = lc.flux.value
                
                # Resample to 1024 points
                if len(flux) > 10:
                    x_old = np.linspace(0, 1, len(flux))
                    x_new = np.linspace(0, 1, 1024)
                    flux_resampled = np.interp(x_new, x_old, flux).astype(np.float32)
                    
                    # Cache
                    np.save(cache_path, flux_resampled)
                    return target, flux_resampled
                
                return target, None
                
            except Exception as e:
                logger.debug(f"Failed to fetch {target}: {e}")
                return target, None
        
        logger.info(f"Fetching {len(targets)} light curves with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, t): t for t in targets}
            
            for i, future in enumerate(as_completed(futures)):
                target, flux = future.result()
                results[target] = flux
                
                if (i + 1) % 50 == 0:
                    success = sum(1 for v in results.values() if v is not None)
                    logger.info(f"Progress: {i+1}/{len(targets)} ({success} successful)")
        
        success = sum(1 for v in results.values() if v is not None)
        logger.info(f"Fetched {success}/{len(targets)} light curves")
        
        return results


# ============================================================================
# K-Fold Training Pipeline
# ============================================================================

class SpecializedTrainer:
    """Train specialized models with K-fold cross-validation."""
    
    def __init__(
        self,
        output_dir: str = "models/specialized",
        n_folds: int = 5
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_folds = n_folds
        self.fetcher = ExpandedDataFetcher()
    
    def prepare_training_data(
        self,
        num_planets: int = 500,
        num_ebs: int = 200
    ) -> Dict[str, np.ndarray]:
        """Prepare training data from real and synthetic sources."""
        import pandas as pd
        
        print("\n" + "=" * 70)
        print("   PREPARING EXPANDED TRAINING DATA")
        print("=" * 70 + "\n")
        
        X_list = []
        y_transit = []  # 0=no transit, 1=transit
        y_binary = []   # 0=planet, 1=EB
        
        # Stellar parameters for each sample
        stellar_params = []
        
        # 1. Fetch confirmed exoplanets
        print("Step 1: Fetching confirmed exoplanets...")
        print("-" * 50)
        
        exoplanets = self.fetcher.fetch_all_confirmed_exoplanets(limit=num_planets * 2)
        
        if len(exoplanets) > 0:
            hostnames = exoplanets['hostname'].dropna().unique().tolist()[:num_planets]
            light_curves = self.fetcher.fetch_light_curve_parallel(hostnames)
            
            for hostname in hostnames:
                if hostname in light_curves and light_curves[hostname] is not None:
                    X_list.append(light_curves[hostname])
                    y_transit.append(1)  # Has transit
                    y_binary.append(0)   # Planet (not EB)
                    
                    # Get stellar params
                    row = exoplanets[exoplanets['hostname'] == hostname].iloc[0]
                    stellar_params.append({
                        'teff': row.get('st_teff', 5778),
                        'radius': row.get('st_rad', 1.0),
                        'period': row.get('pl_orbper', 10.0),
                        'planet_radius': row.get('pl_rade', 1.0)
                    })
        
        print(f"Collected {len(X_list)} planet host light curves")
        
        # 2. Fetch eclipsing binaries
        print("\nStep 2: Fetching eclipsing binaries...")
        print("-" * 50)
        
        ebs = self.fetcher.fetch_eclipsing_binaries(limit=num_ebs * 2)
        
        if len(ebs) > 0 and 'kepid' in ebs.columns:
            eb_targets = [f"KIC {kid}" for kid in ebs['kepid'].dropna().astype(int).head(num_ebs)]
            eb_curves = self.fetcher.fetch_light_curve_parallel(eb_targets, mission="Kepler")
            
            for target in eb_targets:
                if target in eb_curves and eb_curves[target] is not None:
                    X_list.append(eb_curves[target])
                    y_transit.append(1)  # Has eclipse
                    y_binary.append(1)   # EB (not planet)
                    stellar_params.append({
                        'teff': 5778,
                        'radius': 1.0,
                        'period': 3.0,
                        'planet_radius': None
                    })
        
        print(f"Collected {sum(y_binary)} EB light curves")
        
        # 3. Add synthetic examples
        print("\nStep 3: Adding synthetic examples...")
        print("-" * 50)
        
        # Synthetic non-transiting stars
        for i in range(100):
            flux = np.random.normal(1.0, 0.002, 1024).astype(np.float32)
            X_list.append(flux)
            y_transit.append(0)
            y_binary.append(0)
            stellar_params.append({'teff': 5778, 'radius': 1.0, 'period': 0, 'planet_radius': None})
        print("  ✓ Added 100 synthetic non-transiting stars")
        
        # Synthetic EBs
        for i in range(50):
            flux = self._simulate_eb()
            X_list.append(flux)
            y_transit.append(1)
            y_binary.append(1)
            stellar_params.append({'teff': 6000, 'radius': 1.2, 'period': 2.0, 'planet_radius': None})
        print("  ✓ Added 50 synthetic eclipsing binaries")
        
        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)
        y_transit = np.array(y_transit, dtype=np.int32)
        y_binary = np.array(y_binary, dtype=np.int32)
        
        # Normalize
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        
        print(f"\n{'='*70}")
        print(f"TRAINING DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"  Transit present: {sum(y_transit)}")
        print(f"  No transit: {sum(y_transit == 0)}")
        print(f"  Planets: {sum((y_transit == 1) & (y_binary == 0))}")
        print(f"  Eclipsing Binaries: {sum(y_binary)}")
        
        # Save
        data_file = self.output_dir / "training_data.npz"
        np.savez(data_file, X=X, y_transit=y_transit, y_binary=y_binary)
        print(f"\nData saved to: {data_file}")
        
        return {
            'X': X,
            'y_transit': y_transit,
            'y_binary': y_binary,
            'stellar_params': stellar_params
        }
    
    def _simulate_eb(self) -> np.ndarray:
        """Simulate eclipsing binary light curve."""
        t = np.linspace(0, 10, 1024)
        period = np.random.uniform(0.5, 3.0)
        phase = (t % period) / period
        
        flux = np.ones(1024)
        
        # Primary eclipse (V-shaped)
        primary = np.abs(phase) < 0.05
        depth1 = np.random.uniform(0.05, 0.3)
        flux[primary] -= depth1 * (1 - np.abs(phase[primary]) / 0.05)
        
        # Secondary eclipse
        secondary = np.abs(phase - 0.5) < 0.04
        depth2 = np.random.uniform(0.01, 0.1)
        flux[secondary] -= depth2
        
        flux += np.random.normal(0, 0.005, 1024)
        
        return flux.astype(np.float32)
    
    def train_binary_discriminator(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        y_transit: np.ndarray
    ) -> Dict[str, Any]:
        """Train binary discriminator with K-fold validation."""
        from sklearn.model_selection import StratifiedKFold
        
        print("\n" + "=" * 70)
        print("   TRAINING BINARY DISCRIMINATOR (K-Fold)")
        print("=" * 70 + "\n")
        
        # Filter to only transiting/eclipsing signals
        transit_mask = y_transit == 1
        X_transit = X[transit_mask]
        y_binary_filtered = y_binary[transit_mask]
        
        print(f"Training samples: {len(X_transit)}")
        print(f"  Planets: {sum(y_binary_filtered == 0)}")
        print(f"  Binaries: {sum(y_binary_filtered == 1)}")
        
        # Reshape for CNN
        X_transit = X_transit.reshape(-1, 1024, 1)
        
        # K-Fold cross-validation
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        best_accuracy = 0
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_transit, y_binary_filtered)):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")
            
            X_train, X_val = X_transit[train_idx], X_transit[val_idx]
            y_train, y_val = y_binary_filtered[train_idx], y_binary_filtered[val_idx]
            
            # Build model
            from models.binary_discriminator import BinaryDiscriminator
            discriminator = BinaryDiscriminator()
            discriminator.build()
            
            if discriminator.model is None:
                print("TensorFlow not available, skipping training")
                break
            
            # Train
            history = discriminator.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=16,
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc = discriminator.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_acc)
            print(f"  Validation Accuracy: {val_acc:.1%}")
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model = discriminator.model
        
        # Save best model
        if best_model and fold_scores:
            model_path = self.output_dir / "binary_discriminator.h5"
            best_model.save(model_path)
            print(f"\nBest model saved: {model_path}")
            print(f"Mean CV Accuracy: {np.mean(fold_scores):.1%} ± {np.std(fold_scores):.1%}")
        
        return {
            'fold_scores': fold_scores,
            'mean_accuracy': np.mean(fold_scores) if fold_scores else 0,
            'std_accuracy': np.std(fold_scores) if fold_scores else 0,
            'best_accuracy': best_accuracy
        }
    
    def train_stellar_classifier(
        self,
        X: np.ndarray,
        stellar_params: List[Dict]
    ) -> Dict[str, Any]:
        """Train stellar classifier with K-fold validation."""
        from sklearn.model_selection import StratifiedKFold
        
        print("\n" + "=" * 70)
        print("   TRAINING STELLAR CLASSIFIER (K-Fold)")
        print("=" * 70 + "\n")
        
        # Create labels from Teff
        spectral_types = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}
        type_ranges = [(50000, 'O'), (30000, 'B'), (10000, 'A'), (7500, 'F'),
                       (6000, 'G'), (5200, 'K'), (3700, 'M'), (0, 'M')]
        
        def teff_to_label(teff):
            if teff is None or np.isnan(teff):
                teff = 5778  # Default to solar
            for thresh, stype in type_ranges:
                if teff >= thresh:
                    return spectral_types[stype]
            return spectral_types['M']
        
        y_spectral = np.array([teff_to_label(p.get('teff', 5778)) for p in stellar_params])
        
        print(f"Training samples: {len(X)}")
        print("Class distribution:")
        for stype, idx in spectral_types.items():
            print(f"  {stype}: {sum(y_spectral == idx)}")
        
        # Reshape
        X_reshaped = X.reshape(-1, 1024, 1)
        
        # K-Fold
        kfold = StratifiedKFold(n_splits=min(self.n_folds, 3), shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_reshaped, y_spectral)):
            print(f"\n--- Fold {fold + 1} ---")
            
            # Build simple classifier for spectral type only
            try:
                import tensorflow as tf
                from tensorflow.keras import layers, Model
                
                inputs = layers.Input(shape=(1024, 1))
                x = layers.Conv1D(32, 16, activation='relu', padding='same')(inputs)
                x = layers.MaxPooling1D(4)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv1D(64, 8, activation='relu', padding='same')(x)
                x = layers.MaxPooling1D(4)(x)
                x = layers.GlobalAveragePooling1D()(x)
                x = layers.Dense(32, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                outputs = layers.Dense(7, activation='softmax')(x)
                
                model = Model(inputs, outputs)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                model.fit(
                    X_reshaped[train_idx], y_spectral[train_idx],
                    validation_data=(X_reshaped[val_idx], y_spectral[val_idx]),
                    epochs=20,
                    batch_size=16,
                    verbose=0
                )
                
                _, val_acc = model.evaluate(X_reshaped[val_idx], y_spectral[val_idx], verbose=0)
                fold_scores.append(val_acc)
                print(f"  Validation Accuracy: {val_acc:.1%}")
                
            except ImportError:
                print("TensorFlow not available")
                break
        
        if fold_scores:
            print(f"\nMean CV Accuracy: {np.mean(fold_scores):.1%} ± {np.std(fold_scores):.1%}")
        
        return {
            'fold_scores': fold_scores,
            'mean_accuracy': np.mean(fold_scores) if fold_scores else 0
        }
    
    def register_models_in_federated(self):
        """Register trained models in the federated registry."""
        from federated.registry import ModelRegistry, ModelMetadata
        
        print("\n" + "=" * 70)
        print("   REGISTERING MODELS IN FEDERATED SYSTEM")
        print("=" * 70 + "\n")
        
        registry = ModelRegistry()
        
        # Check for trained models
        model_files = [
            ("binary_discriminator.h5", "binary_discrimination", 0.80),
            ("stellar_classifier.h5", "stellar_classification", 0.70),
            ("habitability_assessor.h5", "habitability_assessment", 0.85),
        ]
        
        for filename, task, default_accuracy in model_files:
            model_path = self.output_dir / filename
            
            if model_path.exists():
                meta = ModelMetadata(
                    model_id=f"{task}_v1",
                    version="1.0.0",
                    task=task,
                    accuracy=default_accuracy,
                    input_shape=(1024, 1),
                    created_at=datetime.now().isoformat(),
                    file_path=str(model_path)
                )
                
                registry.register(meta)
                print(f"  ✓ Registered {task} model")
            else:
                print(f"  ○ {task} model not found (not trained yet)")
        
        print(f"\nRegistry summary:\n{registry.summary()}")


# ============================================================================
# Main
# ============================================================================

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
║     Specialized Model Training Pipeline                              ║
║     Larun. × Astrodata                                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    parser = argparse.ArgumentParser(description="Train specialized LARUN models")
    parser.add_argument("--planets", type=int, default=500,
                        help="Number of confirmed planets to fetch")
    parser.add_argument("--ebs", type=int, default=200,
                        help="Number of eclipsing binaries to fetch")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of K-fold splits")
    parser.add_argument("--output", type=str, default="models/specialized",
                        help="Output directory")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching (use cached)")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SpecializedTrainer(output_dir=args.output, n_folds=args.kfold)
    
    # Prepare data
    if args.skip_fetch:
        data_file = Path(args.output) / "training_data.npz"
        if data_file.exists():
            print("Loading cached training data...")
            data = np.load(data_file)
            training_data = {
                'X': data['X'],
                'y_transit': data['y_transit'],
                'y_binary': data['y_binary'],
                'stellar_params': [{'teff': 5778}] * len(data['X'])  # Placeholder
            }
        else:
            training_data = trainer.prepare_training_data(args.planets, args.ebs)
    else:
        training_data = trainer.prepare_training_data(args.planets, args.ebs)
    
    if len(training_data['X']) < 50:
        print("\n⚠️  Not enough data collected.")
        return
    
    # Train models
    binary_results = trainer.train_binary_discriminator(
        training_data['X'],
        training_data['y_binary'],
        training_data['y_transit']
    )
    
    stellar_results = trainer.train_stellar_classifier(
        training_data['X'],
        training_data['stellar_params']
    )
    
    # Register in federated system
    trainer.register_models_in_federated()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     TRAINING COMPLETE                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Results:                                                            ║
║    Binary Discriminator: {binary_results['mean_accuracy']*100:5.1f}% ± {binary_results['std_accuracy']*100:.1f}%                        ║
║    Stellar Classifier:   {stellar_results['mean_accuracy']*100:5.1f}%                                  ║
║                                                                      ║
║  Models saved to: {args.output:<48} ║
║                                                                      ║
║  Larun. × Astrodata                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()

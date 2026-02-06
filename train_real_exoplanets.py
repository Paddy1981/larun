#!/usr/bin/env python3
"""
Real Data Training for EXOPLANET-001
=====================================
Train exoplanet detection on real Kepler/TESS data.

Requirements:
    pip install lightkurve astroquery pandas

Data Sources:
    - NASA Exoplanet Archive: Confirmed planets and false positives
    - MAST/lightkurve: Light curve downloads
    - Kepler/K2/TESS missions

Usage:
    python train_real_exoplanets.py --download   # Download data first
    python train_real_exoplanets.py --train      # Train on cached data
    python train_real_exoplanets.py --full       # Download + Train
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from train_simplified import extract_lightcurve_features, SimpleClassifier, train_classifier


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RealDataConfig:
    """Configuration for real data training."""
    # Data sources
    use_kepler: bool = True
    use_tess: bool = True
    use_k2: bool = False

    # Sample limits
    n_confirmed_planets: int = 500
    n_false_positives: int = 500
    n_field_stars: int = 1000

    # Light curve parameters
    target_length: int = 1024
    min_snr: float = 5.0

    # Training parameters
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.05
    val_split: float = 0.2

    # Output
    cache_dir: str = "data/real_exoplanets"
    output_dir: str = "models/trained"


# =============================================================================
# NASA Exoplanet Archive Query
# =============================================================================

def fetch_confirmed_planets(limit: int = 500) -> pd.DataFrame:
    """
    Fetch confirmed transiting exoplanets from NASA Exoplanet Archive.

    Returns DataFrame with: hostname, pl_name, tic_id, kic_id,
                           pl_trandep, pl_trandur, pl_orbper
    """
    print(f"  Fetching {limit} confirmed transiting planets from NASA Exoplanet Archive...")

    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

        # Query confirmed planets with transit data
        table = NasaExoplanetArchive.query_criteria(
            table='pscomppars',
            select='hostname,pl_name,tic_id,pl_trandep,pl_trandur,pl_orbper,disc_facility',
            where="pl_trandep IS NOT NULL AND tran_flag = 1",
            order="pl_trandep DESC",
        )

        df = table.to_pandas()

        # Filter to Kepler/TESS discoveries
        df = df[df['disc_facility'].str.contains('Kepler|TESS|K2', case=False, na=False)]

        # Remove duplicates by hostname (keep deepest transit)
        df = df.drop_duplicates(subset='hostname', keep='first')

        if len(df) > limit:
            df = df.head(limit)

        print(f"    Found {len(df)} confirmed planets with transit data")
        print(f"    Transit depths: {df['pl_trandep'].min():.0f} - {df['pl_trandep'].max():.0f} ppm")

        return df

    except ImportError:
        print("    astroquery not installed. Install with: pip install astroquery")
        return _get_fallback_planets(limit)
    except Exception as e:
        print(f"    Query failed: {e}")
        return _get_fallback_planets(limit)


def fetch_false_positives(limit: int = 500) -> pd.DataFrame:
    """
    Fetch known false positives (eclipsing binaries, etc.) from catalogs.
    """
    print(f"  Fetching {limit} known false positives...")

    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

        # Query Kepler false positives
        table = NasaExoplanetArchive.query_criteria(
            table='koi',
            select='kepid,koi_disposition,koi_period,koi_depth',
            where="koi_disposition = 'FALSE POSITIVE'",
            order="koi_depth DESC",
        )

        df = table.to_pandas()

        if len(df) > limit:
            df = df.head(limit)

        print(f"    Found {len(df)} false positives")

        return df

    except Exception as e:
        print(f"    Query failed: {e}, using fallback data")
        return _get_fallback_false_positives(limit)


def _get_fallback_planets(limit: int) -> pd.DataFrame:
    """Fallback list of well-known confirmed planets."""
    planets = [
        # Kepler planets
        {"hostname": "Kepler-10", "pl_name": "Kepler-10 b", "kic_id": "11904151", "pl_trandep": 153, "disc_facility": "Kepler"},
        {"hostname": "Kepler-22", "pl_name": "Kepler-22 b", "kic_id": "10593626", "pl_trandep": 492, "disc_facility": "Kepler"},
        {"hostname": "Kepler-452", "pl_name": "Kepler-452 b", "kic_id": "8311864", "pl_trandep": 200, "disc_facility": "Kepler"},
        {"hostname": "Kepler-62", "pl_name": "Kepler-62 f", "kic_id": "9002278", "pl_trandep": 130, "disc_facility": "Kepler"},
        {"hostname": "Kepler-186", "pl_name": "Kepler-186 f", "kic_id": "8120608", "pl_trandep": 100, "disc_facility": "Kepler"},
        # TESS planets
        {"hostname": "TOI-700", "pl_name": "TOI-700 d", "tic_id": "150428135", "pl_trandep": 600, "disc_facility": "TESS"},
        {"hostname": "TOI-1233", "pl_name": "TOI-1233 b", "tic_id": "231670397", "pl_trandep": 3000, "disc_facility": "TESS"},
        {"hostname": "TOI-270", "pl_name": "TOI-270 b", "tic_id": "259377017", "pl_trandep": 1200, "disc_facility": "TESS"},
        {"hostname": "TOI-125", "pl_name": "TOI-125 b", "tic_id": "52368076", "pl_trandep": 800, "disc_facility": "TESS"},
        {"hostname": "TOI-421", "pl_name": "TOI-421 b", "tic_id": "94986319", "pl_trandep": 500, "disc_facility": "TESS"},
    ]

    df = pd.DataFrame(planets[:limit])
    print(f"    Using {len(df)} fallback confirmed planets")
    return df


def _get_fallback_false_positives(limit: int) -> pd.DataFrame:
    """Fallback list of known false positives (eclipsing binaries)."""
    fps = [
        {"kepid": "3128793", "koi_disposition": "FALSE POSITIVE", "koi_depth": 50000},
        {"kepid": "5773345", "koi_disposition": "FALSE POSITIVE", "koi_depth": 45000},
        {"kepid": "8845026", "koi_disposition": "FALSE POSITIVE", "koi_depth": 40000},
        {"kepid": "9851944", "koi_disposition": "FALSE POSITIVE", "koi_depth": 35000},
        {"kepid": "10748390", "koi_disposition": "FALSE POSITIVE", "koi_depth": 30000},
    ]

    df = pd.DataFrame(fps[:limit])
    print(f"    Using {len(df)} fallback false positives")
    return df


# =============================================================================
# Light Curve Download
# =============================================================================

def download_lightcurve(target_id: str, mission: str = "Kepler",
                        target_length: int = 1024) -> Optional[np.ndarray]:
    """
    Download and preprocess a light curve from MAST.

    Args:
        target_id: KIC ID, TIC ID, or target name
        mission: "Kepler", "TESS", or "K2"
        target_length: Desired output length

    Returns:
        Preprocessed flux array or None if download fails
    """
    try:
        import lightkurve as lk

        # Search for light curves
        if mission == "Kepler" and target_id.isdigit():
            search_query = f"KIC {target_id}"
        elif mission == "TESS" and target_id.isdigit():
            search_query = f"TIC {target_id}"
        else:
            search_query = target_id

        search_result = lk.search_lightcurve(
            search_query,
            mission=mission,
            author="Kepler" if mission == "Kepler" else "SPOC"
        )

        if len(search_result) == 0:
            return None

        # Download first quarter/sector
        lc = search_result[0].download()

        if lc is None:
            return None

        # Preprocess
        lc = lc.remove_nans()
        lc = lc.normalize()

        # Remove outliers
        lc = lc.remove_outliers(sigma=5)

        # Flatten (remove stellar variability)
        try:
            lc = lc.flatten(window_length=401)
        except:
            pass

        # Get flux values
        flux = lc.flux.value

        # Handle NaNs
        if np.isnan(flux).any():
            flux = np.nan_to_num(flux, nan=1.0)

        # Resample to target length
        flux = _resample_flux(flux, target_length)

        return flux.astype(np.float32)

    except Exception as e:
        return None


def _resample_flux(flux: np.ndarray, target_length: int) -> np.ndarray:
    """Resample flux array to target length."""
    if len(flux) == target_length:
        return flux
    elif len(flux) > target_length:
        # Downsample by binning
        n_bins = target_length
        bin_size = len(flux) // n_bins
        flux_binned = np.array([
            np.nanmedian(flux[i*bin_size:(i+1)*bin_size])
            for i in range(n_bins)
        ])
        return flux_binned
    else:
        # Upsample by interpolation
        x_old = np.linspace(0, 1, len(flux))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, flux)


# =============================================================================
# Data Collection Pipeline
# =============================================================================

def collect_real_data(config: RealDataConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Collect real light curves and labels.

    Returns:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
        metadata: Collection statistics
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "real_data_cache.npz"
    metadata_file = cache_dir / "collection_metadata.json"

    # Check cache
    if cache_file.exists() and metadata_file.exists():
        print("\nLoading cached real data...")
        data = np.load(cache_file, allow_pickle=True)
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"  Loaded {len(data['X'])} samples from cache")
        return data['X'], data['y'], metadata

    print("\n" + "=" * 60)
    print("COLLECTING REAL EXOPLANET DATA")
    print("=" * 60)

    X_list = []
    y_list = []
    metadata = {
        "confirmed_planets": 0,
        "false_positives": 0,
        "field_stars": 0,
        "failed_downloads": 0,
        "sources": []
    }

    # Check lightkurve availability
    try:
        import lightkurve
        has_lightkurve = True
        print("\nlightkurve available - will download real data")
    except ImportError:
        has_lightkurve = False
        print("\nlightkurve not available - using synthetic data")
        print("Install with: pip install lightkurve")

    if has_lightkurve:
        # 1. Confirmed planets (label = 1: transit)
        print("\n[1/3] Downloading confirmed planet light curves...")
        planets_df = fetch_confirmed_planets(config.n_confirmed_planets)

        for idx, row in planets_df.iterrows():
            target_id = str(row.get('tic_id') or row.get('kic_id', ''))
            if not target_id or target_id == 'nan':
                continue

            mission = "TESS" if pd.notna(row.get('tic_id')) else "Kepler"

            flux = download_lightcurve(target_id, mission, config.target_length)

            if flux is not None:
                features = extract_lightcurve_features(flux)
                X_list.append(features)
                y_list.append(1)  # transit
                metadata["confirmed_planets"] += 1
                metadata["sources"].append({
                    "target": row.get('pl_name', target_id),
                    "type": "confirmed_planet",
                    "mission": mission
                })

                if metadata["confirmed_planets"] % 10 == 0:
                    print(f"    Downloaded {metadata['confirmed_planets']} planets...")
            else:
                metadata["failed_downloads"] += 1

            time.sleep(0.5)  # Rate limiting

        # 2. False positives (label = 2: eclipsing_binary)
        print("\n[2/3] Downloading false positive light curves...")
        fps_df = fetch_false_positives(config.n_false_positives)

        for idx, row in fps_df.iterrows():
            target_id = str(row.get('kepid', ''))
            if not target_id or target_id == 'nan':
                continue

            flux = download_lightcurve(target_id, "Kepler", config.target_length)

            if flux is not None:
                features = extract_lightcurve_features(flux)
                X_list.append(features)
                y_list.append(2)  # eclipsing_binary
                metadata["false_positives"] += 1

                if metadata["false_positives"] % 10 == 0:
                    print(f"    Downloaded {metadata['false_positives']} false positives...")
            else:
                metadata["failed_downloads"] += 1

            time.sleep(0.5)

        # 3. Field stars (label = 0: no_transit)
        print("\n[3/3] Downloading field star light curves...")
        # Get random Kepler field stars (non-planet hosts)
        try:
            import lightkurve as lk

            # Search for random field stars from a Kepler quarter
            search_result = lk.search_lightcurve(
                "Quarter 1",
                mission="Kepler",
                author="Kepler"
            )

            collected = 0
            for result in search_result:
                if collected >= config.n_field_stars:
                    break

                try:
                    flux = download_lightcurve(
                        result.target_name,
                        "Kepler",
                        config.target_length
                    )

                    if flux is not None:
                        # Check it's not too variable (likely non-transit)
                        std = np.std(flux)
                        if std < 0.01:  # Quiet star
                            features = extract_lightcurve_features(flux)
                            X_list.append(features)
                            y_list.append(0)  # no_transit
                            metadata["field_stars"] += 1
                            collected += 1

                            if collected % 20 == 0:
                                print(f"    Downloaded {collected} field stars...")

                    time.sleep(0.3)

                except Exception:
                    continue

        except Exception as e:
            print(f"    Error collecting field stars: {e}")

    # If we don't have enough data, supplement with synthetic
    total_samples = len(X_list)
    target_samples = config.n_confirmed_planets + config.n_false_positives + config.n_field_stars

    if total_samples < target_samples * 0.5:
        print(f"\n  Only collected {total_samples} real samples, supplementing with synthetic...")
        X_synth, y_synth = _generate_synthetic_supplement(
            n_transit=max(0, config.n_confirmed_planets - metadata["confirmed_planets"]),
            n_eb=max(0, config.n_false_positives - metadata["false_positives"]),
            n_noise=max(0, config.n_field_stars - metadata["field_stars"])
        )
        X_list.extend(X_synth)
        y_list.extend(y_synth)
        metadata["synthetic_supplement"] = len(X_synth)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Save cache
    np.savez(cache_file, X=X, y=y)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n  Collection complete:")
    print(f"    Confirmed planets: {metadata['confirmed_planets']}")
    print(f"    False positives: {metadata['false_positives']}")
    print(f"    Field stars: {metadata['field_stars']}")
    print(f"    Failed downloads: {metadata['failed_downloads']}")
    print(f"    Total samples: {len(X)}")

    return X, y, metadata


def _generate_synthetic_supplement(n_transit: int, n_eb: int, n_noise: int) -> Tuple[List, List]:
    """Generate synthetic data to supplement real data."""
    from src.model.data_generators import ExoplanetDataGenerator

    generator = ExoplanetDataGenerator(n_points=1024)
    X_list = []
    y_list = []

    # Generate transits
    for _ in range(n_transit):
        lc = generator.generate_transit(
            period=np.random.uniform(0.1, 0.3),
            depth=np.random.uniform(0.001, 0.02),
            duration=np.random.uniform(0.01, 0.05)
        )
        lc += np.random.normal(0, 0.0005, len(lc))
        X_list.append(extract_lightcurve_features(lc))
        y_list.append(1)

    # Generate eclipsing binaries
    for _ in range(n_eb):
        lc = generator.generate_eclipsing_binary(
            period=np.random.uniform(0.05, 0.2),
            depth1=np.random.uniform(0.05, 0.3),
            depth2=np.random.uniform(0.02, 0.15)
        )
        lc += np.random.normal(0, 0.0005, len(lc))
        X_list.append(extract_lightcurve_features(lc))
        y_list.append(2)

    # Generate noise
    for _ in range(n_noise):
        lc = np.ones(1024) + np.random.normal(0, 0.0005, 1024)
        X_list.append(extract_lightcurve_features(lc))
        y_list.append(0)

    return X_list, y_list


# =============================================================================
# Training Pipeline
# =============================================================================

def train_on_real_data(config: RealDataConfig) -> Dict[str, Any]:
    """
    Train EXOPLANET-001 on real data.
    """
    print("\n" + "=" * 60)
    print("TRAINING ON REAL EXOPLANET DATA")
    print("=" * 60)

    # Collect data
    X, y, collection_metadata = collect_real_data(config)

    if len(X) < 100:
        print("ERROR: Not enough data collected for training")
        return {"success": False, "error": "Insufficient data"}

    # Compute normalization parameters
    norm_mean = X.mean(axis=0)
    norm_std = X.std(axis=0)
    X_normalized = (X - norm_mean) / (norm_std + 1e-8)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_normalized = X_normalized[indices]
    y = y[indices]

    n_val = int(len(X) * config.val_split)
    X_train, y_train = X_normalized[n_val:], y[n_val:]
    X_val, y_val = X_normalized[:n_val], y[:n_val]

    print(f"\n  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")

    # Class distribution
    for i in range(3):
        n_train = (y_train == i).sum()
        n_val = (y_val == i).sum()
        labels = ["no_transit", "transit", "eclipsing_binary"]
        print(f"    {labels[i]}: {n_train} train, {n_val} val")

    # Train model
    print(f"\n  Training for {config.epochs} epochs...")

    model = SimpleClassifier(
        n_features=X.shape[1],
        n_classes=3,
        hidden1=64,
        hidden2=32
    )

    best_acc = train_classifier(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=config.epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size
    )

    # Set normalization and labels
    model.set_normalization(norm_mean, norm_std)
    model.set_class_labels(["no_transit", "transit", "eclipsing_binary"])

    # Save model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "EXOPLANET-001_real_weights.npz"
    model.save(str(model_path))

    print(f"\n  Model saved to: {model_path}")
    print(f"  Best validation accuracy: {best_acc*100:.1f}%")

    # Validation metrics
    val_preds, val_conf, _ = model.predict_from_raw(X[:n_val])

    # Per-class accuracy
    print("\n  Per-class validation accuracy:")
    labels = ["no_transit", "transit", "eclipsing_binary"]
    for i in range(3):
        mask = y[:n_val] == i
        if mask.sum() > 0:
            acc = (val_preds[mask] == i).mean()
            print(f"    {labels[i]}: {acc*100:.1f}%")

    results = {
        "success": True,
        "accuracy": float(best_acc),
        "n_samples": len(X),
        "model_path": str(model_path),
        "collection_metadata": collection_metadata,
        "config": asdict(config)
    }

    # Save results
    results_path = output_dir / "real_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train EXOPLANET-001 on real Kepler/TESS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_real_exoplanets.py --download   # Download data only
    python train_real_exoplanets.py --train      # Train on cached data
    python train_real_exoplanets.py --full       # Download + Train

Requirements:
    pip install lightkurve astroquery pandas
        """
    )

    parser.add_argument('--download', action='store_true',
                        help='Download real data only')
    parser.add_argument('--train', action='store_true',
                        help='Train on cached data')
    parser.add_argument('--full', action='store_true',
                        help='Download and train')
    parser.add_argument('--n-planets', type=int, default=200,
                        help='Number of confirmed planets to download')
    parser.add_argument('--n-fps', type=int, default=200,
                        help='Number of false positives to download')
    parser.add_argument('--n-field', type=int, default=400,
                        help='Number of field stars to download')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached data before downloading')

    args = parser.parse_args()

    config = RealDataConfig(
        n_confirmed_planets=args.n_planets,
        n_false_positives=args.n_fps,
        n_field_stars=args.n_field,
        epochs=args.epochs
    )

    if args.clear_cache:
        cache_dir = Path(config.cache_dir)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")

    if args.download or args.full:
        X, y, metadata = collect_real_data(config)
        print(f"\nData collection complete: {len(X)} samples")

    if args.train or args.full:
        results = train_on_real_data(config)

        if results["success"]:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"  Accuracy: {results['accuracy']*100:.1f}%")
            print(f"  Model: {results['model_path']}")
        else:
            print(f"\nTraining failed: {results.get('error')}")

    if not (args.download or args.train or args.full):
        parser.print_help()


if __name__ == "__main__":
    main()

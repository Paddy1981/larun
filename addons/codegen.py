#!/usr/bin/env python3
"""
LARUN TinyML - Code Generation Addon
Developer tools for generating Python scripts and ML models

This addon is NOT part of the core LARUN TinyML package.
Load with: larun --addon codegen

Created by: Padmanaban Veeraragavalu (Larun Engineering)
With AI assistance from: Claude (Anthropic)

Project: LARUN TinyML × Astrodata
License: MIT
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# =============================================================================
# CODE TEMPLATES
# =============================================================================

TEMPLATES = {
    # -------------------------------------------------------------------------
    # DATA FETCH TEMPLATE
    # -------------------------------------------------------------------------
    "data_fetch": '''#!/usr/bin/env python3
"""
LARUN TinyML - Data Fetch Script
Generated: {timestamp}
Target: {target}

Fetch astronomical data from NASA archives.
"""

import numpy as np
from pathlib import Path

try:
    import lightkurve as lk
except ImportError:
    print("Installing lightkurve...")
    import subprocess
    subprocess.check_call(["pip", "install", "lightkurve"])
    import lightkurve as lk

# Configuration
TARGET = "{target}"
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_tess_data(target: str, limit: int = 10):
    """Fetch TESS light curves for a target."""
    print(f"Searching TESS for: {{target}}")

    search_result = lk.search_lightcurve(target, mission="TESS")

    if len(search_result) == 0:
        print(f"No TESS data found for {{target}}")
        return None

    print(f"Found {{len(search_result)}} light curves")

    # Download and stitch light curves
    lc_collection = search_result[:limit].download_all()
    lc = lc_collection.stitch()

    return lc


def fetch_kepler_data(target: str, limit: int = 10):
    """Fetch Kepler light curves for a target."""
    print(f"Searching Kepler for: {{target}}")

    search_result = lk.search_lightcurve(target, mission="Kepler")

    if len(search_result) == 0:
        print(f"No Kepler data found for {{target}}")
        return None

    print(f"Found {{len(search_result)}} light curves")

    lc_collection = search_result[:limit].download_all()
    lc = lc_collection.stitch()

    return lc


def save_lightcurve(lc, filename: str):
    """Save light curve to file."""
    output_path = OUTPUT_DIR / filename

    # Extract data
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else np.zeros_like(flux)

    # Save as NPZ
    np.savez(
        output_path,
        time=time,
        flux=flux,
        flux_err=flux_err,
        target=TARGET
    )

    print(f"Saved: {{output_path}}")
    return output_path


def main():
    """Main execution."""
    print("=" * 60)
    print("LARUN TinyML - Data Fetch")
    print("=" * 60)

    # Try TESS first
    lc = fetch_tess_data(TARGET)

    if lc is None:
        # Fall back to Kepler
        lc = fetch_kepler_data(TARGET)

    if lc is not None:
        # Process and save
        lc = lc.remove_nans().normalize()
        save_lightcurve(lc, f"{{TARGET.replace(' ', '_')}}_lightcurve.npz")

        print(f"\\nData points: {{len(lc.time)}}")
        print(f"Time range: {{lc.time.value.min():.2f}} - {{lc.time.value.max():.2f}}")
    else:
        print(f"No data found for {{TARGET}}")


if __name__ == "__main__":
    main()
''',

    # -------------------------------------------------------------------------
    # LIGHTCURVE ANALYSIS TEMPLATE
    # -------------------------------------------------------------------------
    "lightcurve_analysis": '''#!/usr/bin/env python3
"""
LARUN TinyML - Light Curve Analysis Script
Generated: {timestamp}

Comprehensive light curve analysis pipeline.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter

# Configuration
INPUT_FILE = "{input_file}"
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lightcurve(filepath: str):
    """Load light curve from NPZ file."""
    data = np.load(filepath)
    return data['time'], data['flux'], data.get('flux_err', np.zeros_like(data['flux']))


def remove_outliers(flux: np.ndarray, sigma: float = 3.0):
    """Remove outliers using sigma clipping."""
    median = np.median(flux)
    std = np.std(flux)
    mask = np.abs(flux - median) < sigma * std
    return mask


def detrend_lightcurve(time: np.ndarray, flux: np.ndarray, window: int = 101):
    """Detrend light curve using median filter."""
    trend = median_filter(flux, size=window)
    detrended = flux / trend
    return detrended, trend


def compute_periodogram(time: np.ndarray, flux: np.ndarray,
                        min_period: float = 0.5, max_period: float = 50.0):
    """Compute Lomb-Scargle periodogram."""
    # Create frequency grid
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    freqs = np.linspace(min_freq, max_freq, 10000)

    # Compute periodogram
    pgram = signal.lombscargle(time, flux - np.mean(flux), 2 * np.pi * freqs, normalize=True)
    periods = 1.0 / freqs

    return periods, pgram


def find_best_period(periods: np.ndarray, power: np.ndarray):
    """Find the period with maximum power."""
    best_idx = np.argmax(power)
    return periods[best_idx], power[best_idx]


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float = None):
    """Phase fold the light curve."""
    if epoch is None:
        epoch = time[0]

    phase = ((time - epoch) % period) / period

    # Sort by phase
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def plot_analysis(time, flux, detrended, periods, power, best_period, phase, phase_flux):
    """Create analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original light curve
    axes[0, 0].scatter(time, flux, s=1, alpha=0.5)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Flux')
    axes[0, 0].set_title('Original Light Curve')

    # Detrended light curve
    axes[0, 1].scatter(time, detrended, s=1, alpha=0.5)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Normalized Flux')
    axes[0, 1].set_title('Detrended Light Curve')

    # Periodogram
    axes[1, 0].plot(periods, power)
    axes[1, 0].axvline(best_period, color='r', linestyle='--',
                       label=f'Best period: {{best_period:.4f}} days')
    axes[1, 0].set_xlabel('Period (days)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_title('Lomb-Scargle Periodogram')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')

    # Phase-folded light curve
    axes[1, 1].scatter(phase, phase_flux, s=1, alpha=0.5)
    axes[1, 1].set_xlabel('Phase')
    axes[1, 1].set_ylabel('Normalized Flux')
    axes[1, 1].set_title(f'Phase-Folded (P = {{best_period:.4f}} days)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_results.png', dpi=150)
    plt.close()
    print(f"Saved: {{OUTPUT_DIR / 'analysis_results.png'}}")


def main():
    """Main execution."""
    print("=" * 60)
    print("LARUN TinyML - Light Curve Analysis")
    print("=" * 60)

    # Load data
    print(f"\\nLoading: {{INPUT_FILE}}")
    time, flux, flux_err = load_lightcurve(INPUT_FILE)
    print(f"Data points: {{len(time)}}")

    # Remove outliers
    mask = remove_outliers(flux)
    time, flux = time[mask], flux[mask]
    print(f"After outlier removal: {{len(time)}} points")

    # Detrend
    detrended, trend = detrend_lightcurve(time, flux)

    # Compute periodogram
    print("\\nComputing periodogram...")
    periods, power = compute_periodogram(time, detrended)
    best_period, best_power = find_best_period(periods, power)
    print(f"Best period: {{best_period:.6f}} days (power: {{best_power:.4f}})")

    # Phase fold
    phase, phase_flux = phase_fold(time, detrended, best_period)

    # Create plots
    plot_analysis(time, flux, detrended, periods, power, best_period, phase, phase_flux)

    # Save results
    results = {{
        'best_period': best_period,
        'best_power': best_power,
        'n_points': len(time),
        'time_span': time.max() - time.min()
    }}

    np.savez(OUTPUT_DIR / 'analysis_results.npz', **results)
    print(f"\\nAnalysis complete!")


if __name__ == "__main__":
    main()
''',

    # -------------------------------------------------------------------------
    # TRANSIT SEARCH TEMPLATE
    # -------------------------------------------------------------------------
    "transit_search": '''#!/usr/bin/env python3
"""
LARUN TinyML - Transit Search Script
Generated: {timestamp}

Search for planetary transit signals in light curves.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Configuration
INPUT_FILE = "{input_file}"
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Transit search parameters
MIN_DEPTH = 0.0001  # Minimum transit depth (0.01%)
MIN_DURATION = 0.5   # Minimum duration (hours)
MIN_SNR = 7.0        # Minimum SNR for detection


def load_lightcurve(filepath: str):
    """Load light curve from NPZ file."""
    data = np.load(filepath)
    return data['time'], data['flux']


def compute_running_statistics(flux: np.ndarray, window: int = 50):
    """Compute running mean and std for transit detection."""
    running_mean = uniform_filter1d(flux, window)
    running_std = np.sqrt(uniform_filter1d((flux - running_mean)**2, window))
    return running_mean, running_std


def find_transit_candidates(time: np.ndarray, flux: np.ndarray,
                            running_mean: np.ndarray, running_std: np.ndarray):
    """Find potential transit events."""
    candidates = []

    # Z-score for flux dips
    z_score = (flux - running_mean) / (running_std + 1e-10)

    # Find significant dips
    dip_threshold = -3.0  # 3-sigma dips
    dip_mask = z_score < dip_threshold

    # Group consecutive dips into events
    in_transit = False
    transit_start = 0

    for i in range(len(dip_mask)):
        if dip_mask[i] and not in_transit:
            in_transit = True
            transit_start = i
        elif not dip_mask[i] and in_transit:
            in_transit = False
            transit_end = i

            # Calculate transit properties
            transit_time = time[transit_start:transit_end]
            transit_flux = flux[transit_start:transit_end]

            if len(transit_time) > 3:  # At least 3 points
                duration = (transit_time[-1] - transit_time[0]) * 24  # hours
                depth = 1.0 - np.min(transit_flux) / np.median(flux)

                # Estimate SNR
                noise = np.std(flux)
                snr = depth / noise * np.sqrt(len(transit_flux))

                if duration >= MIN_DURATION and depth >= MIN_DEPTH and snr >= MIN_SNR:
                    candidates.append({{
                        'start_time': transit_time[0],
                        'end_time': transit_time[-1],
                        'mid_time': np.mean(transit_time),
                        'duration_hours': duration,
                        'depth': depth,
                        'snr': snr,
                        'n_points': len(transit_flux)
                    }})

    return candidates


def estimate_period(candidates: list, time: np.ndarray):
    """Estimate orbital period from transit candidates."""
    if len(candidates) < 2:
        return None

    # Calculate time differences between consecutive transits
    mid_times = sorted([c['mid_time'] for c in candidates])
    diffs = np.diff(mid_times)

    if len(diffs) == 0:
        return None

    # Use most common difference as period estimate
    # For more robust estimation, use BLS periodogram
    period = np.median(diffs)

    return period


def plot_transit_search(time, flux, candidates, period):
    """Plot transit search results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full light curve with transits marked
    axes[0].scatter(time, flux, s=1, alpha=0.5, label='Data')

    for i, c in enumerate(candidates):
        axes[0].axvline(c['mid_time'], color='r', alpha=0.5, linestyle='--')
        if i == 0:
            axes[0].axvline(c['mid_time'], color='r', alpha=0.5,
                           linestyle='--', label='Transit candidates')

    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Flux')
    axes[0].set_title(f'Transit Search Results - Found {{len(candidates)}} candidates')
    axes[0].legend()

    # Transit depth histogram
    if candidates:
        depths = [c['depth'] * 100 for c in candidates]  # Convert to percent
        axes[1].hist(depths, bins=20, edgecolor='black')
        axes[1].set_xlabel('Transit Depth (%)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Transit Depth Distribution')

        if period:
            axes[1].text(0.95, 0.95, f'Est. Period: {{period:.4f}} days',
                        transform=axes[1].transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'transit_search_results.png', dpi=150)
    plt.close()
    print(f"Saved: {{OUTPUT_DIR / 'transit_search_results.png'}}")


def main():
    """Main execution."""
    print("=" * 60)
    print("LARUN TinyML - Transit Search")
    print("=" * 60)

    # Load data
    print(f"\\nLoading: {{INPUT_FILE}}")
    time, flux = load_lightcurve(INPUT_FILE)
    print(f"Data points: {{len(time)}}")

    # Compute running statistics
    running_mean, running_std = compute_running_statistics(flux)

    # Find transit candidates
    print("\\nSearching for transits...")
    candidates = find_transit_candidates(time, flux, running_mean, running_std)
    print(f"Found {{len(candidates)}} transit candidates")

    # Estimate period
    period = estimate_period(candidates, time)
    if period:
        print(f"Estimated orbital period: {{period:.4f}} days")

    # Print candidate details
    if candidates:
        print("\\nTransit Candidates:")
        print("-" * 70)
        for i, c in enumerate(candidates):
            print(f"  {{i+1}}. Time: {{c['mid_time']:.4f}} | "
                  f"Depth: {{c['depth']*100:.4f}}% | "
                  f"Duration: {{c['duration_hours']:.2f}}h | "
                  f"SNR: {{c['snr']:.1f}}")

    # Plot results
    plot_transit_search(time, flux, candidates, period)

    # Save results
    import json
    results = {{
        'n_candidates': len(candidates),
        'estimated_period': period,
        'candidates': candidates
    }}

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(OUTPUT_DIR / 'transit_candidates.json', 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\\nResults saved to {{OUTPUT_DIR}}")


if __name__ == "__main__":
    main()
''',

    # -------------------------------------------------------------------------
    # ANOMALY DETECTION TEMPLATE
    # -------------------------------------------------------------------------
    "anomaly_detection": '''#!/usr/bin/env python3
"""
LARUN TinyML - Anomaly Detection Script
Generated: {timestamp}

Detect unusual patterns in astronomical light curves.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import median_filter

# Configuration
INPUT_FILE = "{input_file}"
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Anomaly detection parameters
SENSITIVITY = 0.95  # Detection sensitivity (higher = more anomalies)


class AnomalyDetector:
    """Statistical anomaly detector for light curves."""

    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.threshold = stats.norm.ppf((1 + sensitivity) / 2)

    def detect_statistical(self, flux: np.ndarray):
        """Detect anomalies using statistical methods."""
        # Compute robust statistics
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        robust_std = 1.4826 * mad  # Scale MAD to std

        # Z-scores
        z_scores = np.abs(flux - median) / (robust_std + 1e-10)

        # Anomaly mask
        anomaly_mask = z_scores > self.threshold

        return anomaly_mask, z_scores

    def detect_local_outliers(self, flux: np.ndarray, window: int = 21):
        """Detect local outliers using running median."""
        running_median = median_filter(flux, size=window)
        residuals = flux - running_median

        # Robust std of residuals
        mad = np.median(np.abs(residuals))
        robust_std = 1.4826 * mad

        z_scores = np.abs(residuals) / (robust_std + 1e-10)
        anomaly_mask = z_scores > self.threshold

        return anomaly_mask, z_scores

    def classify_anomaly(self, time: np.ndarray, flux: np.ndarray, idx: int, window: int = 10):
        """Classify the type of anomaly."""
        start = max(0, idx - window)
        end = min(len(flux), idx + window + 1)

        local_flux = flux[start:end]
        local_time = time[start:end]

        center_flux = flux[idx]
        median_local = np.median(local_flux)

        # Classification logic
        if center_flux < median_local:
            # Dip - could be transit, eclipse, or instrumental
            depth = 1 - center_flux / median_local
            if depth > 0.01:  # > 1% depth
                return "deep_dip"
            else:
                return "shallow_dip"
        else:
            # Brightening - could be flare, artifact
            amplitude = center_flux / median_local - 1
            if amplitude > 0.1:  # > 10% brightening
                return "flare"
            else:
                return "brightening"


def load_lightcurve(filepath: str):
    """Load light curve from NPZ file."""
    data = np.load(filepath)
    return data['time'], data['flux']


def plot_anomalies(time, flux, anomaly_mask, z_scores, anomaly_types):
    """Plot anomaly detection results."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Light curve with anomalies highlighted
    axes[0].scatter(time[~anomaly_mask], flux[~anomaly_mask], s=1, alpha=0.5, label='Normal')
    axes[0].scatter(time[anomaly_mask], flux[anomaly_mask], s=10, c='red', label='Anomaly')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Flux')
    axes[0].set_title(f'Anomaly Detection - Found {{np.sum(anomaly_mask)}} anomalies')
    axes[0].legend()

    # Z-scores
    axes[1].plot(time, z_scores, alpha=0.7)
    axes[1].axhline(y=stats.norm.ppf((1 + SENSITIVITY) / 2), color='r', linestyle='--',
                    label=f'Threshold ({{SENSITIVITY*100:.0f}}%)')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Z-score')
    axes[1].set_title('Anomaly Z-scores')
    axes[1].legend()

    # Anomaly type distribution
    if anomaly_types:
        types = list(anomaly_types.values())
        unique_types, counts = np.unique(types, return_counts=True)
        axes[2].bar(unique_types, counts, edgecolor='black')
        axes[2].set_xlabel('Anomaly Type')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Anomaly Classification')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'anomaly_detection_results.png', dpi=150)
    plt.close()
    print(f"Saved: {{OUTPUT_DIR / 'anomaly_detection_results.png'}}")


def main():
    """Main execution."""
    print("=" * 60)
    print("LARUN TinyML - Anomaly Detection")
    print("=" * 60)

    # Load data
    print(f"\\nLoading: {{INPUT_FILE}}")
    time, flux = load_lightcurve(INPUT_FILE)
    print(f"Data points: {{len(time)}}")

    # Initialize detector
    detector = AnomalyDetector(sensitivity=SENSITIVITY)

    # Detect anomalies using both methods
    print("\\nDetecting anomalies...")
    global_mask, global_z = detector.detect_statistical(flux)
    local_mask, local_z = detector.detect_local_outliers(flux)

    # Combine masks
    anomaly_mask = global_mask | local_mask
    z_scores = np.maximum(global_z, local_z)

    print(f"Global anomalies: {{np.sum(global_mask)}}")
    print(f"Local anomalies: {{np.sum(local_mask)}}")
    print(f"Total unique anomalies: {{np.sum(anomaly_mask)}}")

    # Classify anomalies
    anomaly_indices = np.where(anomaly_mask)[0]
    anomaly_types = {{}}

    for idx in anomaly_indices:
        anomaly_type = detector.classify_anomaly(time, flux, idx)
        anomaly_types[int(idx)] = anomaly_type

    # Print classification summary
    print("\\nAnomaly Classification:")
    for atype in set(anomaly_types.values()):
        count = sum(1 for t in anomaly_types.values() if t == atype)
        print(f"  {{atype}}: {{count}}")

    # Plot results
    plot_anomalies(time, flux, anomaly_mask, z_scores, anomaly_types)

    # Save results
    import json
    results = {{
        'n_anomalies': int(np.sum(anomaly_mask)),
        'sensitivity': SENSITIVITY,
        'anomaly_indices': anomaly_indices.tolist(),
        'anomaly_times': time[anomaly_mask].tolist(),
        'anomaly_types': anomaly_types
    }}

    with open(OUTPUT_DIR / 'anomaly_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to {{OUTPUT_DIR}}")


if __name__ == "__main__":
    main()
''',

    # -------------------------------------------------------------------------
    # REPORT GENERATION TEMPLATE
    # -------------------------------------------------------------------------
    "report_generation": '''#!/usr/bin/env python3
"""
LARUN TinyML - Report Generation Script
Generated: {timestamp}

Generate NASA-compatible candidate reports.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_html_report(candidate_data: dict, target: str = "Unknown"):
    """Generate HTML report for a candidate."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LARUN TinyML - Candidate Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px;
                     border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a237e; border-bottom: 2px solid #3f51b5; padding-bottom: 10px; }}
        h2 {{ color: #303f9f; margin-top: 30px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; }}
        .logo {{ font-size: 24px; font-weight: bold; color: #3f51b5; }}
        .meta {{ color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #e8eaf6; color: #1a237e; }}
        tr:hover {{ background: #f5f5f5; }}
        .highlight {{ background: #fff3e0; font-weight: bold; }}
        .status {{ padding: 5px 10px; border-radius: 5px; font-size: 12px; }}
        .status-candidate {{ background: #c8e6c9; color: #2e7d32; }}
        .status-confirmed {{ background: #bbdefb; color: #1565c0; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;
                  color: #666; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">LARUN TinyML</div>
            <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <h1>Exoplanet Candidate Report</h1>

        <h2>Target Information</h2>
        <table>
            <tr><th>Target</th><td>{target}</td></tr>
            <tr><th>Status</th><td><span class="status status-candidate">CANDIDATE</span></td></tr>
            <tr><th>Analysis Date</th><td>{datetime.now().strftime('%Y-%m-%d')}</td></tr>
        </table>

        <h2>Detection Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
            <tr><td>Number of Transits</td><td>{candidate_data.get('n_transits', 'N/A')}</td><td>-</td></tr>
            <tr class="highlight"><td>Orbital Period</td><td>{candidate_data.get('period', 'N/A')}</td><td>days</td></tr>
            <tr><td>Transit Depth</td><td>{candidate_data.get('depth', 'N/A')}</td><td>%</td></tr>
            <tr><td>Transit Duration</td><td>{candidate_data.get('duration', 'N/A')}</td><td>hours</td></tr>
            <tr><td>Signal-to-Noise Ratio</td><td>{candidate_data.get('snr', 'N/A')}</td><td>-</td></tr>
        </table>

        <h2>Data Quality</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Data Points</td><td>{candidate_data.get('n_points', 'N/A')}</td></tr>
            <tr><td>Time Span</td><td>{candidate_data.get('time_span', 'N/A')} days</td></tr>
            <tr><td>Data Source</td><td>{candidate_data.get('source', 'TESS/Kepler')}</td></tr>
        </table>

        <h2>Analysis Notes</h2>
        <p>{candidate_data.get('notes', 'No additional notes.')}</p>

        <h2>Recommendations</h2>
        <ul>
            <li>Verify with independent analysis pipeline</li>
            <li>Check for background eclipsing binaries</li>
            <li>Review centroid motion during transits</li>
            <li>Submit to ExoFOP for community vetting</li>
        </ul>

        <div class="footer">
            <p>LARUN TinyML - Astronomical Data Analysis</p>
            <p>Larun. × Astrodata | MIT License</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_json_report(candidate_data: dict, target: str = "Unknown"):
    """Generate JSON report for a candidate."""
    report = {{
        "larun_version": "2.0.0",
        "report_type": "exoplanet_candidate",
        "generated": datetime.now().isoformat(),
        "target": target,
        "status": "candidate",
        "detection": candidate_data,
        "flags": [],
        "recommendations": [
            "Verify with independent analysis",
            "Check for background binaries",
            "Submit to ExoFOP"
        ]
    }}
    return report


def main():
    """Main execution."""
    print("=" * 60)
    print("LARUN TinyML - Report Generation")
    print("=" * 60)

    # Example candidate data (replace with actual data)
    candidate_data = {{
        "n_transits": 5,
        "period": 3.14159,
        "depth": 0.15,
        "duration": 2.5,
        "snr": 12.3,
        "n_points": 15000,
        "time_span": 27.4,
        "source": "TESS",
        "notes": "Clear transit signal with consistent depth across all events."
    }}

    target = "{target}"

    # Generate HTML report
    html_report = generate_html_report(candidate_data, target)
    html_path = OUTPUT_DIR / f"{{target.replace(' ', '_')}}_report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    print(f"\\nHTML report saved: {{html_path}}")

    # Generate JSON report
    json_report = generate_json_report(candidate_data, target)
    json_path = OUTPUT_DIR / f"{{target.replace(' ', '_')}}_report.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved: {{json_path}}")

    print("\\nReport generation complete!")


if __name__ == "__main__":
    main()
'''
}

# =============================================================================
# ML MODEL TEMPLATES
# =============================================================================

ML_MODEL_TEMPLATES = {
    "cnn_1d": '''#!/usr/bin/env python3
"""
LARUN TinyML - 1D CNN Model
Generated: {timestamp}

1D Convolutional Neural Network for light curve classification.
Architecture: {architecture}
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
INPUT_SHAPE = {input_shape}
NUM_CLASSES = {num_classes}
MODEL_NAME = "{model_name}"


def create_cnn_1d_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Create a 1D CNN model for spectral/light curve classification."""

    inputs = keras.Input(shape=input_shape, name="input_lightcurve")

    # First convolutional block
    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Second convolutional block
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Third convolutional block
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)

    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001):
    """Compile the model with optimizer and loss function."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def export_to_tflite(model: keras.Model, output_path: str, quantize: bool = True):
    """Export model to TensorFlow Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model exported to: {{output_path}}")
    print(f"Size: {{len(tflite_model) / 1024:.2f}} KB")


# Class labels
CLASS_NAMES = [
    "noise",
    "stellar_signal",
    "planetary_transit",
    "eclipsing_binary",
    "instrument_artifact",
    "unknown_anomaly"
]


if __name__ == "__main__":
    print("=" * 60)
    print("LARUN TinyML - 1D CNN Model")
    print("=" * 60)

    # Create model
    model = create_cnn_1d_model(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)

    # Print summary
    print("\\nModel Summary:")
    model.summary()

    # Save model architecture
    keras.utils.plot_model(
        model,
        to_file="model_architecture.png",
        show_shapes=True,
        show_layer_names=True
    )
    print("\\nArchitecture saved to: model_architecture.png")
''',

    "lstm": '''#!/usr/bin/env python3
"""
LARUN TinyML - LSTM Model
Generated: {timestamp}

LSTM Neural Network for time-series light curve analysis.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
INPUT_SHAPE = {input_shape}
NUM_CLASSES = {num_classes}
MODEL_NAME = "{model_name}"


def create_lstm_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Create an LSTM model for time-series analysis."""

    inputs = keras.Input(shape=input_shape, name="input_sequence")

    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Dense layers
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)

    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001):
    """Compile the model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("LARUN TinyML - LSTM Model")
    print("=" * 60)

    model = create_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)
    model.summary()
''',

    "autoencoder": '''#!/usr/bin/env python3
"""
LARUN TinyML - Autoencoder Model
Generated: {timestamp}

Autoencoder for anomaly detection in light curves.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
INPUT_SHAPE = {input_shape}
LATENT_DIM = 16
MODEL_NAME = "{model_name}"


def create_autoencoder(input_shape: tuple, latent_dim: int = 16) -> tuple:
    """Create an autoencoder for anomaly detection."""

    # Encoder
    encoder_inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(encoder_inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(16, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    encoder = keras.Model(encoder_inputs, latent, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(input_shape[0] // 4 * 16, activation="relu")(latent_inputs)
    x = layers.Reshape((input_shape[0] // 4, 16))(x)
    x = layers.Conv1DTranspose(16, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv1DTranspose(32, 7, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = layers.Conv1D(input_shape[1], 3, padding="same", activation="linear")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Full autoencoder
    autoencoder_inputs = keras.Input(shape=input_shape)
    encoded = encoder(autoencoder_inputs)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_inputs, decoded, name=MODEL_NAME)

    return encoder, decoder, autoencoder


def compile_autoencoder(model: keras.Model, learning_rate: float = 0.001):
    """Compile autoencoder with reconstruction loss."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def compute_anomaly_score(model: keras.Model, data: np.ndarray) -> np.ndarray:
    """Compute anomaly scores as reconstruction error."""
    reconstructed = model.predict(data)
    mse = np.mean(np.square(data - reconstructed), axis=(1, 2))
    return mse


if __name__ == "__main__":
    print("=" * 60)
    print("LARUN TinyML - Autoencoder Model")
    print("=" * 60)

    encoder, decoder, autoencoder = create_autoencoder(INPUT_SHAPE, LATENT_DIM)
    autoencoder = compile_autoencoder(autoencoder)

    print("\\nEncoder Summary:")
    encoder.summary()

    print("\\nDecoder Summary:")
    decoder.summary()

    print("\\nAutoencoder Summary:")
    autoencoder.summary()
''',

    "transformer": '''#!/usr/bin/env python3
"""
LARUN TinyML - Transformer Model
Generated: {timestamp}

Transformer architecture for sequence modeling of light curves.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
INPUT_SHAPE = {input_shape}
NUM_CLASSES = {num_classes}
MODEL_NAME = "{model_name}"


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_model(input_shape: tuple, num_classes: int,
                            embed_dim: int = 32, num_heads: int = 2,
                            ff_dim: int = 32, num_blocks: int = 2) -> keras.Model:
    """Create a Transformer model for light curve classification."""

    inputs = keras.Input(shape=input_shape)

    # Initial projection
    x = layers.Dense(embed_dim)(inputs)

    # Transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)


def compile_model(model: keras.Model, learning_rate: float = 0.001):
    """Compile the model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("LARUN TinyML - Transformer Model")
    print("=" * 60)

    model = create_transformer_model(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)
    model.summary()
''',

    "hybrid": '''#!/usr/bin/env python3
"""
LARUN TinyML - Hybrid CNN-LSTM Model
Generated: {timestamp}

Hybrid architecture combining CNN and LSTM for light curve analysis.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
INPUT_SHAPE = {input_shape}
NUM_CLASSES = {num_classes}
MODEL_NAME = "{model_name}"


def create_hybrid_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Create a hybrid CNN-LSTM model."""

    inputs = keras.Input(shape=input_shape)

    # CNN feature extraction
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # LSTM sequence modeling
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(16, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Classification head
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)


def compile_model(model: keras.Model, learning_rate: float = 0.001):
    """Compile the model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("LARUN TinyML - Hybrid CNN-LSTM Model")
    print("=" * 60)

    model = create_hybrid_model(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)
    model.summary()
'''
}


# =============================================================================
# CODE GENERATOR CLASS
# =============================================================================

class CodeGenerator:
    """Generate Python scripts and ML models for LARUN TinyML."""

    def __init__(self, output_dir: Path = Path("./generated")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_script(self, task: str, target: str = "TIC 12345678",
                       output_file: Optional[str] = None) -> Path:
        """Generate a Python script for a specific task."""

        if task not in TEMPLATES:
            raise ValueError(f"Unknown task: {task}. Available: {list(TEMPLATES.keys())}")

        template = TEMPLATES[task]

        # Format template with parameters
        code = template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            target=target,
            input_file="./data/lightcurve.npz",
            output_dir="./output"
        )

        # Determine output filename
        if output_file is None:
            output_file = f"{task}_script.py"

        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            f.write(code)

        return output_path

    def generate_ml_model(self, architecture: str, input_shape: tuple,
                         num_classes: int = 6, model_name: str = "larun_model",
                         output_file: Optional[str] = None) -> Path:
        """Generate an ML model definition."""

        if architecture not in ML_MODEL_TEMPLATES:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: {list(ML_MODEL_TEMPLATES.keys())}")

        template = ML_MODEL_TEMPLATES[architecture]

        # Format template
        code = template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            architecture=architecture,
            input_shape=input_shape,
            num_classes=num_classes,
            model_name=model_name
        )

        # Determine output filename
        if output_file is None:
            output_file = f"{architecture}_model.py"

        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            f.write(code)

        return output_path

    def generate_training_script(self, model_type: str, data_source: str = "synthetic",
                                features: List[str] = None) -> Path:
        """Generate a complete training script."""

        if features is None:
            features = ["early_stopping", "checkpointing"]

        code = f'''#!/usr/bin/env python3
"""
LARUN TinyML - Training Script
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Type: {model_type}
Data Source: {data_source}
Features: {', '.join(features)}
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json

# Configuration
MODEL_TYPE = "{model_type}"
DATA_SOURCE = "{data_source}"
OUTPUT_DIR = Path("./models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def load_data():
    """Load training data."""
    if DATA_SOURCE == "synthetic":
        # Generate synthetic data
        n_samples = 1000
        seq_length = 200
        n_classes = 6

        X = np.random.randn(n_samples, seq_length, 1).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)

        # Add some structure to classes
        for i in range(n_samples):
            if y[i] == 2:  # Transit
                transit_start = np.random.randint(50, 150)
                X[i, transit_start:transit_start+20, 0] -= 0.1
            elif y[i] == 1:  # Stellar
                X[i, :, 0] += 0.05 * np.sin(np.linspace(0, 4*np.pi, seq_length))

        return X, y
    else:
        # Load from file
        data = np.load("./data/training_data.npz")
        return data['X'], data['y']


def create_callbacks():
    """Create training callbacks."""
    callbacks = []

'''

        if "early_stopping" in features:
            code += '''    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ))

'''

        if "checkpointing" in features:
            code += '''    # Model checkpointing
    callbacks.append(keras.callbacks.ModelCheckpoint(
        OUTPUT_DIR / 'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ))

'''

        if "tensorboard" in features:
            code += '''    # TensorBoard logging
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    ))

'''

        if "lr_scheduler" in features:
            code += '''    # Learning rate scheduler
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ))

'''

        code += '''    return callbacks


def build_model(input_shape, num_classes):
    """Build the model."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(32, 7, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("=" * 60)
    print("LARUN TinyML - Model Training")
    print("=" * 60)

    # Load data
    print("\\nLoading data...")
    X, y = load_data()
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")

    # Build model
    print("\\nBuilding model...")
    model = build_model(X.shape[1:], len(np.unique(y)))
    model.summary()

    # Create callbacks
    callbacks = create_callbacks()

    # Train
    print("\\nTraining...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(OUTPUT_DIR / 'final_model.h5')
    print(f"\\nModel saved to {OUTPUT_DIR / 'final_model.h5'}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\\nTraining complete!")


if __name__ == "__main__":
    main()
'''

        output_path = self.output_dir / "training_script.py"
        with open(output_path, 'w') as f:
            f.write(code)

        return output_path

    def generate_pipeline(self, source: str, steps: List[str]) -> Path:
        """Generate a complete analysis pipeline."""

        code = f'''#!/usr/bin/env python3
"""
LARUN TinyML - Analysis Pipeline
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Source: {source}
Steps: {' -> '.join(steps)}
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Pipeline Configuration
SOURCE = "{source}"
STEPS = {steps}
OUTPUT_DIR = Path("./pipeline_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PipelineStep:
    """Base class for pipeline steps."""

    def __init__(self, name: str):
        self.name = name

    def run(self, data: dict) -> dict:
        raise NotImplementedError


'''

        # Add step implementations based on selected steps
        if "fetch" in steps:
            code += '''
class FetchStep(PipelineStep):
    """Fetch data from NASA archives."""

    def __init__(self):
        super().__init__("fetch")

    def run(self, data: dict) -> dict:
        print(f"[{self.name}] Fetching data from {SOURCE}...")

        try:
            import lightkurve as lk

            target = data.get('target', 'TIC 12345678')
            search = lk.search_lightcurve(target, mission=SOURCE.upper())

            if len(search) > 0:
                lc = search[0].download()
                data['time'] = lc.time.value
                data['flux'] = lc.flux.value
                data['source'] = SOURCE
                print(f"[{self.name}] Fetched {len(data['time'])} data points")
            else:
                print(f"[{self.name}] No data found, using synthetic")
                data['time'] = np.linspace(0, 27.4, 1000)
                data['flux'] = 1.0 + 0.001 * np.random.randn(1000)
                data['source'] = 'synthetic'
        except ImportError:
            print(f"[{self.name}] lightkurve not installed, using synthetic data")
            data['time'] = np.linspace(0, 27.4, 1000)
            data['flux'] = 1.0 + 0.001 * np.random.randn(1000)
            data['source'] = 'synthetic'

        return data


'''

        if "clean" in steps:
            code += '''
class CleanStep(PipelineStep):
    """Clean the light curve data."""

    def __init__(self):
        super().__init__("clean")

    def run(self, data: dict) -> dict:
        print(f"[{self.name}] Cleaning data...")

        flux = data['flux']
        time = data['time']

        # Remove NaNs
        mask = ~np.isnan(flux)
        data['flux'] = flux[mask]
        data['time'] = time[mask]

        # Remove outliers (sigma clipping)
        median = np.median(data['flux'])
        std = np.std(data['flux'])
        mask = np.abs(data['flux'] - median) < 3 * std
        data['flux'] = data['flux'][mask]
        data['time'] = data['time'][mask]

        print(f"[{self.name}] Remaining points: {len(data['flux'])}")
        return data


'''

        if "normalize" in steps:
            code += '''
class NormalizeStep(PipelineStep):
    """Normalize the light curve."""

    def __init__(self):
        super().__init__("normalize")

    def run(self, data: dict) -> dict:
        print(f"[{self.name}] Normalizing...")

        median = np.median(data['flux'])
        data['flux'] = data['flux'] / median
        data['normalized'] = True

        print(f"[{self.name}] Flux range: {data['flux'].min():.4f} - {data['flux'].max():.4f}")
        return data


'''

        if "detect" in steps:
            code += '''
class DetectStep(PipelineStep):
    """Detect transit signals."""

    def __init__(self):
        super().__init__("detect")

    def run(self, data: dict) -> dict:
        print(f"[{self.name}] Running transit detection...")

        flux = data['flux']
        time = data['time']

        # Simple dip detection
        threshold = np.median(flux) - 3 * np.std(flux)
        dips = flux < threshold

        data['detections'] = []
        in_dip = False
        dip_start = 0

        for i in range(len(dips)):
            if dips[i] and not in_dip:
                in_dip = True
                dip_start = i
            elif not dips[i] and in_dip:
                in_dip = False
                if i - dip_start > 3:
                    data['detections'].append({
                        'start_time': float(time[dip_start]),
                        'end_time': float(time[i]),
                        'depth': float(1 - np.min(flux[dip_start:i]))
                    })

        print(f"[{self.name}] Found {len(data['detections'])} potential transits")
        return data


'''

        if "report" in steps:
            code += '''
class ReportStep(PipelineStep):
    """Generate analysis report."""

    def __init__(self):
        super().__init__("report")

    def run(self, data: dict) -> dict:
        print(f"[{self.name}] Generating report...")

        report = {
            'generated': datetime.now().isoformat(),
            'source': data.get('source', 'unknown'),
            'n_points': len(data.get('flux', [])),
            'n_detections': len(data.get('detections', [])),
            'detections': data.get('detections', [])
        }

        report_path = OUTPUT_DIR / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[{self.name}] Report saved to {report_path}")
        data['report'] = report
        return data


'''

        # Build pipeline runner
        code += '''
def build_pipeline(steps: list) -> list:
    """Build pipeline from step names."""
    step_classes = {
'''

        if "fetch" in steps:
            code += "        'fetch': FetchStep,\n"
        if "clean" in steps:
            code += "        'clean': CleanStep,\n"
        if "normalize" in steps:
            code += "        'normalize': NormalizeStep,\n"
        if "detect" in steps:
            code += "        'detect': DetectStep,\n"
        if "report" in steps:
            code += "        'report': ReportStep,\n"

        code += '''    }

    pipeline = []
    for step_name in steps:
        if step_name in step_classes:
            pipeline.append(step_classes[step_name]())

    return pipeline


def run_pipeline(pipeline: list, initial_data: dict = None) -> dict:
    """Execute the pipeline."""
    data = initial_data or {}

    print("=" * 60)
    print("LARUN TinyML - Pipeline Execution")
    print("=" * 60)

    for step in pipeline:
        print(f"\\n>>> Running step: {step.name}")
        data = step.run(data)

    print("\\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return data


def main():
    # Build and run pipeline
    pipeline = build_pipeline(STEPS)
    results = run_pipeline(pipeline, {'target': 'TIC 307210830'})

    # Save final results
    np.savez(
        OUTPUT_DIR / 'pipeline_results.npz',
        time=results.get('time', []),
        flux=results.get('flux', [])
    )
    print(f"\\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
'''

        output_path = self.output_dir / "analysis_pipeline.py"
        with open(output_path, 'w') as f:
            f.write(code)

        return output_path


# =============================================================================
# ADDON INTERFACE
# =============================================================================

def get_addon_info() -> dict:
    """Return addon metadata."""
    return {
        "name": "Code Generation",
        "id": "codegen",
        "version": "1.0.0",
        "description": "Generate Python scripts and ML models programmatically",
        "skills": ["CODE-001", "CODE-002", "CODE-003", "CODE-004", "CODE-005"],
        "commands": ["generate"]
    }


def create_generator(output_dir: str = "./generated") -> CodeGenerator:
    """Create a code generator instance."""
    return CodeGenerator(Path(output_dir))


# =============================================================================
# CLI INTERFACE (for standalone use)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LARUN TinyML Code Generator")
    parser.add_argument("command", choices=["script", "model", "training", "pipeline"],
                       help="What to generate")
    parser.add_argument("--type", "-t", help="Task or architecture type")
    parser.add_argument("--target", default="TIC 12345678", help="Target star")
    parser.add_argument("--output", "-o", default="./generated", help="Output directory")
    parser.add_argument("--input-shape", default="200,1", help="Input shape (e.g., 200,1)")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--steps", nargs="+", default=["fetch", "clean", "normalize", "detect", "report"],
                       help="Pipeline steps")

    args = parser.parse_args()

    generator = CodeGenerator(Path(args.output))

    if args.command == "script":
        task = args.type or "data_fetch"
        path = generator.generate_script(task, args.target)
        print(f"Generated: {path}")

    elif args.command == "model":
        arch = args.type or "cnn_1d"
        shape = tuple(int(x) for x in args.input_shape.split(","))
        path = generator.generate_ml_model(arch, shape, args.num_classes)
        print(f"Generated: {path}")

    elif args.command == "training":
        model_type = args.type or "cnn"
        path = generator.generate_training_script(model_type)
        print(f"Generated: {path}")

    elif args.command == "pipeline":
        source = args.type or "tess"
        path = generator.generate_pipeline(source, args.steps)
        print(f"Generated: {path}")

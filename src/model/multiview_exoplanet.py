"""
Multi-View Exoplanet Detection Model
=====================================
Implements a multi-view CNN architecture for exoplanet detection,
similar to ExoMiner, using global, local, and secondary eclipse views.

This architecture targets 95%+ AUC on real Kepler/TESS data.

Views:
- Global View: Full phase-folded light curve (2001 points)
- Local View: Zoomed transit region (201 points)
- Secondary View: Secondary eclipse region (201 points)

Reference: ExoMiner (Valizadegan et al., 2022)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class MultiViewConfig:
    """Configuration for multi-view model."""
    global_view_size: int = 2001  # Full orbital phase
    local_view_size: int = 201   # Zoomed transit
    secondary_view_size: int = 201  # Secondary eclipse
    num_classes: int = 2  # Binary: planet vs non-planet

    # Architecture
    conv_filters: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.3

    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [16, 32, 64]
        if self.dense_units is None:
            self.dense_units = [128, 64]


class MultiViewExoplanetDetector:
    """
    Multi-view CNN for exoplanet detection.

    Architecture:
    - Shared CNN encoder for each view
    - Concatenated features fed to dense classifier
    - Binary output: planet probability

    Designed for TinyML deployment while maintaining high accuracy.
    """

    def __init__(self, config: Optional[MultiViewConfig] = None):
        self.config = config or MultiViewConfig()
        self.weights: Dict[str, np.ndarray] = {}
        self._build()

    def _build(self):
        """Initialize network weights."""
        np.random.seed(42)
        cfg = self.config

        # Shared CNN encoder weights (used for all 3 views)
        # Conv Block 1: 16 filters, kernel size 5
        in_ch = 1
        for i, filters in enumerate(cfg.conv_filters):
            kernel_size = 5 if i == 0 else 3
            self.weights[f"conv{i+1}_w"] = self._init_conv(kernel_size, in_ch, filters)
            self.weights[f"conv{i+1}_b"] = np.zeros(filters, dtype=np.float32)
            self.weights[f"bn{i+1}_gamma"] = np.ones(filters, dtype=np.float32)
            self.weights[f"bn{i+1}_beta"] = np.zeros(filters, dtype=np.float32)
            self.weights[f"bn{i+1}_mean"] = np.zeros(filters, dtype=np.float32)
            self.weights[f"bn{i+1}_var"] = np.ones(filters, dtype=np.float32)
            in_ch = filters

        # Calculate flattened size after pooling
        # Global: 2001 -> 500 -> 125 -> 31 (after 3 maxpool of 4)
        # Local/Secondary: 201 -> 50 -> 12 -> 3
        global_flat = 31 * cfg.conv_filters[-1]
        local_flat = 3 * cfg.conv_filters[-1]
        secondary_flat = 3 * cfg.conv_filters[-1]

        # After global average pooling, each view outputs conv_filters[-1] features
        total_features = cfg.conv_filters[-1] * 3  # 3 views

        # Dense layers for fusion
        prev_size = total_features
        for i, units in enumerate(cfg.dense_units):
            self.weights[f"fc{i+1}_w"] = self._init_dense(prev_size, units)
            self.weights[f"fc{i+1}_b"] = np.zeros(units, dtype=np.float32)
            prev_size = units

        # Output layer (binary classification)
        self.weights["out_w"] = self._init_dense(prev_size, cfg.num_classes)
        self.weights["out_b"] = np.zeros(cfg.num_classes, dtype=np.float32)

    def _init_conv(self, kernel_size: int, in_channels: int, out_channels: int) -> np.ndarray:
        """He initialization for conv weights."""
        std = np.sqrt(2.0 / (kernel_size * in_channels))
        return np.random.randn(kernel_size, in_channels, out_channels).astype(np.float32) * std

    def _init_dense(self, in_features: int, out_features: int) -> np.ndarray:
        """He initialization for dense weights."""
        std = np.sqrt(2.0 / in_features)
        return np.random.randn(in_features, out_features).astype(np.float32) * std

    def _conv1d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """1D convolution with same padding."""
        batch_size, seq_len, in_ch = x.shape
        k_size, _, out_ch = w.shape
        pad = k_size // 2

        x_pad = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')
        out = np.zeros((batch_size, seq_len, out_ch), dtype=np.float32)

        for i in range(seq_len):
            window = x_pad[:, i:i+k_size, :]
            for j in range(out_ch):
                out[:, i, j] = np.sum(window * w[:, :, j], axis=(1, 2)) + b[j]

        return out

    def _batchnorm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   mean: np.ndarray, var: np.ndarray, training: bool = False) -> np.ndarray:
        """Batch normalization."""
        if training:
            batch_mean = np.mean(x, axis=(0, 1))
            batch_var = np.var(x, axis=(0, 1))
            # Update running stats
            mean[:] = 0.9 * mean + 0.1 * batch_mean
            var[:] = 0.9 * var + 0.1 * batch_var
        else:
            batch_mean = mean
            batch_var = var

        eps = 1e-5
        x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        return gamma * x_norm + beta

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _maxpool1d(self, x: np.ndarray, pool_size: int = 4) -> np.ndarray:
        """Max pooling 1D."""
        batch, seq_len, ch = x.shape
        out_len = seq_len // pool_size
        out = np.zeros((batch, out_len, ch), dtype=np.float32)

        for i in range(out_len):
            out[:, i, :] = np.max(x[:, i*pool_size:(i+1)*pool_size, :], axis=1)

        return out

    def _global_avgpool(self, x: np.ndarray) -> np.ndarray:
        """Global average pooling."""
        return np.mean(x, axis=1)

    def _dense(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(x, w) + b

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _encode_view(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Encode a single view through the shared CNN.

        Args:
            x: Input of shape (batch, seq_len, 1)
            training: Whether in training mode

        Returns:
            Encoded features of shape (batch, conv_filters[-1])
        """
        if x.ndim == 2:
            x = x[:, :, np.newaxis]

        # Apply conv blocks
        for i in range(len(self.config.conv_filters)):
            x = self._conv1d(x, self.weights[f"conv{i+1}_w"], self.weights[f"conv{i+1}_b"])
            x = self._batchnorm(
                x,
                self.weights[f"bn{i+1}_gamma"],
                self.weights[f"bn{i+1}_beta"],
                self.weights[f"bn{i+1}_mean"],
                self.weights[f"bn{i+1}_var"],
                training
            )
            x = self._relu(x)
            x = self._maxpool1d(x, 4)

        # Global average pooling
        x = self._global_avgpool(x)

        return x

    def forward(self, global_view: np.ndarray, local_view: np.ndarray,
                secondary_view: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through multi-view network.

        Args:
            global_view: Shape (batch, 2001) or (batch, 2001, 1)
            local_view: Shape (batch, 201) or (batch, 201, 1)
            secondary_view: Shape (batch, 201) or (batch, 201, 1)
            training: Whether in training mode

        Returns:
            Probabilities of shape (batch, 2)
        """
        # Encode each view
        global_features = self._encode_view(global_view, training)
        local_features = self._encode_view(local_view, training)
        secondary_features = self._encode_view(secondary_view, training)

        # Concatenate features
        x = np.concatenate([global_features, local_features, secondary_features], axis=-1)

        # Dense layers
        for i in range(len(self.config.dense_units)):
            x = self._dense(x, self.weights[f"fc{i+1}_w"], self.weights[f"fc{i+1}_b"])
            x = self._relu(x)
            if training:
                # Dropout
                mask = np.random.binomial(1, 1 - self.config.dropout_rate, x.shape)
                x = x * mask / (1 - self.config.dropout_rate)

        # Output
        x = self._dense(x, self.weights["out_w"], self.weights["out_b"])

        return self._softmax(x)

    def predict(self, global_view: np.ndarray, local_view: np.ndarray,
                secondary_view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Returns:
            (predictions, confidences) where predictions are class indices
        """
        probs = self.forward(global_view, local_view, secondary_view, training=False)
        predictions = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return predictions, confidences

    def predict_proba(self, global_view: np.ndarray, local_view: np.ndarray,
                      secondary_view: np.ndarray) -> np.ndarray:
        """Return probability of being a planet (positive class)."""
        probs = self.forward(global_view, local_view, secondary_view, training=False)
        return probs[:, 1]  # Planet probability

    def save(self, filepath: str):
        """Save model weights."""
        np.savez(filepath, **self.weights,
                 _config=json.dumps({
                     'global_view_size': self.config.global_view_size,
                     'local_view_size': self.config.local_view_size,
                     'conv_filters': self.config.conv_filters,
                     'dense_units': self.config.dense_units,
                     'dropout_rate': self.config.dropout_rate
                 }))

    def load(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath, allow_pickle=True)
        self.weights = {k: data[k] for k in data.files if not k.startswith('_')}

    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""
        total_params = sum(w.size for w in self.weights.values())
        return {
            "total_parameters": total_params,
            "size_float32_kb": total_params * 4 / 1024,
            "size_int8_kb": total_params / 1024,
            "views": ["global", "local", "secondary"],
            "architecture": f"Conv{self.config.conv_filters} -> Dense{self.config.dense_units}"
        }


class ViewExtractor:
    """
    Extract global, local, and secondary views from light curves.

    This follows the preprocessing approach used in ExoMiner and AstroNet.
    """

    def __init__(self, global_size: int = 2001, local_size: int = 201):
        self.global_size = global_size
        self.local_size = local_size

    def phase_fold(self, time: np.ndarray, flux: np.ndarray,
                   period: float, t0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase-fold a light curve.

        Args:
            time: Time array
            flux: Flux array
            period: Orbital period in same units as time
            t0: Transit epoch (time of first transit)

        Returns:
            (phase, folded_flux) where phase is in [-0.5, 0.5]
        """
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0

        # Sort by phase
        sort_idx = np.argsort(phase)
        return phase[sort_idx], flux[sort_idx]

    def bin_light_curve(self, phase: np.ndarray, flux: np.ndarray,
                        num_bins: int) -> np.ndarray:
        """
        Bin a phase-folded light curve.

        Args:
            phase: Phase array in [-0.5, 0.5]
            flux: Flux array
            num_bins: Number of output bins

        Returns:
            Binned flux array of length num_bins
        """
        bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
        binned = np.zeros(num_bins, dtype=np.float32)

        for i in range(num_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if mask.sum() > 0:
                binned[i] = np.median(flux[mask])
            else:
                # Interpolate from neighbors
                if i > 0:
                    binned[i] = binned[i - 1]

        return binned

    def extract_views(self, time: np.ndarray, flux: np.ndarray,
                      period: float, t0: float,
                      duration: float = None) -> Dict[str, np.ndarray]:
        """
        Extract all three views from a light curve.

        Args:
            time: Time array
            flux: Flux array (normalized)
            period: Orbital period
            t0: Transit epoch
            duration: Transit duration (optional, estimated if not provided)

        Returns:
            Dictionary with 'global', 'local', 'secondary' views
        """
        # Estimate duration if not provided (typical ~3% of period for hot Jupiters)
        if duration is None:
            duration = 0.03 * period

        # Phase fold
        phase, folded_flux = self.phase_fold(time, flux, period, t0)

        # Global view: full phase
        global_view = self.bin_light_curve(phase, folded_flux, self.global_size)

        # Local view: zoom around transit (phase = 0)
        transit_half_width = (duration / period) * 2  # Double the transit width
        local_mask = np.abs(phase) < transit_half_width
        if local_mask.sum() > 10:
            local_phase = phase[local_mask]
            local_flux = folded_flux[local_mask]
            # Rescale to [-0.5, 0.5] for the zoomed view
            local_phase_rescaled = local_phase / (2 * transit_half_width)
            local_view = self.bin_light_curve(local_phase_rescaled, local_flux, self.local_size)
        else:
            local_view = np.zeros(self.local_size, dtype=np.float32)

        # Secondary view: zoom around secondary eclipse (phase = 0.5 or -0.5)
        secondary_mask = np.abs(np.abs(phase) - 0.5) < transit_half_width
        if secondary_mask.sum() > 10:
            sec_phase = phase[secondary_mask]
            sec_flux = folded_flux[secondary_mask]
            # Shift phase so secondary is at center
            sec_phase_shifted = sec_phase.copy()
            sec_phase_shifted[sec_phase_shifted > 0] -= 0.5
            sec_phase_shifted[sec_phase_shifted < 0] += 0.5
            sec_phase_rescaled = sec_phase_shifted / (2 * transit_half_width)
            secondary_view = self.bin_light_curve(sec_phase_rescaled, sec_flux, self.local_size)
        else:
            secondary_view = np.zeros(self.local_size, dtype=np.float32)

        # Normalize views
        global_view = self._normalize(global_view)
        local_view = self._normalize(local_view)
        secondary_view = self._normalize(secondary_view)

        return {
            'global': global_view,
            'local': local_view,
            'secondary': secondary_view
        }

    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        """Normalize flux to zero mean and unit variance."""
        mean = np.mean(flux)
        std = np.std(flux)
        if std > 0:
            return (flux - mean) / std
        return flux - mean


class TESSDataLoader:
    """
    Load real TESS data for training.

    Uses lightkurve library to download and process TESS light curves.
    """

    def __init__(self, cache_dir: str = "data/tess_cache"):
        self.cache_dir = cache_dir
        self.view_extractor = ViewExtractor()
        self._check_lightkurve()

    def _check_lightkurve(self):
        """Check if lightkurve is available."""
        try:
            import lightkurve as lk
            self.lk = lk
            self._has_lightkurve = True
        except ImportError:
            self._has_lightkurve = False
            print("Warning: lightkurve not installed. Use synthetic data or install with: pip install lightkurve")

    def load_tce(self, tic_id: int, sector: int = None,
                 period: float = None, t0: float = None,
                 duration: float = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Load a TESS TCE (Threshold Crossing Event) and extract views.

        Args:
            tic_id: TESS Input Catalog ID
            sector: TESS sector (optional)
            period: Orbital period in days
            t0: Transit epoch (BJD)
            duration: Transit duration in days

        Returns:
            Dictionary with views or None if failed
        """
        if not self._has_lightkurve:
            return None

        try:
            # Search for light curves
            search = self.lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector)

            if len(search) == 0:
                return None

            # Download and stitch light curves
            lc_collection = search.download_all()
            lc = lc_collection.stitch()

            # Remove outliers and flatten
            lc = lc.remove_outliers(sigma=5)
            lc = lc.flatten(window_length=301)

            # Get time and flux
            time = lc.time.value
            flux = lc.flux.value

            # Normalize flux
            flux = flux / np.median(flux)

            # Extract views
            if period is not None and t0 is not None:
                views = self.view_extractor.extract_views(time, flux, period, t0, duration)
                return views

            return {'time': time, 'flux': flux}

        except Exception as e:
            print(f"Error loading TIC {tic_id}: {e}")
            return None

    def load_confirmed_planets(self, max_samples: int = 1000) -> List[Dict]:
        """
        Load confirmed TESS planets from the NASA Exoplanet Archive.

        Returns:
            List of dictionaries with planet parameters and views
        """
        if not self._has_lightkurve:
            return []

        try:
            # This would require astroquery to access the NASA Exoplanet Archive
            # For now, return empty list - real implementation would query the archive
            print("Note: Full planet catalog loading requires astroquery")
            return []
        except Exception as e:
            print(f"Error loading planet catalog: {e}")
            return []


def generate_synthetic_multiview_data(n_samples: int = 1000,
                                       seed: int = 42) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Generate synthetic multi-view training data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        (views_dict, labels) where views_dict has 'global', 'local', 'secondary' keys
    """
    np.random.seed(seed)

    global_views = np.zeros((n_samples, 2001), dtype=np.float32)
    local_views = np.zeros((n_samples, 201), dtype=np.float32)
    secondary_views = np.zeros((n_samples, 201), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)

    n_planets = n_samples // 2

    for i in range(n_samples):
        is_planet = i < n_planets
        labels[i] = 1 if is_planet else 0

        # Generate phase array
        global_phase = np.linspace(-0.5, 0.5, 2001)
        local_phase = np.linspace(-0.5, 0.5, 201)

        # Base noise
        noise_level = np.random.uniform(0.001, 0.005)
        global_views[i] = np.random.randn(2001) * noise_level
        local_views[i] = np.random.randn(201) * noise_level
        secondary_views[i] = np.random.randn(201) * noise_level

        if is_planet:
            # Add transit signal
            depth = np.random.uniform(0.001, 0.02)  # Transit depth
            width = np.random.uniform(0.01, 0.05)   # Transit width in phase

            # Primary transit in global and local views
            transit_mask = np.abs(global_phase) < width
            global_views[i, transit_mask] -= depth * (1 - (global_phase[transit_mask] / width) ** 2)

            local_transit_mask = np.abs(local_phase) < width * 5
            local_views[i, local_transit_mask] -= depth * (1 - (local_phase[local_transit_mask] / (width * 5)) ** 2)

            # Secondary eclipse (smaller, ~20% of primary)
            secondary_depth = depth * np.random.uniform(0.1, 0.3)
            sec_mask = np.abs(local_phase) < width * 3
            secondary_views[i, sec_mask] -= secondary_depth * (1 - (local_phase[sec_mask] / (width * 3)) ** 2)

        else:
            # Add non-planet signals (eclipsing binaries, noise, etc.)
            signal_type = np.random.choice(['eb', 'noise', 'stellar'])

            if signal_type == 'eb':
                # Eclipsing binary - deep V-shaped eclipses
                depth = np.random.uniform(0.05, 0.3)
                width = np.random.uniform(0.02, 0.1)

                transit_mask = np.abs(global_phase) < width
                global_views[i, transit_mask] -= depth * (1 - np.abs(global_phase[transit_mask]) / width)

                local_transit_mask = np.abs(local_phase) < width * 5
                local_views[i, local_transit_mask] -= depth * (1 - np.abs(local_phase[local_transit_mask]) / (width * 5))

                # Strong secondary (similar depth to primary for EBs)
                sec_depth = depth * np.random.uniform(0.5, 1.0)
                sec_mask = np.abs(local_phase) < width * 3
                secondary_views[i, sec_mask] -= sec_depth * (1 - np.abs(local_phase[sec_mask]) / (width * 3))

            elif signal_type == 'stellar':
                # Stellar variability
                freq = np.random.uniform(1, 5)
                amp = np.random.uniform(0.001, 0.01)
                global_views[i] += amp * np.sin(2 * np.pi * freq * global_phase)
                local_views[i] += amp * np.sin(2 * np.pi * freq * local_phase)

        # Normalize
        global_views[i] = (global_views[i] - np.mean(global_views[i])) / (np.std(global_views[i]) + 1e-8)
        local_views[i] = (local_views[i] - np.mean(local_views[i])) / (np.std(local_views[i]) + 1e-8)
        secondary_views[i] = (secondary_views[i] - np.mean(secondary_views[i])) / (np.std(secondary_views[i]) + 1e-8)

    # Shuffle
    perm = np.random.permutation(n_samples)

    return {
        'global': global_views[perm],
        'local': local_views[perm],
        'secondary': secondary_views[perm]
    }, labels[perm]


# Export
__all__ = [
    'MultiViewConfig',
    'MultiViewExoplanetDetector',
    'ViewExtractor',
    'TESSDataLoader',
    'generate_synthetic_multiview_data'
]

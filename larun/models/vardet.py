"""
VARDET-001 — Variable Source Detector
Inspired by VARnet (Paz, 2024, The Astronomical Journal, arXiv:2409.15499)

Architecture: Fourier (Lomb-Scargle) + Wavelet (db4) feature extraction
              → Random Forest classifier
Target size: ~50 KB (.npz)
Target inference: <500 ms per light curve (CPU)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from larun.models.base import BaseModel

logger = logging.getLogger(__name__)


class VARDET001(BaseModel):
    """
    Variable source detection using Fourier + Wavelet feature extraction.

    Classification scheme (matches VARnet 4-class output):
        0: NON_VARIABLE  — Constant brightness
        1: TRANSIENT     — Sudden events (supernovae, flares, TDEs)
        2: PULSATOR      — Periodic pulsation (Cepheids, RR Lyrae, Delta Scuti)
        3: ECLIPSING     — Eclipsing binaries, planetary transits

    Reference:
        Paz, M. (2024). "A Sub-Millisecond Fourier and Wavelet Based Model to
        Extract Variable Candidates from the NEOWISE Single-Exposure Database."
        The Astronomical Journal. arXiv:2409.15499
    """

    MODEL_ID = "VARDET-001"
    CLASSES = {
        0: "NON_VARIABLE",
        1: "TRANSIENT",
        2: "PULSATOR",
        3: "ECLIPSING",
    }

    # Lomb-Scargle frequency grid (cycles/day)
    _LS_FREQS = np.linspace(0.01, 25.0, 10000)

    def __init__(self, model_path: str | Path | None = None):
        super().__init__(model_path)
        self._rf = None  # Random Forest classifier (sklearn)

    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------

    def extract_features(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Extract 10-dimensional feature vector from a light curve.

        Features:
            1. fourier_power_ratio    — peak LSP power / total power
            2. fourier_peak_freq      — dominant frequency (cycles/day)
            3. fourier_n_significant  — number of peaks >3σ above noise
            4. wavelet_energy_ratio   — max wavelet scale energy / total
            5. wavelet_max_coeff      — maximum absolute wavelet coefficient
            6. wavelet_entropy        — Shannon entropy of wavelet energies
            7. amplitude              — peak-to-peak magnitude range
            8. skewness               — magnitude distribution skewness
            9. kurtosis               — magnitude distribution kurtosis
            10. mad                   — median absolute deviation
        """
        times = np.asarray(times, dtype=float)
        magnitudes = np.asarray(magnitudes, dtype=float)

        # Remove NaNs
        mask = np.isfinite(times) & np.isfinite(magnitudes)
        times, magnitudes = times[mask], magnitudes[mask]

        if len(magnitudes) < 5:
            return np.zeros(10)

        # --- Fourier Features (Lomb-Scargle for unevenly sampled data) ---
        try:
            power = signal.lombscargle(times, magnitudes - magnitudes.mean(), self._LS_FREQS, normalize=True)
            power_sum = power.sum()
            f_power_ratio = float(power.max() / power_sum) if power_sum > 0 else 0.0
            f_peak_freq = float(self._LS_FREQS[np.argmax(power)])
            noise_thresh = float(np.median(power) + 3 * np.std(power))
            f_n_sig = int(np.sum(power > noise_thresh))
        except Exception:
            f_power_ratio, f_peak_freq, f_n_sig = 0.0, 0.0, 0

        # --- Wavelet Features (Daubechies db4, 4 levels) ---
        try:
            import pywt
            n = len(times)
            t_even = np.linspace(times.min(), times.max(), n)
            if len(np.unique(times)) >= 2:
                interp = interp1d(times, magnitudes, kind="linear", bounds_error=False, fill_value="extrapolate")
                mag_even = interp(t_even)
            else:
                mag_even = magnitudes

            coeffs = pywt.wavedec(mag_even, "db4", level=4)
            energies = [float(np.sum(c**2)) for c in coeffs]
            total_e = sum(energies)
            if total_e > 0:
                w_energy_ratio = max(energies) / total_e
                probs = np.array([e / total_e for e in energies if e > 0])
                w_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            else:
                w_energy_ratio, w_entropy = 0.0, 0.0
            w_max_coeff = float(max(np.max(np.abs(c)) for c in coeffs if len(c) > 0))
        except ImportError:
            logger.warning("PyWavelets not installed; wavelet features will be zero. pip install PyWavelets")
            w_energy_ratio, w_max_coeff, w_entropy = 0.0, 0.0, 0.0

        # --- Statistical Features ---
        mag_std = magnitudes.std()
        if mag_std > 0:
            z = (magnitudes - magnitudes.mean()) / mag_std
            skew = float(np.mean(z**3))
            kurt = float(np.mean(z**4))
        else:
            skew, kurt = 0.0, 3.0

        amplitude = float(np.ptp(magnitudes))
        mad = float(np.median(np.abs(magnitudes - np.median(magnitudes))))

        return np.array([
            f_power_ratio,
            f_peak_freq,
            float(f_n_sig),
            w_energy_ratio,
            w_max_coeff,
            w_entropy,
            amplitude,
            skew,
            kurt,
            mad,
        ], dtype=float)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        light_curves: list[dict],
        labels: list[int],
        n_estimators: int = 100,
        max_depth: int = 10,
    ) -> "VARDET001":
        """
        Train on labeled light curves.

        Args:
            light_curves: list of dicts with keys 'times', 'mags' (and optionally 'errors')
            labels: integer class labels (0–3, see CLASSES)
            n_estimators: number of Random Forest trees
            max_depth: max tree depth

        Training data sources:
            - OGLE-III variable star catalog (known pulsators, eclipsing binaries)
            - ASAS-SN transient catalog
            - Kepler/TESS confirmed non-variables
            - Synthetic light curves (astropy + noise models)
        """
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Training VARDET-001 on {len(light_curves)} light curves...")
        X = np.array([
            self.extract_features(lc["times"], lc["mags"], lc.get("errors"))
            for lc in light_curves
        ])
        y = np.array(labels)

        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self._rf.fit(X, y)
        self._loaded = True
        logger.info(f"VARDET-001 trained. Feature importances: {self._rf.feature_importances_.round(3)}")
        return self

    def train_synthetic(self, n_per_class: int = 500) -> "VARDET001":
        """
        Quick training on synthetic light curves for bootstrapping / testing.
        Use real OGLE-III / ASAS-SN data for production training.
        """
        logger.info(f"Training VARDET-001 on synthetic data ({n_per_class} per class)...")
        light_curves, labels = [], []
        rng = np.random.default_rng(42)

        for class_id in range(4):
            for _ in range(n_per_class):
                t = np.sort(rng.uniform(0, 100, rng.integers(50, 200)))
                lc = _synthetic_light_curve(t, class_id, rng)
                light_curves.append(lc)
                labels.append(class_id)

        return self.train(light_curves, labels)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Classify a light curve.

        Returns:
            (label, probability_vector)  e.g. ("PULSATOR", array([0.02, 0.05, 0.91, 0.02]))
        """
        if self._rf is None:
            if not self._loaded:
                self.load()
            if self._rf is None:
                logger.warning("VARDET-001 not trained. Auto-training on synthetic data...")
                self.train_synthetic()

        features = self.extract_features(times, magnitudes, errors)
        proba = self._rf.predict_proba(features.reshape(1, -1))[0]
        label = self.CLASSES[int(np.argmax(proba))]
        return label, proba

    def predict_batch(self, light_curves: list[dict]) -> list[dict]:
        """Classify multiple light curves, return list of result dicts."""
        results = []
        for lc in light_curves:
            label, proba = self.predict(lc["times"], lc["mags"], lc.get("errors"))
            results.append(self.result_dict(label, proba))
        return results

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _get_weights(self) -> dict:
        if self._rf is None:
            return {}
        return {"rf_pickle": np.frombuffer(pickle.dumps(self._rf), dtype=np.uint8)}

    def _load_weights(self, data) -> None:
        if "rf_pickle" in data:
            self._rf = pickle.loads(data["rf_pickle"].tobytes())
            self._loaded = True


# -------------------------------------------------------------------------
# Synthetic data helpers (for bootstrap training & testing)
# -------------------------------------------------------------------------

def _synthetic_light_curve(
    times: np.ndarray,
    class_id: int,
    rng: np.random.Generator,
) -> dict:
    """Generate a synthetic light curve for a given class."""
    noise_level = rng.uniform(0.005, 0.02)
    base_mag = rng.uniform(12.0, 16.0)
    magnitudes = np.full_like(times, base_mag)

    if class_id == 0:  # NON_VARIABLE — just noise
        magnitudes += rng.normal(0, noise_level, len(times))

    elif class_id == 1:  # TRANSIENT — gaussian flare/event
        peak_t = rng.uniform(times[0] + 5, times[-1] - 5)
        width = rng.uniform(1.0, 5.0)
        amplitude = rng.uniform(0.1, 1.5)
        magnitudes -= amplitude * np.exp(-0.5 * ((times - peak_t) / width) ** 2)
        magnitudes += rng.normal(0, noise_level, len(times))

    elif class_id == 2:  # PULSATOR — sinusoidal
        period = rng.uniform(0.5, 20.0)
        amplitude = rng.uniform(0.05, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        magnitudes += amplitude * np.sin(2 * np.pi * times / period + phase)
        magnitudes += rng.normal(0, noise_level, len(times))

    elif class_id == 3:  # ECLIPSING — box-shaped dip
        period = rng.uniform(1.0, 15.0)
        depth = rng.uniform(0.02, 0.3)
        duration = rng.uniform(0.05, 0.2)  # fraction of period
        phase = (times % period) / period
        eclipse = (phase < duration / 2) | (phase > 1 - duration / 2)
        magnitudes[eclipse] += depth
        magnitudes += rng.normal(0, noise_level, len(times))

    return {"times": times, "mags": magnitudes}

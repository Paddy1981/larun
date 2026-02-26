"""
DEBLEND-001 — Light Curve Contamination Separator

Purpose: Detect when a light curve contains blended signals from multiple
sources. Critical for TESS (21-arcsec pixels) where contamination is common.

Architecture: Multi-frequency analysis + crowding metrics → Random Forest
Target size: ~15 KB
Target inference: <300 ms
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import signal

from larun.models.base import BaseModel

logger = logging.getLogger(__name__)


class DEBLEND001(BaseModel):
    """
    Detects contaminated/blended light curves from multiple astrophysical sources.

    Classes:
        0: CLEAN        — Single source, signal is reliable
        1: MILD_BLEND   — Some contamination (<30%), results may be slightly affected
        2: SEVERE_BLEND — Heavy contamination (>30%), results are unreliable

    Key indicators of blending:
    - Multiple significant frequency peaks (>1 dominant signal)
    - Amplitude inconsistency across light curve segments
    - Even/odd transit depth differences (diluted transit vs. EB)
    - High crowding metrics (CROWDSAP < 0.5 in TESS)
    """

    MODEL_ID = "DEBLEND-001"
    CLASSES = {
        0: "CLEAN",
        1: "MILD_BLEND",
        2: "SEVERE_BLEND",
    }

    def __init__(self, model_path: str | Path | None = None):
        super().__init__(model_path)
        self._rf = None

    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------

    def extract_features(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
        crowdsap: float | None = None,
        flfrcsap: float | None = None,
    ) -> np.ndarray:
        """
        Extract 9 blend-detection features.

        Features:
            1. n_dominant_freqs   — number of statistically significant LSP peaks
            2. freq_ratio         — ratio of 2nd to 1st dominant frequency power
            3. amplitude_scatter  — scatter of per-segment amplitudes
            4. phase_coherence    — phase stability across the time series
            5. even_odd_ratio     — ratio of even vs. odd epoch depths (transit check)
            6. crowdsap           — TESS crowding metric (0=fully blended, 1=clean)
            7. flfrcsap           — TESS flux fraction metric
            8. segment_rms_var    — variance of RMS across time segments
            9. residual_power     — power remaining after removing dominant signal

        Args:
            crowdsap: TESS CROWDSAP value from FITS header (optional)
            flfrcsap: TESS FLFRCSAP value from FITS header (optional)
        """
        times = np.asarray(times, dtype=float)
        magnitudes = np.asarray(magnitudes, dtype=float)
        mask = np.isfinite(times) & np.isfinite(magnitudes)
        t, m = times[mask], magnitudes[mask]

        n = len(m)
        if n < 10:
            return np.zeros(9)

        freqs = np.linspace(0.01, 25.0, 5000)

        # --- Multiple frequency analysis ---
        try:
            power = signal.lombscargle(t, m - m.mean(), freqs, normalize=True)
            noise = np.median(power) + 3 * np.std(power)
            sig_peaks = power > noise
            n_dominant_freqs = float(np.sum(sig_peaks))

            # Ratio of 2nd to 1st peak (high ratio → likely multiple sources)
            sorted_power = np.sort(power)[::-1]
            freq_ratio = float(sorted_power[1] / sorted_power[0]) if sorted_power[0] > 0 else 0.0

            # Residual power after removing dominant frequency
            peak_idx = np.argmax(power)
            peak_power = power[peak_idx]
            residual_power = float((power.sum() - peak_power) / power.sum()) if power.sum() > 0 else 0.0
        except Exception:
            n_dominant_freqs, freq_ratio, residual_power = 0.0, 0.0, 0.0

        # --- Amplitude consistency across time segments ---
        n_seg = min(4, n // 10)
        if n_seg >= 2:
            seg_amplitudes = []
            seg_rms = []
            for chunk in np.array_split(m, n_seg):
                seg_amplitudes.append(float(np.ptp(chunk)))
                seg_rms.append(float(np.sqrt(np.mean(chunk**2))))
            amplitude_scatter = float(np.std(seg_amplitudes) / (np.mean(seg_amplitudes) + 1e-9))
            segment_rms_var = float(np.var(seg_rms))
        else:
            amplitude_scatter, segment_rms_var = 0.0, 0.0

        # --- Phase coherence (stability of dominant signal's phase) ---
        try:
            peak_period = 1.0 / freqs[np.argmax(power)] if np.argmax(power) > 0 else 1.0
            phases = (t % peak_period) / peak_period
            # Bin by phase and check consistency
            bins = np.digitize(phases, np.linspace(0, 1, 20))
            bin_means = [m[bins == b].mean() if np.sum(bins == b) > 0 else 0.0 for b in range(1, 21)]
            phase_coherence = float(1.0 / (np.std(bin_means) + 1e-9))
            phase_coherence = min(phase_coherence, 100.0)  # cap
        except Exception:
            phase_coherence = 0.0

        # --- Even/odd epoch ratio (diluted EB indicator) ---
        try:
            peak_period = 1.0 / freqs[np.argmax(power)] if np.argmax(power) > 0 else 1.0
            epoch_num = np.floor(t / peak_period).astype(int)
            even_depths, odd_depths = [], []
            for ep in np.unique(epoch_num):
                idx = epoch_num == ep
                if np.sum(idx) < 3:
                    continue
                depth = float(np.ptp(m[idx]))
                if ep % 2 == 0:
                    even_depths.append(depth)
                else:
                    odd_depths.append(depth)
            if even_depths and odd_depths:
                even_odd_ratio = float(np.mean(even_depths) / (np.mean(odd_depths) + 1e-9))
            else:
                even_odd_ratio = 1.0
        except Exception:
            even_odd_ratio = 1.0

        # --- Crowding metrics (TESS specific) ---
        crowdsap_val = float(crowdsap) if crowdsap is not None else 1.0
        flfrcsap_val = float(flfrcsap) if flfrcsap is not None else 1.0

        return np.array([
            n_dominant_freqs,
            freq_ratio,
            amplitude_scatter,
            phase_coherence,
            even_odd_ratio,
            crowdsap_val,
            flfrcsap_val,
            segment_rms_var,
            residual_power,
        ], dtype=float)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        light_curves: list[dict],
        labels: list[int],
        n_estimators: int = 100,
    ) -> "DEBLEND001":
        """
        Train on labeled light curves.

        Labels:
            0 = CLEAN, 1 = MILD_BLEND, 2 = SEVERE_BLEND

        Training data: TESS light curves with known CROWDSAP values
        + simulated blended signals.
        """
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Training DEBLEND-001 on {len(light_curves)} light curves...")
        X = np.array([
            self.extract_features(
                lc["times"],
                lc["mags"],
                lc.get("errors"),
                lc.get("crowdsap"),
                lc.get("flfrcsap"),
            )
            for lc in light_curves
        ])
        y = np.array(labels)

        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._rf.fit(X, y)
        self._loaded = True
        logger.info("DEBLEND-001 trained successfully.")
        return self

    def train_synthetic(self, n_per_class: int = 300) -> "DEBLEND001":
        """Bootstrap training on synthetic blended/clean signals."""
        from larun.models.vardet import _synthetic_light_curve

        light_curves, labels = [], []
        rng = np.random.default_rng(1)

        for _ in range(n_per_class):
            t = np.sort(rng.uniform(0, 30, rng.integers(100, 500)))

            # CLEAN: single pulsator or quiet star
            lc = _synthetic_light_curve(t, rng.choice([0, 2]), rng)
            lc["crowdsap"] = rng.uniform(0.8, 1.0)
            lc["flfrcsap"] = rng.uniform(0.9, 1.0)
            light_curves.append(lc)
            labels.append(0)

            # MILD_BLEND: add a secondary signal at ~30% amplitude
            lc2 = _synthetic_light_curve(t, 2, rng)
            secondary = 0.3 * np.sin(2 * np.pi * t / rng.uniform(2, 12))
            lc2["mags"] = lc2["mags"] + secondary
            lc2["crowdsap"] = rng.uniform(0.5, 0.8)
            lc2["flfrcsap"] = rng.uniform(0.6, 0.9)
            light_curves.append(lc2)
            labels.append(1)

            # SEVERE_BLEND: dominant contaminating signal
            lc3 = _synthetic_light_curve(t, rng.choice([2, 3]), rng)
            secondary2 = 0.8 * np.sin(2 * np.pi * t / rng.uniform(0.5, 5))
            lc3["mags"] = lc3["mags"] + secondary2
            lc3["crowdsap"] = rng.uniform(0.0, 0.5)
            lc3["flfrcsap"] = rng.uniform(0.1, 0.6)
            light_curves.append(lc3)
            labels.append(2)

        return self.train(light_curves, labels)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
        crowdsap: float | None = None,
        flfrcsap: float | None = None,
    ) -> tuple[str, np.ndarray]:
        if self._rf is None:
            if not self._loaded:
                self.load()
            if self._rf is None:
                logger.warning("DEBLEND-001 not trained. Auto-training on synthetic data...")
                self.train_synthetic()

        features = self.extract_features(times, magnitudes, errors, crowdsap, flfrcsap)
        proba = self._rf.predict_proba(features.reshape(1, -1))[0]
        label = self.CLASSES[int(np.argmax(proba))]
        return label, proba

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

"""
PERIODOGRAM-001 — Advanced Multi-Method Period Finder

Combines four period-finding methods with consensus scoring for
robust period determination on astronomical light curves.

Methods:
    1. Lomb-Scargle   — Standard periodogram, best for sinusoidal signals
    2. BLS            — Box Least Squares, optimized for transit signals
    3. PDM            — Phase Dispersion Minimization, robust for all shapes
    4. ACF            — Autocorrelation Function, best for quasi-periodic signals

Not a classification model — outputs period + confidence + period type.
Target: <2s inference (computationally intensive)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

from larun.models.base import BaseModel

logger = logging.getLogger(__name__)


class PERIODOGRAM001(BaseModel):
    """
    Multi-method period finder with confidence scoring.

    Consensus rule: if 3+ methods agree within 1%, confidence is HIGH.
    """

    MODEL_ID = "PERIODOGRAM-001"
    CLASSES = {
        0: "NO_PERIOD",
        1: "TRANSIT",
        2: "PULSATION",
        3: "ROTATION",
        4: "ECLIPSING",
    }

    # Period search range (days)
    MIN_PERIOD = 0.04   # ~1 hour
    MAX_PERIOD = 100.0  # ~3 months
    N_PERIODS = 5000

    def extract_features(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Not used for classification; returns empty array."""
        return np.array([])

    def find_period(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> dict:
        """
        Run all period-finding methods and return consensus result.

        Returns:
            {
                'best_period': float (days),
                'confidence':  float (0–1),
                'period_type': str,
                'all_methods': {method: {'period': float, 'power': float}},
                'model_id': 'PERIODOGRAM-001'
            }
        """
        times = np.asarray(times, dtype=float)
        magnitudes = np.asarray(magnitudes, dtype=float)
        mask = np.isfinite(times) & np.isfinite(magnitudes)
        t, m = times[mask], magnitudes[mask]

        if len(t) < 10:
            return self._no_period_result()

        results = {}

        # Run each method
        results["lomb_scargle"] = self._lomb_scargle(t, m)
        results["bls"] = self._bls(t, m)
        results["pdm"] = self._pdm(t, m)
        results["acf"] = self._acf(t, m)

        # Consensus
        best_period, confidence = self._consensus(results)
        period_type = self._classify_period_type(t, m, best_period)

        return {
            "model_id": self.MODEL_ID,
            "best_period": round(best_period, 6),
            "confidence": round(confidence, 4),
            "period_type": period_type,
            "all_methods": results,
        }

    def predict(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> tuple[str, np.ndarray]:
        """Returns (period_type, confidence_array) — adapts to BaseModel interface."""
        result = self.find_period(times, magnitudes, errors)
        period_type = result["period_type"]
        confidence = result["confidence"]
        class_idx = next((k for k, v in self.CLASSES.items() if v == period_type), 0)
        proba = np.zeros(len(self.CLASSES))
        proba[class_idx] = confidence
        proba[0] = 1.0 - confidence if class_idx != 0 else 1.0
        proba = np.clip(proba, 0, 1)
        return period_type, proba

    # -------------------------------------------------------------------------
    # Period-Finding Methods
    # -------------------------------------------------------------------------

    def _lomb_scargle(self, times: np.ndarray, magnitudes: np.ndarray) -> dict:
        """Lomb-Scargle periodogram using scipy."""
        try:
            freqs = np.linspace(1.0 / self.MAX_PERIOD, 1.0 / self.MIN_PERIOD, self.N_PERIODS)
            ang_freqs = 2 * np.pi * freqs
            power = signal.lombscargle(times, magnitudes - magnitudes.mean(), ang_freqs, normalize=True)
            best_idx = np.argmax(power)
            return {
                "period": float(1.0 / freqs[best_idx]),
                "power": float(power[best_idx]),
                "method": "lomb_scargle",
            }
        except Exception as e:
            logger.debug(f"LS failed: {e}")
            return {"period": 0.0, "power": 0.0, "method": "lomb_scargle"}

    def _bls(self, times: np.ndarray, magnitudes: np.ndarray) -> dict:
        """
        Box Least Squares — optimized for transit detection.
        Simplified BLS implementation.
        """
        try:
            periods = np.linspace(self.MIN_PERIOD, min(self.MAX_PERIOD, (times[-1] - times[0]) / 2), 2000)
            best_power, best_period = 0.0, periods[0]

            m_norm = magnitudes - magnitudes.mean()
            m_std = magnitudes.std()
            if m_std == 0:
                return {"period": 0.0, "power": 0.0, "method": "bls"}

            for p in periods:
                phases = (times % p) / p
                # BLS power: measure how much variance is explained by a box dip
                # Sort by phase and look for coherent dip in ~10% of phase
                sort_idx = np.argsort(phases)
                ph_sorted = phases[sort_idx]
                m_sorted = m_norm[sort_idx]
                n = len(m_sorted)
                box_width = max(2, int(0.1 * n))  # 10% transit duration
                # Sliding window sum
                cum = np.cumsum(m_sorted)
                window_sums = cum[box_width:] - np.concatenate([[0], cum[:-box_width]])
                if len(window_sums) > 0:
                    power = float(abs(window_sums.min()) / (m_std * np.sqrt(box_width)))
                    if power > best_power:
                        best_power = power
                        best_period = p

            return {"period": float(best_period), "power": float(best_power), "method": "bls"}
        except Exception as e:
            logger.debug(f"BLS failed: {e}")
            return {"period": 0.0, "power": 0.0, "method": "bls"}

    def _pdm(self, times: np.ndarray, magnitudes: np.ndarray) -> dict:
        """
        Phase Dispersion Minimization.
        Period with minimum phase dispersion wins.
        """
        try:
            periods = np.linspace(self.MIN_PERIOD, min(30.0, (times[-1] - times[0]) / 2), 1000)
            best_period = periods[0]
            best_theta = 1.0

            for p in periods:
                phases = (times % p) / p
                # Sort by phase and compute variance in bins
                sort_idx = np.argsort(phases)
                m_sorted = magnitudes[sort_idx]
                n_bins = 10
                bin_vars = []
                for chunk in np.array_split(m_sorted, n_bins):
                    if len(chunk) > 1:
                        bin_vars.append(float(np.var(chunk)))
                if bin_vars:
                    theta = float(np.mean(bin_vars) / (np.var(magnitudes) + 1e-12))
                    if theta < best_theta:
                        best_theta = theta
                        best_period = p

            # Convert theta to "power" (1 - theta, higher is better)
            pdm_power = max(0.0, 1.0 - best_theta)
            return {"period": float(best_period), "power": float(pdm_power), "method": "pdm"}
        except Exception as e:
            logger.debug(f"PDM failed: {e}")
            return {"period": 0.0, "power": 0.0, "method": "pdm"}

    def _acf(self, times: np.ndarray, magnitudes: np.ndarray) -> dict:
        """
        Autocorrelation Function — best for quasi-periodic signals (stellar rotation).
        Interpolates to even grid first.
        """
        try:
            from scipy.interpolate import interp1d
            from scipy.signal import find_peaks

            n = min(len(times), 2048)
            t_even = np.linspace(times.min(), times.max(), n)
            interp = interp1d(times, magnitudes, kind="linear", bounds_error=False, fill_value="extrapolate")
            m_even = interp(t_even) - interp(t_even).mean()

            # Full autocorrelation
            acf = np.correlate(m_even, m_even, mode="full")
            acf = acf[len(acf) // 2:]  # positive lags
            acf /= acf[0]  # normalize

            # Find first significant peak
            dt = (times.max() - times.min()) / (n - 1)
            min_lag = max(1, int(self.MIN_PERIOD / dt))
            max_lag = min(len(acf) - 1, int(self.MAX_PERIOD / dt))

            peaks, props = find_peaks(acf[min_lag:max_lag], height=0.1, distance=max(1, min_lag // 2))
            if len(peaks) > 0:
                best_lag_idx = peaks[np.argmax(props["peak_heights"])] + min_lag
                best_period = float(best_lag_idx * dt)
                best_power = float(acf[best_lag_idx])
            else:
                best_period = 0.0
                best_power = 0.0

            return {"period": best_period, "power": best_power, "method": "acf"}
        except Exception as e:
            logger.debug(f"ACF failed: {e}")
            return {"period": 0.0, "power": 0.0, "method": "acf"}

    # -------------------------------------------------------------------------
    # Consensus & Classification
    # -------------------------------------------------------------------------

    def _consensus(self, results: dict) -> tuple[float, float]:
        """
        Find consensus period. If 3+ methods agree within 1%, confidence is HIGH.

        Returns:
            (best_period, confidence)
        """
        periods = [(r["period"], r["power"]) for r in results.values() if r["period"] > 0]
        if not periods:
            return 0.0, 0.0

        # Weight by power (LSP + PDM + BLS + ACF)
        periods_arr = np.array([p for p, _ in periods])
        powers_arr = np.array([pw for _, pw in periods])

        if powers_arr.sum() == 0:
            best_period = periods_arr[0]
            confidence = 0.1
        else:
            best_period = float(np.average(periods_arr, weights=powers_arr))

        # Count agreements within 1%
        agreements = sum(
            1 for p, _ in periods
            if abs(p - best_period) / (best_period + 1e-9) < 0.01
        )
        confidence = min(1.0, 0.2 * agreements + 0.2 * float(powers_arr.max()))
        return best_period, confidence

    def _classify_period_type(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        period: float,
    ) -> str:
        """
        Heuristic classification of period type.
        Determined by BLS depth symmetry and amplitude.
        """
        if period <= 0:
            return "NO_PERIOD"

        try:
            phases = (times % period) / period
            sort_idx = np.argsort(phases)
            m_sorted = magnitudes[sort_idx]
            amplitude = float(np.ptp(m_sorted))

            # Check for asymmetric dip (transit vs. eclipse)
            n = len(m_sorted)
            half = n // 2
            first_half_min = m_sorted[:half].min()
            second_half_min = m_sorted[half:].min()

            if amplitude < 0.01:
                return "ROTATION"
            elif abs(first_half_min - second_half_min) / (amplitude + 1e-9) > 0.3:
                return "ECLIPSING"
            elif amplitude < 0.05:
                return "TRANSIT"
            else:
                return "PULSATION"
        except Exception:
            return "PULSATION"

    def _no_period_result(self) -> dict:
        return {
            "model_id": self.MODEL_ID,
            "best_period": 0.0,
            "confidence": 0.0,
            "period_type": "NO_PERIOD",
            "all_methods": {},
        }

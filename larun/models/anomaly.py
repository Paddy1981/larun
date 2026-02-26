"""
ANOMALY-001 — Time-Series Anomaly Detector

Architecture: Isolation Forest + optional autoencoder on the same
              10-dimensional feature vector used by VARDET-001.

Purpose: Catch objects that don't fit any known classification —
         the "weird stuff" that leads to new discoveries.
         Classic examples: Boyajian's Star (KIC 8462852), fast-radio
         burst optical counterparts, unusual long-period variables.

Target size: ~20 KB
Target inference: <200 ms
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from larun.models.base import BaseModel
from larun.models.vardet import VARDET001

logger = logging.getLogger(__name__)


class ANOMALY001(BaseModel):
    """
    Anomaly detection for astronomical light curves.

    Uses the same 10-feature vector as VARDET-001 (Fourier + Wavelet + stats)
    and applies Isolation Forest for unsupervised anomaly scoring.
    A secondary threshold on the anomaly score separates MILD from STRONG.

    Classes:
        0: NORMAL        — Consistent with known variability types
        1: MILD_ANOMALY  — Slightly unusual (worth flagging for review)
        2: STRONG_ANOMALY — Highly unusual (priority human review)
    """

    MODEL_ID = "ANOMALY-001"
    CLASSES = {
        0: "NORMAL",
        1: "MILD_ANOMALY",
        2: "STRONG_ANOMALY",
    }

    # Anomaly score thresholds (Isolation Forest scores in [-1, 0] for anomalies
    # and around +0.5 for inliers; we negate for intuitive "higher = more anomalous")
    THRESHOLD_MILD = 0.6      # anomaly_score > this → MILD
    THRESHOLD_STRONG = 0.8    # anomaly_score > this → STRONG

    def __init__(self, model_path: str | Path | None = None):
        super().__init__(model_path)
        self._iso_forest = None
        self._feature_extractor = VARDET001()  # Reuse VARDET feature extraction

    # -------------------------------------------------------------------------
    # Feature Extraction (delegates to VARDET-001's extractor + extra features)
    # -------------------------------------------------------------------------

    def extract_features(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        14-dimensional feature vector:
            - 10 features from VARDET-001 (Fourier + Wavelet + stats)
            - 4 additional variability indices:
              11. stetson_j    — Stetson J index (correlated variability)
              12. von_neumann  — von Neumann ratio (smoothness)
              13. beyond_1std  — fraction of points > 1σ from mean
              14. max_slope    — maximum magnitude change rate between consecutive points
        """
        base_feats = self._feature_extractor.extract_features(times, magnitudes, errors)

        times = np.asarray(times, dtype=float)
        magnitudes = np.asarray(magnitudes, dtype=float)
        mask = np.isfinite(times) & np.isfinite(magnitudes)
        t, m = times[mask], magnitudes[mask]

        if len(m) < 3:
            return np.concatenate([base_feats, np.zeros(4)])

        # Stetson J (simplified — requires pairs of observations)
        m_mean = m.mean()
        residuals = m - m_mean
        n = len(m)
        stetson_j = float(np.sqrt(1.0 / (n * (n - 1))) * np.sum(residuals[:-1] * residuals[1:]))

        # von Neumann ratio (η) — lower means smoother
        delta2 = np.sum(np.diff(m) ** 2) / (n - 1)
        variance = np.var(m)
        von_neumann = float(delta2 / variance) if variance > 0 else 1.0

        # Beyond 1 std dev
        m_std = m.std()
        beyond_1std = float(np.sum(np.abs(residuals) > m_std) / n) if m_std > 0 else 0.0

        # Maximum slope
        if len(t) > 1:
            dt = np.diff(t)
            dt = np.where(dt == 0, 1e-6, dt)
            max_slope = float(np.max(np.abs(np.diff(m) / dt)))
        else:
            max_slope = 0.0

        extra = np.array([stetson_j, von_neumann, beyond_1std, max_slope])
        return np.concatenate([base_feats, extra])

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        light_curves: list[dict],
        contamination: float = 0.05,
        n_estimators: int = 100,
    ) -> "ANOMALY001":
        """
        Train Isolation Forest on light curves (unsupervised).

        Args:
            light_curves: list of dicts with keys 'times', 'mags'
            contamination: expected fraction of anomalies (default 5%)
            n_estimators: number of trees in Isolation Forest

        For production training, use Kepler/TESS non-variable stars as "normal"
        and validate against known anomalies (Boyajian's Star, etc.).
        """
        from sklearn.ensemble import IsolationForest

        logger.info(f"Training ANOMALY-001 on {len(light_curves)} light curves (unsupervised)...")
        X = np.array([
            self.extract_features(lc["times"], lc["mags"], lc.get("errors"))
            for lc in light_curves
        ])

        self._iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._iso_forest.fit(X)
        self._loaded = True
        logger.info("ANOMALY-001 trained successfully.")
        return self

    def train_synthetic(self, n_normal: int = 1000, n_anomaly: int = 50) -> "ANOMALY001":
        """Bootstrap training on synthetic data."""
        from larun.models.vardet import _synthetic_light_curve

        logger.info("Training ANOMALY-001 on synthetic data...")
        light_curves = []
        rng = np.random.default_rng(0)

        # Normal: non-variables and clean pulsators
        for _ in range(n_normal):
            t = np.sort(rng.uniform(0, 100, rng.integers(50, 200)))
            c = rng.choice([0, 2, 3])  # skip transients for "normal"
            light_curves.append(_synthetic_light_curve(t, c, rng))

        # Anomalies will be detected by contamination parameter
        return self.train(light_curves)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def anomaly_score(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> float:
        """
        Return a scalar anomaly score in [0, 1].
        0 = completely normal, 1 = maximum anomaly.
        """
        if self._iso_forest is None:
            if not self._loaded:
                self.load()
            if self._iso_forest is None:
                self.train_synthetic()

        features = self.extract_features(times, magnitudes, errors)
        # Isolation Forest decision_function returns higher values for inliers
        raw_score = self._iso_forest.decision_function(features.reshape(1, -1))[0]
        # Normalize to [0, 1]: inlier scores ~ +0.5, outlier scores ~ -0.5
        normalized = float(0.5 - raw_score)
        return max(0.0, min(1.0, normalized))

    def predict(
        self,
        times: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> tuple[str, np.ndarray]:
        """
        Classify anomaly level.

        Returns:
            (label, probability_vector)
            Note: probabilities are derived from the anomaly score,
            not a true probabilistic model.
        """
        score = self.anomaly_score(times, magnitudes, errors)

        if score >= self.THRESHOLD_STRONG:
            label = "STRONG_ANOMALY"
            proba = np.array([1.0 - score, score * 0.1, score * 0.9])
        elif score >= self.THRESHOLD_MILD:
            label = "MILD_ANOMALY"
            proba = np.array([1.0 - score, score * 0.7, score * 0.3])
        else:
            label = "NORMAL"
            proba = np.array([1.0 - score, score * 0.5, score * 0.5])

        proba = proba / proba.sum()
        return label, proba

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _get_weights(self) -> dict:
        if self._iso_forest is None:
            return {}
        return {"iso_forest_pickle": np.frombuffer(pickle.dumps(self._iso_forest), dtype=np.uint8)}

    def _load_weights(self, data) -> None:
        if "iso_forest_pickle" in data:
            self._iso_forest = pickle.loads(data["iso_forest_pickle"].tobytes())
            self._loaded = True

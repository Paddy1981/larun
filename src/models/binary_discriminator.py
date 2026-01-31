"""
LARUN Binary Discriminator
===========================
Specialized model for distinguishing eclipsing binaries from planetary transits.

Key Features:
- Secondary eclipse detection
- Depth ratio analysis
- V-shape vs U-shape transit profile
- Odd-even transit depth comparison

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BinaryResult:
    """Result of binary vs planet discrimination."""
    is_binary: bool
    binary_probability: float
    planet_probability: float
    classification: str  # "PLANET", "ECLIPSING_BINARY", "UNCERTAIN"
    
    # Diagnostic features
    secondary_eclipse_depth: float
    depth_ratio: float  # secondary/primary
    v_shape_score: float  # 0=U-shape, 1=V-shape
    odd_even_difference: float
    
    # Confidence and explanation
    confidence: float
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_binary': self.is_binary,
            'classification': self.classification,
            'binary_probability': round(self.binary_probability, 4),
            'planet_probability': round(self.planet_probability, 4),
            'confidence': round(self.confidence, 4),
            'secondary_eclipse_depth': round(self.secondary_eclipse_depth, 6),
            'depth_ratio': round(self.depth_ratio, 4),
            'v_shape_score': round(self.v_shape_score, 4),
            'odd_even_difference': round(self.odd_even_difference, 6),
            'explanation': self.explanation
        }
    
    def summary(self) -> str:
        icon = "🌟" if self.classification == "PLANET" else "⚫"
        lines = [
            "═══════════════════════════════════════════════════════════",
            f"  BINARY DISCRIMINATION RESULT {icon}",
            "═══════════════════════════════════════════════════════════",
            f"  Classification: {self.classification}",
            f"  Confidence: {self.confidence:.1%}",
            "───────────────────────────────────────────────────────────",
            f"  Planet Probability: {self.planet_probability:.1%}",
            f"  Binary Probability: {self.binary_probability:.1%}",
            "───────────────────────────────────────────────────────────",
            f"  Diagnostics:",
            f"    Secondary Eclipse Depth: {self.secondary_eclipse_depth*1e6:.0f} ppm",
            f"    Depth Ratio: {self.depth_ratio:.3f}",
            f"    V-shape Score: {self.v_shape_score:.2f}",
            f"    Odd-Even Diff: {self.odd_even_difference*1e6:.1f} ppm",
            "───────────────────────────────────────────────────────────",
            f"  {self.explanation}",
            "═══════════════════════════════════════════════════════════"
        ]
        return "\n".join(lines)


# ============================================================================
# Binary Discriminator Model
# ============================================================================

class BinaryDiscriminator:
    """
    Eclipsing binary vs planetary transit discriminator.
    
    Uses multiple diagnostic features to distinguish between
    genuine planetary transits and eclipsing binary false positives.
    
    Key diagnostics:
    1. Secondary eclipse presence and depth
    2. V-shape vs U-shape transit profile
    3. Odd-even transit depth difference (period confusion)
    4. Duration vs depth relationship
    
    Example:
        >>> discriminator = BinaryDiscriminator()
        >>> result = discriminator.analyze(time, flux, period, t0)
        >>> print(f"Classification: {result.classification}")
    """
    
    # Thresholds for classification
    SECONDARY_THRESHOLD = 0.1    # depth ratio for significant secondary
    V_SHAPE_THRESHOLD = 0.5      # V-shape score threshold
    ODD_EVEN_THRESHOLD = 0.001   # 1000 ppm odd-even difference
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1024, 1),
        model_path: Optional[str] = None
    ):
        """
        Initialize the binary discriminator.
        
        Args:
            input_shape: Shape of input phase-folded curve
            model_path: Path to pre-trained model (optional)
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self._is_built = False
    
    def build(self) -> None:
        """Build the CNN model for classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            
            # Input: phase-folded light curve
            inputs = layers.Input(shape=self.input_shape, name="phase_curve")
            
            # Feature extraction with attention
            x = layers.Conv1D(32, 16, activation='relu', padding='same')(inputs)
            x = layers.MaxPooling1D(4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv1D(64, 8, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(4)(x)
            x = layers.BatchNormalization()(x)
            
            # Attention mechanism for transit shape
            attention = layers.Conv1D(1, 1, activation='sigmoid')(x)
            x = layers.Multiply()([x, attention])
            
            x = layers.Conv1D(64, 4, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
            
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            
            # Binary output: 0=planet, 1=binary
            output = layers.Dense(1, activation='sigmoid', name='binary_prob')(x)
            
            self.model = Model(inputs=inputs, outputs=output, name="BinaryDiscriminator")
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self._is_built = True
            logger.info(f"Built BinaryDiscriminator with {self.model.count_params():,} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. Using rule-based discrimination.")
            self._is_built = True
    
    def analyze(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        depth: Optional[float] = None
    ) -> BinaryResult:
        """
        Analyze a transit signal for binary signatures.
        
        Args:
            time: Time array (days)
            flux: Flux array (normalized)
            period: Orbital period (days)
            t0: Mid-transit epoch
            depth: Transit depth (optional, will be measured)
            
        Returns:
            BinaryResult with classification
        """
        if not self._is_built:
            self.build()
        
        # Phase fold the light curve
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Sort by phase for analysis
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_sorted = flux[sort_idx]
        
        # Measure diagnostic features
        primary_depth = self._measure_primary_depth(phase_sorted, flux_sorted)
        secondary_depth = self._measure_secondary_depth(phase_sorted, flux_sorted)
        v_shape = self._measure_v_shape(phase_sorted, flux_sorted)
        odd_even_diff = self._measure_odd_even(time, flux, period, t0)
        
        # Calculate depth ratio
        depth_ratio = secondary_depth / primary_depth if primary_depth > 0 else 0.0
        
        # Binary probability based on features
        binary_prob = self._calculate_binary_probability(
            depth_ratio, v_shape, odd_even_diff, primary_depth
        )
        
        planet_prob = 1.0 - binary_prob
        
        # Classification
        if binary_prob > 0.7:
            classification = "ECLIPSING_BINARY"
            is_binary = True
        elif binary_prob < 0.3:
            classification = "PLANET"
            is_binary = False
        else:
            classification = "UNCERTAIN"
            is_binary = binary_prob > 0.5
        
        # Confidence
        confidence = max(binary_prob, planet_prob)
        
        # Generate explanation
        explanation = self._generate_explanation(
            classification, depth_ratio, v_shape, odd_even_diff, secondary_depth
        )
        
        return BinaryResult(
            is_binary=is_binary,
            binary_probability=binary_prob,
            planet_probability=planet_prob,
            classification=classification,
            secondary_eclipse_depth=secondary_depth,
            depth_ratio=depth_ratio,
            v_shape_score=v_shape,
            odd_even_difference=odd_even_diff,
            confidence=confidence,
            explanation=explanation
        )
    
    def _measure_primary_depth(
        self,
        phase: np.ndarray,
        flux: np.ndarray
    ) -> float:
        """Measure primary transit/eclipse depth."""
        # Primary transit is centered at phase 0
        in_transit = np.abs(phase) < 0.1
        out_of_transit = np.abs(phase) > 0.3
        
        if np.sum(in_transit) < 3 or np.sum(out_of_transit) < 3:
            return 0.0
        
        baseline = np.median(flux[out_of_transit])
        in_transit_flux = np.median(flux[in_transit])
        
        return max(0, baseline - in_transit_flux)
    
    def _measure_secondary_depth(
        self,
        phase: np.ndarray,
        flux: np.ndarray
    ) -> float:
        """Measure secondary eclipse depth at phase 0.5."""
        # Secondary eclipse is centered at phase 0.5 (or -0.5)
        in_secondary = np.abs(np.abs(phase) - 0.5) < 0.1
        out_of_transit = (np.abs(phase) > 0.2) & (np.abs(phase) < 0.4)
        
        if np.sum(in_secondary) < 3 or np.sum(out_of_transit) < 3:
            return 0.0
        
        baseline = np.median(flux[out_of_transit])
        secondary_flux = np.median(flux[in_secondary])
        
        return max(0, baseline - secondary_flux)
    
    def _measure_v_shape(
        self,
        phase: np.ndarray,
        flux: np.ndarray
    ) -> float:
        """
        Measure V-shape score (0=U-shape, 1=V-shape).
        
        Grazing eclipses are V-shaped, planetary transits are U-shaped.
        """
        # Get transit region
        in_transit = np.abs(phase) < 0.1
        transit_phase = phase[in_transit]
        transit_flux = flux[in_transit]
        
        if len(transit_flux) < 10:
            return 0.5  # Unknown
        
        # Sort by absolute phase (distance from center)
        dist_from_center = np.abs(transit_phase)
        sort_idx = np.argsort(dist_from_center)
        sorted_flux = transit_flux[sort_idx]
        
        # Compare center to edges within transit
        n = len(sorted_flux)
        center_flux = np.mean(sorted_flux[:n//3])
        edge_flux = np.mean(sorted_flux[-n//3:])
        
        # U-shape: center is flat (center ≈ edge within transit)
        # V-shape: center is deeper than edges within transit
        
        transit_depth = np.max(flux[~in_transit]) - center_flux if np.sum(~in_transit) > 0 else 0.1
        
        if transit_depth == 0:
            return 0.5
        
        # V-shape score: how much does center differ from edges
        flatness = abs(center_flux - edge_flux) / transit_depth
        
        return min(1.0, flatness * 2)  # Scale so 0.5 difference = V-shape
    
    def _measure_odd_even(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float
    ) -> float:
        """Measure odd-even transit depth difference."""
        # Get transit number for each point
        transit_num = np.round((time - t0) / period)
        
        odd_mask = (transit_num % 2 == 1) & (np.abs((time - t0) % period) < period * 0.1)
        even_mask = (transit_num % 2 == 0) & (np.abs((time - t0) % period) < period * 0.1)
        
        if np.sum(odd_mask) < 5 or np.sum(even_mask) < 5:
            return 0.0
        
        odd_depth = 1.0 - np.median(flux[odd_mask])
        even_depth = 1.0 - np.median(flux[even_mask])
        
        return abs(odd_depth - even_depth)
    
    def _calculate_binary_probability(
        self,
        depth_ratio: float,
        v_shape: float,
        odd_even_diff: float,
        primary_depth: float
    ) -> float:
        """Calculate probability that signal is from an eclipsing binary."""
        prob = 0.1  # Base prior
        
        # Significant secondary eclipse suggests binary
        if depth_ratio > 0.05:
            prob += 0.3 * min(1.0, depth_ratio / 0.3)
        
        # V-shape transit suggests grazing binary
        if v_shape > 0.3:
            prob += 0.2 * min(1.0, v_shape / 0.5)
        
        # Odd-even difference suggests period is half true period (EB)
        if odd_even_diff > 0.0003:  # 300 ppm
            prob += 0.25 * min(1.0, odd_even_diff / 0.003)
        
        # Very deep eclipses (>3%) are likely binaries
        if primary_depth > 0.03:
            prob += 0.3 * min(1.0, (primary_depth - 0.03) / 0.1)
        
        return min(0.99, max(0.01, prob))
    
    def _generate_explanation(
        self,
        classification: str,
        depth_ratio: float,
        v_shape: float,
        odd_even_diff: float,
        secondary_depth: float
    ) -> str:
        """Generate human-readable explanation."""
        reasons = []
        
        if classification == "PLANET":
            if depth_ratio < 0.05:
                reasons.append("no significant secondary eclipse")
            if v_shape < 0.3:
                reasons.append("U-shaped transit profile")
            if odd_even_diff < 0.0003:
                reasons.append("consistent odd/even depths")
            return "Planet-like: " + ", ".join(reasons) if reasons else "Features consistent with planet"
        
        elif classification == "ECLIPSING_BINARY":
            if depth_ratio > 0.1:
                reasons.append(f"secondary eclipse detected ({secondary_depth*1e6:.0f} ppm)")
            if v_shape > 0.5:
                reasons.append("V-shaped eclipse profile")
            if odd_even_diff > 0.001:
                reasons.append("significant odd/even depth difference")
            return "Binary-like: " + ", ".join(reasons) if reasons else "Features consistent with EB"
        
        else:
            return "Ambiguous signal: additional observations recommended"
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the discriminator model."""
        if not self._is_built:
            self.build()
        
        if self.model is None:
            raise ValueError("Model not available. TensorFlow required for training.")
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            verbose=1
        )
        
        return history.history
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        if self.model:
            self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self._is_built = True


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Binary Discriminator...")
    print("=" * 60)
    
    discriminator = BinaryDiscriminator()
    
    # Create synthetic planet transit
    t = np.linspace(0, 30, 5000)
    period = 3.5
    t0 = 0.5
    
    # Planet: U-shaped, no secondary
    flux_planet = np.ones_like(t)
    phase = ((t - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < 0.02
    flux_planet[in_transit] = 0.99  # 1% depth, flat bottom
    flux_planet += np.random.normal(0, 0.001, len(t))
    
    print("\nTest 1: Planet-like signal")
    result_planet = discriminator.analyze(t, flux_planet, period, t0)
    print(result_planet.summary())
    
    # Create synthetic eclipsing binary
    flux_eb = np.ones_like(t)
    # Primary eclipse
    primary = np.abs(phase) < 0.03
    flux_eb[primary] = 0.90  # 10% depth
    # Secondary eclipse
    secondary = np.abs(np.abs(phase) - 0.5) < 0.03
    flux_eb[secondary] = 0.97  # 3% depth
    flux_eb += np.random.normal(0, 0.001, len(t))
    
    print("\nTest 2: EB-like signal")
    result_eb = discriminator.analyze(t, flux_eb, period, t0)
    print(result_eb.summary())
    
    print("\nTest complete!")

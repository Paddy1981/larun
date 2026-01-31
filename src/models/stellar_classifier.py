"""
LARUN Stellar Classifier
========================
Specialized model for stellar spectral classification.

Classifies stars by:
- Spectral Type: O, B, A, F, G, K, M
- Luminosity Class: I (Supergiant) to V (Dwarf)

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
class StellarClassifierResult:
    """Result of stellar classification."""
    spectral_type: str           # O, B, A, F, G, K, M
    luminosity_class: str        # I, II, III, IV, V
    full_classification: str     # e.g., "G2V"
    spectral_probabilities: Dict[str, float]
    luminosity_probabilities: Dict[str, float]
    confidence: float
    estimated_teff: float        # Estimated Teff from classification
    estimated_radius: float      # Estimated R/R_sun
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spectral_type': self.spectral_type,
            'luminosity_class': self.luminosity_class,
            'full_classification': self.full_classification,
            'confidence': round(self.confidence, 4),
            'estimated_teff': round(self.estimated_teff, 0),
            'estimated_radius': round(self.estimated_radius, 3),
            'spectral_probabilities': {k: round(v, 4) for k, v in self.spectral_probabilities.items()},
            'luminosity_probabilities': {k: round(v, 4) for k, v in self.luminosity_probabilities.items()}
        }
    
    def summary(self) -> str:
        lines = [
            "═══════════════════════════════════════════════════════════",
            "  STELLAR CLASSIFICATION RESULT",
            "═══════════════════════════════════════════════════════════",
            f"  Classification: {self.full_classification}",
            f"  Confidence: {self.confidence:.1%}",
            "───────────────────────────────────────────────────────────",
            f"  Spectral Type: {self.spectral_type}",
            f"  Luminosity Class: {self.luminosity_class}",
            f"  Estimated Teff: {self.estimated_teff:.0f} K",
            f"  Estimated Radius: {self.estimated_radius:.2f} R☉",
            "═══════════════════════════════════════════════════════════"
        ]
        return "\n".join(lines)


# ============================================================================
# Stellar Classifier Model
# ============================================================================

class StellarClassifier:
    """
    Stellar spectral type and luminosity class classifier.
    
    Uses a multi-head CNN architecture to jointly predict
    spectral type (OBAFGKM) and luminosity class (I-V).
    
    Example:
        >>> classifier = StellarClassifier()
        >>> classifier.build()
        >>> result = classifier.classify(light_curve)
        >>> print(f"Star type: {result.full_classification}")
    """
    
    # Spectral types with approximate Teff ranges
    SPECTRAL_TYPES = {
        'O': (30000, 50000),
        'B': (10000, 30000),
        'A': (7500, 10000),
        'F': (6000, 7500),
        'G': (5200, 6000),
        'K': (3700, 5200),
        'M': (2400, 3700)
    }
    
    # Luminosity classes with approximate radius ranges (R_sun)
    LUMINOSITY_CLASSES = {
        'I': (10.0, 100.0),     # Supergiant
        'II': (5.0, 15.0),       # Bright giant
        'III': (2.0, 10.0),      # Giant
        'IV': (1.2, 3.0),        # Subgiant
        'V': (0.1, 1.5)          # Main sequence
    }
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (1024, 1),
        model_path: Optional[str] = None
    ):
        """
        Initialize the stellar classifier.
        
        Args:
            input_shape: Shape of input light curve
            model_path: Path to pre-trained model (optional)
        """
        self.input_shape = input_shape
        self.model_path = model_path
        self.model = None
        self._is_built = False
        
        # Class mappings
        self.spectral_classes = list(self.SPECTRAL_TYPES.keys())
        self.luminosity_classes = list(self.LUMINOSITY_CLASSES.keys())
    
    def build(self) -> None:
        """Build the multi-head classification model."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            
            # Input
            inputs = layers.Input(shape=self.input_shape, name="light_curve")
            
            # Shared feature extractor
            x = layers.Conv1D(32, 16, activation='relu', padding='same')(inputs)
            x = layers.MaxPooling1D(4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv1D(64, 8, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv1D(128, 4, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(4)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.3)(x)
            
            # Shared dense layer
            shared = layers.Dense(64, activation='relu', name="shared_features")(x)
            
            # Spectral type head
            spectral_dense = layers.Dense(32, activation='relu')(shared)
            spectral_output = layers.Dense(
                len(self.spectral_classes),
                activation='softmax',
                name='spectral_type'
            )(spectral_dense)
            
            # Luminosity class head
            lum_dense = layers.Dense(32, activation='relu')(shared)
            luminosity_output = layers.Dense(
                len(self.luminosity_classes),
                activation='softmax',
                name='luminosity_class'
            )(lum_dense)
            
            # Build model
            self.model = Model(
                inputs=inputs,
                outputs=[spectral_output, luminosity_output],
                name="StellarClassifier"
            )
            
            self.model.compile(
                optimizer='adam',
                loss={
                    'spectral_type': 'sparse_categorical_crossentropy',
                    'luminosity_class': 'sparse_categorical_crossentropy'
                },
                metrics=['accuracy']
            )
            
            self._is_built = True
            logger.info(f"Built StellarClassifier with {self.model.count_params():,} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available. Using rule-based classification.")
            self._is_built = True
    
    def classify(
        self,
        light_curve: np.ndarray,
        teff_hint: Optional[float] = None
    ) -> StellarClassifierResult:
        """
        Classify a star from its light curve.
        
        Args:
            light_curve: Flux array of shape (n_points,) or (n_points, 1)
            teff_hint: Optional Teff hint from Gaia for improved accuracy
            
        Returns:
            StellarClassifierResult with classification
        """
        if not self._is_built:
            self.build()
        
        # If we have a model and no Teff hint, use the model
        if self.model is not None and teff_hint is None:
            return self._classify_with_model(light_curve)
        
        # Otherwise use rule-based classification
        return self._classify_rule_based(light_curve, teff_hint)
    
    def _classify_with_model(self, light_curve: np.ndarray) -> StellarClassifierResult:
        """Classify using the neural network model."""
        # Prepare input
        if light_curve.ndim == 1:
            light_curve = light_curve.reshape(1, -1, 1)
        elif light_curve.ndim == 2:
            light_curve = light_curve.reshape(1, *light_curve.shape)
        
        # Predict
        spectral_probs, lum_probs = self.model.predict(light_curve, verbose=0)
        
        # Get best classes
        spectral_idx = np.argmax(spectral_probs[0])
        lum_idx = np.argmax(lum_probs[0])
        
        spectral_type = self.spectral_classes[spectral_idx]
        luminosity_class = self.luminosity_classes[lum_idx]
        
        # Confidence is minimum of both predictions
        confidence = min(spectral_probs[0][spectral_idx], lum_probs[0][lum_idx])
        
        # Estimate Teff and radius
        teff_range = self.SPECTRAL_TYPES[spectral_type]
        estimated_teff = sum(teff_range) / 2
        
        radius_range = self.LUMINOSITY_CLASSES[luminosity_class]
        estimated_radius = sum(radius_range) / 2
        
        return StellarClassifierResult(
            spectral_type=spectral_type,
            luminosity_class=luminosity_class,
            full_classification=f"{spectral_type}2{luminosity_class}",
            spectral_probabilities={
                k: float(v) for k, v in zip(self.spectral_classes, spectral_probs[0])
            },
            luminosity_probabilities={
                k: float(v) for k, v in zip(self.luminosity_classes, lum_probs[0])
            },
            confidence=float(confidence),
            estimated_teff=estimated_teff,
            estimated_radius=estimated_radius
        )
    
    def _classify_rule_based(
        self,
        light_curve: np.ndarray,
        teff_hint: Optional[float] = None
    ) -> StellarClassifierResult:
        """Rule-based classification using light curve features."""
        # Extract features from light curve
        flux = light_curve.flatten()
        flux_std = np.std(flux)
        flux_skew = self._skewness(flux)
        
        # If we have Teff, use it directly
        if teff_hint is not None:
            spectral_type = self._teff_to_spectral(teff_hint)
            # Estimate luminosity from variability
            if flux_std > 0.1:
                luminosity_class = 'I'  # High variability suggests giant
            elif flux_std > 0.02:
                luminosity_class = 'III'
            else:
                luminosity_class = 'V'
            
            teff_range = self.SPECTRAL_TYPES[spectral_type]
            radius_range = self.LUMINOSITY_CLASSES[luminosity_class]
            
            return StellarClassifierResult(
                spectral_type=spectral_type,
                luminosity_class=luminosity_class,
                full_classification=f"{spectral_type}2{luminosity_class}",
                spectral_probabilities={spectral_type: 0.9},
                luminosity_probabilities={luminosity_class: 0.8},
                confidence=0.85,
                estimated_teff=teff_hint,
                estimated_radius=sum(radius_range) / 2
            )
        
        # Without Teff, make educated guesses
        # Most stars are G/K dwarfs
        spectral_probs = {
            'O': 0.01, 'B': 0.03, 'A': 0.05, 'F': 0.15,
            'G': 0.30, 'K': 0.35, 'M': 0.11
        }
        
        # Adjust based on variability
        if flux_std > 0.05:
            # High variability might indicate M dwarf or giant
            spectral_probs['M'] *= 2
            spectral_probs['K'] *= 1.5
        
        # Normalize
        total = sum(spectral_probs.values())
        spectral_probs = {k: v/total for k, v in spectral_probs.items()}
        
        spectral_type = max(spectral_probs, key=spectral_probs.get)
        
        # Luminosity based on variability
        if flux_std > 0.1:
            lum_probs = {'I': 0.5, 'II': 0.2, 'III': 0.2, 'IV': 0.05, 'V': 0.05}
        elif flux_std > 0.02:
            lum_probs = {'I': 0.1, 'II': 0.15, 'III': 0.3, 'IV': 0.25, 'V': 0.2}
        else:
            lum_probs = {'I': 0.02, 'II': 0.03, 'III': 0.1, 'IV': 0.15, 'V': 0.7}
        
        luminosity_class = max(lum_probs, key=lum_probs.get)
        
        confidence = spectral_probs[spectral_type] * lum_probs[luminosity_class]
        
        teff_range = self.SPECTRAL_TYPES[spectral_type]
        radius_range = self.LUMINOSITY_CLASSES[luminosity_class]
        
        return StellarClassifierResult(
            spectral_type=spectral_type,
            luminosity_class=luminosity_class,
            full_classification=f"{spectral_type}2{luminosity_class}",
            spectral_probabilities=spectral_probs,
            luminosity_probabilities=lum_probs,
            confidence=confidence,
            estimated_teff=sum(teff_range) / 2,
            estimated_radius=sum(radius_range) / 2
        )
    
    def _teff_to_spectral(self, teff: float) -> str:
        """Convert Teff to spectral type."""
        for spec_type, (tmin, tmax) in self.SPECTRAL_TYPES.items():
            if tmin <= teff < tmax:
                return spec_type
        if teff >= 50000:
            return 'O'
        return 'M'
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of array."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.sum(((x - mean) / std) ** 3) / n
    
    def train(
        self,
        X: np.ndarray,
        y_spectral: np.ndarray,
        y_luminosity: np.ndarray,
        epochs: int = 50,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Light curves of shape (n_samples, n_points, 1)
            y_spectral: Spectral type labels (0-6 for OBAFGKM)
            y_luminosity: Luminosity class labels (0-4 for I-V)
            epochs: Training epochs
            validation_split: Fraction for validation
            
        Returns:
            Training history
        """
        if not self._is_built:
            self.build()
        
        if self.model is None:
            raise ValueError("Model not available. TensorFlow required for training.")
        
        history = self.model.fit(
            X,
            {'spectral_type': y_spectral, 'luminosity_class': y_luminosity},
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
            logger.info(f"Saved StellarClassifier to {path}")
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self._is_built = True
        logger.info(f"Loaded StellarClassifier from {path}")


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("Testing Stellar Classifier...")
    print("=" * 60)
    
    # Create classifier
    classifier = StellarClassifier()
    
    # Test with synthetic light curve
    test_lc = np.random.normal(1.0, 0.01, 1024).astype(np.float32)
    
    # With Teff hint (Sun-like)
    result = classifier.classify(test_lc, teff_hint=5778)
    print(result.summary())
    
    # Without Teff hint
    result2 = classifier.classify(test_lc)
    print(f"\nWithout hint: {result2.full_classification} ({result2.confidence:.0%})")
    
    print("\nTest complete!")

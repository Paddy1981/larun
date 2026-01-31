"""
LARUN Data Augmentation Module
==============================
Data augmentation techniques for improving model training on light curve data.

These augmentations help prevent overfitting and improve generalization
by creating variations of the training data that preserve the transit signal.

Created by: Padmanaban Veeraragavalu (Larun Engineering)
Reference: docs/research/TINYML_OPTIMIZATION.md
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    noise_level: float = 0.002       # Gaussian noise std
    time_shift_max: float = 0.1      # Max fractional time shift
    flux_scale_range: Tuple[float, float] = (0.95, 1.05)  # Flux scaling range
    dropout_rate: float = 0.05       # Random point dropout rate
    mixup_alpha: float = 0.2         # Mixup interpolation alpha
    enable_noise: bool = True
    enable_time_shift: bool = True
    enable_flux_scale: bool = True
    enable_dropout: bool = False     # Can hurt transit detection
    enable_mixup: bool = False       # For advanced training


class LightCurveAugmenter:
    """
    Augment light curve data for improved model training.

    Implements astronomy-aware augmentations that preserve transit signals
    while adding realistic variations.

    Example:
        >>> augmenter = LightCurveAugmenter()
        >>> X_aug, y_aug = augmenter.augment_batch(X_train, y_train, factor=3)
        >>> print(f"Augmented: {len(X_train)} -> {len(X_aug)} samples")
    """

    def __init__(self, config: Optional[AugmentationConfig] = None, seed: Optional[int] = None):
        """
        Initialize augmenter with configuration.

        Args:
            config: AugmentationConfig or None for defaults
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def add_gaussian_noise(
        self,
        flux: np.ndarray,
        noise_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to flux values.

        This simulates photometric noise that varies between observations.

        Args:
            flux: Flux array (normalized around 1.0)
            noise_level: Noise standard deviation (default from config)

        Returns:
            Noisy flux array
        """
        if noise_level is None:
            noise_level = self.config.noise_level

        noise = self.rng.normal(0, noise_level, flux.shape)
        return flux + noise

    def time_shift(
        self,
        flux: np.ndarray,
        shift_fraction: Optional[float] = None
    ) -> np.ndarray:
        """
        Shift the light curve in time (circular shift).

        This simulates different observation start times while preserving
        the periodic signal structure.

        Args:
            flux: Flux array
            shift_fraction: Fraction of array to shift (random if None)

        Returns:
            Time-shifted flux array
        """
        if shift_fraction is None:
            shift_fraction = self.rng.uniform(
                -self.config.time_shift_max,
                self.config.time_shift_max
            )

        shift_points = int(len(flux) * shift_fraction)
        return np.roll(flux, shift_points, axis=-1)

    def flux_scale(
        self,
        flux: np.ndarray,
        scale_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Scale flux values around the median.

        This simulates variations in stellar brightness and instrument response.

        Args:
            flux: Flux array (normalized around 1.0)
            scale_factor: Scale factor (random from range if None)

        Returns:
            Scaled flux array
        """
        if scale_factor is None:
            scale_factor = self.rng.uniform(*self.config.flux_scale_range)

        # Scale around median to preserve transit depth ratio
        median = np.median(flux)
        return median + (flux - median) * scale_factor

    def random_dropout(
        self,
        flux: np.ndarray,
        dropout_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Randomly mask data points (simulate missing data).

        Args:
            flux: Flux array
            dropout_rate: Fraction of points to mask (default from config)

        Returns:
            Flux array with random points replaced by median
        """
        if dropout_rate is None:
            dropout_rate = self.config.dropout_rate

        mask = self.rng.random(flux.shape) < dropout_rate
        result = flux.copy()
        result[mask] = np.median(flux)
        return result

    def inject_transit(
        self,
        flux: np.ndarray,
        period: float,
        depth: float,
        duration_frac: float = 0.02,
        t0_frac: float = 0.0
    ) -> np.ndarray:
        """
        Inject a synthetic transit signal into a light curve.

        Useful for creating positive training examples from negative samples.

        Args:
            flux: Flux array (normalized)
            period: Transit period in array indices
            depth: Transit depth (fractional, e.g., 0.01 for 1%)
            duration_frac: Transit duration as fraction of period
            t0_frac: Phase offset for transit center

        Returns:
            Flux array with injected transit
        """
        n_points = len(flux)
        indices = np.arange(n_points)

        # Phase fold
        phase = ((indices - t0_frac * period) % period) / period

        # Create box-shaped transit
        in_transit = phase < duration_frac
        result = flux.copy()
        result[in_transit] -= depth

        return result

    def augment_single(
        self,
        flux: np.ndarray,
        augmentations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply random augmentations to a single light curve.

        Args:
            flux: Flux array
            augmentations: List of augmentation names to apply, or None for config

        Returns:
            Augmented flux array
        """
        result = flux.copy()

        if augmentations is None:
            augmentations = []
            if self.config.enable_noise:
                augmentations.append('noise')
            if self.config.enable_time_shift:
                augmentations.append('time_shift')
            if self.config.enable_flux_scale:
                augmentations.append('flux_scale')
            if self.config.enable_dropout:
                augmentations.append('dropout')

        # Shuffle order for variety
        self.rng.shuffle(augmentations)

        for aug in augmentations:
            if aug == 'noise':
                result = self.add_gaussian_noise(result)
            elif aug == 'time_shift':
                result = self.time_shift(result)
            elif aug == 'flux_scale':
                result = self.flux_scale(result)
            elif aug == 'dropout':
                result = self.random_dropout(result)

        return result

    def augment_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factor: int = 2,
        keep_original: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of training data.

        Args:
            X: Input features (N, length, 1) or (N, length)
            y: Labels (N,)
            factor: Number of augmented copies per original
            keep_original: Whether to include original samples

        Returns:
            X_aug: Augmented features
            y_aug: Corresponding labels
        """
        original_shape = X.shape
        was_3d = len(original_shape) == 3

        # Flatten to 2D for processing
        if was_3d:
            X = X.reshape(X.shape[0], -1)

        X_list = []
        y_list = []

        if keep_original:
            X_list.append(X)
            y_list.append(y)

        for _ in range(factor):
            X_aug = np.array([self.augment_single(x) for x in X])
            X_list.append(X_aug)
            y_list.append(y.copy())

        X_result = np.concatenate(X_list, axis=0)
        y_result = np.concatenate(y_list, axis=0)

        # Restore shape
        if was_3d:
            X_result = X_result.reshape(-1, original_shape[1], original_shape[2])

        # Shuffle
        indices = self.rng.permutation(len(X_result))
        X_result = X_result[indices]
        y_result = y_result[indices]

        logger.info(f"Augmented {len(X)} -> {len(X_result)} samples (factor={factor})")

        return X_result, y_result

    def augment_batch_balanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data to achieve balanced class distribution.

        This method augments minority classes more aggressively to match
        the majority class count, solving class imbalance issues.

        Args:
            X: Input features (N, length, 1) or (N, length)
            y: Labels (N,)
            target_per_class: Target samples per class (default: max class count)

        Returns:
            X_balanced: Balanced augmented features
            y_balanced: Corresponding labels
        """
        original_shape = X.shape
        was_3d = len(original_shape) == 3

        # Flatten to 2D for processing
        if was_3d:
            X = X.reshape(X.shape[0], -1)

        # Get class distribution
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        if target_per_class is None:
            target_per_class = counts.max()

        logger.info(f"Balancing classes to {target_per_class} samples each")
        logger.info(f"Original distribution: {dict(zip(classes, counts))}")

        X_list = []
        y_list = []

        for cls, count in zip(classes, counts):
            # Get samples for this class
            mask = y == cls
            X_cls = X[mask]

            # Always keep originals
            X_list.append(X_cls)
            y_list.append(np.full(len(X_cls), cls))

            # How many more do we need?
            needed = target_per_class - count

            if needed > 0:
                # Generate augmented samples
                aug_samples = []
                for _ in range(needed):
                    # Randomly pick an original sample to augment
                    idx = self.rng.integers(0, len(X_cls))
                    aug_sample = self.augment_single(X_cls[idx])
                    aug_samples.append(aug_sample)

                X_list.append(np.array(aug_samples))
                y_list.append(np.full(needed, cls))

        X_result = np.concatenate(X_list, axis=0)
        y_result = np.concatenate(y_list, axis=0)

        # Restore shape
        if was_3d:
            X_result = X_result.reshape(-1, original_shape[1], original_shape[2])

        # Shuffle
        indices = self.rng.permutation(len(X_result))
        X_result = X_result[indices]
        y_result = y_result[indices]

        # Log final distribution
        final_classes, final_counts = np.unique(y_result, return_counts=True)
        logger.info(f"Balanced distribution: {dict(zip(final_classes, final_counts))}")
        logger.info(f"Total samples: {len(X)} -> {len(X_result)}")

        return X_result, y_result

    def mixup(
        self,
        X1: np.ndarray,
        y1: np.ndarray,
        X2: np.ndarray,
        y2: np.ndarray,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation between two batches.

        Mixup creates virtual training examples by interpolating between
        pairs of samples. This can improve generalization.

        Args:
            X1, y1: First batch
            X2, y2: Second batch
            alpha: Beta distribution parameter (default from config)

        Returns:
            X_mixed, y_mixed: Mixed samples and soft labels
        """
        if alpha is None:
            alpha = self.config.mixup_alpha

        # Sample lambda from beta distribution
        lam = self.rng.beta(alpha, alpha, size=len(X1))
        lam = lam.reshape(-1, *([1] * (len(X1.shape) - 1)))

        X_mixed = lam * X1 + (1 - lam) * X2

        # For classification, return soft labels
        lam_flat = lam.flatten()
        y_mixed = np.column_stack([
            y1 * lam_flat + y2 * (1 - lam_flat)
            for y1, y2 in zip(y1.T if y1.ndim > 1 else [y1],
                              y2.T if y2.ndim > 1 else [y2])
        ])

        if y_mixed.shape[1] == 1:
            y_mixed = y_mixed.flatten()

        return X_mixed, y_mixed


def create_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_factor: int = 3,
    config: Optional[AugmentationConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create augmented dataset.

    Args:
        X: Training features
        y: Training labels
        augmentation_factor: How many times to augment
        config: Augmentation configuration

    Returns:
        Augmented X and y arrays

    Example:
        >>> X_aug, y_aug = create_augmented_dataset(X_train, y_train, factor=3)
    """
    augmenter = LightCurveAugmenter(config)
    return augmenter.augment_batch(X, y, factor=augmentation_factor)


def create_balanced_dataset(
    X: np.ndarray,
    y: np.ndarray,
    target_per_class: Optional[int] = None,
    config: Optional[AugmentationConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a class-balanced augmented dataset.

    Augments minority classes more to achieve equal representation.

    Args:
        X: Training features
        y: Training labels
        target_per_class: Target samples per class (default: max class count)
        config: Augmentation configuration

    Returns:
        Balanced X and y arrays

    Example:
        >>> X_bal, y_bal = create_balanced_dataset(X_train, y_train)
    """
    augmenter = LightCurveAugmenter(config)
    return augmenter.augment_batch_balanced(X, y, target_per_class)


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting so minority classes have higher weight.

    Args:
        y: Label array

    Returns:
        Dictionary mapping class index to weight

    Example:
        >>> class_weights = compute_class_weights(y_train)
        >>> model.fit(X, y, class_weight=class_weights)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    # Inverse frequency weighting: weight = n_samples / (n_classes * count)
    weights = n_samples / (n_classes * counts)

    return {cls: weight for cls, weight in zip(classes, weights)}


# ============================================================================
# Transit-Specific Augmentations
# ============================================================================

def augment_transit_depth(
    flux: np.ndarray,
    depth_factor_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Modify transit depth while preserving shape.

    Args:
        flux: Light curve with transit
        depth_factor_range: Range for depth modification

    Returns:
        Modified flux
    """
    # Find transit (minimum flux region)
    threshold = np.percentile(flux, 10)
    in_transit = flux <= threshold

    if not np.any(in_transit):
        return flux

    # Calculate current depth
    out_transit_mean = np.mean(flux[~in_transit])
    in_transit_mean = np.mean(flux[in_transit])
    current_depth = out_transit_mean - in_transit_mean

    # New depth
    factor = np.random.uniform(*depth_factor_range)
    new_depth = current_depth * factor

    # Apply
    result = flux.copy()
    depth_diff = new_depth - current_depth
    result[in_transit] -= depth_diff

    return result


def augment_transit_duration(
    flux: np.ndarray,
    duration_factor_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Modify transit duration (ingress/egress steepness).

    This is more complex and approximates duration changes.

    Args:
        flux: Light curve with transit
        duration_factor_range: Range for duration modification

    Returns:
        Modified flux (approximate)
    """
    # This is a simplified version - full implementation would
    # require knowing the transit model parameters
    # For now, we just apply a time stretch around the transit

    # Find transit center
    min_val = np.min(flux)
    median_val = np.median(flux)
    # Use half-depth threshold to safely catch transit
    threshold = (min_val + median_val) / 2
    transit_indices = np.where(flux < threshold)[0]
    
    if len(transit_indices) == 0:
        center_idx = np.argmin(flux)
    else:
        # Use mean of transit indices as center
        center_idx = int(np.mean(transit_indices))

    n = len(flux)

    # Apply local time stretch
    factor = np.random.uniform(*duration_factor_range)

    # Simple approach: interpolate around transit
    indices = np.arange(n)
    stretched_indices = center_idx + (indices - center_idx) / factor
    stretched_indices = np.clip(stretched_indices, 0, n - 1)

    return np.interp(stretched_indices, indices, flux)


if __name__ == '__main__':
    # Quick test
    print("Testing Data Augmentation...")

    # Create synthetic light curve with transit
    np.random.seed(42)
    n_points = 1024
    flux = np.ones(n_points) + np.random.normal(0, 0.001, n_points)

    # Add transit
    transit_start = 400
    transit_end = 450
    flux[transit_start:transit_end] -= 0.01

    # Create augmenter
    augmenter = LightCurveAugmenter()

    # Test single augmentation
    flux_aug = augmenter.augment_single(flux)
    print(f"Original std: {np.std(flux):.6f}")
    print(f"Augmented std: {np.std(flux_aug):.6f}")

    # Test batch augmentation
    X = np.random.randn(10, 1024, 1).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_aug, y_aug = augmenter.augment_batch(X, y, factor=3)
    print(f"Batch augmented: {len(X)} -> {len(X_aug)} samples")
    print(f"Class distribution: {np.bincount(y_aug)}")

    print("Augmentation test passed!")

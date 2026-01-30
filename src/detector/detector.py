"""
Spectral Anomaly Detector
=========================
Main detection pipeline for identifying astronomical phenomena in spectral data.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection result."""
    detection_id: str
    object_id: str
    classification: str
    confidence: float
    timestamp: datetime
    
    # Transit-specific parameters
    transit_depth: Optional[float] = None  # Fractional flux drop
    transit_duration: Optional[float] = None  # Hours
    transit_midpoint: Optional[float] = None  # Time of mid-transit
    period: Optional[float] = None  # Days
    
    # Signal quality
    snr: float = 0.0
    is_significant: bool = False
    
    # Raw data
    flux_data: Optional[np.ndarray] = None
    time_data: Optional[np.ndarray] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "object_id": self.object_id,
            "classification": self.classification,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "transit_depth": self.transit_depth,
            "transit_duration": self.transit_duration,
            "transit_midpoint": self.transit_midpoint,
            "period": self.period,
            "snr": self.snr,
            "is_significant": self.is_significant,
            "metadata": self.metadata.copy()  # Return copy to prevent mutation
        }


@dataclass
class DetectionBatch:
    """A batch of detections from a single run."""
    batch_id: str
    run_timestamp: datetime
    detections: List[Detection]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def significant_detections(self) -> List[Detection]:
        return [d for d in self.detections if d.is_significant]
    
    @property
    def transit_candidates(self) -> List[Detection]:
        return [d for d in self.detections 
                if d.classification == "planetary_transit" and d.is_significant]


class SpectralDetector:
    """
    Main detector class for spectral analysis.
    
    Combines:
    - TinyML model for classification
    - Signal processing for transit detection
    - Statistical validation
    """
    
    def __init__(
        self,
        model,  # SpectralCNN or TFLiteInference
        config: Dict[str, Any]
    ):
        self.model = model
        self.config = config
        
        # Thresholds
        self.thresholds = config.get("thresholds", {})
        self.min_transit_depth = self.thresholds.get("transit_depth_min", 0.0001)
        self.min_duration = self.thresholds.get("duration_min_hours", 0.5)
        self.min_snr = self.thresholds.get("snr_min", 7.0)
        
        # Detection counter
        self._detection_counter = 0
    
    def _generate_detection_id(self) -> str:
        """Generate unique detection ID."""
        self._detection_counter += 1
        return f"DET-{datetime.now().strftime('%Y%m%d')}-{self._detection_counter:06d}"
    
    def detect_single(
        self,
        flux: np.ndarray,
        time: Optional[np.ndarray] = None,
        object_id: str = "unknown"
    ) -> Detection:
        """
        Run detection on a single light curve.
        
        Args:
            flux: Flux array
            time: Time array (optional)
            object_id: Identifier for the object
            
        Returns:
            Detection result
        """
        # Preprocess
        flux_processed = self._preprocess(flux)
        
        # Run model
        predictions = self.model.predict(flux_processed[np.newaxis, ..., np.newaxis])
        
        if isinstance(predictions, tuple):
            pred_class, confidence = predictions
            pred_class = pred_class[0]
            confidence = confidence[0]
        else:
            confidence = np.max(predictions[0])
            pred_class = np.argmax(predictions[0])
        
        class_labels = [
            "noise", "stellar_signal", "planetary_transit",
            "eclipsing_binary", "instrument_artifact", "unknown_anomaly"
        ]
        classification = class_labels[pred_class]
        
        # Create detection
        detection = Detection(
            detection_id=self._generate_detection_id(),
            object_id=object_id,
            classification=classification,
            confidence=float(confidence),
            timestamp=datetime.now(),
            flux_data=flux,
            time_data=time
        )
        
        # Additional analysis for transits
        if classification == "planetary_transit":
            detection = self._analyze_transit(detection, flux, time)
        
        # Calculate SNR
        detection.snr = self._calculate_snr(flux)
        
        # Determine significance
        detection.is_significant = self._is_significant(detection)
        
        return detection
    
    def detect_batch(
        self,
        data_list: List[Tuple[np.ndarray, Optional[np.ndarray], str]],
        batch_id: Optional[str] = None
    ) -> DetectionBatch:
        """
        Run detection on a batch of light curves.
        
        Args:
            data_list: List of (flux, time, object_id) tuples
            batch_id: Optional batch identifier
            
        Returns:
            DetectionBatch with all results
        """
        if batch_id is None:
            batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Processing batch {batch_id} with {len(data_list)} items")
        
        detections = []
        for flux, time, object_id in data_list:
            try:
                detection = self.detect_single(flux, time, object_id)
                detections.append(detection)
            except Exception as e:
                logger.error(f"Error processing {object_id}: {e}")
        
        # Generate summary
        summary = self._generate_batch_summary(detections)
        
        batch = DetectionBatch(
            batch_id=batch_id,
            run_timestamp=datetime.now(),
            detections=detections,
            summary=summary
        )
        
        logger.info(
            f"Batch complete: {len(batch.significant_detections)} significant detections, "
            f"{len(batch.transit_candidates)} transit candidates"
        )
        
        return batch
    
    def _preprocess(self, flux: np.ndarray) -> np.ndarray:
        """Preprocess flux for model input."""
        # Remove NaN
        flux = np.nan_to_num(flux, nan=np.nanmedian(flux))
        
        # Normalize
        flux_median = np.median(flux)
        flux_std = np.std(flux)
        if flux_std > 0:
            flux = (flux - flux_median) / flux_std
        
        # Resample to model input size (1024)
        target_size = 1024
        if len(flux) != target_size:
            x_old = np.linspace(0, 1, len(flux))
            x_new = np.linspace(0, 1, target_size)
            flux = np.interp(x_new, x_old, flux)
        
        return flux.astype(np.float32)
    
    def _analyze_transit(
        self,
        detection: Detection,
        flux: np.ndarray,
        time: Optional[np.ndarray]
    ) -> Detection:
        """
        Detailed transit analysis.
        
        Extracts:
        - Transit depth
        - Duration
        - Mid-transit time
        - Period (if multiple transits)
        """
        # Detrend
        flux_detrended = self._detrend(flux)
        
        # Find transit events
        transit_mask = flux_detrended < (np.median(flux_detrended) - 2 * np.std(flux_detrended))
        
        if not np.any(transit_mask):
            return detection
        
        # Calculate transit depth
        baseline = np.median(flux_detrended[~transit_mask])
        transit_flux = np.median(flux_detrended[transit_mask])
        detection.transit_depth = abs(baseline - transit_flux)
        
        # Calculate duration (in indices, convert to hours if time available)
        transit_indices = np.where(transit_mask)[0]
        if len(transit_indices) > 0:
            duration_indices = transit_indices[-1] - transit_indices[0]
            
            if time is not None and len(time) > 0:
                time_span = time[-1] - time[0]
                detection.transit_duration = duration_indices / len(flux) * time_span * 24  # hours
            else:
                detection.transit_duration = duration_indices / len(flux) * 100  # arbitrary units
        
        # Find mid-transit
        if len(transit_indices) > 0:
            mid_idx = transit_indices[len(transit_indices) // 2]
            if time is not None and len(time) > mid_idx:
                detection.transit_midpoint = time[mid_idx]
        
        # Try to find period using BLS
        if time is not None and len(time) > 0:
            detection.period = self._estimate_period_bls(flux_detrended, time)
        
        return detection
    
    def _detrend(self, flux: np.ndarray, window: int = 101) -> np.ndarray:
        """Remove long-term trends from flux."""
        # Median filter for robust trend estimation
        from scipy.ndimage import median_filter
        trend = median_filter(flux, size=window, mode='nearest')
        return flux / trend
    
    def _estimate_period_bls(
        self,
        flux: np.ndarray,
        time: np.ndarray,
        min_period: float = 0.5,
        max_period: float = 100.0
    ) -> Optional[float]:
        """
        Estimate period using Box Least Squares (simplified).
        
        Returns period in days.
        """
        try:
            # Trial periods
            periods = np.linspace(min_period, max_period, 1000)
            
            best_power = 0
            best_period = None
            
            for period in periods:
                # Phase fold
                phase = (time % period) / period
                sorted_idx = np.argsort(phase)
                phase_sorted = phase[sorted_idx]
                flux_sorted = flux[sorted_idx]
                
                # Simple box fit
                n_bins = 50
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_means = []
                
                for i in range(n_bins):
                    mask = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i + 1])
                    if np.any(mask):
                        bin_means.append(np.mean(flux_sorted[mask]))
                    else:
                        bin_means.append(np.nan)
                
                bin_means = np.array(bin_means)
                
                # Power = variance of binned data
                power = np.nanstd(bin_means)
                
                if power > best_power:
                    best_power = power
                    best_period = period
            
            return best_period
            
        except Exception as e:
            logger.warning(f"Period estimation failed: {e}")
            return None
    
    def _calculate_snr(self, flux: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Estimate noise using MAD (Median Absolute Deviation)
        mad = np.median(np.abs(flux - np.median(flux)))
        noise = mad * 1.4826  # Scale to approximate std
        
        # Signal is the range
        signal = np.ptp(flux)
        
        if noise > 0:
            return signal / noise
        return 0.0
    
    def _is_significant(self, detection: Detection) -> bool:
        """Determine if detection is statistically significant."""
        
        # Must have minimum SNR
        if detection.snr < self.min_snr:
            return False
        
        # Must have minimum confidence
        if detection.confidence < 0.7:
            return False
        
        # For transits, check depth
        if detection.classification == "planetary_transit":
            if detection.transit_depth is None:
                return False
            if detection.transit_depth < self.min_transit_depth:
                return False
        
        return True
    
    def _generate_batch_summary(self, detections: List[Detection]) -> Dict[str, Any]:
        """Generate summary statistics for a batch."""
        
        classifications = [d.classification for d in detections]
        
        summary = {
            "total_processed": len(detections),
            "significant_count": sum(1 for d in detections if d.is_significant),
            "classification_counts": {},
            "mean_confidence": np.mean([d.confidence for d in detections]) if detections else 0,
            "mean_snr": np.mean([d.snr for d in detections]) if detections else 0,
            "transit_candidates": []
        }
        
        # Count by classification
        for label in set(classifications):
            summary["classification_counts"][label] = classifications.count(label)
        
        # Transit candidate details
        for d in detections:
            if d.classification == "planetary_transit" and d.is_significant:
                summary["transit_candidates"].append({
                    "object_id": d.object_id,
                    "confidence": d.confidence,
                    "transit_depth": d.transit_depth,
                    "period": d.period
                })
        
        return summary
    
    def save_batch_results(self, batch: DetectionBatch, output_dir: str):
        """Save batch results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        summary_file = output_path / f"{batch.batch_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "batch_id": batch.batch_id,
                "run_timestamp": batch.run_timestamp.isoformat(),
                "summary": batch.summary
            }, f, indent=2)
        
        # Save all detections
        detections_file = output_path / f"{batch.batch_id}_detections.json"
        with open(detections_file, 'w') as f:
            json.dump([d.to_dict() for d in batch.detections], f, indent=2)
        
        # Save significant detections separately
        if batch.significant_detections:
            sig_file = output_path / f"{batch.batch_id}_significant.json"
            with open(sig_file, 'w') as f:
                json.dump([d.to_dict() for d in batch.significant_detections], f, indent=2)
        
        logger.info(f"Saved batch results to {output_path}")


class RealTimeDetector:
    """
    Real-time detection handler for streaming data.
    """
    
    def __init__(
        self,
        detector: SpectralDetector,
        buffer_size: int = 1024,
        overlap: int = 512
    ):
        self.detector = detector
        self.buffer_size = buffer_size
        self.overlap = overlap
        
        self.buffer: List[float] = []
        self.time_buffer: List[float] = []
        self.detections: List[Detection] = []
    
    def add_data_point(self, flux: float, time: Optional[float] = None):
        """Add a single data point to the buffer."""
        self.buffer.append(flux)
        if time is not None:
            self.time_buffer.append(time)
        
        # Process when buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._process_buffer()
    
    def _process_buffer(self):
        """Process current buffer and detect."""
        flux = np.array(self.buffer)
        time = np.array(self.time_buffer) if self.time_buffer else None
        
        detection = self.detector.detect_single(flux, time, "realtime")
        
        if detection.is_significant:
            self.detections.append(detection)
            logger.info(f"Real-time detection: {detection.classification} (conf={detection.confidence:.2f})")
        
        # Slide buffer
        self.buffer = self.buffer[self.overlap:]
        if self.time_buffer:
            self.time_buffer = self.time_buffer[self.overlap:]
    
    def get_detections(self) -> List[Detection]:
        """Get all detections so far."""
        return self.detections
    
    def clear(self):
        """Clear buffers and detections."""
        self.buffer = []
        self.time_buffer = []
        self.detections = []


# CLI interface
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Spectral Anomaly Detector")
    parser.add_argument("--input", type=str, required=True, help="Input data directory or file")
    parser.add_argument("--output", type=str, default="reports/detections", help="Output directory")
    parser.add_argument("--model", type=str, help="Path to TFLite model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    if args.model and args.model.endswith('.tflite'):
        from src.model.spectral_cnn import TFLiteInference
        model = TFLiteInference(args.model)
    else:
        from src.model.spectral_cnn import SpectralCNN
        model = SpectralCNN()
        model.build_model()
        model.compile()
    
    # Initialize detector
    detector = SpectralDetector(model, config.get("detection", {}))
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        from astropy.io import fits
        with fits.open(str(input_path)) as hdul:
            flux = hdul[1].data['FLUX']
            time = hdul[1].data.get('TIME')
        
        detection = detector.detect_single(flux, time, input_path.stem)
        print(f"Classification: {detection.classification}")
        print(f"Confidence: {detection.confidence:.2%}")
        print(f"Significant: {detection.is_significant}")
        
    else:
        # Directory - batch processing
        data_list = []
        for fits_file in input_path.glob("*.fits"):
            try:
                from astropy.io import fits
                with fits.open(str(fits_file)) as hdul:
                    flux = hdul[1].data['FLUX']
                    time = hdul[1].data.get('TIME')
                data_list.append((flux, time, fits_file.stem))
            except Exception as e:
                logger.error(f"Error loading {fits_file}: {e}")
        
        if data_list:
            batch = detector.detect_batch(data_list)
            detector.save_batch_results(batch, args.output)
            print(f"Processed {len(batch.detections)} items")
            print(f"Significant detections: {len(batch.significant_detections)}")
            print(f"Transit candidates: {len(batch.transit_candidates)}")

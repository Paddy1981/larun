#!/usr/bin/env python3
"""
AstroTinyML - Main Orchestrator
===============================
Complete pipeline for NASA spectral data analysis.

Usage:
    python main.py --mode full --target Kepler-186 --mission kepler
    python main.py --mode train --data data/processed/training.npz
    python main.py --mode calibrate --reference data/calibration/exoplanets.csv
    python main.py --mode detect --input data/raw/ --output reports/
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import yaml
import json

import numpy as np

# Set up logging - ensure logs directory exists first
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/astro_tinyml.log')
    ]
)
logger = logging.getLogger("AstroTinyML")


class AstroTinyMLPipeline:
    """
    Main orchestrator for the AstroTinyML system.
    
    Coordinates:
    - Data ingestion from NASA archives
    - Model training and inference
    - Auto-calibration
    - Detection and report generation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components (lazy loading)
        self._pipeline = None
        self._model = None
        self._calibrator = None
        self._detector = None
        self._reporter = None
        
        # Create directories
        self._setup_directories()
        
        logger.info("AstroTinyML Pipeline initialized")
    
    def _setup_directories(self):
        """Create required directories."""
        dirs = [
            "data/raw", "data/processed", "data/calibration",
            "models/tflite", "models/checkpoints",
            "reports", "logs"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    @property
    def pipeline(self):
        """Lazy-load NASA data pipeline."""
        if self._pipeline is None:
            from src.pipeline.nasa_pipeline import NASADataPipeline
            self._pipeline = NASADataPipeline(
                self.config.get("nasa", {}),
                cache_dir="data/raw"
            )
        return self._pipeline
    
    @property
    def model(self):
        """Lazy-load SpectralCNN model."""
        if self._model is None:
            from src.model.spectral_cnn import SpectralCNN
            self._model = SpectralCNN(
                input_shape=tuple(self.config["model"]["input_shape"]),
                num_classes=len(self.config["model"]["output_classes"])
            )
            self._model.build_model()
            self._model.compile(
                learning_rate=self.config["model"]["training"]["learning_rate"]
            )
        return self._model
    
    @property
    def calibration_db(self):
        """Lazy-load calibration database."""
        from src.calibration.calibrator import CalibrationDatabase
        return CalibrationDatabase()
    
    @property
    def calibrator(self):
        """Lazy-load auto-calibrator."""
        if self._calibrator is None:
            from src.calibration.calibrator import AutoCalibrator
            self._calibrator = AutoCalibrator(
                self.model,
                self.calibration_db,
                self.config.get("calibration", {})
            )
        return self._calibrator
    
    @property
    def detector(self):
        """Lazy-load spectral detector."""
        if self._detector is None:
            from src.detector.detector import SpectralDetector
            self._detector = SpectralDetector(
                self.model,
                self.config.get("detection", {})
            )
        return self._detector
    
    @property
    def reporter(self):
        """Lazy-load report generator."""
        if self._reporter is None:
            from src.reporter.report_generator import NASAReportGenerator, ReportConfig
            report_config = ReportConfig(
                title=self.config["reporting"].get("title", "Spectral Analysis Report"),
                institution=os.getenv("INSTITUTION_NAME", "AstroTinyML Research"),
                contact_email=os.getenv("RESEARCHER_EMAIL", "researcher@example.com"),
                data_source=self.config.get("nasa", {}).get("mast", {}).get("missions", ["MAST"])[0]
            )
            self._reporter = NASAReportGenerator(
                report_config,
                output_dir=self.config["reporting"]["output_dir"]
            )
        return self._reporter
    
    async def fetch_data(
        self,
        targets: list,
        mission: str = "kepler"
    ) -> Dict[str, list]:
        """
        Fetch data from NASA archives.
        
        Args:
            targets: List of target names
            mission: Mission to query
            
        Returns:
            Dictionary mapping targets to spectral data
        """
        logger.info(f"Fetching data for {len(targets)} targets from {mission}")
        return await self.pipeline.fetch_spectral_data_batch(targets, mission)
    
    async def fetch_calibration_data(self, limit: int = 1000):
        """Fetch confirmed exoplanet data for calibration."""
        logger.info("Fetching calibration data from NASA Exoplanet Archive")
        df = await self.pipeline.fetch_confirmed_exoplanets(limit=limit)
        
        # Add to calibration database
        self.calibration_db.add_references_from_exoplanet_archive(df)
        
        return df
    
    def train_model(
        self,
        training_data_path: Optional[str] = None,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train the spectral classification model.
        
        Args:
            training_data_path: Path to .npz file with training data
            epochs: Number of training epochs
            batch_size: Batch size
        """
        logger.info("Starting model training")
        
        if training_data_path and Path(training_data_path).exists():
            # Load from file
            data = np.load(training_data_path)
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data.get('X_val')
            y_val = data.get('y_val')
        else:
            # Use calibration database
            X_train, y_train = self.calibration_db.get_training_data()
            
            if len(X_train) == 0:
                logger.error("No training data available. Fetch calibration data first.")
                return
            
            # Split for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        val_count = len(X_val) if X_val is not None and len(X_val) > 0 else 0
        logger.info(f"Training with {len(X_train)} samples, validating with {val_count}")
        
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_dir="models/checkpoints"
        )
        
        # Save model
        self.model.save("models/checkpoints/spectral_cnn_final.keras")
        
        # Export TFLite
        self.model.export_for_edge("models/tflite")
        
        logger.info("Training complete")
        
        return history
    
    def run_calibration(self):
        """Run calibration and generate report."""
        logger.info("Running calibration")
        
        metrics = self.calibrator.run_calibration()
        
        # Print report
        report = self.calibrator.get_calibration_report()
        print(report)
        
        return metrics
    
    def run_detection(
        self,
        input_path: str,
        output_dir: str = "reports"
    ):
        """
        Run detection on input data and generate report.
        
        Args:
            input_path: Path to FITS file or directory
            output_dir: Output directory for reports
        """
        logger.info(f"Running detection on {input_path}")
        
        input_path = Path(input_path)
        data_list = []
        
        if input_path.is_file():
            # Single file
            spectral_data = self.pipeline.load_local_fits(str(input_path))
            if spectral_data:
                data_list.append((
                    spectral_data.flux,
                    spectral_data.time,
                    spectral_data.object_id or input_path.stem
                ))
        else:
            # Directory
            for fits_file in input_path.glob("**/*.fits"):
                try:
                    spectral_data = self.pipeline.load_local_fits(str(fits_file))
                    if spectral_data:
                        data_list.append((
                            spectral_data.flux,
                            spectral_data.time,
                            spectral_data.object_id or fits_file.stem
                        ))
                except Exception as e:
                    logger.error(f"Error loading {fits_file}: {e}")
        
        if not data_list:
            logger.warning("No data loaded for detection")
            return None
        
        # Run detection
        batch = self.detector.detect_batch(data_list)
        
        # Generate report
        calibration_metrics = None
        if self.calibrator.last_calibration:
            recent = self.calibration_db.get_recent_metrics(1)
            if recent:
                calibration_metrics = {
                    "timestamp": recent[0].timestamp.isoformat(),
                    "accuracy": recent[0].accuracy,
                    "drift_detected": recent[0].drift_detected,
                    "reference_count": len(self.calibration_db.references)
                }
        
        output_files = self.reporter.generate_report(
            batch,
            calibration_metrics,
            output_formats=["pdf", "json", "fits", "csv"]
        )
        
        logger.info(f"Generated reports: {list(output_files.keys())}")
        
        return batch, output_files
    
    async def run_full_pipeline(
        self,
        targets: list,
        mission: str = "kepler",
        train: bool = False,
        calibrate: bool = True
    ):
        """
        Run the complete pipeline from data fetch to report generation.
        
        Args:
            targets: List of target names
            mission: Mission to query
            train: Whether to train the model
            calibrate: Whether to run calibration
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL ASTROTINYML PIPELINE")
        logger.info("=" * 60)
        
        results = {
            "start_time": datetime.now(),
            "targets": targets,
            "mission": mission
        }
        
        # Step 1: Fetch calibration data if needed
        if calibrate and len(self.calibration_db.references) < 100:
            logger.info("Step 1: Fetching calibration data...")
            await self.fetch_calibration_data(limit=1000)
        
        # Step 2: Train model if requested
        if train:
            logger.info("Step 2: Training model...")
            self.train_model()
        
        # Step 3: Run calibration
        if calibrate:
            logger.info("Step 3: Running calibration...")
            calibration_metrics = self.run_calibration()
            results["calibration"] = {
                "accuracy": calibration_metrics.accuracy,
                "drift_detected": calibration_metrics.drift_detected
            }
        
        # Step 4: Fetch target data
        logger.info("Step 4: Fetching target data...")
        target_data = await self.fetch_data(targets, mission)
        
        # Step 5: Run detection
        logger.info("Step 5: Running detection...")
        data_list = []
        for target, spectral_list in target_data.items():
            for spectral_data in spectral_list:
                data_list.append((
                    spectral_data.flux,
                    spectral_data.time,
                    f"{target}"
                ))
        
        if data_list:
            batch = self.detector.detect_batch(data_list)
            
            # Step 6: Generate report
            logger.info("Step 6: Generating report...")
            output_files = self.reporter.generate_report(
                batch,
                results.get("calibration"),
                output_formats=["pdf", "json", "fits", "csv"]
            )
            
            results["detection"] = {
                "total_processed": len(batch.detections),
                "significant": len(batch.significant_detections),
                "transit_candidates": len(batch.transit_candidates)
            }
            results["output_files"] = output_files
        
        results["end_time"] = datetime.now()
        results["duration"] = str(results["end_time"] - results["start_time"])
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Duration: {results['duration']}")
        if "detection" in results:
            logger.info(f"Processed: {results['detection']['total_processed']} items")
            logger.info(f"Transit candidates: {results['detection']['transit_candidates']}")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AstroTinyML - Spectral Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for a target
  python main.py --mode full --target Kepler-186 --mission kepler

  # Train model with calibration data
  python main.py --mode train --fetch-calibration

  # Run calibration only
  python main.py --mode calibrate

  # Detect anomalies in local files
  python main.py --mode detect --input data/raw/my_data.fits

  # Generate submission package
  python main.py --mode report --input reports/detections.json --submit-ready
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "fetch", "train", "calibrate", "detect", "report"],
        default="full",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        help="Target name(s) to analyze"
    )
    
    parser.add_argument(
        "--mission",
        choices=["kepler", "tess"],
        default="kepler",
        help="NASA mission to query"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--fetch-calibration",
        action="store_true",
        help="Fetch calibration data from NASA Exoplanet Archive"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    
    parser.add_argument(
        "--submit-ready",
        action="store_true",
        help="Generate NASA submission package"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = AstroTinyMLPipeline(args.config)
    
    # Run based on mode
    if args.mode == "full":
        targets = args.target or ["Kepler-186"]
        asyncio.run(pipeline.run_full_pipeline(
            targets,
            args.mission,
            train=args.train,
            calibrate=True
        ))
    
    elif args.mode == "fetch":
        targets = args.target or ["Kepler-186"]
        data = asyncio.run(pipeline.fetch_data(targets, args.mission))
        print(f"Fetched data for {len(data)} targets")
        for target, items in data.items():
            print(f"  {target}: {len(items)} light curves")
    
    elif args.mode == "train":
        if args.fetch_calibration:
            asyncio.run(pipeline.fetch_calibration_data())
        pipeline.train_model(
            training_data_path=args.input,
            epochs=args.epochs
        )
    
    elif args.mode == "calibrate":
        if args.fetch_calibration:
            asyncio.run(pipeline.fetch_calibration_data())
        pipeline.run_calibration()
    
    elif args.mode == "detect":
        if not args.input:
            print("Error: --input required for detect mode")
            sys.exit(1)
        result = pipeline.run_detection(args.input, args.output)
        if result is None:
            print("Error: No data loaded for detection. Check input path.")
            sys.exit(1)
        batch, files = result
        print(f"\nDetection complete!")
        print(f"Processed: {len(batch.detections)} items")
        print(f"Significant: {len(batch.significant_detections)}")
        print(f"Transit candidates: {len(batch.transit_candidates)}")
        print(f"\nOutput files:")
        for fmt, path in files.items():
            print(f"  {fmt}: {path}")
    
    elif args.mode == "report":
        if not args.input:
            print("Error: --input required for report mode")
            sys.exit(1)
        
        # Load existing detection results
        with open(args.input) as f:
            data = json.load(f)
        
        from src.detector.detector import Detection, DetectionBatch
        
        detections = []
        for d in data.get("detections", []):
            detection = Detection(
                detection_id=d.get("detection_id", ""),
                object_id=d.get("object_id", ""),
                classification=d.get("classification", "unknown"),
                confidence=d.get("confidence", 0),
                timestamp=datetime.fromisoformat(d.get("timestamp", datetime.now().isoformat())),
                transit_depth=d.get("transit_depth"),
                transit_duration=d.get("transit_duration"),
                period=d.get("period"),
                snr=d.get("snr", 0),
                is_significant=d.get("is_significant", False)
            )
            detections.append(detection)
        
        batch = DetectionBatch(
            batch_id=data.get("batch_id", "REPORT"),
            run_timestamp=datetime.now(),
            detections=detections,
            summary=data.get("summary", {})
        )
        
        if args.submit_ready:
            package = pipeline.reporter.generate_submission_package(batch)
            print(f"Submission package: {package}")
        else:
            files = pipeline.reporter.generate_report(batch)
            print("Generated reports:")
            for fmt, path in files.items():
                print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()

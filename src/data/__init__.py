"""
LARUN Data Module
=================
Automated data collection and training for astronomical TinyML models.

Data Sources:
- MAST (TESS, Kepler): Light curves for exoplanet detection
- Gaia DR3: Stellar parameters and photometry
- ZTF/ATLAS: Transient event light curves
- OGLE: Variable stars and microlensing events
- Galaxy Zoo: Galaxy morphology labels

Usage:
    # Collect data for a specific model
    from src.data.collectors import MASTCollector
    collector = MASTCollector()
    X, y, info = collector.collect(n_samples=1000)

    # Run automated training
    from src.data.automated_trainer import AutomatedTrainer
    trainer = AutomatedTrainer()
    results = trainer.train_all()
"""

from src.data.automated_trainer import AutomatedTrainer, TrainingConfig, ModelTrainingResult

__all__ = [
    'AutomatedTrainer',
    'TrainingConfig',
    'ModelTrainingResult',
]

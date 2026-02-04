"""
LARUN TinyML Models
===================
8 specialized TinyML models for astronomical analysis.

Models:
    - EXOPLANET-001: Exoplanet transit detection
    - VSTAR-001: Variable star classification
    - FLARE-001: Stellar flare detection
    - ASTERO-001: Asteroseismology analysis
    - SUPERNOVA-001: Supernova/transient detection
    - GALAXY-001: Galaxy morphology classification
    - SPECTYPE-001: Stellar spectral type classification
    - MICROLENS-001: Microlensing event detection

Usage:
    from src.model import get_model, get_generator, Pipeline

    # Get a model
    model = get_model("EXOPLANET-001")

    # Generate training data
    generator = get_generator("EXOPLANET-001")
    X, y = generator.generate_dataset(DatasetConfig(n_samples=1000))

    # Create a pipeline
    from src.model.pipeline_framework import create_exoplanet_detection_pipeline
    pipeline = create_exoplanet_detection_pipeline(models)
"""

# Model classes
from src.model.specialized_models import (
    MODEL_SPECS,
    MODEL_CLASSES,
    get_model,
    list_models,
    BaseNumpyModel,
    ExoplanetDetector,
    VariableStarClassifier,
    FlareDetector,
    AsteroseismologyAnalyzer,
    SupernovaDetector,
    GalaxyClassifier,
    SpectralTypeClassifier,
    MicrolensingDetector,
)

# Data generators
from src.model.data_generators import (
    DATA_GENERATORS,
    DatasetConfig,
    get_generator,
    ExoplanetDataGenerator,
    VariableStarDataGenerator,
    FlareDataGenerator,
    AsteroseismologyDataGenerator,
    SupernovaDataGenerator,
    GalaxyDataGenerator,
    SpectralTypeDataGenerator,
    MicrolensingDataGenerator,
)

# Pipeline framework
from src.model.pipeline_framework import (
    Pipeline,
    PipelineNode,
    PipelineResult,
    PipelineStatus,
    PipelineRunner,
    ModelNode,
    ParallelNode,
    SeriesNode,
    EnsembleNode,
    ConditionalNode,
    HumanCheckpointNode,
    DataTransformNode,
    HumanReviewRequest,
    HumanReviewResponse,
    create_exoplanet_detection_pipeline,
    create_stellar_classification_pipeline,
    create_transient_detection_pipeline,
)

# Visual guide
from src.model.visual_guide import (
    PipelineVisualizer,
    MODEL_DESCRIPTIONS,
    Colors,
)

# Legacy models
from src.model.spectral_cnn import SpectralCNN, TFLiteInference
from src.model.numpy_cnn import NumpyCNN

__all__ = [
    # Specs and factories
    'MODEL_SPECS',
    'MODEL_CLASSES',
    'get_model',
    'list_models',
    'DATA_GENERATORS',
    'get_generator',
    'DatasetConfig',

    # Model classes
    'BaseNumpyModel',
    'ExoplanetDetector',
    'VariableStarClassifier',
    'FlareDetector',
    'AsteroseismologyAnalyzer',
    'SupernovaDetector',
    'GalaxyClassifier',
    'SpectralTypeClassifier',
    'MicrolensingDetector',

    # Data generators
    'ExoplanetDataGenerator',
    'VariableStarDataGenerator',
    'FlareDataGenerator',
    'AsteroseismologyDataGenerator',
    'SupernovaDataGenerator',
    'GalaxyDataGenerator',
    'SpectralTypeDataGenerator',
    'MicrolensingDataGenerator',

    # Pipeline
    'Pipeline',
    'PipelineNode',
    'PipelineResult',
    'PipelineStatus',
    'PipelineRunner',
    'ModelNode',
    'ParallelNode',
    'SeriesNode',
    'EnsembleNode',
    'ConditionalNode',
    'HumanCheckpointNode',
    'DataTransformNode',
    'HumanReviewRequest',
    'HumanReviewResponse',
    'create_exoplanet_detection_pipeline',
    'create_stellar_classification_pipeline',
    'create_transient_detection_pipeline',

    # Visual
    'PipelineVisualizer',
    'MODEL_DESCRIPTIONS',
    'Colors',

    # Legacy
    'SpectralCNN',
    'TFLiteInference',
    'NumpyCNN',
]

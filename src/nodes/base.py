"""
BaseNode Abstract Class for LARUN Federated TinyML System

Defines the interface that all analysis nodes must implement.
Each node is a self-contained TinyML model with preprocessing and postprocessing.

Node Types:
- EXOPLANET-001: Transit detection (48KB, input 1024x1)
- VSTAR-001: Variable star classification (72KB, input 512x1)
- FLARE-001: Stellar flare detection (32KB, input 256x1)
- ASTERO-001: Asteroseismology (60KB, input 512x1)
- SUPERNOVA-001: Transient detection (80KB, input 128x1)
- GALAXY-001: Galaxy morphology (88KB, input 64x64x3)
- SPECTYPE-001: Spectral classification (40KB, input 8x1)
- MICROLENS-001: Microlensing events (72KB, input 512x1)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np
import yaml
import time


class NodeStatus(Enum):
    """Status of a node in the registry."""
    AVAILABLE = "available"       # Listed but not installed
    INSTALLED = "installed"       # Installed but not enabled
    ENABLED = "enabled"           # Active and will run
    DISABLED = "disabled"         # Installed but turned off
    ERROR = "error"               # Failed to load/run


class NodeCategory(Enum):
    """Category of astronomical analysis."""
    EXOPLANET = "exoplanet"
    STELLAR = "stellar"
    TRANSIENT = "transient"
    GALACTIC = "galactic"
    SPECTROSCOPY = "spectroscopy"


@dataclass
class NodeMetadata:
    """Metadata about a node loaded from manifest.yaml."""
    node_id: str                          # e.g., "EXOPLANET-001"
    name: str                             # Human-readable name
    version: str                          # Semantic version
    description: str                      # What the node does
    category: NodeCategory                # Analysis category
    model_size_kb: int                    # Model size in KB (must be <100)
    input_shape: Tuple[int, ...]          # Expected input shape
    output_classes: List[str]             # Classification labels
    author: str = "LARUN Project"
    license: str = "MIT"
    min_larun_version: str = "2.0.0"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, manifest_path: Path) -> 'NodeMetadata':
        """Load metadata from a manifest.yaml file."""
        with open(manifest_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse input_shape from string or list
        input_shape = data.get('input_shape', (1024, 1))
        if isinstance(input_shape, str):
            input_shape = tuple(map(int, input_shape.strip('()').split(',')))
        elif isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        return cls(
            node_id=data['node_id'],
            name=data['name'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            category=NodeCategory(data.get('category', 'stellar')),
            model_size_kb=data.get('model_size_kb', 50),
            input_shape=input_shape,
            output_classes=data.get('output_classes', []),
            author=data.get('author', 'LARUN Project'),
            license=data.get('license', 'MIT'),
            min_larun_version=data.get('min_larun_version', '2.0.0'),
            dependencies=data.get('dependencies', []),
            tags=data.get('tags', []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'category': self.category.value,
            'model_size_kb': self.model_size_kb,
            'input_shape': list(self.input_shape),
            'output_classes': self.output_classes,
            'author': self.author,
            'license': self.license,
            'min_larun_version': self.min_larun_version,
            'dependencies': self.dependencies,
            'tags': self.tags,
        }


@dataclass
class NodeResult:
    """Result from running a node's inference."""
    node_id: str                          # Which node produced this
    classification: str                   # Primary classification label
    confidence: float                     # Confidence score (0-1)
    probabilities: Dict[str, float]       # All class probabilities
    detections: List[Dict[str, Any]]      # List of detected events
    metadata: Dict[str, Any]              # Additional node-specific data
    inference_time_ms: float              # How long inference took
    success: bool = True                  # Whether inference succeeded
    error_message: Optional[str] = None   # Error message if failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'node_id': self.node_id,
            'classification': self.classification,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'detections': self.detections,
            'metadata': self.metadata,
            'inference_time_ms': self.inference_time_ms,
            'success': self.success,
            'error_message': self.error_message,
        }

    @classmethod
    def error(cls, node_id: str, message: str) -> 'NodeResult':
        """Create an error result."""
        return cls(
            node_id=node_id,
            classification='error',
            confidence=0.0,
            probabilities={},
            detections=[],
            metadata={},
            inference_time_ms=0.0,
            success=False,
            error_message=message,
        )


class BaseNode(ABC):
    """
    Abstract base class for all LARUN analysis nodes.

    Each node must implement:
    - preprocess(): Transform raw input data for the model
    - infer(): Run the TinyML model
    - postprocess(): Transform model output to NodeResult

    The run() method chains these together with timing and error handling.

    Example Implementation:
        class ExoplanetNode(BaseNode):
            def preprocess(self, light_curve):
                # Normalize and reshape to (1024, 1)
                return normalized_data

            def infer(self, data):
                # Run TFLite model
                return model_output

            def postprocess(self, output, raw_input):
                # Create NodeResult with transit detections
                return NodeResult(...)
    """

    def __init__(self, node_path: Path):
        """
        Initialize node from its directory.

        Args:
            node_path: Path to the node directory containing manifest.yaml
        """
        self.node_path = Path(node_path)
        self.manifest_path = self.node_path / 'manifest.yaml'
        self.model_path = self.node_path / 'model'

        # Load metadata
        if self.manifest_path.exists():
            self.metadata = NodeMetadata.from_yaml(self.manifest_path)
        else:
            raise FileNotFoundError(f"No manifest.yaml found at {self.manifest_path}")

        self.node_id = self.metadata.node_id
        self.status = NodeStatus.INSTALLED
        self._model = None
        self._interpreter = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model is not None or self._interpreter is not None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the TinyML model from disk.

        Should load either:
        - A TFLite model using tf.lite.Interpreter
        - A NumPy model from weights file
        - Any other lightweight model format

        Must set self._model or self._interpreter.
        """
        pass

    @abstractmethod
    def preprocess(self, raw_input: np.ndarray) -> np.ndarray:
        """
        Preprocess raw input data for the model.

        Args:
            raw_input: Raw input data (e.g., light curve, spectrum)

        Returns:
            Preprocessed data matching model's expected input shape

        Common preprocessing steps:
        - Normalization (zero-mean, unit variance or min-max)
        - Resampling to expected length
        - Gap interpolation
        - Detrending
        - Phase folding (for periodic signals)
        """
        pass

    @abstractmethod
    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model.

        Args:
            preprocessed_data: Output from preprocess()

        Returns:
            Raw model output (typically class probabilities)
        """
        pass

    @abstractmethod
    def postprocess(self, model_output: np.ndarray,
                    raw_input: np.ndarray) -> NodeResult:
        """
        Convert model output to a NodeResult.

        Args:
            model_output: Raw output from infer()
            raw_input: Original input data (for additional analysis)

        Returns:
            NodeResult with classification, confidence, and detections
        """
        pass

    def run(self, raw_input: np.ndarray) -> NodeResult:
        """
        Execute the full node pipeline: preprocess -> infer -> postprocess.

        Args:
            raw_input: Raw input data

        Returns:
            NodeResult with all analysis results
        """
        start_time = time.perf_counter()

        try:
            # Ensure model is loaded
            if not self.is_loaded:
                self.load_model()

            # Run pipeline
            preprocessed = self.preprocess(raw_input)
            output = self.infer(preprocessed)
            result = self.postprocess(output, raw_input)

            # Update timing
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.inference_time_ms = elapsed_ms

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return NodeResult(
                node_id=self.node_id,
                classification='error',
                confidence=0.0,
                probabilities={},
                detections=[],
                metadata={'error_type': type(e).__name__},
                inference_time_ms=elapsed_ms,
                success=False,
                error_message=str(e),
            )

    def validate_input(self, raw_input: np.ndarray) -> Tuple[bool, str]:
        """
        Validate that input data is suitable for this node.

        Args:
            raw_input: Input data to validate

        Returns:
            (is_valid, error_message) tuple
        """
        if not isinstance(raw_input, np.ndarray):
            return False, f"Expected np.ndarray, got {type(raw_input)}"

        if raw_input.size == 0:
            return False, "Input array is empty"

        if np.isnan(raw_input).any():
            # Allow NaNs but warn
            pass

        return True, ""

    def get_info(self) -> Dict[str, Any]:
        """Get node information for display."""
        return {
            'node_id': self.node_id,
            'name': self.metadata.name,
            'version': self.metadata.version,
            'description': self.metadata.description,
            'category': self.metadata.category.value,
            'model_size_kb': self.metadata.model_size_kb,
            'input_shape': self.metadata.input_shape,
            'output_classes': self.metadata.output_classes,
            'status': self.status.value,
            'is_loaded': self.is_loaded,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.node_id} ({self.status.value})>"


class TFLiteNode(BaseNode):
    """
    Base class for nodes using TensorFlow Lite models.

    Provides default implementation of load_model() and infer()
    for TFLite models. Subclasses only need to implement
    preprocess() and postprocess().
    """

    def __init__(self, node_path: Path, model_filename: str = 'detector.tflite'):
        super().__init__(node_path)
        self.model_filename = model_filename
        self.tflite_path = self.model_path / model_filename

    def load_model(self) -> None:
        """Load TFLite model using interpreter."""
        try:
            import tensorflow as tf

            if not self.tflite_path.exists():
                raise FileNotFoundError(f"Model not found: {self.tflite_path}")

            self._interpreter = tf.lite.Interpreter(
                model_path=str(self.tflite_path)
            )
            self._interpreter.allocate_tensors()

            # Get input/output details
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

        except ImportError:
            # Fall back to tflite_runtime if full TF not available
            try:
                import tflite_runtime.interpreter as tflite

                self._interpreter = tflite.Interpreter(
                    model_path=str(self.tflite_path)
                )
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()

            except ImportError:
                raise ImportError(
                    "Neither tensorflow nor tflite_runtime is available. "
                    "Install with: pip install tflite-runtime"
                )

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
        if self._interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get expected input type
        input_dtype = self._input_details[0]['dtype']

        # Ensure batch dimension
        if len(preprocessed_data.shape) == len(self.metadata.input_shape):
            preprocessed_data = np.expand_dims(preprocessed_data, axis=0)

        # Cast to expected type
        input_data = preprocessed_data.astype(input_dtype)

        # Run inference
        self._interpreter.set_tensor(
            self._input_details[0]['index'],
            input_data
        )
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(
            self._output_details[0]['index']
        )

        return output


class NumpyNode(BaseNode):
    """
    Base class for nodes using pure NumPy CNN models.

    Uses the NumPy CNN implementation for environments without TensorFlow.
    Subclasses must implement preprocess() and postprocess().
    """

    def __init__(self, node_path: Path, weights_filename: str = 'weights.npz'):
        super().__init__(node_path)
        self.weights_filename = weights_filename
        self.weights_path = self.model_path / weights_filename

    def load_model(self) -> None:
        """Load NumPy model weights."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        # Import the NumPy CNN implementation
        import sys
        src_path = self.node_path.parent.parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from model.numpy_cnn import SpectralCNN

        # Load weights
        weights = np.load(self.weights_path, allow_pickle=True)

        # Initialize model
        self._model = SpectralCNN(
            input_length=self.metadata.input_shape[0],
            num_classes=len(self.metadata.output_classes)
        )
        self._model.load_weights(weights)

    def infer(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """Run NumPy CNN inference."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Ensure batch dimension
        if len(preprocessed_data.shape) == len(self.metadata.input_shape):
            preprocessed_data = np.expand_dims(preprocessed_data, axis=0)

        return self._model.predict(preprocessed_data)

"""
Dynamic Node Loader for LARUN Federated TinyML System

Handles runtime loading of node modules from their directories.
Supports both Python module loading and dynamic class instantiation.

Usage:
    registry = NodeRegistry()
    loader = NodeLoader(registry)

    # Load a specific node
    node = loader.load_node("EXOPLANET-001")

    # Load all enabled nodes
    nodes = loader.load_enabled_nodes()

    # Run analysis
    results = [node.run(light_curve) for node in nodes]
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

from .base import BaseNode, NodeStatus
from .registry import NodeRegistry, NodeEntry


class NodeLoadError(Exception):
    """Exception raised when a node fails to load."""
    pass


class NodeLoader:
    """
    Dynamic loader for LARUN analysis nodes.

    Loads node implementations from their directories at runtime,
    allowing for modular installation and updates.
    """

    # Mapping of node IDs to their implementation class names
    # Used when class name doesn't follow the default pattern
    CLASS_NAME_OVERRIDES: Dict[str, str] = {
        'EXOPLANET-001': 'ExoplanetNode',
        'VSTAR-001': 'VariableStarNode',
        'FLARE-001': 'FlareNode',
        'ASTERO-001': 'AsteroseismoNode',
        'SUPERNOVA-001': 'SupernovaNode',
        'GALAXY-001': 'GalaxyNode',
        'SPECTYPE-001': 'SpectralTypeNode',
        'MICROLENS-001': 'MicrolensingNode',
    }

    def __init__(self, registry: Optional[NodeRegistry] = None):
        """
        Initialize the node loader.

        Args:
            registry: NodeRegistry instance. Creates new one if not provided.
        """
        self.registry = registry or NodeRegistry()
        self._loaded_nodes: Dict[str, BaseNode] = {}
        self._node_classes: Dict[str, Type[BaseNode]] = {}

    def load_node(self, node_id: str, force_reload: bool = False) -> BaseNode:
        """
        Load a node by its ID.

        Args:
            node_id: The node identifier (e.g., "EXOPLANET-001")
            force_reload: If True, reload even if already loaded

        Returns:
            Instantiated BaseNode subclass

        Raises:
            NodeLoadError: If node cannot be loaded
        """
        # Check cache
        if not force_reload and node_id in self._loaded_nodes:
            return self._loaded_nodes[node_id]

        # Get node entry from registry
        entry = self.registry.get_node(node_id)
        if entry is None:
            raise NodeLoadError(f"Node {node_id} not found in registry")

        if entry.status == 'available':
            raise NodeLoadError(
                f"Node {node_id} is not installed. "
                f"Run 'larun node install {node_id}' first."
            )

        if entry.path is None:
            raise NodeLoadError(f"Node {node_id} has no path in registry")

        node_path = Path(entry.path)
        if not node_path.exists():
            raise NodeLoadError(f"Node path does not exist: {node_path}")

        # Load the node class
        node_class = self._load_node_class(node_id, node_path)

        # Instantiate the node
        try:
            node = node_class(node_path)
            node.status = NodeStatus(entry.status)
            self._loaded_nodes[node_id] = node
            return node
        except Exception as e:
            raise NodeLoadError(f"Failed to instantiate {node_id}: {e}")

    def _load_node_class(self, node_id: str, node_path: Path) -> Type[BaseNode]:
        """
        Load the node class from its source file.

        Args:
            node_id: Node identifier
            node_path: Path to node directory

        Returns:
            The node class (not instantiated)
        """
        # Check cache
        if node_id in self._node_classes:
            return self._node_classes[node_id]

        # Find the detector/node implementation
        src_dir = node_path / 'src'
        if not src_dir.exists():
            src_dir = node_path  # Fall back to node root

        # Look for common implementation file names
        impl_files = [
            src_dir / 'detector.py',
            src_dir / 'node.py',
            src_dir / 'classifier.py',
            src_dir / f"{node_path.name}.py",
        ]

        impl_file = None
        for f in impl_files:
            if f.exists():
                impl_file = f
                break

        if impl_file is None:
            raise NodeLoadError(
                f"No implementation file found in {src_dir}. "
                f"Expected: detector.py, node.py, or classifier.py"
            )

        # Load the module dynamically
        module_name = f"larun_node_{node_id.lower().replace('-', '_')}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, impl_file)
            if spec is None or spec.loader is None:
                raise NodeLoadError(f"Could not load spec for {impl_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise NodeLoadError(f"Failed to load module {impl_file}: {e}")

        # Find the node class in the module
        class_name = self.CLASS_NAME_OVERRIDES.get(node_id)

        if class_name and hasattr(module, class_name):
            node_class = getattr(module, class_name)
        else:
            # Search for BaseNode subclass
            node_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, BaseNode) and
                    obj is not BaseNode):
                    node_class = obj
                    break

        if node_class is None:
            raise NodeLoadError(
                f"No BaseNode subclass found in {impl_file}. "
                f"Make sure your node class inherits from BaseNode."
            )

        self._node_classes[node_id] = node_class
        return node_class

    def load_enabled_nodes(self) -> List[BaseNode]:
        """
        Load all enabled nodes from the registry.

        Returns:
            List of loaded BaseNode instances
        """
        enabled = self.registry.get_enabled_nodes()
        nodes = []

        for entry in enabled:
            try:
                node = self.load_node(entry.node_id)
                nodes.append(node)
            except NodeLoadError as e:
                print(f"Warning: Could not load {entry.node_id}: {e}")

        return nodes

    def load_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """
        Load specific nodes by their IDs.

        Args:
            node_ids: List of node IDs to load

        Returns:
            List of loaded nodes (skips failures with warnings)
        """
        nodes = []

        for node_id in node_ids:
            try:
                node = self.load_node(node_id)
                nodes.append(node)
            except NodeLoadError as e:
                print(f"Warning: Could not load {node_id}: {e}")

        return nodes

    def preload_all(self) -> Dict[str, BaseNode]:
        """
        Preload all installed nodes into cache.

        Returns:
            Dict of node_id -> BaseNode for all successfully loaded nodes
        """
        installed = self.registry.get_installed_nodes()

        for entry in installed:
            try:
                self.load_node(entry.node_id)
            except NodeLoadError as e:
                print(f"Warning: Could not preload {entry.node_id}: {e}")

        return dict(self._loaded_nodes)

    def unload_node(self, node_id: str) -> bool:
        """
        Unload a node from cache.

        Args:
            node_id: Node to unload

        Returns:
            True if node was unloaded
        """
        if node_id in self._loaded_nodes:
            del self._loaded_nodes[node_id]
            return True
        return False

    def unload_all(self) -> None:
        """Unload all cached nodes."""
        self._loaded_nodes.clear()
        self._node_classes.clear()

    def get_loaded_nodes(self) -> Dict[str, BaseNode]:
        """Get all currently loaded nodes."""
        return dict(self._loaded_nodes)

    def is_loaded(self, node_id: str) -> bool:
        """Check if a node is loaded."""
        return node_id in self._loaded_nodes

    def reload_node(self, node_id: str) -> BaseNode:
        """
        Reload a node, refreshing from disk.

        Useful when node code has been updated.
        """
        # Clear from cache
        if node_id in self._loaded_nodes:
            del self._loaded_nodes[node_id]
        if node_id in self._node_classes:
            del self._node_classes[node_id]

        # Clear from sys.modules
        module_name = f"larun_node_{node_id.lower().replace('-', '_')}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        return self.load_node(node_id)

    def validate_node(self, node_id: str) -> Dict[str, Any]:
        """
        Validate a node's implementation.

        Checks:
        - Node can be loaded
        - Model can be loaded
        - Model size is within limits
        - Required methods are implemented

        Returns:
            Validation report dict
        """
        report = {
            'node_id': node_id,
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        try:
            node = self.load_node(node_id)

            # Check model size
            if node.metadata.model_size_kb > 100:
                report['warnings'].append(
                    f"Model size ({node.metadata.model_size_kb}KB) exceeds 100KB target"
                )

            # Try to load model
            try:
                node.load_model()
            except FileNotFoundError as e:
                report['warnings'].append(f"Model file not found: {e}")
            except Exception as e:
                report['errors'].append(f"Model load failed: {e}")
                report['valid'] = False

            # Check required methods
            import inspect
            for method in ['preprocess', 'infer', 'postprocess']:
                if not hasattr(node, method):
                    report['errors'].append(f"Missing required method: {method}")
                    report['valid'] = False
                elif getattr(node, method).__func__ is getattr(BaseNode, method, None):
                    # Method not overridden
                    report['errors'].append(f"Method {method} not implemented")
                    report['valid'] = False

        except NodeLoadError as e:
            report['valid'] = False
            report['errors'].append(str(e))

        return report

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a node.

        Returns node metadata and status without fully loading the model.
        """
        entry = self.registry.get_node(node_id)
        if entry is None:
            return None

        info = entry.to_dict()
        info['loaded'] = self.is_loaded(node_id)

        if self.is_loaded(node_id):
            node = self._loaded_nodes[node_id]
            info['model_loaded'] = node.is_loaded
            info['input_shape'] = node.metadata.input_shape
            info['output_classes'] = node.metadata.output_classes

        return info

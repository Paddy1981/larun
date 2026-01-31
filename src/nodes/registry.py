"""
Node Registry for LARUN Federated TinyML System

Manages node installation, enabling, disabling, and discovery.
Maintains a registry.yaml file in the nodes directory that tracks
all installed nodes and their status.

Usage:
    registry = NodeRegistry()
    registry.list_nodes()              # Show all nodes
    registry.enable_node("VSTAR-001")  # Enable a node
    registry.disable_node("VSTAR-001") # Disable a node
    registry.install_node(url)         # Install from URL
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import json
import hashlib
import urllib.request
import zipfile
import tarfile

from .base import NodeMetadata, NodeStatus, NodeCategory


@dataclass
class NodeEntry:
    """Entry in the node registry."""
    node_id: str
    name: str
    version: str
    category: str
    status: str  # available, installed, enabled, disabled
    path: Optional[str] = None
    model_size_kb: int = 0
    description: str = ""
    install_date: Optional[str] = None
    checksum: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'name': self.name,
            'version': self.version,
            'category': self.category,
            'status': self.status,
            'path': self.path,
            'model_size_kb': self.model_size_kb,
            'description': self.description,
            'install_date': self.install_date,
            'checksum': self.checksum,
            'source_url': self.source_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeEntry':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NodeRegistry:
    """
    Manages the registry of LARUN analysis nodes.

    The registry tracks:
    - Available nodes (built-in and community)
    - Installed nodes
    - Enabled/disabled status
    - Version information
    """

    # Built-in nodes that ship with LARUN
    BUILTIN_NODES = {
        'EXOPLANET-001': {
            'name': 'Exoplanet Transit Detector',
            'description': 'Detects exoplanet transits in light curves using CNN',
            'category': 'exoplanet',
            'model_size_kb': 48,
            'folder': 'exoplanet',
        },
        'VSTAR-001': {
            'name': 'Variable Star Classifier',
            'description': 'Classifies variable stars from phase-folded light curves',
            'category': 'stellar',
            'model_size_kb': 72,
            'folder': 'variable_star',
        },
        'FLARE-001': {
            'name': 'Stellar Flare Detector',
            'description': 'Detects stellar flares in short time windows',
            'category': 'stellar',
            'model_size_kb': 32,
            'folder': 'flare',
        },
        'ASTERO-001': {
            'name': 'Asteroseismology Analyzer',
            'description': 'Analyzes stellar oscillations from power spectra',
            'category': 'stellar',
            'model_size_kb': 60,
            'folder': 'asteroseismo',
        },
        'SUPERNOVA-001': {
            'name': 'Supernova/Transient Detector',
            'description': 'Detects supernovae and transient events',
            'category': 'transient',
            'model_size_kb': 80,
            'folder': 'supernova',
        },
        'GALAXY-001': {
            'name': 'Galaxy Morphology Classifier',
            'description': 'Classifies galaxy morphology from image cutouts',
            'category': 'galactic',
            'model_size_kb': 88,
            'folder': 'galaxy',
        },
        'SPECTYPE-001': {
            'name': 'Spectral Type Classifier',
            'description': 'Classifies stellar spectral types from photometry',
            'category': 'spectroscopy',
            'model_size_kb': 40,
            'folder': 'spectral_type',
        },
        'MICROLENS-001': {
            'name': 'Microlensing Event Detector',
            'description': 'Detects gravitational microlensing events',
            'category': 'transient',
            'model_size_kb': 72,
            'folder': 'microlensing',
        },
    }

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the node registry.

        Args:
            base_path: Base path for LARUN installation.
                      Defaults to parent of src/nodes.
        """
        if base_path is None:
            # Default to larun directory
            base_path = Path(__file__).parent.parent.parent

        self.base_path = Path(base_path)
        self.nodes_path = self.base_path / 'nodes'
        self.registry_file = self.nodes_path / 'registry.yaml'
        self.config_path = self.base_path / 'config'
        self.user_config_file = self.config_path / 'user_config.yaml'

        # Ensure directories exist
        self.nodes_path.mkdir(parents=True, exist_ok=True)
        self.config_path.mkdir(parents=True, exist_ok=True)

        # Load or initialize registry
        self._registry: Dict[str, NodeEntry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from YAML file or initialize from built-in nodes."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = yaml.safe_load(f) or {}

            for node_id, node_data in data.get('nodes', {}).items():
                self._registry[node_id] = NodeEntry.from_dict(node_data)
        else:
            # Initialize with built-in nodes
            self._initialize_builtin_nodes()
            self._save_registry()

    def _initialize_builtin_nodes(self) -> None:
        """Initialize registry with built-in nodes."""
        for node_id, info in self.BUILTIN_NODES.items():
            node_path = self.nodes_path / info['folder']

            # Determine status based on presence
            if node_path.exists() and (node_path / 'manifest.yaml').exists():
                status = 'installed'
            else:
                status = 'available'

            self._registry[node_id] = NodeEntry(
                node_id=node_id,
                name=info['name'],
                version='1.0.0',
                category=info['category'],
                status=status,
                path=str(node_path) if status == 'installed' else None,
                model_size_kb=info['model_size_kb'],
                description=info['description'],
            )

    def _save_registry(self) -> None:
        """Save registry to YAML file."""
        data = {
            'version': '1.0',
            'nodes': {
                node_id: entry.to_dict()
                for node_id, entry in self._registry.items()
            }
        }

        with open(self.registry_file, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def list_nodes(self, category: Optional[str] = None,
                   status: Optional[str] = None) -> List[NodeEntry]:
        """
        List nodes with optional filtering.

        Args:
            category: Filter by category (exoplanet, stellar, etc.)
            status: Filter by status (available, installed, enabled, disabled)

        Returns:
            List of NodeEntry objects
        """
        nodes = list(self._registry.values())

        if category:
            nodes = [n for n in nodes if n.category == category]

        if status:
            nodes = [n for n in nodes if n.status == status]

        return sorted(nodes, key=lambda n: n.node_id)

    def get_node(self, node_id: str) -> Optional[NodeEntry]:
        """Get a specific node by ID."""
        return self._registry.get(node_id)

    def get_enabled_nodes(self) -> List[NodeEntry]:
        """Get all enabled nodes."""
        return self.list_nodes(status='enabled')

    def get_installed_nodes(self) -> List[NodeEntry]:
        """Get all installed nodes (enabled or disabled)."""
        return [n for n in self._registry.values()
                if n.status in ('installed', 'enabled', 'disabled')]

    def enable_node(self, node_id: str) -> bool:
        """
        Enable a node for use in analysis.

        Args:
            node_id: The node to enable (e.g., "VSTAR-001")

        Returns:
            True if successful, False otherwise
        """
        node = self._registry.get(node_id)

        if node is None:
            print(f"Error: Node {node_id} not found in registry")
            return False

        if node.status == 'available':
            print(f"Error: Node {node_id} is not installed. Run 'larun node install {node_id}' first.")
            return False

        if node.status == 'enabled':
            print(f"Node {node_id} is already enabled")
            return True

        node.status = 'enabled'
        self._save_registry()
        self._update_user_config()
        print(f"Enabled node: {node_id}")
        return True

    def disable_node(self, node_id: str) -> bool:
        """
        Disable a node (will not run in analysis).

        Args:
            node_id: The node to disable

        Returns:
            True if successful, False otherwise
        """
        node = self._registry.get(node_id)

        if node is None:
            print(f"Error: Node {node_id} not found in registry")
            return False

        if node.status in ('available', 'disabled'):
            print(f"Node {node_id} is already disabled or not installed")
            return True

        node.status = 'disabled'
        self._save_registry()
        self._update_user_config()
        print(f"Disabled node: {node_id}")
        return True

    def install_node(self, node_id_or_url: str) -> bool:
        """
        Install a node from built-in list or URL.

        Args:
            node_id_or_url: Either a node ID (e.g., "VSTAR-001") or a URL

        Returns:
            True if successful, False otherwise
        """
        # Check if it's a built-in node ID
        if node_id_or_url in self.BUILTIN_NODES:
            return self._install_builtin_node(node_id_or_url)

        # Check if it's a URL
        if node_id_or_url.startswith(('http://', 'https://')):
            return self._install_from_url(node_id_or_url)

        print(f"Error: Unknown node '{node_id_or_url}'")
        print("Use 'larun node list --available' to see available nodes")
        return False

    def _install_builtin_node(self, node_id: str) -> bool:
        """Install a built-in node."""
        if node_id not in self.BUILTIN_NODES:
            return False

        node = self._registry.get(node_id)
        if node and node.status in ('installed', 'enabled', 'disabled'):
            print(f"Node {node_id} is already installed")
            return True

        # Built-in nodes should already have their code in place
        # Just update the registry status
        info = self.BUILTIN_NODES[node_id]
        node_path = self.nodes_path / info['folder']

        if node_path.exists() and (node_path / 'manifest.yaml').exists():
            self._registry[node_id] = NodeEntry(
                node_id=node_id,
                name=info['name'],
                version='1.0.0',
                category=info['category'],
                status='installed',
                path=str(node_path),
                model_size_kb=info['model_size_kb'],
                description=info['description'],
            )
            self._save_registry()
            print(f"Installed node: {node_id}")
            return True
        else:
            print(f"Error: Node files not found at {node_path}")
            print("The node implementation may be missing.")
            return False

    def _install_from_url(self, url: str) -> bool:
        """Install a node from a URL (zip or tar.gz)."""
        print(f"Downloading node from: {url}")

        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                tmp_path = tmp.name

            # Extract and validate
            with tempfile.TemporaryDirectory() as extract_dir:
                if url.endswith('.zip'):
                    with zipfile.ZipFile(tmp_path, 'r') as zf:
                        zf.extractall(extract_dir)
                elif url.endswith(('.tar.gz', '.tgz')):
                    with tarfile.open(tmp_path, 'r:gz') as tf:
                        tf.extractall(extract_dir)
                else:
                    print("Error: URL must point to .zip or .tar.gz file")
                    return False

                # Find manifest.yaml
                manifest_files = list(Path(extract_dir).rglob('manifest.yaml'))
                if not manifest_files:
                    print("Error: No manifest.yaml found in archive")
                    return False

                manifest_path = manifest_files[0]
                node_dir = manifest_path.parent

                # Load metadata
                metadata = NodeMetadata.from_yaml(manifest_path)

                # Validate model size
                if metadata.model_size_kb > 100:
                    print(f"Warning: Model size ({metadata.model_size_kb}KB) exceeds 100KB limit")

                # Check for conflicts
                if metadata.node_id in self._registry:
                    existing = self._registry[metadata.node_id]
                    if existing.status != 'available':
                        print(f"Node {metadata.node_id} already installed. Use 'larun node update' to upgrade.")
                        return False

                # Install to nodes directory
                target_path = self.nodes_path / metadata.node_id.lower().replace('-', '_')
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(node_dir, target_path)

                # Calculate checksum
                checksum = self._calculate_checksum(target_path)

                # Add to registry
                from datetime import datetime
                self._registry[metadata.node_id] = NodeEntry(
                    node_id=metadata.node_id,
                    name=metadata.name,
                    version=metadata.version,
                    category=metadata.category.value,
                    status='installed',
                    path=str(target_path),
                    model_size_kb=metadata.model_size_kb,
                    description=metadata.description,
                    install_date=datetime.now().isoformat(),
                    checksum=checksum,
                    source_url=url,
                )
                self._save_registry()

                print(f"Installed node: {metadata.node_id} v{metadata.version}")
                return True

        except Exception as e:
            print(f"Error installing node: {e}")
            return False
        finally:
            # Cleanup temp file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def uninstall_node(self, node_id: str, keep_data: bool = False) -> bool:
        """
        Uninstall a node.

        Args:
            node_id: The node to uninstall
            keep_data: If True, keep the node files but mark as available

        Returns:
            True if successful
        """
        node = self._registry.get(node_id)

        if node is None:
            print(f"Error: Node {node_id} not found")
            return False

        if node.status == 'available':
            print(f"Node {node_id} is not installed")
            return True

        # Check if it's a built-in node
        if node_id in self.BUILTIN_NODES:
            print(f"Cannot uninstall built-in node {node_id}")
            print("You can disable it with 'larun node disable {}'".format(node_id))
            return False

        # Remove files if not keeping data
        if not keep_data and node.path:
            node_path = Path(node.path)
            if node_path.exists():
                shutil.rmtree(node_path)

        # Update registry
        node.status = 'available'
        node.path = None
        node.install_date = None
        self._save_registry()
        self._update_user_config()

        print(f"Uninstalled node: {node_id}")
        return True

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of node directory."""
        hasher = hashlib.sha256()

        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                hasher.update(str(file_path.relative_to(path)).encode())
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()[:16]

    def _update_user_config(self) -> None:
        """Update user config with enabled nodes."""
        enabled = [n.node_id for n in self.get_enabled_nodes()]

        # Load existing config
        if self.user_config_file.exists():
            with open(self.user_config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        config['enabled_nodes'] = enabled
        config['last_updated'] = str(Path(__file__).stat().st_mtime)

        with open(self.user_config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)

    def refresh(self) -> None:
        """Refresh registry by scanning nodes directory."""
        # Scan for any new nodes in the nodes directory
        for node_dir in self.nodes_path.iterdir():
            if not node_dir.is_dir():
                continue

            manifest_path = node_dir / 'manifest.yaml'
            if not manifest_path.exists():
                continue

            try:
                metadata = NodeMetadata.from_yaml(manifest_path)

                # Check if already in registry
                if metadata.node_id not in self._registry:
                    self._registry[metadata.node_id] = NodeEntry(
                        node_id=metadata.node_id,
                        name=metadata.name,
                        version=metadata.version,
                        category=metadata.category.value,
                        status='installed',
                        path=str(node_dir),
                        model_size_kb=metadata.model_size_kb,
                        description=metadata.description,
                    )
            except Exception as e:
                print(f"Warning: Could not load node from {node_dir}: {e}")

        self._save_registry()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        nodes = list(self._registry.values())

        return {
            'total_nodes': len(nodes),
            'available': len([n for n in nodes if n.status == 'available']),
            'installed': len([n for n in nodes if n.status in ('installed', 'enabled', 'disabled')]),
            'enabled': len([n for n in nodes if n.status == 'enabled']),
            'disabled': len([n for n in nodes if n.status == 'disabled']),
            'total_model_size_kb': sum(n.model_size_kb for n in nodes if n.status == 'enabled'),
            'by_category': {
                cat: len([n for n in nodes if n.category == cat])
                for cat in set(n.category for n in nodes)
            },
        }

    def format_node_list(self, nodes: Optional[List[NodeEntry]] = None,
                         verbose: bool = False) -> str:
        """Format node list for display."""
        if nodes is None:
            nodes = self.list_nodes()

        if not nodes:
            return "No nodes found."

        lines = []
        lines.append("\n LARUN Node Registry")
        lines.append("=" * 70)

        # Group by category
        categories = {}
        for node in nodes:
            cat = node.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(node)

        for category, cat_nodes in sorted(categories.items()):
            lines.append(f"\n[{category.upper()}]")

            for node in cat_nodes:
                status_icon = {
                    'available': ' ',
                    'installed': '-',
                    'enabled': '+',
                    'disabled': 'x',
                }.get(node.status, '?')

                size_str = f"{node.model_size_kb}KB"
                line = f"  [{status_icon}] {node.node_id:<14} {node.name:<30} {size_str:>6}"
                lines.append(line)

                if verbose:
                    lines.append(f"      {node.description}")
                    if node.path:
                        lines.append(f"      Path: {node.path}")

        lines.append("\n" + "-" * 70)
        lines.append("Legend: [+] enabled  [-] installed  [x] disabled  [ ] available")

        stats = self.get_stats()
        lines.append(f"Total: {stats['enabled']} enabled, {stats['installed']} installed, "
                     f"{stats['total_model_size_kb']}KB total model size")

        return "\n".join(lines)

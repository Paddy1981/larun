"""
LARUN Federated Multi-Model TinyML Node System

This module provides the infrastructure for running multiple specialized
TinyML models (<100KB each) for different astronomical analysis tasks.

Components:
- BaseNode: Abstract base class for all analysis nodes
- NodeRegistry: Install, enable, disable, list nodes
- NodeLoader: Dynamic loading of node modules
- NodeAggregator: Combine results from multiple nodes

Usage:
    from src.nodes import NodeRegistry, NodeLoader, NodeAggregator

    registry = NodeRegistry()
    loader = NodeLoader(registry)
    aggregator = NodeAggregator()

    # Load enabled nodes
    nodes = loader.load_enabled_nodes()

    # Run analysis
    results = [node.run(light_curve) for node in nodes]

    # Aggregate results
    combined = aggregator.aggregate(results)
"""

from .base import BaseNode, NodeResult, NodeMetadata
from .registry import NodeRegistry
from .loader import NodeLoader
from .aggregator import NodeAggregator

__all__ = [
    'BaseNode',
    'NodeResult',
    'NodeMetadata',
    'NodeRegistry',
    'NodeLoader',
    'NodeAggregator',
]

#!/usr/bin/env python3
"""
Test script for LARUN Federated Multi-Model TinyML System

Verifies:
1. Node registry functionality
2. Node loading
3. Multi-node analysis with aggregation
4. CLI commands

Run: python test_nodes.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np


def test_registry():
    """Test node registry functionality."""
    print("\n" + "=" * 60)
    print(" Testing Node Registry")
    print("=" * 60)

    from nodes.registry import NodeRegistry

    registry = NodeRegistry()

    # List all nodes
    print("\nListing all nodes:")
    nodes = registry.list_nodes()
    for node in nodes:
        print(f"  [{node.status}] {node.node_id}: {node.name} ({node.model_size_kb}KB)")

    # Get stats
    print("\nRegistry stats:")
    stats = registry.get_stats()
    print(f"  Total: {stats['total_nodes']}")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  By category: {stats['by_category']}")

    # Enable a node
    print("\nEnabling VSTAR-001...")
    registry.enable_node('VSTAR-001')

    print("\nEnabled nodes:")
    for node in registry.get_enabled_nodes():
        print(f"  {node.node_id}")

    return True


def test_loader():
    """Test node loading."""
    print("\n" + "=" * 60)
    print(" Testing Node Loader")
    print("=" * 60)

    from nodes.registry import NodeRegistry
    from nodes.loader import NodeLoader

    registry = NodeRegistry()
    loader = NodeLoader(registry)

    # Load exoplanet node
    print("\nLoading EXOPLANET-001...")
    try:
        node = loader.load_node('EXOPLANET-001')
        print(f"  Loaded: {node}")
        print(f"  Metadata: {node.metadata.name}")
        print(f"  Input shape: {node.metadata.input_shape}")
        print(f"  Classes: {node.metadata.output_classes}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_inference():
    """Test node inference."""
    print("\n" + "=" * 60)
    print(" Testing Node Inference")
    print("=" * 60)

    from nodes.registry import NodeRegistry
    from nodes.loader import NodeLoader

    registry = NodeRegistry()
    loader = NodeLoader(registry)

    # Create test light curve
    print("\nGenerating synthetic light curve...")
    np.random.seed(42)
    n_points = 1024
    flux = 1.0 + 0.001 * np.random.randn(n_points)

    # Add a transit-like dip
    transit_center = 512
    transit_width = 20
    flux[transit_center - transit_width:transit_center + transit_width] -= 0.01

    print(f"  Data shape: {flux.shape}")
    print(f"  Mean flux: {np.mean(flux):.4f}")
    print(f"  Transit depth: ~1%")

    # Test exoplanet node
    print("\nRunning EXOPLANET-001...")
    node = loader.load_node('EXOPLANET-001')
    result = node.run(flux)

    print(f"  Classification: {result.classification}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Inference time: {result.inference_time_ms:.2f}ms")
    print(f"  Detections: {len(result.detections)}")

    if result.detections:
        print(f"  Detection details: {result.detections[0]}")

    return result.success


def test_multi_node():
    """Test multi-node analysis with aggregation."""
    print("\n" + "=" * 60)
    print(" Testing Multi-Node Analysis")
    print("=" * 60)

    from nodes.registry import NodeRegistry
    from nodes.loader import NodeLoader
    from nodes.aggregator import NodeAggregator

    registry = NodeRegistry()
    loader = NodeLoader(registry)
    aggregator = NodeAggregator()

    # Enable multiple nodes
    registry.enable_node('EXOPLANET-001')
    registry.enable_node('VSTAR-001')
    registry.enable_node('FLARE-001')

    # Load enabled nodes
    nodes = loader.load_enabled_nodes()
    print(f"\nLoaded {len(nodes)} nodes:")
    for node in nodes:
        print(f"  - {node.node_id}")

    # Create test data
    np.random.seed(123)
    flux = 1.0 + 0.002 * np.random.randn(1024)
    flux[400:450] -= 0.008  # Transit-like feature

    # Run all nodes
    print("\nRunning analysis...")
    results = []
    for node in nodes:
        print(f"  {node.node_id}...", end=" ")
        result = node.run(flux)
        results.append(result)
        print(f"{result.classification} ({result.confidence:.1%})")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregator.aggregate(results, target_id="TEST_TARGET")

    print(f"\n  Primary classification: {aggregated.primary_classification}")
    print(f"  Overall confidence: {aggregated.overall_confidence:.1%}")
    print(f"  Consensus: {aggregated.consensus}")
    print(f"  Total detections: {aggregated.summary['total_detections']}")

    # Format output
    print("\n" + "-" * 40)
    print(aggregator.format_result(aggregated))

    return True


def test_cli():
    """Test CLI functionality."""
    print("\n" + "=" * 60)
    print(" Testing CLI")
    print("=" * 60)

    from cli.node_commands import NodeCommands

    cli = NodeCommands()

    # Test list command
    print("\nTesting 'node list' command:")
    cli.run(['list'])

    # Test stats command
    print("\nTesting 'node stats' command:")
    cli.run(['stats'])

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  LARUN Federated Multi-Model TinyML System - Test Suite")
    print("=" * 60)

    tests = [
        ("Registry", test_registry),
        ("Loader", test_loader),
        ("Inference", test_inference),
        ("Multi-Node", test_multi_node),
        ("CLI", test_cli),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "" if success else ""
        print(f"  {status} {name}")

    print(f"\n  Passed: {passed}/{total}")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

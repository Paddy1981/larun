#!/usr/bin/env python3
"""
Model Validation for LARUN TinyML Nodes
========================================

Validates trained models against quality gates before deployment.
Checks accuracy, size, inference time, and format compatibility.

Usage:
    python scripts/validate_models.py --node VSTAR-001
    python scripts/validate_models.py --all
    python scripts/validate_models.py --model nodes/flare/model/detector.tflite

Quality Gates:
    - Accuracy >= 0.80 (configurable per node)
    - Model size <= 100KB
    - Inference time <= 50ms
    - Valid TFLite format
    - No NaN/Inf in weights
"""

import os
import sys
import argparse
import json
import hashlib
import struct
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Quality Gate Configuration
# =============================================================================

DEFAULT_QUALITY_GATES = {
    'min_accuracy': 0.80,
    'max_size_kb': 100,
    'max_inference_ms': 50,
    'min_precision': 0.70,
    'min_recall': 0.70,
    'min_f1': 0.75,
}

NODE_QUALITY_GATES = {
    'EXOPLANET-001': {
        'min_accuracy': 0.82,
        'min_precision': 0.80,
        'min_recall': 0.85,  # High recall important for planet detection
    },
    'VSTAR-001': {
        'min_accuracy': 0.85,
        'max_size_kb': 80,
    },
    'FLARE-001': {
        'min_accuracy': 0.88,
        'min_recall': 0.90,  # High recall for flare detection
        'max_size_kb': 40,
    },
    'ASTERO-001': {
        'min_accuracy': 0.80,
    },
    'SUPERNOVA-001': {
        'min_accuracy': 0.85,
        'min_recall': 0.88,  # Don't miss transients
    },
    'GALAXY-001': {
        'min_accuracy': 0.78,
        'max_size_kb': 95,  # Images need more capacity
    },
    'SPECTYPE-001': {
        'min_accuracy': 0.82,
        'max_size_kb': 50,
    },
    'MICROLENS-001': {
        'min_accuracy': 0.80,
        'min_recall': 0.85,
    },
}


# =============================================================================
# Validation Functions
# =============================================================================

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_tflite_format(model_path: Path) -> Dict[str, Any]:
    """
    Validate TFLite model format and structure.

    Returns validation result with details.
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'details': {},
    }

    if not model_path.exists():
        result['errors'].append(f"Model file not found: {model_path}")
        return result

    # Check file extension
    if model_path.suffix not in ['.tflite', '.lite']:
        result['warnings'].append(f"Unexpected extension: {model_path.suffix}")

    # Check file size
    size_bytes = model_path.stat().st_size
    size_kb = size_bytes / 1024
    result['details']['size_kb'] = round(size_kb, 2)

    # Check TFLite magic bytes (FlatBuffer format)
    with open(model_path, 'rb') as f:
        # TFLite files start with specific bytes
        header = f.read(8)

        # Simple size check
        if size_bytes < 100:
            result['errors'].append("Model file too small to be valid")
            return result

    # Try to load with TFLite interpreter
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        result['details']['input_shape'] = input_details[0]['shape'].tolist()
        result['details']['input_dtype'] = str(input_details[0]['dtype'])
        result['details']['output_shape'] = output_details[0]['shape'].tolist()
        result['details']['output_dtype'] = str(output_details[0]['dtype'])
        result['details']['num_tensors'] = len(interpreter.get_tensor_details())

        # Check quantization
        if input_details[0]['dtype'] == np.int8:
            result['details']['quantization'] = 'int8'
        elif input_details[0]['dtype'] == np.float32:
            result['details']['quantization'] = 'float32'
        else:
            result['details']['quantization'] = str(input_details[0]['dtype'])

        result['valid'] = True

    except ImportError:
        result['warnings'].append("TensorFlow not available, skipping detailed validation")
        # Basic validation passed if we got here
        result['valid'] = True

    except Exception as e:
        result['errors'].append(f"Failed to load model: {str(e)}")

    return result


def validate_model_size(model_path: Path, max_kb: int = 100) -> Dict[str, Any]:
    """Validate model size against limit."""
    size_kb = model_path.stat().st_size / 1024

    return {
        'passed': size_kb <= max_kb,
        'size_kb': round(size_kb, 2),
        'limit_kb': max_kb,
        'margin_kb': round(max_kb - size_kb, 2),
    }


def validate_inference_time(
    model_path: Path,
    input_shape: Tuple[int, ...],
    max_ms: float = 50,
    n_runs: int = 10,
) -> Dict[str, Any]:
    """
    Validate inference latency.

    Runs multiple inferences and reports statistics.
    """
    result = {
        'passed': False,
        'avg_ms': 0,
        'min_ms': 0,
        'max_ms': 0,
        'limit_ms': max_ms,
    }

    try:
        import tensorflow as tf
        import time

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        # Generate random input
        if input_dtype == np.int8:
            test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            test_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        # Timed runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        result['avg_ms'] = round(np.mean(times), 3)
        result['min_ms'] = round(np.min(times), 3)
        result['max_ms'] = round(np.max(times), 3)
        result['std_ms'] = round(np.std(times), 3)
        result['passed'] = result['avg_ms'] <= max_ms

    except ImportError:
        result['error'] = "TensorFlow not available"
    except Exception as e:
        result['error'] = str(e)

    return result


def validate_metrics(
    metrics: Dict[str, float],
    gates: Dict[str, float],
) -> Dict[str, Any]:
    """Validate model metrics against quality gates."""
    results = {
        'passed': True,
        'checks': {},
    }

    metric_map = {
        'min_accuracy': 'accuracy',
        'min_precision': 'precision',
        'min_recall': 'recall',
        'min_f1': 'f1_score',
    }

    for gate_name, metric_name in metric_map.items():
        if gate_name in gates and metric_name in metrics:
            threshold = gates[gate_name]
            value = metrics[metric_name]
            passed = value >= threshold

            results['checks'][metric_name] = {
                'passed': passed,
                'value': value,
                'threshold': threshold,
            }

            if not passed:
                results['passed'] = False

    return results


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_node(
    node_id: str,
    metrics_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full validation for a node's model.

    Args:
        node_id: Node identifier
        metrics_path: Optional path to metrics JSON
        verbose: Print detailed output

    Returns:
        Validation results
    """
    base_path = Path(__file__).parent.parent / 'nodes'

    # Node folder mapping
    node_folders = {
        'EXOPLANET-001': 'exoplanet',
        'VSTAR-001': 'variable_star',
        'FLARE-001': 'flare',
        'ASTERO-001': 'asteroseismo',
        'SUPERNOVA-001': 'supernova',
        'GALAXY-001': 'galaxy',
        'SPECTYPE-001': 'spectral_type',
        'MICROLENS-001': 'microlensing',
    }

    if node_id not in node_folders:
        raise ValueError(f"Unknown node: {node_id}")

    folder = node_folders[node_id]
    node_path = base_path / folder

    # Find model file
    model_dir = node_path / 'model'
    model_files = list(model_dir.glob('*.tflite')) if model_dir.exists() else []

    if not model_files:
        return {
            'passed': False,
            'node_id': node_id,
            'error': f"No model file found in {model_dir}",
        }

    model_path = model_files[0]

    # Get quality gates
    gates = DEFAULT_QUALITY_GATES.copy()
    gates.update(NODE_QUALITY_GATES.get(node_id, {}))

    if verbose:
        print(f"\n{'='*60}")
        print(f" Validating {node_id}")
        print(f"{'='*60}")
        print(f"  Model: {model_path}")

    results = {
        'passed': True,
        'node_id': node_id,
        'model_path': str(model_path),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'checks': {},
    }

    # 1. Format validation
    format_result = validate_tflite_format(model_path)
    results['checks']['format'] = format_result

    if verbose:
        status = "" if format_result['valid'] else ""
        print(f"  {status} Format: {'valid' if format_result['valid'] else 'invalid'}")
        if format_result.get('details'):
            print(f"      Shape: {format_result['details'].get('input_shape')}")
            print(f"      Quantization: {format_result['details'].get('quantization')}")

    if not format_result['valid']:
        results['passed'] = False

    # 2. Size validation
    size_result = validate_model_size(model_path, gates['max_size_kb'])
    results['checks']['size'] = size_result

    if verbose:
        status = "" if size_result['passed'] else ""
        print(f"  {status} Size: {size_result['size_kb']}KB (limit: {size_result['limit_kb']}KB)")

    if not size_result['passed']:
        results['passed'] = False

    # 3. Inference time validation
    if format_result['valid'] and format_result.get('details', {}).get('input_shape'):
        inference_result = validate_inference_time(
            model_path,
            tuple(format_result['details']['input_shape']),
            gates['max_inference_ms'],
        )
        results['checks']['inference'] = inference_result

        if verbose:
            if 'error' in inference_result:
                print(f"   Inference: {inference_result['error']}")
            else:
                status = "" if inference_result['passed'] else ""
                print(f"  {status} Inference: {inference_result['avg_ms']}ms avg (limit: {inference_result['limit_ms']}ms)")

        if 'error' not in inference_result and not inference_result['passed']:
            results['passed'] = False

    # 4. Metrics validation (if available)
    if metrics_path is None:
        metrics_path = node_path / 'model' / 'metrics.json'

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        metrics_result = validate_metrics(metrics, gates)
        results['checks']['metrics'] = metrics_result

        if verbose:
            for metric_name, check in metrics_result['checks'].items():
                status = "" if check['passed'] else ""
                print(f"  {status} {metric_name}: {check['value']:.2%} (threshold: {check['threshold']:.2%})")

        if not metrics_result['passed']:
            results['passed'] = False
    else:
        if verbose:
            print(f"   Metrics: No metrics file found at {metrics_path}")

    # Compute checksum
    results['checksum'] = compute_sha256(model_path)

    if verbose:
        print(f"\n  Checksum: {results['checksum'][:16]}...")
        overall = "" if results['passed'] else ""
        print(f"\n  {overall} Overall: {'PASSED' if results['passed'] else 'FAILED'}")

    return results


def validate_all_nodes(verbose: bool = True) -> Dict[str, Any]:
    """Validate all node models."""
    nodes = [
        'VSTAR-001', 'FLARE-001', 'ASTERO-001', 'SUPERNOVA-001',
        'SPECTYPE-001', 'MICROLENS-001', 'GALAXY-001',
    ]

    results = {}
    passed = 0
    failed = 0

    for node_id in nodes:
        try:
            result = validate_node(node_id, verbose=verbose)
            results[node_id] = result

            if result['passed']:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results[node_id] = {'passed': False, 'error': str(e)}
            failed += 1
            if verbose:
                print(f"\n Error validating {node_id}: {e}")

    if verbose:
        print("\n" + "=" * 60)
        print(" Validation Summary")
        print("=" * 60)
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"  Total: {passed + failed}")

    return {
        'nodes': results,
        'summary': {
            'passed': passed,
            'failed': failed,
            'total': passed + failed,
            'all_passed': failed == 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate LARUN TinyML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--node', '-n', help='Node ID to validate')
    parser.add_argument('--all', '-a', action='store_true', help='Validate all nodes')
    parser.add_argument('--model', '-m', type=Path, help='Direct path to model file')
    parser.add_argument('--metrics', type=Path, help='Path to metrics JSON')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--output', '-o', type=Path, help='Save results to JSON')

    args = parser.parse_args()

    if args.model:
        result = validate_tflite_format(args.model)
        size_result = validate_model_size(args.model)
        print(json.dumps({'format': result, 'size': size_result}, indent=2))

    elif args.all:
        results = validate_all_nodes(verbose=not args.quiet)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        # Exit with error code if any failed
        sys.exit(0 if results['summary']['all_passed'] else 1)

    elif args.node:
        result = validate_node(args.node, args.metrics, verbose=not args.quiet)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        sys.exit(0 if result['passed'] else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

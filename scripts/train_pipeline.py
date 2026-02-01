#!/usr/bin/env python3
"""
LARUN Training Pipeline Orchestrator
=====================================

Orchestrates the full training pipeline:
1. Fetch training data
2. Train models
3. Validate models
4. Generate metrics
5. Optionally publish to release

Usage:
    python scripts/train_pipeline.py --node VSTAR-001
    python scripts/train_pipeline.py --all
    python scripts/train_pipeline.py --all --publish --version v2.1.0
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Pipeline Steps
# =============================================================================

def run_step(name: str, script: str, args: List[str]) -> Dict[str, Any]:
    """Run a pipeline step and capture results."""
    print(f"\n{'='*60}")
    print(f" Step: {name}")
    print(f"{'='*60}")

    start_time = datetime.utcnow()

    try:
        result = subprocess.run(
            [sys.executable, script] + args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        success = result.returncode == 0
        duration = (datetime.utcnow() - start_time).total_seconds()

        if result.stdout:
            print(result.stdout)

        if result.stderr and not success:
            print(f"STDERR: {result.stderr}", file=sys.stderr)

        return {
            'name': name,
            'success': success,
            'duration_seconds': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        return {
            'name': name,
            'success': False,
            'duration_seconds': duration,
            'error': str(e),
        }


def fetch_data(nodes: List[str], limit: Optional[int] = None) -> Dict[str, Any]:
    """Fetch training data for nodes."""
    script = 'scripts/fetch_training_data.py'
    args = []

    if len(nodes) == 1:
        args = ['--node', nodes[0]]
    else:
        args = ['--all']

    if limit:
        args.extend(['--limit', str(limit)])

    return run_step('Fetch Training Data', script, args)


def train_models(nodes: List[str], epochs: int = 30) -> Dict[str, Any]:
    """Train models for nodes."""
    script = 'train_nodes.py'
    args = ['--epochs', str(epochs)]

    if len(nodes) == 1:
        args.extend(['--node', nodes[0]])
    else:
        args.append('--all')

    return run_step('Train Models', script, args)


def validate_models(nodes: List[str]) -> Dict[str, Any]:
    """Validate trained models."""
    script = 'scripts/validate_models.py'
    args = []

    if len(nodes) == 1:
        args = ['--node', nodes[0]]
    else:
        args = ['--all']

    return run_step('Validate Models', script, args)


def bundle_models(version: str) -> Dict[str, Any]:
    """Create model bundles."""
    script = 'scripts/bundle_models.py'
    args = ['--version', version]

    return run_step('Bundle Models', script, args)


def generate_changelog(version: str, since: Optional[str] = None) -> Dict[str, Any]:
    """Generate changelog."""
    script = 'scripts/generate_changelog.py'
    args = ['--version', version, '--output', 'CHANGELOG.md']

    if since:
        args.extend(['--since', since])

    return run_step('Generate Changelog', script, args)


def publish_release(version: str, draft: bool = True) -> Dict[str, Any]:
    """Publish to GitHub release."""
    script = 'scripts/publish_models.py'
    args = ['--version', version]

    if draft:
        args.append('--draft')

    return run_step('Publish Release', script, args)


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_pipeline(
    nodes: Optional[List[str]] = None,
    epochs: int = 30,
    data_limit: Optional[int] = None,
    skip_fetch: bool = False,
    skip_train: bool = False,
    skip_validate: bool = False,
    publish: bool = False,
    version: Optional[str] = None,
    draft: bool = True,
) -> Dict[str, Any]:
    """
    Run the full training pipeline.

    Args:
        nodes: List of node IDs to process (None for all)
        epochs: Training epochs
        data_limit: Limit on training data samples
        skip_fetch: Skip data fetching step
        skip_train: Skip training step
        skip_validate: Skip validation step
        publish: Publish to GitHub release
        version: Release version
        draft: Create draft release

    Returns:
        Pipeline results
    """
    all_nodes = [
        'VSTAR-001', 'FLARE-001', 'ASTERO-001', 'SUPERNOVA-001',
        'SPECTYPE-001', 'MICROLENS-001', 'GALAXY-001',
    ]

    if nodes is None or 'all' in nodes:
        nodes = all_nodes

    print(f"\n{'#'*60}")
    print(f" LARUN Training Pipeline")
    print(f"{'#'*60}")
    print(f"  Nodes: {', '.join(nodes)}")
    print(f"  Epochs: {epochs}")
    print(f"  Publish: {publish}")
    if version:
        print(f"  Version: {version}")

    pipeline_start = datetime.utcnow()
    results = {
        'started_at': pipeline_start.isoformat() + 'Z',
        'nodes': nodes,
        'steps': [],
    }

    # Step 1: Fetch data
    if not skip_fetch:
        step = fetch_data(nodes, data_limit)
        results['steps'].append(step)

        if not step['success']:
            print("\n Pipeline failed at: Fetch Data")
            results['success'] = False
            return results

    # Step 2: Train models
    if not skip_train:
        step = train_models(nodes, epochs)
        results['steps'].append(step)

        if not step['success']:
            print("\n Pipeline failed at: Train Models")
            results['success'] = False
            return results

    # Step 3: Validate models
    if not skip_validate:
        step = validate_models(nodes)
        results['steps'].append(step)

        if not step['success']:
            print("\n Pipeline failed at: Validate Models")
            results['success'] = False
            return results

    # Step 4: Bundle and publish (optional)
    if publish and version:
        # Bundle
        step = bundle_models(version)
        results['steps'].append(step)

        if not step['success']:
            print("\n Pipeline failed at: Bundle Models")
            results['success'] = False
            return results

        # Changelog
        step = generate_changelog(version)
        results['steps'].append(step)

        # Publish
        step = publish_release(version, draft=draft)
        results['steps'].append(step)

        if not step['success']:
            print("\n Pipeline failed at: Publish Release")
            results['success'] = False
            return results

    # Pipeline complete
    pipeline_end = datetime.utcnow()
    duration = (pipeline_end - pipeline_start).total_seconds()

    results['success'] = True
    results['completed_at'] = pipeline_end.isoformat() + 'Z'
    results['total_duration_seconds'] = duration

    # Summary
    print(f"\n{'#'*60}")
    print(f" Pipeline Complete")
    print(f"{'#'*60}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Steps completed: {len(results['steps'])}")

    for step in results['steps']:
        status = "" if step['success'] else ""
        print(f"    {status} {step['name']}: {step['duration_seconds']:.1f}s")

    return results


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run LARUN training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all nodes
    python scripts/train_pipeline.py --all

    # Train single node with more epochs
    python scripts/train_pipeline.py --node VSTAR-001 --epochs 50

    # Full pipeline with release
    python scripts/train_pipeline.py --all --publish --version v2.1.0

    # Skip data fetching (use existing data)
    python scripts/train_pipeline.py --all --skip-fetch
        """
    )

    parser.add_argument('--node', '-n', action='append',
                       help='Node ID to train (can specify multiple)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Train all nodes')
    parser.add_argument('--epochs', '-e', type=int, default=30,
                       help='Training epochs (default: 30)')
    parser.add_argument('--data-limit', type=int,
                       help='Limit training data samples')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip data fetching step')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip-validate', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--publish', '-p', action='store_true',
                       help='Publish to GitHub release')
    parser.add_argument('--version', '-v',
                       help='Release version (required for publish)')
    parser.add_argument('--draft', action='store_true', default=True,
                       help='Create draft release (default: True)')
    parser.add_argument('--no-draft', action='store_false', dest='draft',
                       help='Create non-draft release')
    parser.add_argument('--output', '-o', type=Path,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Determine nodes
    if args.all:
        nodes = None  # Will be expanded to all nodes
    elif args.node:
        nodes = args.node
    else:
        parser.print_help()
        print("\nError: Specify --node or --all")
        sys.exit(1)

    # Check version for publish
    if args.publish and not args.version:
        print("Error: --version is required when using --publish")
        sys.exit(1)

    # Run pipeline
    results = run_pipeline(
        nodes=nodes,
        epochs=args.epochs,
        data_limit=args.data_limit,
        skip_fetch=args.skip_fetch,
        skip_train=args.skip_train,
        skip_validate=args.skip_validate,
        publish=args.publish,
        version=args.version,
        draft=args.draft,
    )

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    # Exit code
    sys.exit(0 if results.get('success') else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Bundle LARUN Models for Release
================================

Creates distribution bundles of trained models for release.
Generates ZIP archives with models, metadata, and checksums.

Usage:
    python scripts/bundle_models.py
    python scripts/bundle_models.py --version v2.1.0
    python scripts/bundle_models.py --output dist/
"""

import os
import sys
import argparse
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import zipfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Configuration
# =============================================================================

NODE_CONFIGS = {
    'EXOPLANET-001': {
        'folder': 'exoplanet',
        'model_file': 'detector.tflite',
        'description': 'Exoplanet transit detector',
    },
    'VSTAR-001': {
        'folder': 'variable_star',
        'model_file': 'classifier.tflite',
        'description': 'Variable star classifier',
    },
    'FLARE-001': {
        'folder': 'flare',
        'model_file': 'detector.tflite',
        'description': 'Stellar flare detector',
    },
    'ASTERO-001': {
        'folder': 'asteroseismo',
        'model_file': 'analyzer.tflite',
        'description': 'Asteroseismology analyzer',
    },
    'SUPERNOVA-001': {
        'folder': 'supernova',
        'model_file': 'detector.tflite',
        'description': 'Supernova/transient detector',
    },
    'GALAXY-001': {
        'folder': 'galaxy',
        'model_file': 'classifier.tflite',
        'description': 'Galaxy morphology classifier',
    },
    'SPECTYPE-001': {
        'folder': 'spectral_type',
        'model_file': 'classifier.tflite',
        'description': 'Spectral type classifier',
    },
    'MICROLENS-001': {
        'folder': 'microlensing',
        'model_file': 'detector.tflite',
        'description': 'Microlensing event detector',
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_version_from_pyproject() -> str:
    """Read version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'

    if pyproject_path.exists():
        content = pyproject_path.read_text()
        for line in content.split('\n'):
            if line.strip().startswith('version'):
                # Parse: version = "0.1.0"
                return line.split('=')[1].strip().strip('"\'')

    return "0.0.0"


def load_node_manifest(node_path: Path) -> Dict[str, Any]:
    """Load node manifest.yaml."""
    manifest_path = node_path / 'manifest.yaml'

    if manifest_path.exists():
        try:
            import yaml
            with open(manifest_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            # Parse YAML manually for basic fields
            content = manifest_path.read_text()
            manifest = {}
            for line in content.split('\n'):
                if ':' in line and not line.strip().startswith('#'):
                    key, value = line.split(':', 1)
                    manifest[key.strip()] = value.strip()
            return manifest

    return {}


# =============================================================================
# Bundle Functions
# =============================================================================

def collect_model_info(base_path: Path) -> List[Dict[str, Any]]:
    """Collect information about all models."""
    models = []
    nodes_path = base_path / 'nodes'

    for node_id, config in NODE_CONFIGS.items():
        node_path = nodes_path / config['folder']
        model_path = node_path / 'model' / config['model_file']

        if not model_path.exists():
            print(f"  Skipping {node_id}: model not found")
            continue

        # Get model stats
        size_bytes = model_path.stat().st_size
        checksum = compute_sha256(model_path)

        # Load manifest
        manifest = load_node_manifest(node_path)

        # Load training metrics if available
        metrics_path = node_path / 'model' / 'metrics.json'
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        models.append({
            'node_id': node_id,
            'model_path': model_path,
            'model_file': config['model_file'],
            'description': config['description'],
            'size_bytes': size_bytes,
            'size_kb': round(size_bytes / 1024, 2),
            'checksum': checksum,
            'version': manifest.get('version', '1.0.0'),
            'metrics': metrics,
        })

    return models


def create_main_bundle(
    models: List[Dict[str, Any]],
    version: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Create the main model bundle ZIP."""
    bundle_name = f"larun-models-{version}.zip"
    bundle_path = output_path / bundle_name

    print(f"\nCreating bundle: {bundle_name}")

    total_size = 0

    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add each model
        for model in models:
            arc_name = f"models/{model['node_id']}/{model['model_file']}"
            zf.write(model['model_path'], arc_name)
            print(f"  Added: {arc_name} ({model['size_kb']} KB)")
            total_size += model['size_bytes']

        # Add registry
        registry = {
            'version': version,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'models': [
                {
                    'node_id': m['node_id'],
                    'file': f"models/{m['node_id']}/{m['model_file']}",
                    'description': m['description'],
                    'size_kb': m['size_kb'],
                    'checksum_sha256': m['checksum'],
                    'model_version': m['version'],
                    'metrics': m['metrics'],
                }
                for m in models
            ],
        }
        zf.writestr('registry.json', json.dumps(registry, indent=2))

        # Add README
        readme = generate_bundle_readme(models, version)
        zf.writestr('README.md', readme)

    bundle_size = bundle_path.stat().st_size
    print(f"\nBundle created: {bundle_path}")
    print(f"  Total model size: {total_size / 1024:.1f} KB")
    print(f"  Bundle size: {bundle_size / 1024:.1f} KB")

    return {
        'path': bundle_path,
        'size_bytes': bundle_size,
        'models_count': len(models),
        'checksum': compute_sha256(bundle_path),
    }


def create_individual_bundles(
    models: List[Dict[str, Any]],
    version: str,
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Create individual model bundles for each node."""
    bundles = []

    for model in models:
        bundle_name = f"{model['node_id']}-{version}.zip"
        bundle_path = output_path / bundle_name

        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model
            zf.write(model['model_path'], model['model_file'])

            # Add metadata
            metadata = {
                'node_id': model['node_id'],
                'version': model['version'],
                'file': model['model_file'],
                'size_kb': model['size_kb'],
                'checksum_sha256': model['checksum'],
                'metrics': model['metrics'],
                'created_at': datetime.utcnow().isoformat() + 'Z',
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))

        bundles.append({
            'node_id': model['node_id'],
            'path': bundle_path,
            'size_bytes': bundle_path.stat().st_size,
            'checksum': compute_sha256(bundle_path),
        })

    return bundles


def create_checksums_file(
    main_bundle: Dict[str, Any],
    individual_bundles: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Create checksums.txt file."""
    checksums_path = output_path / 'checksums.txt'

    with open(checksums_path, 'w') as f:
        f.write("# LARUN Model Bundle Checksums\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("# Main Bundle\n")
        f.write(f"{main_bundle['checksum']}  {main_bundle['path'].name}\n\n")

        f.write("# Individual Bundles\n")
        for bundle in individual_bundles:
            f.write(f"{bundle['checksum']}  {bundle['path'].name}\n")

        f.write("\n# Individual Model Files\n")
        for model in models:
            f.write(f"{model['checksum']}  {model['node_id']}/{model['model_file']}\n")

    return checksums_path


def generate_bundle_readme(models: List[Dict[str, Any]], version: str) -> str:
    """Generate README for the bundle."""
    lines = [
        f"# LARUN Models Bundle v{version}",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Contents",
        "",
    ]

    total_size = 0
    for model in models:
        lines.append(f"### {model['node_id']}")
        lines.append(f"- **Description**: {model['description']}")
        lines.append(f"- **File**: `models/{model['node_id']}/{model['model_file']}`")
        lines.append(f"- **Size**: {model['size_kb']} KB")
        lines.append(f"- **Version**: {model['version']}")

        if model['metrics']:
            if 'accuracy' in model['metrics']:
                lines.append(f"- **Accuracy**: {model['metrics']['accuracy']:.1%}")

        lines.append("")
        total_size += model['size_kb']

    lines.extend([
        "## Installation",
        "",
        "```bash",
        "# Extract bundle",
        f"unzip larun-models-{version}.zip",
        "",
        "# Copy models to LARUN nodes directory",
        "cp -r models/* ~/.larun/nodes/",
        "",
        "# Or use LARUN CLI",
        f"larun node install --bundle larun-models-{version}.zip",
        "```",
        "",
        "## Verification",
        "",
        "Verify file integrity using checksums:",
        "",
        "```bash",
        "sha256sum -c checksums.txt",
        "```",
        "",
        f"**Total size**: {total_size:.1f} KB",
    ])

    return "\n".join(lines)


# =============================================================================
# Main Function
# =============================================================================

def bundle_models(
    version: Optional[str] = None,
    output_dir: Optional[Path] = None,
    individual: bool = True,
) -> Dict[str, Any]:
    """
    Create model bundles for release.

    Args:
        version: Version string (defaults to pyproject.toml version)
        output_dir: Output directory (defaults to dist/)
        individual: Also create individual node bundles

    Returns:
        Dict with bundle information
    """
    base_path = Path(__file__).parent.parent

    if version is None:
        version = get_version_from_pyproject()

    if output_dir is None:
        output_dir = base_path / 'dist'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f" Bundling LARUN Models v{version}")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")

    # Collect model info
    models = collect_model_info(base_path)

    if not models:
        print("\nNo models found!")
        return {'success': False, 'error': 'No models found'}

    print(f"\nFound {len(models)} models")

    # Create main bundle
    main_bundle = create_main_bundle(models, version, output_dir)

    # Create individual bundles
    individual_bundles = []
    if individual:
        print("\nCreating individual bundles...")
        individual_bundles = create_individual_bundles(models, version, output_dir)
        for bundle in individual_bundles:
            print(f"  {bundle['path'].name}: {bundle['size_bytes'] / 1024:.1f} KB")

    # Create checksums
    checksums_path = create_checksums_file(
        main_bundle, individual_bundles, models, output_dir
    )
    print(f"\nChecksums saved to: {checksums_path}")

    # Summary
    print(f"\n{'='*60}")
    print(" Bundle Summary")
    print(f"{'='*60}")
    print(f"  Main bundle: {main_bundle['path'].name}")
    print(f"  Size: {main_bundle['size_bytes'] / 1024:.1f} KB")
    print(f"  Models: {main_bundle['models_count']}")
    print(f"  Individual bundles: {len(individual_bundles)}")

    return {
        'success': True,
        'version': version,
        'main_bundle': str(main_bundle['path']),
        'individual_bundles': [str(b['path']) for b in individual_bundles],
        'checksums': str(checksums_path),
        'models_count': len(models),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Bundle LARUN models for release',
    )

    parser.add_argument('--version', '-v',
                       help='Version string (e.g., v2.1.0)')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output directory (default: dist/)')
    parser.add_argument('--no-individual', action='store_true',
                       help='Skip individual node bundles')

    args = parser.parse_args()

    result = bundle_models(
        version=args.version,
        output_dir=args.output,
        individual=not args.no_individual,
    )

    if not result['success']:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()

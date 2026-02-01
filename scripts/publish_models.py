#!/usr/bin/env python3
"""
Publish Models to GitHub Releases
==================================

Uploads trained models to GitHub Releases for distribution.
Creates release assets with checksums and metadata.

Usage:
    python scripts/publish_models.py --version v2.1.0
    python scripts/publish_models.py --version v2.1.0 --draft
    python scripts/publish_models.py --version v2.1.0 --node VSTAR-001

Requires:
    - GitHub CLI (gh) installed and authenticated
    - GITHUB_TOKEN environment variable (for CI)
"""

import os
import sys
import argparse
import json
import hashlib
import subprocess
import tempfile
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

NODE_INFO = {
    'EXOPLANET-001': {'folder': 'exoplanet', 'model': 'detector.tflite'},
    'VSTAR-001': {'folder': 'variable_star', 'model': 'classifier.tflite'},
    'FLARE-001': {'folder': 'flare', 'model': 'detector.tflite'},
    'ASTERO-001': {'folder': 'asteroseismo', 'model': 'analyzer.tflite'},
    'SUPERNOVA-001': {'folder': 'supernova', 'model': 'detector.tflite'},
    'GALAXY-001': {'folder': 'galaxy', 'model': 'classifier.tflite'},
    'SPECTYPE-001': {'folder': 'spectral_type', 'model': 'classifier.tflite'},
    'MICROLENS-001': {'folder': 'microlensing', 'model': 'detector.tflite'},
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


def run_gh_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run GitHub CLI command."""
    cmd = ['gh'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running gh command: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)

    return result


def check_gh_auth() -> bool:
    """Check if GitHub CLI is authenticated."""
    try:
        result = run_gh_command(['auth', 'status'], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: GitHub CLI (gh) not found. Install from https://cli.github.com")
        return False


# =============================================================================
# Model Collection Functions
# =============================================================================

def collect_model(node_id: str, base_path: Path) -> Optional[Dict[str, Any]]:
    """
    Collect model file and metadata for a node.

    Returns dict with path, checksum, and metadata.
    """
    if node_id not in NODE_INFO:
        print(f"  Warning: Unknown node {node_id}")
        return None

    info = NODE_INFO[node_id]
    node_path = base_path / 'nodes' / info['folder']
    model_path = node_path / 'model' / info['model']

    if not model_path.exists():
        print(f"  Warning: Model not found for {node_id}: {model_path}")
        return None

    # Get model stats
    size_kb = model_path.stat().st_size / 1024
    checksum = compute_sha256(model_path)

    # Load metrics if available
    metrics_path = node_path / 'model' / 'metrics.json'
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load manifest
    manifest_path = node_path / 'manifest.yaml'
    version = "1.0.0"
    if manifest_path.exists():
        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
            version = manifest.get('version', '1.0.0')

    return {
        'node_id': node_id,
        'path': model_path,
        'size_kb': round(size_kb, 2),
        'checksum': checksum,
        'version': version,
        'metrics': metrics,
    }


def collect_all_models(base_path: Path) -> List[Dict[str, Any]]:
    """Collect all node models."""
    models = []

    for node_id in NODE_INFO:
        model = collect_model(node_id, base_path)
        if model:
            models.append(model)

    return models


# =============================================================================
# Asset Creation Functions
# =============================================================================

def create_bundle(
    models: List[Dict[str, Any]],
    version: str,
    output_dir: Path,
) -> Path:
    """
    Create a zip bundle containing all models.

    Returns path to the bundle.
    """
    bundle_name = f"larun-models-{version}.zip"
    bundle_path = output_dir / bundle_name

    print(f"\n  Creating bundle: {bundle_name}")

    with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for model in models:
            # Add model file
            arc_name = f"{model['node_id']}/{model['path'].name}"
            zf.write(model['path'], arc_name)
            print(f"    Added: {arc_name} ({model['size_kb']}KB)")

        # Add metadata
        metadata = {
            'version': version,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'models': [
                {
                    'node_id': m['node_id'],
                    'file': f"{m['node_id']}/{m['path'].name}",
                    'size_kb': m['size_kb'],
                    'checksum': m['checksum'],
                    'model_version': m['version'],
                    'metrics': m['metrics'],
                }
                for m in models
            ],
        }

        zf.writestr('metadata.json', json.dumps(metadata, indent=2))

    bundle_size = bundle_path.stat().st_size / 1024
    print(f"    Bundle size: {bundle_size:.1f}KB")

    return bundle_path


def create_checksums(
    models: List[Dict[str, Any]],
    bundle_path: Path,
    output_dir: Path,
) -> Path:
    """Create checksums file for all assets."""
    checksums_path = output_dir / 'checksums.txt'

    with open(checksums_path, 'w') as f:
        f.write(f"# LARUN Model Checksums\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")

        # Bundle checksum
        bundle_checksum = compute_sha256(bundle_path)
        f.write(f"{bundle_checksum}  {bundle_path.name}\n")

        # Individual model checksums
        f.write("\n# Individual Models\n")
        for model in models:
            f.write(f"{model['checksum']}  {model['node_id']}-{model['path'].name}\n")

    return checksums_path


def copy_individual_models(
    models: List[Dict[str, Any]],
    version: str,
    output_dir: Path,
) -> List[Path]:
    """Copy individual model files for release."""
    paths = []

    for model in models:
        # Name format: VSTAR-001-v1.2.0.tflite
        new_name = f"{model['node_id']}-v{model['version']}.tflite"
        dest = output_dir / new_name
        shutil.copy2(model['path'], dest)
        paths.append(dest)

    return paths


# =============================================================================
# GitHub Release Functions
# =============================================================================

def create_release(
    version: str,
    assets: List[Path],
    changelog: str,
    draft: bool = False,
    prerelease: bool = False,
) -> str:
    """
    Create a GitHub release with assets.

    Returns the release URL.
    """
    print(f"\n  Creating GitHub release {version}...")

    # Build gh release create command
    args = ['release', 'create', version]

    if draft:
        args.append('--draft')

    if prerelease:
        args.append('--prerelease')

    args.extend(['--title', f"LARUN Models {version}"])
    args.extend(['--notes', changelog])

    # Add assets
    for asset in assets:
        args.append(str(asset))

    result = run_gh_command(args)

    # Parse release URL from output
    release_url = result.stdout.strip()
    print(f"  Release created: {release_url}")

    return release_url


def upload_asset(version: str, asset_path: Path) -> None:
    """Upload an additional asset to an existing release."""
    args = ['release', 'upload', version, str(asset_path)]
    run_gh_command(args)
    print(f"    Uploaded: {asset_path.name}")


# =============================================================================
# Main Function
# =============================================================================

def publish_models(
    version: str,
    node_id: Optional[str] = None,
    draft: bool = False,
    prerelease: bool = False,
    changelog: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Publish models to GitHub release.

    Args:
        version: Release version (e.g., 'v2.1.0')
        node_id: Optional specific node to publish
        draft: Create as draft release
        prerelease: Mark as prerelease
        changelog: Release notes content

    Returns:
        Dict with release information
    """
    base_path = Path(__file__).parent.parent

    print(f"\n{'='*60}")
    print(f" Publishing LARUN Models {version}")
    print(f"{'='*60}")

    # Check GitHub auth
    if not check_gh_auth():
        return {'success': False, 'error': 'GitHub CLI not authenticated'}

    # Collect models
    if node_id:
        model = collect_model(node_id, base_path)
        models = [model] if model else []
    else:
        models = collect_all_models(base_path)

    if not models:
        return {'success': False, 'error': 'No models found'}

    print(f"\n  Found {len(models)} models:")
    for m in models:
        print(f"    - {m['node_id']}: {m['size_kb']}KB (v{m['version']})")

    # Create temporary directory for assets
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create bundle
        bundle_path = create_bundle(models, version, temp_path)

        # Copy individual models
        model_paths = copy_individual_models(models, version, temp_path)

        # Create checksums
        checksums_path = create_checksums(models, bundle_path, temp_path)

        # Prepare assets list
        assets = [bundle_path, checksums_path] + model_paths

        # Generate changelog if not provided
        if changelog is None:
            changelog = generate_changelog(models, version)

        # Save changelog
        changelog_path = temp_path / 'CHANGELOG.md'
        with open(changelog_path, 'w') as f:
            f.write(changelog)

        # Create release
        try:
            release_url = create_release(
                version,
                assets,
                changelog,
                draft=draft,
                prerelease=prerelease,
            )

            return {
                'success': True,
                'version': version,
                'release_url': release_url,
                'models_published': len(models),
                'bundle_size_kb': bundle_path.stat().st_size / 1024,
            }

        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f"Failed to create release: {e}",
            }


def generate_changelog(models: List[Dict[str, Any]], version: str) -> str:
    """Generate release notes from model metadata."""
    lines = [
        f"# LARUN Models {version}",
        "",
        f"Released: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "## Models",
        "",
    ]

    total_size = 0
    for m in models:
        lines.append(f"### {m['node_id']} (v{m['version']})")
        lines.append(f"- Size: {m['size_kb']}KB")

        if m['metrics']:
            if 'accuracy' in m['metrics']:
                lines.append(f"- Accuracy: {m['metrics']['accuracy']:.1%}")
            if 'f1_score' in m['metrics']:
                lines.append(f"- F1 Score: {m['metrics']['f1_score']:.1%}")

        lines.append("")
        total_size += m['size_kb']

    lines.extend([
        "## Installation",
        "",
        "```bash",
        f"# Download and install all models",
        f"larun node install --from-release {version}",
        "",
        f"# Or install individual models",
        f"larun node install VSTAR-001 --version {version}",
        "```",
        "",
        "## Checksums",
        "",
        "See `checksums.txt` for SHA256 verification.",
        "",
        f"**Total size**: {total_size:.1f}KB",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Publish LARUN models to GitHub Releases',
    )

    parser.add_argument('--version', '-v', required=True,
                       help='Release version (e.g., v2.1.0)')
    parser.add_argument('--node', '-n',
                       help='Publish specific node only')
    parser.add_argument('--draft', action='store_true',
                       help='Create as draft release')
    parser.add_argument('--prerelease', action='store_true',
                       help='Mark as prerelease')
    parser.add_argument('--changelog', '-c', type=Path,
                       help='Path to changelog file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be published without creating release')

    args = parser.parse_args()

    # Load changelog if provided
    changelog = None
    if args.changelog and args.changelog.exists():
        changelog = args.changelog.read_text()

    if args.dry_run:
        print("\n[DRY RUN] Would publish:")
        print(f"  Version: {args.version}")
        print(f"  Draft: {args.draft}")
        print(f"  Prerelease: {args.prerelease}")

        base_path = Path(__file__).parent.parent
        models = collect_all_models(base_path) if not args.node else [collect_model(args.node, base_path)]
        print(f"  Models: {len([m for m in models if m])} found")
        return

    result = publish_models(
        args.version,
        node_id=args.node,
        draft=args.draft,
        prerelease=args.prerelease,
        changelog=changelog,
    )

    if result['success']:
        print(f"\n Successfully published {result['models_published']} models")
        print(f"  Release URL: {result['release_url']}")
    else:
        print(f"\n Failed to publish: {result['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()

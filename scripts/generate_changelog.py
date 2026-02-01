#!/usr/bin/env python3
"""
Generate Changelog for LARUN Releases
======================================

Generates release notes from git commits and model metadata.
Follows Keep a Changelog format.

Usage:
    python scripts/generate_changelog.py
    python scripts/generate_changelog.py --version v2.1.0
    python scripts/generate_changelog.py --since v2.0.0
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Git Functions
# =============================================================================

def run_git(args: List[str]) -> str:
    """Run git command and return output."""
    result = subprocess.run(
        ['git'] + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.stdout.strip()


def get_commits_since(since_tag: Optional[str] = None) -> List[Dict[str, str]]:
    """Get commits since a tag or all commits."""
    args = ['log', '--pretty=format:%H|%s|%an|%ai']

    if since_tag:
        args.append(f'{since_tag}..HEAD')

    output = run_git(args)

    commits = []
    for line in output.split('\n'):
        if '|' in line:
            parts = line.split('|')
            commits.append({
                'hash': parts[0][:8],
                'message': parts[1],
                'author': parts[2],
                'date': parts[3][:10],
            })

    return commits


def get_tags() -> List[str]:
    """Get list of git tags sorted by version."""
    output = run_git(['tag', '-l', 'v*', '--sort=-version:refname'])
    return output.split('\n') if output else []


def get_latest_tag() -> Optional[str]:
    """Get the most recent version tag."""
    tags = get_tags()
    return tags[0] if tags else None


# =============================================================================
# Commit Classification
# =============================================================================

def classify_commit(message: str) -> str:
    """Classify a commit message into a changelog category."""
    msg_lower = message.lower()

    # Feature additions
    if any(word in msg_lower for word in ['add', 'feat', 'new', 'implement', 'create']):
        return 'added'

    # Changes/improvements
    if any(word in msg_lower for word in ['change', 'update', 'improve', 'enhance', 'refactor']):
        return 'changed'

    # Deprecations
    if any(word in msg_lower for word in ['deprecate', 'obsolete']):
        return 'deprecated'

    # Removals
    if any(word in msg_lower for word in ['remove', 'delete', 'drop']):
        return 'removed'

    # Bug fixes
    if any(word in msg_lower for word in ['fix', 'bug', 'patch', 'resolve', 'repair']):
        return 'fixed'

    # Security
    if any(word in msg_lower for word in ['security', 'vuln', 'cve', 'auth']):
        return 'security'

    # Default to changed
    return 'changed'


def categorize_commits(commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize commits by type."""
    categories = {
        'added': [],
        'changed': [],
        'deprecated': [],
        'removed': [],
        'fixed': [],
        'security': [],
    }

    for commit in commits:
        category = classify_commit(commit['message'])
        categories[category].append(commit)

    return categories


# =============================================================================
# Model Metrics
# =============================================================================

def collect_model_metrics(base_path: Path) -> Dict[str, Dict[str, Any]]:
    """Collect metrics for all models."""
    metrics = {}
    nodes_path = base_path / 'nodes'

    if not nodes_path.exists():
        return metrics

    for node_dir in nodes_path.iterdir():
        if not node_dir.is_dir():
            continue

        metrics_path = node_dir / 'model' / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                node_id = data.get('node_id', node_dir.name.upper())
                metrics[node_id] = data

    return metrics


def get_training_results(base_path: Path) -> Optional[Dict[str, Any]]:
    """Get latest training results."""
    results_path = base_path / 'nodes' / 'training_results.json'

    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    return None


# =============================================================================
# Changelog Generation
# =============================================================================

def generate_changelog(
    version: str,
    since_tag: Optional[str] = None,
    include_commits: bool = True,
    include_models: bool = True,
) -> str:
    """Generate changelog content."""
    base_path = Path(__file__).parent.parent

    lines = [
        f"# Changelog",
        "",
        "All notable changes to LARUN will be documented in this file.",
        "",
        "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),",
        "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).",
        "",
        f"## [{version}] - {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
    ]

    # Commits section
    if include_commits:
        commits = get_commits_since(since_tag)
        if commits:
            categories = categorize_commits(commits)

            category_names = {
                'added': 'Added',
                'changed': 'Changed',
                'deprecated': 'Deprecated',
                'removed': 'Removed',
                'fixed': 'Fixed',
                'security': 'Security',
            }

            for cat_key, cat_name in category_names.items():
                cat_commits = categories.get(cat_key, [])
                if cat_commits:
                    lines.append(f"### {cat_name}")
                    lines.append("")
                    for commit in cat_commits:
                        lines.append(f"- {commit['message']} ({commit['hash']})")
                    lines.append("")

    # Model metrics section
    if include_models:
        training_results = get_training_results(base_path)

        if training_results:
            lines.append("### Model Performance")
            lines.append("")
            lines.append("| Node | Accuracy | Size (KB) |")
            lines.append("|------|----------|-----------|")

            for result in training_results:
                if 'error' not in result:
                    acc = f"{result.get('accuracy', 0):.1%}"
                    size = result.get('model_size_kb', 'N/A')
                    lines.append(f"| {result['node_id']} | {acc} | {size} |")

            lines.append("")

    return "\n".join(lines)


def generate_release_notes(
    version: str,
    since_tag: Optional[str] = None,
) -> str:
    """Generate concise release notes for GitHub release."""
    base_path = Path(__file__).parent.parent

    lines = [
        f"## LARUN {version}",
        "",
        f"Released: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
    ]

    # Get commits
    commits = get_commits_since(since_tag)
    if commits:
        categories = categorize_commits(commits)

        # Only include categories with commits
        if categories['added']:
            lines.append("### What's New")
            lines.append("")
            for commit in categories['added'][:5]:  # Limit to 5
                lines.append(f"- {commit['message']}")
            lines.append("")

        if categories['fixed']:
            lines.append("### Bug Fixes")
            lines.append("")
            for commit in categories['fixed'][:5]:
                lines.append(f"- {commit['message']}")
            lines.append("")

        if categories['changed']:
            lines.append("### Improvements")
            lines.append("")
            for commit in categories['changed'][:5]:
                lines.append(f"- {commit['message']}")
            lines.append("")

    # Model info
    training_results = get_training_results(base_path)
    if training_results:
        total_size = sum(r.get('model_size_kb', 0) for r in training_results if 'error' not in r)
        avg_accuracy = sum(r.get('accuracy', 0) for r in training_results if 'error' not in r) / len(training_results)

        lines.extend([
            "### Models",
            "",
            f"- **{len(training_results)} models** included",
            f"- **Total size**: {total_size:.1f} KB",
            f"- **Average accuracy**: {avg_accuracy:.1%}",
            "",
        ])

    # Installation
    lines.extend([
        "### Installation",
        "",
        "```bash",
        "# Install from release",
        f"larun node install --from-release {version}",
        "",
        "# Or download bundle",
        f"gh release download {version} -p 'larun-models-*.zip'",
        "```",
        "",
    ])

    return "\n".join(lines)


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate changelog for LARUN releases',
    )

    parser.add_argument('--version', '-v',
                       help='Version for the changelog (e.g., v2.1.0)')
    parser.add_argument('--since', '-s',
                       help='Generate changes since this tag')
    parser.add_argument('--release-notes', '-r', action='store_true',
                       help='Generate concise release notes instead of full changelog')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output file (default: stdout)')
    parser.add_argument('--no-commits', action='store_true',
                       help='Exclude git commits')
    parser.add_argument('--no-models', action='store_true',
                       help='Exclude model metrics')

    args = parser.parse_args()

    # Determine version
    if args.version:
        version = args.version
    else:
        # Try to get from latest tag or default
        latest = get_latest_tag()
        if latest:
            # Increment patch version
            parts = latest.lstrip('v').split('.')
            parts[-1] = str(int(parts[-1]) + 1)
            version = 'v' + '.'.join(parts)
        else:
            version = 'v0.1.0'

    # Determine since tag
    since = args.since
    if not since:
        tags = get_tags()
        since = tags[0] if tags else None

    print(f"Generating changelog for {version}", file=sys.stderr)
    if since:
        print(f"  Changes since: {since}", file=sys.stderr)

    # Generate content
    if args.release_notes:
        content = generate_release_notes(version, since)
    else:
        content = generate_changelog(
            version,
            since,
            include_commits=not args.no_commits,
            include_models=not args.no_models,
        )

    # Output
    if args.output:
        args.output.write_text(content)
        print(f"Changelog saved to: {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == '__main__':
    main()

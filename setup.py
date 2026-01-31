#!/usr/bin/env python3
"""
LARUN TinyML - Setup Script

Backward-compatible setup.py for older pip versions.
Modern installation uses pyproject.toml.

Usage:
    pip install .           # Install from source
    pip install -e .        # Install in editable/development mode
    pip install -e .[dev]   # Install with development dependencies
"""

from setuptools import setup

if __name__ == "__main__":
    setup()

#!/usr/bin/env python3
"""
LARUN Discovery Pipeline
========================
Main entry point for the discovery pipeline system.

Usage:
    python larun_pipeline.py                    # Start interactive mode
    python larun_pipeline.py run TIC_123456     # Process a target
    python larun_pipeline.py dashboard          # Start web dashboard
    python larun_pipeline.py --help             # Show help

Created by: Padmanaban Veeraragavalu (Larun Engineering)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.cli import main

if __name__ == '__main__':
    main()

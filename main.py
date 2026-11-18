#!/usr/bin/env python3
"""
NEXUS - Autonomous AI Model Development Platform

Main entry point for the NEXUS CLI. Provides a unified interface for managing
the complete model development workflow from user requests to deployed models.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from nexus import __version__

from nexus.cli import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
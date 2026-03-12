"""
Unit tests for the QDS research codebase.

Run all tests with:
    python -m pytest qds_project/tests/ -v
or from within qds_project/:
    python -m pytest tests/ -v
"""

import sys
import os

# Ensure the project root is on the path so imports work regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

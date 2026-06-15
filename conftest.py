"""Pytest bootstrap: ensure the package root is importable as ``yonder_hud``
regardless of the current working directory when tests are run."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

"""Pytest bootstrap: make the module root importable as ``app.*``."""
import os
import sys

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

"""The MIV encoder and the sequence contract must not depend on SHARP/3DGS (spec §24)."""

from __future__ import annotations

import subprocess
import sys

import pytest

BLOCKER = """
import importlib.abc, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {"torch", "sharp", "sharp_spatialize", "gsplat"}:
            raise ImportError(f"blocked import of {name}")
        return None
sys.meta_path.insert(0, Blocker())
"""


@pytest.mark.parametrize(
    "module",
    ["sharp_video.contract", "sharp_video.sequence_io", "sharp_video.miv"],
)
def test_module_imports_without_sharp_or_torch(module):
    """Module imports without sharp or torch."""
    code = BLOCKER + f"import {module}\nprint('ok')\n"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"

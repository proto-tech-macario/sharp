"""Tests for scripts/stage2/run_test_matrix.py (categorisation and summary only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage2" / "run_test_matrix.py"


@pytest.fixture(scope="module")
def matrix():
    """The run_test_matrix script, imported as a module."""
    spec = importlib.util.spec_from_file_location("run_test_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,category", [
    ("static_office.mp4", "static"),
    ("camera_motion_street.MP4", "camera_motion"),
    ("camera_object_skate.mov", "camera_object"),
    ("moving_object_dog.mp4", "moving_object"),
    ("difficult_foliage.mkv", "difficult"),
    ("holiday.mp4", "uncategorized"),
])
def test_categorize(matrix, name, category):
    """Categorize."""
    assert matrix.categorize(name) == category


def test_summary_lists_missing_categories(matrix):
    """Summary lists missing categories."""
    rows = [{"clip": "static_a.mp4", "category": "static", "pipeline_ok": True,
             "sequence_ok": True, "frames": 30, "miv_bytes": 1000, "bitrate_kbps": 8.0}]
    text = matrix.summarize(rows)
    assert "| static_a.mp4 | static | PASS | PASS | 30 | 1000 | 8.000 |" in text
    assert "Test 2 - camera movement" in text and "Test 1 - static scene" not in text


def test_summary_all_covered(matrix):
    """Summary all covered."""
    rows = [{"clip": f"{key}.mp4", "category": key, "pipeline_ok": True, "sequence_ok": False}
            for key in matrix.CATEGORIES]
    text = matrix.summarize(rows)
    assert "All five" in text and "FAIL" in text

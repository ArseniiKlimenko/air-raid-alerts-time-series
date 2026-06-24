"""Tests for the visualization layer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.visualization import AlertVisualizer


def test_visualizer_init(tmp_path: Path):
    """Test that visualizer uses the correct output directory."""
    viz = AlertVisualizer(output_dir=tmp_path)
    assert viz.output_dir == tmp_path
    assert not (tmp_path / "index.html").exists()

# Detailed rendering tests would require mocking plotly, but this ensures the module is importable and constructable.

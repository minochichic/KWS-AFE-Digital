"""Tk GUI package with explicit state and per-workflow controllers."""

from .app import SweeperGUI
from .cursors import InteractivePlotCursor, MetricSelectionOverlay
from .detector_tab import DetectorTabController
from .state import AppState

__all__ = [
    "AppState",
    "DetectorTabController",
    "InteractivePlotCursor",
    "MetricSelectionOverlay",
    "SweeperGUI",
]

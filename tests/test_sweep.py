"""Sweep helper tests (no dataset, no training)."""

from __future__ import annotations

from pathlib import Path

from experiments.sweep import (
    envelope_ms_for_T,
    print_summary,
    time_axis_broken,
)

BASE = str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")


def test_envelope_window_follows_T() -> None:
    assert envelope_ms_for_T(40, 0.0) == 25.0        # native_T 40 covers T=40
    assert envelope_ms_for_T(64, 0.0) == 10.0        # native_T 100 covers T<=100
    assert envelope_ms_for_T(128, 0.0) == 10.0
    assert envelope_ms_for_T(40, 15.0) == 15.0       # explicit override wins


def test_time_axis_broken_flags_small_T() -> None:
    # T=96 -> 48 frames after conv1's stride 2 < conv2 span 57 -> broken
    broken = time_axis_broken(BASE, C=32, T=96, envelope_ms=10.0)
    assert broken and any("conv2" in n for n in broken)
    # T=128 -> 64 frames > 57 -> sound
    assert time_axis_broken(BASE, C=32, T=128, envelope_ms=10.0) == []


def test_print_summary_picks_smallest_passing(capsys) -> None:
    results = [
        {"C": 16, "T": 40, "params": 40_000, "test_acc": 0.78, "meets_target": False},
        {"C": 32, "T": 64, "params": 89_000, "test_acc": 0.86, "meets_target": True},
        {"C": 64, "T": 128, "params": 96_000, "test_acc": 0.88, "meets_target": True},
    ]
    print_summary(results)
    out = capsys.readouterr().out
    assert "smallest config >= 85%: C=32 T=64" in out


def test_print_summary_reports_no_pass(capsys) -> None:
    results = [{"C": 16, "T": 40, "params": 40_000, "test_acc": 0.80,
                "meets_target": False}]
    print_summary(results)
    assert "no config reached 85%" in capsys.readouterr().out

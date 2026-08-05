#!/usr/bin/env python3
"""16-channel ngspice AC/OP/transient runner and GUI compatibility entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from sweeper import *  # noqa: F401,F403 - preserve the v13 public import surface.
from sweeper.constants import DEFAULT_NETLIST, DEFAULT_RESULTS, DEFAULT_TABLE
from sweeper.gui import SweeperGUI
from sweeper.models import SweepSettings
from sweeper.results import (
    _RAW_NUMBER,
    _canonical_vector_name,
    _log_frequency_crossing,
    _quadratic_peak,
    _raw_numbers,
    _vector_index,
)
from sweeper.simulation import (
    Simulator,
    find_ngspice,
    parse_channel_selection,
    validate_generation,
)
from sweeper.components import load_channels


def build_parser() -> argparse.ArgumentParser:
    """Create CLI options for validation, headless execution, or the GUI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--netlist", type=Path, default=DEFAULT_NETLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ngspice", default="")
    parser.add_argument("--channels", default="all", help="예: 0,3,7 또는 all")
    parser.add_argument(
        "--analysis",
        choices=("ac", "op", "both"),
        default="both",
        help="실행할 해석 종류: ac, op 또는 both (기본: both)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested CLI mode and translate failures into exit code 1."""

    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            validate_generation(args.table, args.netlist, args.output_dir)
            return 0
        if args.headless:
            channels = load_channels(args.table)
            selected = parse_channel_selection(args.channels, channels)
            simulator = Simulator()
            results = simulator.run(
                selected,
                args.netlist,
                args.output_dir,
                find_ngspice(args.ngspice),
                SweepSettings(),
                args.analysis,
            )
            return 0 if len(results) == len(selected) else 2
        import tkinter as tk

        root = tk.Tk()
        SweeperGUI(root)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

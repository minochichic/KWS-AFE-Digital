#!/usr/bin/env python3
"""Small ngspice command-line test double that writes deterministic rawfiles."""

from __future__ import annotations

import sys
from pathlib import Path


def _transient_raw() -> str:
    """Return four deterministic points for the four analog viewer traces."""

    return (
        "Title: fake transient\n"
        "Plotname: Transient Analysis\n"
        "Flags: real\n"
        "No. Variables: 5\n"
        "No. Points: 4\n"
        "Variables:\n"
        "\t0\ttime\ttime\n"
        "\t1\tv(/vin)\tvoltage\n"
        "\t2\tv(/v_filt_out)\tvoltage\n"
        "\t3\tv(/v_detect_out)\tvoltage\n"
        "\t4\tv(/v_ref)\tvoltage\n"
        "Values:\n"
        "0\t0\n\t0.9\n\t0.9\n\t0.90\n\t0.91\n"
        "1\t0.001\n\t0.901\n\t0.9005\n\t0.92\n\t0.91\n"
        "2\t0.002\n\t0.899\n\t0.8995\n\t0.93\n\t0.91\n"
        "3\t0.003\n\t0.9\n\t0.9\n\t0.90\n\t0.91\n"
    )


def main() -> int:
    """Implement only the ``-o``/``-r`` batch contract used by the runner."""

    args = sys.argv[1:]
    log_path = Path(args[args.index("-o") + 1])
    raw_path = Path(args[args.index("-r") + 1])
    netlist_path = Path(args[-1])
    text = netlist_path.read_text(encoding="utf-8")
    if ".tran " not in text.lower():
        log_path.write_text("missing .tran directive\n", encoding="utf-8")
        return 2
    if "stimulus.pwl.inc" not in text:
        log_path.write_text("missing PWL include\n", encoding="utf-8")
        return 3
    raw_path.write_text(_transient_raw(), encoding="utf-8")
    log_path.write_text(
        "No. of Data Rows : 4\nfake transient done\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

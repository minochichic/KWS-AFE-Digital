"""Hardware figures drawn from the Vivado report files (no hand-copied values).

    out/.venv_report/Scripts/python docs/report/make_hw_figures.py out/build/bd_base_bal
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("out/report_build/img")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

BLOCKS = [("u_c1", "conv1"), ("u_b1", "B1"), ("u_b2", "B2"), ("u_b3", "B3"),
          ("u_c2", "conv2"), ("u_tail", "tail"), ("u_pa", "plane A"),
          ("u_pb", "plane B"), ("u_pc", "plane C"), ("u_pd", "plane D")]


def slack_hist(rep: Path) -> None:
    s = []
    with open(rep / "setup_slack_endpoints.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                s.append(float(row["slack_ns"]))
            except ValueError:
                pass
    s = np.asarray(s)
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    ax.hist(s, bins=60, color="0.55", edgecolor="k", linewidth=0.3)
    ax.axvline(s.min(), color="k", linestyle="--", linewidth=0.7)
    ax.text(s.min() + 0.2, ax.get_ylim()[1] * 0.85, f"WNS = {s.min():+.3f} ns",
            fontsize=7)
    ax.set_xlabel("Setup slack (ns), clock period 20 ns")
    ax.set_ylabel("Endpoints")
    ax.set_yscale("log")
    fig.savefig(OUT / "fig_slack_hist.png", dpi=600)
    plt.close(fig)
    print(f"slack: {len(s)} endpoints, min {s.min():.3f}")


def util_bars(rep: Path) -> None:
    txt = (rep / "util_hier.rpt").read_text()
    lut, ff = [], []
    for inst, _ in BLOCKS:
        m = re.search(rf"^\|\s+{inst}\s+\|[^|]+\|\s+(\d+)\s+\|\s+\d+\s+\|\s+\d+\s+\|"
                      rf"\s+\d+\s+\|\s+(\d+)\s+\|", txt, re.M)
        lut.append(int(m.group(1)))
        ff.append(int(m.group(2)))
    x = np.arange(len(BLOCKS))
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.bar(x - 0.2, lut, 0.4, color="0.3", edgecolor="k", linewidth=0.4, label="LUT")
    ax.bar(x + 0.2, ff, 0.4, color="0.8", edgecolor="k", linewidth=0.4, label="FF")
    ax.set_xticks(x)
    ax.set_xticklabels([b for _, b in BLOCKS], rotation=45, ha="right")
    ax.set_ylabel("Cells")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linewidth=0.3, color="0.8")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_util_blocks.png", dpi=600)
    plt.close(fig)
    print("util:", dict(zip([b for _, b in BLOCKS], lut)))


if __name__ == "__main__":
    rep = Path(sys.argv[1]) / "report"
    OUT.mkdir(parents=True, exist_ok=True)
    slack_hist(rep)
    util_bars(rep)

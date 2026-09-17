"""Block diagrams for the interim report -- plain black-on-white, paper style.

The diagrams under docs/diagrams/ are working notes (colour, Korean prose inside
the boxes, some describing superseded settings). A report figure is a different
object: English labels, no fill beyond light grey, the caption carries the prose.

    out/.venv_report/Scripts/python docs/report/make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path("out/report_build/img")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "mathtext.fontset": "stix",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def box(ax, x, y, w, h, text, fill="white", lw=0.7, ls="-", fs=6.5, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="k",
                           linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, text="", above=True, fs=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.6, color="k"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.12 if above else -0.12),
                text, ha="center", va="bottom" if above else "top", fontsize=fs)


def canvas(w, h):
    # drawn at 1.6x and placed at column width in the document: 7 pt text
    # ends up ~6.5 pt on paper, and the boxes have room for three lines.
    fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6))
    ax.set_xlim(0, w * 2)
    ax.set_ylim(0, h * 2)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(OUT / f"{name}.png")


def fig_system():
    """Signal chain: microphone -> analog feature extractor -> FPGA."""
    fig, ax = canvas(7.16, 1.9)
    # analog side
    ax.add_patch(Rectangle((0.15, 0.35), 5.35, 3.15, fill=False, linewidth=0.6,
                           linestyle="--"))
    ax.text(2.82, 3.3, "Analog front end, 16 ch (PCB designed; simulated input in this work)",
            ha="center", fontsize=6.5, style="italic")
    box(ax, 0.35, 1.55, 0.8, 0.8, "MEMS\nmic")
    box(ax, 1.5, 1.35, 1.1, 1.2, "Band-pass\nfilter bank")
    box(ax, 2.95, 1.35, 1.05, 1.2, "Envelope\ndetector")
    box(ax, 4.35, 1.35, 0.95, 1.2, "Comparator\n$V_{env} \\geq V_{th,i}$\n(R divider)")
    arrow(ax, 1.15, 1.95, 1.5, 1.95)
    arrow(ax, 2.6, 1.95, 2.95, 1.95)
    arrow(ax, 4.0, 1.95, 4.35, 1.95)
    ax.text(2.82, 0.6, "16 × (filter, detector, comparator); thresholds $V_{th,i}$ learned end-to-end",
            ha="center", fontsize=6)
    # FPGA side
    ax.add_patch(Rectangle((6.05, 0.35), 8.1, 3.15, fill=False, linewidth=0.9))
    ax.text(10.1, 3.3, "Digital back end (Verilog RTL on XC7S75 FPGA)",
            ha="center", fontsize=6.5, style="italic")
    arrow(ax, 5.3, 1.95, 6.25, 1.95)
    ax.text(5.78, 2.1, "cmp[15:0]", ha="center", va="bottom", fontsize=6)
    box(ax, 6.25, 1.35, 1.35, 1.2, "Frame capture\n10 ms sticky-OR\n→ 16-bit frame")
    box(ax, 7.95, 1.35, 1.45, 1.2, "Sliding window\n100-frame history\nhop 10 frames\npad → T=128")
    box(ax, 9.75, 1.35, 2.25, 1.2, "Folded BinaryMatchboxNet\nconv1(int8) → B1–B3(binary)\n→ conv2(binary) → conv3(int8)\n→ conv4(fixed) → pool",
        fill="0.93")
    box(ax, 12.35, 1.35, 1.6, 1.2, "Decision\nN=5 consecutive\nmargin ≥ 4301\ncooldown 1 s")
    arrow(ax, 7.6, 1.95, 7.95, 1.95)
    arrow(ax, 9.4, 1.95, 9.75, 1.95)
    arrow(ax, 12.0, 1.95, 12.35, 1.95)
    arrow(ax, 13.95, 1.95, 14.3, 1.95)
    ax.text(14.33, 1.95, "keyword", va="center", fontsize=6.5)
    ax.text(10.1, 0.6, "weights and thresholds loaded from ROM files ($readmemh); no retraining of RTL",
            ha="center", fontsize=6)
    save(fig, "fig_system")


def fig_flow():
    """Design and verification flow with the three evaluation tiers."""
    fig, ax = canvas(7.16, 2.3)
    y = 2.3
    h = 1.3
    steps = [
        (0.2, 2.0, "Training (PyTorch)\nAFE sim + BNN, QAT\nSTE thresholds"),
        (2.55, 2.0, "Export\nBN → integer threshold\nbit-packed weights\nQ*.6 tail ROMs"),
        (4.9, 2.0, "Golden vectors\nper-layer integer\nactivations"),
        (7.25, 2.0, "RTL simulation\n(XSim / Icarus)\nunit + top, bit-exact"),
        (9.6, 2.0, "Synthesis & P&R\nVivado 2026.1\n50 MHz, XC7S75-1"),
        (11.95, 2.0, "On-chip self-test\nclip ROM → net → score\nVIO over JTAG"),
    ]
    for x, w, t in steps:
        box(ax, x, y, w, h, t)
    for (x, w, _), (x2, _, _) in zip(steps, steps[1:]):
        arrow(ax, x + w, y + h / 2, x2, y + h / 2)
    # tier brackets
    def bracket(x0, x1, label):
        ax.plot([x0, x0, x1, x1], [2.05, 1.85, 1.85, 2.05], color="k", linewidth=0.6)
        ax.text((x0 + x1) / 2, 1.7, label, ha="center", va="top", fontsize=6.5)
    bracket(0.2, 2.2, "T1: clip accuracy\n(full test set, float & integer)")
    bracket(4.9, 9.25, "bit-exact equivalence\n(layer by layer, simulation)")
    bracket(11.95, 13.95, "T2: hardware reproduction\n(chip class = integer prediction)")
    ax.text(7.1, 4.3, "T3 (always-on): 3-s spliced streams, 100-ms hop, N=5 vote — Python, validation split",
            ha="center", fontsize=6.5, style="italic")
    save(fig, "fig_flow")


def fig_selftest():
    """On-chip self-test harness."""
    fig, ax = canvas(3.5, 2.2)
    ax.add_patch(Rectangle((0.1, 0.25), 5.35, 4.0, fill=False, linewidth=0.9))
    ax.text(2.8, 4.02, "XC7S75 (kws_selftest_board)", ha="center", fontsize=6.5,
            style="italic")
    box(ax, 0.3, 2.55, 1.35, 1.15, "Clip ROM\nBRAM, 600 clips\n× 128 × 16 bit")
    box(ax, 0.3, 0.55, 1.35, 1.15, "Expected ROM\npython integer\nprediction")
    box(ax, 2.05, 2.55, 1.5, 1.15, "Network\n(kws_top,\nas deployed)", fill="0.93")
    box(ax, 2.05, 0.55, 1.5, 1.15, "Scorer FSM\ntotal / match\nfirst failure")
    box(ax, 3.95, 0.55, 1.3, 3.15, "VIO\n\nprobe_in: 58 bit\nprobe_out:\ngo, soft_rst")
    arrow(ax, 1.65, 3.12, 2.05, 3.12, "frame", fs=5.5)
    arrow(ax, 2.8, 2.55, 2.8, 1.7)
    ax.text(2.88, 2.12, "class", ha="left", va="center", fontsize=5.5)
    arrow(ax, 1.65, 1.12, 2.05, 1.12)
    arrow(ax, 3.55, 1.12, 3.95, 1.12)
    ax.add_patch(FancyArrowPatch((5.45, 2.1), (6.3, 2.1), arrowstyle="<|-|>",
                                 mutation_scale=6, linewidth=0.6, color="k"))
    ax.text(5.88, 2.22, "JTAG", ha="center", va="bottom", fontsize=5.5)
    ax.text(6.35, 2.1, "PC\n(Tcl)", va="center", fontsize=6.5)
    save(fig, "fig_selftest")


if __name__ == "__main__":
    fig_system()
    fig_flow()
    fig_selftest()

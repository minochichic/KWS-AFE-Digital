"""Figures for the project plan — plain line art, IEEE-paper style.

Emits PNG (300 dpi, for the .docx) and SVG (for the .html) from one source.

  python3 make_figures.py

Labels are English; the document's captions stay Korean. The analog side is the
colleague's work, so its internals stay as blocks -- these figures carry the
digital decision stage and the offline training instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

CH = 16                      # colleague's schematic: GIC_0 .. GIC_15
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none",
})

K = "black"
LW = 0.9
FILL = "#F2F2F2"


# ───────────────────────────────────────────── primitives
def box(ax, x, y, w, h, label="", sub="", fs=8.5, fill="white", lw=LW):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=K, lw=lw, zorder=2))
    if label:
        ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), label, ha="center",
                va="center", fontsize=fs, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.19, sub, ha="center", va="center",
                fontsize=fs - 1.7, color="#444444", zorder=3)


def dashbox(ax, x, y, w, h, title="", fs=8.4):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=K, lw=0.7,
                           linestyle=(0, (4, 3))))
    if title:
        ax.text(x + w / 2, y + h + 0.16, title, ha="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, lw=LW, head=0.09, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle=f"-|>,head_width={head*1.6:.2f},"
                                           f"head_length={head*2.6:.2f}",
                                color=K, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def wire(ax, pts, lw=LW, ls="-"):
    p = np.asarray(pts, float)
    ax.plot(p[:, 0], p[:, 1], color=K, lw=lw, ls=ls, solid_capstyle="round", zorder=1)


def bus(ax, x1, y, x2, n, lw=1.6):
    """Thick line + slash + width — how papers draw a bus."""
    ax.plot([x1, x2], [y, y], color=K, lw=lw, zorder=1)
    m = (x1 + x2) / 2
    ax.plot([m - .07, m + .07], [y - .11, y + .11], color=K, lw=LW, zorder=2)
    ax.text(m + .06, y + .26, str(n), fontsize=7.6, ha="center", va="bottom")


def dot(ax, x, y):
    ax.plot([x], [y], marker="o", ms=3, color=K, zorder=3)


def res(ax, x, y, w=0.62, h=0.2, label="", vert=False, fs=7.2):
    if vert:
        ax.add_patch(Rectangle((x - h / 2, y - w / 2), h, w, facecolor="white",
                               edgecolor=K, lw=LW, zorder=3))
        if label: ax.text(x + h / 2 + 0.1, y, label, fontsize=fs, va="center")
    else:
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="white",
                               edgecolor=K, lw=LW, zorder=3))
        if label: ax.text(x, y + h / 2 + 0.11, label, fontsize=fs, ha="center")


def cap(ax, x, y, label="", fs=7.2):
    for dy in (-0.05, 0.05):
        ax.plot([x - 0.17, x + 0.17], [y + dy, y + dy], color=K, lw=LW, zorder=3)
    if label: ax.text(x + 0.25, y, label, fontsize=fs, va="center")


def opamp(ax, x, y, w=0.82, h=0.82, label=""):
    ax.add_patch(Polygon([(x, y - h / 2), (x, y + h / 2), (x + w, y)], closed=True,
                         facecolor="white", edgecolor=K, lw=LW, zorder=3))
    for txt, yy in (("+", y + h / 4), ("-", y - h / 4)):
        ax.text(x + 0.15, yy, txt, fontsize=8.5, ha="center", va="center", zorder=4)
    if label:
        ax.text(x + w / 2, y - h / 2 - 0.24, label, fontsize=7.2, ha="center")


def diode(ax, x, y, label=""):
    ax.add_patch(Polygon([(x - 0.16, y - 0.15), (x - 0.16, y + 0.15), (x + 0.16, y)],
                         closed=True, facecolor="white", edgecolor=K, lw=LW, zorder=3))
    ax.plot([x + 0.16, x + 0.16], [y - 0.16, y + 0.16], color=K, lw=LW, zorder=3)
    if label: ax.text(x, y + 0.31, label, fontsize=7, ha="center")


def gnd(ax, x, y, size=0.17):
    wire(ax, [(x, y), (x, y - 0.16)])
    for i, s in enumerate((1.0, 0.62, 0.28)):
        ax.plot([x - size * s, x + size * s], [y - 0.16 - i * 0.075] * 2, color=K, lw=LW)


def finish(fig, name):
    for ext, kw in (("png", dict(dpi=300)), ("svg", {})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)
    print(f"  {name}.png / .svg")


def blank(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    return fig, ax


# ═════════════════════════════════ Fig. 1 — system block diagram
def fig_block():
    fig, ax = blank(7.4, 2.95)
    ax.set_xlim(0, 14.8); ax.set_ylim(0, 5.9)

    ax.add_patch(Polygon([(0.30, 3.05), (0.30, 3.95), (1.00, 4.25), (1.00, 2.75)],
                         closed=True, facecolor="white", edgecolor=K, lw=LW))
    ax.text(0.65, 2.35, "Mic", ha="center", fontsize=8)
    arrow(ax, 1.05, 3.5, 1.60, 3.5)
    box(ax, 1.60, 3.08, 1.10, 0.84, "Preamp", fs=7.6)
    arrow(ax, 2.75, 3.5, 3.35, 3.5)

    dashbox(ax, 3.35, 1.15, 4.70, 4.15, f"Analog front-end   ({CH} channels)")
    ax.text(5.70, 0.78, "(colleague)", ha="center", fontsize=7.2,
            color="#555555", style="italic")

    rows = [4.35, 3.35, 1.65]
    names = ["CH0", "CH1", f"CH{CH-1}"]
    for yy, nm in zip(rows, names):
        box(ax, 3.62, yy - 0.34, 1.02, 0.68, "BPF", fs=7.2)
        box(ax, 4.92, yy - 0.34, 1.32, 0.68, "Envelope\ndetector", fs=6.6)
        box(ax, 6.52, yy - 0.34, 1.20, 0.68, "Comparator", fs=6.4)
        arrow(ax, 4.67, yy, 4.89, yy, head=0.07)
        arrow(ax, 6.27, yy, 6.49, yy, head=0.07)
        wire(ax, [(3.35, yy), (3.59, yy)])
        wire(ax, [(7.72, yy), (8.55, yy)])
        ax.text(8.30, yy + 0.21, nm, fontsize=6.8, ha="center")
    for yy in (2.62, 2.42, 2.22):
        ax.plot([5.70], [yy], marker=".", ms=2.2, color=K)
    wire(ax, [(3.35, 1.65), (3.35, 4.35)])
    dot(ax, 3.35, 3.5)

    bus(ax, 8.55, 3.5, 9.20, CH)
    wire(ax, [(8.55, 1.65), (8.55, 4.35)])

    dashbox(ax, 9.20, 1.15, 3.55, 4.15, "Digital decision stage")
    ax.text(10.98, 0.78, "(this work)", ha="center", fontsize=7.2,
            color="#555555", style="italic")
    box(ax, 9.46, 4.18, 3.03, 0.72, "START detect", fs=7.6, fill=FILL)
    box(ax, 9.46, 3.24, 3.03, 0.72, "Timing counter  (100 Hz)", fs=7.2, fill=FILL)
    box(ax, 9.46, 2.30, 3.03, 0.72, "Template match  (4 states)", fs=7.2, fill=FILL)
    box(ax, 9.46, 1.36, 3.03, 0.72, "Latch  &  final AND", fs=7.4, fill=FILL)
    for y1, y2 in ((4.18, 3.96), (3.24, 3.02), (2.30, 2.08)):
        arrow(ax, 10.98, y1, 10.98, y2, head=0.07)

    arrow(ax, 12.78, 1.72, 13.48, 1.72)
    ax.text(13.13, 1.96, "WAKE", ha="center", fontsize=7.8)
    ax.add_patch(Polygon([(13.58, 1.42), (13.58, 2.02), (14.08, 1.72)], closed=True,
                         facecolor="white", edgecolor=K, lw=LW))
    wire(ax, [(14.08, 1.42), (14.08, 2.02)])
    for dx in (0.02, 0.18):
        arrow(ax, 13.79 + dx, 2.10, 13.97 + dx, 2.34, lw=0.7, head=0.05)
    ax.text(13.83, 1.05, "LED", ha="center", fontsize=7.6)

    finish(fig, "fig1_block")


# ═════════════════════════════════ Fig. 2 — one analog channel
def fig_channel():
    fig, ax = blank(7.4, 2.45)
    ax.set_xlim(0, 14.8); ax.set_ylim(0, 4.9)

    for x0, x1, name in ((0.25, 4.75, "(a)  Bandpass filter"),
                         (5.00, 10.05, "(b)  Envelope detector"),
                         (10.30, 14.55, "(c)  Comparator")):
        dashbox(ax, x0, 0.50, x1 - x0, 3.55, name)

    Y = 2.35
    # (a) The GIC needs two op-amps; kept as a block so the figure stays readable.
    ax.text(0.34, Y + 0.28, "$v_{in}$", fontsize=8)
    wire(ax, [(0.42, Y), (1.15, Y)])
    res(ax, 1.46, Y, label="R1")
    wire(ax, [(1.77, Y), (2.30, Y)])
    box(ax, 2.30, Y - 0.62, 1.85, 1.24, "GIC\nbandpass", "2 op-amps", fs=8, fill=FILL)
    ax.text(3.22, Y - 0.97, "$f_c$ set by RA, C", fontsize=7.4, ha="center",
            color="#333333")
    wire(ax, [(4.15, Y), (4.75, Y)])
    ax.text(4.44, Y + 0.28, "$v_{filt}$", fontsize=8, ha="center")

    # (b)
    wire(ax, [(5.00, Y), (5.45, Y)])
    res(ax, 5.76, Y, label="R4")
    wire(ax, [(6.07, Y), (6.55, Y)]); dot(ax, 6.55, Y)
    wire(ax, [(6.55, Y), (7.00, Y)])
    opamp(ax, 7.00, Y, label="U3")
    wire(ax, [(6.55, Y), (6.55, Y + 1.12)])
    res(ax, 7.55, Y + 1.12, label="R5")
    wire(ax, [(6.55, Y + 1.12), (7.24, Y + 1.12)])
    wire(ax, [(7.86, Y + 1.12), (8.90, Y + 1.12), (8.90, Y)])
    diode(ax, 8.30, Y, label="D")
    wire(ax, [(7.82, Y), (8.14, Y)]); wire(ax, [(8.46, Y), (8.90, Y)])
    dot(ax, 8.90, Y)
    cap(ax, 8.90, Y - 0.80, label="C3")
    wire(ax, [(8.90, Y), (8.90, Y - 0.70)])
    wire(ax, [(8.90, Y - 0.90), (8.90, Y - 1.08)]); gnd(ax, 8.90, Y - 1.08)
    wire(ax, [(8.90, Y), (10.05, Y)])
    ax.text(9.52, Y + 0.28, "$v_{env}$", fontsize=8, ha="center")

    # (c)
    wire(ax, [(10.30, Y), (12.05, Y)])
    opamp(ax, 12.05, Y, w=0.88, h=0.88)
    wire(ax, [(12.93, Y), (14.10, Y)])
    ax.text(13.52, Y + 0.30, "CH$_n$", fontsize=8, ha="center")
    ax.text(13.52, Y - 0.32, "0 / 1.8 V", fontsize=7.0, ha="center", color="#444444")

    XD = 11.05
    ax.text(XD - 0.44, Y + 1.36, "1.8 V", fontsize=7.4, ha="right")
    wire(ax, [(XD, Y + 1.36), (XD, Y + 1.15)])
    res(ax, XD, Y + 0.84, w=0.62, vert=True, label="R7")
    wire(ax, [(XD, Y + 0.53), (XD, Y - 0.22)]); dot(ax, XD, Y - 0.22)
    res(ax, XD, Y - 0.68, w=0.62, vert=True, label="R8")
    wire(ax, [(XD, Y - 0.99), (XD, Y - 1.18)]); gnd(ax, XD, Y - 1.18)
    wire(ax, [(XD, Y - 0.22), (12.05, Y - 0.22)])
    ax.text(11.54, Y - 0.46, "$V_{th}$", fontsize=7.6, ha="center")

    ax.text(7.4, 0.13, "R7 / R8 ratio sets the channel threshold — fixed by "
                       "offline training.",
            fontsize=7.6, ha="center", style="italic", color="#333333")
    finish(fig, "fig2_channel")


# ═════════════════════════════════ Fig. 3 — timing
def fig_timing():
    rng = np.random.default_rng(7)
    T, dt = 620, 10
    n = T // dt
    times = [12, 22, 34, 47]
    t0 = 6

    pat = np.zeros((CH, n), int)
    for c in range(CH):
        a = 7 + int(round((c * 5) % 21))          # onset spread over the word
        b = a + 15 + (c % 5) * 3
        for t in range(n):
            p = 0.88 if a <= t <= b else (0.03 if t < a - 2 or t > b + 3 else 0.35)
            pat[c, t] = rng.random() < p

    fig = plt.figure(figsize=(7.4, 4.1))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 6.6, 1.0], hspace=0.26,
                          left=0.095, right=0.985, top=0.915, bottom=0.11)

    ax0 = fig.add_subplot(gs[0])
    tt = np.linspace(0, T, 2400)
    env = np.exp(-((tt - 270) / 165) ** 2) * 0.92
    ax0.plot(tt, env * np.sin(2 * np.pi * tt / 5.5) * (0.35 + 0.65 * env),
             color=K, lw=0.45)
    ax0.set_ylabel("Mic", fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
    ax0.set_ylim(-1.15, 1.15)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for c in range(CH):
        y = CH - 1 - c
        for t in range(n):
            if pat[c, t]:
                ax1.add_patch(Rectangle((t * dt, y + 0.16), dt, 0.68,
                                        facecolor="#2B2B2B", edgecolor="none"))
        ax1.axhline(y, color="#D8D8D8", lw=0.35, zorder=0)
    ax1.set_ylim(-0.15, CH)
    ax1.set_yticks([CH - 1 - c + 0.5 for c in range(CH)])
    ax1.set_yticklabels([f"CH{c}" for c in range(CH)], fontsize=5.6)
    ax1.set_ylabel("Channel", fontsize=8, labelpad=6)

    for ax in (ax0, ax1):
        ax.axvline(t0 * dt, color=K, lw=0.9, ls=(0, (5, 3)))
    ax1.text(t0 * dt - 5, CH + 0.22, "START", fontsize=7.4, ha="right")
    for i, s in enumerate(times):
        ax1.axvline(s * dt, color=K, lw=0.8, ls=(0, (1.6, 2)))
        ax1.text(s * dt, CH + 0.22, f"$S_{i+1}$", fontsize=8, ha="center")
        ax1.annotate("", xy=(s * dt, -0.05), xytext=(s * dt, -0.85),
                     annotation_clip=False,
                     arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.3",
                                     color=K, lw=0.8))

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for t in range(t0, times[-1] + 1):
        ax2.plot([t * dt, t * dt], [0.66, 0.96], color="#888888", lw=0.4)
    ax2.text(t0 * dt + 4, 1.05, "10 ms ticks", fontsize=7, ha="left", color="#555555")
    w0 = times[-1] * dt
    ax2.plot([0, w0, w0, w0 + 90, w0 + 90, T], [0.06, 0.06, 0.44, 0.44, 0.06, 0.06],
             color=K, lw=1.1)
    ax2.text(w0 + 45, 0.52, "WAKE", fontsize=7.6, ha="center", va="bottom")
    ax2.set_ylim(-0.05, 1.30)
    ax2.set_xlabel("Time [ms]", fontsize=8)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0, T)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.7)
        ax.tick_params(labelsize=7, width=0.7, length=2.5)
    for ax in (ax0, ax2): ax.set_yticks([])
    for ax in (ax0, ax1):
        ax.tick_params(labelbottom=False, bottom=False)
        ax.spines["bottom"].set_visible(False)

    finish(fig, "fig3_timing")


# ═════════════════════════════════ Fig. 4 — offline training flow
def fig_training():
    fig, ax = blank(7.4, 2.75)
    ax.set_xlim(0, 14.8); ax.set_ylim(1.02, 6.35)

    dashbox(ax, 0.10, 1.30, 11.15, 4.45, "")
    ax.text(0.10, 5.92, "Offline  (PC)", fontsize=8.4, ha="left")

    box(ax, 0.45, 4.35, 2.15, 0.78, "Target word", "positive", fs=7.8)
    box(ax, 0.45, 3.35, 2.15, 0.78, "Other words, silence", "negative", fs=7.0)
    box(ax, 0.45, 2.35, 2.15, 0.78, "Continuous audio", "synthesized", fs=7.4)

    box(ax, 3.25, 3.20, 1.80, 1.95, "AFE\nmodel", fs=8, fill=FILL)
    for y in (4.74, 3.74, 2.74):
        arrow(ax, 2.62, y, 3.22, y, head=0.07)

    bus(ax, 5.05, 4.18, 5.80, f"{CH} bits")

    box(ax, 5.85, 4.62, 2.40, 0.72, "1.  START search", fs=7.8)
    box(ax, 5.85, 3.72, 2.40, 0.72, "2.  Times & templates", fs=7.5)
    box(ax, 5.85, 2.82, 2.40, 0.72, "3.  Match tolerance", fs=7.8)
    for y1, y2 in ((4.62, 4.44), (3.72, 3.54)):
        arrow(ax, 7.05, y1, 7.05, y2, head=0.07)
    wire(ax, [(5.82, 4.18), (5.82, 4.98)])

    box(ax, 8.70, 3.30, 2.25, 1.75, "State-machine\neval", "TPR  /  false alarms",
        fs=7.8, fill=FILL)
    arrow(ax, 8.25, 4.18, 8.67, 4.18, head=0.07)
    wire(ax, [(9.82, 3.28), (9.82, 1.80), (4.15, 1.80)], ls=(0, (4, 3)))
    arrow(ax, 4.15, 1.80, 4.15, 3.16, head=0.07, ls=(0, (4, 3)))
    ax.text(7.0, 1.52, "threshold refinement  (iterate)", fontsize=7.4, ha="center",
            color="#333333", style="italic")

    arrow(ax, 11.28, 4.18, 12.00, 4.18)
    dashbox(ax, 12.00, 1.30, 2.65, 4.45, "")
    ax.text(13.32, 5.92, "Circuit  (fixed)", fontsize=8.4, ha="center")
    box(ax, 12.22, 4.42, 2.20, 0.72, "Comparator refs", f"{CH} dividers", fs=7.2)
    box(ax, 12.22, 3.42, 2.20, 0.72, "Template wiring", "resistor array", fs=7.2)
    box(ax, 12.22, 2.42, 2.20, 0.72, "Decode counts", "decoder wiring", fs=7.2)
    ax.text(13.32, 1.88, "no software", fontsize=7.6, ha="center", style="italic",
            color="#333333")

    finish(fig, "fig4_training")


if __name__ == "__main__":
    print(f"CH = {CH}\nout: {OUT}")
    fig_block(); fig_channel(); fig_timing(); fig_training()

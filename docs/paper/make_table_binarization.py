"""IEEE-style tables: what binarization costs and saves, on the same FPGA.

    .venv/bin/python docs/paper/make_table_binarization.py

Writes, each as .tex (booktabs, for the paper), .svg and .png (preview):
  table_binarization  -- TABLE IV, the three networks (power in W)
  table_mac_bench     -- TABLE V, one MAC engine of each kind (rtl/bench)

Every number is derived here from its source -- op counts
(docs/conv2_int8_compare.md 4, 9.2), the MAC micro-benchmark (9.6,
rtl/bench/run_mac_bench.tcl), accuracies (9.1, 9.5) -- so the files cannot
drift from each other or from the measurements.
"""
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- MAC micro-benchmark (Vivado post-route, vectorless, xc7s75-1) --------- #
CLK = 50e6
DUTY = 64 / 65                       # one start cycle per 64-word vector
LANE_RATE = CLK * 32 * DUTY          # terms/s at 32 terms/cycle
#        name, operands, terms/cycle, LUT, FF, DSP, WNS, engine mW (net of stim)
BENCH = [
    ("Binary, 1 engine",  "1\\,b $\\times$ 1\\,b",  32,  173,  49,  0,  9.08,  3.0),
    ("Binary, 8 engines", "1\\,b $\\times$ 1\\,b", 256,  707, 336,  0,  8.57, 15.0),
    ("INT8 $\\times\\,\\pm1$ (add/sub)", "1\\,b $\\times$ 8\\,b", 32, 1050, 775, 0, 9.94, 24.0),
    ("INT8 MAC, DSP48", "8\\,b $\\times$ 8\\,b",   32, 1042, 787, 32, 12.03, 43.0),
    ("INT8 MAC, LUT only", "8\\,b $\\times$ 8\\,b", 32, 3646, 786,  0,  8.69, 61.0),
]
BENCH_REF = 1                         # the 8-engine build: 1 mW steps matter less


def p_at32(b):                        # engine power scaled to 32 terms/cycle
    return b[7] * 32 / b[2]


P32 = {"xnor": p_at32(BENCH[1]), "add": p_at32(BENCH[2]), "mac": p_at32(BENCH[3])}   # mW
PJ = {k: v * 1e-3 / LANE_RATE * 1e12 for k, v in P32.items()}                          # pJ/term

# ---- networks ----------------------------------------------------------------- #
BASE_CYC = 2.87e6                     # bd_base, on chip
BIN_LAYER_CYC = 122_880               # B1-B3 + conv2 binary engines
CONV2_BIN_CYC = 20_480
RATE_HZ = 10                          # inference rate (CLAUDE.md 0)
NETS = [
    # name, binarized, weight bits, xnor, add, mac, acc, latency_cycles, dsp, measured
    ("Binary (proposed)", "B1--B3, Conv2", 375_552,
     3_424_256, 1_441_792, 1_146_880, 82.49, BASE_CYC, 0, True),
    ("Conv2 $\\rightarrow$ INT8", "B1--B3", 445_888,
     2_781_184, 1_560_576, 1_671_168, 82.84,
     BASE_CYC - CONV2_BIN_CYC + 1_856 * 64 + 8_192 * 64, 1, False),
    ("All INT8 (W8A8)", "--", 750_080,
     0, 1_441_792, 4_571_136, 86.68,
     BASE_CYC - BIN_LAYER_CYC + 3_424_256 / 2, 30, False),
]
NET_REF, ACC_REF = 2, 0


def energy_j(r):
    return (r[3] * PJ["xnor"] + r[4] * PJ["add"] + r[5] * PJ["mac"]) * 1e-12


net_cells = []
for i, r in enumerate(NETS):
    lat = r[7] / CLK
    p_inf = energy_j(r) / lat * 1e3                 # mW, while the network runs
    p_avg = energy_j(r) * RATE_HZ * 1e3             # mW, averaged at 10 Hz
    p_avg_ref = energy_j(NETS[NET_REF]) * RATE_HZ * 1e3
    d = round(r[6], 1) - round(NETS[ACC_REF][6], 1)  # of the DISPLAYED values
    net_cells.append([
        r[0], r[1], f"{r[2] / 1000:.1f}",
        *[(f"{v / 1e6:.2f}" if v else "--") for v in (r[3], r[4], r[5])],
        f"{p_inf:.2f}", f"{p_avg:.2f}",
        "(ref.)" if i == NET_REF else f"({p_avg_ref / p_avg:.1f}$\\times$)",
        f"{lat * 1e3:.1f}" + ("$^{\\ddagger}$" if r[9] else ("$^{\\S}$" if r[8] >= 30 else "")),
        f"{r[8]}",
        f"{r[6]:.1f}", "(ref.)" if i == ACC_REF else f"({d:+.1f})",
    ])

bench_cells = []
for i, b in enumerate(BENCH):
    coarse = b[7] < 10                              # 1 mW steps: >= +-10 %
    mark = "$^{\\P}$" if coarse else ""
    p32 = p_at32(b)
    rel = ("(ref.)" if i == BENCH_REF else
           "--" if coarse else f"({p32 / p_at32(BENCH[BENCH_REF]):.1f}$\\times$)")
    bench_cells.append([
        b[0], b[1], f"{b[2]}", f"{b[3]:,}", f"{b[4]:,}", f"{b[5]}", f"{b[6]:.2f}",
        f"{b[7]:.1f}{mark}", f"{p32:.1f}{mark}", rel,
    ])

# ---- footnotes (one source for .tex and the previews) ----------------------- #
NET_NOTES = [
    ("", "All three networks have the same topology and 93,952 weights; only layer precision "
         "differs. XNOR: 32-bit packed XNOR-popcount term; ADD: INT8 weight × ±1 activation "
         "(sign-controlled add); MAC: INT8 multiply-accumulate."),
    ("*", "Dynamic compute power only. Per-operation cost comes from Table V: engine power at "
          f"32 terms/cycle and 50 MHz is {P32['xnor']:.1f} mW (XNOR), {P32['add']:.0f} mW (ADD) and "
          f"{P32['mac']:.0f} mW (MAC on DSP48); dividing by the term rate "
          f"(50 MHz × 32 × 64/65 = {LANE_RATE / 1e9:.2f} G terms/s) gives "
          f"{PJ['xnor']:.2f}, {PJ['add']:.1f} and {PJ['mac']:.1f} pJ per term. Network power = "
          "Σ ops × (pJ per term), divided by the latency (during inference) or multiplied by "
          f"{RATE_HZ} Hz (average at the {RATE_HZ}-Hz inference rate). Memory access, control and "
          "static power are excluded; for reference, the whole device draws 0.184 W during "
          "inference of the proposed network, 0.094 W of it static (vectorless)."),
    ("†", "Relative to All INT8, on the 10-Hz average."),
    ("‡", "Measured on chip (600/600 clips bit-exact with the integer reference). Other "
          "latencies are estimated from cycle counts."),
    ("§", "With two DSP48 per engine; with one, 123 ms. Matching the binary latency would need "
          "480 DSP48 (the device has 140)."),
    ("‖", "DSP48 slices for the layers that are binary in the proposed network; the INT8 tail "
          "(Conv3, Conv4) uses 7 in all three."),
    ("", "Accuracy: test split, 12 classes; W8A8 = 8-bit weights and activations (QAT)."),
]
BENCH_NOTES = [
    ("", "One engine per row, placed and routed alone on XC7S75-1 at 50 MHz. Operands come "
         "from on-chip xorshift64 generators (identical in every build); each engine accumulates "
         "64-word vectors, and every accumulator is XOR-reduced to an output pin so that no logic "
         "is pruned. LUT, FF and power are net of a stimulus-only build. Act. × weight: "
         "operand widths."),
    ("*", "Engine dynamic power: report_power (post-route, vectorless default switching) of the "
          "build minus that of the stimulus-only build. Comparable across rows; not a "
          "measurement on silicon."),
    ("†", f"Scaled to the same throughput, 32 terms/cycle (= {LANE_RATE / 1e9:.2f} G terms/s with "
          "one start cycle per 64-word vector). Per-term energy = this power ÷ term rate: "
          + ", ".join(f"{p_at32(b) * 1e-3 / LANE_RATE * 1e12:.3g}" for b in BENCH[1:])
          + " pJ for rows 2–5. Table IV uses rows 2–4."),
    ("¶", "report_power prints 1 mW steps (±17% at 3 mW), so the binary reference is the "
          "8-engine build. Shared control logic makes it 88 LUT per engine against 173 alone; "
          "the real design, with one engine per layer, is the latter."),
]


def tex_note(sym, s):
    s = (s.replace("×", "$\\times$").replace("±", "$\\pm$").replace("Σ", "$\\Sigma$")
          .replace("÷", "$/$").replace("–", "--").replace("%", "\\%").replace(",", "{,}")
          .replace("{,} ", ", ").replace("report_power", "\\texttt{report\\_power}"))
    s = re.sub(r"(\d) (mW|MHz|pJ|W|Hz|ms)\b", r"\1\\,\2", s)     # thin space, numbers only
    mark = {"*": "$^{*}$", "†": "$^{\\dagger}$", "‡": "$^{\\ddagger}$", "§": "$^{\\S}$",
            "‖": "$^{\\|}$", "¶": "$^{\\P}$", "": ""}[sym]
    return mark + s


# ---- LaTeX -------------------------------------------------------------------- #
def write_tex(stem, label, caption, colspec, head, cells, notes):
    tex = (f"% GENERATED by docs/paper/make_table_binarization.py -- do not edit by hand.\n"
           f"% needs \\usepackage{{booktabs}}\n\\begin{{table*}}[t]\n"
           f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n"
           f"\\setlength{{\\tabcolsep}}{{4.2pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
           + head + "\\midrule\n")
    for c in cells:
        tex += " & ".join(c) + " \\\\\n"
    tex += ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n"
            "\\begin{minipage}{\\textwidth}\\footnotesize\n"
            + " \\\\\n".join(tex_note(*n) for n in notes) + "\n\\end{minipage}\n\\end{table*}\n")
    (HERE / f"{stem}.tex").write_text(tex)


write_tex(
    "table_binarization", "tab:binarization",
    "Hardware Cost of Binarization: Ablation of the Proposed Network\\\\\n"
    "on the Same FPGA (Xilinx XC7S75, 50\\,MHz), Google Speech Commands v2",
    "llr rrr rrl r r rl",
    " & & & \\multicolumn{3}{c}{Ops / inference [M]} & \\multicolumn{3}{c}{Compute power$^{*}$ [mW]}"
    " & & & \\multicolumn{2}{c}{Accuracy [\\%]} \\\\\n"
    "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){12-13}\n"
    "Method & Binarized layers & Weights [kbit] & XNOR & ADD & MAC & inference & avg.\\ @10\\,Hz"
    " & (improv.$^{\\dagger}$) & Latency [ms] & DSP$^{\\|}$ & 12 classes & \\\\\n",
    net_cells, NET_NOTES)

write_tex(
    "table_mac_bench", "tab:macbench",
    "Post-Route Cost of One MAC Engine of Each Kind\\\\\n"
    "(Xilinx XC7S75, 50\\,MHz, Vivado, Vectorless)",
    "ll r rrr r rrl",
    " & & & & & & & \\multicolumn{3}{c}{Engine power [mW]} \\\\\n"
    "\\cmidrule(lr){8-10}\n"
    "Engine & Act.\\ $\\times$ weight & Terms/cycle & LUT & FF & DSP & WNS [ns]"
    " & as built$^{*}$ & @32 terms/cyc.$^{\\dagger}$ & (rel.) \\\\\n",
    bench_cells, BENCH_NOTES)

# ---- previews (.svg + .png) ---------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "svg.fonttype": "path"})
SZ_CELL, SZ_HEAD, SZ_NOTE = 9.0, 8.6, 6.8


def plain(s):
    return (s.replace("$\\rightarrow$", "→").replace("$\\times$", "×").replace("$\\times\\,\\pm1$", "× ±1")
             .replace("$^{\\dagger}$", "†").replace("$^{\\ddagger}$", "‡").replace("$^{\\S}$", "§")
             .replace("$^{\\P}$", "¶").replace("$^{\\|}$", "‖").replace("$^{*}$", "*")
             .replace("\\,", " ").replace("\\%", "%").replace("--", "–").replace("\\ ", " "))


def render(stem, number, title, heads, groups, cells, align, tight, notes):
    """heads: column headers; groups: (first_col, last_col, label);
    tight: columns that hug the one before them (the '(x)' columns)."""
    fig = plt.figure(dpi=220)
    rnd = fig.canvas.get_renderer()

    def width(s, size):
        t = fig.text(0, 0, s, size=size)
        w = t.get_window_extent(rnd).width / fig.dpi
        t.remove()
        return w

    rows = [[plain(c) for c in r] for r in cells]
    heads = [plain(h) for h in heads]
    n = len(heads)
    colw = [max([width(heads[j], SZ_HEAD)] + [width(r[j], SZ_CELL) for r in rows]) for j in range(n)]
    gap = [0.06 if j in tight else (0.0 if j == 0 else 0.2) for j in range(n)]
    for a, b, lab in groups:                     # a group label must fit over its columns
        span = sum(colw[a:b + 1]) + sum(gap[a + 1:b + 1])
        need = width(lab, SZ_HEAD) + 0.1
        if need > span:
            colw[b] += need - span
    M = 0.12
    W = M * 2 + sum(colw) + sum(gap)
    x0 = []
    x = M
    for j in range(n):
        x += gap[j]
        x0.append(x)
        x += colw[j]

    note_chars = int(W / 0.0445)                  # STIX at SZ_NOTE, ~0.0445 in per char
    note_lines = []
    for sym, s in notes:
        note_lines += textwrap.wrap((sym + s) if sym else s, note_chars)
    title_lines = textwrap.wrap(title.upper(), int(W / 0.07))
    H = (0.30 + 0.20 * len(title_lines) + 0.12 + (0.30 if groups else 0) + 0.26
         + 0.25 * len(rows) + 0.14 + 0.135 * len(note_lines) + 0.08)
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def t(xx, yy, s, ha="left", size=SZ_CELL):
        ax.text(xx, yy, s, ha=ha, va="center", size=size)

    def cx(j):
        return x0[j] + (colw[j] if align[j] == "right" else 0)

    y = H - 0.16
    t(W / 2, y, f"TABLE {number}", "center", 9.5)
    for line in title_lines:
        y -= 0.20
        t(W / 2, y, line, "center", 7.6)
    y -= 0.22
    ax.plot([M - 0.04, W - M + 0.04], [y, y], color="k", lw=1.1)
    if groups:
        y -= 0.17
        for a, b, lab in groups:
            t((x0[a] + x0[b] + colw[b]) / 2, y, lab, "center", SZ_HEAD)
            ax.plot([x0[a], x0[b] + colw[b]], [y - 0.11, y - 0.11], color="k", lw=0.5)
        y -= 0.13
    y -= 0.13
    for j, h in enumerate(heads):
        t(cx(j), y, h, align[j], SZ_HEAD)
    y -= 0.15
    ax.plot([M - 0.04, W - M + 0.04], [y, y], color="k", lw=0.6)
    y -= 0.10
    for r in rows:
        y -= 0.125
        for j, s in enumerate(r):
            t(cx(j), y, s, align[j])
        y -= 0.125
    y -= 0.05
    ax.plot([M - 0.04, W - M + 0.04], [y, y], color="k", lw=1.1)
    ax.text(M, y - 0.06, "\n".join(note_lines), ha="left", va="top", size=SZ_NOTE, linespacing=1.38)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


render("table_binarization", "IV",
       "Hardware cost of binarization: ablation of the proposed network on the same FPGA "
       "(Xilinx XC7S75, 50 MHz), Google Speech Commands v2",
       ["Method", "Binarized layers", "Weights [kbit]", "XNOR", "ADD", "MAC",
        "inference", "avg. @10 Hz", "(improv.†)", "Latency [ms]", "DSP‖", "12 classes", ""],
       [(3, 5, "Ops / inference [M]"), (6, 8, "Compute power* [mW]"), (11, 12, "Accuracy [%]")],
       net_cells,
       ["left", "left", "right", "right", "right", "right", "right", "right", "left",
        "right", "right", "right", "left"],
       {8, 12}, NET_NOTES)

render("table_mac_bench", "V",
       "Post-route cost of one MAC engine of each kind (Xilinx XC7S75, 50 MHz, Vivado, vectorless)",
       ["Engine", "Act. × weight", "Terms/cycle", "LUT", "FF", "DSP", "WNS [ns]",
        "as built*", "@32 terms/cyc.†", "(rel.)"],
       [(7, 9, "Engine power [mW]")],
       bench_cells,
       ["left", "left", "right", "right", "right", "right", "right", "right", "right", "left"],
       {9}, BENCH_NOTES)

print("wrote table_binarization.{tex,svg,png} and table_mac_bench.{tex,svg,png} in", HERE)
for c in net_cells:
    print(" ", plain(c[0]), "| inference", c[6], "mW | 10 Hz", c[7], "mW", c[8], "|", c[11], c[12])
for c in bench_cells:
    print(" ", plain(c[0]), "|", c[7], "mW as built |", c[8], "mW @32 |", plain(c[9]))

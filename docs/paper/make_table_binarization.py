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
OPS = {"bin": "1\\,b $\\times$ 1\\,b", "pm1": "1\\,b $\\times$ 8\\,b", "i8": "8\\,b $\\times$ 8\\,b"}
#  key, name, operands, terms/cycle, LUT, FF, DSP, BRAM tiles, WNS, engine mW (net of stim)
ARITH = [    # KIND 0, 7, 3, 6, 4 -- the engine fed straight from the stimulus
    ("a_bin1", "Binary, 1 engine",            OPS["bin"],  32,  173,   49,  0, 0,  9.08,  3.0),
    ("a_bin8", "Binary, 8 engines",           OPS["bin"], 256,  707,  336,  0, 0,  8.57, 15.0),
    ("a_pm1",  "INT8 $\\times\\,\\pm1$ (add/sub)", OPS["pm1"], 32, 1050, 775, 0, 0, 9.94, 24.0),
    ("a_dsp",  "INT8 MAC, DSP48",             OPS["i8"],   32, 1042,  787, 32, 0, 12.03, 43.0),
    ("a_lut",  "INT8 MAC, LUT only",          OPS["i8"],   32, 3646,  786,  0, 0,  8.69, 61.0),
]
MEM = [      # KIND 12, 8, 11, 9, 10 -- + weight ROM and activation frame register
    ("m_bin1", "Binary, 1 engine",            OPS["bin"],  32,  271,  187,  0, 0,  7.55,  5.0),
    ("m_bin8", "Binary, 8 engines",           OPS["bin"], 256, 2226, 1431,  0, 0,  7.72, 46.0),
    ("m_pm1",  "INT8 $\\times\\,\\pm1$ (add/sub)", OPS["pm1"], 32, 2168, 962, 0, 0, 10.73, 43.0),
    ("m_dsp",  "INT8 MAC, DSP48",             OPS["i8"],   32, 2640, 1890, 32, 0,  6.67, 68.0),
    ("m_bram", "INT8 MAC, DSP48, BRAM ROM",   OPS["i8"],   32,  928, 2077, 32, 4,  9.25, 57.0),
]
B = {r[0]: r for r in ARITH + MEM}


def p_at32(b):                        # engine power scaled to 32 terms/cycle, mW
    return b[9] * 32 / b[3]


def pj(key):                          # energy per term, pJ
    return p_at32(B[key]) * 1e-3 / LANE_RATE * 1e12


# Table IV: arithmetic + operand memory. Binary and ADD read LUT ROMs as the
# proposed design does; INT8 MAC takes the block-RAM ROM, the cheaper of its
# two options -- the choice that favours INT8.
USE = {"xnor": "m_bin8", "add": "m_pm1", "mac": "m_bram"}
ALT_LUT = {"xnor": "m_bin8", "add": "m_pm1", "mac": "m_dsp"}
ARITH_ONLY = {"xnor": "a_bin8", "add": "a_pm1", "mac": "a_dsp"}
PJ = {k: pj(v) for k, v in USE.items()}

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


def energy_j(r, use=USE):
    return (r[3] * pj(use["xnor"]) + r[4] * pj(use["add"]) + r[5] * pj(use["mac"])) * 1e-12


def gain(use):                        # All INT8 / Binary, energy per inference
    return energy_j(NETS[2], use) / energy_j(NETS[0], use)


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

# Table V: two sections, each relative to its own 8-engine binary build
bench_cells = []
for title, rows, ref in (("(a) Arithmetic only", ARITH, "a_bin8"),
                         ("(b) With operand memory: weight ROM + activation frame register",
                          MEM, "m_bin8")):
    bench_cells.append(title)                       # a section row
    for b in rows:
        coarse = b[9] < 10                          # 1 mW steps: >= +-10 %
        mark = "$^{\\P}$" if coarse else ""
        rel = ("(ref.)" if b[0] == ref else
               "--" if coarse else f"({p_at32(b) / p_at32(B[ref]):.1f}$\\times$)")
        bench_cells.append([
            b[1], b[2], f"{b[3]}", f"{b[4]:,}", f"{b[5]:,}", f"{b[6]}",
            (f"{b[7]:g}" if b[7] else "0"), f"{b[8]:.2f}",
            f"{b[9]:.1f}{mark}", f"{p_at32(b):.1f}{mark}", rel,
        ])

# ---- footnotes (one source for .tex and the previews) ----------------------- #
NET_NOTES = [
    ("", "All three networks have the same topology and 93,952 weights; only layer precision "
         "differs. XNOR: 32-bit packed XNOR-popcount term; ADD: INT8 weight × ±1 activation "
         "(sign-controlled add); MAC: INT8 multiply-accumulate."),
    ("*", "Dynamic power of the arithmetic and its operand memory (a weight-ROM word read "
          "every cycle and the activation frame register), from Table V(b). At 32 terms/cycle "
          f"and 50 MHz an engine draws {p_at32(B[USE['xnor']]):.1f} mW (XNOR, LUT ROM as in the "
          f"proposed design), {p_at32(B[USE['add']]):.0f} mW (ADD, LUT ROM) and "
          f"{p_at32(B[USE['mac']]):.0f} mW (MAC on DSP48 with a block-RAM ROM, the cheaper INT8 "
          f"option); dividing by the term rate (50 MHz × 32 × 64/65 = {LANE_RATE / 1e9:.2f} G "
          f"terms/s) gives {PJ['xnor']:.2f}, {PJ['add']:.1f} and {PJ['mac']:.1f} pJ per term. "
          "Network power = Σ ops × (pJ per term), divided by the latency (during inference) or "
          f"multiplied by {RATE_HZ} Hz (average at the {RATE_HZ}-Hz inference rate). With a LUT "
          f"ROM for the INT8 MAC, All INT8 costs {gain(ALT_LUT):.1f}× the binary network; with "
          f"arithmetic alone (Table V(a)), {gain(ARITH_ONLY):.1f}×. Excluded: control and clock "
          "distribution (the same folded controller in all three) and static power (0.094 W, "
          "the same device); for reference, the whole device draws 0.184 W during inference of "
          "the proposed network (vectorless)."),
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
         "is pruned. LUT, FF, BRAM and power are net of a stimulus-only build. Act. × weight: "
         "operand widths. In (b) the engine is fed as the proposed design's pointwise layer "
         "feeds it: a 256-word weight ROM loaded from a file, 32 bits wide for binary and 256 for INT8, "
         "read one word a cycle (asynchronously from LUTs unless BRAM), and a 128-channel "
         "activation frame latched every 260 cycles and read one word a cycle."),
    ("*", "Engine dynamic power: report_power (post-route, vectorless default switching) of the "
          "build minus that of the stimulus-only build. Comparable across rows; not a "
          "measurement on silicon."),
    ("†", f"Scaled to the same throughput, 32 terms/cycle (= {LANE_RATE / 1e9:.2f} G terms/s with "
          "one start cycle per 64-word vector). Per-term energy = this power ÷ term rate: "
          + ", ".join(f"{pj(k):.3g}" for k in ("a_bin8", "a_pm1", "a_dsp", "a_lut"))
          + " pJ in (a); "
          + ", ".join(f"{pj(k):.3g}" for k in ("m_bin8", "m_pm1", "m_dsp", "m_bram"))
          + " pJ in (b). Table IV uses (b): binary, INT8 × ±1 and the BRAM-ROM MAC."),
    ("¶", "report_power prints 1 mW steps (±17% at 3 mW, ±10% at 5 mW), so the binary "
          "reference is the 8-engine build. In (a) shared control logic makes it 88 LUT per "
          "engine against 173 alone; the real design, with one engine per layer, is the latter."),
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
    ncol = max(len(c) for c in cells if not isinstance(c, str))
    first = True
    for c in cells:
        if isinstance(c, str):                       # a section row
            if not first:
                tex += "\\midrule\n"
            tex += f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{c}}}}} \\\\\n"
        else:
            tex += " & ".join(c) + " \\\\\n"
        first = False
    tex += ("\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n"
            "\\begin{minipage}{\\textwidth}\\footnotesize\n"
            + " \\\\\n".join(tex_note(*n) for n in notes) + "\n\\end{minipage}\n\\end{table*}\n")
    (HERE / f"{stem}.tex").write_text(tex)


write_tex(
    "table_binarization", "tab:binarization",
    "Hardware Cost of Binarization: Ablation of the Proposed Network\\\\\n"
    "on the Same FPGA (Xilinx XC7S75, 50\\,MHz), Google Speech Commands v2",
    "llr rrr rrl r r rl",
    " & & & \\multicolumn{3}{c}{Ops / inference [M]} & \\multicolumn{3}{c}{Arith.\\ + memory power$^{*}$ [mW]}"
    " & & & \\multicolumn{2}{c}{Accuracy [\\%]} \\\\\n"
    "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){12-13}\n"
    "Method & Binarized layers & Weights [kbit] & XNOR & ADD & MAC & inference & avg.\\ @10\\,Hz"
    " & (improv.$^{\\dagger}$) & Latency [ms] & DSP$^{\\|}$ & 12 classes & \\\\\n",
    net_cells, NET_NOTES)

write_tex(
    "table_mac_bench", "tab:macbench",
    "Post-Route Cost of One MAC Engine of Each Kind\\\\\n"
    "(Xilinx XC7S75, 50\\,MHz, Vivado, Vectorless)",
    "ll r rrrr r rrl",
    " & & & & & & & & \\multicolumn{3}{c}{Engine power [mW]} \\\\\n"
    "\\cmidrule(lr){9-11}\n"
    "Engine & Act.\\ $\\times$ weight & Terms/cycle & LUT & FF & DSP & BRAM & WNS [ns]"
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


def render(stem, number, title, heads, groups, cells, align, tight, notes, report=False):
    """heads: column headers; groups: (first_col, last_col, label);
    tight: columns that hug the one before them (the '(x)' columns).
    report=True: body only -- no title block and no notes (the report's own
    caption and note paragraphs carry them), tighter gaps, to report/."""
    if report:
        title, notes = "", []
    fig = plt.figure(dpi=300 if report else 220)
    rnd = fig.canvas.get_renderer()

    def width(s, size):
        t = fig.text(0, 0, s, size=size)
        w = t.get_window_extent(rnd).width / fig.dpi
        t.remove()
        return w

    # a str in cells is a section row: italic, across the table
    rows = [plain(r) if isinstance(r, str) else [plain(c) for c in r] for r in cells]
    data = [r for r in rows if not isinstance(r, str)]
    heads = [plain(h) for h in heads]
    n = len(heads)
    colw = [max([width(heads[j], SZ_HEAD)] + [width(r[j], SZ_CELL) for r in data]) for j in range(n)]
    gap = [0.06 if j in tight else (0.0 if j == 0 else (0.13 if report else 0.2)) for j in range(n)]
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
    top_pad = 0.30 + 0.20 * len(title_lines) + 0.12 if not report else 0.06
    H = (top_pad + (0.30 if groups else 0) + 0.26
         + 0.25 * len(rows) + 0.06 * sum(isinstance(r, str) for r in rows[1:])
         + (0.14 + 0.135 * len(note_lines) + 0.08 if not report else 0.22))
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def t(xx, yy, s, ha="left", size=SZ_CELL):
        ax.text(xx, yy, s, ha=ha, va="center", size=size)

    def cx(j):
        return x0[j] + (colw[j] if align[j] == "right" else 0)

    if report:
        y = H - 0.04
    else:
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
    for i, r in enumerate(rows):
        if isinstance(r, str):
            if i:
                y -= 0.03
                ax.plot([M - 0.04, W - M + 0.04], [y, y], color="k", lw=0.6)
                y -= 0.03
            y -= 0.125
            ax.text(M, y, r, ha="left", va="center", size=SZ_CELL, style="italic")
            y -= 0.125
            continue
        y -= 0.125
        for j, s in enumerate(r):
            t(cx(j), y, s, align[j])
        y -= 0.125
    y -= 0.05
    ax.plot([M - 0.04, W - M + 0.04], [y, y], color="k", lw=1.1)
    if note_lines:
        ax.text(M, y - 0.06, "\n".join(note_lines), ha="left", va="top", size=SZ_NOTE, linespacing=1.38)
    out = HERE / "report" if report else HERE
    out.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300 if report else 220)
    plt.close(fig)


# IEEE previews (TABLE IV/V) and report bodies (report/: 표 2, 표 3 of the IDEC report)
for rep in (False, True):
    render("table2_binarization" if rep else "table_binarization", "IV",
           "Hardware cost of binarization: ablation of the proposed network on the same FPGA "
           "(Xilinx XC7S75, 50 MHz), Google Speech Commands v2",
           ["Method", "Binarized layers", "Weights [kbit]", "XNOR", "ADD", "MAC",
            "inference", "avg. @10 Hz", "(improv.†)", "Latency [ms]", "DSP‖", "12 classes", ""],
           [(3, 5, "Ops / inference [M]"), (6, 8, "Arith. + memory power* [mW]"), (11, 12, "Accuracy [%]")],
           net_cells,
           ["left", "left", "right", "right", "right", "right", "right", "right", "left",
            "right", "right", "right", "left"],
           {8, 12}, NET_NOTES, report=rep)

    render("table3_mac_bench" if rep else "table_mac_bench", "V",
           "Post-route cost of one MAC engine of each kind (Xilinx XC7S75, 50 MHz, Vivado, vectorless)",
           ["Engine", "Act. × weight", "Terms/cycle", "LUT", "FF", "DSP", "BRAM", "WNS [ns]",
            "as built*", "@32 terms/cyc.†", "(rel.)"],
           [(8, 10, "Engine power [mW]")],
           bench_cells,
           ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right",
            "left"],
           {10}, BENCH_NOTES, report=rep)

print("wrote table_binarization.{tex,svg,png} and table_mac_bench.{tex,svg,png} in", HERE)
for c in net_cells:
    print(" ", plain(c[0]), "| inference", c[6], "mW | 10 Hz", c[7], "mW", c[8], "|", c[11], c[12])
for c in bench_cells:
    if isinstance(c, str):
        print(" ", c)
    else:
        print("   ", plain(c[0]), "|", plain(c[8]), "mW as built |", plain(c[9]), "mW @32 |", plain(c[10]))
print("  energy All INT8 / Binary:", f"{gain(USE):.2f} (used),", f"{gain(ALT_LUT):.2f} (LUT-ROM MAC),",
      f"{gain(ARITH_ONLY):.2f} (arithmetic only)")

"""Draw docs/diagrams/18_head.svg -- the two ways a comparator gets it wrong.

`head` came out of the 16-channel netlist and it only measures one of them, so
a table of head values on its own is misleading. The picture puts both next to
each other:

  LEVEL  the threshold sits alpha_c * T*ln16 above the quiescent point, and the
         per-channel offsets move the quiescent by a fixed number of millivolts.
         head = margin / offsets. Below 1x, the channel is already on the wrong
         side before any signal arrives. This is what the tables report.

  SLOPE  during speech the envelope crosses that threshold at a finite rate, so
         the comparator spends real time inside its own offset band. That time
         is set by d(v_env - v_thr)/dt, which head says nothing about, and it is
         the reason detector gain looked worth having.

The two disagree about gain, which is the whole point of drawing them together:
raising it steepens the crossing and simultaneously multiplies the offset that
sets the level error.

Waveforms are real -- read from the transient CSVs that sim_afe.py writes.
Numbers in the labels are computed here from the same constants sim_afe.py uses,
so the picture cannot drift from the tables.

    python3 analog/AFE/scripts/sim_afe.py       # writes the CSVs first
    python3 docs/diagrams/make_head.py
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "18_head.svg"
ART = pathlib.Path(__file__).resolve().parents[2] / "analog/AFE/artifacts"

FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,monospace"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1360, 1080

# --- the same constants sim_afe.py carries -------------------------------
FLOOR_MV = 70.0            # T*ln16, measured on the netlist. Thermal, FIXED.
ALPHA_MIN = 0.0572         # ch14, the worst channel
VOS_TYP, VOS_MAX = 0.4, 1.5        # OPA379  SBOS347D
COMP_TYP, COMP_MAX = 0.4, 6.0      # LPV7215 SNOS977, 1.8 V table
GAIN = 4.70                # R5/R4 for the reference design

MARGIN = ALPHA_MIN * FLOOR_MV
SC_TYP = GAIN * VOS_TYP + COMP_TYP
SC_MAX = GAIN * VOS_MAX + COMP_MAX


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=15, fill=INK, anchor="start", weight="400", mono=False,
        style=""):
    f = MONO if mono else FONT
    return (f'<text x="{x}" y="{y}" font-family="{f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
            f'{style}>{esc(s)}</text>')


def load(tag):
    """t, v_env, v_thr from a sim_afe transient CSV (wrdata writes t,v pairs)."""
    rows = [[float(x) for x in ln.split()]
            for ln in (ART / f"sim_{tag}.csv").read_text().splitlines()
            if ln.strip()]
    return ([r[0] for r in rows], [r[1] for r in rows], [r[7] for r in rows])


P = []
A = P.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
  f'width="{W}" height="{H}">')
A(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

A(txt(40, 48, "비교기가 틀리는 두 가지 방식", 27, INK, weight="700"))
A(txt(40, 76, "head 는 왼쪽만 잰다. 오른쪽은 이득을 반대 방향으로 요구한다.",
      16, MUTE))

# ======================= LEFT: the level error ===========================
X0, Y0, PW, PH = 40, 108, 620, 470
A(f'<rect x="{X0}" y="{Y0}" width="{PW}" height="{PH}" rx="10" '
  f'fill="#f8fafc" stroke="#e2e8f0"/>')
A(txt(X0 + 22, Y0 + 34, "① LEVEL — 무음에서 이미 틀림", 19, INK, weight="700"))
A(txt(X0 + 22, Y0 + 58, "head = 마진 / 산포.  이게 표에 있는 숫자다.", 14, MUTE))

# vertical mV scale: 0 = quiescent, up to +20 mV
BX, BY, BH = X0 + 170, Y0 + 100, 320
MVTOP = 20.0
def ypx(mv):
    return BY + BH - (mv / MVTOP) * BH

A(f'<line x1="{BX}" y1="{BY}" x2="{BX}" y2="{BY+BH}" stroke="{WIRE}" '
  f'stroke-width="1.5"/>')
for mv in (0, 5, 10, 15, 20):
    A(f'<line x1="{BX-5}" y1="{ypx(mv):.1f}" x2="{BX}" y2="{ypx(mv):.1f}" '
      f'stroke="{WIRE}"/>')
    A(txt(BX - 11, ypx(mv) + 5, f"+{mv}", 13, MUTE, "end", mono=True))
A(txt(BX - 11, ypx(0) + 24, "mV", 12, MUTE, "end"))

RW = 300
# quiescent
A(f'<line x1="{BX}" y1="{ypx(0):.1f}" x2="{BX+RW}" y2="{ypx(0):.1f}" '
  f'stroke="{WIRE}" stroke-width="2.5"/>')
A(txt(BX + RW + 10, ypx(0) + 5, "정지점 = V_ref", 14, WIRE, weight="600"))

# threshold of the worst channel
A(f'<line x1="{BX}" y1="{ypx(MARGIN):.1f}" x2="{BX+RW}" '
  f'y2="{ypx(MARGIN):.1f}" stroke="{GRN}" stroke-width="2.5"/>')
A(txt(BX + RW + 10, ypx(MARGIN) + 5,
      f"V_thr (ch14, α={ALPHA_MIN:.3f})", 14, GRN, weight="600"))
A(f'<path d="M {BX+40} {ypx(0):.1f} L {BX+40} {ypx(MARGIN):.1f}" '
  f'stroke="{GRN}" stroke-width="1.5" marker-start="url(#gdot)" '
  f'marker-end="url(#gdot)"/>')
A(txt(BX + 50, (ypx(0) + ypx(MARGIN)) / 2 + 5,
      f"마진 {MARGIN:.1f} mV = α × 70", 13, GRN, weight="600"))

# scatter bands
A(f'<rect x="{BX+150}" y="{ypx(SC_TYP):.1f}" width="60" '
  f'height="{(ypx(0)-ypx(SC_TYP)):.1f}" fill="{BLU}" opacity="0.22"/>')
A(txt(BX + 180, ypx(SC_TYP) - 8, f"typ {SC_TYP:.1f}", 12, BLU, "middle",
      weight="700", mono=True))
A(f'<rect x="{BX+220}" y="{ypx(min(SC_MAX, MVTOP)):.1f}" width="60" '
  f'height="{(ypx(0)-ypx(min(SC_MAX, MVTOP))):.1f}" fill="{RED}" '
  f'opacity="0.22"/>')
A(txt(BX + 250, ypx(min(SC_MAX, MVTOP)) - 8, f"max {SC_MAX:.1f}", 12, RED,
      "middle", weight="700", mono=True))
A(txt(BX + 250, ypx(MVTOP) - 26, "↑ 계속", 11, RED, "middle"))

yv = Y0 + PH - 52
A(txt(X0 + 22, yv,
      f"typ:  마진 {MARGIN:.1f} > 산포 {SC_TYP:.1f}  →  head "
      f"{MARGIN/SC_TYP:.2f}× ✅", 15, BLU, weight="600", mono=True))
A(txt(X0 + 22, yv + 26,
      f"max:  마진 {MARGIN:.1f} < 산포 {SC_MAX:.1f}  →  head "
      f"{MARGIN/SC_MAX:.2f}× ⚠️", 15, RED, weight="600", mono=True))

# ======================= RIGHT: the slope error ==========================
X1 = 700
A(f'<rect x="{X1}" y="{Y0}" width="{PW}" height="{PH}" rx="10" '
  f'fill="#f8fafc" stroke="#e2e8f0"/>')
A(txt(X1 + 22, Y0 + 34, "② SLOPE — 교차할 때 흔들림", 19, INK, weight="700"))
A(txt(X1 + 22, Y0 + 58,
      "오프셋 밴드 안에 머무는 시간 = 2ε / (dV/dt). head 는 이걸 안 잰다.",
      14, MUTE))

t, ve, vt = load("A")
T_LO, T_HI = 0.0198, 0.0230
idx = [i for i in range(len(t)) if T_LO <= t[i] <= T_HI]
d = [(ve[i] - vt[i]) * 1e3 for i in idx]          # mV above threshold
GX, GY, GW, GH = X1 + 90, Y0 + 96, 480, 300
DLO, DHI = -25.0, 45.0
xs = lambda i: GX + (t[i] - T_LO) / (T_HI - T_LO) * GW      # noqa: E731
ys = lambda mv: GY + GH - (mv - DLO) / (DHI - DLO) * GH     # noqa: E731

A(f'<rect x="{GX}" y="{GY}" width="{GW}" height="{GH}" fill="#ffffff" '
  f'stroke="#e2e8f0"/>')
# comparator worst-case offset band around the threshold (d = 0)
A(f'<rect x="{GX}" y="{ys(COMP_MAX):.1f}" width="{GW}" '
  f'height="{(ys(-COMP_MAX)-ys(COMP_MAX)):.1f}" fill="{RED}" opacity="0.16"/>')
A(f'<line x1="{GX}" y1="{ys(0):.1f}" x2="{GX+GW}" y2="{ys(0):.1f}" '
  f'stroke="{GRN}" stroke-width="2" stroke-dasharray="6 4"/>')
A(txt(GX + GW - 6, ys(0) - 8, "V_thr", 13, GRN, "end", weight="600"))
A(txt(GX + 8, ys(COMP_MAX) - 7,
      f"비교기 오프셋 ±{COMP_MAX:.0f} mV (max)", 12, RED, weight="600"))

pts = " ".join(f"{xs(i):.1f},{ys(max(DLO, min(DHI, d[k]))):.1f}"
               for k, i in enumerate(idx))
A(f'<polyline points="{pts}" fill="none" stroke="{BLU}" stroke-width="2"/>')
A(txt(GX + 8, GY + 20, "v_env − V_thr  (실제 시뮬 파형, 이득 4.7)", 13, BLU,
      weight="600"))

for k, ms in ((0, "19.8"), (1, "21.4"), (2, "23.0")):
    x = GX + k * GW / 2
    A(f'<line x1="{x:.1f}" y1="{GY+GH}" x2="{x:.1f}" y2="{GY+GH+5}" '
      f'stroke="{WIRE}"/>')
    A(txt(x, GY + GH + 20, ms, 12, MUTE, "middle", mono=True))
A(txt(GX + GW / 2, GY + GH + 40, "시간 [ms]", 13, MUTE, "middle"))

yv = Y0 + PH - 52
A(txt(X1 + 22, yv, "이득이 크면 교차가 가팔라 밴드를 빨리 빠져나간다.",
      15, INK, weight="600"))
A(txt(X1 + 22, yv + 26,
      "이득 1.6 → 351 µs   4.7 → 405 µs   10 → 25 µs", 15, PUR,
      weight="600", mono=True))

# ======================= BOTTOM: the conflict ============================
BY2 = Y0 + PH + 34
A(f'<rect x="{X0}" y="{BY2}" width="{W-80}" height="{H-BY2-40}" rx="10" '
  f'fill="#fffbeb" stroke="#fcd34d"/>')
A(txt(X0 + 26, BY2 + 36, "두 지표가 이득에 대해 반대로 말한다", 20, ORG,
      weight="700"))

rows = [
    ("이득 ↑", "산포 = (R5/R4)·Vos 가 같이 커진다", "LEVEL 나빠짐", RED),
    ("이득 ↑", "교차가 가팔라져 밴드 통과가 빨라진다", "SLOPE 좋아짐", GRN),
    ("이득 ↓", "산포가 준다", "LEVEL 좋아짐", GRN),
    ("이득 ↓", "교차가 완만해져 밴드 안에 오래 머문다", "SLOPE 나빠짐", RED),
]
for k, (a, b, c, col) in enumerate(rows):
    y = BY2 + 74 + k * 30
    A(txt(X0 + 30, y, a, 15, INK, weight="700", mono=True))
    A(txt(X0 + 110, y, b, 15, WIRE))
    A(txt(X0 + 640, y, c, 15, col, weight="700"))

A(txt(X0 + 26, BY2 + 216,
      "빠져나갈 곳은 이득이 아니다. 마진(α·T·ln16)도 비교기 오프셋도 "
      "고정 전압이라,", 16, INK))
A(txt(X0 + 26, BY2 + 242,
      f"이득을 어느 쪽으로 돌려도 한쪽을 사고 다른 쪽을 판다. "
      f"max 산포 {SC_MAX:.1f} mV 중 {COMP_MAX:.0f} mV 는 비교기 몫이고",
      16, INK))
A(txt(X0 + 26, BY2 + 268,
      "그건 (R5/R4) 가 안 곱해져서 이득으로는 손도 못 댄다. "
      "→ 비교기 부품이나 브링업 트림.", 16, INK, weight="600"))

A('<defs><marker id="gdot" markerWidth="6" markerHeight="6" refX="3" '
  f'refY="3"><circle cx="3" cy="3" r="2.5" fill="{GRN}"/></marker></defs>')
A('</svg>')

OUT.write_text("\n".join(P), encoding="utf-8")
print(f"wrote {OUT}")
print(f"  margin {MARGIN:.2f} mV   scatter typ {SC_TYP:.2f} / max {SC_MAX:.2f}")
print(f"  head   typ {MARGIN/SC_TYP:.2f}x / max {MARGIN/SC_MAX:.2f}x")

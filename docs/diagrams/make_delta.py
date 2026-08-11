"""Draw docs/diagrams/15_delta.svg -- what delta is, and why it would not hold still.

delta is the one number in the threshold circuit that the analog side has to
build and that we estimate from data. It swung 2.4x between two seeds, which
made it unusable as a spec. The picture is the fastest way to see both what it
means and why the estimator was never going to be stable: it is targeting a
value near zero, and the compression guard puts a mass point exactly there.

Scale numbers come from one anchor: lse_temp is physically n*V_T = 25.85 mV, so
the run's lse_temp of 0.58329 code units fixes 1 code unit = 44.3 mV. The check
that this is the right anchor is that it sends TYP (0.71955) to 31.9 mV, which
is the 33 mV the trained frac of 0.78 asks for.
"""
import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "15_delta.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,monospace"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1340, 1150

VT = 25.852
T_CODE = 0.58329                 # af_lse078 lse_temp
MV = VT / T_CODE                 # 44.3 mV per code unit
TYP = 0.71955 * MV               # 31.9 mV, typical frame rise
GUARD = 0.00100 * MV             # 0.044 mV, sqrt(1e-6)
D1, D2 = 0.00100 * MV, 0.00244 * MV
QUIESCENT = 917.8


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=12, fill=INK, anchor="middle", weight=None, mono=False):
    f = f' font-family="{MONO}"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{w}{f}>{esc(s)}</text>')


def box(x, y, w, h, fill="#ffffff", stroke=WIRE, rx=6, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, c=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
            f'stroke-width="{sw}"{d}/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{WIRE}"/></marker>'
     '</defs>',
     t(W / 2, 36, "δ — comparator 의 기준선이 정지점보다 얼마나 위인가", 23, INK,
       weight="700"),
     t(W / 2, 60, "저항 두 개로 만드는 값이고, 동료에게 넘길 스펙이다. "
                  "그런데 seed 를 바꾸면 2.4배 흔들렸다.", 13, MUTE)]

# ── A. the circuit picture ─────────────────────────────────────────────────
ay = 86
s += [box(40, ay, 620, 330, "#f8fafc", "#cbd5e1"),
      t(60, ay + 28, "A. 회로에서 — detector 출력의 좌표계", 15, INK, "start",
        weight="700")]

BASE, TOP = ay + 290, ay + 90            # y of quiescent, y of typical peak
bx = 150
s += [line(bx - 40, BASE, 620, BASE, WIRE, 2),
      t(bx - 48, BASE + 5, f"{QUIESCENT} mV", 11, INK, "end", weight="700"),
      t(bx - 48, BASE + 21, "quiescent", 10, MUTE, "end"),
      t(bx - 48, BASE + 35, "(무신호)", 9.5, MUTE, "end")]

# envelope rise
s += [f'<path d="M {bx} {BASE} C {bx+40} {BASE} {bx+55} {TOP} {bx+95} {TOP} '
      f'C {bx+150} {TOP} {bx+165} {BASE-30} {bx+230} {BASE-18} '
      f'L {bx+330} {BASE-10} L {bx+330} {BASE} Z" fill="{BLU}" opacity="0.18"/>',
      f'<path d="M {bx} {BASE} C {bx+40} {BASE} {bx+55} {TOP} {bx+95} {TOP} '
      f'C {bx+150} {TOP} {bx+165} {BASE-30} {bx+230} {BASE-18} '
      f'L {bx+330} {BASE-10}" fill="none" stroke="{BLU}" stroke-width="2"/>',
      t(bx + 130, TOP - 12, "envelope (V_env,c)", 11.5, BLU, weight="700"),
      line(bx + 95, TOP, bx + 95, BASE, BLU, 1.2, "4 3"),
      t(bx + 105, (TOP + BASE) / 2, f"전형 상승 {TYP:.0f} mV", 11, BLU, "start")]

# V_ref, just above quiescent -- deliberately drawn at true scale (invisible)
VREF_Y = BASE - 2
s += [line(bx - 20, VREF_Y, 540, VREF_Y, RED, 2),
      t(548, VREF_Y - 4, "V_ref", 11.5, RED, "start", weight="700"),
      t(548, VREF_Y + 11, "= 정지점 + δ", 9.5, RED, "start")]
s.append(t(350, ay + 322, f"δ = {D2:.3f} mV 는 이 그림에서 선 두께보다 얇다 "
                          f"— 그게 결론의 절반이다", 12, RED, weight="700"))

# zoom inset
s += [box(430, ay + 46, 210, 132, "#ffffff", RED, sw=1.2, dash="4 3"),
      t(535, ay + 66, "×300 확대", 10.5, RED, weight="700")]
zb = ay + 148
s += [line(450, zb, 620, zb, WIRE, 2),
      t(444, zb + 4, "0", 9.5, MUTE, "end"),
      line(450, zb - 26, 620, zb - 26, RED, 2),
      t(444, zb - 22, "δ", 11, RED, "end", weight="700"),
      t(535, zb - 32, f"{D2:.3f} mV", 9.5, RED),
      t(535, zb + 18, "= 전형 상승의 0.34%", 9.5, MUTE)]

# ── B. the algebra ─────────────────────────────────────────────────────────
s += [box(680, ay, 620, 330, "#eff6ff", "#93c5fd"),
      t(700, ay + 28, "B. 수식에서 — 왜 나눗셈에 나타나나", 15, BLU, "start",
        weight="700"),
      t(700, ay + 50, "저항 divider 가 만들 수 있는 건 linear mix 뿐이다.",
        11.5, MUTE, "start")]
eqs = [("V_thr,c = α_c · V_max + (1−α_c) · V_ref", 13, INK,
        "divider 가 실제로 만드는 것"),
       ("V_env,c > V_thr,c", 13, INK, "comparator 발화 조건"),
       ("모든 항에서 quiescent 를 뺀다", 11.5, MUTE, "→ '정지점 위 상승' 좌표로"),
       ("env_c > α_c · S + (1−α_c) · δ", 13, INK, "S = 채널간 max, δ = V_ref − 정지점"),
       ("(env_c − δ) / (S − δ)  >  α_c", 14, BLU, "← _xmix / _xlse 의 그 줄")]
for i, (e, sz, col, note) in enumerate(eqs):
    yy = ay + 84 + i * 46
    s += [t(700, yy, e, sz, col, "start", weight="700" if col != MUTE else None,
            mono=(col != MUTE)),
          t(700, yy + 17, note, 10, MUTE, "start")]
s += [box(700, ay + 292, 580, 26, "#ffffff", "#93c5fd", sw=1.1),
      t(990, ay + 310, "δ 는 floor 다 — α=0 이어도 임계가 정확히 δ 이므로, "
        "δ 만큼도 못 올라오면 절대 발화 못 한다", 11, INK)]

# ── C. the atom ────────────────────────────────────────────────────────────
cy = 440
s += [box(40, cy, 1260, 430, "#fff1f2", "#fca5a5"),
      t(60, cy + 28, "C. 왜 안 멈추나 — quantile 이 atom 위에 앉아 있다", 15, RED,
        "start", weight="700"),
      t(60, cy + 50, "sqrt(mel + 1e-6) 은 √(1e-6) = 0.044 mV 에서 바닥을 친다. "
                     "무음·zero padding 프레임이 전부 그 값 하나에 쌓인다 = atom.",
        11.5, MUTE, "start")]

# log axis 0.01 .. 1000 mV, so the speech hump sits inside the frame
X0, XW, YB = 150, 980, cy + 310
DECADES = (-2, 3)
def xv(v):
    return X0 + (math.log10(v) - DECADES[0]) / (DECADES[1] - DECADES[0]) * XW
s.append(line(X0, YB, X0 + XW, YB, WIRE, 1.6))
for d in range(DECADES[0], DECADES[1] + 1):
    x = xv(10.0 ** d)
    s += [line(x, YB, x, YB + 6, WIRE, 1.2),
          t(x, YB + 20, f"{10.0**d:g}", 10, MUTE)]
s.append(t(X0 + XW / 2, YB + 38, "정지점 위 상승 (mV, log scale)", 10.5, INK))

# speech frames (schematic shape; only the annotated numbers are measured)
pts = []
for i in range(161):
    v = 10 ** (DECADES[0] + (DECADES[1] - DECADES[0]) * i / 160)
    z = (math.log10(v) - math.log10(TYP)) / 0.55
    pts.append(f"{X0 + i / 160 * XW},{YB - 120 * math.exp(-0.5 * z * z)}")
s += [f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BLU}" '
      f'stroke-width="2"/>',
      t(xv(TYP), YB - 138, "speech frames", 12, BLU, weight="700"),
      t(xv(TYP), YB - 123, f"중앙 {TYP:.0f} mV", 10.5, BLU)]

# the atom
s += [f'<rect x="{xv(GUARD) - 9}" y="{YB - 200}" width="18" height="200" '
      f'fill="{RED}" opacity="0.30" stroke="{RED}" stroke-width="1.6"/>',
      t(xv(GUARD), YB - 210, "atom", 12.5, RED, weight="700"),
      t(xv(GUARD), YB - 226, "무음 프레임 2.5%", 10.5, RED)]

# seed 1234 lands INSIDE the atom -- annotate from the left
s += [t(xv(GUARD) - 24, YB - 120, "seed 1234", 11, RED, "end", weight="700"),
      t(xv(GUARD) - 24, YB - 105, f"δ = {D1:.3f} mV", 10, RED, "end"),
      t(xv(GUARD) - 24, YB - 91, "atom 안", 10, RED, "end"),
      line(xv(GUARD) - 20, YB - 108, xv(GUARD) - 11, YB - 108, RED, 1.2)]
# seed 1235 lands just outside -- annotate from the right
s += [line(xv(D2), YB - 160, xv(D2), YB, ORG, 2, "5 3"),
      t(xv(D2) + 10, YB - 150, "seed 1235", 11, ORG, "start", weight="700"),
      t(xv(D2) + 10, YB - 135, f"δ = {D2:.3f} mV", 10, ORG, "start"),
      t(xv(D2) + 10, YB - 121, "atom 밖", 10, ORG, "start")]

# the swing, drawn on the axis where both markers live
sy = YB - 46
s += [line(xv(GUARD), sy, xv(D2), sy, RED, 1.8),
      line(xv(GUARD), sy - 5, xv(GUARD), sy + 5, RED, 1.8),
      line(xv(D2), sy - 5, xv(D2), sy + 5, RED, 1.8),
      t((xv(GUARD) + xv(D2)) / 2, sy - 9, "2.4배", 12, RED, weight="700")]

s += [box(60, cy + 352, 600, 48, "#ffffff", RED, sw=1.1),
      t(72, cy + 373, "무음 프레임 2.5%   vs   xmax_floor_frac 2.0%", 12, RED,
        "start", weight="700"),
      t(72, cy + 391, "2% quantile 을 요구하는데 무음이 2.5% → quantile 이 "
        "atom 안으로 들어간다", 10.5, INK, "start")]
s += [box(680, cy + 352, 620, 48, "#ffffff", RED, sw=1.1),
      t(692, cy + 373, "표본을 늘려도 절벽은 안 움직인다", 12, RED, "start",
        weight="700"),
      t(692, cy + 391, "분산은 줄지만 quantile 은 여전히 discontinuity 위다 "
        "→ floor_frac 을 올려야 한다", 10.5, INK, "start")]

# ── D. what to hand over ───────────────────────────────────────────────────
dy = 894
s += [box(40, dy, 1260, 226, "#f0fdf4", "#86efac"),
      t(60, dy + 28, "D. 그래서 동료에게 뭐라고 하나", 15, GRN, "start",
        weight="700")]
s += [box(60, dy + 44, 600, 76, "#ffffff", "#86efac", sw=1.1),
      t(72, dy + 66, "0.1 mV 를 저항으로 맞추라는 건 불가능하다", 12.5, GRN,
        "start", weight="700"),
      t(72, dy + 86, "저항 공차도 op-amp offset 도 그보다 훨씬 크다.", 11, INK,
        "start"),
      t(72, dy + 104, "그런데 맞출 필요가 없다 — 오른쪽이 이유다.", 11, INK,
        "start")]
s += [box(680, dy + 44, 620, 76, "#ffffff", "#86efac", sw=1.1),
      t(692, dy + 66, "δ ≈ 0 이면  (env−δ)/(S−δ) ≈ env/S", 12.5, GRN, "start",
        weight="700", mono=True),
      t(692, dy + 86, "floor 가 사실상 아무 일도 안 한다.", 11, INK, "start"),
      t(692, dy + 104, "→ V_ref 를 detector 정지점(917.8 mV)에 두면 된다.",
        11, INK, "start")]
s.append(t(670, dy + 142, "실제 일은 학습된 α 16개가 한다. δ 는 그 위에 얹힌 "
                          "아주 얇은 floor 일 뿐이다.", 12.5, GRN, weight="700"))

opts = [("① floor_frac 을 0.03~0.05 로",
         "δ 가 atom 을 벗어나 조용한 음성을 재게 된다. 단 xmix 에서 0.05 는 "
         "채널 4개를 죽였다 — xlse 에서 재확인 필요"),
        ("② δ = 0 으로 확정",
         "어차피 0.34%다. 회로가 단순해지고(V_ref = 정지점) 불안정성이 원천 "
         "제거된다. 무음에서 잡음÷잡음이 되는지만 확인")]
for i, (a, b) in enumerate(opts):
    yy = dy + 174 + i * 26
    s += [t(70, yy, a, 11.5, PUR, "start", weight="700"),
          t(320, yy, b, 10.5, MUTE, "start")]

s.append(t(40, H - 16, "환산 앵커: lse_temp 0.58329 ≡ n·V_T 25.85 mV → "
                       "1 code = 44.3 mV (검산: TYP → 31.9 mV ≈ 요구 33 mV) · "
                       "생성: docs/diagrams/make_delta.py", 10, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

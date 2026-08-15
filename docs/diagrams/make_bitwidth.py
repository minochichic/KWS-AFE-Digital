"""20_bitwidth.svg -- why measuring the accumulator range saved nothing.

The one idea: a register's width is the LOG of the range it holds. Shrinking a
range by 37% therefore buys 0.54 bits, and bits do not come in fractions. The
staircase in panel C is the whole argument; everything else is setup.

Numbers are the ones actually measured on xl_g12 (runs/xl_g12/ranges.json), so
this diagram cannot drift from the result it explains.

Run:  python3 docs/diagrams/make_bitwidth.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "diagrams" / "20_bitwidth.svg"

W, H = 1200, 892
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
BAD_BG, BAD_ED = "#fdf0ef", "#cc6155"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"
HI_BG, HI_ED = "#eaf1fb", "#4a7fd0"
WARN = "#c9902a"

# the site the story is told with: b1.subs.0.pw
SITE = "stages.b1.subs.0.pw"
MEAS_LO, MEAS_HI, BOUND = -80, 74, 128
_rj = ROOT / "runs" / "xl_g12" / "ranges.json"
if _rj.exists():                                  # prefer the real file
    for s in json.loads(_rj.read_text())["sites"]:
        if s["name"] == SITE:
            MEAS_LO, MEAS_HI = math.floor(s["lo"]), math.ceil(s["hi"])
            BOUND = s["n_terms"]


def signed_bits(lo: int, hi: int) -> int:
    return 1 + max(1, math.ceil(math.log2(max(hi + 1, -lo, 1))))


MEAS_B = signed_bits(MEAS_LO, MEAS_HI)
BOUND_B = signed_bits(-BOUND, BOUND)
NEED_M = max(MEAS_HI + 1, -MEAS_LO)
NEED_B = BOUND + 1

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=9, sw=1.4, dash=None):
    BOXES.append((x, y, w, h))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{edge}" stroke-width="{sw}"{d}/>')


def text(s, x, y, size=12.5, fill=INK, anchor="middle", weight="400",
         font=FONT, limit=None):
    TEXTS.append((s, x, y, size, anchor))
    if limit is not None and tw(s, size) > limit:
        raise ValueError(f"overflows {limit:.0f}px: {s!r} -> {tw(s, size):.0f}px")
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}'
        f'</text>')


def line(x1, y1, x2, y2, color=LINE, sw=1.6, dash=None, marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#a)"' if marker else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d}{m}/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("비트폭 — 왜 실측해도 안 줄어들었나", W / 2, 36, size=21, weight="700",
     limit=W - 60)
text("한 문장: 레지스터의 폭은 담는 범위의 log 다. 범위를 37% 줄이면 0.54비트가 "
     "줄고, 비트는 쪼갤 수 없다.", W / 2, 58, size=12.5, fill=MUTE, limit=W - 60)

# ══ A ═════════════════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 156, "#ffffff", LINE, r=12, sw=1.2)
text("A.  레지스터는 칸이 정해진 그릇이다", 44, AY + 25, size=14.5, weight="700",
     anchor="start")

for i in range(9):
    x = 46 + i * 34
    box(x, AY + 42, 28, 30, HI_BG, HI_ED, r=4, sw=1)
    text("0", x + 14, AY + 62, size=13, font=MONO, limit=24)
text("9 비트", 46 + 4.5 * 34, AY + 88, size=11.5, fill=MUTE, limit=90)

text("담는 범위", 400, AY + 46, size=11.5, fill=MUTE, anchor="start", limit=90)
for j, (b, lo, hi) in enumerate([(8, -128, 127), (9, -256, 255),
                                 (10, -512, 511)]):
    text(f"{b} 비트", 400, AY + 66 + j * 19, size=12, font=MONO, anchor="start",
         weight="600", limit=60)
    text(f"−{-lo} … +{hi}", 470, AY + 66 + j * 19, size=12, font=MONO,
         anchor="start", limit=110)
    text("← 한 비트 늘 때마다 범위는 2배", 590, AY + 66 + j * 19, size=11,
         fill=MUTE, anchor="start", limit=200) if j == 1 else None

box(818, AY + 40, W - 24 - 818 - 16, 96, BAD_BG, BAD_ED, r=8, sw=1.3)
text("넘치면 —  조용히 감싼다", 818 + (W - 24 - 818 - 16) / 2, AY + 60,
     size=12.5, weight="700", fill="#a3453a", limit=310)
text("8비트에서  127 + 1  =  −128", 818 + (W - 24 - 818 - 16) / 2, AY + 84,
     size=13, font=MONO, limit=310)
text("에러가 안 난다. 정확도만 떨어진다.", 818 + (W - 24 - 818 - 16) / 2,
     AY + 106, size=11.5, fill=MUTE, limit=310)
text("→ 폭은 「일어날 수 있는 최댓값」으로 정한다",
     818 + (W - 24 - 818 - 16) / 2, AY + 125, size=11, fill=MUTE, limit=310)

# ══ B ═════════════════════════════════════════════════════════════════════
BY = 250
box(24, BY, W - 48, 132, "#ffffff", LINE, r=12, sw=1.2)
text("B.  이진 누산기의 최댓값은 「잴」 필요가 없다 — 계산된다", 44, BY + 25,
     size=14.5, weight="700", anchor="start")

text("입력도 ±1, 가중치도 ±1  →  곱은 언제나 ±1.  그걸 K개 더한다.",
     44, BY + 48, size=12.5, anchor="start", limit=560)
text("n  =  (±1) + (±1) + … + (±1)      K개", 60, BY + 76, size=13.5,
     font=MONO, anchor="start", limit=400)
text("전부 +1 이면 +K,  전부 −1 이면 −K.  그 밖은 불가능.", 60, BY + 100,
     size=12.5, anchor="start", weight="600", limit=520)

box(624, BY + 40, W - 24 - 624 - 16, 76, OK_BG, OK_ED, r=8, sw=1.3)
text(f"{SITE.split('.', 1)[1]}  는  K = {BOUND}",
     624 + (W - 24 - 624 - 16) / 2, BY + 64, size=13, font=MONO, limit=500)
text(f"경계 ±{BOUND}  →  {BOUND_B}비트.  측정 없이 확정이고, 절대 안 넘친다.",
     624 + (W - 24 - 624 - 16) / 2, BY + 90, size=12.5, weight="600",
     limit=500)

# ══ C -- the staircase ════════════════════════════════════════════════════
CY = 398
box(24, CY, W - 48, 300, "#ffffff", LINE, r=12, sw=1.2)
text("C.  ★ 그런데 비트는 범위의 log 다 — 계단으로 오른다", 44, CY + 25,
     size=14.5, weight="700", anchor="start")

PX0, PX1 = 108, 1080          # plot x range
NMAX = 300                    # need axis max
BASE = CY + 250               # y of the lowest bit level
STEP_H = 36


def px(need):
    return PX0 + (PX1 - PX0) * min(need, NMAX) / NMAX


def py(bits):
    return BASE - (bits - 6) * STEP_H


# axes
line(PX0, BASE + 10, PX1, BASE + 10, MUTE, 1.4)
line(PX0, BASE + 10, PX0, py(10) - 14, MUTE, 1.4)
text("필요한 범위 (need)", (PX0 + PX1) / 2, BASE + 34, size=11.5, fill=MUTE,
     limit=200)
text("비트", PX0 - 46, py(8) - 4, size=11.5, fill=MUTE, limit=50)

# the step function: w bits covers need in (2^(w-2), 2^(w-1)]
for w in range(7, 11):
    lo_n, hi_n = 2 ** (w - 2), 2 ** (w - 1)
    if lo_n > NMAX:
        break
    x1, x2, y = px(lo_n), px(hi_n), py(w)
    add(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{INK}" '
        f'stroke-width="2.6"/>')
    add(f'<line x1="{x1}" y1="{y}" x2="{x1}" y2="{py(w - 1)}" stroke="{LINE}" '
        f'stroke-width="1.4" stroke-dasharray="3 3"/>')
    text(f"{w}", PX0 - 22, y + 4, size=12.5, font=MONO, weight="600", limit=26)
    if hi_n <= NMAX:
        text(f"{hi_n}", x2, BASE + 26, size=10.5, fill=MUTE, limit=40)

# markers
for need, bits, col, lab, dy in [
        (NEED_M, MEAS_B, HI_ED, f"실측  need={NEED_M}  →  {MEAS_B}비트", -18),
        (NEED_B, BOUND_B, WARN, f"경계  need={NEED_B}  →  {BOUND_B}비트", -18)]:
    x, y = px(need), py(bits)
    add(f'<line x1="{x}" y1="{BASE + 10}" x2="{x}" y2="{y}" stroke="{col}" '
        f'stroke-width="1.8" stroke-dasharray="5 3"/>')
    add(f'<circle cx="{x}" cy="{y}" r="5.5" fill="{col}"/>')
    text(lab, x + 12, y + dy, size=12, fill=col, anchor="start", weight="600",
         limit=280)

text(f"범위는 {NEED_M} → {NEED_B} 로 {100 * (NEED_B - NEED_M) / NEED_M:.0f}% "
     f"커지는데, 비트는 {MEAS_B} → {BOUND_B}, 딱 1개 차이다.",
     (PX0 + PX1) / 2, CY + 56, size=12.5, weight="600", limit=W - 200)
text("1비트를 온전히 아끼려면 범위가 절반이 되어야 한다. 37% 줄이는 걸로는 "
     "계단 한 칸도 못 내려갈 때가 많다.",
     (PX0 + PX1) / 2, CY + 76, size=11.5, fill=MUTE, limit=W - 160)

# ══ D ═════════════════════════════════════════════════════════════════════
DY = 714
box(24, DY, W - 48, 154, "#ffffff", LINE, r=12, sw=1.2)
text("D.  그리고 실측에는 guard 비트가 따라붙는다", 44, DY + 25, size=14.5,
     weight="700", anchor="start")

CW = (W - 48 - 24 - 16) / 2
for i, (bg, ed, head, rows, tot, note) in enumerate([
    (HI_BG, HI_ED, "실측으로 잡으면",
     [(f"측정 범위  [{MEAS_LO}, {MEAS_HI}]", f"{MEAS_B}비트"),
      ("+ guard 1비트  (분포 밖 입력 대비)", "+1"),
      ("+ RTL saturate 로직", "필요")],
     f"= {MEAS_B + 1}비트", "본 데이터만큼만 믿을 수 있다"),
    (OK_BG, OK_ED, "경계로 잡으면",
     [(f"±K = ±{BOUND}  (계산으로 확정)", f"{BOUND_B}비트"),
      ("guard 불필요  (원리적으로 못 넘침)", "+0"),
      ("saturate 불필요", "없음")],
     f"= {BOUND_B}비트", "어떤 입력에도 안전"),
]):
    x = 36 + i * (CW + 16)
    box(x, DY + 38, CW, 104, bg, ed, r=9)
    text(head, x + 14, DY + 60, size=13, weight="700", anchor="start",
         limit=CW - 120)
    text(tot, x + CW - 14, DY + 60, size=14, weight="700", anchor="end",
         font=MONO, limit=110)
    for j, (a, b) in enumerate(rows):
        text("· " + a, x + 14, DY + 82 + j * 17, size=11.5, anchor="start",
             limit=CW - 90)
        text(b, x + CW - 14, DY + 82 + j * 17, size=11, fill=MUTE, anchor="end",
             font=MONO, limit=80)
    text(note, x + 14, DY + 136, size=11, fill=MUTE, anchor="start",
         limit=CW - 24)

add("</svg>")

bad = [b for b in BOXES if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W
       or b[1] + b[3] > H]
if bad:
    raise ValueError(f"box outside canvas: {bad[:2]}")
for s, x, y, size, anchor in TEXTS:
    w = tw(s, size)
    lo = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    if lo < -1 or lo + w > W + 1 or y > H:
        raise ValueError(f"text outside canvas: {s!r} at {lo:.0f}..{lo + w:.0f}")

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT}  ({W}x{H})")
print(f"site {SITE}: measured [{MEAS_LO},{MEAS_HI}] -> {MEAS_B}b (+1 guard), "
      f"bound +-{BOUND} -> {BOUND_B}b")

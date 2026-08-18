"""31_fold_arithmetic.svg -- how the tail's floats become an integer multiply.

Companion to 30_bn_vanishes.svg. That one ends at "the tail has to actually
compute A*n + B"; this one is where A and B come from, and why there is a
suspicious-looking constant added before every shift.

  A. constants in front of the accumulator collapse into one
  B. a float gain becomes an integer by multiplying by a power of two
  C. the same value down both paths, digit for digit
  D. adding half before a truncate IS rounding

Every number below is computed, not typed, and the two paths are asserted equal.

Run:  python3 docs/diagrams/make_fold_arithmetic.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "31_fold_arithmetic.svg"

W, H = 1240, 1044
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
CO_BG, CO_ED = "#e4eefb", "#2f6fd0"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"
NO_BG, NO_ED = "#fdf0ef", "#cc6155"
W_BG, W_ED = "#fdf3e3", "#c9902a"

# ---- the worked example, computed here so it cannot be wrong -------------- #
BN_G, ALPHA, BIAS = 0.17, 0.02, 0.21     # what training left behind
ACC = 17                                  # one accumulator value
FRAC, SHIFT = 6, 21                       # 8.6 output; 2^21 of gain headroom

GAIN = BN_G * ALPHA                                  # the fold: one constant
# 0.17*0.02 lands on 0.0034000000000000002 in binary floating point. The label
# is rounded for reading; every computation below uses GAIN itself.
GAIN_LBL = f"{GAIN:.4f}"
A = round(GAIN * (1 << FRAC) * (1 << SHIFT))
B = round(BIAS * (1 << FRAC) * (1 << SHIFT))
HALF = 1 << (SHIFT - 1)

P1 = A * ACC
P2 = P1 + B
P3 = P2 + HALF
Y_HW = P3 >> SHIFT

REAL = GAIN * ACC + BIAS                             # what PyTorch computes
Y_SW = round(REAL * (1 << FRAC))

assert Y_HW == Y_SW, f"paths disagree: hw {Y_HW} vs sw {Y_SW}"
# A is a rounded integer, so it cannot match the gain exactly -- it can only
# be within half a unit of it. Asserting the right bound rather than an
# arbitrary small one is the difference between checking the fold and
# checking that floats happen to be close.
assert abs(A / (1 << FRAC) / (1 << SHIFT) - GAIN) <= 0.5 / (1 << FRAC) / (1 << SHIFT)

# the decimal illustration of round-half-up, also computed
DIV = 10
CASES = [37, 34, 35]
TRUNC = [(v, v / DIV, v // DIV, (v + DIV // 2) // DIV, round(v / DIV))
         for v in CASES]
for v, exact, tr, half_up, want in TRUNC:
    assert half_up == want or abs(exact - int(exact) - 0.5) < 1e-9, (v, half_up)

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=6, sw=1.3, dash=None, track=True):
    if track:
        BOXES.append((x, y, w, h))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
        f'stroke="{edge}" stroke-width="{sw}"{d}/>')


def text(s, x, y, size=11.5, fill=INK, anchor="middle", weight="400",
         font=FONT, limit=None):
    TEXTS.append((s, x, y, size, anchor))
    if limit is not None and tw(s, size) > limit:
        raise ValueError(f"overflows {limit:.0f}px: {s!r} -> {tw(s, size):.0f}px")
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}'
        f'</text>')


def arrow(x1, y1, x2, y2, color=MUTE, sw=1.7, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d} marker-end="url(#a)"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("접기 —  소수 상수들을 정수 두 개로 만든다", W / 2, 36, size=21,
     weight="700", limit=W - 60)
text("꼬리는 A × 누산기 + B 를 실제로 계산해야 한다. 그 A 와 B 는 학습이 끝나면 "
     "전부 고정이라, 미리 곱해서 정수로 만들어 둘 수 있다.", W / 2, 58,
     size=12.5, fill=MUTE, limit=W - 60)

# ══ A. constants collapse ═════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 196, "#ffffff", LINE, r=12, sw=1.2)
text("A.  누산기 앞의 상수들은 하나로 뭉친다", 44, AY + 25, size=14.5,
     weight="700", anchor="start")
text("conv2_pw 가 실제로 하는 계산.  누산기 말고는 전부 학습 후 고정된 값이다.",
     44, AY + 46, size=11.5, fill=MUTE, anchor="start", limit=900)

text(f"진짜 출력  =  BN게인 × ( α × 누산기 )  +  BN오프셋", 44, AY + 76,
     size=13.5, font=MONO, anchor="start", limit=700)

# Only the two CONSTANTS merge. Laying the accumulator out in the same row with
# an arrow through it reads as though it merges too, which is the one thing that
# must not be implied -- it is the only value that changes at run time.
CW, CYY = 168, AY + 94
for i, (lbl, val) in enumerate([("BN게인", f"{BN_G}"),
                                ("α  (이진 스케일)", f"{ALPHA}")]):
    x = 44 + i * (CW + 14)
    box(x, CYY, CW, 54, CO_BG, CO_ED, r=8, sw=1.3)
    text(lbl, x + CW / 2, CYY + 21, size=11, fill=CO_ED, weight="700",
         limit=CW - 14)
    text(val, x + CW / 2, CYY + 42, size=14, font=MONO, weight="700",
         limit=CW - 14)
    text("학습 후 고정", x + CW / 2, CYY + 72, size=10, fill=MUTE, limit=CW)

arrow(400, CYY + 27, 424, CYY + 27, CO_ED)
box(430, CYY, 200, 54, OK_BG, OK_ED, r=8, sw=1.6)
text("게인  =  BN게인 × α", 530, CYY + 21, size=11, fill=OK_ED, weight="700",
     limit=184)
text(GAIN_LBL, 530, CYY + 42, size=14, font=MONO, weight="700", limit=184)
text("미리 곱해서 하나로 — 이게 「접기」", 530, CYY + 72, size=10,
     fill=OK_ED, weight="700", limit=260)

box(680, CYY, CW, 54, W_BG, W_ED, r=8, sw=1.3)
text("누산기 n", 680 + CW / 2, CYY + 21, size=11, fill=W_ED, weight="700",
     limit=CW - 14)
text(f"{ACC}", 680 + CW / 2, CYY + 42, size=14, font=MONO, weight="700",
     limit=CW - 14)
text("★ 뭉쳐지지 않는다", 680 + CW / 2, CYY + 72, size=10, fill=W_ED,
     weight="700", limit=CW + 20)

box(872, CYY - 12, W - 24 - 872 - 16, 82, "#ffffff", MUTE, r=9, sw=1.2)
text("게인과 오프셋은 학습이 끝나면 다시는", 892, CYY + 10, size=11,
     anchor="start", limit=300)
text("변하지 않는다. 누산기만 프레임마다", 892, CYY + 30, size=11,
     anchor="start", limit=300)
text("바뀐다 — 그래서 접을 수 있는 건 상수뿐.", 892, CYY + 50, size=11,
     anchor="start", limit=300)

text("conv3 은 여기에 int8 스케일이 하나 더 붙는데 똑같이 뭉치고, conv4 는 "
     "뭉친 결과가 2 의 거듭제곱이라 곱셈기조차 사라진다.", W / 2, AY + 182,
     size=11, fill=MUTE, limit=W - 90)

# ══ B. float -> integer ═══════════════════════════════════════════════════
BY = 290
box(24, BY, W - 48, 158, "#f6f9fe", CO_ED, r=12, sw=1.5)
text("B.  소수 게인을 정수로 —  2 의 거듭제곱을 곱한다", 44, BY + 25,
     size=14.5, weight="700", anchor="start")

for i, (eq, why) in enumerate([
        (f"게인 = {GAIN_LBL}", "소수. 하드웨어가 싫어한다"),
        (f"× 2^{FRAC}  (= {1 << FRAC})",
         f"출력이 8.{FRAC} 포맷이라 {1 << FRAC} 배로 저장한다"),
        (f"× 2^{SHIFT}", "정수로 만들 만큼의 여유. 나중에 되돌린다"),
        (f"A = {A:,}", "정수! 이게 .hex 에 들어간다")]):
    yy = BY + 46 + i * 26
    last = i == 3
    text(eq, 60, yy + 14, size=12.5, font=MONO, anchor="start",
         weight="700" if last else "400",
         fill=CO_ED if last else INK, limit=290)
    text(why, 370, yy + 14, size=11, fill=MUTE, anchor="start", limit=380)

box(788, BY + 44, W - 24 - 788 - 16, 96, "#ffffff", CO_ED, r=9, sw=1.3)
text(f"「나중에 되돌린다」  =  >> {SHIFT}", 808, BY + 68, size=12,
     weight="700", anchor="start", limit=390)
text(f"2^{SHIFT} 로 나누기다. 2 의 거듭제곱 나누기는", 808, BY + 90, size=11,
     anchor="start", limit=390)
text("하드웨어에서 배선을 옮기는 것이라 공짜.", 808, BY + 108, size=11,
     anchor="start", limit=390)
text(f"오프셋 B 도 같은 방식으로 = {B:,}", 808, BY + 130, size=11,
     fill=MUTE, anchor="start", limit=390)

# ══ C. the two paths ══════════════════════════════════════════════════════
CY2 = 464
box(24, CY2, W - 48, 268, "#ffffff", LINE, r=12, sw=1.2)
text(f"C.  숫자 하나를 끝까지 —  누산기에 {ACC} 이 담겼다고 하자", 44,
     CY2 + 25, size=14.5, weight="700", anchor="start")

PW = 556
for i, (title, rows, bg, ed) in enumerate([
        ("PyTorch (소수)", [
            (f"{GAIN_LBL} × {ACC} + {BIAS}", f"= {REAL:.4f}"),
            ("relu", f"= {max(0.0, REAL):.4f}"),
            (f"8.{FRAC} 로 저장:  × {1 << FRAC}",
             f"= {REAL * (1 << FRAC):.4f}"),
            ("반올림", f"= {Y_SW}")], CO_BG, CO_ED),
        ("FPGA (정수)", [
            (f"A × {ACC}", f"= {P1:,}"),
            ("+ B", f"= {P2:,}"),
            (f"+ 절반 ({HALF:,})", f"= {P3:,}"),
            (f">> {SHIFT}", f"= {Y_HW}")], OK_BG, OK_ED)]):
    x = 44 + i * (PW + 32)
    box(x, CY2 + 44, PW, 172, bg, ed, r=10, sw=1.4)
    text(title, x + 18, CY2 + 67, size=12, weight="700", fill=ed,
         anchor="start", limit=PW - 36)
    for j, (lhs, rhs) in enumerate(rows):
        yy = CY2 + 88 + j * 30
        last = j == len(rows) - 1
        if last:
            box(x + 14, yy - 18, PW - 28, 28, "#ffffff", ed, r=6, sw=1.2,
                track=False)
        text(lhs, x + 30, yy, size=12, font=MONO, anchor="start",
             weight="700" if last else "400", limit=300)
        text(rhs, x + PW - 30, yy, size=12, font=MONO, anchor="end",
             weight="700" if last else "400", limit=220)

text(f"둘 다 {Y_HW}.  이게 접기가 맞게 됐다는 뜻이다.", W / 2, CY2 + 246,
     size=13, weight="700", limit=W - 90)

# ══ D. why add half ═══════════════════════════════════════════════════════
DY = 748
box(24, DY, W - 48, 268, "#fffbf0", W_ED, r=12, sw=1.5)
text("D.  「반올림용 절반 더하기」가 뭔가", 44, DY + 25, size=14.5,
     weight="700", anchor="start")
text(">> 는 나누고 나서 소수점 아래를 그냥 버린다 (버림). 항상 아래로 깎이니 "
     "평균 0.5 씩 손해다.  10 으로 나누는 걸로 보면 쉽다.", 44, DY + 46,
     size=11.5, fill=MUTE, anchor="start", limit=1080)

COLS = [("나눌 값", 150), ("정확한 답", 190), ("버림만", 230),
        ("절반(5) 더하고 버림", 320)]
hx = 90
text("나눌 값", hx + 40, DY + 82, size=11, weight="700", fill=MUTE, limit=120)
text("정확한 답", hx + 220, DY + 82, size=11, weight="700", fill=MUTE, limit=140)
text("버림만", hx + 430, DY + 82, size=11, weight="700", fill=NO_ED, limit=140)
text("절반(5) 더하고 버림", hx + 700, DY + 82, size=11, weight="700",
     fill=OK_ED, limit=240)

for j, (v, exact, tr, half_up, want) in enumerate(TRUNC):
    yy = DY + 96 + j * 40
    box(60, yy, 1120, 34, "#ffffff", LINE, r=7, sw=1.1)
    text(f"{v} / {DIV}", hx + 40, yy + 22, size=12, font=MONO, limit=120)
    text(f"{exact}", hx + 220, yy + 22, size=12, font=MONO, limit=140)
    ok_tr = tr == want
    text(f"{tr}", hx + 430, yy + 22, size=12, font=MONO, weight="700",
         fill=OK_ED if ok_tr else NO_ED, limit=140)
    text("✓" if ok_tr else f"✗  {want} 가 맞다", hx + 500, yy + 22, size=11,
         fill=OK_ED if ok_tr else NO_ED, anchor="start", limit=160)
    text(f"({v}+5) / {DIV} = {(v + 5) / DIV}  →  {half_up}", hx + 700,
         yy + 22, size=12, font=MONO, weight="700", limit=280)
    text("✓", hx + 960, yy + 22, size=11, fill=OK_ED, anchor="start", limit=40)

text("절반을 더하고 버리면 = 반올림.  " + f"2^{SHIFT} 로 나눌 때 그 절반이 "
     f"2^{SHIFT - 1} = {HALF:,} 이고, 위 C 에 나온 그 숫자다.", W / 2,
     DY + 236, size=12, weight="700", limit=W - 90)
text("하드웨어에서 이렇게 하는 이유: 상수 하나 더하기는 덧셈기 하나(누산기 "
     "초기값으로 넣으면 사실상 공짜)인데, 진짜 반올림 로직은 비트를 들여다봐야 "
     "해서 더 비싸다.", W / 2, DY + 256, size=11, fill=MUTE, limit=W - 90)

add("</svg>")

bad = [b for b in BOXES if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W
       or b[1] + b[3] > H]
if bad:
    raise ValueError(f"box outside canvas: {bad[:2]}")
for s, x, y, size, anchor in TEXTS:
    ww = tw(s, size)
    lo = x if anchor == "start" else (x - ww if anchor == "end" else x - ww / 2)
    if lo < -1 or lo + ww > W + 1 or y > H:
        raise ValueError(f"text outside canvas: {s!r} at {lo:.0f}..{lo + ww:.0f}")

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT}  ({W}x{H})")
print(f"worked example: gain {GAIN_LBL} -> A={A:,}  B={B:,}  shift={SHIFT}")
print(f"both paths give {Y_HW} (float said {REAL * (1 << FRAC):.4f})")

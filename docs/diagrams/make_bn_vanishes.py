"""30_bn_vanishes.svg -- why sign() removes BN and relu() does not.

The objection that prompted this is the right one: "you have to compute BN to
compare it with zero, don't you?" You do, if you compute then compare. The trick
is that you can rearrange the inequality FIRST, on a laptop, before the chip
ever runs -- and what comes out the other side is a constant.

  A. the network ends two different ways, and that is the whole split
  B. the thermometer: an inequality can be pre-solved
  C. the same rearrangement on BN, term by term
  D. why relu cannot be pre-solved -- it asks a different question

Run:  python3 docs/diagrams/make_bn_vanishes.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "30_bn_vanishes.svg"

W, H = 1240, 1010
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"          # sign path: BN disappears
NO_BG, NO_ED = "#fdf0ef", "#cc6155"          # relu path: BN survives
CO_BG, CO_ED = "#e4eefb", "#2f6fd0"          # constants
W_BG, W_ED = "#fdf3e3", "#c9902a"

# the thermometer, computed rather than typed
F_THRESH = 100.0
C_THRESH = (F_THRESH - 32.0) / 1.8
assert abs(C_THRESH * 1.8 + 32.0 - F_THRESH) < 1e-9

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

text("sign 이면 BN 이 사라지고, relu 면 안 사라진다", W / 2, 36, size=21,
     weight="700", limit=W - 60)
text("「계산을 해야 0 과 비교하지 않나?」 — 계산하고 비교하면 그렇다. "
     "부등식은 미리 풀어둘 수 있다.", W / 2, 58, size=12.5, fill=MUTE,
     limit=W - 60)

# ══ A. the network ends two ways ══════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 174, "#ffffff", LINE, r=12, sw=1.2)
text("A.  네트워크는 두 가지 방식으로 끝난다", 44, AY + 25, size=14.5,
     weight="700", anchor="start")

CH = 46
for i, (title, expr, ask, bg, ed, verdict) in enumerate([
        ("몸통  conv1 · b1 · b2 · b3 · conv2_dw",
         "출력 = sign( BN(누산기) )", "묻는 것:  「어느 쪽이냐」",
         OK_BG, OK_ED, "BN 이 사라진다  →  정수 비교 하나"),
        ("꼬리  conv2_pw · conv3 · conv4",
         "출력 = relu( BN(누산기) )", "묻는 것:  「얼마냐」",
         NO_BG, NO_ED, "BN 이 남는다  →  곱셈 + 덧셈 + 시프트")]):
    x = 44 + i * 588
    box(x, AY + 40, 564, 120, bg, ed, r=10, sw=1.4)
    text(title, x + 18, AY + 63, size=11.5, weight="700", anchor="start",
         fill=ed, limit=528)
    text(expr, x + 18, AY + 88, size=13.5, font=MONO, anchor="start", limit=528)
    text(ask, x + 18, AY + 111, size=11.5, fill=MUTE, anchor="start", limit=528)
    box(x + 14, AY + 122, 536, 28, "#ffffff", ed, r=6, sw=1.1, track=False)
    text(verdict, x + 28, AY + 141, size=11.5, weight="700", anchor="start",
         limit=508)

# ══ B. the thermometer ════════════════════════════════════════════════════
BY = 268
box(24, BY, W - 48, 218, "#f7fbf8", OK_ED, r=12, sw=1.5)
text("B.  온도계 —  부등식은 미리 풀어둘 수 있다", 44, BY + 25, size=14.5,
     weight="700", anchor="start")
text("「섭씨를 화씨로 바꿔서 100°F 가 넘으면 알람을 울려라」", 44, BY + 46,
     size=11.5, fill=MUTE, anchor="start", limit=800)

STEPS = [("C × 1.8 + 32  >  100", "시키는 대로.  측정할 때마다 곱셈 한 번"),
         ("C × 1.8       >  68", "양변에서 32 를 뺀다"),
         (f"C             >  {C_THRESH:.1f}", "양변을 1.8 로 나눈다")]
for i, (eq, why) in enumerate(STEPS):
    yy = BY + 66 + i * 38
    last = i == len(STEPS) - 1
    box(60, yy, 400, 32, "#ffffff" if not last else OK_BG,
        OK_ED if last else LINE, r=7, sw=1.5 if last else 1.1)
    text(eq, 78, yy + 21, size=13, font=MONO, anchor="start",
         weight="700" if last else "400", limit=368)
    text(why, 478, yy + 21, size=11, fill=MUTE, anchor="start", limit=280)
    if not last:
        arrow(260, yy + 33, 260, yy + 37, MUTE, sw=1.4)

box(786, BY + 62, W - 24 - 786 - 16, 122, "#ffffff", OK_ED, r=9, sw=1.3)
text(f"「화씨 100 도」는 「섭씨 {C_THRESH:.1f} 도」다", 806, BY + 86, size=12.5,
     weight="700", anchor="start", limit=380)
text(f"{C_THRESH:.1f} 은 측정값과 무관하다. 노트북에서", 806, BY + 108,
     size=11, anchor="start", limit=380)
text("미리 한 번 계산해서 적어두면 된다.", 806, BY + 126, size=11,
     anchor="start", limit=380)
text("온도계는 변환 없이 눈금만 비교한다 —", 806, BY + 150, size=11,
     weight="700", anchor="start", limit=380)
text("곱셈이 사라졌다.", 806, BY + 168, size=11, weight="700", anchor="start",
     limit=380)

text(f"검산:  {C_THRESH:.1f} × 1.8 + 32 = {C_THRESH * 1.8 + 32:.0f}", 60,
     BY + 200, size=10.5, fill=MUTE, anchor="start", limit=400)

# ══ C. the same thing on BN ═══════════════════════════════════════════════
CY = 502
box(24, CY, W - 48, 286, "#ffffff", LINE, r=12, sw=1.2)
text("C.  BN 에 똑같이 한다", 44, CY + 25, size=14.5, weight="700",
     anchor="start")
text("BN 은 학습이 끝나면  BN(n) = g × (α×n − μ) + β  다.  변하는 건 누산기 n "
     "하나뿐이고 나머지는 전부 고정된 상수다.", 44, CY + 46, size=11.5,
     fill=MUTE, anchor="start", limit=1000)

BSTEPS = [("BN(n)  >  0", None),
          ("g × (α×n − μ) + β  >  0", "BN 을 펼친다"),
          ("g×α×n  >  g×μ − β", "n 이 든 항만 왼쪽으로"),
          ("n  >  (g×μ − β) / (g×α)", "양변을 g×α 로 나눈다")]
for i, (eq, why) in enumerate(BSTEPS):
    yy = CY + 66 + i * 38
    last = i == len(BSTEPS) - 1
    box(60, yy, 470, 32, CO_BG if last else "#ffffff",
        CO_ED if last else LINE, r=7, sw=1.5 if last else 1.1)
    text(eq, 78, yy + 21, size=13, font=MONO, anchor="start",
         weight="700" if last else "400", limit=438)
    if why:
        text(why, 548, yy + 21, size=11, fill=MUTE, anchor="start", limit=220)
    if not last:
        arrow(295, yy + 33, 295, yy + 37, MUTE, sw=1.4)

text("└──────── 이 덩어리가 T ────────┘", 296, CY + 232, size=11,
     font=MONO, fill=CO_ED, weight="700", limit=470)
text("T 에는 n 이 없다 → 미리 계산해서 .hex 에 적어둔다.", 296, CY + 254,
     size=11.5, weight="600", limit=520)
text("칩은 실행할 때 「n ≥ T」 비교 하나만 한다.", 296, CY + 272, size=11.5,
     weight="600", limit=520)

box(786, CY + 66, W - 24 - 786 - 16, 82, W_BG, W_ED, r=9, sw=1.3)
text("n 은 정수다", 806, CY + 88, size=12, weight="700", anchor="start",
     limit=380)
text("n > 37.8  과  n ≥ 38  은 완전히 같다.", 806, CY + 110, size=11,
     anchor="start", limit=380)
text("저장되는 건 정수 38 → 반올림 오차 0.", 806, CY + 132, size=11,
     weight="700", anchor="start", limit=380)

box(786, CY + 158, W - 24 - 786 - 16, 82, W_BG, W_ED, r=9, sw=1.3)
text("g 가 음수면 부등호가 뒤집힌다", 806, CY + 180, size=12, weight="700",
     anchor="start", limit=380)
text("음수로 나누면 > 가 < 가 된다. 그래서", 806, CY + 202, size=11,
     anchor="start", limit=380)
text("채널당 (문턱값, 방향 1비트) 두 개를 저장.", 806, CY + 224, size=11,
     anchor="start", limit=380)

text("이 두 숫자가 b1_s0_dw_t.hex 파일 내용의 전부다.",
     786 + (W - 24 - 786 - 16) / 2, CY + 256, size=11, fill=MUTE, limit=430)
text("g·α·μ·β 는 하드웨어에 하나도 남지 않는다.",
     786 + (W - 24 - 786 - 16) / 2, CY + 274, size=11, fill=MUTE, limit=430)

# ══ D. why relu cannot ════════════════════════════════════════════════════
DY = 804
box(24, DY, W - 48, 178, "#fffbf7", NO_ED, r=12, sw=1.5)
text("D.  relu 는 왜 안 되나 — 다른 질문을 하기 때문", 44, DY + 25, size=14.5,
     weight="700", anchor="start")

for i, (who, q, form, note, bg, ed) in enumerate([
        ("sign", "「어느 쪽이냐」", "BN(n) > 0", "부등식 → 양변을 옮길 수 있다",
         OK_BG, OK_ED),
        ("relu", "「얼마냐」", "relu(BN(n)) = 0.2678", "값 → 옮길 변이 없다",
         NO_BG, NO_ED)]):
    yy = DY + 44 + i * 58
    box(60, yy, W - 120, 50, bg, ed, r=8, sw=1.3)
    text(who, 82, yy + 31, size=13, font=MONO, weight="700", fill=ed,
         anchor="start", limit=70)
    text(q, 160, yy + 31, size=12, weight="700", anchor="start", limit=150)
    text(form, 330, yy + 31, size=12.5, font=MONO, anchor="start", limit=260)
    text(note, 620, yy + 31, size=11.5, fill=MUTE, anchor="start", limit=480)

text("그래서 꼬리는 A × n + B 를 실제로 계산해야 한다 — 그 A 와 B 를 정수로 "
     "만드는 게 31_fold_arithmetic.svg 다.", W / 2, DY + 168, size=11.5,
     weight="600", limit=W - 90)

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
print(f"thermometer verified: {F_THRESH:.0f}F = {C_THRESH:.4f}C")

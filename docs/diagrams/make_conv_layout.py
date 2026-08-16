"""22_conv_layout.svg -- what depthwise/pointwise actually read, and why padding
breaks the binary MAC.

Written because the RTL notes in rtl/README.md are correct but unreadable
without a picture: they talk about tap runs, n_valid and a shift, and none of
that means anything until you can see which cells a convolution touches.

Three things, in order:
  A. the [C, T] plane, and the two directions the two convs read it in
  B. what padding inserts at the edge, and why "0" is a problem for 2P - N
  C. one word = one frame's channels, and what that costs each conv

Run:  python3 docs/diagrams/make_conv_layout.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "22_conv_layout.svg"

W, H = 1240, 1012
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
GRID = "#e8ecf2"
DW_BG, DW_ED = "#e4eefb", "#2f6fd0"       # depthwise
PW_BG, PW_ED = "#fdf3e3", "#c9902a"       # pointwise
PAD_BG, PAD_ED = "#fdf0ef", "#cc6155"     # padding
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=9, sw=1.4, dash=None, track=True):
    if track:
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


def arrow(x1, y1, x2, y2, color=MUTE, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')


CELL = 21
NC, NT = 8, 16          # a zoomed-in corner of the real [16 or 128, 128] plane


def grid(x0, y0, hi_cells, hi_bg, hi_ed, pad_cells=()):
    """Draw the [C, T] plane; highlight cells given as (channel, frame)."""
    for c in range(NC):
        for t in range(NT):
            x, y = x0 + t * CELL, y0 + c * CELL
            if (c, t) in pad_cells:
                f, e = PAD_BG, PAD_ED
            elif (c, t) in hi_cells:
                f, e = hi_bg, hi_ed
            else:
                f, e = "#ffffff", GRID
            box(x, y, CELL, CELL, f, e, r=2, sw=1, track=False)
    text("frame (시간) →", x0 + NT * CELL / 2, y0 + NC * CELL + 17, size=10.5,
         fill=MUTE, limit=NT * CELL)
    add(f'<text x="{x0 - 10}" y="{y0 + NC * CELL / 2}" font-family="{FONT}" '
        f'font-size="10.5" fill="{MUTE}" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 10} {y0 + NC * CELL / 2})">channel ↑</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("convolution 이 무엇을 읽는가 — 그리고 padding 이 왜 문제인가", W / 2, 36,
     size=21, weight="700", limit=W - 60)
text("NN 의 입력은 [channel, frame] 평면이다. 아래 격자는 그 평면의 한 귀퉁이를 "
     "확대한 것.", W / 2, 58, size=12.5, fill=MUTE, limit=W - 60)

# ══ A ═════════════════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 292, "#ffffff", LINE, r=12, sw=1.2)
text("A.  두 conv 는 같은 평면을 서로 다른 방향으로 읽는다", 44, AY + 25,
     size=14.5, weight="700", anchor="start")

gx1, gy = 90, AY + 62
grid(gx1, gy, {(3, t) for t in range(2, 9)}, DW_BG, DW_ED)
box(gx1 + 2 * CELL - 2, gy + 3 * CELL - 2, 7 * CELL + 4, CELL + 4,
    "none", DW_ED, r=4, sw=2.4, track=False)
text("depthwise", gx1 + NT * CELL / 2, gy - 26, size=13.5, weight="700",
     fill=DW_ED, limit=NT * CELL)
text("채널 하나 · 시간 방향 K칸 (가로)", gx1 + NT * CELL / 2, gy - 8, size=11,
     fill=MUTE, limit=NT * CELL + 40)

gx2 = 700
grid(gx2, gy, {(c, 6) for c in range(NC)}, PW_BG, PW_ED)
box(gx2 + 6 * CELL - 2, gy - 2, CELL + 4, NC * CELL + 4, "none", PW_ED, r=4,
    sw=2.4, track=False)
text("pointwise", gx2 + NT * CELL / 2, gy - 26, size=13.5, weight="700",
     fill="#8a6410", limit=NT * CELL)
text("프레임 하나 · 전 채널 (세로)", gx2 + NT * CELL / 2, gy - 8, size=11,
     fill=MUTE, limit=NT * CELL + 40)

for x, s in [(gx1 + NT * CELL / 2, "출력 한 칸 = 가로 13칸의 합  (K=13)"),
             (gx2 + NT * CELL / 2, "출력 한 칸 = 세로 128칸의 합  (C=128)")]:
    text(s, x, gy + NC * CELL + 42, size=12, weight="600", limit=NT * CELL + 90)
text("이 방향 차이가 아래 모든 것을 결정한다.", W / 2, AY + 272, size=12,
     fill=MUTE, limit=600)

# ══ B ═════════════════════════════════════════════════════════════════════
BY = 386
box(24, BY, W - 48, 296, "#ffffff", LINE, r=12, sw=1.2)
text("B.  왼쪽 끝에서 창이 평면 밖으로 나간다 — 거기 뭘 넣나", 44, BY + 25,
     size=14.5, weight="700", anchor="start")

gx3, gy3 = 130, BY + 66
pad = {(3, t) for t in range(-6, 0)}
for c in range(NC):
    for t in range(-6, NT):
        x, y = gx3 + (t + 6) * CELL, gy3 + c * CELL
        if (c, t) in pad:
            f, e = PAD_BG, PAD_ED
        elif c == 3 and 0 <= t <= 6:
            f, e = DW_BG, DW_ED
        elif t < 0:
            f, e = "#fafbfc", GRID
        else:
            f, e = "#ffffff", GRID
        box(x, y, CELL, CELL, f, e, r=2, sw=1, track=False)
add(f'<line x1="{gx3 + 6 * CELL}" y1="{gy3 - 8}" x2="{gx3 + 6 * CELL}" '
    f'y2="{gy3 + NC * CELL + 8}" stroke="{INK}" stroke-width="2" '
    f'stroke-dasharray="5 3"/>')
text("← 평면 밖 (padding)", gx3 + 3 * CELL, gy3 - 16, size=11, fill=PAD_ED,
     weight="600", limit=6 * CELL + 60)
text("t=0 의 창", gx3 + 9.5 * CELL, gy3 - 16, size=11, fill=DW_ED,
     weight="600", limit=140)
text("padding 6칸", gx3 + 3 * CELL, gy3 + NC * CELL + 20, size=10.5,
     fill=PAD_ED, limit=6 * CELL + 20)
text("유효 7칸", gx3 + 9.5 * CELL, gy3 + NC * CELL + 20, size=10.5, fill=DW_ED,
     limit=7 * CELL)

px = 620
box(px, BY + 50, W - 24 - px - 20, 96, PAD_BG, PAD_ED, r=9)
text("padding 에 들어가는 값은 0 이다. −1 이 아니다.", px + 16, BY + 74,
     size=13, weight="700", anchor="start", limit=W - px - 60)
text("우리 인코딩 {−1, +1} 안에 0 은 없다. 기여가 0 —  더해지지도 "
     "빼지지도 않는다.", px + 16, BY + 96, size=11.5,
     anchor="start", limit=W - px - 60)
text("PyTorch conv1d 의 기본 동작이고, 학습된 가중치가 그 전제로 맞춰져 있다.",
     px + 16, BY + 116, size=11.5, fill=MUTE, anchor="start",
     limit=W - px - 60)
text("골든 벡터로 확인: t=0 의 max|acc| 가 정확히 7 이었다.", px + 16,
     BY + 136, size=11.5, fill=MUTE, anchor="start", limit=W - px - 60)

box(px, BY + 158, W - 24 - px - 20, 120, "#ffffff", LINE, r=9, sw=1.2)
text("그런데 우리 MAC 은 0 을 표현 못 한다", px + 16, BY + 180, size=12.5,
     weight="700", anchor="start", limit=W - px - 60)
text("acc = 2·popcount(XNOR) − N", px + 16, BY + 204, size=13, font=MONO,
     anchor="start", limit=W - px - 60)
text("N 은 「항이 몇 개인가」다. 모든 항이 ±1 이라고 가정한다.",
     px + 16, BY + 224, size=11.5, anchor="start", limit=W - px - 60)
text("→ padding 칸은 N 에서도 빼야 한다.  N = 13 이 아니라 7.",
     px + 16, BY + 248, size=12.5, weight="700", fill=OK_ED, anchor="start",
     limit=W - px - 60)
text("이걸 놓치면 가장자리 프레임만 틀린다 — 값은 그럴듯하다.",
     px + 16, BY + 268, size=11, fill=MUTE, anchor="start", limit=W - px - 60)

# ══ C ═════════════════════════════════════════════════════════════════════
CY = 698
box(24, CY, W - 48, 290, "#ffffff", LINE, r=12, sw=1.2)
text("C.  메모리에는 세로 한 줄이 한 word 로 들어간다 (frame-major)",
     44, CY + 25, size=14.5, weight="700", anchor="start")

gx4, gy4 = 110, CY + 62
grid(gx4, gy4, set(), DW_BG, DW_ED)
for t in range(NT):
    box(gx4 + t * CELL - 1, gy4 - 1, CELL + 2, NC * CELL + 2, "none",
        PW_ED if t == 6 else "#dfe5ee", r=3, sw=2 if t == 6 else 1,
        track=False)
text("세로 한 줄 = 1 word (채널 c 가 비트 c)", gx4 + NT * CELL / 2, gy4 - 14,
     size=11.5, weight="600", limit=NT * CELL + 90)

qx = 560
for i, (nm, col, lines, cost) in enumerate([
    ("pointwise", PW_ED,
     ["필요한 것: 세로 한 줄 = 그 word 자체",
      "word 를 그대로 MAC 에 흘려보내면 끝"],
     "128채널 = 4 word = 4 사이클"),
    ("depthwise", DW_ED,
     ["필요한 것: 가로 13칸 = 서로 다른 13개 word 에서",
      "각각 비트 c 하나씩 → 모아야 한다 (bit gather)"],
     "13프레임을 레지스터에 붙들고 있는다 (line buffer)"),
]):
    y = CY + 56 + i * 106
    box(qx, y, W - 24 - qx - 20, 92, "#ffffff", col, r=9, sw=1.5)
    text(nm, qx + 16, y + 24, size=13, weight="700", fill=col, anchor="start",
         limit=200)
    for j, s in enumerate(lines):
        text(s, qx + 16, y + 46 + j * 18, size=11.5, anchor="start",
             limit=W - qx - 60)
    text(cost, qx + 16, y + 82, size=11, fill=MUTE, anchor="start",
         limit=W - qx - 60)

text("둘 중 하나는 반드시 대가를 치른다. pointwise 쪽이 층 수도 많고 항 수도 "
     "커서(64~128 vs 13~29) 그쪽을 공짜로 만들었다.",
     W / 2, CY + 272, size=11.5, fill=MUTE, limit=W - 90)

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

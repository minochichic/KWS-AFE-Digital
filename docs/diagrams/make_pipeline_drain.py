"""24_pipeline_drain.svg -- why the line buffer keeps shifting after the last
frame arrives.

One rule explains the whole timeline: output t needs input t+6, so the buffer
runs 6 pushes ahead of the outputs. That forces a fill phase at the start where
nothing comes out, and a drain phase at the end where nothing goes in.

The pads land on opposite ends in the two phases, which is why only the left
edge needs a shift.

Run:  python3 docs/diagrams/make_pipeline_drain.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "24_pipeline_drain.svg"

W, H = 1240, 872
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
VAL_BG, VAL_ED = "#e4eefb", "#2f6fd0"
PAD_BG, PAD_ED = "#fdf0ef", "#cc6155"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"
W_ED = "#c9902a"

K, P, T = 13, 6, 64

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
        f'stroke-width="{sw}"{d} marker-end="url(#a)"/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("line buffer 의 채우기와 비우기 — 「입력이 끝난다」의 의미", W / 2, 36,
     size=21, weight="700", limit=W - 60)
text(f"K={K}, padding={P}, 프레임 {T}개짜리 클립 하나를 처리하는 전 과정.",
     W / 2, 58, size=12.5, fill=MUTE, limit=W - 60)

# ══ A. the rule ═══════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 158, "#fffbf0", W_ED, r=12, sw=1.4)
text("A.  규칙 하나 —  출력 t 를 만들려면 입력 t+6 까지 있어야 한다",
     44, AY + 26, size=14.5, weight="700", anchor="start")

CW, CH = 44, 30
gx = 250
gy = AY + 52
for j in range(K):
    x = gx + j * CW
    box(x, gy, CW - 3, CH, VAL_BG, VAL_ED, r=3, sw=1.2, track=False)
    lbl = "t" if j == P else (f"t{j-P:+d}")
    text(lbl, x + (CW - 3) / 2, gy + 20, size=10.5, font=MONO, limit=CW - 6)
text("창 = 앞뒤 6칸", gx - 16, gy + 20, size=11.5, weight="600", anchor="end",
     limit=180)
add(f'<rect x="{gx + P * CW - 3}" y="{gy - 4}" width="{CW + 1}" '
    f'height="{CH + 8}" rx="4" fill="none" stroke="{INK}" stroke-width="2"/>')
arrow(gx + 12.5 * CW + 14, gy + CH / 2, gx + 13 * CW + 40, gy + CH / 2)
text("가장 최근에 필요한 것", gx + 13 * CW + 52, gy + 14, size=11.5,
     weight="600", anchor="start", limit=220)
text("= 입력 t+6", gx + 13 * CW + 52, gy + 32, size=12, font=MONO,
     anchor="start", weight="700", limit=220)
text("→ buffer 는 출력보다 6칸 앞서 달린다. 그래서 앞뒤로 6칸씩 어긋난 구간이 "
     "생긴다.", W / 2, AY + 132, size=12.5, weight="600", limit=W - 90)

# ══ B. timeline ═══════════════════════════════════════════════════════════
BY = 254
box(24, BY, W - 48, 402, "#ffffff", LINE, r=12, sw=1.2)
text("B.  70번 밀어서 출력 64개", 44, BY + 25, size=14.5, weight="700",
     anchor="start")

BW, BH = 44, 28
bx = 268
hdr = BY + 50
text("미는 것", bx - 16, hdr, size=10.5, fill=MUTE, anchor="end", limit=110)
text("line buffer  (오래된 것 ←  → 새 것)", bx + 6.5 * BW, hdr, size=10.5,
     fill=MUTE, limit=13 * BW)
text("valid", bx + 13 * BW + 90, hdr, size=10.5, fill=MUTE, limit=110)
text("출력", bx + 13 * BW + 230, hdr, size=10.5, fill=MUTE, limit=80)

# (push label, list of 13 slots: frame no or None, output label, phase)
ROWS = [
    ("f0",  [None] * 12 + [0],                              "—", "fill"),
    ("f5",  [None] * 7 + [0, 1, 2, 3, 4, 5],                 "—", "fill"),
    ("f6",  [None] * 6 + [0, 1, 2, 3, 4, 5, 6],              "t=0", "run"),
    ("f63", list(range(51, 64)),                             "t=57", "run"),
    ("(없음)", list(range(52, 64)) + [None],                 "t=58", "drain"),
    ("(없음) ×6", list(range(57, 64)) + [None] * 6,          "t=63", "drain"),
]
for _p, _s, _o, _ph in ROWS:
    assert len(_s) == K, f"{_p}: {len(_s)} slots, expected {K}"
for i, (push, slots, outp, phase) in enumerate(ROWS):
    y = BY + 66 + i * 52
    text(push, bx - 16, y + 19, size=11.5, weight="600", anchor="end",
         limit=110, font=MONO if push.startswith("f") else FONT)
    bits = ""
    for j, fr in enumerate(slots):
        x = bx + j * BW
        if fr is None:
            box(x, y, BW - 3, BH, PAD_BG, PAD_ED, r=3, sw=1.1, track=False)
            text("pad", x + (BW - 3) / 2, y + 19, size=9.5, fill=PAD_ED,
                 limit=BW - 8)
            bits += "0"
        else:
            box(x, y, BW - 3, BH, VAL_BG, VAL_ED, r=3, sw=1.1, track=False)
            text(f"f{fr}", x + (BW - 3) / 2, y + 19, size=10.5, font=MONO,
                 limit=BW - 8)
            bits += "1"
    text(bits, bx + 13 * BW + 90, y + 19, size=10.5, font=MONO, limit=130)
    oc = MUTE if outp == "—" else OK_ED
    text(outp, bx + 13 * BW + 230, y + 19, size=12, font=MONO, fill=oc,
         weight="700", limit=80)
    if i in (1, 3):
        add(f'<line x1="44" y1="{y + BH + 12}" x2="{W - 44}" '
            f'y2="{y + BH + 12}" stroke="{GRID}" stroke-width="1.5"/>')

text("마지막 실제 프레임 f63 을 밀어도 출력은 t=57 까지다. 남은 t=58~63 의 창은 "
     "f64~f69 를 필요로 하는데 그런 건 없다 — 그래서 빈 칸을 6번 더 민다.",
     W / 2, BY + 386, size=12, weight="600", limit=W - 90)

# ══ C. phases ═════════════════════════════════════════════════════════════
CY = 674
box(24, CY, W - 48, 178, "#ffffff", LINE, r=12, sw=1.2)
text("C.  세 구간, 그리고 pad 가 어느 쪽에 붙는가", 44, CY + 25, size=14.5,
     weight="700", anchor="start")

PH = [("채우기 (fill)", "6 push", "없음", "0000001111111",
       "낮은 쪽 (오래된 칸)", "shift = 6 … 1", PAD_BG, PAD_ED),
      ("본류", "58 push", "t=0 … 57", "1111111111111",
       "없음", "shift = 0", OK_BG, OK_ED),
      ("비우기 (drain)", "6 push", "t=58 … 63", "1111111000000",
       "높은 쪽 (새 칸)", "shift = 0", "#eef2f7", "#7d8896")]
PW_ = (W - 88 - 32) / 3
for i, (nm, n, out, bits, where, sh, bg, ed) in enumerate(PH):
    x = 44 + i * (PW_ + 16)
    box(x, CY + 44, PW_, 118, bg, ed, r=9, sw=1.4)
    text(nm, x + 14, CY + 68, size=13, weight="700", anchor="start",
         limit=PW_ - 90)
    text(n, x + PW_ - 14, CY + 68, size=11.5, fill=MUTE, anchor="end",
         font=MONO, limit=80)
    text(f"출력  {out}", x + 14, CY + 90, size=11.5, anchor="start",
         limit=PW_ - 28)
    text(bits, x + 14, CY + 110, size=11, font=MONO, anchor="start",
         limit=PW_ - 28)
    text(f"pad 위치  {where}", x + 14, CY + 130, size=11, fill=MUTE,
         anchor="start", limit=PW_ - 28)
    text(sh, x + 14, CY + 150, size=11.5, weight="600", anchor="start",
         limit=PW_ - 28)

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
print(f"check: {T} outputs need {T + P} pushes ({P} fill + {T - P} run + {P} drain)")

"""23_mac_alignment.svg -- three questions the RTL notes assume away.

  A. n_valid says HOW MANY taps, not WHICH. At the left edge the valid ones sit
     high, and the mask covers low bits -- so the count can be right and the
     taps still wrong. Shifting activation AND weight together fixes it.
  B. why a word is 32 bits, and why 32 channels are wanted at once.
  C. what a line buffer actually does, frame by frame.

Run:  python3 docs/diagrams/make_mac_alignment.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "23_mac_alignment.svg"

W, H = 1240, 1030
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
VAL_BG, VAL_ED = "#e4eefb", "#2f6fd0"      # valid tap
PAD_BG, PAD_ED = "#fdf0ef", "#cc6155"      # padding tap
W_BG, W_ED = "#fdf3e3", "#c9902a"          # weight
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


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("mask · shift · 정렬 — 그리고 word 와 line buffer", W / 2, 36, size=21,
     weight="700", limit=W - 60)
text("K=13, padding=6, 출력 프레임 t=0 인 경우로 전부 설명한다.", W / 2, 58,
     size=12.5, fill=MUTE, limit=W - 60)

# ══ A. tap alignment ══════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 338, "#ffffff", LINE, r=12, sw=1.2)
text("A.  n_valid 은 「몇 개」지 「어느 것」이 아니다", 44, AY + 25, size=14.5,
     weight="700", anchor="start")
text("t=0 의 창은 입력 위치 −6 … +6 을 덮는다. tap 번호 0~5 가 padding, "
     "6~12 가 실제 데이터.", 44, AY + 44, size=11.5, fill=MUTE, anchor="start",
     limit=800)

CW, CH = 42, 30
x0 = 150


def taprow(y, order, label, sub, hi_lo=None, wrow=False):
    """order[i] = tap number sitting at bit position i, or None for empty."""
    text(label, x0 - 14, y + 20, size=11.5, weight="600", anchor="end",
         limit=120)
    if sub:
        text(sub, x0 - 14, y + 36, size=10, fill=MUTE, anchor="end", limit=120)
    for i, tapno in enumerate(order):
        x = x0 + i * CW
        if tapno is None:
            box(x, y, CW - 3, CH, "#fafbfc", GRID, r=3, sw=1, track=False)
            continue
        pad = tapno < 6
        if wrow:
            f, e = W_BG, W_ED
        else:
            f, e = (PAD_BG, PAD_ED) if pad else (VAL_BG, VAL_ED)
        box(x, y, CW - 3, CH, f, e, r=3, sw=1.3, track=False)
        text(f"{'w' if wrow else 't'}{tapno}", x + (CW - 3) / 2, y + 20,
             size=11, font=MONO,
             fill=(PAD_ED if (pad and not wrow) else INK), limit=CW - 8)
    if hi_lo is not None:
        a, b = hi_lo
        add(f'<rect x="{x0 + a * CW - 3}" y="{y - 5}" '
            f'width="{(b - a + 1) * CW - 1}" height="{CH + 10}" rx="5" '
            f'fill="none" stroke="{OK_ED}" stroke-width="2.4"/>')


# bit position ruler
text("bit position →", x0 + 6.5 * CW, AY + 68, size=10.5, fill=MUTE, limit=160)
for i in range(13):
    text(str(i), x0 + i * CW + (CW - 3) / 2, AY + 84, size=10, fill=MUTE,
         limit=CW - 6)

y1 = AY + 92
taprow(y1, list(range(13)), "그대로 넘기면", "activation")
taprow(y1 + 38, list(range(13)), "", "weight", wrow=True)
add(f'<rect x="{x0 - 3}" y="{y1 - 5}" width="{7 * CW - 1}" '
    f'height="{CH + 48}" rx="5" fill="none" stroke="{PAD_ED}" '
    f'stroke-width="2.4"/>')
text("mask = 낮은 7비트", x0 + 3.5 * CW, y1 + 92, size=11.5, weight="600",
     fill=PAD_ED, limit=7 * CW)
text("→ padding 6개 + 실제 1개를 센다. 완전히 틀림.", x0 + 8 * CW, y1 + 92,
     size=11.5, weight="600", fill=PAD_ED, anchor="start", limit=420)

y2 = AY + 216
taprow(y2, list(range(6, 13)) + [None] * 6, "6칸 shift 후", "activation",
       hi_lo=(0, 6))
taprow(y2 + 38, list(range(6, 13)) + [None] * 6, "", "weight", wrow=True)
text("mask = 낮은 7비트", x0 + 3.5 * CW, y2 + 92, size=11.5, weight="600",
     fill=OK_ED, limit=7 * CW)
text("→ 실제 7개만. t6 이 w6 과 만난다.", x0 + 8 * CW, y2 + 92, size=11.5,
     weight="600", fill=OK_ED, anchor="start", limit=420)

box(880, AY + 92, W - 24 - 880 - 16, 172, "#fffbf0", W_ED, r=9, sw=1.4)
text("둘 다 shift 해야 한다", 880 + 16, AY + 116, size=12.5, weight="700",
     anchor="start", limit=300)
for j, s in enumerate([
        "activation 만 옮기면 t6 이 bit 0 으로",
        "가서 w0 과 만난다 — 짝이 어긋난다.", "",
        "합은 여전히 ±1 의 합이라 값이",
        "그럴듯하다. 가장자리 프레임만",
        "조용히 틀린다."]):
    text(s, 880 + 16, AY + 140 + j * 19, size=11, anchor="start", limit=300)

# ══ B. why 32 ═════════════════════════════════════════════════════════════
BY = 432
box(24, BY, W - 48, 236, "#ffffff", LINE, r=12, sw=1.2)
text("B.  word 가 왜 32비트이고, 왜 32채널이 한꺼번에 필요한가", 44, BY + 25,
     size=14.5, weight="700", anchor="start")

box(44, BY + 46, 560, 78, "#ffffff", LINE, r=8, sw=1.2)
text("왜 32인가 — 관례다", 60, BY + 68, size=12.5, weight="700",
     anchor="start", limit=300)
text("FPGA BRAM 이 36비트(32+parity), 버스가 32/64비트라 딱 맞는다.",
     60, BY + 90, size=11.5, anchor="start", limit=530)
text("요구사항이 아니라 선택이다. 64로 바꿔도 된다 (pack.py WORD_BITS).",
     60, BY + 110, size=11, fill=MUTE, anchor="start", limit=530)

box(44, BY + 134, 560, 84, OK_BG, OK_ED, r=8)
text("왜 한꺼번에 — 이게 binary network 의 존재 이유다", 60, BY + 156,
     size=12.5, weight="700", anchor="start", limit=530)
text("XNOR + popcount 한 번에 곱셈-누산 32개가 끝난다. 곱셈기 0개로.",
     60, BY + 178, size=11.5, anchor="start", limit=530)
text("128채널 pointwise: 비트 하나씩이면 128 사이클, word 단위면 4 사이클.",
     60, BY + 198, size=11.5, fill=MUTE, anchor="start", limit=530)

px = 636
box(px, BY + 46, W - 24 - px - 16, 172, "#ffffff", LINE, r=8, sw=1.2)
text("그래서 packing 이득은 pointwise 에 몰린다", px + 16, BY + 68, size=12.5,
     weight="700", anchor="start", limit=W - px - 50)
rows = [("pointwise", "128 terms", "4 word", "32× 병렬이 그대로"),
        ("depthwise", "13 terms", "1 word", "32칸 중 13칸만 씀")]
text(f"{'conv':<12}{'항 수':<12}{'word':<10}", px + 16, BY + 94, size=11,
     fill=MUTE, anchor="start", font=MONO, limit=W - px - 50)
for j, (a, b, c, d) in enumerate(rows):
    y = BY + 116 + j * 40
    text(a, px + 16, y, size=11.5, weight="600", anchor="start", limit=110)
    text(b, px + 124, y, size=11.5, anchor="start", limit=90)
    text(c, px + 224, y, size=11.5, anchor="start", limit=80)
    text(d, px + 16, y + 18, size=11, fill=MUTE, anchor="start",
         limit=W - px - 50)
text("CLAUDE.md 3.4 의 「두 PE 를 하나로 합치지 말 것」이 이 얘기다.",
     px + 16, BY + 202, size=11, fill=MUTE, anchor="start", limit=W - px - 50)

# ══ C. line buffer ════════════════════════════════════════════════════════
CY = 684
box(24, CY, W - 48, 322, "#ffffff", LINE, r=12, sw=1.2)
text("C.  line buffer — 13프레임을 붙들고 있는 shift register", 44, CY + 25,
     size=14.5, weight="700", anchor="start")
text("depthwise 는 가로 13칸이 필요한데 메모리에는 세로로 들어 있다. "
     "13개 word 를 레지스터에 들고 있으면 gather 가 공짜가 된다.",
     44, CY + 44, size=11.5, fill=MUTE, anchor="start", limit=900)

SW_, SH_ = 62, 34
sx = 150
STEPS = [("t=0", [None] * 6 + [0, 1, 2, 3, 4, 5, 6]),
         ("t=1", [None] * 5 + [0, 1, 2, 3, 4, 5, 6, 7]),
         ("t=7", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])]
for i, (lab, frames) in enumerate(STEPS):
    y = CY + 74 + i * 62
    text(lab, sx - 16, y + 22, size=12, weight="700", anchor="end", limit=60)
    for j, fr in enumerate(frames):
        x = sx + j * SW_
        if fr is None:
            box(x, y, SW_ - 4, SH_, PAD_BG, PAD_ED, r=3, sw=1.2, track=False)
            text("pad", x + (SW_ - 4) / 2, y + 22, size=10, fill=PAD_ED,
                 limit=SW_ - 10)
        else:
            box(x, y, SW_ - 4, SH_, VAL_BG, VAL_ED, r=3, sw=1.2, track=False)
            text(f"f{fr}", x + (SW_ - 4) / 2, y + 22, size=11, font=MONO,
                 limit=SW_ - 10)
    if i < 2:
        arrow(sx + 13 * SW_ + 6, y + SH_ / 2, sx + 13 * SW_ + 34, y + SH_ / 2)
text("각 칸 = word 하나 = 그 프레임의 전 채널(128비트)", sx + 6.5 * SW_,
     CY + 68, size=11, fill=MUTE, limit=13 * SW_)
text("새 프레임이 오른쪽으로 들어오고 가장 오래된 것이 왼쪽으로 빠진다",
     sx + 6.5 * SW_, CY + 264, size=11.5, weight="600", limit=13 * SW_)

box(44, CY + 282, W - 88, 28, OK_BG, OK_ED, r=7, sw=1.2)
text("channel c 의 13-bit tap 벡터 = 13개 레지스터에서 비트 c 만 뽑은 것 — "
     "전부 이미 들고 있으니 조합논리 한 방, 추가 사이클 0.  그리고 "
     "「pad 칸이 몇 개인가」가 곧 A의 shift 량이다.",
     W / 2, CY + 300, size=11.5, limit=W - 110)

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

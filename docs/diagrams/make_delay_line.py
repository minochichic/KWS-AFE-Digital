"""26_delay_line.svg -- what a delay line and a tap actually are.

rtl/README.md defines both in a sentence each, which is not enough: they are
physical arrangements, and the words only land once you see the samples moving
through the registers.

  A. a delay line is a chain of registers; each stage holds an older sample
  B. a tap is a wire pulled out at a stage -- taps x coefficients, summed, IS
     the convolution
  C. our slots hold a whole 128-channel frame, so one tap is 128 bits and the
     gather picks one channel out of each

Illustrated with K=5 for legibility; the design uses K=13.

Run:  python3 docs/diagrams/make_delay_line.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "26_delay_line.svg"

W, H = 1240, 1000
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
REG_BG, REG_ED = "#ffffff", "#8a93a6"
HOT_BG, HOT_ED = "#e4eefb", "#2f6fd0"      # the sample we follow
W_BG, W_ED = "#fdf3e3", "#c9902a"          # coefficients
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"

KDEMO = 5          # drawn; the real design uses 13

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


def arrow(x1, y1, x2, y2, color=MUTE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d} marker-end="url(#a)"/>')


def line(x1, y1, x2, y2, color=MUTE, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d}/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("지연선(delay line)과 tap", W / 2, 36, size=21, weight="700", limit=W - 60)
text(f"그림은 K={KDEMO} 로 그렸다. 실제 설계는 K=13 이고, 나머지는 똑같다.",
     W / 2, 58, size=12.5, fill=MUTE, limit=W - 60)

# ══ A. what a delay line is ═══════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 288, "#ffffff", LINE, r=12, sw=1.2)
text("A.  지연선 = 레지스터를 줄지어 놓은 것", 44, AY + 25, size=14.5,
     weight="700", anchor="start")
text("새 샘플이 한쪽으로 들어오고, 한 칸씩 밀린다. 각 칸은 「몇 스텝 전의 "
     "샘플」을 들고 있다.", 44, AY + 44, size=11.5, fill=MUTE, anchor="start",
     limit=800)

CW, CH = 74, 40
gx = 300
STEPS = [("t=0", [None, None, None, None, "A"]),
         ("t=1", [None, None, None, "A", "B"]),
         ("t=2", [None, None, "A", "B", "C"])]
for r, (lab, cells) in enumerate(STEPS):
    y = AY + 78 + r * 54
    text(lab, gx - 24, y + 26, size=12, weight="700", anchor="end", limit=60)
    for j, v in enumerate(cells):
        x = gx + j * CW
        hot = (v == "A")
        box(x, y, CW - 6, CH, HOT_BG if hot else REG_BG,
            HOT_ED if hot else REG_ED, track=False)
        text(v if v else "·", x + (CW - 6) / 2, y + 26, size=13,
             font=MONO, weight="700" if hot else "400",
             fill=HOT_ED if hot else MUTE, limit=CW - 14)
    if r == 0:
        arrow(gx + KDEMO * CW + 4, y + 20, gx + KDEMO * CW - 24, y + 20)
        text("새 샘플", gx + KDEMO * CW + 12, y + 24, size=11, weight="600",
             anchor="start", limit=110)

for j in range(KDEMO):
    text(f"슬롯 {j}", gx + j * CW + (CW - 6) / 2, AY + 244, size=10.5,
         fill=MUTE, limit=CW - 6)
text("오래된 것 ←", gx - 24, AY + 244, size=10.5, fill=MUTE, anchor="end",
     limit=110)
text("→ 새 것", gx + KDEMO * CW + 12, AY + 244, size=10.5, fill=MUTE,
     anchor="start", limit=110)
text("A 를 따라가 보면: 들어온 뒤 매 스텝 한 칸씩 왼쪽으로 간다. "
     "슬롯 번호가 곧 「몇 스텝 전인가」다.",
     W / 2, AY + 272, size=11.5, weight="600", limit=W - 90)

# ══ B. tap ════════════════════════════════════════════════════════════════
BY = 384
box(24, BY, W - 48, 300, "#ffffff", LINE, r=12, sw=1.2)
text("B.  tap = 각 칸에서 선을 딴 것", 44, BY + 25, size=14.5, weight="700",
     anchor="start")
text("옛 아날로그 지연선에서 중간중간 전선을 뽑아내던 데서 온 말이다. "
     "뽑은 선마다 계수를 곱해 전부 더하면 — 그게 convolution 이다.",
     44, BY + 44, size=11.5, fill=MUTE, anchor="start", limit=900)

ty = BY + 74
tx = 300
for j in range(KDEMO):
    x = tx + j * CW
    box(x, ty, CW - 6, CH, REG_BG, REG_ED, track=False)
    text(f"x[t{j-2:+d}]" if j != 2 else "x[t]", x + (CW - 6) / 2, ty + 26,
         size=11, font=MONO, limit=CW - 12)
    # the tap wire
    line(x + (CW - 6) / 2, ty + CH, x + (CW - 6) / 2, ty + 66, HOT_ED, 1.6)
    box(x + 6, ty + 66, CW - 18, 30, W_BG, W_ED, r=5, sw=1.2, track=False)
    text(f"× w{j}", x + (CW - 6) / 2, ty + 86, size=11, font=MONO,
         limit=CW - 22)
    line(x + (CW - 6) / 2, ty + 96, x + (CW - 6) / 2, ty + 120, MUTE, 1.4)
line(tx + (CW - 6) / 2, ty + 120, tx + (KDEMO - 1) * CW + (CW - 6) / 2,
     ty + 120, MUTE, 1.4)
mid = tx + (KDEMO - 1) * CW / 2 + (CW - 6) / 2
arrow(mid, ty + 120, mid, ty + 146)
box(mid - 70, ty + 146, 140, 38, OK_BG, OK_ED, r=8, track=False)
text("Σ  (전부 더함)", mid, ty + 170, size=12, weight="700", limit=130)

text("tap", tx - 24, ty + 86, size=12, weight="700", fill=HOT_ED, anchor="end",
     limit=60)
text("계수", tx - 24, ty + 106, size=10.5, fill=MUTE, anchor="end", limit=60)

box(880, BY + 74, W - 24 - 880 - 16, 174, "#fffbf0", W_ED, r=9)
text("이게 곧 convolution 이다", 896, BY + 96, size=12, weight="700",
     anchor="start", limit=300)
text("out[t] = Σ w[j] · x[t−2+j]", 896, BY + 122, size=12, font=MONO,
     anchor="start", limit=300)
for k, s in enumerate([
        "우리 경우 K=13 이라 앞뒤 6칸,",
        "계수는 w_rom[ch] 의 13 비트.",
        "",
        "곱셈이 없는 이유: x 도 w 도 ±1",
        "이라 곱은 「부호가 같은가」고,",
        "XNOR 이 그걸 센다."]):
    text(s, 896, BY + 146 + k * 18, size=10.5, anchor="start", limit=310)

# ══ C. our case ═══════════════════════════════════════════════════════════
CY = 702
box(24, CY, W - 48, 268, "#ffffff", LINE, r=12, sw=1.2)
text("C.  우리 슬롯 하나는 프레임 전체 — 128 채널이 들어 있다", 44, CY + 25,
     size=14.5, weight="700", anchor="start")
text("그래서 tap 하나가 128 비트다. 채널 하나짜리 필터를 돌리려면 각 tap 에서 "
     "그 채널의 비트만 뽑아야 한다 — 그게 gather.", 44, CY + 44, size=11.5,
     fill=MUTE, anchor="start", limit=940)

sx, sy = 300, CY + 74
SW_, SH_ = 74, 118
NB = 7                                   # bit rows drawn per slot
HOT_ROW = 3
for j in range(KDEMO):
    x = sx + j * SW_
    box(x, sy, SW_ - 10, SH_, "#fbfcfe", REG_ED, r=6, track=False)
    for b in range(NB):
        yy = sy + 8 + b * 15
        hot = (b == HOT_ROW)
        box(x + 8, yy, SW_ - 26, 12, HOT_BG if hot else "#ffffff",
            HOT_ED if hot else GRID, r=2, sw=1 if hot else 0.8, track=False)
    text(f"tap {j}", x + (SW_ - 10) / 2, sy + SH_ + 16, size=10.5, fill=MUTE,
         limit=SW_ - 10)
text("128 비트", sx - 20, sy + 20, size=10.5, fill=MUTE, anchor="end", limit=80)
text("채널 ch", sx - 20, sy + 8 + HOT_ROW * 15 + 10, size=11, weight="700",
     fill=HOT_ED, anchor="end", limit=80)

for j in range(KDEMO):
    x = sx + j * SW_ + (SW_ - 10) / 2
    line(x, sy + 8 + HOT_ROW * 15 + 12, x, sy + SH_ + 30, HOT_ED, 1.5)
gy2 = sy + SH_ + 30
line(sx + (SW_ - 10) / 2, gy2, sx + (KDEMO - 1) * SW_ + (SW_ - 10) / 2, gy2,
     HOT_ED, 1.5)
mid2 = sx + (KDEMO - 1) * SW_ / 2 + (SW_ - 10) / 2
arrow(mid2, gy2, mid2, gy2 + 24, HOT_ED)
box(mid2 - 96, gy2 + 24, 192, 34, HOT_BG, HOT_ED, r=8, track=False)
text(f"taps[{KDEMO-1}:0]  —  {KDEMO} 비트", mid2, gy2 + 46, size=11.5,
     weight="700", font=MONO, limit=182)

box(880, CY + 74, W - 24 - 880 - 16, 158, "#f7fbf8", OK_ED, r=9)
text("gather 가 공짜인 이유", 896, CY + 96, size=12, weight="700",
     anchor="start", limit=300)
for k, s in enumerate([
        "13 개 슬롯은 전부 레지스터다.",
        "이미 손에 들고 있으니 비트를",
        "뽑는 건 배선일 뿐 — 조합논리",
        "한 방, 추가 사이클 0.",
        "",
        "메모리였다면 13 번 읽어야 했다.",
        "line buffer 를 두는 이유가 이것."]):
    text(s, 896, CY + 120 + k * 18, size=10.5, anchor="start", limit=310)

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

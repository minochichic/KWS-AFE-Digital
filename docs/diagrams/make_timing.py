"""27_timing.svg -- the two time scales as waveforms.

Every other picture in this set is structural. This one is temporal, because
the question it answers is temporal: the delay-line diagram's t=0,1,2 read as
clock edges, and they are not -- they are frames, 1613 clocks apart.

  A. frame scale   : the line buffer advances once per frame, and only there
  B. clock scale   : inside one frame the buffer is FROZEN and the channel
                     index sweeps; three clocks per channel
  C. what changes at which rate, and what never changes

Numbers are the measured kws_dw_conv / kws_block schedule.

Run:  python3 docs/diagrams/make_timing.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "27_timing.svg"

W, H = 1240, 1010
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
SIG = "#2f6fd0"                     # waveform ink
FROZEN_BG, FROZEN_ED = "#e4eefb", "#2f6fd0"
MOVE_BG, MOVE_ED = "#fdf3e3", "#c9902a"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"

CLKS_PER_FRAME = 1613               # measured, kws_block
CYC_PER_CH = 3                      # kws_dw_conv: S_START, S_FEED, S_TAKE

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


def text(s, x, y, size=11, fill=INK, anchor="middle", weight="400",
         font=FONT, limit=None):
    TEXTS.append((s, x, y, size, anchor))
    if limit is not None and tw(s, size) > limit:
        raise ValueError(f"overflows {limit:.0f}px: {s!r} -> {tw(s, size):.0f}px")
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}'
        f'</text>')


def poly(pts, color=SIG, sw=1.8):
    d = " ".join(f"{x},{y}" for x, y in pts)
    add(f'<polyline points="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}"/>')


LANE_H = 26
HI, LO = 5, 20                       # offsets inside a lane


def lane_label(x, y, name, sub=None):
    text(name, x - 12, y + 15, size=11, weight="600", anchor="end", font=MONO,
         limit=130)
    if sub:
        text(sub, x - 12, y + 27, size=9, fill=MUTE, anchor="end", limit=130)


def wave_clk(x, y, n, step):
    pts = []
    for i in range(n):
        pts += [(x + i * step, y + LO), (x + i * step, y + HI),
                (x + i * step + step / 2, y + HI),
                (x + i * step + step / 2, y + LO)]
    pts.append((x + n * step, y + LO))
    poly(pts)


def wave_bit(x, y, pattern, step):
    """pattern: list of 0/1, one per cycle."""
    pts = [(x, y + (HI if pattern[0] else LO))]
    for i, v in enumerate(pattern):
        yy = y + (HI if v else LO)
        pts.append((x + i * step, yy))
        pts.append((x + (i + 1) * step, yy))
    poly(pts)


def wave_bus(x, y, segs, step, bg=None, ed=None):
    """segs: list of (label, n_cycles). Drawn as boxes with a notch."""
    cx = x
    for lab, n in segs:
        wdt = n * step
        box(cx + 2, y + HI - 1, wdt - 4, LO - HI + 2, bg or "#ffffff",
            ed or SIG, r=3, sw=1.2, track=False)
        if lab:
            text(lab, cx + wdt / 2, y + 16, size=9.5, font=MONO,
                 limit=wdt - 8)
        cx += wdt


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("타이밍 — 클럭과 프레임, 두 척도", W / 2, 36, size=21, weight="700",
     limit=W - 60)
text(f"지연선 그림의 t=0,1,2 는 프레임이지 클럭이 아니다. "
     f"프레임 하나가 {CLKS_PER_FRAME:,} 클럭이다.", W / 2, 58, size=12.5,
     fill=MUTE, limit=W - 60)

# ══ A. frame scale ════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 250, "#ffffff", LINE, r=12, sw=1.2)
text("A.  프레임 척도 — 지연선은 여기서만 움직인다", 44, AY + 25, size=14.5,
     weight="700", anchor="start")
text(f"한 칸이 프레임 하나 = {CLKS_PER_FRAME:,} 클럭 ≈ 16 µs (@100 MHz).",
     44, AY + 44, size=11.5, fill=MUTE, anchor="start", limit=700)

FX, FSTEP, NFR = 200, 220, 4
y = AY + 62
# quarter-frame resolution, so the brief gaps at the boundary are visible --
# busy really does drop there, and that is when the next push is allowed
lane_label(FX, y, "in_push", "프레임 투입")
wave_bit(FX, y, [1, 0, 0, 0] * NFR, FSTEP / 4)
y += LANE_H + 4
lane_label(FX, y, "busy", "계산 중")
wave_bit(FX, y, [1, 1, 1, 0] * NFR, FSTEP / 4)
y += LANE_H + 4
lane_label(FX, y, "fbuf", "line buffer")
def _rng(i):                       # the 13 frames in the buffer, no double signs
    lo, hi = i - 1 - 6, i - 1 + 6
    f = lambda v: "t" if v == 0 else f"t{v:+d}"
    return f"x[{f(lo)} … {f(hi)}]"
wave_bus(FX, y, [(_rng(i), 1) for i in range(NFR)], FSTEP, MOVE_BG, MOVE_ED)
y += LANE_H + 4
lane_label(FX, y, "out_valid", "프레임 완료")
wave_bit(FX, y, [0, 0, 0, 1] * NFR, FSTEP / 4)

for i in range(NFR + 1):
    add(f'<line x1="{FX + i * FSTEP}" y1="{AY + 58}" x2="{FX + i * FSTEP}" '
        f'y2="{AY + 196}" stroke="{GRID}" stroke-width="1"/>')
for i in range(NFR):
    text(f"프레임 t{i-1:+d}" if i != 1 else "프레임 t",
         FX + i * FSTEP + FSTEP / 2, AY + 212, size=10.5, fill=MUTE,
         limit=FSTEP - 10)
text("← 지연선이 한 칸 미는 순간은 여기, 프레임 경계뿐이다", W / 2, AY + 236,
     size=11.5, weight="600", limit=W - 90)

# ══ B. clock scale ════════════════════════════════════════════════════════
BY = 346
box(24, BY, W - 48, 346, "#ffffff", LINE, r=12, sw=1.2)
text("B.  한 프레임 안을 확대 — 여기서 지연선은 얼어 있다", 44, BY + 25,
     size=14.5, weight="700", anchor="start")
text(f"채널 하나에 {CYC_PER_CH} 클럭. 채널 인덱스만 훑고, tap 은 그대로다 "
     f"(kws_dw_conv).", 44, BY + 44, size=11.5, fill=MUTE, anchor="start",
     limit=800)

CX, CSTEP, NCY = 200, 78, 12
y = BY + 64
lane_label(CX, y, "clk", "10 ns")
wave_clk(CX, y, NCY, CSTEP)
y += LANE_H + 4
lane_label(CX, y, "st", "FSM")
wave_bus(CX, y, [(s, 1) for s in
                 ["START", "FEED", "TAKE"] * (NCY // CYC_PER_CH)], CSTEP)
y += LANE_H + 4
lane_label(CX, y, "ch", "채널 인덱스")
wave_bus(CX, y, [(f"{i}", CYC_PER_CH) for i in range(NCY // CYC_PER_CH)],
         CSTEP, OK_BG, OK_ED)
y += LANE_H + 4
lane_label(CX, y, "taps", "gather 결과")
wave_bus(CX, y, [(f"비트 {i} ×13", CYC_PER_CH)
                 for i in range(NCY // CYC_PER_CH)], CSTEP)
y += LANE_H + 4
lane_label(CX, y, "mac_feed", "word 투입")
wave_bit(CX, y, [0, 1, 0] * (NCY // CYC_PER_CH), CSTEP)
y += LANE_H + 4
lane_label(CX, y, "acc", "2P−N")
wave_bus(CX, y, sum([[("", 2), (f"a{i}", 1)]
                     for i in range(NCY // CYC_PER_CH)], []), CSTEP)
y += LANE_H + 4
lane_label(CX, y, "out_frame", "비트가 채워짐")
wave_bus(CX, y, [(f"[{i}] 확정", CYC_PER_CH)
                 for i in range(NCY // CYC_PER_CH)], CSTEP, OK_BG, OK_ED)
y += LANE_H + 6
lane_label(CX, y, "fbuf", "line buffer")
box(CX + 2, y + HI - 1, NCY * CSTEP - 4, LO - HI + 2, FROZEN_BG, FROZEN_ED,
    r=3, sw=1.6, track=False)
text("변하지 않음 — 같은 13 프레임을 128 채널이 나눠 쓴다", CX + NCY * CSTEP / 2,
     y + 16, size=10, weight="600", fill=FROZEN_ED, limit=NCY * CSTEP - 20)

for i in range(0, NCY + 1, CYC_PER_CH):
    add(f'<line x1="{CX + i * CSTEP}" y1="{BY + 60}" x2="{CX + i * CSTEP}" '
        f'y2="{BY + 296}" stroke="{GRID}" stroke-width="1"/>')
text(f"… 채널 127 까지 반복 → 128 × {CYC_PER_CH} = 384 클럭", CX + 6 * CSTEP,
     BY + 320, size=11.5, weight="600", limit=560)

# ══ C. summary ════════════════════════════════════════════════════════════
CY = 708
box(24, CY, W - 48, 282, "#ffffff", LINE, r=12, sw=1.2)
text("C.  무엇이 어느 주기로 바뀌나", 44, CY + 25, size=14.5, weight="700",
     anchor="start")

ROWS = [("클럭마다", "10 ns",
         "st (FSM) · ch/co 채널 인덱스 · wi/wa 워드 주소 · MAC 내부 P",
         "가장 빠르다. 하드웨어 하나를 채널마다 갈아 끼우는 것이 folded 다.",
         OK_BG, OK_ED),
        ("채널마다", f"{CYC_PER_CH} 클럭",
         "taps (gather 결과) · acc · out_frame 의 비트 하나",
         "tap 은 그대로고 뽑는 비트 위치만 바뀐다.",
         "#ffffff", LINE),
        ("프레임마다", f"{CLKS_PER_FRAME:,} 클럭 ≈ 16 µs",
         "fbuf 슬롯 · valid · xdly — 지연선 전부",
         "여기서만 창이 한 칸 움직인다. conv 의 시간축이 이것.",
         MOVE_BG, MOVE_ED),
        ("절대 안 바뀜", "—",
         "가중치 ROM · threshold ROM · 슬롯↔tap 대응",
         "재학습하면 .hex 만 갈아 끼운다 (docs/ICD.md 6).",
         "#f2f4f7", "#a8b0bf")]
for i, (nm, per, what, why, bg, ed) in enumerate(ROWS):
    yy = CY + 46 + i * 58
    box(44, yy, W - 88, 50, bg, ed, r=8, sw=1.2)
    text(nm, 60, yy + 21, size=12, weight="700", anchor="start", limit=130)
    text(per, 60, yy + 39, size=10.5, fill=MUTE, anchor="start", font=MONO,
         limit=150)
    text(what, 216, yy + 21, size=11.5, anchor="start", limit=620)
    text(why, 216, yy + 39, size=10.5, fill=MUTE, anchor="start", limit=620)

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

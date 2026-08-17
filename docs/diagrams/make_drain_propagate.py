"""29_drain_propagate.svg -- why a flush does not travel downstream.

"drain does not propagate" is the sentence that keeps not landing, and it is not
obvious: a flush looks like it should flow through the pipeline the way data
does. It does not, and the reason is one line -- a stage that receives a flush
still emits REAL data, so the stage after it never sees a flush at all.

  A. what a flush is for: the buffer must shift, and something must enter
  B. the crux: flush in, real out
  C. the timeline, computed here so it cannot be wrong
  D. and why a plane buffer makes the whole question disappear

Run:  python3 docs/diagrams/make_drain_propagate.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "29_drain_propagate.svg"

W, H = 1240, 1126
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
REAL_BG, REAL_ED = "#e4eefb", "#2f6fd0"
FLUSH_BG, FLUSH_ED = "#fdf0ef", "#cc6155"
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"
W_BG, W_ED = "#fdf3e3", "#c9902a"

T, PAD = 64, 6


def timeline():
    """(push, sub0 in, sub0 out, s1 in, block out) -- the real schedule."""
    rows, s1_push = [], 0
    for i in range(T + 2 * PAD):
        s0_in = "real" if i < T else "flush"
        s0_out = i - PAD if (i >= PAD and i - PAD < T) else None
        if s0_out is not None:
            s1_in, idx = "real", s1_push
            s1_push += 1
        elif i >= T:
            s1_in, idx = "flush", s1_push
            s1_push += 1
        else:
            s1_in, idx = "—", None
        out = (idx - PAD) if (idx is not None and PAD <= idx < T + PAD) else None
        rows.append((i, s0_in, s0_out, s1_in, out))
    return rows


ROWS = timeline()
N_OUT = sum(1 for r in ROWS if r[4] is not None)
assert N_OUT == T, f"{N_OUT} outputs, expected {T}"

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

text("flush 는 하류로 흘러가지 않는다", W / 2, 36, size=21, weight="700",
     limit=W - 60)
text("데이터처럼 파이프를 타고 갈 것 같지만 안 간다. 이유는 한 줄이다 — "
     "flush 를 받은 단도 real 을 내놓는다.", W / 2, 58, size=12.5, fill=MUTE,
     limit=W - 60)

# ══ A. what a flush is for ════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 200, "#ffffff", LINE, r=12, sw=1.2)
text("A.  flush 는 「데이터 없이 한 칸 밀어라」는 지시다", 44, AY + 25,
     size=14.5, weight="700", anchor="start")
text("line buffer 가 밀려야 마지막 출력이 나온다. 밀려면 뭐라도 들어가야 하는데, "
     "입력은 끝났다. 그래서 빈 칸을 넣는다.", 44, AY + 44, size=11.5,
     fill=MUTE, anchor="start", limit=940)

CWD, CHT = 62, 34
ax, ay = 200, AY + 74
CELLS = ["f58", "f59", "f60", "f61", "f62", "f63", "빈"]
for j, v in enumerate(CELLS):
    x = ax + j * CWD
    fl = (v == "빈")
    box(x, ay, CWD - 5, CHT, FLUSH_BG if fl else REAL_BG,
        FLUSH_ED if fl else REAL_ED, track=False)
    text(v, x + (CWD - 5) / 2, ay + 22, size=11, font=MONO,
         fill=FLUSH_ED if fl else INK, limit=CWD - 12)
arrow(ax + len(CELLS) * CWD + 6, ay + 17, ax + len(CELLS) * CWD - 18, ay + 17,
      FLUSH_ED)
text("flush", ax + len(CELLS) * CWD + 14, ay + 21, size=11, weight="700",
     fill=FLUSH_ED, anchor="start", limit=80)
text("(슬롯 13개 중 오른쪽 일부만 그림)", ax + 3.5 * CWD, ay + CHT + 18,
     size=10, fill=MUTE, limit=320)

box(ax + 560, ay - 8, W - 24 - (ax + 560) - 16, 74, W_BG, W_ED, r=8)
text("flush = 데이터가 아니다", ax + 576, ay + 14, size=11.5, weight="700",
     anchor="start", limit=280)
text("들어간 칸은 「무효」로 표시되고 (valid=0),", ax + 576, ay + 34, size=10.5,
     anchor="start", limit=300)
text("합에서 빠진다. 미는 것 자체가 목적이다.", ax + 576, ay + 52, size=10.5,
     anchor="start", limit=300)

text(f"{PAD}번 밀어야 남은 출력 {PAD}개(t=58…63)가 나온다.", W / 2, AY + 176,
     size=11.5, weight="600", limit=W - 90)

# ══ B. the crux ═══════════════════════════════════════════════════════════
BY = 294
box(24, BY, W - 48, 244, "#fffbf0", W_ED, r=12, sw=1.5)
text("B.  ★ 여기가 핵심 — flush 가 들어가는데 real 이 나온다", 44, BY + 25,
     size=14.5, weight="700", anchor="start")

bx, by = 70, BY + 56
box(bx, by, 150, 40, FLUSH_BG, FLUSH_ED, r=7)
text("flush 투입", bx + 75, by + 25, size=11.5, weight="600", limit=140)
arrow(bx + 152, by + 20, bx + 196, by + 20)

box(bx + 198, by - 14, 300, 96, "#ffffff", REAL_ED, r=8)
text("sub0 의 line buffer", bx + 348, by + 6, size=11.5, weight="700",
     limit=280)
text("[f52 … f63][빈]", bx + 348, by + 28, size=11, font=MONO, limit=280)
text("중앙 tap = f58  →  real", bx + 348, by + 50, size=11.5, weight="700",
     fill=REAL_ED, limit=280)
text("중앙이 real 이면 출력한다", bx + 348, by + 70, size=10, fill=MUTE,
     limit=280)
arrow(bx + 500, by + 20, bx + 544, by + 20)

box(bx + 546, by, 150, 40, REAL_BG, REAL_ED, r=7)
text("f58 출력 (real)", bx + 621, by + 25, size=11.5, weight="600", limit=140)
arrow(bx + 698, by + 20, bx + 742, by + 20)

box(bx + 744, by, 190, 40, REAL_BG, REAL_ED, r=7)
text("s1 은 real 을 받는다", bx + 839, by + 25, size=11.5, weight="600",
     limit=180)

box(70, BY + 150, W - 140, 78, FLUSH_BG, FLUSH_ED, r=9)
text("그래서 s1 은 flush 를 한 번도 못 본다", 86, BY + 172, size=12.5,
     weight="700", anchor="start", limit=460)
text("sub0 은 「중앙 tap 이 real 일 때」만 출력한다. 자기 drain 중에 나오는 "
     "출력도 진짜 데이터다.", 86, BY + 192, size=11, anchor="start", limit=1000)
text("그런데 s1 도 line buffer 가 있어서 자기 flush 6개가 필요하다. "
     "아무도 안 준다 → 블록이 직접 만들어 넣는다.",
     86, BY + 210, size=11, weight="600", anchor="start", limit=1000)

# ══ C. the timeline ═══════════════════════════════════════════════════════
CY = 554
box(24, CY, W - 48, 344, "#ffffff", LINE, r=12, sw=1.2)
text(f"C.  타임라인 — 블록 push {len(ROWS)}회 → 출력 {N_OUT}개", 44, CY + 25,
     size=14.5, weight="700", anchor="start")

COLS = [("블록 push", 110), ("sub0 받음", 150), ("sub0 내놓음", 160),
        ("s1 받음", 150), ("블록 출력", 150)]
tx, ty = 100, CY + 50
cx = tx
for nm, wd in COLS:
    text(nm, cx + wd / 2, ty, size=10.5, fill=MUTE, weight="600", limit=wd - 6)
    cx += wd

SHOW = [(0, "0 … 5"), (6, "6"), (12, "12"), (63, "63"),
        (64, "64 … 69"), (70, "70 … 75")]
for r, (i, lab) in enumerate(SHOW):
    _, s0i, s0o, s1i, out = ROWS[i]
    yy = ty + 18 + r * 38
    hot = 64 <= i <= 69
    box(tx, yy, sum(w for _, w in COLS), 32,
        "#fff6e8" if hot else "#ffffff", W_ED if hot else GRID,
        r=6, sw=1.6 if hot else 1)
    cx = tx
    vals = [lab, s0i, (f"f{s0o}" if s0o is not None else "없음"), s1i,
            (f"t={out}" if out is not None else "—")]
    if i == 0:
        vals = [lab, "real", "없음", "—", "—"]
    if i == 64:
        vals = [lab, "flush", f"f58 … f63", "real", "t=52 … 57"]
    if i == 70:
        vals = [lab, "flush", "없음", "flush ←주입", "t=58 … 63"]
    if i == 6:
        vals = [lab, "real", "f0", "real", "—"]
    if i == 12:
        vals = [lab, "real", "f6", "real", "t=0"]
    for (nm, wd), v in zip(COLS, vals):
        col = INK
        if v.startswith("flush"):
            col = FLUSH_ED
        elif v.startswith("real") or v.startswith("f") or v.startswith("t="):
            col = REAL_ED if not v.startswith("t=") else OK_ED
        text(v, cx + wd / 2, yy + 21, size=11,
             font=MONO if v[0].isdigit() or v[0] in "ft" else FONT,
             fill=col, weight="700" if hot else "400", limit=wd - 8)
        cx += wd

text("노란 두 줄이 전부다. push 64~69 는 flush 를 넣는데 sub0 이 real 을 내놓아 "
     "s1 은 여전히 real 을 받고,", W / 2, CY + 310, size=11.5, weight="600",
     limit=W - 90)
text("push 70~75 에서야 블록이 s1 용 flush 를 만들어 넣는다. "
     "그래서 caller 는 T + 2·PAD 번 밀어야 한다.", W / 2, CY + 330, size=11,
     fill=MUTE, limit=W - 90)

# ══ D. plane ══════════════════════════════════════════════════════════════
DY = 916
box(24, DY, W - 48, 186, "#f7fbf8", OK_ED, r=12, sw=1.4)
text("D.  평면 버퍼에서는 이 질문이 사라진다", 44, DY + 25, size=14.5,
     weight="700", anchor="start")

for i, (a, b, bg, ed) in enumerate([
        ("스트리밍", "「이 프레임은 padding 인가?」 → 언제 flush 가 도착했는가",
         FLUSH_BG, FLUSH_ED),
        ("평면 버퍼", "「이 프레임은 padding 인가?」 → 주소가 범위 안인가",
         OK_BG, OK_ED)]):
    yy = DY + 46 + i * 44
    box(60, yy, W - 120, 38, bg, ed, r=8, sw=1.2)
    text(a, 78, yy + 24, size=12, weight="700", anchor="start", limit=150)
    text(b, 236, yy + 24, size=11.5, anchor="start", limit=880)

text("if (addr < 0 || addr > T−1) → padding.   끝이다. flush 도, valid shift "
     "register 도, drain 전파도 없다.", W / 2, DY + 152, size=11.5,
     weight="600", limit=W - 90)
text("타이밍 질문이 주소 질문으로 바뀐다. line buffer 는 남지만, 상류가 밀어넣는 "
     "대신 주소 생성기가 읽어온다.", W / 2, DY + 172, size=11, fill=MUTE,
     limit=W - 90)

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
print(f"timeline verified: {len(ROWS)} pushes -> {N_OUT} outputs")

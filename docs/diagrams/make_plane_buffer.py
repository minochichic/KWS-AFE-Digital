"""28_plane_buffer.svg -- streaming versus a plane buffer, and the middle path.

The question kws_top forces: how do twenty-one layers hand activations to each
other? Streaming works inside a layer and is already built, but every junction
between layers needs its latencies matched by hand -- kws_block cost two bugs
doing exactly that, and there are twenty more junctions ahead.

  A. streaming     -- a conveyor. Everything a layer needs, it must be holding.
  B. plane buffer  -- a warehouse. Everything is addressable, so timing goes away.
  C. the middle    -- streaming inside a block, planes between them. Keeps the
                     five verified modules and kws_block untouched.

Plane sizes and the BRAM share are computed from the manifest.

Run:  python3 docs/diagrams/make_plane_buffer.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "diagrams" / "28_plane_buffer.svg"

W, H = 1240, 1024
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
STREAM_BG, STREAM_ED = "#fdf0ef", "#cc6155"      # the conveyor, and its cost
PLANE_BG, PLANE_ED = "#eaf1fb", "#4a7fd0"        # memory
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"
W_BG, W_ED = "#fdf3e3", "#c9902a"

# ---- numbers from the manifest -------------------------------------------
_man = json.loads((ROOT / "rtl" / "gen" / "xl_g12" / "manifest.json").read_text())
T_FR = 64                                        # frames after conv1's stride 2
BIG_CH = max(l["out_ch"] for l in _man["layers"])
BIG_BITS = BIG_CH * T_FR
BRAM_BITS = 16 * 1024 * 1024                     # KC705, 16 Mb
N_PLANES = 3                                     # ping, pong, skip
SHARE = N_PLANES * BIG_BITS / BRAM_BITS * 100

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=8, sw=1.3, dash=None, track=True):
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

text("층 사이를 어떻게 잇나 — 컨베이어 vs 창고", W / 2, 36, size=21,
     weight="700", limit=W - 60)
text("21개 층이 활성을 서로에게 넘기는 방식. 층 안에서는 스트리밍이 맞는데, "
     "층 사이에서는 얘기가 다르다.", W / 2, 58, size=12.5, fill=MUTE,
     limit=W - 60)

# ══ A. streaming ══════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 268, "#ffffff", LINE, r=12, sw=1.2)
text("A.  스트리밍 = 컨베이어 벨트", 44, AY + 25, size=14.5, weight="700",
     anchor="start")
text("활성이 프레임 단위로 흘러간다. 층이 필요한 것은 그 순간 손에 들고 있어야 "
     "한다 — 그게 line buffer 다.", 44, AY + 44, size=11.5, fill=MUTE,
     anchor="start", limit=880)

sx, sy = 70, AY + 70
BW2, BG2 = 150, 44
for i, nm in enumerate(["conv1", "b1", "b2", "b3", "꼬리"]):
    x = sx + i * (BW2 + BG2)
    box(x, sy, BW2, 46, "#ffffff", MUTE, r=8)
    text(nm, x + BW2 / 2, sy + 28, size=12.5, weight="600", font=MONO,
         limit=BW2 - 10)
    if i < 4:
        arrow(x + BW2 + 2, sy + 23, x + BW2 + BG2 - 2, sy + 23)
        text("프레임", x + BW2 + BG2 / 2, sy + 14, size=9.5, fill=MUTE,
             limit=BG2 + 10)

# the cost: every junction needs its latency matched
for i in range(4):
    x = sx + i * (BW2 + BG2) + BW2 + BG2 / 2
    add(f'<line x1="{x}" y1="{sy + 46}" x2="{x}" y2="{sy + 74}" '
        f'stroke="{STREAM_ED}" stroke-width="1.5" stroke-dasharray="3 3"/>')
    box(x - 34, sy + 74, 68, 26, STREAM_BG, STREAM_ED, r=5, sw=1.2, track=False)
    text("지연 맞춤", x, sy + 91, size=9.5, fill="#a3453a", limit=62)

box(70, AY + 190, W - 140, 62, STREAM_BG, STREAM_ED, r=9)
text("접점마다 도착 시각을 손으로 맞춰야 한다", 86, AY + 212, size=12,
     weight="700", anchor="start", limit=420)
text("kws_block 하나에서 이걸로 버그 두 개가 났다 — 지연선 깊이 off-by-one "
     "(103/128 실패), drain 이 하류로 안 전파.",
     86, AY + 232, size=11, anchor="start", limit=980)
text("층이 21개면 그런 접점이 20개다.", 86, AY + 248, size=11, fill=MUTE,
     anchor="start", limit=400)

# ══ B. plane buffer ═══════════════════════════════════════════════════════
BY = 362
box(24, BY, W - 48, 300, "#ffffff", LINE, r=12, sw=1.2)
text("B.  평면 버퍼 = 창고", 44, BY + 25, size=14.5, weight="700",
     anchor="start")
text(f"활성 평면 [채널, 프레임] 을 온칩 메모리에 통째로 둔다. 필요할 때 "
     f"주소로 꺼내 쓴다.", 44, BY + 44, size=11.5, fill=MUTE, anchor="start",
     limit=880)

px, py = 90, BY + 70
for i, (nm, bg, ed) in enumerate([("평면 A", PLANE_BG, PLANE_ED),
                                  ("평면 B", PLANE_BG, PLANE_ED)]):
    x = px + i * 470
    box(x, py, 130, 92, bg, ed, r=8)
    text(nm, x + 65, py + 24, size=12.5, weight="700", font=MONO, limit=120)
    text(f"{BIG_CH}ch × {T_FR}fr", x + 65, py + 44, size=10, fill=MUTE,
         limit=120)
    text(f"{BIG_BITS/8/1024:.2f} KB", x + 65, py + 62, size=11, weight="600",
         limit=120)
    text("주소로 접근", x + 65, py + 80, size=9.5, fill=MUTE, limit=120)

box(px + 178, py + 22, 240, 48, "#ffffff", MUTE, r=8)
text("층 하나 (읽고 → 쓰고)", px + 298, py + 42, size=11.5, weight="600",
     limit=230)
text("kws_block 등", px + 298, py + 60, size=10, fill=MUTE, font=MONO,
     limit=230)
arrow(px + 132, py + 46, px + 176, py + 46)
arrow(px + 420, py + 46, px + 468, py + 46)
add(f'<path d="M{px + 600} {py + 92} L{px + 600} {py + 116} '
    f'L{px + 65} {py + 116} L{px + 65} {py + 94}" fill="none" '
    f'stroke={chr(34)}{PLANE_ED}{chr(34)} stroke-width="1.6" '
    f'stroke-dasharray="5 3" marker-end="url(#a)"/>')
text("다음 층은 방향만 바꿔 재사용 (ping-pong)", px + 330, py + 132, size=11,
     fill=PLANE_ED, weight="600", limit=520)

ROWS = [("「6 프레임 전」 값", "지연선에 붙들고 있어야", "주소 −6 으로 읽으면 끝"),
        ("drain", "손으로 전파", "없음 — 필요한 걸 읽는다"),
        ("residual skip", "12 프레임 지연선", "평면 하나 더"),
        ("kws_top", "21 층 스케줄 맞추기", "층을 순서대로 도는 루프")]
ty = BY + 158
box(70, ty, W - 140, 22 + len(ROWS) * 22, "#ffffff", LINE, r=8, sw=1.2)
text("스트리밍", 470, ty + 17, size=10.5, fill="#a3453a", weight="700",
     limit=120)
text("평면 버퍼", 800, ty + 17, size=10.5, fill=PLANE_ED, weight="700",
     limit=120)
for i, (what, a, b) in enumerate(ROWS):
    yy = ty + 38 + i * 22
    text(what, 86, yy, size=10.5, anchor="start", limit=250)
    text(a, 470, yy, size=10.5, fill=MUTE, limit=280)
    text(b, 800, yy, size=10.5, weight="600", limit=300)

box(70, BY + 268, W - 140, 24, OK_BG, OK_ED, r=7, sw=1.2)
text(f"메모리 비용: 평면 {N_PLANES}개 = {N_PLANES*BIG_BITS/8/1024:.2f} KB, "
     f"KC705 BRAM 16 Mb 의 {SHARE:.2f} %", W / 2, BY + 285, size=11.5,
     weight="600", limit=W - 160)

# ══ C. the middle path ════════════════════════════════════════════════════
CY = 678
box(24, CY, W - 48, 322, "#f7fbf8", OK_ED, r=12, sw=1.4)
text("C.  권고 — 블록 안은 스트리밍, 블록 사이는 평면", 44, CY + 25,
     size=14.5, weight="700", anchor="start")

mx, my = 60, CY + 54
CHAIN = [("AFE", "#ffffff", MUTE), ("conv1", "#ffffff", MUTE),
         ("평면 A", PLANE_BG, PLANE_ED), ("b1", W_BG, W_ED),
         ("평면 B", PLANE_BG, PLANE_ED), ("b2", W_BG, W_ED),
         ("평면 A", PLANE_BG, PLANE_ED), ("b3", W_BG, W_ED),
         ("평면 B", PLANE_BG, PLANE_ED), ("꼬리", "#ffffff", MUTE)]
CWD, CGP = 100, 16
for i, (nm, bg, ed) in enumerate(CHAIN):
    x = mx + i * (CWD + CGP)
    box(x, my, CWD, 40, bg, ed, r=7)
    text(nm, x + CWD / 2, my + 25, size=11, weight="600", font=MONO,
         limit=CWD - 8)
    if i < len(CHAIN) - 1:
        arrow(x + CWD + 1, my + 20, x + CWD + CGP - 1, my + 20, sw=1.4)
text("블록 안은 지금 그대로 (검증 완료)", mx + 3.5 * (CWD + CGP), my + 60,
     size=10.5, fill="#8a6410", weight="600", limit=340)

GOOD = [("kws_block 을 안 고친다",
         "지금 인터페이스(프레임 push → 프레임 받기)를 평면 리더가 그대로 먹인다"),
        ("정렬 문제가 블록 안으로 격리된다", "거기는 이미 풀렸다"),
        ("kws_top 이 5 단계 루프가 된다", "conv1 · b1 · b2 · b3 · 꼬리"),
        ("남은 접점이 20개가 아니라 4개", "버그가 날 자리가 그만큼 준다")]
for i, (a, b) in enumerate(GOOD):
    yy = CY + 130 + i * 40
    box(60, yy, W - 120, 34, "#ffffff", OK_ED, r=7, sw=1.1)
    text(f"·  {a}", 78, yy + 15, size=11.5, weight="600", anchor="start",
         limit=520)
    text(b, 78, yy + 29, size=10, fill=MUTE, anchor="start", limit=1040)

box(60, CY + 292, W - 120, 22, "#fffbf0", W_ED, r=6, sw=1.1)
text("순수 평면으로 가면 kws_block 의 지연선·drain 로직(~80줄)이 필요 없어져 "
     "방금 한 작업 일부가 죽는다. 절충안은 그걸 살린다.",
     W / 2, CY + 307, size=10.5, limit=W - 160)

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
print(f"plane {BIG_CH}ch x {T_FR}fr = {BIG_BITS} b = {BIG_BITS/8/1024:.2f} KB; "
      f"{N_PLANES} planes = {SHARE:.3f}% of 16 Mb")

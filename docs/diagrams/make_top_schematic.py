"""32_top_schematic.svg -- the whole datapath as a schematic sheet.

Every other diagram here explains ONE idea. This one is the sheet you keep open
while reading the rest: which module is which, what its ports are actually
called, and what runs between them.

Drawn in the style of an HDL schematic viewer on purpose -- port names inside
the box at the edge they belong to, nets labelled on the wire, the module name
and its role underneath -- because that is the form an EE reader already knows
how to scan.

  A. the datapath, left to right and top to bottom
  B. the sequencer, which is what turns eleven modules into five phases

Run:  python3 docs/diagrams/make_top_schematic.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "32_top_schematic.svg"

W, H = 1680, 1180
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
NET = "#2f6fd0"          # data nets
CTL = "#8a8f9c"          # control / handshake
BIN_BG, BIN_ED = "#eef4fd", "#2f6fd0"   # binary datapath
MEM_BG, MEM_ED = "#fdf3e3", "#c9902a"   # storage
FIX_BG, FIX_ED = "#e9f5ec", "#3f9a5d"   # fixed point
ANA_BG, ANA_ED = "#fdeeec", "#cc6155"   # the analog boundary
SEQ_BG, SEQ_ED = "#f2f0fa", "#6b5bb5"   # control

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=4, sw=1.6, track=True, dash=None):
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


def wire(pts, color=NET, sw=1.6, arrow=True, dash=None):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#a)"' if color == NET else ' marker-end="url(#g)"'
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}'
        f'{mk if arrow else ""}/>')


def netlabel(s, x, y, color=NET):
    text(s, x, y, size=10, fill=color, font=MONO, weight="600")


def blk(x, y, w, h, ref, name, role, lefts, rights, bg, ed):
    """One schematic symbol. Returns pin coordinates by name."""
    box(x, y, w, h, bg, ed)
    text(ref, x + 4, y - 6, size=11.5, fill=ed, weight="700", anchor="start")
    text(name, x + w / 2, y + h + 16, size=12, fill=ed, weight="700",
         font=MONO, limit=w + 90)
    text(role, x + w / 2, y + h + 31, size=10, fill=MUTE, limit=w + 110)
    pins = {}
    for i, p in enumerate(lefts):
        py = y + 22 + i * 15
        text(p, x + 7, py + 4, size=9.5, font=MONO, anchor="start",
             fill=INK, limit=w / 2 - 6)
        add(f'<line x1="{x-9}" y1="{py}" x2="{x}" y2="{py}" stroke="{MUTE}" '
            f'stroke-width="1.2"/>')
        pins[p] = (x - 9, py)
    for i, p in enumerate(rights):
        py = y + 22 + i * 15
        text(p, x + w - 7, py + 4, size=9.5, font=MONO, anchor="end",
             fill=INK, limit=w / 2 - 6)
        add(f'<line x1="{x+w}" y1="{py}" x2="{x+w+9}" y2="{py}" stroke="{MUTE}" '
            f'stroke-width="1.2"/>')
        pins[p] = (x + w + 9, py)
    return pins


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs>'
    f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{NET}"/></marker>'
    f'<marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{CTL}"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("KWS 가속기 — 상위 스키매틱", W / 2, 32, size=22, weight="700")
text("비교기 16가닥이 들어가서 클래스 인덱스 하나가 나온다. "
     "굵은 파랑 = 데이터(프레임), 회색 = 핸드셰이크.",
     W / 2, 54, size=12.5, fill=MUTE)

# ══ ROW 1 ═════════════════════════════════════════════════════════════════
R1 = 110
fc = blk(40, R1, 190, 130, "U1", "kws_frame_ctrl", "아날로그 경계 · ICD §5",
         ["clk", "rst_n", "start", "cmp[15:0]", "out_ready"],
         ["out_valid", "out_frame", "busy"], ANA_BG, ANA_ED)
c1 = blk(360, R1, 190, 130, "U2", "kws_conv1", "int8 × ±1 · stride 2",
         ["clk", "rst_n", "start", "in_push", "in_real", "in_frame[15:0]"],
         ["busy", "out_valid", "out_frame"], BIN_BG, BIN_ED)
pa = blk(680, R1, 175, 130, "U3", "kws_plane  A", "128 ch × 64 · FLUSH 12",
         ["wr_start", "wr_valid", "wr_frame", "rd_start", "rd_ready"],
         ["wr_full", "rd_push", "rd_real", "rd_frame", "rd_done"],
         MEM_BG, MEM_ED)
b1 = blk(985, R1, 190, 130, "U4", "kws_block  b1", "TCS ×2 + projection skip",
         ["clk", "rst_n", "start", "in_push", "in_real", "in_frame[127:0]"],
         ["busy", "out_valid", "out_frame"], BIN_BG, BIN_ED)

text("아날로그", 135, R1 - 24, size=10.5, fill=ANA_ED, weight="700")
add(f'<line x1="270" y1="{R1-30}" x2="270" y2="{R1+180}" stroke="{ANA_ED}" '
    f'stroke-width="1.4" stroke-dasharray="5 4"/>')
text("여기부터 전부 동기 · 태그를 모른다", 300, R1 - 24, size=10.5,
     fill=MUTE, anchor="start")

wire([fc["out_frame"], (300, fc["out_frame"][1]), (300, c1["in_frame[15:0]"][1]),
      c1["in_frame[15:0]"]])
netlabel("FRAME[15:0]", 300, R1 + 8)
wire([c1["out_frame"], (620, c1["out_frame"][1]),
      (620, pa["wr_frame"][1]), pa["wr_frame"]])
netlabel("C1_OUT[127:0]", 620, R1 + 8)
wire([pa["rd_frame"], (930, pa["rd_frame"][1]),
      (930, b1["in_frame[127:0]"][1]), b1["in_frame[127:0]"]])
netlabel("A_FRAME", 930, R1 + 8)

# ══ ROW 2 ═════════════════════════════════════════════════════════════════
R2 = 400
pb = blk(40, R2, 175, 130, "U5", "kws_plane  B", "64 ch × 64 · FLUSH 14",
         ["wr_start", "wr_valid", "wr_frame", "rd_start", "rd_ready"],
         ["wr_full", "rd_push", "rd_real", "rd_frame", "rd_done"],
         MEM_BG, MEM_ED)
b2 = blk(345, R2, 190, 130, "U6", "kws_block  b2", "TCS ×2 + identity skip",
         ["clk", "rst_n", "start", "in_push", "in_real", "in_frame[63:0]"],
         ["busy", "out_valid", "out_frame"], BIN_BG, BIN_ED)
pc = blk(680, R2, 175, 130, "U7", "kws_plane  C", "64 ch × 64 · FLUSH 16",
         ["wr_start", "wr_valid", "wr_frame", "rd_start", "rd_ready"],
         ["wr_full", "rd_push", "rd_real", "rd_frame", "rd_done"],
         MEM_BG, MEM_ED)
b3 = blk(985, R2, 190, 130, "U8", "kws_block  b3", "TCS ×2 + identity skip",
         ["clk", "rst_n", "start", "in_push", "in_real", "in_frame[63:0]"],
         ["busy", "out_valid", "out_frame"], BIN_BG, BIN_ED)

wire([b1["out_frame"], (1230, b1["out_frame"][1]), (1230, R1 + 205),
      (18, R1 + 205), (18, pb["wr_frame"][1]), pb["wr_frame"]])
netlabel("B1_OUT[63:0]", 1150, R1 + 197)
wire([pb["rd_frame"], (290, pb["rd_frame"][1]),
      (290, b2["in_frame[63:0]"][1]), b2["in_frame[63:0]"]])
wire([b2["out_frame"], (620, b2["out_frame"][1]),
      (620, pc["wr_frame"][1]), pc["wr_frame"]])
netlabel("B2_OUT", 620, R2 + 8)
wire([pc["rd_frame"], (930, pc["rd_frame"][1]),
      (930, b3["in_frame[63:0]"][1]), b3["in_frame[63:0]"]])

# ══ ROW 3 ═════════════════════════════════════════════════════════════════
R3 = 690
pd = blk(40, R3, 175, 130, "U9", "kws_plane  D", "64 ch × 64 · FLUSH 28",
         ["wr_start", "wr_valid", "wr_frame", "rd_start", "rd_ready"],
         ["wr_full", "rd_push", "rd_real", "rd_frame", "rd_done"],
         MEM_BG, MEM_ED)
c2 = blk(345, R3, 190, 130, "U10", "kws_dw_conv", "conv2_dw · k=29 dil=2",
         ["clk", "rst_n", "start", "in_push", "in_real", "in_frame[63:0]"],
         ["busy", "out_valid", "out_frame"], BIN_BG, BIN_ED)
tl = blk(680, R3, 200, 130, "U11", "kws_tail", "여기서 산술이 시작된다",
         ["clk", "rst_n", "start", "in_valid", "in_frame[63:0]"],
         ["busy", "class_valid", "class_idx[3:0]"], FIX_BG, FIX_ED)

wire([b3["out_frame"], (1230, b3["out_frame"][1]), (1230, R2 + 205),
      (18, R2 + 205), (18, pd["wr_frame"][1]), pd["wr_frame"]])
netlabel("B3_OUT[63:0]", 1150, R2 + 197)
wire([pd["rd_frame"], (290, pd["rd_frame"][1]),
      (290, c2["in_frame[63:0]"][1]), c2["in_frame[63:0]"]])
wire([c2["out_frame"], (615, c2["out_frame"][1]),
      (615, tl["in_frame[63:0]"][1]), tl["in_frame[63:0]"]])
netlabel("C2_OUT", 615, R3 + 8)

# routed downward, not right: the "inside U11" panel starts at x=950 and a
# rightward arrow would run under it
_cx, _cy = tl["class_idx[3:0]"]
add(f'<path d="M{_cx},{_cy} L912,{_cy} L912,{R3+150}" fill="none" '
    f'stroke="{NET}" stroke-width="2.4" marker-end="url(#a)"/>')
text("CLASS[3:0]", 912, R3 + 168, size=12, weight="700", fill=NET, font=MONO)
text("0..11", 912, R3 + 182, size=9.5, fill=MUTE)

# ---- what is inside kws_tail --------------------------------------------- #
box(950, R3 + 40, 700, 118, "#ffffff", FIX_ED, r=8, sw=1.3)
text("U11 안쪽 — 여섯 단계가 접착제 없이 이어진다", 966, R3 + 60, size=11,
     weight="700", fill=FIX_ED, anchor="start")
chain = [("kws_pw_conv", "conv2_pw\n이진 MAC"), ("kws_affine", "→ 8.6, relu"),
         ("kws_dense_conv", "conv3\nint8 MAC"), ("kws_affine", "→ 5.6, relu"),
         ("kws_dense_conv", "conv4\n고정 MAC"), ("kws_affine", "→ 8.6 logits")]
cw = 104
for i, (mod, what) in enumerate(chain):
    cx = 966 + i * (cw + 10)
    box(cx, R3 + 72, cw, 40, FIX_BG, FIX_ED, r=5, sw=1.1, track=False)
    text(mod, cx + cw / 2, R3 + 88, size=8.5, font=MONO, weight="700",
         limit=cw - 4)
    for j, ln in enumerate(what.split("\n")):
        text(ln, cx + cw / 2, R3 + 100 + j * 9, size=8, fill=MUTE, limit=cw - 4)
    if i:
        add(f'<line x1="{cx-10}" y1="{R3+92}" x2="{cx-2}" y2="{R3+92}" '
            f'stroke="{FIX_ED}" stroke-width="1.4" marker-end="url(#a)"/>')
text("affine 은 사이클당 채널 하나를 내보내고 dense 는 하나를 싣는다 — "
     "직결이라 평면이 필요 없다. 마지막에 T=64 프레임 합 → argmax.",
     966, R3 + 128, size=10, fill=MUTE, anchor="start", limit=680)

# ══ SEQUENCER ═════════════════════════════════════════════════════════════
SQ = 900
box(24, SQ, W - 48, 150, SEQ_BG, SEQ_ED, r=10, sw=1.6)
text("kws_top 의 시퀀서 — 열한 개 모듈을 다섯 단계로", 44, SQ + 24,
     size=14, weight="700", fill=SEQ_ED, anchor="start")
text("층 L 이 클립을 다 끝낸 뒤에 L+1 이 읽는다. 그래서 접합부마다 평면 한 장이면 "
     "되고 핑퐁이 필요 없다.", 44, SQ + 43, size=11, fill=MUTE, anchor="start")

phases = [("S_C1", "U2 → U3", "U3 가 찰 때까지\n(flush 4개 자체 생성)"),
          ("S_B1", "U3 → U4 → U5", "pa_done ∧ pb_full"),
          ("S_B2", "U5 → U6 → U7", "pb_done ∧ pc_full"),
          ("S_B3", "U7 → U8 → U9", "pc_done ∧ pd_full"),
          ("S_TL", "U9 → U10 → U11", "class_valid")]
pw_ = 290
for i, (st, path, wait) in enumerate(phases):
    px = 44 + i * (pw_ + 26)
    box(px, SQ + 56, pw_, 74, "#ffffff", SEQ_ED, r=6, sw=1.2, track=False)
    text(st, px + 12, SQ + 76, size=12, font=MONO, weight="700", fill=SEQ_ED,
         anchor="start", limit=70)
    text(path, px + 90, SQ + 76, size=10.5, font=MONO, anchor="start",
         limit=pw_ - 100)
    for j, ln in enumerate(wait.split("\n")):
        text("기다리는 것: " + ln if j == 0 else ln, px + 12,
             SQ + 96 + j * 13, size=9.5, fill=MUTE, anchor="start",
             limit=pw_ - 24)
    if i:
        add(f'<line x1="{px-24}" y1="{SQ+93}" x2="{px-6}" y2="{SQ+93}" '
            f'stroke="{SEQ_ED}" stroke-width="1.6" marker-end="url(#a)"/>')

# ══ LEGEND ════════════════════════════════════════════════════════════════
LG = 1075
keys = [(BIN_BG, BIN_ED, "이진 데이터패스", "곱셈기 없음 · XNOR+popcount"),
        (MEM_BG, MEM_ED, "평면 (BRAM)", "층 사이 · 「6프레임 전」이 주소가 된다"),
        (FIX_BG, FIX_ED, "고정소수점 꼬리", "BN 이 살아남아 곱셈기가 필요"),
        (ANA_BG, ANA_ED, "아날로그 경계", "유일한 비동기 · 2FF 동기화")]
for i, (bg, ed, name, note) in enumerate(keys):
    kx = 44 + i * 405
    box(kx, LG, 22, 22, bg, ed, r=3, sw=1.4, track=False)
    text(name, kx + 32, LG + 11, size=11, weight="700", anchor="start",
         limit=150)
    text(note, kx + 32, LG + 26, size=9.5, fill=MUTE, anchor="start", limit=360)

text("모든 폭·커널·ROM 경로는 export 가 만든 parameters.vh 에서 온다 — "
     "RTL 소스에는 태그도 숫자도 없다 (docs/ICD.md, rtl/RUNNING.md).",
     W / 2, LG + 62, size=11, fill=MUTE)

add("</svg>")

bad = [b for b in BOXES if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W
       or b[1] + b[3] > H]
if bad:
    raise ValueError(f"box outside canvas: {bad[:2]}")
for s, x, y, size, anchor in TEXTS:
    ww = tw(s, size)
    lo = x if anchor == "start" else (x - ww if anchor == "end" else x - ww / 2)
    if lo < -1 or lo + ww > W + 1 or y > H:
        raise ValueError(f"text outside canvas: {s!r} at {lo:.0f}..{lo+ww:.0f}")

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT}  ({W}x{H})")
print(f"{len(BOXES)} tracked boxes, {len(TEXTS)} labels")

"""25_module_internals.svg -- what is inside each verified RTL module.

The progress diagram (21) says which modules exist. This one says what they
contain and how a value moves through them, which is what you need in order to
review the design or to add the next module on top of it.

Four panels, bottom of the hierarchy first, because each is built from the one
above it:
  A. kws_bin_mac   -- the only place arithmetic happens
  B. kws_dw_conv   -- line buffer, the edge machinery, one MAC
  C. kws_pw_conv   -- word streaming, one MAC
  D. kws_tcs_sub   -- composition, and the handshake contract they all share

Cycle counts are computed from the FSMs rather than typed, so they cannot drift
from the modules.

Run:  python3 docs/diagrams/make_module_internals.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / "25_module_internals.svg"

W, H = 1240, 1400
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE, GRID = "#1c2333", "#5b6478", "#c3cbd9", "#e8ecf2"
MAC_BG, MAC_ED = "#eef2f7", "#7d8896"      # the arithmetic core
DW_BG, DW_ED = "#e4eefb", "#2f6fd0"
PW_BG, PW_ED = "#fdf3e3", "#c9902a"
REG_BG, REG_ED = "#ffffff", "#8a93a6"      # registers
ROM_BG, ROM_ED = "#f3eefb", "#7a5bb5"      # ROMs
OK_BG, OK_ED = "#e6f4ea", "#3f9a5d"

# --- the design's actual numbers -------------------------------------------
C_DW, K, PAD = 128, 13, 6
C_PWI, C_PWO, NW = 128, 64, 4
DW_CYC_PER_CH = 3           # S_START, S_FEED, S_TAKE
PW_CYC_PER_CH = 2 + NW      # S_START, S_FEED x NW, S_TAKE
DW_FRAME = C_DW * DW_CYC_PER_CH
PW_FRAME = C_PWO * PW_CYC_PER_CH

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


def arrow(x1, y1, x2, y2, color=MUTE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d} marker-end="url(#a)"/>')


def unit(x, y, w, h, title, sub, bg, ed, note=None):
    """A functional block with a name, a one-line role, and an optional note."""
    box(x, y, w, h, bg, ed)
    text(title, x + w / 2, y + 20, size=12, weight="700", font=MONO,
         limit=w - 12)
    text(sub, x + w / 2, y + 37, size=10.5, limit=w - 10)
    if note:                       # below the box: inside it collides with sub
        text(note, x + w / 2, y + h + 13, size=9.5, fill=MUTE, limit=w + 30)


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("검증된 RTL 모듈의 내부 구조", W / 2, 36, size=21, weight="700",
     limit=W - 60)
text("아래에서 위로. 각 모듈은 그 아래 것을 인스턴스한다 — 그래서 위 모듈의 "
     "테스트벤치는 아래가 맞다는 걸 알고 시작한다.", W / 2, 58, size=12.5,
     fill=MUTE, limit=W - 60)

# ══ A. kws_bin_mac ════════════════════════════════════════════════════════
AY = 78
box(24, AY, W - 48, 246, "#ffffff", LINE, r=12, sw=1.2)
text("A.  kws_bin_mac — 이 설계에서 산술이 일어나는 유일한 곳", 44, AY + 25,
     size=14.5, weight="700", anchor="start")
text("입력도 ±1, 가중치도 ±1 이면 곱은 「부호가 같은가」다. XNOR 이 그걸 세고, "
     "곱셈기는 하나도 없다.", 44, AY + 44, size=11.5, fill=MUTE, anchor="start",
     limit=860)

ax, ay = 60, AY + 62
unit(ax, ay, 118, 46, "act[31:0]", "활성 word", REG_BG, REG_ED)
unit(ax, ay + 62, 118, 46, "wgt[31:0]", "가중치 word", ROM_BG, ROM_ED)
unit(ax + 152, ay + 30, 96, 46, "XNOR", "부호 일치", MAC_BG, MAC_ED)
arrow(ax + 118, ay + 23, ax + 150, ay + 45)
arrow(ax + 118, ay + 85, ax + 150, ay + 63)

unit(ax + 282, ay + 30, 110, 46, "& mask", "tail 차단", MAC_BG, MAC_ED)
arrow(ax + 248, ay + 53, ax + 280, ay + 53)
unit(ax + 282, ay + 108, 110, 42, "n_valid", "유효 항 수", REG_BG, REG_ED)
add(f'<line x1="{ax + 337}" y1="{ay + 108}" x2="{ax + 337}" y2="{ay + 78}" '
    f'stroke="{MUTE}" stroke-width="1.6" marker-end="url(#a)"/>')

unit(ax + 426, ay + 30, 112, 46, "popcount", "1 의 개수 P", MAC_BG, MAC_ED)
arrow(ax + 392, ay + 53, ax + 424, ay + 53)
unit(ax + 572, ay + 30, 132, 46, "2·P − N", "부호 있는 합", MAC_BG, MAC_ED)
arrow(ax + 538, ay + 53, ax + 570, ay + 53)
add(f'<path d="M{ax + 337} {ay + 150} L{ax + 337} {ay + 168} '
    f'L{ax + 638} {ay + 168} L{ax + 638} {ay + 80}" fill="none" '
    f'stroke="{MUTE}" stroke-width="1.6" stroke-dasharray="4 3" '
    f'marker-end="url(#a)"/>')
text("N 으로도 쓰인다", ax + 490, ay + 163, size=10, fill=MUTE, limit=200)

unit(ax + 738, ay + 30, 118, 46, "acc", "정수 누산기", OK_BG, OK_ED)
arrow(ax + 704, ay + 53, ax + 736, ay + 53)

box(ax + 880, ay - 4, W - 24 - (ax + 880) - 16, 156, "#fffbf0", "#c9902a", r=9)
text("word 가 여러 개면 P 를 누적", ax + 896, ay + 18, size=11.5, weight="700",
     anchor="start", limit=260)
for j, s in enumerate([
        "start → n_valid 를 래치, P=0",
        "in_valid 마다 한 word 씩",
        "마지막 word 에서 2P−N 을 확정",
        "",
        "tail mask 가 모듈 안에 있는 이유:",
        "0 패딩 비트는 진짜 −1 이라",
        "빼먹으면 합에 조용히 낀다"]):
    text(s, ax + 896, ay + 40 + j * 17, size=10.5, anchor="start", limit=270)

# ══ B. kws_dw_conv ════════════════════════════════════════════════════════
BY = 342
box(24, BY, W - 48, 410, "#ffffff", LINE, r=12, sw=1.2)
text(f"B.  kws_dw_conv — 시간 방향 K={K} 탭, 채널을 섞지 않는다", 44, BY + 25,
     size=14.5, weight="700", anchor="start")
text("메모리에는 세로(프레임)로 들어 있는데 필요한 건 가로(시간)라, K 프레임을 "
     "레지스터에 붙들어 gather 를 공짜로 만든다.", 44, BY + 44, size=11.5,
     fill=MUTE, anchor="start", limit=880)

bx, by = 60, BY + 62
unit(bx, by, 128, 50, "in_frame", f"{C_DW}b, 한 프레임", REG_BG, REG_ED)
unit(bx, by + 86, 128, 50, "in_real", "1=데이터 0=drain", REG_BG, REG_ED)

unit(bx + 162, by, 152, 50, "fbuf[0:12]", f"{K}×{C_DW}b shift reg",
     DW_BG, DW_ED, "line buffer")
unit(bx + 162, by + 86, 152, 50, "valid[12:0]", f"{K}b shift reg",
     DW_BG, DW_ED, "pad 위치를 기억")
arrow(bx + 128, by + 25, bx + 160, by + 25)
arrow(bx + 128, by + 111, bx + 160, by + 111)

unit(bx + 348, by, 132, 50, "gather", "슬롯 j 의 비트 c", DW_BG, DW_ED,
     "조합, 0 사이클")
arrow(bx + 314, by + 25, bx + 346, by + 25)
unit(bx + 348, by + 86, 132, 34, "tzc → shift", "", DW_BG, DW_ED)
unit(bx + 348, by + 128, 132, 34, "popcnt → n_valid", "", DW_BG, DW_ED)
arrow(bx + 314, by + 104, bx + 346, by + 104)
arrow(bx + 314, by + 118, bx + 346, by + 142)

unit(bx + 514, by, 118, 50, ">> shift", "정렬", DW_BG, DW_ED)
arrow(bx + 480, by + 25, bx + 512, by + 25)
unit(bx + 514, by + 86, 118, 50, ">> shift", "가중치도 같이", DW_BG, DW_ED)
# shift goes to BOTH shifters. Routed around the right of the tzc block --
# drawing it straight would run the wire through the box.
add(f'<path d="M{bx + 480} {by + 103} L{bx + 496} {by + 103} L{bx + 496} '
    f'{by + 38} L{bx + 512} {by + 38}" fill="none" stroke="{MUTE}" '
    f'stroke-width="1.4" marker-end="url(#a)"/>')
add(f'<path d="M{bx + 496} {by + 103} L{bx + 496} {by + 124} L{bx + 512} '
    f'{by + 124}" fill="none" stroke="{MUTE}" stroke-width="1.4" '
    f'marker-end="url(#a)"/>')

unit(bx + 514, by + 172, 118, 44, "w_rom[ch]", f"{K}b 탭", ROM_BG, ROM_ED)
add(f'<line x1="{bx + 573}" y1="{by + 172}" x2="{bx + 573}" y2="{by + 138}" '
    f'stroke="{MUTE}" stroke-width="1.6" marker-end="url(#a)"/>')

unit(bx + 666, by + 30, 128, 50, "kws_bin_mac", "A", MAC_BG, MAC_ED)
arrow(bx + 632, by + 25, bx + 664, by + 48)
arrow(bx + 632, by + 111, bx + 664, by + 66)

unit(bx + 828, by + 30, 132, 50, "acc ≥ thr ?", "융합된 threshold",
     OK_BG, OK_ED, "BN 이 여기 접혔다")
arrow(bx + 794, by + 55, bx + 826, by + 55)
unit(bx + 828, by + 116, 132, 44, "t_rom[ch]", "정수 + 극성", ROM_BG, ROM_ED)
add(f'<line x1="{bx + 894}" y1="{by + 116}" x2="{bx + 894}" y2="{by + 84}" '
    f'stroke="{MUTE}" stroke-width="1.6" marker-end="url(#a)"/>')

unit(bx + 994, by + 30, 118, 50, "out_frame", f"{C_DW}b ±1", OK_BG, OK_ED)
arrow(bx + 960, by + 55, bx + 992, by + 55)

box(60, BY + 310, W - 120, 82, "#f7fbf8", OK_ED, r=9)
text("한 프레임 = 채널 128개를 차례로 — 시분할(folded)", 76, BY + 310,
     size=12, weight="700", anchor="start", limit=440)
text(f"채널당 {DW_CYC_PER_CH} 사이클 (S_START → S_FEED → S_TAKE) × "
     f"{C_DW}채널 = 프레임당 {DW_FRAME} 사이클", 76, BY + 354, size=11.5,
     anchor="start", limit=700)
text("shift 와 n_valid 는 valid 레지스터 하나에서 나온다 — 프레임 카운터도, "
     "양끝 별도 산술도 없다.", 76, BY + 374, size=11, fill=MUTE, anchor="start",
     limit=800)

# ══ C. kws_pw_conv ════════════════════════════════════════════════════════
CY = 770
box(24, CY, W - 48, 258, "#ffffff", LINE, r=12, sw=1.2)
text("C.  kws_pw_conv — 채널 방향 1×1, 시간을 섞지 않는다", 44, CY + 25,
     size=14.5, weight="700", anchor="start")
text("필요한 벡터가 저장된 word 그 자체다. line buffer 도 shift 도 없고 "
     "n_valid 는 언제나 C_IN 이다.", 44, CY + 44, size=11.5, fill=MUTE,
     anchor="start", limit=880)

cx, cy = 60, CY + 66
unit(cx, cy, 128, 50, "in_frame", f"{C_PWI}b", REG_BG, REG_ED)
unit(cx + 162, cy, 128, 50, "act (latch)", "프레임 보관", PW_BG, PW_ED)
arrow(cx + 128, cy + 25, cx + 160, cy + 25)
unit(cx + 324, cy, 128, 50, "word 선택", f"wi = 0..{NW-1}", PW_BG, PW_ED)
arrow(cx + 290, cy + 25, cx + 322, cy + 25)

unit(cx + 324, cy + 74, 128, 46, "w_rom[wa]", "주소 카운터", ROM_BG, ROM_ED,
     "곱셈 없음")
unit(cx + 496, cy + 12, 128, 50, "kws_bin_mac", "A", MAC_BG, MAC_ED)
arrow(cx + 452, cy + 25, cx + 494, cy + 30)
arrow(cx + 452, cy + 95, cx + 494, cy + 52)

unit(cx + 664, cy + 12, 132, 50, "acc ≥ thr ?", "융합된 threshold",
     OK_BG, OK_ED)
arrow(cx + 624, cy + 37, cx + 662, cy + 37)
unit(cx + 838, cy + 12, 118, 50, "out_frame", f"{C_PWO}b ±1", OK_BG, OK_ED)
arrow(cx + 796, cy + 37, cx + 836, cy + 37)

box(60, CY + 190, W - 120, 56, "#fffbf0", "#c9902a", r=9)
text(f"채널당 {PW_CYC_PER_CH} 사이클 (S_START → S_FEED ×{NW} → S_TAKE) × "
     f"{C_PWO}채널 = 프레임당 {PW_FRAME} 사이클", 76, CY + 214, size=11.5,
     weight="600", anchor="start", limit=700)
text(f"S_FEED 가 {NW}번인 것이 packing 이득이다 — {C_PWI}개 항을 {NW} 사이클에. "
     f"depthwise 는 {K}개 항에 1 사이클이라 32칸 중 {K}칸만 쓴다.",
     76, CY + 234, size=11, fill=MUTE, anchor="start", limit=1040)

# ══ D. kws_tcs_sub ════════════════════════════════════════════════════════
DY = 1050
box(24, DY, W - 48, 320, "#ffffff", LINE, r=12, sw=1.2)
text("D.  kws_tcs_sub — 합성, 그리고 모든 모듈이 공유하는 handshake",
     44, DY + 25, size=14.5, weight="700", anchor="start")

dx, dy = 60, DY + 54
unit(dx, dy, 118, 46, "in_frame", "블록 입력", REG_BG, REG_ED)
unit(dx + 152, dy, 150, 46, "kws_dw_conv", "B", DW_BG, DW_ED)
arrow(dx + 118, dy + 23, dx + 150, dy + 23)
unit(dx + 336, dy, 150, 46, "handoff reg", "1 사이클", REG_BG, REG_ED)
arrow(dx + 302, dy + 23, dx + 334, dy + 23)
unit(dx + 520, dy, 150, 46, "kws_pw_conv", "C", PW_BG, PW_ED)
arrow(dx + 486, dy + 23, dx + 518, dy + 23)
unit(dx + 704, dy, 128, 46, "out_frame", "sub 출력", OK_BG, OK_ED)
arrow(dx + 670, dy + 23, dx + 702, dy + 23)
text("±1 (재이진화 완료)", dx + 411, dy + 66, size=10, fill=MUTE, limit=200)

box(dx, dy + 82, 832, 62, "#fdf0ef", "#cc6155", r=9)
text("busy = dw_busy | dw_ov | pw_iv | pw_busy", dx + 16, dy + 104, size=12,
     font=MONO, weight="700", anchor="start", limit=400)
text("네 개를 다 넣어야 한다. dw_busy 는 dw_ov 를 올리는 엣지에 떨어지고 "
     "pw_iv 는 그 다음 엣지에 오른다 —", dx + 16, dy + 124, size=10.5,
     anchor="start", limit=800)
text("두 개만 보면 프레임이 공중에 뜬 채 busy 가 한 사이클 0 이 된다.",
     dx + 16, dy + 138, size=10.5, fill=MUTE, anchor="start", limit=800)

box(dx + 856, dy - 6, W - 24 - (dx + 856) - 16, 152, "#f7fbf8", OK_ED, r=9)
text("공통 인터페이스 규약", dx + 872, dy + 16, size=11.5, weight="700",
     anchor="start", limit=240)
for j, s in enumerate([
        "· busy 중에는 절대 push 금지",
        "· out_valid 는 busy 가 내려가는",
        "  그 엣지에 1 사이클 뜬다",
        "· MAC strobe 는 상태에서 조합",
        "· start 는 클립 시작에만"]):
    text(s, dx + 872, dy + 38 + j * 18, size=10.5, anchor="start", limit=250)

box(60, DY + 226, W - 120, 78, "#ffffff", LINE, r=9, sw=1.2)
text("다음 — kws_block 이 재사용하지 않는 것", 76, DY + 248, size=12,
     weight="700", anchor="start", limit=420)
text("residual block 의 **마지막** sub-block 은 pointwise 가 threshold 로 "
     "끝나지 않는다 (manifest epilogue \"none\").".replace("**", ""),
     76, DY + 270, size=11, anchor="start", limit=1040)
text("raw 정수 누산기를 skip 누산기와 더한 뒤 threshold 를 한 번만 건다. "
     "여기에 kws_tcs_sub 을 쓰면 학습된 망에 없는 threshold 가 끼어든다.",
     76, DY + 290, size=11, fill=MUTE, anchor="start", limit=1040)

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
print(f"dw {DW_CYC_PER_CH}cyc/ch x {C_DW} = {DW_FRAME}/frame | "
      f"pw {PW_CYC_PER_CH}cyc/ch x {C_PWO} = {PW_FRAME}/frame")

"""21_rtl_progress.svg -- what has been built for the FPGA, and what golden
vectors are for.

Three questions, in this order:
  A. What does the build actually produce, and from what?
  B. Which RTL modules exist, and which are next?
  C. What do golden vectors buy? (the answer is: a bug names its own module)

Module status is read from the filesystem rather than typed, so a module that
gets written or deleted cannot leave this diagram stale.

Run:  python3 docs/diagrams/make_rtl_progress.py
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "diagrams" / "21_rtl_progress.svg"

W, H = 1240, 968
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK, MUTE, LINE = "#1c2333", "#5b6478", "#c3cbd9"
DONE_BG, DONE_ED = "#e6f4ea", "#3f9a5d"
NOW_BG, NOW_ED = "#e4eefb", "#2f6fd0"
WAIT_BG, WAIT_ED = "#f2f4f7", "#a8b0bf"
SW_BG, SW_ED = "#fdf3e3", "#d9a441"       # software side
HW_BG, HW_ED = "#eaf1fb", "#4a7fd0"       # hardware side
BAD_BG, BAD_ED = "#fdf0ef", "#cc6155"

RTLDIR = ROOT / "rtl"

parts: list[str] = []
add = parts.append
BOXES: list[tuple] = []
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def box(x, y, w, h, fill, edge, r=9, sw=1.4, dash=None):
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


def arrow(x1, y1, x2, y2, color=LINE, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MUTE}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("FPGA 설계 — 지금까지 만든 것", W / 2, 36, size=21, weight="700",
     limit=W - 60)
text("학습된 모델에서 RTL 이 읽는 파일까지는 끝났다. 이제 그 파일들을 쓰는 "
     "하드웨어를 아래에서 위로 쌓는다.", W / 2, 58, size=12.5, fill=MUTE,
     limit=W - 60)

# ══ A. build flow ═════════════════════════════════════════════════════════
AY, AH = 78, 224
box(24, AY, W - 48, AH, "#ffffff", LINE, r=12, sw=1.2)
text("A.  빌드 흐름 — 학습된 모델이 어떻게 하드웨어가 읽는 파일이 되나",
     44, AY + 25, size=14.5, weight="700", anchor="start")

box(44, AY + 46, 190, 62, SW_BG, SW_ED, r=8)
text("학습된 체크포인트", 139, AY + 70, size=12.5, weight="600", limit=180)
text("runs/xl_g12/best.pt", 139, AY + 90, size=10.5, fill=MUTE, font=MONO,
     limit=180)

TOOLS = [("export/emit.py", "치수 · 가중치 · threshold", AY + 46),
         ("export/golden.py", "층별 기준값", AY + 132)]
for t, sub, y in TOOLS:
    box(268, y, 196, 62, "#ffffff", MUTE, r=8, sw=1.2)
    text(t, 366, y + 26, size=12, font=MONO, weight="600", limit=186)
    text(sub, 366, y + 46, size=11, fill=MUTE, limit=186)
    arrow(236, AY + 77, 266, y + 31)

OUTS = [("parameters.vh", "치수 — RTL 이 include", AY + 40, HW_BG, HW_ED),
        ("*.hex  (32개)", "가중치 · threshold → BRAM", AY + 92, HW_BG, HW_ED),
        ("golden/  (36개)", "층별 정답 — 테스트벤치용", AY + 144, DONE_BG, DONE_ED)]
for t, sub, y, bg, ed in OUTS:
    box(500, y, 250, 44, bg, ed, r=8)
    text(t, 625, y + 20, size=12, font=MONO, weight="600", limit=240)
    text(sub, 625, y + 36, size=10.5, fill=MUTE, limit=240)
arrow(466, AY + 77, 498, AY + 62)
arrow(466, AY + 77, 498, AY + 114)
arrow(466, AY + 163, 498, AY + 166)

box(786, AY + 46, W - 24 - 786 - 20, 148, HW_BG, HW_ED, r=9)
text("RTL", 786 + (W - 830) / 2, AY + 72, size=15, weight="700", limit=390)
text("숫자를 하나도 안 갖는다", 786 + (W - 830) / 2, AY + 94, size=12,
     fill=MUTE, limit=390)
for j, s in enumerate(["치수는 parameters.vh 에서",
                       "가중치는 $readmemh 로 BRAM 에",
                       "→ 재학습하면 .hex 만 교체"]):
    text("· " + s, 806, AY + 120 + j * 20, size=11.5, anchor="start", limit=380)
arrow(752, AY + 62, 784, AY + 100)
arrow(752, AY + 114, 784, AY + 110)

# ══ B. module tree ════════════════════════════════════════════════════════
BY, BH = 322, 256
box(24, BY, W - 48, BH, "#ffffff", LINE, r=12, sw=1.2)
text("B.  RTL 모듈 — 아래에서 위로", 44, BY + 25, size=14.5, weight="700",
     anchor="start")
text("작은 것부터 만들고 각 단계를 골든 벡터로 끊어 검증한다. 위 모듈은 아래가 "
     "맞다는 걸 알고 시작한다.", 44, BY + 44, size=11.5, fill=MUTE,
     anchor="start", limit=760)

# rtl/README.md 3 의 분해와 같아야 한다. dw 와 pw 는 메모리 접근 패턴이
# 다르고 골든 파일도 따로라, kws_tcs_sub 한 덩어리로 두지 않았다.
MODS = [
    ("kws_bin_mac", "2·popcount(XNOR) − N", "자체 벡터"),
    ("kws_dw_conv", "라인버퍼 + 게더 + shift", "b1_s0_dw 골든"),
    ("kws_pw_conv", "워드 스트리밍", "b1_s0_pw 골든"),
    ("kws_tcs_sub", "dw → pw 배선", "b1_s0_* 연결"),
    ("kws_block", "sub ×2 + residual add", "b1_add 골든"),
    ("kws_top", "21개 층 시분할 (folded)", "logits · predictions"),
    ("kws_frame_ctrl", "2FF 동기화 + sticky OR", "ICD §5 경계"),
]
MG = 10
MW = (W - 88 - (len(MODS) - 1) * MG) / len(MODS)
for i, (nm, what, verify) in enumerate(MODS):
    exists = (RTLDIR / f"{nm}.v").is_file()
    bg, ed, lab = ((DONE_BG, DONE_ED, "완료") if exists else
                   (NOW_BG, NOW_ED, "▶ 다음") if not any(
                       (RTLDIR / f"{m}.v").is_file() for m, _, _ in MODS[:i]
                   ) or (i > 0 and (RTLDIR / f"{MODS[i-1][0]}.v").is_file()) else
                   (WAIT_BG, WAIT_ED, "대기"))
    x = 44 + i * (MW + MG)
    box(x, BY + 64, MW, 118, bg, ed, r=9, sw=1.9 if lab == "▶ 다음" else 1.3)
    box(x + MW - 58, BY + 74, 48, 17, "#ffffff", ed, r=8, sw=1)
    text(lab, x + MW - 34, BY + 86, size=9.5, fill=ed, weight="700", limit=44)
    text(nm, x + 12, BY + 87, size=11.5, font=MONO, weight="700",
         anchor="start", limit=MW - 66)
    text(what, x + MW / 2, BY + 116, size=11, limit=MW - 16)
    text("검증", x + MW / 2, BY + 146, size=10, fill=MUTE, limit=MW - 16)
    text(verify, x + MW / 2, BY + 163, size=10, fill=MUTE, limit=MW - 12)
    if i < len(MODS) - 1:
        arrow(x + MW + 1, BY + 123, x + MW + MG - 1, BY + 123, sw=1.5)

text("남은 것:  합성 · 타이밍 → KC705 비트스트림", 44, BY + 212, size=12,
     anchor="start", fill=MUTE, limit=440)
text("아날로그는 동료 담당이라 경계(kws_frame_ctrl)까지만 우리 몫이다.",
     44, BY + 232, size=11.5, anchor="start", fill=MUTE, limit=600)

# ══ C. what golden vectors buy ════════════════════════════════════════════
CY, CHH = 598, 214
box(24, CY, W - 48, CHH, "#f7fbf8", DONE_ED, r=12, sw=1.4)
text("C.  골든 벡터 = 층마다 미리 적어둔 정답", 44, CY + 25, size=14.5,
     weight="700", anchor="start")
text("없으면 보이는 건 「12개 로짓이 이상하다」 하나뿐이다. 21개 층 어디가 "
     "틀렸는지 알 방법이 없다.", 44, CY + 44, size=11.5, fill=MUTE,
     anchor="start", limit=760)

CHAIN = [("conv1", True), ("b1_s0_dw", True), ("b1_s0_pw", False),
         ("b1_s1_dw", None), ("…", None), ("conv4", None)]
CW, CG = 140, 14
for i, (nm, ok) in enumerate(CHAIN):
    x = 60 + i * (CW + CG)
    bg, ed = ((DONE_BG, DONE_ED) if ok else
              (BAD_BG, BAD_ED) if ok is False else (WAIT_BG, WAIT_ED))
    box(x, CY + 66, CW, 52, bg, ed, r=8)
    text(nm, x + CW / 2, CY + 88, size=11.5, font=MONO, weight="600",
         limit=CW - 10)
    mark = "일치" if ok else ("불일치" if ok is False else "—")
    text(mark, x + CW / 2, CY + 106, size=10.5,
         fill=(DONE_ED if ok else BAD_ED if ok is False else MUTE), limit=CW - 10)
    if i < len(CHAIN) - 1:
        arrow(x + CW + 1, CY + 92, x + CW + CG - 1, CY + 92, sw=1.4)

x_bad = 60 + 2 * (CW + CG)
add(f'<path d="M{x_bad + CW / 2} {CY + 124} L{x_bad + CW / 2} {CY + 146}" '
    f'stroke="{BAD_ED}" stroke-width="2" marker-end="url(#a)"/>')
text("첫 불일치가 범인을 지목한다 — 이분 탐색이 아니라 바로 그 모듈이다",
     x_bad + CW / 2, CY + 164, size=12, weight="600", fill=BAD_ED, limit=620)

text("우리 파일: input.hex(입력) · <층>_acc.hex(정수 누산기) · "
     "<층>_out.hex(±1 출력) · logits.txt  — 8클립 36개",
     44, CY + 194, size=11, fill=MUTE, anchor="start", limit=W - 90)

# ══ D. lint vs sim ════════════════════════════════════════════════════════
DY = 828
box(24, DY, W - 48, 116, "#ffffff", LINE, r=12, sw=1.2)
text("D.  린트와 시뮬은 다른 것을 잡는다", 44, DY + 25, size=14.5,
     weight="700", anchor="start")

PAIR = [("린트 (verilator --lint-only)", "코드를 **읽어서** 잡는다",
         ["비트폭 불일치 → 조용한 잘림", "의도 안 한 래치", "미구동 · 미사용 신호"],
         "돌리지 않아도 나온다", SW_BG, SW_ED),
        ("시뮬 (iverilog / XSim)", "코드를 **돌려서** 잡는다",
         ["골든 벡터와 값이 다른가", "타이밍 · 상태 전이", "리셋 안 된 X 전파"],
         "데이터가 그 경우를 밟아야 나온다", HW_BG, HW_ED)]
PW = (W - 88 - 16) / 2
for i, (head, how, items, note, bg, ed) in enumerate(PAIR):
    x = 44 + i * (PW + 16)
    box(x, DY + 38, PW, 66, bg, ed, r=8)
    text(head.replace("**", ""), x + 12, DY + 58, size=12, weight="700",
         anchor="start", limit=PW - 24)
    text(how.replace("**", ""), x + PW - 12, DY + 58, size=11, fill=MUTE,
         anchor="end", limit=170)
    text("  ·  ".join(items), x + 12, DY + 78, size=10.5, anchor="start",
         limit=PW - 24)
    text(note, x + 12, DY + 96, size=10.5, fill=MUTE, anchor="start",
         limit=PW - 24)

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
built = [m for m, _, _ in MODS if (RTLDIR / f"{m}.v").is_file()]
print(f"wrote {OUT}  ({W}x{H})")
print(f"modules on disk: {', '.join(built) if built else '(none)'}")

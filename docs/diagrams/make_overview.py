"""19_overview.svg -- where the project is and what happens next, on one page.

Three questions the diagram answers, in this order:
  1. What is the physical chain, and where does the analog work stop being ours?
  2. Where do the two tracks differ? (exactly one spot -- worth seeing)
  3. What is the FPGA work sequence, and what does each step feed?

Numbers come from the repo, not from memory: channel count and corner
frequencies from the SPICE filterbank design, frame geometry from the AFE
config. Accuracies are cited with their source run so a stale number cannot
quietly survive a rewrite.

Run:  python3 docs/diagrams/make_overview.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "diagrams" / "19_overview.svg"

W, H = 1240, 976
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

INK = "#1c2333"
MUTE = "#5b6478"
LINE = "#c3cbd9"
ANALOG_BG, ANALOG_ED = "#fdf3e3", "#d9a441"     # colleague's side
FPGA_BG, FPGA_ED = "#eaf1fb", "#4a7fd0"         # ours
DONE_BG, DONE_ED = "#e6f4ea", "#3f9a5d"
NOW_BG, NOW_ED = "#e4eefb", "#2f6fd0"
WAIT_BG, WAIT_ED = "#f2f4f7", "#a8b0bf"
T1_BG, T1_ED = "#eaf1fb", "#4a7fd0"
T2_BG, T2_ED = "#fdf0ef", "#cc6155"

# ── repo facts ────────────────────────────────────────────────────────────
def _channels() -> tuple[int, float, float]:
    p = ROOT / "analog" / "AFE" / "artifacts" / "filterbank_design.csv"
    fc = [float(r["f_c"]) for r in csv.DictReader(p.open())
          if r.get("f_c")] if p.exists() else []
    if not fc:                                   # fall back to the design doc
        md = (ROOT / "analog" / "AFE" / "artifacts" / "component_table.md").read_text()
        fc = [float(m) for m in re.findall(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|", md, re.M)]
    return len(fc), min(fc), max(fc)


def _frames() -> tuple[int, int, int, float]:
    y = (ROOT / "configs" / "base.yaml").read_text()
    def num(k, d):
        m = re.search(rf"^\s*{k}:\s*([0-9.]+)", y, re.M)
        return float(m.group(1)) if m else d
    win = num("envelope_win_ms", 10.0)
    clip = num("clip_ms", 1000.0)
    T = int(num("time_steps", 128))
    native = int(round(clip / win))
    return native, T, (T - native) // 2, win


NCH, FMIN, FMAX = _channels()
NATIVE, TPAD, PAD_L, WIN_MS = _frames()

parts: list[str] = []
add = parts.append


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s: str, size: float) -> float:
    """Estimate rendered width; CJK glyphs are ~full-em, latin ~0.55em."""
    w = 0.0
    for ch in s:
        w += 1.02 if ord(ch) > 0x2E80 else 0.55
    return w * size


BOXES: list[tuple[float, float, float, float]] = []
TEXTS: list[tuple[str, float, float, float, str]] = []


def box(x, y, w, h, fill, edge, r=9, sw=1.4, dash=None):
    BOXES.append((x, y, w, h))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="{sw}"{d}/>')


def text(s, x, y, size=12.5, fill=INK, anchor="middle", weight="400",
         font=FONT, limit=None):
    TEXTS.append((s, x, y, size, anchor))
    if limit is not None and tw(s, size) > limit:
        raise ValueError(f"text overflows {limit:.0f}px: {s!r} "
                         f"needs {tw(s, size):.0f}px @ {size}px")
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f'{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color=LINE, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')


# ── canvas ────────────────────────────────────────────────────────────────
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
    '<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{DONE_ED}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

text("KWS AFE — 지금 어디에 있고, 다음에 무엇을 하는가", W / 2, 36,
     size=21, weight="700", limit=W - 60)
text(f"{NCH}채널 아날로그 프론트엔드 → 부분 이진화 MatchboxNet → 12-class  "
     f"|  아날로그=동료 담당, FPGA=우리 담당", W / 2, 58,
     size=12.5, fill=MUTE, limit=W - 60)

# ══ A. physical chain ═════════════════════════════════════════════════════
AY, AH = 84, 176
box(24, AY, W - 48, AH, "#ffffff", LINE, r=12, sw=1.2)
text("A.  물리 체인 — 소리에서 키워드까지", 44, AY + 24, size=14.5,
     weight="700", anchor="start")

# who-owns-what bands
box(36, AY + 36, 700, 20, ANALOG_BG, ANALOG_ED, r=5, sw=1)
text("동료 담당 — 아날로그 회로 (PCB)", 386, AY + 50, size=11.5,
     fill="#8a6410", weight="600", limit=690)
box(748, AY + 36, W - 784, 20, FPGA_BG, FPGA_ED, r=5, sw=1)
text("우리 담당 — FPGA (KC705)", 748 + (W - 784) / 2, AY + 50, size=11.5,
     fill="#2f5aa0", weight="600", limit=440)

CH = [
    ("마이크", "MM20-33366", ANALOG_BG, ANALOG_ED),
    ("프리앰프", "×10", ANALOG_BG, ANALOG_ED),
    (f"GIC 필터 ×{NCH}", f"{FMIN:.0f}–{FMAX:.0f} Hz", ANALOG_BG, ANALOG_ED),
    (f"검출기 ×{NCH}", "정류+평활", ANALOG_BG, ANALOG_ED),
    ("★ 분기점", "트랙 1 / 2", "#fff8e6", "#c9902a"),
    (f"비교기 ×{NCH}", "→ 0/1 펄스", ANALOG_BG, ANALOG_ED),
    ("동기화 + OR", f"{WIN_MS:.0f} ms 창", FPGA_BG, FPGA_ED),
    ("BinaryMatchboxNet", f"[{NCH},{TPAD}] 이진", FPGA_BG, FPGA_ED),
    ("12 클래스", "yes/no/…", FPGA_BG, FPGA_ED),
]
BW, GAP, BY, BH = 120, 8.5, AY + 68, 60
x0 = 36
for i, (t1, t2, bg, ed) in enumerate(CH):
    x = x0 + i * (BW + GAP)
    box(x, BY, BW, BH, bg, ed, r=8)
    text(t1, x + BW / 2, BY + 25, size=11.5, weight="600", limit=BW - 8)
    text(t2, x + BW / 2, BY + 43, size=10.5, fill=MUTE, limit=BW - 8)
    if i < len(CH) - 1:
        arrow(x + BW + 1, BY + BH / 2, x + BW + GAP - 1, BY + BH / 2, sw=1.5)

# the boundary, between comparator (i=5) and sync (i=6)
bx = x0 + 6 * (BW + GAP) - GAP / 2
add(f'<line x1="{bx}" y1="{AY + 32}" x2="{bx}" y2="{AY + AH - 12}" '
    f'stroke="{INK}" stroke-width="2.4" stroke-dasharray="7 4"/>')
text("경 계", bx, AY + AH - 26, size=12, weight="700", limit=60)
text(f"{NCH}가닥 + 프레임 타이밍  ·  그 외에는 아무것도 넘어오지 않는다",
     bx, AY + AH - 10, size=11, fill=MUTE, limit=520)

# ══ B. the two tracks ═════════════════════════════════════════════════════
BY0, BH0 = 280, 214
box(24, BY0, W - 48, BH0, "#ffffff", LINE, r=12, sw=1.2)
text("B.  두 트랙 — 갈리는 곳은 ★ 한 군데뿐이다", 44, BY0 + 24, size=14.5,
     weight="700", anchor="start")
text("검출기 16채널을 비교기에 어떻게 물리느냐. 그 앞뒤는 완전히 같다.",
     44, BY0 + 43, size=11.5, fill=MUTE, anchor="start", limit=700)

TRK = [
    (T1_BG, T1_ED, "트랙 1 — xlse (다이오드-OR)", "xl_g12",
     ["다이오드 16개 → 한 노드에서 wired-OR", "버퍼 1개 + 공유 분압 1벌 → V_ref",
      "다이오드가 만드는 것은 max 가 아니라 log-sum-exp"],
     "음량 방어: 구조적 부분 불변 + 게인 증강",
     "벤치마크 0.8445 · 배치 추정 ≈0.69 (측정 완료)", DONE_BG, DONE_ED),
    (T2_BG, T2_ED, "트랙 2 — fixed (다이오드-OR 없음)", "fx_g12",
     ["다이오드 0 · 버퍼 0 · 공유 노드 없음", "채널마다 자기 분압 → 절대 임계 16벌",
      "정규화 자체가 없다"],
     "음량 방어: 게인 증강뿐 (분모가 없다)",
     "재측정 필요 — 기록된 0.726 은 낡았다", WAIT_BG, WAIT_ED),
]
TW_, TH_ = (W - 48 - 24 - 14) / 2, 138
for i, (bg, ed, name, tag, bullets, defense, verdict, vbg, ved) in enumerate(TRK):
    x = 36 + i * (TW_ + 14)
    y = BY0 + 56
    box(x, y, TW_, TH_, bg, ed, r=9)
    text(name, x + 14, y + 23, size=13, weight="700", anchor="start",
         limit=TW_ - 100)
    text(tag, x + TW_ - 14, y + 23, size=11.5, fill=MUTE, anchor="end",
         font=MONO, limit=90)
    for j, b in enumerate(bullets):
        text("·  " + b, x + 14, y + 46 + j * 17, size=11.5, anchor="start",
             limit=TW_ - 24)
    text(defense, x + 14, y + 105, size=11.5, weight="600", anchor="start",
         limit=TW_ - 24)
    box(x + 12, y + 114, TW_ - 24, 20, vbg, ved, r=5, sw=1)
    text(verdict, x + TW_ / 2, y + 128, size=11, fill=INK, limit=TW_ - 34)

text("공통 잣대:  §6f  gain_sweep()  — 화자가 마이크에서 멀어지고 가까워지는 축. "
     "정규화 방식과 무관하게 물리적 의미가 같다.",
     W / 2, BY0 + BH0 - 12, size=11.5, fill=MUTE, limit=W - 90)

# ══ C. why the RTL does not move ══════════════════════════════════════════
CY, CHH = 514, 116
box(24, CY, W - 48, CHH, "#f7fbf8", DONE_ED, r=12, sw=1.4)
text("C.  경계 계약 (docs/ICD.md) — 아날로그가 바뀌어도 RTL 은 안 바뀐다",
     44, CY + 25, size=14.5, weight="700", anchor="start")

LEFT = ["필터 f_c / Q", "채널 threshold (R7/R8)", "프리앰프 이득 (frac)",
        "V_ref 오프셋 (δ)", "검출기 τ", "마이크 교체"]
RIGHT = [("채널 수 16 → N", "N_CH"), ("비교기 극성 반전", "CMP_INVERT"),
         ("프레임 폭 10 ms → x", "FRAME_CYCLES")]

box(40, CY + 36, 520, 68, "#ffffff", LINE, r=8, sw=1)
text("이것들이 바뀌면 →  가중치만 재학습, RTL 무수정", 300, CY + 54,
     size=12, weight="600", fill="#2f7a4c", limit=505)
for j, s in enumerate(LEFT):
    text("· " + s, 54 + (j % 3) * 172, CY + 76 + (j // 3) * 17, size=11,
         anchor="start", limit=168)

box(578, CY + 36, 300, 68, "#ffffff", LINE, r=8, sw=1)
text("이것들만 →  파라미터 1줄", 728, CY + 54, size=12, weight="600",
     limit=285)
for j, (s, p) in enumerate(RIGHT):
    text("· " + s, 592, CY + 74 + j * 15, size=10.5, anchor="start", limit=190)
    text(p, 866, CY + 74 + j * 15, size=10, fill=MUTE, anchor="end",
         font=MONO, limit=110)

box(896, CY + 36, W - 24 - 896 - 16, 68, "#ffffff", LINE, r=8, sw=1)
text("그래서 지키는 규칙", 896 + (W - 24 - 896 - 16) / 2, CY + 54, size=12,
     weight="600", limit=280)
for j, s in enumerate(["가중치 → BRAM $readmemh", "치수 → 매니페스트가 생성",
                       "로직에 상수로 박지 않는다"]):
    text("· " + s, 910, CY + 74 + j * 15, size=10.5, anchor="start", limit=280)

# ══ D. the work sequence ══════════════════════════════════════════════════
DY, DHH = 648, 304
box(24, DY, W - 48, DHH, "#ffffff", LINE, r=12, sw=1.2)
text("D.  진행 순서 — FPGA 트랙", 44, DY + 25, size=14.5, weight="700",
     anchor="start")
text("가로 = 의존 순서. 왼쪽이 끝나야 오른쪽이 시작된다.", 44, DY + 44,
     size=11.5, fill=MUTE, anchor="start", limit=600)

STEPS = [
    ("P5-0", "경계 규약", "docs/ICD.md", "동료와 나눌 계약", "done"),
    ("P5-1", "비트폭 실측", "export/ranges.py", "노트북 §8b — 다음 실행", "now"),
    ("P5-c", "매니페스트\n+ ROM .hex", "parameters.vh", "치수 자동 생성", "wait"),
    ("P5-d", "골든 벡터", "층별 중간 활성", "없으면 디버깅 불가", "wait"),
    ("P5-e", "RTL", "PE → TCS → top", "바텀업", "wait"),
    ("P5-f", "검증", "비트 정확 일치", "골든 벡터 대조", "wait"),
    ("P5-g", "합성·보드", "타이밍 · 비트스트림", "KC705", "wait"),
]
SW_, SH_, SG = 152, 108, 12.4
sx0 = 40
sy = DY + 62
STY = {"done": (DONE_BG, DONE_ED, "완료"), "now": (NOW_BG, NOW_ED, "▶ 다음"),
       "wait": (WAIT_BG, WAIT_ED, "대기")}
for i, (pid, title, sub, note, st) in enumerate(STEPS):
    bg, ed, lab = STY[st]
    x = sx0 + i * (SW_ + SG)
    box(x, sy, SW_, SH_, bg, ed, r=9, sw=1.9 if st == "now" else 1.3)
    text(pid, x + 12, sy + 22, size=11.5, fill=MUTE, anchor="start",
         font=MONO, weight="600", limit=60)
    box(x + SW_ - 62, sy + 10, 52, 17, "#ffffff", ed, r=8, sw=1)
    text(lab, x + SW_ - 36, sy + 22, size=9.5, fill=ed, weight="700", limit=48)
    for j, ln in enumerate(title.split("\n")):
        text(ln, x + SW_ / 2, sy + 48 + j * 15, size=13, weight="700",
             limit=SW_ - 16)
    text(sub, x + SW_ / 2, sy + 80, size=10, fill=MUTE, font=MONO,
         limit=SW_ - 10)
    text(note, x + SW_ / 2, sy + 96, size=10, fill=MUTE, limit=SW_ - 10)
    if i < len(STEPS) - 1:
        arrow(x + SW_ + 1, sy + SH_ / 2, x + SW_ + SG - 1, sy + SH_ / 2,
              sw=1.6)

# what feeds what
fy = sy + SH_ + 30
add(f'<path d="M{sx0 + SW_ / 2 + SW_ + SG} {sy + SH_} '
    f'L{sx0 + SW_ / 2 + SW_ + SG} {fy} '
    f'L{sx0 + 2 * (SW_ + SG) + SW_ / 2} {fy} '
    f'L{sx0 + 2 * (SW_ + SG) + SW_ / 2} {sy + SH_ + 4}" '
    f'fill="none" stroke="{DONE_ED}" stroke-width="1.8" '
    f'stroke-dasharray="5 3" marker-end="url(#ag)"/>')
# kept clear of the dashed path (which spans x 280..445) -- anchored to its
# right so the arrow cannot run through the glyphs
text("실측한 비트폭이 매니페스트의 입력이 된다 — 재지 않고 최악값으로 잡으면 "
     "데이터패스 전체가 과설계된다",
     sx0 + 2 * (SW_ + SG) + SW_ / 2 + 40, fy + 4, size=11, fill=DONE_ED,
     anchor="start", limit=W - 560)

# parallel track
py = DY + DHH - 52
box(40, py, W - 104, 38, "#fdfaf3", ANALOG_ED, r=8, sw=1.2, dash="6 4")
text("병행 (학습 필요 없음):", 56, py + 24, size=11.5, weight="700",
     anchor="start", fill="#8a6410", limit=170)
text("트랙 2 학습 fx_g12  ·  §6f gain_sweep 두 트랙 비교  ·  dead_channels "
     "진단  ·  P3-b 최대 확신도 선택  ·  동료 답변 대기: ICD §7 (비교기 레벨, "
     "핀↔채널 매핑)",
     232, py + 24, size=11, anchor="start", limit=W - 300)

add("</svg>")

# ── layout self-check ─────────────────────────────────────────────────────
bad = [b for b in BOXES if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W
       or b[1] + b[3] > H]
if bad:
    raise ValueError(f"{len(bad)} box(es) outside the {W}x{H} canvas: {bad[:3]}")
for s, x, y, size, anchor in TEXTS:
    w = tw(s, size)
    lo = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    if lo < -1 or lo + w > W + 1 or y > H:
        raise ValueError(f"text outside canvas: {s!r} at x={lo:.0f}..{lo + w:.0f}")

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT}  ({W}x{H}, {len(BOXES)} boxes, {len(TEXTS)} texts)")
print(f"facts: {NCH}ch {FMIN:.0f}-{FMAX:.0f} Hz | "
      f"{NATIVE} frames -> T={TPAD} (pad {PAD_L}/{PAD_L}) @ {WIN_MS:.0f} ms")

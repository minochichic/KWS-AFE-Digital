"""Draw proposal/artifacts/fpga_blocks.svg -- the KC705 block diagram.

Two things this page has to get across, because both overturn an assumption we
were carrying:

  * the accelerator is FOLDED, not fully unrolled. 10 Hz needs 0.6 MAC/cycle at
    100 MHz, so even the "fast" design uses 1.3% of the KC705. Fully unrolling
    6 M MACs would need millions of LUTs and does not fit anyway.
  * classification is a free-running SLIDING window, not wake-on-event. The
    relative threshold fires on 95% of frames even in quiet, so "first event =
    speech started" -- which worked for Cerutti's fixed threshold -- is dead.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "fpga_blocks.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
BG_B, LN_B = "#eff6ff", "#93c5fd"
BG_G, LN_G = "#dcfce7", "#86efac"
BG_O, LN_O = "#fef9f3", "#fdba74"
W, H = 1300, 900


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=12, fill=INK, anchor="middle", weight=None, mono=False):
    f = ' font-family="ui-monospace,Menlo,monospace"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{w}{f}>{esc(s)}</text>')


def box(x, y, w, h, fill="#ffffff", stroke=WIRE, rx=6, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arw(x1, y1, x2, y2, color=WIRE, sw=1.8):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#a)"/>')


def wire(pts, color=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x},{y}" for x, y in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}"{d}/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{WIRE}"/></marker>'
     '</defs>',
     t(W/2, 34, "FPGA 블록 구성 — AMD Kintex-7 KC705", 22, INK, weight="700"),
     t(W/2, 57, "접은(folded) 가속기 + 자유 구동 슬라이딩 창. "
                "완전 펼침은 불필요하고 애초에 들어가지도 않는다.", 12.5, MUTE)]

# ── analog side (outside the FPGA) ─────────────────────────────────────────
s += [box(40, 96, 150, 108, "#f1f5f9", "#94a3b8", dash="5 3"),
      t(115, 118, "아날로그 AFE", 12.5, MUTE, weight="700"),
      t(115, 137, "(FPGA 밖)", 10, MUTE),
      t(115, 160, "비교기 16개", 11.5, ORG, weight="700"),
      t(115, 177, "0/1 라인 ×16", 10, MUTE),
      t(115, 194, "→ proposal/THRESHOLD.md", 9, MUTE),
      arw(190, 150, 232, 150)]

# ── FPGA outline ───────────────────────────────────────────────────────────
s += [box(232, 80, 1028, 560, "#ffffff", INK, rx=10, sw=2),
      t(250, 102, "XC7K325T", 13, INK, "start", weight="700"),
      t(322, 102, "— 아래 설계는 LUT 1.3% / BRAM 3.3%만 쓴다", 10.5, MUTE, "start")]

# ── A. frame builder ───────────────────────────────────────────────────────
ax, ay = 256, 122
s += [box(ax, ay, 300, 250, BG_O, LN_O),
      t(ax + 150, ay + 24, "A. 프레임 조립", 13.5, ORG, weight="700"),
      t(ax + 150, ay + 42, "AFE 비트열 → [16, 128] 이미지", 10.5, MUTE)]

sub = [("동기화 2FF ×16", "비동기 입력 → 클럭 도메인", ay + 56),
       ("10 ms OR 누산", "창 안에 한 번이라도 1이면 1", ay + 110),
       ("시프트 레지스터", "16비트 × 100단 = 1600 FF", ay + 164)]
for i, (title, note, yy) in enumerate(sub):
    s += [box(ax + 18, yy, 264, 44, "#ffffff", LN_O, sw=1.1),
          t(ax + 150, yy + 19, title, 11.5, INK, weight="700"),
          t(ax + 150, yy + 34, note, 9.5, MUTE)]
    if i < 2:
        s.append(arw(ax + 150, yy + 44, ax + 150, yy + 54))
s += [t(ax + 150, ay + 226, "패딩 28칸은 저장 안 한다 — 항상 −1인 상수", 10,
        ORG, weight="700"),
      t(ax + 150, ay + 241, "주소 <14 또는 ≥114 이면 −1 반환", 9.5, MUTE)]

# ── B. accelerator ─────────────────────────────────────────────────────────
bx = 576
s += [box(bx, ay, 330, 250, BG_B, LN_B),
      t(bx + 165, ay + 24, "B. 접은 가속기", 13.5, BLU, weight="700"),
      t(bx + 165, ay + 42, "6.0 M MAC / 추론", 10.5, MUTE)]

pe = [("XNOR-popcount PE", "이진 3.42 M MAC (57%)", "P=64 → 53.5 k cyc"),
      ("부호 누산기", "conv1 1.44 M — 곱셈기 없음", "입력이 ±1이라"),
      ("DSP48 MAC", "conv3·conv4 1.15 M (int8)", "진짜 곱셈은 여기뿐")]
for i, (title, mid, note) in enumerate(pe):
    yy = ay + 58 + i * 58
    s += [box(bx + 16, yy, 298, 50, "#ffffff", LN_B, sw=1.1),
          t(bx + 165, yy + 17, title, 11.5, INK, weight="700"),
          t(bx + 165, yy + 31, mid, 9.5, MUTE),
          t(bx + 165, yy + 44, note, 9.5, BLU)]
s.append(t(bx + 165, ay + 240, "10 Hz에 필요한 건 0.6 MAC/cycle뿐", 10.5, BLU,
           weight="700"))
s.append(arw(ax + 300, ay + 150, bx, ay + 150))
s.append(t((ax + 300 + bx) / 2, ay + 142, "[16,128]", 9.5, ORG))

# ── C. memory ──────────────────────────────────────────────────────────────
cx = 926
s += [box(cx, ay, 314, 250, BG_G, LN_G),
      t(cx + 157, ay + 24, "C. 온칩 메모리", 13.5, GRN, weight="700"),
      t(cx + 157, ay + 42, "외부 메모리 불필요", 10.5, MUTE)]
mem = [("이진 가중치 (bit-packed)", "6.53 KB"),
       ("int8 가중치", "38.00 KB"),
       ("fixed 가중치", "3.02 KB"),
       ("활성 핑퐁 버퍼", "16.00 KB"),
       ("정수 threshold + 극성", "채널당 2개")]
for i, (a, b) in enumerate(mem):
    yy = ay + 64 + i * 22
    s += [t(cx + 18, yy, a, 10.5, INK, "start"),
          t(cx + 296, yy, b, 10.5, INK, "end", weight="700")]
s += [wire([(cx + 18, ay + 180), (cx + 296, ay + 180)], LN_G, 1.2),
      t(cx + 18, ay + 198, "합계", 11, GRN, "start", weight="700"),
      t(cx + 296, ay + 198, "63.6 KB", 11, GRN, "end", weight="700"),
      t(cx + 157, ay + 222, "KC705 BRAM의 3.3%", 10, MUTE),
      t(cx + 157, ay + 240, "$readmemh ROM으로 초기화", 9.5, MUTE)]
s.append(wire([(cx, ay + 150), (bx + 330, ay + 150)], WIRE, 1.6))

# ── D. sequencer / decision ────────────────────────────────────────────────
dy = 392
s += [box(ax, dy, 654, 226, "#faf5ff", "#d8b4fe"),
      t(ax + 327, dy + 24, "D. 시퀀서 + 판정", 13.5, PUR, weight="700"),
      t(ax + 327, dy + 42, "자유 구동 — 웨이크온이벤트가 아니다", 10.5, MUTE)]

tl = [("10 ms 프레임 타이머", "시프트 레지스터를 한 칸 민다"),
      ("100 ms마다 스냅샷", "추론 5.2 ms < 프레임 10 ms → 더블버퍼 불필요"),
      ("층 시퀀서 FSM", "스테이지 순회, 주소 생성"),
      ("posterior 평활 + 불응기", "한 발화가 창 8개에 걸린다 → 억제 필수")]
for i, (a, b) in enumerate(tl):
    yy = dy + 56 + i * 40
    s += [box(ax + 18, yy, 618, 34, "#ffffff", "#d8b4fe", sw=1.1),
          t(ax + 30, yy + 21, a, 11.5, INK, "start", weight="700"),
          t(ax + 626, yy + 21, b, 10, MUTE, "end")]
s.append(wire([(bx + 165, ay + 250), (bx + 165, dy)], WIRE, 1.6))

# ── E. output ──────────────────────────────────────────────────────────────
s += [box(cx, dy, 314, 226, BG_G, LN_G),
      t(cx + 157, dy + 24, "E. 출력", 13.5, GRN, weight="700"),
      t(cx + 157, dy + 52, "12-class argmax", 12, INK, weight="700"),
      t(cx + 157, dy + 72, "키워드 10 + silence + unknown", 9.5, MUTE),
      t(cx + 157, dy + 104, "UART / LED / GPIO", 11.5, INK),
      t(cx + 157, dy + 124, "보드 검증용", 9.5, MUTE)]
s += [box(cx + 18, dy + 146, 278, 62, "#ffffff", LN_G, sw=1.1),
      t(cx + 157, dy + 166, "디버그 여유가 넘친다", 11, GRN, weight="700"),
      t(cx + 157, dy + 182, "LUT 98%가 남으므로 ILA·중간값", 9.5, MUTE),
      t(cx + 157, dy + 196, "덤프를 마음껏 넣을 수 있다", 9.5, MUTE)]
s.append(arw(ax + 654, dy + 113, cx, dy + 113))

# ── timing strip ───────────────────────────────────────────────────────────
ty = 664
s += [box(40, ty, 1220, 128, "#f8fafc", "#cbd5e1"),
      t(60, ty + 24, "타이밍 — 슬라이딩 창", 13, INK, "start", weight="700"),
      t(60, ty + 42, "단어 하나(270 ms)가 창 8개에 걸린다. 그래서 불응기가 필요하다.",
        10.5, MUTE, "start")]
x0, xw = 250, 900
for i in range(9):
    x = x0 + i * (xw / 9)
    w_ = xw / 9 - 6
    hit = 3 <= i <= 5
    s += [box(x, ty + 58, w_, 26, "#fecaca" if hit else "#ffffff",
              RED if hit else "#cbd5e1", rx=3, sw=1.1),
          t(x + w_ / 2, ty + 76, f"{i*100}", 9.5, RED if hit else MUTE)]
s += [t(60, ty + 76, "창 시작(ms)", 10.5, INK, "start"),
      t(60, ty + 104, "판정", 10.5, INK, "start"),
      t(x0 + xw / 2, ty + 104, "단어가 학습 분포와 맞는 창은 가운데 2~3개뿐 "
        "— ±300 ms에서는 −11~17pp", 10.5, RED, weight="700"),
      t(x0 + xw / 2, ty + 119, "→ time-shift 증강으로 곡선을 평평하게 만드는 것이 "
        "근본 해결 (탐색 중)", 10, MUTE)]

s.append(t(40, H - 22, "연산량 근거: troubleshooting/scripts/count_ops.py · "
                       "리소스: estimate_resources.py · 이동 곡선: "
                       "notebooks/workbench.ipynb §5", 10, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

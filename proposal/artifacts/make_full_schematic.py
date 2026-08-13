"""Draw proposal/artifacts/afe_full.svg -- the whole AFE as it now stands.

Everything on this page is either a measured component value or a trained
constant, and each is labelled with which. It exists so the analog side has one
sheet to build from instead of five documents to reconcile.

What changed from the board that exists today: only the comparator reference.
The old design put a fixed R7/R8 divider across the full 1.8 V rail per channel.
The new one derives the reference from a diode-OR of all sixteen envelopes, so
the threshold follows the loudest channel instead of a fixed voltage -- and it
costs less power, because the divider now sits across V_max - V_ref (~0.1 V)
rather than the whole rail.

delta = 0 is the other change, and it simplifies rather than complicates: V_ref
goes to the detector quiescent point, so the reference generator has no offset
to trim. That was measured, not assumed -- see docs/EXPERIMENT_MAP.md A-3.
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / "afe_full.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,monospace"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1560, 2000

DESIGN = np.loadtxt(ROOT / "analog/AFE/artifacts/filterbank_design.csv",
                    delimiter=",", skiprows=1)
ALPHA = [0.8745, 0.7804, 0.4788, 0.4482, 0.5423, 0.4072, 0.1725, 0.1326,
         0.1799, 0.1242, 0.2096, 0.2005, 0.1244, 0.0788, 0.0572, 0.3787]
RTOT_K = 1000.0                    # Ra + Rb, kohm
QUIESCENT = 917.8


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=12, fill=INK, anchor="middle", weight=None, mono=False):
    f = f' font-family="{MONO}"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{w}{f}>{esc(s)}</text>')


def box(x, y, w, h, fill="#ffffff", stroke=WIRE, rx=6, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def wire(pts, c=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x},{y}" for x, y in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{c}" '
            f'stroke-width="{sw}"{d}/>')


def arw(x1, y1, x2, y2, c=WIRE, sw=1.8):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
            f'stroke-width="{sw}" marker-end="url(#a)"/>')


def dot(x, y, c=WIRE, r=3.2):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'


def res(x, y, w=46, h=15, lab="", sub="", vert=False, c=WIRE):
    s = []
    if vert:
        s.append(f'<rect x="{x - h/2}" y="{y}" width="{h}" height="{w}" rx="2" '
                 f'fill="#ffffff" stroke="{c}" stroke-width="1.5"/>')
        if lab:
            s.append(t(x + h / 2 + 6, y + w / 2 - 1, lab, 10.5, INK, "start",
                       weight="700"))
        if sub:
            s.append(t(x + h / 2 + 6, y + w / 2 + 12, sub, 9.5, MUTE, "start"))
    else:
        s.append(f'<rect x="{x}" y="{y - h/2}" width="{w}" height="{h}" rx="2" '
                 f'fill="#ffffff" stroke="{c}" stroke-width="1.5"/>')
        if lab:
            s.append(t(x + w / 2, y - h / 2 - 5, lab, 10.5, INK, weight="700"))
        if sub:
            s.append(t(x + w / 2, y + h / 2 + 11, sub, 9.5, MUTE))
    return "".join(s)


def cap(x, y, c=WIRE, lab="", sub=""):
    s = [f'<line x1="{x-9}" y1="{y-11}" x2="{x-9}" y2="{y+11}" stroke="{c}" '
         f'stroke-width="2.2"/>',
         f'<line x1="{x+1}" y1="{y-11}" x2="{x+1}" y2="{y+11}" stroke="{c}" '
         f'stroke-width="2.2"/>']
    if lab:
        s.append(t(x - 4, y - 18, lab, 10.5, INK, weight="700"))
    if sub:
        s.append(t(x - 4, y + 30, sub, 9.5, MUTE))
    return "".join(s)


def diode(x, y, size=8, c=WIRE):
    return (f'<polygon points="{x-size},{y-size} {x-size},{y+size} {x+size},{y}" '
            f'fill="#ffffff" stroke="{c}" stroke-width="1.5"/>'
            f'<line x1="{x+size}" y1="{y-size}" x2="{x+size}" y2="{y+size}" '
            f'stroke="{c}" stroke-width="2.2"/>')


def amp(x, y, w=52, h=48, lab="", plus_top=True, c=WIRE):
    s = [f'<polygon points="{x},{y-h/2} {x},{y+h/2} {x+w},{y}" fill="#ffffff" '
         f'stroke="{c}" stroke-width="1.7"/>']
    a, b = ("+", "−") if plus_top else ("−", "+")
    s += [t(x + 11, y - h / 4 + 4, a, 12, INK),
          t(x + 11, y + h / 4 + 5, b, 12, INK)]
    if lab:
        s.append(t(x + w / 2, y + h / 2 + 15, lab, 10, MUTE))
    return "".join(s)


def gnd(x, y, c=WIRE):
    return (f'<line x1="{x-10}" y1="{y}" x2="{x+10}" y2="{y}" stroke="{c}" '
            f'stroke-width="2"/><line x1="{x-6}" y1="{y+4}" x2="{x+6}" '
            f'y2="{y+4}" stroke="{c}" stroke-width="2"/>'
            f'<line x1="{x-2}" y1="{y+8}" x2="{x+2}" y2="{y+8}" stroke="{c}" '
            f'stroke-width="2"/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{WIRE}"/></marker>'
     '</defs>',
     t(W / 2, 38, "AFE 전체 회로 — 현재 확정 상태", 25, INK, weight="700"),
     t(W / 2, 62, "부품값은 SPICE 실측, α 는 학습 결과. 단일 1.8 V 전원, "
                  "마이크만 0.9 V.", 13, MUTE),
     t(W / 2, 80, "기존 보드에서 바뀌는 곳은 §C 비교기 기준전압 하나뿐이다.",
       12, RED, weight="700")]

# ══ A. one channel, component level ════════════════════════════════════════
ay = 100
s += [box(30, ay, 1500, 300, "#f8fafc", "#cbd5e1"),
      t(52, ay + 26, "A. 신호 체인 — 채널 하나 (×16 동일 구조, 부품값만 다름)",
        16, INK, "start", weight="700"),
      t(52, ay + 46, "굵은 선 = 신호 경로. 값은 analog/AFE/artifacts/"
        "component_table.md", 10.5, MUTE, "start")]

Y = ay + 185
# mic
s += [box(60, Y - 40, 96, 80, "#ffffff", WIRE, sw=1.2),
      t(108, Y - 16, "MIC", 12, INK, weight="700"),
      t(108, Y + 2, "Knowles", 9.5, MUTE),
      t(108, Y + 15, "MM20-33366", 9, MUTE),
      t(108, Y + 31, "0.9 V 전용", 9.5, ORG, weight="700"),
      wire([(156, Y), (200, Y)], WIRE, 2.4)]

# preamp: non-inverting, feedback drawn clear of the signal path
s += [amp(200, Y, lab=""),
      t(226, Y + 40, "OPA379", 9.5, MUTE),
      wire([(200, Y - 12), (188, Y - 12), (188, Y - 62)], WIRE, 1.4),
      dot(188, Y - 62),
      res(196, Y - 62, 56, 13, "Rf 90 k", c=WIRE),
      wire([(252, Y - 62), (268, Y - 62), (268, Y)]),
      res(126, Y - 62, 50, 13, "Rg 10 k", c=WIRE),
      wire([(126, Y - 62), (106, Y - 62)]), gnd(106, Y - 54),
      wire([(252, Y), (300, Y)], WIRE, 2.4), dot(268, Y),
      t(212, Y + 58, "G = 1 + Rf/Rg = ×10", 11, INK, weight="700"),
      t(212, Y + 73, "GBW·헤드룸 양쪽에서 ×10 이 상한", 9.5, MUTE)]

# GIC bandpass
s += [box(300, Y - 50, 166, 100, "#eff6ff", "#93c5fd", sw=1.3),
      t(383, Y - 30, "2차 GIC 밴드패스", 11.5, BLU, weight="700"),
      t(383, Y - 10, "R_A  10.6 ~ 12.5 k", 10, INK, mono=True),
      t(383, Y + 5, "C    1.86 ~ 76.6 n", 10, INK, mono=True),
      t(383, Y + 20, "R1   16.5 ~ 59.3 k", 10, INK, mono=True),
      t(383, Y + 40, "R2 = R3 = 100 k (공통)", 9.5, MUTE),
      wire([(466, Y), (512, Y)], WIRE, 2.4),
      t(383, Y + 66, "f_c = 1/(2π R_A C),   Q = R1/R_A", 10.5, BLU,
        weight="700"),
      t(383, Y + 81, "이득 2.00 — .ac 실측이 공식과 일치", 9.5, MUTE)]

# active detector
s += [box(512, Y - 50, 232, 100, "#f0fdf4", "#86efac", sw=1.3),
      t(628, Y - 30, "능동 검출기 (정밀정류 + 평활)", 11.5, GRN, weight="700"),
      diode(538, Y - 4, 6, GRN), diode(538, Y + 12, 6, GRN),
      t(538, Y + 30, "D1/D2", 9, GRN, weight="700"),
      t(566, Y - 8, "BAT54  R4 10 k  R5 47 k", 9.5, INK, "start", mono=True),
      t(566, Y + 7, "R6 8.25 k       C3 100 n", 9.5, INK, "start", mono=True),
      t(646, Y + 32, "G = R5/R4 = 4.7   τ = R5·C3 = 4.7 ms", 9.5, GRN,
        weight="700"),
      wire([(744, Y), (800, Y)], WIRE, 2.4), dot(800, Y),
      t(628, Y + 66, "실측 τ_방전 4.62 ms, 충전 t90 0.65 ms", 9.5, MUTE),
      t(628, Y + 81, "⚠️ 데드존 5.9~12.1 mV (저GBW 크로스오버) — 모델 미반영",
        9.5, ORG)]

# envelope node
s += [t(806, Y - 40, "V_env,c", 13.5, PUR, "start", weight="700"),
      t(806, Y - 24, "정지점 917.8 mV 위로 상승", 9.5, MUTE, "start"),
      t(806, Y - 10, "66 dB SPL 에서 33 mV", 10.5, PUR, "start", weight="700")]

# comparator
s += [wire([(800, Y), (930, Y)], WIRE, 2.4),
      amp(930, Y, 58, 54, lab=""),
      t(959, Y + 44, "LPV7215   0.58 µA", 9.5, MUTE),
      wire([(988, Y), (1044, Y)], WIRE, 2.4),
      t(1050, Y - 6, "→ FPGA", 12.5, INK, "start", weight="700"),
      t(1050, Y + 10, "1 비트 / 채널", 9.5, MUTE, "start"),
      wire([(910, Y + 14), (884, Y + 14), (884, Y + 76)], RED, 1.8),
      t(880, Y + 80, "V_thr,c  ← §C", 11, RED, "end", weight="700")]

s += [box(1130, Y - 50, 380, 100, "#ffffff", "#cbd5e1", sw=1.1),
      t(1142, Y - 28, "채널당: GIC op-amp 2 + 검출기 1 + 비교기 1", 10.5, INK,
        "start"),
      t(1142, Y - 10, "16채널 = op-amp 48개 + 비교기 16개", 10.5, INK, "start"),
      t(1142, Y + 10, "출력 16 라인, 각 0/1 — ADC 없음", 10.5, GRN, "start",
        weight="700"),
      t(1142, Y + 30, "FPGA 가 10 ms 창으로 OR 누산 → [16,128]", 9.5, MUTE,
        "start")]

# ══ B. shared: diode-OR + buffer ═══════════════════════════════════════════
by = 420
s += [box(30, by, 740, 330, "#fff1f2", "#fca5a5"),
      t(52, by + 26, "B. 다이오드-OR + 버퍼 (뱅크 전체에 1조)", 16, RED,
        "start", weight="700"),
      t(52, by + 46, "16채널 엔벨로프에서 '지금 가장 큰 것' 을 만든다", 10.5,
        MUTE, "start")]

BUS = 300
for i, lab in enumerate(("V_env,0", "V_env,1", "V_env,15")):
    yy = by + 84 + i * 34
    s += [t(66, yy + 4, lab, 10.5, PUR, "start", weight="700"),
          wire([(130, yy), (BUS - 22, yy)]),
          diode(BUS - 12, yy, 7, RED),
          wire([(BUS - 4, yy), (BUS, yy)]), dot(BUS, yy)]
s += [t(150, by + 148, "⋮  16채널 전부", 10.5, MUTE, "start"),
      wire([(BUS, by + 84), (BUS, by + 210)], RED, 2.2),
      t(BUS + 14, by + 100, "BAT54 ×16", 10.5, RED, "start", weight="700"),
      t(BUS + 14, by + 114, "(검출기와 같은 부품)", 9, MUTE, "start")]
# current sink
s += [f'<circle cx="{BUS}" cy="{by+228}" r="12" fill="#ffffff" stroke="{RED}" '
      f'stroke-width="1.5"/>',
      f'<line x1="{BUS}" y1="{by+222}" x2="{BUS}" y2="{by+234}" stroke="{RED}" '
      f'stroke-width="1.5"/>',
      f'<polygon points="{BUS-4},{by+228} {BUS+4},{by+228} {BUS},{by+235}" '
      f'fill="{RED}"/>',
      wire([(BUS, by + 240), (BUS, by + 252)], RED, 2), gnd(BUS, by + 252, RED),
      t(BUS - 22, by + 232, "정전류 싱크 ×1", 10, RED, "end", weight="700")]

s += [wire([(BUS, by + 84), (400, by + 84), (400, by + 118), (430, by + 118)],
           RED, 2.2),
      t(360, by + 76, "V_or = LSE − V_d", 10.5, RED, weight="700"),
      amp(430, by + 130, 52, 48, lab=""),
      diode(500, by + 100, 7, RED),
      wire([(430, by + 142), (412, by + 142), (412, by + 100), (492, by + 100)],
           RED, 1.6),
      wire([(508, by + 100), (560, by + 100), (560, by + 130)], RED, 1.6),
      wire([(482, by + 130), (560, by + 130)], RED, 2.2), dot(560, by + 130),
      t(500, by + 84, "짝 다이오드", 9.5, RED, weight="700"),
      t(500, by + 72, "BAT54WT1 같은 다이", 9, MUTE),
      wire([(560, by + 130), (650, by + 130)], RED, 2.4),
      t(658, by + 126, "V_max", 14, RED, "start", weight="700"),
      t(658, by + 142, "(강하 보상 완료)", 9.5, MUTE, "start")]

s += [box(52, by + 268, 696, 50, "#ffffff", RED, sw=1.1),
      t(64, by + 288, "⚠️ 이건 max 가 아니라 soft max 다: "
        "V_or = max + T·ln k,  T = n·V_T = 26 mV", 11, RED, "start",
        weight="700"),
      t(64, by + 306, "다이오드가 지수라 진 채널도 샌다. 그 결과가 하드 max 보다 "
        "+6.7pp 좋았다 (docs/diagrams/17_loudness.svg)", 10, INK, "start")]

# ══ C. threshold divider ═══════════════════════════════════════════════════
s += [box(790, by, 740, 330, "#faf5ff", "#d8b4fe"),
      t(812, by + 26, "C. 채널별 분압 + 기준전압  ← 유일하게 바뀌는 곳", 16, PUR,
        "start", weight="700"),
      t(812, by + 46, "기존: R7/R8 이 1.8 V 레일 전체에 걸린 고정 분압", 10.5,
        MUTE, "start")]

DX = 900
s += [wire([(830, by + 96), (DX, by + 96)], RED, 2.2),
      t(824, by + 100, "V_max", 11, RED, "end", weight="700"),
      res(DX, by + 96, 50, 40, vert=True, lab="Ra", sub="= (1−α)·1 MΩ", c=PUR),
      wire([(DX, by + 146), (DX, by + 166)], PUR, 1.8), dot(DX, by + 166),
      res(DX, by + 166, 50, 40, vert=True, lab="Rb", sub="= α·1 MΩ", c=PUR),
      wire([(DX, by + 216), (DX, by + 240), (830, by + 240)], GRN, 2.2),
      t(824, by + 244, "V_ref", 11, GRN, "end", weight="700"),
      wire([(DX, by + 166), (1010, by + 166)], PUR, 2.2),
      t(1018, by + 162, "V_thr,c", 12.5, PUR, "start", weight="700"),
      t(1018, by + 178, "= α·V_max + (1−α)·V_ref", 9.5, MUTE, "start"),
      t(DX + 100, by + 96, "×16 (채널마다 다른 α)", 10.5, PUR, "start",
        weight="700")]

s += [box(812, by + 268, 700, 50, "#ffffff", GRN, sw=1.2),
      t(824, by + 288, "✅ δ = 0 확정 → V_ref = 검출기 정지점 917.8 mV", 12, GRN,
        "start", weight="700"),
      t(824, by + 306, "오프셋 트리밍 불필요. 복제 검출기(입력 개방)로 만들면 "
        "온도·전원 드리프트까지 따라간다 — 회로 측 선택", 10, INK, "start")]

# ══ D. the 16-channel table ════════════════════════════════════════════════
dy = 770
s += [box(30, dy, 1500, 480, "#ffffff", "#cbd5e1"),
      t(52, dy + 26, "D. 16채널 값 — 필터는 SPICE 설계, α 는 학습 결과 "
        "(xl_g12)", 16, INK, "start", weight="700"),
      t(52, dy + 46, "Ra + Rb = 1 MΩ 로 정규화. 전류 = (V_max−V_ref)/1 MΩ ≈ 0.1 µA/채널",
        10.5, MUTE, "start")]

hdr = [(90, "ch"), (185, "f_c [Hz]"), (290, "Q"), (400, "R_A [kΩ]"),
       (510, "C [nF]"), (620, "R1 [kΩ]"), (760, "α (학습)"),
       (890, "Rb [kΩ]"), (1010, "Ra [kΩ]")]
for x, lab in hdr:
    s.append(t(x, dy + 78, lab, 11, INK, "end", weight="700"))
s.append(wire([(50, dy + 86), (1040, dy + 86)], WIRE, 1.2))
for i in range(16):
    r = DESIGN[i]
    a = ALPHA[i]
    yy = dy + 106 + i * 22
    if i % 2 == 0:
        s.append(f'<rect x="50" y="{yy-15}" width="990" height="20" '
                 f'fill="{MUTE}" opacity="0.05"/>')
    vals = [(90, f"{i}"), (185, f"{r[2]:.0f}"), (290, f"{r[4]:.2f}"),
            (400, f"{r[6]/1e3:.2f}"), (510, f"{r[7]*1e9:.2f}"),
            (620, f"{r[8]/1e3:.1f}"), (760, f"{a:.4f}"),
            (890, f"{a*RTOT_K:.1f}"), (1010, f"{(1-a)*RTOT_K:.1f}")]
    for x, v in vals:
        col = PUR if x >= 760 else INK
        s.append(t(x, yy, v, 10.5, col, "end",
                   weight="700" if x == 760 else None, mono=True))

s += [box(1070, dy + 70, 440, 180, "#f0fdf4", "#86efac", sw=1.1),
      t(1090, dy + 92, "α 범위 0.057 ~ 0.874 — 전부 제작 가능", 12, GRN,
        "start", weight="700"),
      t(1090, dy + 112, "죽은 채널 0/16 (0 또는 1 에 붙은 것 없음)", 10.5, INK,
        "start"),
      t(1090, dy + 136, "α 가 낮은 채널(6~14, 1.3~5.8 kHz)은", 10.5, INK,
        "start"),
      t(1090, dy + 152, "임계가 낮다 = 잘 발화한다. 그 대역의", 10.5, INK,
        "start"),
      t(1090, dy + 168, "엔벨로프가 원래 작기 때문이다.", 10.5, INK, "start"),
      t(1090, dy + 194, "⚠️ α 는 최종 학습이 끝나면 다시 뽑아야 한다.", 10.5,
        ORG, "start", weight="700"),
      t(1090, dy + 210, "지금 값은 xl_g12 (게인 증강, 150 에폭).", 10, MUTE,
        "start"),
      t(1090, dy + 232, "반드시 effective_alpha() 로 읽을 것 — 원값은", 9.5,
        RED, "start"),
      t(1090, dy + 246, "straight-through clamp 때문에 [0,1] 밖으로 뜬다.", 9.5,
        RED, "start")]

s += [box(1070, dy + 268, 440, 190, "#fffbeb", "#fcd34d", sw=1.1),
      t(1090, dy + 290, "전력 수지 (제안 회로의 순증)", 12, ORG, "start",
        weight="700")]
pw = [("버퍼 op-amp ×1", "+13 µW"), ("OR 다이오드 ×16 + 싱크", "+3 µW"),
      ("새 분압 ×16 (~0.1 V)", "+3 µW"),
      ("기존 분압 ×16 (1.8 V 전체)", "−52 µW")]
for i, (a, b) in enumerate(pw):
    yy = dy + 314 + i * 20
    s += [t(1090, yy, a, 10.5, INK, "start"),
          t(1494, yy, b, 10.5, RED if b.startswith("+") else GRN, "end",
            weight="700")]
s += [wire([(1090, dy + 400), (1494, dy + 400)], "#fcd34d", 1.2),
      t(1090, dy + 418, "순증", 12, ORG, "start", weight="700"),
      t(1494, dy + 418, "≈ −33 µW", 13, GRN, "end", weight="700"),
      t(1090, dy + 440, "임계를 똑똑하게 만들면서 전력이 오히려 준다 —", 10, INK,
        "start"),
      t(1090, dy + 454, "분압이 레일 전체가 아니라 V_max−V_ref 에만 걸린다.",
        10, INK, "start")]

# ══ E. spec + open items ═══════════════════════════════════════════════════
ey = 1270
s += [box(30, ey, 740, 250, "#f0fdf4", "#86efac"),
      t(52, ey + 28, "E. 동료에게 넘길 스펙", 16, GRN, "start", weight="700")]
spec = [("요구 엔벨로프 상승", "66 dB SPL 에서 33 mV", True),
        ("= 프리앰프", "×10 (현재 값 그대로)", True),
        ("V_ref", "검출기 정지점 917.8 mV (δ=0)", True),
        ("α ×16", "위 표의 Ra/Rb (Ra+Rb = 1 MΩ)", True),
        ("비교기", "채널당 1개, LPV7215", True),
        ("다이오드-OR", "BAT54 ×16 + 정전류 싱크 ×1 + 버퍼 1", True)]
for i, (a, b, ok) in enumerate(spec):
    yy = ey + 60 + i * 24
    s += [t(64, yy, "✅" if ok else "⬜", 11, GRN, "start"),
          t(90, yy, a, 11, INK, "start", weight="700"),
          t(300, yy, b, 11, INK, "start")]
s += [box(52, ey + 208, 696, 30, "#ffffff", GRN, sw=1.1),
      t(64, ey + 228, "\"이보다 조용하면 정확도가 떨어진다. 회로 특성이지 "
        "튜닝으로 못 고친다\" 를 함께 전달할 것", 11, RED, "start",
        weight="700")]

s += [box(790, ey, 740, 250, "#fff1f2", "#fca5a5"),
      t(812, ey + 28, "F. 아직 모델에 없는 것 (= 이 회로도의 한계)", 16, RED,
        "start", weight="700")]
gaps = [("τ (C3 평활)", "실측 4.62 ms. baseline 에 안 넣음 — 논문에 값이 없어 "
         "임의값이 섞이면 원인 분리가 안 된다"),
        ("comparator offset", "mV 급. 보드 만든 뒤 측정해서 그 숫자로 재학습"),
        ("검출기 데드존", "5.9~12.1 mV 실측. spice_deadzone 스위치는 있으나 미사용"),
        ("채널 이득 편차", "1.8 dB. soft max 는 16채널을 섞으므로 하드 max 때와 "
         "달리 흡수가 안 된다 — 미시험"),
        ("저항 공차 / E12·E24", "제작 단계에서")]
for i, (a, b) in enumerate(gaps):
    yy = ey + 62 + i * 36
    s += [t(824, yy, a, 11, RED, "start", weight="700"),
          t(824, yy + 15, b, 10, MUTE, "start")]

# ══ G. accuracy ════════════════════════════════════════════════════════════
gy = 1540
s += [box(30, gy, 1500, 200, "#eff6ff", "#93c5fd"),
      t(52, gy + 28, "G. 이 회로로 기대되는 정확도", 16, BLU, "start",
        weight="700"),
      t(52, gy + 48, "두 숫자는 서로 다른 질문에 대한 답이다", 10.5, MUTE,
        "start")]
s += [box(52, gy + 62, 700, 120, "#ffffff", "#93c5fd", sw=1.1),
      t(64, gy + 84, "① 벤치마크 (GSC test 그대로) — 논문 비교용", 12, INK,
        "start", weight="700"),
      t(400, gy + 112, "0.8445", 26, BLU, weight="700"),
      t(64, gy + 140, "Cerutti 8채널 0.763 / 우리 16채널 0.845 / 64채널 0.860",
        10.5, INK, "start"),
      t(64, gy + 158, "채널수 곡선 보간 예측치(0.795)보다 5pp 위. 목표 85% 까지 "
        "0.6pp", 10, MUTE, "start")]
s += [box(778, gy + 62, 730, 120, "#ffffff", "#93c5fd", sw=1.1),
      t(790, gy + 84, "② 배치 추정 (실환경)", 12, INK, "start", weight="700"),
      t(1000, gy + 112, "≈ 0.69", 26, ORG, weight="700"),
      t(790, gy + 140, "0.8445 − 3.2pp(음량 60~85 dB) − 12.5pp(낯선 배경·위치)",
        10.5, INK, "start"),
      t(790, gy + 158, "두 벌점이 더해진다는 가정. 최대 확신도 선택(창 8개 중 "
        "고르기)은 아직 미구현 — 상방 여지", 10, MUTE, "start")]

s.append(t(30, H - 24, "부품값: analog/AFE/artifacts/component_table.md · "
                       "α: runs/xl_g12 · 정지점 917.8 mV: 동료 실측 · "
                       "전력: proposal/ANALOG.md 3-4 · "
                       "생성: proposal/artifacts/make_full_schematic.py",
           10, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

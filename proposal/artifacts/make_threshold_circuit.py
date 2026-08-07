"""Draw proposal/artifacts/threshold_circuit.svg -- the threshold block alone,
at component level, because that is the ONE part of the AFE that changes.

Everything else (mic, preamp, GIC filters, detectors) is already built and
verified. A circuit person opening the proposal needs exactly this page.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "threshold_circuit.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG = "#dc2626", "#15803d", "#1e3a8a", "#9a3412"
W, H = 1280, 860


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=12, fill=INK, anchor="middle", weight=None, mono=False):
    f = ' font-family="ui-monospace,Menlo,monospace"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{w}{f}>{esc(s)}</text>')


def wire(pts, color=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x},{y}" for x, y in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}"{d}/>')


def dot(x, y, color=WIRE, r=3.2):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>'


def res(x, y, w=54, h=17, label="", sub="", vertical=False, color=WIRE):
    """Resistor as an IEC box, with the label above (or right if vertical)."""
    s = []
    if vertical:
        s.append(f'<rect x="{x - h/2}" y="{y}" width="{h}" height="{w}" rx="2" '
                 f'fill="#ffffff" stroke="{color}" stroke-width="1.6"/>')
        if label:
            s.append(t(x + h/2 + 7, y + w/2 - 2, label, 11.5, INK, "start",
                       weight="700"))
        if sub:
            s.append(t(x + h/2 + 7, y + w/2 + 12, sub, 10, MUTE, "start"))
    else:
        s.append(f'<rect x="{x}" y="{y - h/2}" width="{w}" height="{h}" rx="2" '
                 f'fill="#ffffff" stroke="{color}" stroke-width="1.6"/>')
        if label:
            s.append(t(x + w/2, y - h/2 - 6, label, 11.5, INK, weight="700"))
        if sub:
            s.append(t(x + w/2, y + h/2 + 13, sub, 10, MUTE))
    return "".join(s)


def diode(x, y, size=9, color=WIRE, left=True):
    """Diode pointing right (anode left) by default."""
    if left:
        tri = f"{x - size},{y - size} {x - size},{y + size} {x + size},{y}"
        bar = f'<line x1="{x + size}" y1="{y - size}" x2="{x + size}" ' \
              f'y2="{y + size}" stroke="{color}" stroke-width="2.2"/>'
    else:
        tri = f"{x + size},{y - size} {x + size},{y + size} {x - size},{y}"
        bar = f'<line x1="{x - size}" y1="{y - size}" x2="{x - size}" ' \
              f'y2="{y + size}" stroke="{color}" stroke-width="2.2"/>'
    return (f'<polygon points="{tri}" fill="#ffffff" stroke="{color}" '
            f'stroke-width="1.6"/>{bar}')


def amp(x, y, w=54, h=52, label="", plus_top=True, color=WIRE):
    """Op-amp / comparator triangle, input side left, output right."""
    s = [f'<polygon points="{x},{y - h/2} {x},{y + h/2} {x + w},{y}" '
         f'fill="#ffffff" stroke="{color}" stroke-width="1.8"/>']
    a, b = ("+", "−") if plus_top else ("−", "+")
    s += [t(x + 11, y - h/4 + 4, a, 12, INK),
          t(x + 11, y + h/4 + 5, b, 12, INK)]
    if label:
        s.append(t(x + w/2, y + h/2 + 16, label, 10.5, MUTE))
    return "".join(s)


def gnd(x, y, color=WIRE):
    return (f'<line x1="{x-11}" y1="{y}" x2="{x+11}" y2="{y}" stroke="{color}" '
            f'stroke-width="2"/><line x1="{x-7}" y1="{y+4}" x2="{x+7}" '
            f'y2="{y+4}" stroke="{color}" stroke-width="2"/>'
            f'<line x1="{x-3}" y1="{y+8}" x2="{x+3}" y2="{y+8}" '
            f'stroke="{color}" stroke-width="2"/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     t(W/2, 34, "비교기 임계 회로 — AFE에서 유일하게 바뀌는 부분", 22, INK,
       weight="700"),
     t(W/2, 57, "마이크 · 프리앰프 · GIC 필터 · 검출기는 그대로다. "
                "R7/R8 고정 분압만 아래 회로로 교체한다.", 12.5, MUTE)]

# ── Block A: diode-OR ──────────────────────────────────────────────────────
ax, ay = 60, 100
s += [f'<rect x="{ax}" y="{ay}" width="330" height="300" rx="8" fill="#fff1f2" '
      f'stroke="{RED}" stroke-width="1.6"/>',
      t(ax + 16, ay + 24, "A. 다이오드-OR — 지금 가장 큰 채널 찾기", 13, RED,
        "start", weight="700")]

bus = ax + 250                              # OR bus x
rows = [(ay + 62, "v_env,0"), (ay + 102, "v_env,1"), (ay + 178, "v_env,15")]
for yy, lab in rows:
    s += [t(ax + 20, yy + 4, lab, 11, ORG, "start", weight="700"),
          wire([(ax + 78, yy), (bus - 26, yy)]),
          diode(bus - 14, yy, color=RED),
          wire([(bus - 5, yy), (bus, yy)]), dot(bus, yy, RED)]
s.append(t(ax + 100, ay + 143, "⋮   16채널 전부", 11, MUTE, "start"))
s.append(wire([(bus, ay + 62), (bus, ay + 250)], RED, 2))
s.append(t(bus + 16, ay + 150, "BAT54 ×16", 10.5, RED, "start", weight="700"))
s.append(t(bus + 16, ay + 165, "(검출기와 동일 부품)", 9.5, MUTE, "start"))

# current sink
s += [wire([(bus, ay + 250), (bus, ay + 258)], RED, 2),
      f'<circle cx="{bus}" cy="{ay + 270}" r="12" fill="#ffffff" stroke="{RED}" '
      f'stroke-width="1.6"/>',
      f'<line x1="{bus}" y1="{ay+264}" x2="{bus}" y2="{ay+276}" stroke="{RED}" '
      f'stroke-width="1.6"/>',
      f'<polygon points="{bus-4},{ay+270} {bus+4},{ay+270} {bus},{ay+277}" '
      f'fill="{RED}"/>',
      wire([(bus, ay + 282), (bus, ay + 292)], RED, 2),
      gnd(bus, ay + 292, RED),
      t(bus - 24, ay + 274, "정전류 싱크 ×1", 10.5, RED, "end", weight="700")]
s.append(t(ax + 16, ay + 42, "V_or = V_max − V_d  (강하만큼 낮다)", 10.5, MUTE,
           "start"))

# ── Block B: buffer + Vd compensation ──────────────────────────────────────
bx = 420
s += [f'<rect x="{bx}" y="{ay}" width="300" height="300" rx="8" fill="#fff1f2" '
      f'stroke="{RED}" stroke-width="1.6"/>',
      t(bx + 16, ay + 24, "B. 버퍼 + 다이오드 강하 보상", 13, RED, "start",
        weight="700"),
      t(bx + 16, ay + 42, "op-amp 1개로 뱅크 전체를 담당한다", 10.5, MUTE, "start")]

opx, opy = bx + 90, ay + 120
s += [wire([(bus, ay + 62), (opx - 40, ay + 62), (opx - 40, opy - 13),
            (opx, opy - 13)], RED, 2),
      amp(opx, opy, label="", color=RED),
      wire([(opx + 54, opy), (opx + 110, opy)], RED, 2),
      dot(opx + 110, opy, RED),
      wire([(opx + 110, opy), (opx + 110, opy + 66), (opx - 18, opy + 66),
            (opx - 18, opy + 13), (opx, opy + 13)], RED, 1.6),
      t(opx + 20, opy + 82, "피드백", 10, MUTE)]
s.append(t(opx + 118, opy - 8, "V_max", 13, GRN, "start", weight="700"))
s.append(t(opx + 118, opy + 9, "(강하 보상됨)", 9.5, MUTE, "start"))

# matched diode note
s += [f'<rect x="{bx + 16}" y="{ay + 210}" width="268" height="72" rx="6" '
      f'fill="#ffffff" stroke="{RED}" stroke-width="1.2"/>',
      t(bx + 24, ay + 231, "왜 수동 다이오드만으로는 안 되나", 11, RED, "start",
        weight="700"),
      t(bx + 24, ay + 249, "다이오드 강하 ~0.25 V ≫ v_env 스윙 ~50 mV.", 10, INK,
        "start"),
      t(bx + 24, ay + 265, "→ 같은 다이 위의 짝(BAT54WT1)을 기준 경로에 넣어", 10,
        INK, "start"),
      t(bx + 24, ay + 279, "   온도 드리프트를 상쇄한다.", 10, INK, "start")]

# ── Block C: per-channel divider + comparator ──────────────────────────────
cx = 760
s += [f'<rect x="{cx}" y="{ay}" width="460" height="300" rx="8" fill="#fff1f2" '
      f'stroke="{RED}" stroke-width="1.6"/>',
      t(cx + 16, ay + 24, "C. 채널별 분압 + 비교기  (×16)", 13, RED, "start",
        weight="700"),
      t(cx + 16, ay + 42, "R7/R8을 대신하는 부분. 채널마다 저항 2개.", 10.5, MUTE,
        "start")]

dx = cx + 60          # divider column x
top, mid, bot = ay + 78, ay + 150, ay + 222
s += [wire([(opx + 110, opy), (dx, opy), (dx, top)], GRN, 2),
      t(dx, top - 10, "V_max", 11, GRN, weight="700"),
      res(dx, top + 6, w=44, h=18, vertical=True, label="Ra_c", color=RED),
      wire([(dx, top + 50), (dx, mid)], RED, 2), dot(dx, mid, RED),
      res(dx, mid + 6, w=44, h=18, vertical=True, label="Rb_c", color=RED),
      wire([(dx, mid + 50), (dx, bot)], RED, 2),
      t(dx, bot + 16, "V_ref", 11, GRN, weight="700")]

cmpx, cmpy = dx + 200, mid
s += [amp(cmpx, cmpy, w=58, h=56, plus_top=True, color=WIRE),
      # V_thr into the (-) input
      wire([(dx, mid), (cmpx - 52, mid), (cmpx - 52, cmpy + 14),
            (cmpx, cmpy + 14)], RED, 2),
      t(dx + 62, mid - 9, "V_thr,c", 11.5, RED, weight="700"),
      # v_env into the (+) input, from well above so the labels cannot collide
      wire([(cmpx - 70, cmpy - 52), (cmpx - 20, cmpy - 52),
            (cmpx - 20, cmpy - 14), (cmpx, cmpy - 14)], ORG, 2),
      t(cmpx - 74, cmpy - 48, "v_env,c", 11, ORG, "end", weight="700"),
      wire([(cmpx + 58, cmpy), (cmpx + 100, cmpy)], WIRE, 2),
      t(cmpx + 110, cmpy + 5, "0 / 1", 13, INK, "start", weight="700"),
      t(cmpx + 29, cmpy + 44, "LPV7215", 10, MUTE)]

# the formula, spelled out
s += [f'<rect x="{cx + 16}" y="{ay + 246}" width="428" height="40" rx="6" '
      f'fill="#ffffff" stroke="{RED}" stroke-width="1.2"/>',
      t(cx + 230, ay + 264, "V_thr,c  =  α_c · V_max  +  (1 − α_c) · V_ref",
        13, RED, weight="700"),
      t(cx + 230, ay + 280, "α_c = Rb_c / (Ra_c + Rb_c)   — 학습이 준 16개 상수",
        10.5, MUTE)]

# ── Block D: V_ref ─────────────────────────────────────────────────────────
dy = 425
s += [f'<rect x="{ax}" y="{dy}" width="660" height="120" rx="8" fill="#dcfce7" '
      f'stroke="{GRN}" stroke-width="1.6"/>',
      t(ax + 16, dy + 24, "D. V_ref — 무음일 때의 바닥", 13, GRN, "start",
        weight="700"),
      t(ax + 16, dy + 46, "V_ref = (검출기 정지점 917.8 mV) + δ", 12.5, INK,
        "start", weight="700"),
      t(ax + 16, dy + 68, "δ = 전형 음성 엔벨로프 상승의 0.4%  (ML 실측, "
                          "floor_frac = 0.02)", 11, INK, "start"),
      t(ax + 16, dy + 88, "만드는 법: 입력을 끊은 더미 검출기 1채널 + 작은 오프셋.",
        11, INK, "start"),
      t(ax + 16, dy + 105, "→ 정지점을 자동 추종하므로 온도·공정·전원 변동이 "
                           "함께 상쇄된다.", 10.5, MUTE, "start")]

# ── BOM ────────────────────────────────────────────────────────────────────
s += [f'<rect x="{ax + 690}" y="{dy}" width="530" height="120" rx="8" '
      f'fill="#eff6ff" stroke="#93c5fd" stroke-width="1.6"/>',
      t(ax + 706, dy + 24, "추가 부품 (뱅크 전체)", 13, BLU, "start", weight="700")]
bom = [("BAT54 다이오드", "16 + 1", "OR용 16개 + 강하 보상 1개"),
       ("op-amp (버퍼)", "1", "뱅크 공유"),
       ("정전류 싱크", "1", "OR 노드 풀다운"),
       ("저항 Ra / Rb", "32", "채널당 2개 — R7/R8 자리를 대체")]
for i, (a, n, note) in enumerate(bom):
    y = dy + 46 + i * 18
    s += [t(ax + 706, y, a, 10.5, INK, "start"),
          t(ax + 856, y, n, 10.5, INK, "end", weight="700"),
          t(ax + 872, y, note, 10, MUTE, "start")]

# ── before / after ─────────────────────────────────────────────────────────
fy = 575
s += [f'<rect x="{ax}" y="{fy}" width="1160" height="130" rx="8" '
      f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>',
      t(ax + 16, fy + 24, "무엇이 바뀌나 — 한 장 요약", 13, INK, "start",
        weight="700")]
s += [t(ax + 90, fy + 52, "기존 (고정 임계)", 12, MUTE, weight="700"),
      t(ax + 90, fy + 76, "1.8V ─[R7]─┬─[R8]─ GND", 12, INK, mono=True),
      t(ax + 90, fy + 94, "           │", 12, INK, mono=True),
      t(ax + 90, fy + 112, "        V_thr (고정)", 12, INK, mono=True),
      t(ax + 400, fy + 84, "→", 26, RED, weight="700"),
      t(ax + 690, fy + 52, "새로 (상대 임계)", 12, RED, weight="700"),
      t(ax + 690, fy + 76, "V_max ─[Ra]─┬─[Rb]─ V_ref", 12, INK, mono=True),
      t(ax + 690, fy + 94, "            │", 12, INK, mono=True),
      t(ax + 690, fy + 112, "         V_thr (지금 최대의 α배)", 12, INK, mono=True)]

s += [t(W/2, 745, "임계가 고정 전압이 아니라 '지금 이 순간 16채널 중 최대'를 "
                  "따라가므로, 말하는 사람과 거리가 바뀌어도 이진 이미지가 그대로다.",
        12, RED, weight="700"),
      t(W/2, 768, "음량이 분자·분모에서 약분되기 때문이고, AGC와 달리 "
                  "피드백 루프가 없어 발진·어택/릴리즈·첫 단어 문제가 없다.",
        11.5, MUTE)]

s.append(t(ax, H - 24, "α 16개 값 = proposal/artifacts/channel_table.md   ·   "
                       "전력 순증 ≈ −30 µW (분압이 1.8 V가 아니라 ~0.1 V에만 걸린다)",
           10.5, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

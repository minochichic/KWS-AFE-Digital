"""Draw docs/diagrams/16_quiescent.svg -- why everything is measured as a RISE.

"delta is how far V_ref sits above the quiescent point" is the sentence the
delta explanation rests on, and it does not land without seeing the supply
rail. The detector output is a voltage on a wire, and with no sound at all
that wire is not at 0 V -- it rests at 918 mV, because the circuit runs on a
single 1.8 V supply and has no negative voltage to swing into.

So every voltage in the threshold equation carries the same uninformative
918 mV, and the only thing that says anything is how far above rest each one
sits. Subtracting the rest point from all of them is a change of origin, not
an approximation, and it is what turns the divider equation into a ratio.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "16_quiescent.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,monospace"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1320, 700

RAIL = 1800.0
Q = 917.8
RISE = 31.9
DELTA = 0.108


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


def line(x1, y1, x2, y2, c=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
            f'stroke-width="{sw}"{d}/>')


def arw(x1, y1, x2, y2, c=WIRE, sw=2.0):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
            f'stroke-width="{sw}" marker-end="url(#a)"/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{WIRE}"/></marker>'
     '</defs>',
     t(W / 2, 36, "왜 \"quiescent 위로 얼마나\" 로 재나", 23, INK, weight="700"),
     t(W / 2, 60, "소리가 하나도 없어도 detector 출력선은 0 V 가 아니다. "
                  "918 mV 에 앉아 쉬고 있다.", 13, MUTE)]

# ── 1. absolute volts ──────────────────────────────────────────────────────
px, py, pw, ph = 40, 88, 400, 460
s += [box(px, py, pw, ph, "#f8fafc", "#cbd5e1"),
      t(px + 20, py + 28, "① 절대 전압으로 보면", 15, INK, "start", weight="700"),
      t(px + 20, py + 48, "1.8 V 단일 전원. 음전압이 없어서 모든 신호가", 11,
        MUTE, "start"),
      t(px + 20, py + 64, "0 V 위에 떠 있어야 한다 — 그래서 쉬는 자리가 "
        "생긴다.", 11, MUTE, "start")]

AX, AB, AT = px + 120, py + 372, py + 96     # axis x, y of 0 V, y of 1.8 V
def yv(mv):
    return AB - (mv / RAIL) * (AB - AT)

s += [line(AX, AB, AX, AT, WIRE, 2),
      line(AX - 10, AB, AX + 220, AB, WIRE, 2),
      t(AX - 16, AB + 5, "0 V", 11, INK, "end", weight="700"),
      t(AX - 16, AB + 20, "(GND)", 9.5, MUTE, "end"),
      line(AX - 10, yv(RAIL), AX + 220, yv(RAIL), WIRE, 2, "5 3"),
      t(AX - 16, yv(RAIL) + 4, "1.8 V", 11, INK, "end", weight="700"),
      t(AX - 16, yv(RAIL) + 19, "(전원)", 9.5, MUTE, "end")]

# quiescent + the tiny bump on top of it
qy = yv(Q)
s += [f'<rect x="{AX}" y="{qy}" width="220" height="{AB - qy}" '
      f'fill="{MUTE}" opacity="0.12"/>',
      line(AX, qy, AX + 220, qy, GRN, 2.4),
      t(AX + 228, qy - 3, "918 mV", 12, GRN, "start", weight="700"),
      t(AX + 228, qy + 13, "quiescent", 10.5, GRN, "start")]
bump = (RISE / RAIL) * (AB - AT)
s += [f'<path d="M {AX+30} {qy} Q {AX+70} {qy - bump*3} {AX+110} {qy} '
      f'Q {AX+150} {qy - bump} {AX+200} {qy}" fill="none" stroke="{BLU}" '
      f'stroke-width="2"/>',
      t(AX + 110, qy - bump * 3 - 10, "envelope", 11, BLU, weight="700")]
s += [t(px + pw / 2, py + 420, "envelope 도 V_env,c 도 V_ref 도 전부 "
        "918 mV 를 깔고 있다", 11.5, INK),
      t(px + pw / 2, py + 440, "→ 그 공통분은 아무 정보가 없다", 12.5, RED,
        weight="700")]

# ── arrow ──────────────────────────────────────────────────────────────────
s += [arw(px + pw + 20, py + 240, px + pw + 100, py + 240, PUR, 2.4),
      t(px + pw + 60, py + 224, "원점을", 11.5, PUR, weight="700"),
      t(px + pw + 60, py + 268, "여기로", 11.5, PUR, weight="700"),
      t(px + pw + 60, py + 284, "옮긴다", 11.5, PUR, weight="700")]

# ── 2. rise coordinate ─────────────────────────────────────────────────────
qx, qy0, qw = 580, 88, 400
s += [box(qx, qy0, qw, ph, "#eff6ff", "#93c5fd"),
      t(qx + 20, qy0 + 28, "② quiescent 를 빼면", 15, BLU, "start",
        weight="700"),
      t(qx + 20, qy0 + 48, "모든 항에서 똑같이 뺀다 = 좌표 이동.", 11, MUTE,
        "start"),
      t(qx + 20, qy0 + 64, "근사가 아니라 정확히 같은 식이다.", 11, MUTE, "start")]

BX, BB, BT = qx + 130, qy0 + 372, qy0 + 124
def yr(mv):
    return BB - (mv / 40.0) * (BB - BT)

s += [line(BX, BB, BX, BT, WIRE, 2),
      line(BX - 10, BB, BX + 220, BB, GRN, 2.4),
      t(BX - 16, BB + 5, "0", 12, GRN, "end", weight="700"),
      t(BX - 16, BB + 20, "= quiescent", 9.5, GRN, "end")]
for mv in (10, 20, 30):
    s += [line(BX - 5, yr(mv), BX + 5, yr(mv), WIRE, 1.2),
          t(BX - 12, yr(mv) + 4, f"{mv}", 10, MUTE, "end")]
s.append(t(BX - 12, yr(38) + 4, "mV", 10, INK, "end"))

s += [f'<path d="M {BX+20} {BB} Q {BX+70} {yr(RISE)-14} {BX+120} {BB-6} '
      f'Q {BX+165} {yr(12)} {BX+215} {BB-3}" fill="none" stroke="{BLU}" '
      f'stroke-width="2.4"/>',
      t(BX + 80, yr(RISE) - 22, f"envelope 상승 ~{RISE:.0f} mV", 11.5, BLU,
        weight="700")]
dy_ = yr(DELTA)
s += [line(BX - 10, dy_, BX + 220, dy_, RED, 2),
      t(BX + 228, dy_ + 4, "V_ref", 11.5, RED, "start", weight="700"),
      t(BX + 228, dy_ + 19, f"δ = {DELTA:.3f} mV", 10, RED, "start")]
s.append(t(qx + qw / 2, qy0 + 440, "이제 δ 와 신호가 같은 자 위에 있다", 12.5, BLU, weight="700"))

# ── 3. the analogy ─────────────────────────────────────────────────────────
rx, rw = 1000, 280
s += [box(rx, qy0, rw, ph, "#f0fdf4", "#86efac"),
      t(rx + 20, qy0 + 28, "③ 비유", 15, GRN, "start", weight="700")]
for i, (ln, bold) in enumerate([
        ("파도 높이를 지구 중심에서", True), ("재지 않는다. 해수면에서 잰다.", True),
        ("", False),
        ("quiescent = 해수면", False),
        ("δ = 해수면 위에 세운", False), ("      기준 막대의 높이", False),
        ("envelope = 파도 높이", False),
        ("", False),
        ("해수면이 918 mV 인 건", False), ("파도와 아무 상관이 없다.", False)]):
    if ln:
        s.append(t(rx + 20, qy0 + 58 + i * 20, ln, 11.5 if bold else 11,
                   INK if bold else MUTE, "start",
                   weight="700" if bold else None))

sea = qy0 + 320
s += [f'<rect x="{rx+30}" y="{sea}" width="220" height="62" fill="{BLU}" '
      f'opacity="0.14"/>',
      line(rx + 30, sea, rx + 250, sea, BLU, 2.4),
      t(rx + 140, sea + 40, "바다", 11, BLU),
      f'<path d="M {rx+50} {sea} Q {rx+80} {sea-38} {rx+110} {sea} '
      f'Q {rx+140} {sea-16} {rx+180} {sea}" fill="none" stroke="{BLU}" '
      f'stroke-width="2"/>',
      t(rx + 150, sea - 46, "파도 = envelope", 11, BLU, weight="700"),
      line(rx + 30, sea - 7, rx + 250, sea - 7, RED, 1.8)]
for i, (c, lab) in enumerate(((BLU, "해수면 = quiescent"), (RED, "기준 막대 = δ"))):
    ly = sea + 84 + i * 19
    s += [line(rx + 30, ly - 4, rx + 52, ly - 4, c, 2.4),
          t(rx + 60, ly, lab, 10.5, c, "start", weight="700")]

s.append(t(40, H - 20, "918 mV = analog/AFE_tuning 실측 정지점 · "
                       "31.9 mV = 전형 프레임 상승 (docs/diagrams/15_delta.svg) · "
                       "생성: docs/diagrams/make_quiescent.py", 10, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

"""Draw docs/diagrams/14_diode_softmax.svg -- the diode-OR in plain language.

The xlse review is correct but reads as algebra. The picture underneath it is
simple: a diode is not a switch, it is a blurry valve about 26 mV wide, and
26 mV is fixed by physics while the speech signal is not. So the same circuit
behaves differently depending on how loudly somebody talks -- and it is worst
exactly when they talk quietly.

Both panels share one millivolt axis on purpose. That is the whole argument:
the blur band is the same height in both, and only the bars shrink.
"""
import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "14_diode_softmax.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1300, 1010

VT = 25.852                              # kT/q at 300 K, in mV
DROP_DB = [0, -2, -5, -9, -12, -14, -17, -19, -22, -24,
           -26, -28, -30, -32, -34, -36]  # a typical speech frame, peak-relative


def lse(v, t):
    m = max(v)
    return m + t * math.log(sum(math.exp((x - m) / t) for x in v))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t_(x, y, s, size=12, fill=INK, anchor="middle", weight=None, mono=False):
    f = ' font-family="ui-monospace,Menlo,monospace"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{w}{f}>{esc(s)}</text>')


def box(x, y, w, h, fill="#ffffff", stroke=WIRE, rx=6, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, color=WIRE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d}/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     t_(W / 2, 36, "다이오드-OR은 왜 \"제일 큰 값\"을 못 주나", 23, INK,
        weight="700"),
     t_(W / 2, 60, "다이오드는 스위치가 아니라 폭 26 mV짜리 흐릿한 밸브다. "
                   "그 폭은 물리가 고정하고, 목소리 크기는 안 그렇다.", 13, MUTE)]

# ── A. switch vs valve ─────────────────────────────────────────────────────
ay = 84
s += [box(40, ay, 610, 176, "#f0fdf4", "#86efac"),
      t_(60, ay + 26, "우리가 기대한 것 — 스위치", 15, GRN, "start",
         weight="700")]
for i, hgt in enumerate([54, 30, 20, 12]):
    x = 90 + i * 130
    on = i == 0
    s += [f'<rect x="{x}" y="{ay + 128 - hgt}" width="44" height="{hgt}" rx="3" '
          f'fill="{GRN if on else "#cbd5e1"}" opacity="{0.75 if on else 0.5}"/>',
          t_(x + 22, ay + 146, "켜짐" if on else "꺼짐", 10.5,
             GRN if on else MUTE, weight="700" if on else None)]
s.append(t_(345, ay + 166, "제일 큰 놈만 통과, 나머지는 완전히 차단",
            12, GRN, weight="700"))

s += [box(670, ay, 590, 176, "#fef2f2", "#fca5a5"),
      t_(690, ay + 26, "실제 — 흐릿한 밸브", 15, RED, "start", weight="700")]
for i, hgt in enumerate([54, 30, 20, 12]):
    x = 715 + i * 125
    s += [f'<rect x="{x}" y="{ay + 128 - hgt}" width="44" height="{hgt}" rx="3" '
          f'fill="{RED}" opacity="{0.7 - i * 0.13}"/>',
          t_(x + 22, ay + 146, ["많이", "조금", "찔끔", "찔끔"][i], 10.5, RED)]
s.append(t_(965, ay + 166, "진 채널도 계속 샌다 → 새는 것들이 합쳐져 "
                           "노드가 들린다", 12, RED, weight="700"))

# ── B. the same circuit, two volumes ───────────────────────────────────────
by = 288
s += [box(40, by, 1220, 540, "#eff6ff", "#93c5fd"),
      t_(60, by + 28, "B. 같은 회로인데 목소리 크기에 따라 딴판이 된다",
         16, BLU, "start", weight="700"),
      t_(60, by + 50, "두 그래프는 같은 mV 자를 쓴다. 회색 띠(26 mV, "
                      "다이오드의 흐린 폭)는 양쪽에서 높이가 똑같고, "
                      "막대만 줄어든다 — 그게 전부다.", 11.5, MUTE, "start")]

MV = 0.78                                   # px per mV
BASE = by + 355                             # y of 0 mV -- ABOVE the summary box
for pi, (name, pk, note) in enumerate(
        [("74 dB — 보통 대화", 237.0, "채널 차이가 흐린 폭보다 훨씬 크다"),
         ("60 dB — 조용한 대화", 47.0, "채널 차이가 흐린 폭과 비슷하다")]):
    x0 = 90 + pi * 610
    v = [pk * 10 ** (d / 20) for d in DROP_DB]
    node = lse(v, VT)
    col = GRN if node / max(v) < 1.1 else RED
    y_max, y_node = BASE - max(v) * MV, BASE - node * MV

    s += [box(x0 - 20, by + 70, 540, 300, "#ffffff", "#cbd5e1", sw=1.1),
          t_(x0 + 250, by + 96, name, 14, INK, weight="700"),
          t_(x0 + 250, by + 114, note, 10.5, MUTE)]

    # 26 mV blur band, anchored on the true max -- identical height both sides,
    # which is the entire point of sharing one axis.
    s += [f'<rect x="{x0}" y="{BASE - (max(v) + VT) * MV}" width="440" '
          f'height="{VT * MV}" fill="{MUTE}" opacity="0.18"/>',
          t_(x0 + 8, BASE - (max(v) + VT / 2) * MV + 4,
             "다이오드 흐린 폭 26 mV", 9.5, MUTE, "start")]

    for i, val in enumerate(v):
        s.append(f'<rect x="{x0 + 6 + i * 27}" y="{BASE - val * MV}" width="19" '
                 f'height="{val * MV}" rx="2" fill="{BLU}" opacity="0.55"/>')

    # In the loud panel the two lines are 3 px apart, so the OR label is pushed
    # clear of the max label rather than stacked on top of it.
    y_lab = min(y_node - 10, y_max - 20)
    s += [line(x0, y_max, x0 + 440, y_max, INK, 1.6, "5 3"),
          t_(x0 + 448, y_max + 4, f"진짜 max {max(v):.0f}", 10.5, INK, "start"),
          line(x0, y_node, x0 + 440, y_node, col, 2.6),
          line(x0 + 440, y_node, x0 + 448, y_lab - 4, col, 1.2),
          t_(x0 + 452, y_lab, f"OR 노드 {node:.0f}", 11.5, col, "start",
             weight="700"),
          line(x0, BASE, x0 + 460, BASE, WIRE, 1.4)]

    over = node / max(v)
    s += [box(x0 - 20, by + 386, 540, 100, "#f8fafc", "#cbd5e1", sw=1.1),
          t_(x0 + 250, by + 412, f"분모가 {over:.2f}배 부풀었다", 15, col,
             weight="700"),
          t_(x0 + 250, by + 434,
             f"실제로 분모에 기여한 채널 "
             f"{math.exp((node - max(v)) / VT):.1f}개 / 16개", 11.5, INK),
          t_(x0 + 250, by + 456,
             "→ 사실상 진짜 max. 문제 없음" if over < 1.1
             else "→ 모든 채널이 훨씬 덜 발화한다", 11.5, col),
          t_(x0 + 250, by + 476,
             f"흐린 폭 / 신호 = {VT / pk:.2f}", 10.5, MUTE)]

s.append(t_(650, by + 522, "조용히 말할수록 나빠진다 — 하필 제일 필요한 곳에서.",
            13, RED, weight="700"))

# ── C. so what ─────────────────────────────────────────────────────────────
cy = 856
s += [box(40, cy, 1220, 128, "#faf5ff", "#d8b4fe"),
      t_(60, cy + 28, "C. 그래서 바꿔야 할 것 두 가지", 16, PUR, "start",
         weight="700")]
items = [("1", "\"흐린 정도\"는 우리가 고르는 값이 아니다",
          "코드의 lse_temp_frac은 곧 1/음량이다. 하나로 학습하면 한 음량만 배운다."),
         ("", "→ 게인 증강을 다시 본다",
          "xlse에서 클립 게인을 바꾸는 건 lse_temp_frac을 바꾸는 것과 같다. "
          "예전 기각(−9pp)은 고정 임계 얘기라 여기 적용 안 됨."),
         ("2", "spice_gain_restore를 켠다",
          "\"채널별 threshold가 흡수한다\"는 진짜 max에서만 맞다. 소프트 max는 "
          "16채널을 한 숫자로 섞으므로,"),
         ("", "", "한 채널의 오차가 모두의 분모를 밀어올린다 — "
          "채널별 상수로는 되돌릴 수 없다.")]
for i, (n_, a, b) in enumerate(items):
    yy = cy + 54 + i * 21
    s += [t_(70, yy, n_, 12.5, PUR, "start", weight="700"),
          t_(88, yy, a, 12, INK, "start", weight="700"),
          t_(430, yy, b, 11, MUTE, "start")]

s.append(t_(40, H - 16, "숫자 재현: docs/diagrams/make_softmax_intuition.py · "
                        "대수 유도: docs/experiments_log.md §6-6", 10, MUTE,
            "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

"""Generate docs/diagrams/11_xmax_mechanism.svg and 12_xmax_floor.svg.

All numbers are measured, not illustrative:
  * the 16-channel frame is frame 49 of Speech Commands yes/<first>.wav through
    the SPICE filterbank with sqrt compression;
  * the quantiles, the histogram and the firing rates come from
    experiments/xmax_event_rate.py.

Run:  python docs/diagrams/make_xmax_diagrams.py
"""
import math, pathlib

OUT = pathlib.Path(__file__).resolve().parent
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, RULE = "#0f172a", "#64748b", "#334155"
BLUE_BG, BLUE_LN, BLUE_TX = "#eff6ff", "#93c5fd", "#1e3a8a"
ORNG_BG, ORNG_LN, ORNG_TX = "#fef9f3", "#fdba74", "#9a3412"
RED, GREEN = "#dc2626", "#15803d"

ENV = [0.7878, 1.0275, 1.3422, 1.1721, 1.2205, 1.4579, 1.8364, 2.4447,
       3.8439, 7.4211, 9.9820, 9.9984, 7.1905, 4.1697, 2.4274, 1.6438]
FC = [166, 295, 447, 631, 832, 1072, 1349, 1660,
      2042, 2455, 2951, 3467, 4169, 4898, 5754, 6761]
ALPHA = 0.35                      # a plausible learned alpha, for the picture


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12, fill=INK, anchor="middle", weight=None, style=None):
    a = f' font-weight="{weight}"' if weight else ""
    b = f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{a}{b}>{esc(s)}</text>')


# =========================================================================== #
# 11: the mechanism -- fixed vs xmax under a 2x input gain
# =========================================================================== #
def panel(ox, oy, gain, mode, w=430, h=142):
    """One bar chart of the 16 envelopes with the firing threshold drawn on it."""
    env = [v * gain for v in ENV]
    mx = max(env)
    thr = [ALPHA * mx] * 16 if mode == "xmax" else [ALPHA * max(ENV)] * 16
    top = max(max(env), max(thr)) * 1.12
    bw, gap = w / 16 * 0.62, w / 16
    s = []
    for i, (v, t) in enumerate(zip(env, thr)):
        x = ox + i * gap + (gap - bw) / 2
        bh = v / top * h
        on = v > t
        s.append(f'<rect x="{x:.1f}" y="{oy + h - bh:.1f}" width="{bw:.1f}" '
                 f'height="{bh:.1f}" rx="1.5" '
                 f'fill="{ORNG_LN if on else "#cbd5e1"}" '
                 f'stroke="{ORNG_TX if on else "#94a3b8"}" stroke-width="0.8"/>')
        s.append(txt(x + bw / 2, oy + h + 18, "1" if on else "0", 11,
                     ORNG_TX if on else "#94a3b8", weight="700"))
    ty = oy + h - thr[0] / top * h
    s.append(f'<line x1="{ox - 6}" y1="{ty:.1f}" x2="{ox + w + 6}" y2="{ty:.1f}" '
             f'stroke="{RED}" stroke-width="2" stroke-dasharray="6 3"/>')
    s.append(f'<line x1="{ox}" y1="{oy + h}" x2="{ox + w}" y2="{oy + h}" '
             f'stroke="{RULE}" stroke-width="1.3"/>')
    n_on = sum(1 for v, t in zip(env, thr) if v > t)
    return "\n".join(s), ty, n_on


def diagram_mechanism():
    W, H = 1040, 604
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         txt(W / 2, 34, "xmax — 채널간 상대 임계가 음량을 지우는 원리", 22, INK,
             weight="700"),
         txt(W / 2, 58, "실측: Speech Commands \"yes\" 한 클립의 프레임 49, "
                        "SPICE 필터뱅크 + √ 압축", 13, MUTE)]

    for col, (mode, title, sub, bg, ln, tx) in enumerate([
            ("fixed", "fixed — 절대 임계", "임계가 고정 전압. 입력이 커지면 켜지는 채널이 늘어난다",
             BLUE_BG, BLUE_LN, BLUE_TX),
            ("xmax", "xmax — 상대 임계", "임계가 그 순간 최대의 α배. 입력과 함께 따라 올라간다",
             ORNG_BG, ORNG_LN, ORNG_TX)]):
        ox = 45 + col * 505
        s.append(f'<rect x="{ox - 20}" y="82" width="480" height="478" rx="8" '
                 f'fill="{bg}" stroke="{ln}" stroke-width="1.4"/>')
        s.append(txt(ox + 220, 106, title, 15, tx, weight="700"))
        s.append(txt(ox + 220, 126, sub, 11, MUTE))

        for row, g in enumerate([1.0, 2.0]):
            oy = 158 + row * 186
            body, ty, n_on = panel(ox, oy, g, mode)
            s.append(txt(ox - 8, oy - 10, f"입력 ×{g:g}", 12, INK, anchor="start",
                         weight="700"))
            s.append(txt(ox + 430, oy - 10, f"발화 {n_on}/16 채널", 12,
                         GREEN if mode == "xmax" else RED, anchor="end",
                         weight="700"))
            s.append(body)
            s.append(txt(ox + 430 + 12, ty + 4, "임계", 10, RED, anchor="start"))

        verdict = ("두 이미지가 동일 ✓  음량 불변" if mode == "xmax"
                   else "두 이미지가 다름 ✗  음량이 그대로 샌다")
        s.append(f'<rect x="{ox - 8}" y="518" width="456" height="30" rx="5" '
                 f'fill="#ffffff" stroke="{ln}"/>')
        s.append(txt(ox + 220, 538, verdict, 13,
                     GREEN if mode == "xmax" else RED, weight="700"))

    s.append(txt(45, 588, f"가로축 = 16개 채널 (좌 {FC[0]} Hz → 우 {FC[-1]} Hz)   |   "
                          f"세로축 = 엔벨로프 크기   |   그림의 α = {ALPHA} (실제로는 채널마다 학습됨)",
                 11, MUTE, anchor="start"))
    s.append("</svg>")
    (OUT / "11_xmax_mechanism.svg").write_text("\n".join(s), encoding="utf-8")


# =========================================================================== #
# 12: the floor -- two regimes, and the sqrt-guard trap
# =========================================================================== #
HIST = [0.0, 0.0, 0.03319, 0.00021, 0.0, 0.0, 0.0, 0.00278, 0.01674, 0.02229,
        0.01826, 0.01931, 0.02285, 0.02097, 0.02097, 0.02437, 0.02674, 0.03715,
        0.03646, 0.03542, 0.03007, 0.03604, 0.02618, 0.01667, 0.01681, 0.01708,
        0.01653, 0.03750, 0.05472, 0.03583, 0.03701, 0.04000, 0.03687, 0.06674,
        0.03694, 0.01979, 0.03063, 0.04639, 0.04674, 0.01375]


def diagram_floor():
    W, H = 1040, 660
    GUARD, LO, HI, NB = 1e-3, math.log10(5e-4), math.log10(50.0), 40
    x0, x1, ay, ht = 105, 960, 356, 190

    def X(v):
        return x0 + (math.log10(v) - LO) / (HI - LO) * (x1 - x0)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         txt(W / 2, 34, "floor — 상대 모드와 절대 모드의 경계, 그리고 그 함정", 22,
             INK, weight="700"),
         txt(W / 2, 58, "프레임별 채널간 max의 실측 분포 (144클립 × 100프레임 = 14,400 프레임, "
                        "로그 스케일)", 13, MUTE)]

    fx = X(0.00634)
    s.append(f'<rect x="{x0}" y="{ay - ht}" width="{fx - x0:.1f}" height="{ht}" '
             f'fill="{BLUE_BG}"/>')
    s.append(f'<rect x="{fx:.1f}" y="{ay - ht}" width="{x1 - fx:.1f}" '
             f'height="{ht}" fill="{ORNG_BG}"/>')

    # histogram
    top = max(HIST) * 1.15
    for i, v in enumerate(HIST):
        if v <= 0:
            continue
        xa = x0 + i / NB * (x1 - x0)
        xb = x0 + (i + 1) / NB * (x1 - x0)
        bh = v / top * ht
        guard_bin = int((math.log10(GUARD) - LO) / (HI - LO) * NB)
        c, e = ((RED, "#991b1b") if i == guard_bin
                else (ORNG_LN, ORNG_TX) if xa >= fx else (BLUE_LN, BLUE_TX))
        s.append(f'<rect x="{xa + 0.8:.1f}" y="{ay - bh:.1f}" '
                 f'width="{xb - xa - 1.6:.1f}" height="{bh:.1f}" '
                 f'fill="{c}" stroke="{e}" stroke-width="0.7"/>')

    s.append(txt(x0 + 96, ay - ht - 32, "√(1e-6) = 1e-3", 13, RED, weight="700"))
    s.append(txt(x0 + 96, ay - ht - 15, "sqrt 수치 가드", 11, RED))
    s.append(txt(X(GUARD), ay - 90, "3.3%", 14, RED, weight="700"))

    gap_a, gap_b = X(1.2e-3), X(3.7e-3)
    s.append(f'<line x1="{gap_a:.1f}" y1="{ay - 44}" x2="{gap_b:.1f}" '
             f'y2="{ay - 44}" stroke="{MUTE}" stroke-width="1.2" '
             f'stroke-dasharray="4 3"/>')
    s.append(txt((gap_a + gap_b) / 2, ay - 52, "빈 구간", 10, MUTE))
    s.append(txt((gap_a + gap_b) / 2, ay - 24, "↑ 가드는 신호와", 9, MUTE))
    s.append(txt((gap_a + gap_b) / 2, ay - 13, "완전히 분리돼 있다", 9, MUTE))

    # regime captions
    s.append(txt((x0 + fx) / 2, ay - ht + 22, "절대 모드", 13, BLUE_TX,
                 weight="700"))
    s.append(txt((x0 + fx) / 2, ay - ht + 40, "분모 = floor", 10, BLUE_TX))
    s.append(txt(fx + (x1 - fx) / 2, ay - ht + 22,
                 "상대 모드 — 음성은 전부 여기", 15, ORNG_TX, weight="700"))
    s.append(txt(fx + (x1 - fx) / 2, ay - ht + 42,
                 "분모 = 채널간 max  →  게인이 분자·분모에서 상쇄 → 음량 불변",
                 11, ORNG_TX))
    s.append(f'<line x1="{fx:.1f}" y1="{ay - ht}" x2="{fx:.1f}" y2="{ay + 8}" '
             f'stroke="{GREEN}" stroke-width="2.2"/>')
    s.append(txt(fx, ay - ht - 15, "floor (q=0.05)", 11, GREEN, weight="700"))

    # axis
    s.append(f'<line x1="{x0}" y1="{ay}" x2="{x1}" y2="{ay}" '
             f'stroke="{RULE}" stroke-width="1.6"/>')
    for dec in (-3, -2, -1, 0, 1):
        xv = X(10.0 ** dec)
        s.append(f'<line x1="{xv:.1f}" y1="{ay}" x2="{xv:.1f}" y2="{ay + 6}" '
                 f'stroke="{RULE}" stroke-width="1"/>')
        s.append(txt(xv, ay + 20, f"1e{dec}", 10, MUTE))
    s.append(txt(x1, ay + 20, "채널간 max", 10, MUTE, anchor="end"))

    # the two boxes
    ty = 420
    s.append(f'<rect x="{x0}" y="{ty}" width="405" height="176" rx="7" '
             f'fill="#fee2e2" stroke="{RED}" stroke-width="1.4"/>')
    s.append(txt(x0 + 202, ty + 26, "함정: floor_frac = 0.02", 14, RED,
                 weight="700"))
    s.append(txt(x0 + 202, ty + 45, "→ floor가 가드 위에 정확히 착지", 11, RED))
    for i, line in enumerate([
            "무음 프레임은 16채널이 전부 1e-3으로 같다",
            "xmax = 1e-3 / 1e-3 = 1.0   (전 채널)",
            "무신호가 '가장 활발한 이미지'가 된다",
            "",
            "실측: 제로 클립 발화율 100%,",
            "×0.01 조용한 클립이 음성보다 더 발화",
            "(0.538 > 0.386) — 하드웨어는 이럴 수 없다"]):
        if not line:
            continue
        s.append(txt(x0 + 18, ty + 72 + i * 15, line, 10.5,
                     RED if i == 2 else INK, anchor="start",
                     weight="700" if i == 2 else None))

    bx = x0 + 430
    s.append(f'<rect x="{bx}" y="{ty}" width="425" height="176" rx="7" '
             f'fill="#dcfce7" stroke="{GREEN}" stroke-width="1.4"/>')
    s.append(txt(bx + 212, ty + 26, "해결: floor_frac = 0.05 → 가드의 6.3배", 14,
                 GREEN, weight="700"))
    rows = [("floor_frac", "floor", "제로 발화", "음성 발화"),
            ("0.02", "0.00100", "1.000  ✗", "0.386"),
            ("0.05", "0.00634", "0.000  ✓", "0.366"),
            ("0.10", "0.01311", "0.000  ✓", "0.341")]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            s.append(txt(bx + 20 + c * 105, ty + 56 + r * 21, cell, 11,
                         MUTE if r == 0 else INK, anchor="start",
                         weight="700" if r == 0 else None))
    s.append(txt(bx + 212, ty + 158,
                 "무음은 막고 음성 발화율은 거의 그대로 (0.366 vs 0.386)", 10.5, MUTE))

    s.append(txt(x0, H - 14, "floor는 분포의 분위수로 정해진다 — 데이터셋 최댓값의 "
                             "비율로 잡으면 87%의 프레임을 지배해 fixed로 퇴화한다.",
                 10.5, MUTE, anchor="start"))
    s.append("</svg>")
    (OUT / "12_xmax_floor.svg").write_text("\n".join(s), encoding="utf-8")


diagram_mechanism()
diagram_floor()
print("wrote", OUT / "11_xmax_mechanism.svg")
print("wrote", OUT / "12_xmax_floor.svg")

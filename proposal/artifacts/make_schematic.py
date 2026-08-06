"""Draw proposal/artifacts/afe_xmax_system.svg -- the AFE with the cross-channel
relative threshold, which the earlier afe_final_system.svg predates.

The only structural change from that drawing is the threshold source: the 16
per-channel dividers no longer hang off the 1.8 V rail (a FIXED voltage), they
hang between the cross-channel max node and a shared reference. Everything to
the left of the comparator is unchanged and already verified in ngspice.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "afe_xmax_system.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, RULE = "#0f172a", "#64748b", "#334155"
BLUE_BG, BLUE_LN, BLUE_TX = "#eff6ff", "#93c5fd", "#1e3a8a"
ORNG_BG, ORNG_LN, ORNG_TX = "#fef9f3", "#fdba74", "#9a3412"
GRN_BG, GRN_LN, GRN_TX = "#dcfce7", "#86efac", "#15803d"
RED = "#dc2626"
W, H = 1240, 830


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=12, fill=INK, anchor="middle", weight=None):
    a = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{fill}"{a}>{esc(s)}</text>')


def box(x, y, w, h, fill, stroke, rx=6, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, color=RULE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')


def line(x1, y1, x2, y2, color=RULE, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d}/>')


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{RULE}"/></marker>'
     '<marker id="ar" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{RED}"/></marker></defs>',
     t(W / 2, 32, "AFE 시스템 — 채널간 상대 임계 + 채널당 비교기 2개", 22, INK, weight="700"),
     t(W / 2, 55, "붉은 부분만 새로 추가된다. 왼쪽 신호 체인은 ngspice 검증 완료, 변경 없음.",
       12.5, MUTE)]

# ── row 1: mic -> preamp -> fanout ─────────────────────────────────────────
y0 = 88
s += [box(40, y0, 105, 56, "#f1f5f9", "#94a3b8"),
      t(92, y0 + 24, "MIC", 13, INK, weight="700"),
      t(92, y0 + 41, "~0.4 mVpp", 10, MUTE),
      arrow(145, y0 + 28, 185, y0 + 28),
      box(185, y0, 130, 56, "#f1f5f9", "#94a3b8"),
      t(250, y0 + 22, "프리앰프", 12, INK, weight="700"),
      t(250, y0 + 38, "G = 1+Rf/Rg = 10", 10, MUTE),
      t(250, y0 + 51, "16 µA", 9, MUTE),
      arrow(315, y0 + 28, 360, y0 + 28),
      t(392, y0 + 24, "16 채널로", 11, MUTE),
      t(392, y0 + 38, "분배", 11, MUTE)]

# ── channel block (one shown, x16) ─────────────────────────────────────────
cy = 175
s += [box(40, cy, 700, 118, BLUE_BG, BLUE_LN),
      t(60, cy + 20, "채널 c  (16개 동일 구조 — RA / C / R1 만 채널별로 다름)", 13,
        BLUE_TX, anchor="start", weight="700"),
      box(70, cy + 32, 190, 68, "#ffffff", BLUE_LN, sw=1.1),
      t(165, cy + 54, "2차 GIC 밴드패스", 12, INK, weight="700"),
      t(165, cy + 71, "RA 10.6–12.5k · C 1.86–76.6n", 9.5, MUTE),
      t(165, cy + 85, "R1 16.5–59k · R2=R3=100k", 9.5, MUTE),
      arrow(260, cy + 66, 300, cy + 66),
      t(280, cy + 56, "v_filt", 9, MUTE),
      box(300, cy + 32, 200, 68, "#ffffff", BLUE_LN, sw=1.1),
      t(400, cy + 54, "능동 검출기 (정밀정류)", 12, INK, weight="700"),
      t(400, cy + 71, "D1/D2 BAT54 · R4=10k · R5=47k", 9.5, MUTE),
      t(400, cy + 85, "C3=100n → τ=4.7 ms · G=4.7", 9.5, MUTE),
      arrow(500, cy + 66, 545, cy + 66),
      t(524, cy + 56, "v_env,c", 9.5, ORNG_TX, weight="700"),
      box(545, cy + 34, 120, 64, "#fff1f2", RED, sw=1.4),
      t(605, cy + 52, "비교기 ×2", 12, RED, weight="700"),
      t(605, cy + 68, "LPV7215 · 0.58 µA", 9.5, MUTE),
      t(605, cy + 82, "α_c,0 / α_c,1", 9.5, RED, weight="700"),
      arrow(665, cy + 66, 720, cy + 66),
      t(738, cy + 62, "2비트", 11, RED, weight="700"),
      t(738, cy + 77, "온도계", 9.5, MUTE)]

# tap from v_env down to the OR bus
s += [line(524, cy + 78, 524, 330, RED, 1.8),
      t(534, 322, "16채널 전부", 9.5, RED, anchor="start")]

# ── the new block ──────────────────────────────────────────────────────────
ny = 340
s += [box(40, ny, 700, 210, "#fff1f2", RED, sw=1.8),
      t(60, ny + 22, "NEW — 채널간 최댓값 회로 + 상대 임계 분압", 14, RED,
        anchor="start", weight="700"),
      t(60, ny + 40, "기존 회로에서 바뀌는 곳은 여기 하나뿐이다: 비교기 기준 전압을 "
                     "고정 분압이 아니라 '지금 16채널 중 최대'에서 만든다.",
        10.5, MUTE, anchor="start")]

# diode OR
s += [box(70, ny + 55, 175, 82, "#ffffff", RED, sw=1.2),
      t(157, ny + 76, "다이오드-OR ×16", 12, RED, weight="700"),
      t(157, ny + 93, "BAT54 (기존과 동일 부품)", 9.5, MUTE),
      t(157, ny + 107, "정전류 싱크 1개", 9.5, MUTE),
      t(157, ny + 124, "→ V_max − V_d", 10, INK, weight="700"),
      arrow(245, ny + 96, 285, ny + 96, RED)]
# buffer + Vd compensation
s += [box(285, ny + 55, 175, 82, "#ffffff", RED, sw=1.2),
      t(372, ny + 76, "버퍼 + V_d 보상", 12, RED, weight="700"),
      t(372, ny + 93, "op-amp ×1 (뱅크 공유)", 9.5, MUTE),
      t(372, ny + 107, "정합 다이오드로 강하 상쇄", 9.5, MUTE),
      t(372, ny + 124, "→ V_max", 10, INK, weight="700"),
      arrow(460, ny + 96, 500, ny + 96, RED)]
# per-channel divider
s += [box(500, ny + 48, 215, 100, "#ffffff", RED, sw=1.2),
      t(607, ny + 68, "채널별 분압 ×32  (채널당 2개)", 12, RED, weight="700"),
      t(607, ny + 88, "V_max ─[Ra]─┬─[Rb]─ V_ref", 11, INK),
      t(607, ny + 104, "│", 10, INK),
      t(607, ny + 118, "V_thr,c", 11, ORNG_TX, weight="700"),
      t(607, ny + 138, "α_c = Rb/(Ra+Rb) ∈ [0,1]", 10.5, RED, weight="700")]
# V_thr back up to comparator
s += [line(715, ny + 108, 760, ny + 108, RED, 1.8),
      line(760, ny + 108, 760, cy + 80, RED, 1.8),
      arrow(760, cy + 80, 667, cy + 80, RED),
      t(770, (ny + 108 + cy + 80) / 2, "V_thr,c", 10, RED, anchor="start", weight="700")]

# V_ref block
s += [box(40, ny + 158, 700, 40, GRN_BG, GRN_LN, sw=1.2),
      t(58, ny + 175, "V_ref = 검출기 정지점(917.8 mV) + δ", 12, GRN_TX,
        anchor="start", weight="700"),
      t(58, ny + 191, "δ = 전형 음성 엔벨로프 상승의 2.4% (ML 실측). 더미 검출기 1채널로 "
                      "만들면 온도·공정이 자동 추종된다.", 10, MUTE, anchor="start")]

# ── right column: what alpha is, and the power budget ──────────────────────
rx = 770
s += [box(rx, 88, 430, 205, ORNG_BG, ORNG_LN),
      t(rx + 215, 112, "α는 클립마다 변하지 않는다", 14, ORNG_TX, weight="700"),
      t(rx + 16, 138, "학습이 끝나면 α는 32개의 상수다. PCB에 납땜되는 저항비이고,",
        10.5, INK, anchor="start"),
      t(rx + 16, 154, "추론 중에는 절대 변하지 않는다.", 10.5, INK, anchor="start")]
rows = [("", "무엇", "언제 정해지나", "회로"),
        ("α_c", "채널당 2개 (32개)", "학습 중 1회 → 동결", "저항 4개"),
        ("V_max", "프레임마다 다름", "런타임, 매 순간", "다이오드-OR"),
        ("δ", "전체 공유 1개", "설계 시 1회", "V_ref 분압")]
for i, r in enumerate(rows):
    for c, cell in enumerate(r):
        s.append(t(rx + 16 + c * 108, 182 + i * 20, cell, 10,
                   MUTE if i == 0 else INK, anchor="start",
                   weight="700" if i == 0 or c == 0 else None))
s.append(t(rx + 215, 278, "즉 클립마다 변하는 것은 분모(V_max)뿐이다.", 11,
           ORNG_TX, weight="700"))

s += [box(rx, 310, 430, 240, GRN_BG, GRN_LN),
      t(rx + 215, 334, "전력 수지 — 순증이 거의 0", 14, GRN_TX, weight="700")]
pw = [("항목", "전력", ""),
      ("추가: 비교기 16개 (2비트화)", "+17 µW", "◆"),
      ("추가: 버퍼 op-amp ×1", "+13 µW", ""),
      ("추가: OR 다이오드 ×16 + 싱크", "+3 µW", ""),
      ("추가: 새 분압 ×32", "+6 µW", ""),
      ("절감: 기존 분압 ×16", "−52 µW", "★"),
      ("순증", "≈ −13 µW", "")]
for i, (a, b, c) in enumerate(pw):
    y = 344 + i * 21
    bold = "700" if i in (0, len(pw) - 1) else None
    col = GRN_TX if i == len(pw) - 1 else (MUTE if i == 0 else INK)
    if i == len(pw) - 1:
        s.append(line(rx + 16, y - 16, rx + 400, y - 16, GRN_LN, 1.2))
    s += [t(rx + 16, y, a, 11, col, anchor="start", weight=bold),
          t(rx + 330, y, b, 11, col, anchor="end", weight=bold),
          t(rx + 350, y, c, 11, GRN_TX, anchor="start")]
s.append(t(rx + 16, 496, "◆ 비교기 2개 = 2비트 온도계 코드. 정확도 0.778 → 0.802 (+2.5pp).",
           10, MUTE, anchor="start"))
s.append(t(rx + 16, 512, "★ 기존 분압은 1.8 V 전체, 새 분압은 V_max−V_ref(~0.1 V)에만 걸린다.",
           10, MUTE, anchor="start"))
s.append(t(rx + 16, 536, "AGC와 달리 피드백 루프가 없다 → 발진·어택/릴리즈·첫 단어 문제 없음.",
           10.5, GRN_TX, anchor="start", weight="700"))

# ── bottom: what stays fixed vs what this replaces ─────────────────────────
by = 578
s += [box(40, by, 1160, 120, "#f8fafc", "#cbd5e1"),
      t(60, by + 24, "이 설계가 바꾸는 것 / 바꾸지 않는 것", 13, INK,
        anchor="start", weight="700")]
keep = ["GIC 필터뱅크 16채널 (RA/C/R1 — mel 매칭, 변경 금지)",
        "능동 검출기 (D1/D2, R4=10k, R5=47k, C3=100n)",
        "비교기 LPV7215, 단일 1.8 V 공급, 0.9 V 바이어스",
        "마이크 프리앰프 G=10"]
chg = ["R7/R8 (1.8 V 고정 분압)  →  Ra/Rb (V_max ↔ V_ref 분압)",
       "임계가 고정 전압  →  지금 최대의 α배",
       "추가: 다이오드-OR + 버퍼 1조 (뱅크 전체 공유)",
       "추가: V_ref 생성 (더미 검출기 + δ)"]
for i, k in enumerate(keep):
    s.append(t(70, by + 48 + i * 17, f"✓  {k}", 10.5, GRN_TX, anchor="start"))
for i, k in enumerate(chg):
    s.append(t(640, by + 48 + i * 17, f"→  {k}", 10.5, RED, anchor="start"))

s.append(t(40, H - 22, "수치 출처: 필터·검출기 = analog/AFE_tuning (실제 벤더 모델 "
                       "ngspice 실측) · α와 δ = ML 실측 (proposal/artifacts/channel_table.md)",
           10, MUTE, anchor="start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

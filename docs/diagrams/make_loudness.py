"""Draw docs/diagrams/17_loudness.svg -- why dividing by the diode-OR is not
loudness-invariant.

Dividing by a max IS loudness-invariant: max(g*v) = g*max(v), so g cancels
exactly. That is the whole reason for the cross-channel threshold. But a
diode-OR does not give the max -- it gives max + T*ln(k), and T = n*V_T is a
physical constant that does not shrink when the talker gets quieter. So the
excess is 1% of the signal at 85 dB SPL and 380% at 60 dB, g never cancels, and
the same spectral shape produces a different binary image at every loudness.

The measured consequence is the frac sweep: 0.845 at the training loudness,
0.75 twelve dB below it. That is not a training failure. It is the front end
losing invariance, and no threshold placement recovers it, because the
channel-to-channel contrast itself collapses (0.204 -> 0.043).
"""
import math
import pathlib

import torch

OUT = pathlib.Path(__file__).resolve().parent / "17_loudness.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,monospace"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1360, 1180

VT = 25.852
ALPHA = 0.30                       # illustrative; learned alphas span 0.06-0.87
DROP_DB = [0, -2, -5, -9, -12, -14, -17, -19, -22, -24,
           -26, -28, -30, -32, -34, -36]
PROF = [10 ** (d / 20) for d in DROP_DB]
# ANALOG.md 6-2, pre-amp x10, formant channel gets about a third of the band
SPLS = [(85, 280.3), (74, 79.0), (66, 31.4), (60, 15.8)]


def lse(v):
    return float(VT * torch.logsumexp(torch.tensor(v) / VT, 0))


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


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     t(W / 2, 36, "왜 최댓값으로 나눴는데도 음량이 남나", 23, INK, weight="700"),
     t(W / 2, 60, "다이오드-OR 은 max 를 주지 않는다. "
                  "max + (신호와 무관한 고정 초과분) 을 준다.", 13, MUTE)]

# ── A. what we expected vs what we get ─────────────────────────────────────
ay = 88
s += [box(40, ay, 640, 220, "#f0fdf4", "#86efac"),
      t(60, ay + 28, "① 기대 — 진짜 max 로 나누면", 15, GRN, "start",
        weight="700")]
s += [t(60, ay + 62, "max(g·v) = g·max(v)", 15, INK, "start", weight="700",
        mono=True),
      t(60, ay + 82, "max 는 1차 동차 — 크기를 곱하면 그대로 곱해진다", 10.5,
        MUTE, "start"),
      t(60, ay + 126, "g·v_c / (g·max(v))  =  v_c / max(v)", 16, GRN, "start",
        weight="700", mono=True),
      t(60, ay + 148, "→ g 가 정확히 약분된다. 음량 완전 불변 ✅", 12, GRN,
        "start", weight="700"),
      t(60, ay + 186, "이게 채널간 임계를 쓰기로 한 이유 전부다.", 11.5, INK,
        "start")]

s += [box(720, ay, 600, 220, "#fef2f2", "#fca5a5"),
      t(740, ay + 28, "② 실제 — 다이오드-OR 은", 15, RED, "start", weight="700")]
s += [t(740, ay + 62, "V_or = max(v) + T·ln(k)", 15, INK, "start",
        weight="700", mono=True),
      t(740, ay + 82, "k = 최대 근처 채널 수, T = n·V_T = 26 mV", 10.5, MUTE,
        "start"),
      t(740, ay + 126, "g·v_c / (g·max(v) + T·ln k)", 16, RED, "start",
        weight="700", mono=True),
      t(740, ay + 148, "→ 초과분에 g 가 없다. 약분 안 된다 ❌", 12, RED,
        "start", weight="700"),
      t(740, ay + 186, "T 는 물리 상수라 화자가 조용해져도 안 줄어든다.", 11.5,
        INK, "start")]

# ── B. the excess, measured ────────────────────────────────────────────────
by = 330
s += [box(40, by, 640, 250, "#fff1f2", "#fca5a5"),
      t(60, by + 28, "③ 그 초과분이 조용해질수록 신호를 삼킨다", 15, RED,
        "start", weight="700"),
      t(60, by + 48, "프리앰프 ×10, 포먼트 채널 기준 (ANALOG.md 6-2)", 10.5,
        MUTE, "start")]
cols = [(90, "SPL"), (200, "스윙"), (330, "OR 노드"), (455, "초과분"),
        (610, "초과/신호")]
for x, lab in cols:
    s.append(t(x, by + 78, lab, 11, INK, "end" if x > 100 else "middle",
               weight="700"))
for i, (spl, pk) in enumerate(SPLS):
    v = [pk * p for p in PROF]
    S = lse(v)
    yy = by + 104 + i * 30
    frac_ex = (S - max(v)) / max(v)
    col = GRN if frac_ex < 0.1 else (ORG if frac_ex < 1.0 else RED)
    s += [t(90, yy, f"{spl} dB", 12, INK, weight="700"),
          t(200, yy, f"{pk:.1f} mV", 11.5, INK, "end"),
          t(330, yy, f"{S:.1f} mV", 11.5, INK, "end"),
          t(455, yy, f"+{S - max(v):.1f} mV", 11.5, col, "end", weight="700"),
          t(610, yy, f"{frac_ex:.0%}", 13, col, "end", weight="700")]
s.append(t(360, by + 232, "60 dB 에서는 초과분이 신호의 4배다", 12.5, RED,
           weight="700"))

# ── C. the ruler ───────────────────────────────────────────────────────────
s += [box(720, by, 600, 250, "#fffbeb", "#fcd34d"),
      t(740, by + 28, "④ 같은 얘기를 자로 비유하면", 15, ORG, "start",
        weight="700")]
rx, ry = 760, by + 96
s.append(line(rx, ry, rx + 520, ry, WIRE, 2))
for i in range(6):
    x = rx + i * 104
    s += [line(x, ry - 12, x, ry + 12, WIRE, 2),
          t(x, ry + 28, f"{i * 26}", 10, MUTE)]
s.append(t(rx + 260, ry + 46, "다이오드의 눈금 = 26 mV (T)", 11.5, ORG,
           weight="700"))
s += [f'<rect x="{rx}" y="{ry - 44}" width="{520 * 15.8 / 130:.0f}" '
      f'height="20" rx="3" fill="{RED}" opacity="0.30" stroke="{RED}"/>',
      t(rx + 520 * 15.8 / 130 / 2, ry - 30, "16", 10.5, RED, weight="700"),
      t(rx + 520 * 15.8 / 130 + 10, ry - 30, "← 60 dB 의 신호 전체", 11, RED,
        "start", weight="700")]
s += [t(740, by + 196, "눈금이 26 mm 인 자로 16 mm 를 재는 셈이다.", 12.5, INK,
        "start"),
      t(740, by + 218, "자를 어디에 대든(=α 를 어디에 두든) 안 된다.", 12, ORG,
        "start", weight="700")]

# ── D. same sound, different image ─────────────────────────────────────────
dy = 602
s += [box(40, dy, 1280, 330, "#eff6ff", "#93c5fd"),
      t(60, dy + 28, "⑤ 그래서 같은 소리가 크기에 따라 다른 이진 이미지가 된다",
        15, BLU, "start", weight="700"),
      t(60, dy + 48, "스펙트럼 모양은 완전히 동일. 크기만 다르다. "
                     f"α = {ALPHA} 로 고정 (예시값 — 학습된 α 는 0.06~0.87)",
        10.5, MUTE, "start")]

BW, BH, GAP = 13, 96, 5
for i, (spl, pk) in enumerate(SPLS):
    v = [pk * p for p in PROF]
    S = lse(v)
    n = [x / S for x in v]
    x0 = 78 + i * 305
    yb = dy + 200
    s += [t(x0 + 117, dy + 84, f"{spl} dB SPL", 13, INK, weight="700"),
          t(x0 + 117, dy + 100, f"스윙 {pk:.0f} mV → OR 노드 {S:.0f} mV", 10, MUTE)]
    s.append(line(x0, yb, x0 + 16 * (BW + GAP), yb, WIRE, 1.4))
    ty = yb - BH * ALPHA
    s.append(line(x0 - 4, ty, x0 + 16 * (BW + GAP), ty, RED, 1.8, "5 3"))
    s.append(t(x0 - 8, ty + 4, "α", 11, RED, "end", weight="700"))
    fire = 0
    for c, val in enumerate(n):
        h = BH * min(val, 1.0)
        on = val > ALPHA
        fire += on
        s.append(f'<rect x="{x0 + c * (BW + GAP)}" y="{yb - h}" width="{BW}" '
                 f'height="{h}" rx="2" fill="{BLU if on else MUTE}" '
                 f'opacity="{0.75 if on else 0.30}"/>')
    s += [t(x0 + 117, yb + 22, "".join("1" if x > ALPHA else "0" for x in n),
            13, INK, mono=True),
          t(x0 + 117, yb + 44, f"{fire}/16 발화", 12.5,
            GRN if fire >= 3 else RED, weight="700")]
s.append(t(680, dy + 306, "채널 4개 → 0개. α 를 낮춰 개수는 맞출 수 있어도, "
                          "격차가 0.204 → 0.043 으로 줄어 패턴은 복구 안 된다",
           12.5, RED, weight="700"))

# ── E. what can and cannot fix it ──────────────────────────────────────────
ey = 954
s += [box(40, ey, 1280, 196, "#faf5ff", "#d8b4fe"),
      t(60, ey + 28, "⑥ 고칠 수 있나", 15, PUR, "start", weight="700"),
      t(60, ey + 48, "임계가 만들어질 수 있는 형태는 하나뿐이다: "
                     "V_thr = α·V_or + (1−α)·V_ref. "
                     "소프트웨어에 자유도가 없다.", 11, MUTE, "start")]
rows = [("❌", "정규화 바꾸기", "이미 xmax / xmix / xlse 를 다 해봤다. "
         "초과분이 스펙트럼에 따라 변해서 상수로 뺄 수 없다 (δ 가 그 시도, 0 이 최적)"),
        ("❌", "게인 증강", "곡선을 +1.3pp 올렸지만 평평하게는 못 만든다 — "
         "네트워크는 훈련되지만 프론트엔드 불변성은 안 생긴다"),
        ("⚠️", "프리앰프 이득 ↑", "×10 이 레일 상한. 그래도 초과분이 1% 아래가 "
         "되는 건 78 dB SPL 이상뿐"),
        ("⭕", "진짜 max 회로", "초과분 제거. 단 증폭기 추가 — 그게 `xmax` 이고 "
         "0.781 이었다 (soft max 가 +6.7pp 더 높다)"),
        ("❌", "AGC", "피드백 루프(발진·어택/릴리즈) — 이미 기각")]
for i, (mark, what, why) in enumerate(rows):
    yy = ey + 82 + i * 22
    s += [t(66, yy, mark, 12, INK, "start"),
          t(96, yy, what, 11.5, INK, "start", weight="700"),
          t(250, yy, why, 10.5, MUTE, "start")]

s.append(t(40, H - 16, "실측 결과: frac 스윕 xl_g12 — 학습 음량 0.845, "
                       "12 dB 아래 0.743 · 생성: docs/diagrams/make_loudness.py",
           10, MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

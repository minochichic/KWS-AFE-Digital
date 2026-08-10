"""Draw docs/diagrams/13_sliding_window.svg -- why a deployed FPGA needs
something training never did.

The question this answers: "augmentation and window width were never an issue
while training; why do they appear the moment we target hardware?"

Because the dataset does a job for us that nobody does on a board. Speech
Commands hands over clips that are ALREADY cut to 1 s with the word roughly
centred, so the model has never in its life seen a word anywhere else. A
free-running microphone has no clips and no centre -- the FPGA has to invent
the cut itself, and it has no idea when the talker will start.
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "13_sliding_window.svg"
FONT = "'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUTE, WIRE = "#0f172a", "#64748b", "#334155"
RED, GRN, BLU, ORG, PUR = "#dc2626", "#15803d", "#1e3a8a", "#9a3412", "#7e22ce"
W, H = 1340, 1210


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


def arw(x1, y1, x2, y2, color=WIRE, sw=1.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#a)"{d}/>')


def word(x, y, w, h=26, color=GRN, label="yes"):
    """A spoken word as a blob on a timeline."""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
            f'fill="{color}" opacity="0.22" stroke="{color}" '
            f'stroke-width="1.6"/>' + t(x + w / 2, y + h / 2 + 4.5, label, 11,
                                        color, weight="700"))


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5" '
     f'orient="auto"><polygon points="0 0, 9 3.5, 0 7" fill="{WIRE}"/></marker>'
     '</defs>',
     t(W / 2, 36, "왜 학습 때는 없던 문제가 FPGA에서 생기나", 23, INK,
       weight="700"),
     t(W / 2, 60, "데이터셋이 대신 해주던 일이 하나 있다 — 그걸 보드에서는 "
                  "아무도 안 해준다.", 13, MUTE)]

# ── A. training ────────────────────────────────────────────────────────────
ay = 88
s += [box(40, ay, 620, 200, "#f0fdf4", "#86efac"),
      t(60, ay + 26, "A. 학습할 때 — 문제가 없었던 이유", 15, GRN, "start",
        weight="700"),
      t(60, ay + 47, "Speech Commands가 이미 1초로 잘라서 준다. "
                     "단어는 대충 가운데.", 11.5, MUTE, "start")]
for i, (off, lab) in enumerate([(0.30, "yes"), (0.26, "go"), (0.34, "stop")]):
    yy = ay + 66 + i * 40
    s += [box(70, yy, 500, 28, "#ffffff", "#cbd5e1", rx=4, sw=1.1),
          word(70 + 500 * off, yy + 1, 500 * 0.30, 26, GRN, lab),
          t(586, yy + 19, "1초 클립", 10, MUTE, "start")]
s.append(t(350, ay + 190, "모델은 평생 \"가운데 있는 단어\"만 봤다.", 12.5, GRN,
           weight="700"))

# ── B. deployment ──────────────────────────────────────────────────────────
s += [box(680, ay, 620, 200, "#fef2f2", "#fca5a5"),
      t(700, ay + 26, "B. FPGA에서 — always-on", 15, RED, "start", weight="700"),
      t(700, ay + 47, "마이크는 끝없이 흐른다. 클립도, 경계도, 가운데도 없다.",
        11.5, MUTE, "start")]
s += [box(710, ay + 74, 570, 30, "#ffffff", "#cbd5e1", rx=4, sw=1.1),
      word(710 + 300, ay + 75, 130, 28, RED, "yes"),
      t(716, ay + 122, "⟵ 끝없는 스트림 ⟶", 10.5, MUTE, "start"),
      t(1274, ay + 122, "화자는 아무 때나 말한다", 10.5, MUTE, "end")]
s += [box(700, ay + 134, 580, 54, "#ffffff", "#fca5a5", sw=1.1),
      t(712, ay + 155, "누가 1초를 끊어줄 것인가?", 12, RED, "start",
        weight="700"),
      t(712, ay + 175, "데이터셋이 해주던 그 일을, 이제 FPGA가 스스로 해야 한다.",
        11, INK, "start")]

# ── C. why not just wait for the sound? ────────────────────────────────────
cy = 306
s += [box(40, cy, 1260, 128, "#fffbeb", "#fcd34d"),
      t(60, cy + 26, "C. \"소리가 나면 그때부터 1초 재면 되지 않나?\" — "
                     "그게 원래 계획이었고, 죽었다", 15, ORG, "start",
        weight="700"),
      t(60, cy + 50, "Cerutti는 그렇게 했다. 되는 이유는 그쪽 임계가 "
                     "고정 전압이라 조용하면 아무도 안 넘기 때문이다.", 11.5,
        MUTE, "start")]
s += [box(60, cy + 64, 600, 50, "#ffffff", "#fcd34d", sw=1.1),
      t(72, cy + 84, "우리 임계는 상대적이다 (xmix/xlse)", 11.5, ORG, "start",
        weight="700"),
      t(72, cy + 103, "\"16채널 중 지금 제일 큰 놈\"에 대한 비율 — "
                      "조용해도 누군가는 제일 크다.", 10.5, INK, "start")]
s += [box(680, cy + 64, 600, 50, "#ffffff", "#fcd34d", sw=1.1),
      t(692, cy + 84, "실측: 조용한 구간에서도 프레임의 95.2%가 발화", 11.5, RED,
        "start", weight="700"),
      t(692, cy + 103, "→ 트리거가 상시 걸린다. \"첫 이벤트 = 말 시작\"이 "
                       "성립하지 않는다.", 10.5, INK, "start")]

# ── D. sliding window ──────────────────────────────────────────────────────
dy = 456
s += [box(40, dy, 1260, 330, "#eff6ff", "#93c5fd"),
      t(60, dy + 26, "D. 그래서 슬라이딩 — 100 ms마다 1초 창을 계속 뽑는다",
        15, BLU, "start", weight="700"),
      t(60, dy + 47, "언제 말할지 모르니 항상 본다. 단어를 온전히 담는 창이 "
                     "약 8개 생기고, 창마다 단어 위치가 다르다.", 11.5, MUTE,
        "start")]

# stream with the word.  Geometry is chosen so the FIRST window (word flush
# right) and the LAST (word flush left) both fit inside the panel -- the whole
# point is that the reader sees the word sitting still while the windows walk.
sx, sw_, wx, ww, win = 150, 1120, 560, 170, 560
s += [box(sx, dy + 70, sw_, 30, "#ffffff", "#cbd5e1", rx=4, sw=1.1),
      word(wx, dy + 71, ww, 28, BLU, "yes  (270 ms)"),
      t(sx - 8, dy + 90, "스트림", 10.5, MUTE, "end")]

n = 8
for i in range(n):
    yy = dy + 116 + i * 25
    x0 = wx + ww - win + i * (win - ww) / (n - 1)
    good = 3 <= i <= 4
    col = GRN if good else RED
    s += [f'<rect x="{x0}" y="{yy}" width="{win}" height="19" rx="3" '
          f'fill="none" stroke="{col}" stroke-width="1.5" opacity="0.75"/>',
          f'<rect x="{wx}" y="{yy + 2}" width="{ww}" height="15" rx="2" '
          f'fill="{col}" opacity="0.28"/>']
    pos = (wx - x0) / (win - ww)
    s.append(t(1280, yy + 14, f"단어가 창의 {pos * 100:>3.0f}% 지점", 9.5,
               col, "end"))
s += [t(70, dy + 130, "창 1", 10, MUTE, "start"),
      t(70, dy + 305, f"창 {n}", 10, MUTE, "start"),
      t(560, dy + 322, "같은 한 단어인데, 창마다 다른 위치에서 보인다 "
        "— 이게 학습 분포와 어긋난다.", 12, BLU, weight="700")]

# ── E. what we measured wrong ──────────────────────────────────────────────
ey = 808
s += [box(40, ey, 620, 210, "#fef2f2", "#fca5a5"),
      t(60, ey + 26, "E. 우리가 잰 것 (틀렸다)", 15, RED, "start", weight="700"),
      t(60, ey + 47, "1초 클립을 그냥 밀었다 → 단어가 클립 밖으로 나가 "
                     "잘린다.", 11.5, MUTE, "start")]
CLIP_X, CLIP_W = 70, 500                       # the 1 s buffer, both rows
s += [box(CLIP_X, ey + 66, CLIP_W, 28, "#ffffff", "#cbd5e1", rx=4, sw=1.1),
      word(CLIP_X + 150, ey + 67, 170, 26, GRN, "yes"),
      t(640, ey + 85, "원본", 10, MUTE, "end")]
# Shift right by 300 ms: the tail crosses the buffer edge and is DISCARDED.
# Drawn past the edge on purpose -- that overhang is the information the
# experiment destroyed and augmentation could never give back.
s += [box(CLIP_X, ey + 106, CLIP_W, 28, "#ffffff", "#cbd5e1", rx=4, sw=1.1),
      f'<rect x="{CLIP_X + 400}" y="{ey + 107}" width="100" height="26" rx="4" '
      f'fill="{GRN}" opacity="0.22" stroke="{GRN}" stroke-width="1.6"/>',
      t(CLIP_X + 450, ey + 125, "ye", 11, GRN, weight="700"),
      f'<rect x="{CLIP_X + CLIP_W}" y="{ey + 107}" width="70" height="26" '
      f'rx="4" fill="{RED}" opacity="0.15" stroke="{RED}" stroke-width="1.4" '
      f'stroke-dasharray="4 3"/>',
      t(CLIP_X + CLIP_W + 35, ey + 125, "s", 11, RED, weight="700"),
      f'<line x1="{CLIP_X + CLIP_W}" y1="{ey + 100}" x2="{CLIP_X + CLIP_W}" '
      f'y2="{ey + 140}" stroke="{RED}" stroke-width="2"/>',
      t(CLIP_X + 200, ey + 125, "0 으로 채움 (인공 무음)", 10, RED),
      t(645, ey + 152, "버려짐 ↑", 9.5, RED, "end")]
s += [t(350, ey + 158, "단어의 뒷부분이 사라졌다 → 정보 소실", 12, RED,
        weight="700"),
      t(350, ey + 178, "증강으로 못 고친다. 실제로 ±300 증강은 +0.33pp뿐이었다.",
        11, INK),
      t(350, ey + 196, "곡선의 −11pp는 오정렬이 아니라 대부분 잘림이었다.", 11,
        INK)]

# ── F. what it really is ───────────────────────────────────────────────────
s += [box(680, ey, 620, 210, "#f0fdf4", "#86efac"),
      t(700, ey + 26, "F. 실제 슬라이딩 (맞는 측정)", 15, GRN, "start",
        weight="700"),
      t(700, ey + 47, "연속 스트림이라 단어는 안 잘린다. 위치만 바뀌고 "
                      "나머지는 방 잡음.", 11.5, MUTE, "start")]
for i, off in enumerate((0.10, 0.45, 0.78)):
    yy = ey + 66 + i * 34
    s += [box(710, yy, 560, 28, "#ffffff", "#cbd5e1", rx=4, sw=1.1)]
    for k in range(28):
        xx = 714 + k * 20
        s.append(f'<line x1="{xx}" y1="{yy + 6}" x2="{xx}" y2="{yy + 22}" '
                 f'stroke="{MUTE}" stroke-width="0.8" opacity="0.35"/>')
    s.append(word(710 + 560 * off, yy + 1, 170, 26, GRN, "yes"))
s += [t(990, ey + 178, "단어는 항상 온전하다 — 에너지 보존 1.00", 12, GRN,
        weight="700"),
      t(990, ey + 196, "회색 빗금 = 실제 방 잡음 (디지털 무음이 아니다)", 11,
        INK)]

# ── G. what to do ──────────────────────────────────────────────────────────
gy = 1040
s += [box(40, gy, 1260, 128, "#faf5ff", "#d8b4fe"),
      t(60, gy + 26, "G. 그래서 할 일 — 순서가 중요하다", 15, PUR, "start",
        weight="700")]
todo = [("1", "순수 오정렬 비용을 잰다", "experiments/window_offset.py — "
         "자르지 않고 창 안에서만 옮긴다"),
        ("2", "낙폭 ≤ 5pp 면", "슬라이딩이 사실상 공짜. 최대 확신도 선택만 "
         "붙이면 끝 (재학습 불필요)"),
        ("3", "낙폭 ≥ 15pp 면", "그때 비로소 증강 — 단, 1초를 미는 게 아니라 "
         "긴 캔버스에 단어를 심어 잘리지 않게")]
for i, (n_, a, b) in enumerate(todo):
    yy = gy + 48 + i * 26
    s += [t(70, yy, n_, 12, PUR, "start", weight="700"),
          t(88, yy, a, 12, INK, "start", weight="700"),
          t(320, yy, b, 11, MUTE, "start")]

s.append(t(40, H - 16, "근거: docs/experiments_log.md §6-3 / §6-5 · "
                       "그림 생성: docs/diagrams/make_sliding_window.py", 10,
           MUTE, "start"))
s.append("</svg>")
OUT.write_text("\n".join(s), encoding="utf-8")
print("wrote", OUT)

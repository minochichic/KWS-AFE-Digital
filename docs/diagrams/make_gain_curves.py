"""21_gain_curves.svg -- the track-1 vs track-2 decision, on the common axis.

frac_sweep cannot compare the tracks (frac is an xlse concept), so notebook 6f
sweeps input gain instead: the speaker moving toward or away from the mic, which
means the same thing to both front ends. Under xlse with delta=0 the gain sweep
and the frac sweep were already shown identical to 1e-7, so nothing is lost.

Three curves, one axis, measured on the test split (n=4888, SE 0.52pp):
  xl_g12   xlse  + gain aug +-12 dB   -- the confirmed model
  fx_d0    fixed + no aug             -- track 2 at its best
  fx_q75   fixed + gain aug +-12 dB   -- track 2 with the augmentation it needs

Palette validated with the dataviz skill's checker (light surface): lightness
band, chroma floor, CVD separation and normal-vision floor all PASS. The amber
carries a contrast WARN against the surface, which obligates visible labels --
every series is directly labelled at its right end, and the table repeats the
numbers, so identity is never colour-alone.

Run:  python3 docs/diagrams/make_gain_curves.py
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "diagrams" / "21_gain_curves.svg"

W, H = 1180, 700
FONT = "'Apple SD Gothic Neo','Noto Sans KR','Helvetica Neue',sans-serif"
MONO = "'SF Mono','Menlo',monospace"

SURFACE = "#fcfcfb"
INK, MUTE, GRID, AXIS = "#1c2333", "#5b6478", "#e7eaef", "#aab2c0"

# validated categorical palette, fixed order -- never cycled, never by rank
SERIES = [
    ("xl_g12", "xlse · 다이오드-OR", "#4a7fd0",
     [(-18, .6324), (-12, .7459), (-6, .8239), (-3, .8337), (0, .8445),
      (3, .8419), (6, .8361), (12, .8155), (18, .7827)], "±12 dB 증강"),
    ("fx_d0", "fixed · 증강 없음", "#cc6155",
     [(-18, .2480), (-12, .4253), (-6, .5943), (-3, .6671), (0, .7171),
      (3, .7496), (6, .7539), (12, .7250), (18, .6301)], "증강 없음"),
    ("fx_q75", "fixed · ±12 dB 증강", "#d9a441",
     [(-18, .2050), (-12, .3165), (-6, .4583), (-3, .5456), (0, .6164),
      (3, .6659), (6, .6933), (12, .7054), (18, .6461)], "±12 dB 증강"),
]

PX0, PX1, PY0, PY1 = 108, 998, 118, 486      # plot rect
LABEL_W = 152                                # right-hand direct labels
GMIN, GMAX, AMIN, AMAX = -18, 18, 0.15, 0.90

parts: list[str] = []
add = parts.append
TEXTS: list[tuple] = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.02 if ord(c) > 0x2E80 else 0.55 for c in s) * size


def text(s, x, y, size=12.5, fill=INK, anchor="middle", weight="400",
         font=FONT, limit=None):
    TEXTS.append((s, x, y, size, anchor))
    if limit is not None and tw(s, size) > limit:
        raise ValueError(f"overflows {limit:.0f}px: {s!r} -> {tw(s, size):.0f}px")
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}'
        f'</text>')


def gx(g):
    return PX0 + (PX1 - PX0) * (g - GMIN) / (GMAX - GMIN)


def ay(a):
    return PY1 - (PY1 - PY0) * (a - AMIN) / (AMAX - AMIN)


add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}">')
add(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

text("소리 크기가 변할 때 — 두 트랙의 공통 축", W / 2, 40, size=21,
     weight="700", limit=W - 60)
text("Speech Commands v2 12-class, test n=4888 (SE 0.52pp).  "
     "가로축은 입력 게인 = 화자가 마이크에서 멀어지고 가까워지는 것.",
     W / 2, 62, size=12.5, fill=MUTE, limit=W - 60)

# ── grid + axes ───────────────────────────────────────────────────────────
for a in [x / 10 for x in range(2, 10)]:
    y = ay(a)
    add(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" '
        f'stroke="{GRID}" stroke-width="1"/>')
    text(f"{a:.1f}", PX0 - 12, y + 4, size=11, fill=MUTE, anchor="end",
         font=MONO, limit=34)
add(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="{AXIS}" '
    f'stroke-width="1.2"/>')
add(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="{AXIS}" '
    f'stroke-width="1.2"/>')

# 0 dB = the level everything was trained at
x0 = gx(0)
add(f'<line x1="{x0:.1f}" y1="{PY0}" x2="{x0:.1f}" y2="{PY1}" '
    f'stroke="{AXIS}" stroke-width="1.4" stroke-dasharray="5 4"/>')
text("학습 레벨", x0, PY0 - 8, size=11, fill=MUTE, limit=90)

for g in (-18, -12, -6, -3, 0, 3, 6, 12, 18):
    text(f"{g:+d}" if g else "0", gx(g), PY1 + 20, size=11, fill=MUTE,
         font=MONO, limit=40)
text("입력 게인 [dB]", (PX0 + PX1) / 2, PY1 + 44, size=12, fill=MUTE, limit=200)
text("test 정확도", PX0 - 46, PY0 - 14, size=12, fill=MUTE, anchor="start",
     limit=120)

# ── series ────────────────────────────────────────────────────────────────
for tag, label, col, pts, _aug in SERIES:
    d = " ".join(("M" if i == 0 else "L") + f"{gx(g):.1f} {ay(a):.1f}"
                 for i, (g, a) in enumerate(pts))
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
        f'stroke-linejoin="round"/>')
    peak = max(pts, key=lambda p: p[1])
    for g, a in pts:
        r = 5.5 if (g, a) == peak else 4
        add(f'<circle cx="{gx(g):.1f}" cy="{ay(a):.1f}" r="{r}" fill="{col}" '
            f'stroke="{SURFACE}" stroke-width="2"/>')
# Direct labels at the right end. The contrast WARN on the amber obligates
# visible text, so these are not optional -- which means they must not collide.
# fx_d0 ends at 0.630 and fx_q75 at 0.646, eight pixels apart on this scale, so
# they get pushed to a minimum spacing and a leader line back to the point.
_ends = sorted(((ay(s[3][-1][1]), gx(s[3][-1][0]), s[1], s[2]) for s in SERIES),
               key=lambda e: e[0])
_placed: list[float] = []
for y, x, label, col in _ends:
    ty = y if not _placed else max(y, _placed[-1] + 18)
    _placed.append(ty)
    if abs(ty - y) > 1:
        add(f'<path d="M{x + 6:.1f} {y:.1f} L{x + 14:.1f} {ty:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="1.2"/>')
    text(label, x + 18, ty + 4, size=12, fill=INK, anchor="start",
         weight="600", limit=LABEL_W)

# peak markers: where each track actually wants to sit
for tag, label, col, pts, _ in SERIES:
    g, a = max(pts, key=lambda p: p[1])
    if g != 0:
        text(f"최고 {g:+d} dB", gx(g), ay(a) - 14, size=10.5, fill=col,
             weight="700", limit=90)

# ── summary table (bottom-left) ──────────────────────────────────────────
BY = 544
TX = 24
add(f'<rect x="{TX}" y="{BY}" width="416" height="122" rx="10" '
    f'fill="#ffffff" stroke="{GRID}" stroke-width="1.4"/>')
cols = [("평균", 82), ("최저", 74), ("낙폭", 78)]
xc = TX + 150
for name, wd in cols:
    text(name, xc + wd - 10, BY + 24, size=11, fill=MUTE, anchor="end",
         limit=wd)
    xc += wd
for i, (tag, label, col, pts, _) in enumerate(SERIES):
    y = BY + 50 + i * 24
    acc = [a for _, a in pts]
    add(f'<rect x="{TX + 18}" y="{y - 9}" width="10" height="10" rx="2" '
        f'fill="{col}"/>')
    text(tag, TX + 34, y, size=11.5, anchor="start", font=MONO, limit=104)
    xc = TX + 150
    for val, wd in ((f"{sum(acc) / len(acc):.4f}", 82),
                    (f"{min(acc):.4f}", 74),
                    (f"{(max(acc) - min(acc)) * 100:.1f}pp", 78)):
        text(val, xc + wd - 10, y, size=11.5, anchor="end", font=MONO,
             limit=wd)
        xc += wd

# ── takeaway (bottom-right) ──────────────────────────────────────────────
KX = 456
add(f'<rect x="{KX}" y="{BY}" width="{W - KX - 24}" height="122" rx="10" '
    f'fill="#ffffff" stroke="{GRID}" stroke-width="1.4"/>')
text("판정 — 다이오드 16개 + 버퍼 1개가 사는 값", KX + 18, BY + 26, size=13.5,
     weight="700", anchor="start", limit=460)
for j, line in enumerate([
    "−18 dB 에서 fixed 는 0.248 로 무너진다 (우연 0.083). xlse 는 0.632.",
    "절대 임계는 소리가 작아지면 아무 채널도 안 켜져 이미지가 통째로 −1 이 된다.",
    "다이오드-OR 은 분모도 같이 작아지므로 상대 패턴이 남는다.",
    "증강으로 낙폭이 안 줄었다 (50.6 → 50.0pp). 상쇄할 분모가 없기 때문.",
    "최고점이 0 dB 가 아닌 +6/+12 dB — 학습 지점에서 이미 비트가 모자란다.",
]):
    text("· " + line, KX + 18, BY + 48 + j * 15, size=10.5, fill=MUTE,
         anchor="start", limit=W - KX - 54)

add("</svg>")

for s, x, y, size, anchor in TEXTS:
    w = tw(s, size)
    lo = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    if lo < -1 or lo + w > W + 1 or y > H:
        raise ValueError(f"text outside canvas: {s!r} at {lo:.0f}..{lo + w:.0f}")

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT}  ({W}x{H})")
for tag, _l, _c, pts, _a in SERIES:
    acc = [a for _, a in pts]
    pg, pa = max(pts, key=lambda p: p[1])
    print(f"  {tag:<8} mean {sum(acc)/len(acc):.4f}  min {min(acc):.4f}  "
          f"drop {(max(acc)-min(acc))*100:4.1f}pp  peak {pa:.4f} @ {pg:+d} dB")

"""5절 도판 2장 — 전체 배선, 학습 모델. 논문 도판 스타일(흑백 선화).

  python3 make_sec5_figs.py     ->  figures/sec5_wiring.png, sec5_model.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

_HAVE = {f.name for f in fm.fontManager.ttflist}
KO = next((f for f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
                       "Malgun Gothic", "Noto Sans CJK KR") if f in _HAVE), "DejaVu Sans")
# AppleGothic 은 대괄호를 전각으로 그린다 -> 도판에서는 소괄호만 쓴다.
matplotlib.rcParams.update({
    "font.family": KO,
    "axes.unicode_minus": False, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

K, LW = "black", 0.9
FILL, HOT, HOTBG, COOL, COOLBG = "#F2F2F2", "#8C4A1C", "#F6EBE1", "#25646F", "#E4EEF0"


def blank(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    return fig, ax


def box(ax, x0, y0, x1, y1, name, sub=None, sub2=None, fc="white", ec=K, lw=LW,
        fs=8.4, round_=False):
    P = FancyBboxPatch if round_ else Rectangle
    if round_:
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
    else:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                               edgecolor=ec, lw=lw, zorder=2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lines = [l for l in (name, sub, sub2) if l]
    step = 13
    top = cy + step * (len(lines) - 1) / 2
    for i, l in enumerate(lines):
        ax.text(cx, top - i * step, l, ha="center", va="center", zorder=3,
                fontsize=fs if i == 0 else fs - 1.6,
                color=K if i == 0 else "#444444")


def arrow(ax, x1, y1, x2, y2, c=K, lw=LW, ls="-", head=0.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_width=.16,head_length=.34",
                                color=c, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def path(ax, pts, c=K, lw=LW, ls="-", arrow_end=True):
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        if arrow_end and i == len(pts) - 2:
            arrow(ax, x1, y1, x2, y2, c=c, lw=lw, ls=ls)
        else:
            ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=1,
                    solid_capstyle="round")


def bus(ax, pts, lw=2.4):
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        ax.plot([x1, x2], [y1, y2], color=K, lw=lw, zorder=1, solid_capstyle="round")


def dot(ax, x, y):
    ax.plot([x], [y], marker="o", ms=3.4, color=K, zorder=4)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  {name}.png")


# ═════════════════════════ 그림 1 — 전체 배선
def wiring():
    fig, ax = blank(8.4, 4.9, (-38, 800), (-6, 452))

    # 채널 버스
    bus(ax, [(-30, 220), (30, 220)])
    ax.plot([-4, 4], [212, 228], color=K, lw=LW, zorder=2)
    ax.text(0, 234, "N", ha="center", fontsize=7.4)
    ax.text(-34, 220, "CH0…CHn", ha="right", va="center", fontsize=8.4)
    ax.text(-34, 206, "비교기 출력", ha="right", va="center", fontsize=7,
            color="#555555")
    bus(ax, [(30, 64), (30, 380)])
    dot(ax, 30, 220)
    bus(ax, [(30, 380), (56, 380)])
    bus(ax, [(30, 64), (306, 64)])

    # 윗줄
    box(ax, 60, 355, 180, 405, "START detect", "gate 조합 또는 R+comp", fc=FILL)
    box(ax, 230, 355, 342, 405, "RUN latch", "D flip-flop", fc=FILL)
    box(ax, 396, 349, 528, 411, "Timing counter", "8 bit, 100 Hz", fc=FILL)
    box(ax, 580, 349, 720, 411, "Time decode", "3-to-8 dec ×2 + NOR", fc=FILL)
    box(ax, 396, 245, 528, 291, "100 Hz osc", "Schmitt + RC", fc=FILL)

    arrow(ax, 180, 392, 226, 392)
    ax.text(203, 400, "→ CLK", ha="center", fontsize=7.2, color="#555555")
    arrow(ax, 342, 392, 392, 392)
    ax.text(367, 400, "CE", ha="center", fontsize=7.2, color="#555555")
    path(ax, [(342, 368), (368, 368), (368, 362), (392, 362)])
    ax.text(360, 344, "/RUN → MR", ha="center", fontsize=7.2, color="#555555")
    arrow(ax, 462, 291, 462, 345)
    ax.text(470, 318, "CLK", ha="left", fontsize=7.2, color="#555555")
    bus(ax, [(528, 380), (552, 380)]); ax.plot([536, 544], [372, 388], color=K, lw=LW)
    ax.text(540, 394, "8", ha="center", fontsize=7.4)
    arrow(ax, 552, 380, 576, 380)

    # 디코드 출력
    path(ax, [(720, 395), (748, 395), (748, 222), (100, 222), (100, 165), (126, 165)])
    ax.text(744, 426, "DECODE S1–S4", ha="right", fontsize=7.6)

    # 아랫줄
    box(ax, 130, 141, 260, 189, "SAMPLE gate", "DECODE · /CLK", fc=FILL)
    box(ax, 310, 141, 450, 189, "PASS F/F ×4", "edge-triggered", fc=FILL)
    box(ax, 500, 141, 612, 189, "Final AND", "2-input ×3", fc=FILL)
    box(ax, 310, 40, 450, 88, "Template match", "R network + comp ×4", fc=FILL)

    arrow(ax, 260, 165, 306, 165)
    ax.text(312, 198, "SAMPLE → CLK", ha="left", fontsize=7.2, color="#555555")
    arrow(ax, 450, 165, 496, 165)
    ax.text(473, 173, "PASS 1-4", ha="center", fontsize=7.2, color="#555555")
    arrow(ax, 612, 165, 668, 165)
    ax.text(674, 165, "WAKE", ha="left", va="center", fontsize=9)
    arrow(ax, 366, 88, 366, 137)
    ax.text(358, 112, "MATCH 1-4  → D", ha="right", fontsize=7.2, color="#555555")

    # 타임아웃
    TO = HOT
    path(ax, [(720, 365), (768, 365), (768, 14), (286, 14), (286, 351)], c=TO,
         ls=(0, (4, 3)))
    ax.text(762, 330, "DECODE_TO", ha="right", fontsize=7.6, color=TO)
    path(ax, [(470, 14), (470, 137)], c=TO, ls=(0, (4, 3)))
    dot_ = ax.plot([470], [14], marker="o", ms=3.4, color=TO, zorder=4)
    ax.text(294, 344, "CLR", ha="left", fontsize=7.2, color=TO)
    ax.text(478, 130, "CLR", ha="left", fontsize=7.2, color=TO)
    ax.text(150, 22, "timeout → clear RUN and PASS", ha="center", fontsize=7.6,
            color=TO)

    save(fig, "sec5_wiring")


# ═════════════════════════ 그림 2 — 학습 모델
def model():
    fig, ax = blank(7.8, 7.0, (-46, 784), (-10, 700))
    CX = 175

    # 이진 특징 래스터
    rows = [[(0, 1), (2, 3)], [(1, 3), (5, 8)], [(3, 6), (9, 11)],
            [(0, 1), (4, 8)], [(2, 4), (7, 10)], [(5, 7), (10, 13)],
            [(1, 2), (6, 9)], [(4, 6), (8, 13)]]
    x0, y0, cw, ch = 76, 45, 12.4, 7.4
    for r, segs in enumerate(rows):
        for a, b in segs:
            ax.add_patch(Rectangle((x0 + a * cw, y0 + (7 - r) * (ch + 1.7)),
                                   (b - a + 1) * cw, ch, facecolor="#2B2B2B",
                                   edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((70, 38), 210, 80, facecolor="none", edgecolor=K,
                           lw=1.1, zorder=2))
    ax.text(CX, 26, "Binary feature", ha="center", fontsize=8.6)
    ax.text(CX, 12, "X(c,t) ∈ {0,1}, 10 ms 격자", ha="center", fontsize=7.2,
            color="#444444")
    ax.text(CX, -2, "아날로그 전단 출력 — 학습 대상 아님", ha="center", fontsize=6.8,
            color="#888888")

    box(ax, 70, 168, 280, 226, "START detect + Align", "전수 탐색", "정렬 ±1 프레임",
        fc=HOTBG, ec=HOT, lw=1.3, round_=True)
    box(ax, 70, 268, 280, 346, "Template Head", "M(s,c), k(s), τ(s)", "× 4 states",
        fc=HOTBG, ec=HOT, lw=1.3, round_=True)
    box(ax, 70, 388, 280, 446, "AND over states", "WAKE score", fc=FILL, round_=True)
    box(ax, 70, 488, 280, 546, "Binary Cross Entropy", "대상 단어 vs 나머지",
        fc=COOLBG, ec=COOL, lw=1.3, round_=True)

    for a, b in ((118, 164), (226, 264), (346, 384), (446, 484)):
        arrow(ax, CX, a, CX, b, lw=1.25)
    ax.text(62, 249, "정렬됨", ha="right", va="center", fontsize=7, color="#888888")
    ax.text(62, 369, "PASS(s)", ha="right", va="center", fontsize=7, color="#888888")
    ax.text(62, 197, "탐색", ha="right", va="center", fontsize=8.4, color="#555555")
    ax.text(62, 307, "학습", ha="right", va="center", fontsize=8.4, color="#555555")

    # 확대 패널
    ax.add_patch(FancyBboxPatch((396, 46), 358, 540,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="none", edgecolor=K, lw=0.75,
                                linestyle=(0, (4, 3)), zorder=1))
    ax.plot([280, 396], [342, 586], color=K, lw=0.75, ls=(0, (4, 3)), zorder=1)
    ax.plot([280, 396], [272, 46],  color=K, lw=0.75, ls=(0, (4, 3)), zorder=1)
    ax.text(410, 566, "한 상태를 펼친 것", ha="left", fontsize=7.6, color="#666666")

    PX = 552
    box(ax, 447, 66,  657, 116, "Gather at τ(s)", "한 프레임만 뽑는다", fc=FILL)
    box(ax, 447, 146, 657, 202, "Ternary template", "M = g · sign(w), STE",
        "g = 사용 / 무시", fc=HOTBG, ec=HOT, lw=1.3)
    box(ax, 447, 232, 657, 282, "Match count", "쓰는 채널의 단순 합", fc=FILL)
    box(ax, 447, 312, 657, 362, "Tolerance k(s)", "count ≥ k", fc=HOTBG, ec=HOT, lw=1.3)
    box(ax, 447, 392, 657, 442, "PASS(s)", "0 – 1", fc=FILL)
    for a, b in ((116, 142), (202, 228), (282, 308), (362, 388)):
        arrow(ax, PX, a, PX, b, lw=1.25)
    arrow(ax, PX, 442, PX, 476, lw=1.25)
    ax.text(PX, 492, "s = 1, 2, 3, 4 반복", ha="center", fontsize=8.4)

    # L1 가지
    path(ax, [(657, 174), (690, 174), (690, 512)], c=HOT, ls=(0, (4, 3)))
    box(ax, 618, 512, 730, 562, "L1 on g", "X 를 늘린다", fc=HOTBG, ec=HOT, lw=1.2)
    ax.text(698, 340, "저항 수 감소", rotation=90, ha="center", va="center",
            fontsize=7.2, color=HOT)

    save(fig, "sec5_model")


if __name__ == "__main__":
    print(f"글꼴 {KO}\n저장 {OUT}")
    wiring(); model()

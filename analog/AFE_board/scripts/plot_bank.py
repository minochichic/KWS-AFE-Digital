"""What peak-normalization erases from the board's filterbank.

filterbank_matrix_board.csv stores each channel scaled so its peak is 1.0, so
the real per-channel passband gain is gone. This draws both forms side by side
and puts a number on the difference, because on THIS board the answer is
"almost nothing" (0.5 dB spread) and that is worth seeing rather than assuming.

  .venv/bin/python analog/AFE_board/scripts/plot_bank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm                             # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

# Labels are Korean; DejaVu has no Hangul and silently draws tofu boxes.
_HAVE = {f.name for f in fm.fontManager.ttflist}
for _f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic",
           "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
    if _f in _HAVE:
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False   # the CJK fonts lack U+2212

BOARD = Path(__file__).resolve().parents[1]
AFE = BOARD.parent / "AFE"
SR, N_FFT = 16000, 512


def load(suffix: str):
    m = np.loadtxt(AFE / "artifacts" / f"filterbank_matrix{suffix}.csv", delimiter=",")
    d = np.loadtxt(AFE / "artifacts" / f"filterbank_design{suffix}.csv",
                   delimiter=",", skiprows=1)
    return m, d[:, 5]                                            # matrix, gain_dB


def main() -> int:
    grid = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    m, gdb = load("_board")
    lin = 10 ** (gdb / 20.0)
    restored = m * lin[:, None]
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(m)))

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.33, wspace=0.22)

    def bank(ax, mat, title, sub):
        for k, row in enumerate(mat):
            ax.plot(grid[1:], row[1:], color=colors[k], lw=1.5)
        ax.set_xscale("log")
        ax.set_xlim(80, 8000)
        ax.set_xlabel("frequency [Hz]")
        ax.set_title(f"{title}\n{sub}", fontsize=11, fontweight="bold",
                     loc="left")
        ax.title.set_linespacing(1.6)
        ax.grid(alpha=0.25, which="both", lw=0.5)
        for f in (125, 5000):                       # the stated band edges
            ax.axvline(f, color="#c0392b", ls=":", lw=1.2)

    ax = fig.add_subplot(gs[0, 0])
    bank(ax, m, "peak-normalized  (학습이 실제로 보는 것)",
         "행마다 max=1.0 -- 채널 간 이득 차이가 지워진 상태")
    ax.set_ylabel("|H| (peak = 1)")
    ax.set_ylim(0, 1.08)
    ax.annotate("저역 꼭대기가 각진 건 회로가 아니라 격자 탓.\n"
                "ch0 대역폭 30 Hz < STFT 빈 31.25 Hz 라\n"
                "봉우리를 1~2점으로밖에 못 찍는다.",
                xy=(205, 1.0), xytext=(700, 0.80), fontsize=8.5,
                color="#c0392b", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b",
                          alpha=0.92, lw=0.8),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax = fig.add_subplot(gs[0, 1])
    bank(ax, restored, "gain restored  (spice_gain_restore: True)",
         "통과대역 실제 이득으로 다시 세운 것")
    ax.set_ylabel("|H| (linear, x)")
    ax.set_ylim(0, 1.08 * restored.max())

    # --- per-channel gain, the thing that was erased ---
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(16), gdb, color=colors, width=0.72)
    ax.set_ylim(gdb.min() - 0.9, gdb.max() + 0.5)
    ax.set_xticks(range(16))
    ax.set_xlabel("channel")
    ax.set_ylabel("passband gain [dB]")
    ax.set_title(f"지워진 정보: 채널별 이득  (폭 {gdb.max()-gdb.min():.2f} dB "
                 f"= x{lin.max()/lin.min():.3f})",
                 fontsize=11, fontweight="bold", loc="left")
    ax.axhline(gdb.mean(), color="#c0392b", ls="--", lw=1,
               label=f"평균 {gdb.mean():.2f} dB")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # --- how much the two forms actually differ, vs the older bank ---
    ax = fig.add_subplot(gs[1, 1])
    labels, spreads = [], []
    for suf, name in (("_board", "동료 보드\n(Q 4.5 고정)"),
                      ("_125_5000", "우리 설계\n125-5000"),
                      ("", "우리 설계\n50-8000")):
        try:
            _, g = load(suf)
        except OSError:
            continue
        labels.append(name)
        spreads.append(g.max() - g.min())
    bars = ax.bar(labels, spreads, color=["#2e7d32", "#7f8c8d", "#7f8c8d"][:len(spreads)],
                  width=0.55)
    for b, s in zip(bars, spreads):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.04, f"{s:.2f} dB",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("이득 폭 [dB]")
    ax.set_ylim(0, max(spreads) * 1.28)
    ax.set_title("정규화가 지우는 양 — 뱅크별 비교\n"
                 "작을수록 peak-normalization 이 잃는 것이 적다",
                 fontsize=11, fontweight="bold", loc="left")
    ax.title.set_linespacing(1.6)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("filterbank_matrix_board.csv — 피크 정규화가 지운 것",
                 fontsize=13.5, fontweight="bold", x=0.5, y=0.985)
    out = BOARD / "artifacts" / "bank_normalization.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"저장: {out.relative_to(BOARD.parent.parent)}")
    print(f"이득 {gdb.min():.2f}~{gdb.max():.2f} dB, 폭 {gdb.max()-gdb.min():.2f} dB "
          f"= 진폭 x{lin.max()/lin.min():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

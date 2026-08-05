"""Compare 2nd-order vs 4th-order (cascaded) GIC channel selectivity.

Runs gic_4th.cir (.ac), reads the stage-1 (2nd) and stage-2 (4th) magnitude
responses, and measures for each: peak freq f_c, -3 dB bandwidth, Q = f_c/BW,
and the skirt roll-off (dB/octave) an octave-plus away from the peak. The point:
does 4th order actually sharpen the skirts (less spectral blur -> the Phase B
gap), and at what bandwidth cost?

Output: artifacts/order_compare.png, artifacts/order_compare.md
Run from repo root: .venv/bin/python AFE_highorder/scripts/compare_order.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HO = Path(__file__).resolve().parents[1]


def run():
    (HO / "sim").mkdir(exist_ok=True)
    subprocess.run(["ngspice", "-b", str(HO / "netlists" / "gic_4th.cir")],
                   cwd=HO, capture_output=True, text=True, timeout=120)
    d = np.loadtxt(HO / "sim" / "gic4.csv")
    f = d[:, 0]
    return f, d[:, 1], d[:, 3]        # freq, 2nd-order dB, 4th-order dB


def analyze(f, mag):
    mag = mag - mag.max()                       # normalize peak to 0 dB
    i_pk = int(np.argmax(mag))
    fc = f[i_pk]
    # -3 dB bandwidth (cross below/above peak)
    below = np.where(mag[:i_pk] <= -3)[0]
    above = np.where(mag[i_pk:] <= -3)[0]
    f_lo = f[below[-1]] if len(below) else f[0]
    f_hi = f[i_pk + above[0]] if len(above) else f[-1]
    bw = f_hi - f_lo
    Q = fc / bw if bw > 0 else float("nan")
    # lower-skirt slope: fit dB vs log2(f) over [fc/8, fc/3]
    m = (f >= fc / 8) & (f <= fc / 3)
    slope = np.polyfit(np.log2(f[m]), mag[m], 1)[0] if m.sum() > 3 else float("nan")
    return dict(fc=fc, f_lo=f_lo, f_hi=f_hi, bw=bw, Q=Q, slope=slope, mag=mag)


def main():
    f, m2, m4 = run()
    a2, a4 = analyze(f, m2), analyze(f, m4)
    for name, a in [("2nd order (1 stage)", a2), ("4th order (cascade)", a4)]:
        print(f"{name:22s} f_c={a['fc']:6.0f} Hz  BW={a['bw']:6.0f} Hz  "
              f"Q={a['Q']:.2f}  lower-skirt={a['slope']:.1f} dB/oct")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogx(f, a2["mag"], color="tab:gray", lw=2, label="2nd order (±6 dB/oct)")
    ax.semilogx(f, a4["mag"], color="tab:blue", lw=2, label="4th order cascade (±12 dB/oct)")
    ax.axhline(-3, color="tab:red", ls="--", lw=1, label="-3 dB")
    ax.set_ylim(-60, 3); ax.set_xlim(f[0], f[-1])
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("normalized |H| [dB]")
    ax.set_title("GIC channel selectivity: 2nd vs 4th order (skirt steepness)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(HO / "artifacts" / "order_compare.png", dpi=130)

    md = ["# 2차 vs 4차 GIC 채널 선택도\n",
          "동일 2차 GIC 2단 캐스케이드 = 4차. 스커트가 날카로워지지만 대역폭이 좁아짐.\n",
          "| 차수 | f_c [Hz] | -3dB 대역 [Hz] | Q | 하단 스커트 [dB/oct] |",
          "|---|---:|---:|---:|---:|"]
    for name, a in [("2차 (1단)", a2), ("4차 (캐스케이드)", a4)]:
        md.append(f"| {name} | {a['fc']:.0f} | {a['bw']:.0f} | {a['Q']:.2f} | {a['slope']:.1f} |")
    md += ["",
           "- **스커트**: 4차가 2차의 ~2배 기울기(±6→±12 dB/oct)면 블러가 준다.",
           "- **대역폭 대가**: 캐스케이드는 -3dB 대역이 좁아짐 → 16채널 고정 시 대역 사이 "
           "**커버리지 공백** 위험. 실사용엔 stagger-tuning(스테이지별 f_c 약간 엇갈림)이나 "
           "Q 조정으로 대역 유지 필요.",
           "- **비용**: 채널당 opamp 2→4개 → 전력 ~2배(µW 예산에 직접 부담).",
           "",
           "→ 다음: 대역 유지(stagger)한 4차 채널, 그리고 16채널 뱅크로 확장 시 커버리지 점검."]
    (HO / "artifacts" / "order_compare.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/order_compare.png, order_compare.md")


if __name__ == "__main__":
    main()

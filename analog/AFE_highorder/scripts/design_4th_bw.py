"""Design a bandwidth-preserved 4th-order GIC channel.

Cascading two 2nd-order stages steepens skirts but narrows the band. Here we
BROADEN each stage (lower Q via R1c) until the 4th-order cascade's -3 dB
bandwidth equals the 2nd-order reference -- keeping coverage while gaining the
steeper 4th-order skirts. Bisects R1c in SPICE, then reports both filters'
bandwidth / skirt / passband flatness.

Output: artifacts/order4_bw.png, artifacts/order4_bw.md
Run from repo root: .venv/bin/python AFE_highorder/scripts/design_4th_bw.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HO = Path(__file__).resolve().parents[1]
NET = (HO / "netlists" / "gic_4th_bw.cir").read_text()


def run(r1c):
    net = re.sub(r"\.param R1c\s*=\s*\S+", f".param R1c = {r1c:.6g}", NET)
    (HO / "sim").mkdir(exist_ok=True)
    (HO / "sim" / "tmp.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(HO / "sim" / "tmp.cir")],
                   cwd=HO, capture_output=True, text=True, timeout=120)
    d = np.loadtxt(HO / "sim" / "gic4bw.csv")
    return d[:, 0], d[:, 1], d[:, 3]        # f, vref(2nd) dB, v4(4th) dB


def metrics(f, mag):
    mag = mag - mag.max()
    i = int(np.argmax(mag)); fc = f[i]
    lo = np.where(mag[:i] <= -3)[0]; hi = np.where(mag[i:] <= -3)[0]
    f_lo = f[lo[-1]] if len(lo) else f[0]
    f_hi = f[i + hi[0]] if len(hi) else f[-1]
    bw = f_hi - f_lo
    m = (f >= fc / 8) & (f <= fc / 3)
    slope = np.polyfit(np.log2(f[m]), mag[m], 1)[0] if m.sum() > 3 else np.nan
    return fc, bw, fc / bw, slope, mag


def bw4(r1c):
    f, _, v4 = run(r1c)
    return metrics(f, v4)[1]


def main():
    # target BW from the 2nd-order reference
    f, vref, _ = run(30e3)
    fc_ref, bw_ref, Q_ref, sl_ref, mref = metrics(f, vref)
    print(f"reference 2nd order: f_c={fc_ref:.0f}  BW={bw_ref:.0f}  Q={Q_ref:.2f}  "
          f"skirt={sl_ref:.1f} dB/oct  (target BW={bw_ref:.0f})")

    # bisect R1c so cascade BW == reference BW (lower R1c -> lower Q -> wider)
    lo, hi = 5e3, 53.3e3
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if bw4(mid) > bw_ref:      # too wide -> raise Q (raise R1c)
            lo = mid
        else:
            hi = mid
    r1c = np.sqrt(lo * hi)

    f, vref, v4 = run(r1c)
    fc_ref, bw_ref, Q_ref, sl_ref, mref = metrics(f, vref)
    fc4, bw4v, Q4, sl4, m4 = metrics(f, v4)
    ripple = m4.max() - m4[(f >= fc4 * 0.9) & (f <= fc4 * 1.1)].min()   # in-band
    print(f"4th order (R1c={r1c/1e3:.2f}k): f_c={fc4:.0f}  BW={bw4v:.0f}  Q={Q4:.2f}  "
          f"skirt={sl4:.1f} dB/oct")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogx(f, mref, color="tab:gray", lw=2, label=f"2nd order (BW {bw_ref:.0f}, ±6 dB/oct)")
    ax.semilogx(f, m4, color="tab:blue", lw=2,
                label=f"4th order, BW-preserved (BW {bw4v:.0f}, ±12 dB/oct)")
    ax.axhline(-3, color="tab:red", ls="--", lw=1)
    ax.set_ylim(-60, 3); ax.set_xlim(f[0], f[-1])
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("normalized |H| [dB]")
    ax.set_title("Bandwidth-preserved 4th order: same -3dB band, 2x steeper skirts")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(HO / "artifacts" / "order4_bw.png", dpi=130)

    md = ["# 대역폭 보존 4차 채널 설계\n",
          "각 스테이지 Q를 낮춰(R1c↓) 캐스케이드 후 -3dB 대역을 2차와 맞춤. 스커트는 4차 유지.\n",
          "| 차수 | f_c [Hz] | -3dB 대역 [Hz] | Q | 스커트 [dB/oct] |",
          "|---|---:|---:|---:|---:|",
          f"| 2차 (기준) | {fc_ref:.0f} | {bw_ref:.0f} | {Q_ref:.2f} | {sl_ref:.1f} |",
          f"| 4차 (대역보존) | {fc4:.0f} | {bw4v:.0f} | {Q4:.2f} | {sl4:.1f} |",
          "",
          f"- 설계값: 각 스테이지 **R1c = {r1c/1e3:.2f}kΩ** (기준 R1=53.3k보다 낮음=Q↓=대역↑).",
          f"- **대역 동일(≈{bw_ref:.0f}Hz)인데 스커트 {sl_ref:.1f}→{sl4:.1f} dB/oct** → 이웃 채널 "
          "겹침(블러) 감소, 커버리지 공백 없음.",
          f"- 통과대역 리플 ≈ {ripple:.2f} dB (동기동조라 평탄, 큰 리플 없음).",
          "- 대가: 채널당 opamp 2→4개(전력 ~2배).",
          "",
          "→ 다음: 이 방식으로 16채널 4차 뱅크 생성 + 블러(겹침) 정량 비교."]
    (HO / "artifacts" / "order4_bw.md").write_text("\n".join(md) + "\n")
    print(f"통과대역 리플 {ripple:.2f} dB.  저장: artifacts/order4_bw.png, order4_bw.md")


if __name__ == "__main__":
    main()

"""Sweep detector gain (via R4) to enlarge the V+ swing for comparator margin.

Gain G = R5/R4. Lowering R4 raises gain while keeping tau = R5*C3 fixed (~4.7 ms),
so it is the clean knob. For each R4 we measure the compression curve (V+ vs
input amplitude at 1 kHz) and extract:
  - deadzone (turn-on amplitude)  -- does higher loop gain shrink it?
  - V+ swing at a weak (10 mV) and loud (100 mV) input
  - saturation: whether V+ hits the rail headroom (~900 mV above 0.9 V bias)
Goal: pick R4 so even weak/HF signals give V+ swing >> comparator offset
(~5 mV) without clipping loud signals.

Outputs: artifacts/detector_gain_sweep.png, artifacts/detector_gain_sweep.md
Run:  .venv/bin/python AFE/scripts/sweep_detector_gain.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AFE = Path(__file__).resolve().parents[1]
NET = (AFE / "netlists" / "detector.cir").read_text()
BIAS = 0.9
FREQ = 1000.0
R5 = 47e3
VOS = 5e-3                       # assumed comparator input offset (order of a few mV)
HEADROOM = 0.9                   # 0.9 V bias -> 1.8 V rail
R4_LIST = [10e3, 4.7e3, 2.2e3, 1e3, 470.0]
AMPS = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1])


def run(r4, amp):
    net = re.sub(r"R4\s+vfilt\s+da\s+\S+", f"R4 vfilt da {r4:.6g}", NET)
    net = re.sub(r"\.param AMP=\S+", f".param AMP={amp:.6g}", net)
    net = re.sub(r"\.param FREQ=\S+", f".param FREQ={FREQ:.6g}", net)
    (AFE / "sim").mkdir(exist_ok=True)
    (AFE / "sim" / "tmp_g.cir").write_text(net)
    subprocess.run(["ngspice", "-b", str(AFE / "sim" / "tmp_g.cir")],
                   cwd=AFE, capture_output=True, text=True, timeout=300)
    d = np.loadtxt(AFE / "sim" / "det.csv")
    t, vdet = d[:, 0], d[:, 3]
    return vdet[t >= 30e-3].mean() - BIAS       # steady-state V+ excursion


def turn_on(exc):
    pos = np.where(exc > 0)[0]
    if len(pos) == 0 or pos[0] == 0:
        return np.nan
    i = pos[0]
    f = (0 - exc[i-1]) / (exc[i] - exc[i-1])
    return np.exp(np.log(AMPS[i-1]) + f * (np.log(AMPS[i]) - np.log(AMPS[i-1])))


def at(exc, a):
    """Interpolate V+ excursion at input amplitude a (log-x)."""
    m = exc > 0
    if m.sum() < 2:
        return np.nan
    return float(np.interp(np.log(a), np.log(AMPS[m]), exc[m]))


def main():
    curves = {}
    rows = []
    for r4 in R4_LIST:
        exc = np.array([run(r4, A) for A in AMPS])
        curves[r4] = exc
        dz = turn_on(exc)
        v10, v100 = at(exc, 1e-2), at(exc, 1e-1)
        vmax = exc.max()
        sat = vmax > 0.85 * HEADROOM
        rows.append((r4, R5/r4, dz, v10, v100, vmax, sat))
        print(f"R4={r4/1e3:5.2f}k G={R5/r4:5.1f}: deadzone={dz*1e3:5.2f}mV "
              f"V+@10mV={v10*1e3:6.1f}mV V+@100mV={v100*1e3:6.1f}mV "
              f"max={vmax*1e3:5.0f}mV{' CLIP' if sat else ''}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(R4_LIST)))
    for (r4), c in zip(R4_LIST, cmap):
        e = curves[r4]; m = e > 0
        ax.loglog(AMPS[m]*1e3, e[m]*1e3, "o-", color=c,
                  label=f"R4={r4/1e3:.2f}k (G={R5/r4:.0f})")
    ax.axhline(VOS*1e3, color="red", ls="--", lw=1, label=f"comparator Vos ~{VOS*1e3:.0f}mV")
    ax.axhline(HEADROOM*1e3, color="black", ls=":", lw=1, label="rail headroom 900mV")
    ax.set_xlabel("input amplitude A [mV]"); ax.set_ylabel("envelope V+ − 0.9V [mV]")
    ax.set_title("Detector gain sweep (R4) — V+ swing vs comparator margin (1 kHz)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(AFE / "artifacts" / "detector_gain_sweep.png", dpi=130)

    md = ["# 검출기 이득(R4) sweep — 비교기 마진 확보\n",
          f"G=R5/R4 (R5={R5/1e3:.0f}k 고정 → τ=R5·C3 불변). 1 kHz, 비교기 Vos≈{VOS*1e3:.0f}mV 가정, "
          f"레일 여유 {HEADROOM*1e3:.0f}mV.\n",
          "| R4 [kΩ] | G | 데드존 [mV] | V+@10mV | V+@100mV | max [mV] | 포화 |",
          "|---:|---:|---:|---:|---:|---:|:--|"]
    for r4, g, dz, v10, v100, vmax, sat in rows:
        md.append(f"| {r4/1e3:.2f} | {g:.0f} | {dz*1e3:.2f} | {v10*1e3:.1f} | "
                  f"{v100*1e3:.1f} | {vmax*1e3:.0f} | {'CLIP' if sat else '-'} |")
    md += ["", "- **데드존**은 이득을 올려도 opamp GBW가 정해서 크게 안 줄 수 있음(측정치 참조).",
           "- **V+@10mV**(약신호)가 Vos(~5mV)의 수십 배가 되도록 R4를 낮추면 HF 마진 확보.",
           "- 단 이득이 너무 크면 큰 입력에서 **레일 포화(CLIP)** → 정보 손실. 그 사이가 최적.",
           "- 대가: R4↓ = 필터 출력 부하↑·소비전류↑(µW 예산). 이득단 추가는 opamp 1개 추가.",
           "",
           "→ 표에서 CLIP 없이 V+@10mV가 가장 큰 R4가 1차 후보. 확정 후 full_chain/threshold "
           "재캘리브레이션 + 비교기 오프셋 모델로 검증."]
    (AFE / "artifacts" / "detector_gain_sweep.md").write_text("\n".join(md) + "\n")
    print("저장: artifacts/detector_gain_sweep.png, detector_gain_sweep.md")


if __name__ == "__main__":
    main()
